# Deadlock Prevention Design

**Date:** 2026-03-01
**Status:** Approved

## Problem

The TTS server deadlocks when a streaming inference hangs on GPU. Health checks pass but all synthesis requests block forever.

### Root Cause

1. `_stream_synthesize()` bypasses `PriorityInferQueue` — calls `_infer_executor.submit()` directly
2. No timeout on `asyncio.Queue.get()` in `_stream_synthesize` — waits forever if GPU stalls
3. `_infer_executor` has 1 thread — once stuck, all queue-based inference is blocked
4. No mechanism to detect or recover from a stuck executor thread
5. Streaming endpoints don't track `_queue_tracker` depth — invisible to health checks
6. Model load/unload uses same executor — also blocked when inference is stuck

### Observed Failure

- Streaming request produced 6 token chunks, then GPU hung (CUDA OOM from competing process)
- Executor thread permanently occupied
- `queue_depth: 2` showed stuck requests; health check returned `status: ok`
- Server unrecoverable without container restart

## Design

### 1. Route streaming through PriorityInferQueue

**Current:** `_stream_synthesize()` calls `_infer_executor.submit(_run)` directly.

**New:** Submit `_run` via `_infer_queue.submit()` with `PRIORITY_REALTIME`. The queue worker runs it in the executor, serializing with batch jobs. The chunk-level asyncio.Queue bridge remains — consumer reads chunks with timeout while the queue job runs to completion.

The existing `submit()` method works because the worker awaits `run_in_executor(fn)` to completion. For streaming, `fn` is `_run()` which pushes chunks and returns when done. The consumer reads chunks concurrently from the asyncio.Queue.

### 2. Timeout on `queue.get()` in `_stream_synthesize`

```python
item = await asyncio.wait_for(queue.get(), timeout=REQUEST_TIMEOUT)
```

Prevents infinite hang. Consumer raises `TimeoutError` after `REQUEST_TIMEOUT` seconds.

### 3. Queue depth tracking for streaming endpoints

Add `_queue_tracker.acquire/release` calls to:
- `POST /v1/audio/speech/stream`
- `POST /v1/audio/speech/stream/pcm`
- `GET /v1/audio/speech/ws` (per-message)

Same pattern as `/speech` and `/clone`.

### 4. Inference watchdog with `os._exit(1)`

New background task `_inference_watchdog()`:
- Global `_infer_job_started_at: float | None` set by the queue worker when a job starts, cleared when it completes
- Checks every 10 seconds
- If a job has been running longer than `REQUEST_TIMEOUT + 30` seconds, logs CRITICAL and calls `os._exit(1)`
- Docker `restart: unless-stopped` brings the container back

### 5. Robust error propagation in `_stream_synthesize._run()`

Add `finally` block to `_run()` ensuring the sentinel `None` is always pushed to the chunk queue, even if `asyncio.run_coroutine_threadsafe(...).result()` itself fails (e.g., event loop closed).

```python
def _run():
    sentinel_sent = False
    try:
        for chunk, sr in _do_synthesize_streaming(...):
            asyncio.run_coroutine_threadsafe(queue.put((chunk, sr)), loop).result()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
        sentinel_sent = True
    except Exception as e:
        try:
            asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()
            sentinel_sent = True
        except Exception:
            pass
    finally:
        if not sentinel_sent:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)
            except Exception:
                pass
```

## Files Modified

- `server.py` — all changes in this single file

## Testing

- Unit tests in `server_test.py` covering:
  - Timeout fires on stuck `queue.get()`
  - Watchdog detects stuck job and exits
  - Streaming routes through queue (mock verification)
  - Queue depth tracks streaming requests
- Manual verification: restart container, confirm synthesis works

## Recovery Behavior

| Scenario | Before | After |
|----------|--------|-------|
| GPU inference hangs | Server deadlocked forever | Watchdog kills process after REQUEST_TIMEOUT+30s, Docker restarts |
| Streaming generator stalls | `queue.get()` blocks forever | TimeoutError after REQUEST_TIMEOUT seconds |
| Client disconnects mid-stream | Executor thread still stuck | Same timeout + watchdog applies |
| Queue full during streaming | Invisible, no rejection | 503 rejection via `_queue_tracker` |
