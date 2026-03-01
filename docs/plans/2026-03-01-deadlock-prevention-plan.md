# Deadlock Prevention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate inference deadlocks by routing all GPU work through the PriorityInferQueue, adding timeouts to all blocking waits, tracking queue depth for streaming endpoints, and adding an inference watchdog that self-terminates on unrecoverable hangs.

**Architecture:** Five changes to `server.py`: (1) `_stream_synthesize` submits to `_infer_queue` instead of `_infer_executor` directly, (2) `queue.get()` calls have `REQUEST_TIMEOUT`, (3) streaming endpoints track `_queue_tracker`, (4) `_inference_watchdog` background task monitors job duration and calls `os._exit(1)` on stuck jobs, (5) `_run()` gets a `finally` block ensuring the sentinel is always pushed. All unit tests in `server_test.py`.

**Tech Stack:** Python asyncio, ThreadPoolExecutor, pytest with mocks

**Design doc:** `docs/plans/2026-03-01-deadlock-prevention-design.md`

---

### Task 1: Add inference watchdog timestamp tracking to PriorityInferQueue

**Files:**
- Modify: `server.py:305-370` (PriorityInferQueue class)
- Test: `server_test.py` (new TestInferenceWatchdog class)

**Step 1: Write the failing test**

Add to `server_test.py` at the end:

```python
class TestInferenceWatchdog:
    """Tests for inference job duration tracking and watchdog."""

    def test_job_started_at_set_during_execution(self):
        """_infer_job_started_at is set when a job runs and cleared after."""
        timestamps = []

        def capture_fn():
            timestamps.append(server._infer_job_started_at)
            return "done"

        async def run():
            server._infer_job_started_at = None
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()
            await queue.submit(capture_fn, priority=1)
            await asyncio.sleep(0.05)  # let worker loop clear the timestamp

        asyncio.run(run())
        assert len(timestamps) == 1
        assert timestamps[0] is not None  # was set during execution
        assert server._infer_job_started_at is None  # cleared after
```

**Step 2: Run test to verify it fails**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestInferenceWatchdog::test_job_started_at_set_during_execution -v`
Expected: FAIL — `server._infer_job_started_at` does not exist

**Step 3: Add `_infer_job_started_at` global and update `_worker` to set/clear it**

In `server.py`, after line 398 (`_infer_queue = PriorityInferQueue()`), add:

```python
# Timestamp when the current inference job started (None = idle)
_infer_job_started_at: float | None = None
```

In `PriorityInferQueue._worker`, wrap each `run_in_executor` call with timestamp tracking. The worker method (lines 318-369) should set `_infer_job_started_at = time.monotonic()` before the executor call and `_infer_job_started_at = None` after (in a `finally` block).

Modify the batch branch (lines 345-358):
```python
                    try:
                        global _infer_job_started_at
                        _infer_job_started_at = time.monotonic()
                        wavs, srs = await loop.run_in_executor(
                            self._infer_executor,
                            lambda: _do_synthesize_batch(texts, langs, voice_files, kwargs_list),
                        )
                        sr = srs[0] if isinstance(srs, list) else srs
                        for job, wav in zip(batch_jobs, wavs):
                            if not job.future.done():
                                job.future.set_result(([wav], sr))
                    except Exception as exc:
                        logger.opt(exception=True).error("Batch inference failed", batch_size=len(batch_jobs))
                        for job in batch_jobs:
                            if not job.future.done():
                                job.future.set_exception(exc)
                    finally:
                        _infer_job_started_at = None
```

Modify the single-job branch (lines 360-369):
```python
                else:
                    logger.debug("Queue dispatching single job", priority=single_job.priority,
                                 remaining=len(self._heap))
                    try:
                        global _infer_job_started_at
                        _infer_job_started_at = time.monotonic()
                        result = await loop.run_in_executor(self._infer_executor, single_job.fn)
                        if not single_job.future.done():
                            single_job.future.set_result(result)
                    except Exception as exc:
                        logger.opt(exception=True).error("Single job inference failed")
                        if not single_job.future.done():
                            single_job.future.set_exception(exc)
                    finally:
                        _infer_job_started_at = None
```

**Step 4: Run test to verify it passes**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestInferenceWatchdog::test_job_started_at_set_during_execution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: track inference job start time in PriorityInferQueue worker"
```

---

### Task 2: Add `_inference_watchdog` background task with `os._exit(1)`

**Files:**
- Modify: `server.py:215-228` (lifespan), `server.py:901-913` (after _idle_watchdog)
- Test: `server_test.py` (add to TestInferenceWatchdog class)

**Step 1: Write the failing test**

Add to `TestInferenceWatchdog` in `server_test.py`:

```python
    def test_watchdog_exits_on_stuck_job(self):
        """Watchdog calls os._exit(1) when a job exceeds the timeout."""
        async def run():
            server._infer_job_started_at = time.monotonic() - 999  # stuck for 999s
            with patch.object(server, "REQUEST_TIMEOUT", 10), \
                 patch("os._exit") as mock_exit:
                await server._inference_watchdog_check()
                mock_exit.assert_called_once_with(1)

        asyncio.run(run())

    def test_watchdog_no_exit_when_idle(self):
        """Watchdog does nothing when no job is running."""
        async def run():
            server._infer_job_started_at = None
            with patch("os._exit") as mock_exit:
                await server._inference_watchdog_check()
                mock_exit.assert_not_called()

        asyncio.run(run())

    def test_watchdog_no_exit_within_timeout(self):
        """Watchdog does nothing when job is within timeout."""
        async def run():
            server._infer_job_started_at = time.monotonic()  # just started
            with patch.object(server, "REQUEST_TIMEOUT", 300), \
                 patch("os._exit") as mock_exit:
                await server._inference_watchdog_check()
                mock_exit.assert_not_called()

        asyncio.run(run())
```

Also add `import time` at the top of the test file if not already present.

**Step 2: Run tests to verify they fail**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestInferenceWatchdog -v`
Expected: FAIL — `server._inference_watchdog_check` does not exist

**Step 3: Implement `_inference_watchdog_check` and `_inference_watchdog`**

Add after `_idle_watchdog` in `server.py` (after line ~913):

```python
_WATCHDOG_GRACE = 30  # seconds beyond REQUEST_TIMEOUT before kill


async def _inference_watchdog_check() -> None:
    """Single check: if a job has been running too long, exit the process."""
    started = _infer_job_started_at
    if started is None:
        return
    elapsed = time.monotonic() - started
    deadline = REQUEST_TIMEOUT + _WATCHDOG_GRACE
    if elapsed > deadline:
        logger.bind(elapsed_s=round(elapsed), deadline_s=deadline).critical(
            "Inference watchdog: job stuck, terminating process"
        )
        os._exit(1)


async def _inference_watchdog() -> None:
    """Background loop that checks for stuck inference jobs every 10 seconds."""
    while True:
        await asyncio.sleep(10)
        await _inference_watchdog_check()
```

In `lifespan()` (line ~223), add after the `_idle_watchdog` task creation:

```python
    asyncio.create_task(_inference_watchdog())
```

**Step 4: Run tests to verify they pass**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestInferenceWatchdog -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: add inference watchdog that self-terminates on stuck GPU jobs"
```

---

### Task 3: Route `_stream_synthesize` through PriorityInferQueue with timeout

**Files:**
- Modify: `server.py:1348-1369` (_stream_synthesize function)
- Test: `server_test.py` (new TestStreamSynthesizeDeadlockPrevention class)

**Step 1: Write the failing tests**

Add to `server_test.py`:

```python
class TestStreamSynthesizeDeadlockPrevention:
    """Tests for _stream_synthesize deadlock prevention."""

    def test_stream_uses_infer_queue_not_executor(self):
        """_stream_synthesize submits work through _infer_queue, not _infer_executor directly."""
        chunks_received = []

        async def run():
            # Mock _do_synthesize_streaming to yield 2 chunks
            def mock_streaming(*args, **kwargs):
                yield np.zeros(100), 24000
                yield np.zeros(100), 24000

            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_infer_queue", queue), \
                 patch.object(server, "_do_synthesize_streaming", mock_streaming), \
                 patch.object(server, "REQUEST_TIMEOUT", 5):
                async for chunk, sr in server._stream_synthesize("hello", "en", "ryan.wav", {}):
                    chunks_received.append(chunk)

        asyncio.run(run())
        assert len(chunks_received) == 2

    def test_stream_timeout_on_stuck_queue_get(self):
        """queue.get() raises TimeoutError if no chunks arrive within REQUEST_TIMEOUT."""
        async def run():
            # Mock _do_synthesize_streaming to block forever
            import threading
            block = threading.Event()

            def mock_streaming(*args, **kwargs):
                block.wait()  # never unblocked
                yield np.zeros(100), 24000

            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_infer_queue", queue), \
                 patch.object(server, "_do_synthesize_streaming", mock_streaming), \
                 patch.object(server, "REQUEST_TIMEOUT", 0.2):
                with pytest.raises(asyncio.TimeoutError):
                    async for chunk, sr in server._stream_synthesize("hello", "en", "ryan.wav", {}):
                        pass
            block.set()  # unblock so thread can exit

        asyncio.run(run())

    def test_stream_sentinel_always_sent_on_error(self):
        """_run() sends sentinel even when _do_synthesize_streaming raises."""
        chunks_received = []
        errors_received = []

        async def run():
            def mock_streaming(*args, **kwargs):
                yield np.zeros(100), 24000
                raise RuntimeError("GPU OOM")

            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_infer_queue", queue), \
                 patch.object(server, "_do_synthesize_streaming", mock_streaming), \
                 patch.object(server, "REQUEST_TIMEOUT", 5):
                try:
                    async for chunk, sr in server._stream_synthesize("hello", "en", "ryan.wav", {}):
                        chunks_received.append(chunk)
                except RuntimeError as e:
                    errors_received.append(str(e))

        asyncio.run(run())
        assert len(chunks_received) == 1  # got first chunk before error
        assert len(errors_received) == 1
        assert "GPU OOM" in errors_received[0]
```

**Step 2: Run tests to verify they fail**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestStreamSynthesizeDeadlockPrevention -v`
Expected: FAIL — streaming still uses `_infer_executor.submit` directly, no timeout on `queue.get()`

**Step 3: Rewrite `_stream_synthesize` to use queue + timeout + robust sentinel**

Replace `server.py` lines 1348-1369 with:

```python
async def _stream_synthesize(text, language, voice_file, gen_kwargs):
    """Async generator bridging sync GPU streaming -> async endpoint.

    Routes through _infer_queue (not _infer_executor directly) so streaming
    jobs are serialized with batch/single jobs.  queue.get() has a timeout
    to prevent infinite hangs if the GPU thread stalls.
    """
    chunk_queue = asyncio.Queue(maxsize=2)
    loop = asyncio.get_running_loop()

    def _run():
        sentinel_sent = False
        try:
            for chunk, sr in _do_synthesize_streaming(text, language, voice_file, gen_kwargs):
                asyncio.run_coroutine_threadsafe(chunk_queue.put((chunk, sr)), loop).result()
            asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop).result()
            sentinel_sent = True
        except Exception as e:
            try:
                asyncio.run_coroutine_threadsafe(chunk_queue.put(e), loop).result()
                sentinel_sent = True
            except Exception:
                pass
        finally:
            if not sentinel_sent:
                try:
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop).result(timeout=5)
                except Exception:
                    pass

    # Submit through PriorityInferQueue instead of _infer_executor directly.
    # We don't await the submit future — we consume chunks from chunk_queue.
    asyncio.ensure_future(_infer_queue.submit(_run, priority=PRIORITY_REALTIME))

    while True:
        item = await asyncio.wait_for(chunk_queue.get(), timeout=REQUEST_TIMEOUT)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item
```

**Step 4: Run tests to verify they pass**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestStreamSynthesizeDeadlockPrevention -v`
Expected: PASS (all 3 tests)

**Step 5: Run all existing tests to check for regressions**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py -v`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: route streaming through PriorityInferQueue with timeout on queue.get()"
```

---

### Task 4: Add queue depth tracking to streaming endpoints

**Files:**
- Modify: `server.py:1489-1643` (/stream endpoint), `server.py:1741-1890` (/stream/pcm endpoint), `server.py:1893-end` (/ws endpoint)
- Test: `server_test.py` (new TestStreamingQueueDepth class)

**Step 1: Write the failing test**

Add to `server_test.py`:

```python
class TestStreamingQueueDepth:
    """Streaming endpoints track queue depth via _queue_tracker."""

    def test_stream_endpoint_acquires_and_releases_queue(self):
        """POST /v1/audio/speech/stream calls _queue_tracker.acquire and release."""
        acquire_calls = []
        release_calls = []

        original_acquire = server._queue_tracker.acquire
        original_release = server._queue_tracker.release

        async def mock_acquire(request_id, endpoint):
            acquire_calls.append((request_id, endpoint))

        async def mock_release():
            release_calls.append(True)

        async def run():
            from httpx import AsyncClient, ASGITransport

            # Mock out model loading and synthesis
            with patch.object(server, "model", MagicMock()), \
                 patch.object(server, "_voice_prompts", {"ryan.wav": []}), \
                 patch.object(server, "_HAS_STREAMING", False), \
                 patch.object(server, "STREAM_TYPE", "sentence"), \
                 patch.object(server._queue_tracker, "acquire", mock_acquire), \
                 patch.object(server._queue_tracker, "release", mock_release), \
                 patch.object(server, "_infer_queue") as mock_queue:
                # Make submit return immediately with fake audio
                mock_future = asyncio.get_running_loop().create_future()
                mock_future.set_result(([np.zeros(100)], 24000))
                mock_queue.submit = MagicMock(return_value=mock_future)

                transport = ASGITransport(app=server.app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/v1/audio/speech/stream", json={
                        "input": "Hello world.",
                        "voice": "alloy",
                    })

            assert len(acquire_calls) == 1
            assert acquire_calls[0][1] == "/v1/audio/speech/stream"
            assert len(release_calls) == 1

        asyncio.run(run())
```

**Step 2: Run test to verify it fails**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestStreamingQueueDepth -v`
Expected: FAIL — streaming endpoint doesn't call `_queue_tracker`

**Step 3: Add `_queue_tracker.acquire/release` to streaming endpoints**

For `/v1/audio/speech/stream` (around line 1489-1643):

Add after `await _ensure_model_loaded()` (line 1494):
```python
    await _queue_tracker.acquire(request_id, "/v1/audio/speech/stream")
```

Wrap the `generate()` inner function and return in a try/finally. The `release` needs to happen after the streaming response is fully consumed. Since `StreamingResponse` consumes the generator lazily, put the release at the end of `generate()` in a finally:

In the `generate()` async generator, wrap the entire body in try/finally:
```python
    async def generate():
        try:
            # ... existing generate body ...
        finally:
            await _queue_tracker.release()
```

Apply the same pattern to `/v1/audio/speech/stream/pcm` (around line 1741):
- Add `await _queue_tracker.acquire(request_id, "/v1/audio/speech/stream/pcm")` after `_ensure_model_loaded()`
- Wrap `pcm_generator()` body in try/finally with `await _queue_tracker.release()`

For `/v1/audio/speech/ws` (around line 1893):
- Add `await _queue_tracker.acquire(request_id, "/v1/audio/speech/ws")` after `_ensure_model_loaded()` (line 1919), per-message
- Add `await _queue_tracker.release()` at the end of each message processing, in a finally block around the synthesis logic

**Step 4: Run test to verify it passes**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py::TestStreamingQueueDepth -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: track queue depth for streaming endpoints (/stream, /stream/pcm, /ws)"
```

---

### Task 5: Run full test suite, create GitHub issue, PR, merge, and update docs

**Files:**
- Modify: `CHANGELOG.md`, `LEARNING_LOG.md`, `README.md`

**Step 1: Run full test suite**

Run: `cd /volume3/docker/qwen3-tts && python -m pytest server_test.py -v`
Expected: All tests PASS

**Step 2: Create GitHub issue**

```bash
gh issue create --title "fix: inference deadlock when streaming GPU thread stalls" \
  --body "## Problem
Streaming inference (\`_stream_synthesize\`) bypasses PriorityInferQueue, submitting directly to the single-thread executor. When GPU hangs, the executor thread is permanently occupied, blocking all inference (batch and streaming). Health checks pass but synthesis requests hang forever.

## Root Cause
1. Streaming bypasses PriorityInferQueue — uses \`_infer_executor.submit()\` directly
2. No timeout on \`asyncio.Queue.get()\` — hangs forever if GPU stalls
3. Streaming endpoints don't track \`_queue_tracker\` depth
4. No watchdog to detect/recover from stuck inference threads

## Solution
- Route streaming through PriorityInferQueue
- Add \`REQUEST_TIMEOUT\` to \`queue.get()\` calls
- Add \`_queue_tracker.acquire/release\` to streaming endpoints
- Add \`_inference_watchdog\` that calls \`os._exit(1)\` when a job exceeds \`REQUEST_TIMEOUT + 30s\`
- Robust sentinel handling in \`_run()\` with finally block

Design: docs/plans/2026-03-01-deadlock-prevention-design.md"
```

**Step 3: Create branch, commit, push, PR**

```bash
git checkout -b fix/issue-N-deadlock-prevention
# (commits already made per task)
git push -u origin fix/issue-N-deadlock-prevention
gh pr create --title "fix: prevent inference deadlock on streaming GPU hang" \
  --body "Closes #N

## Changes
- Route streaming through PriorityInferQueue (was bypassing it)
- Add REQUEST_TIMEOUT to queue.get() in _stream_synthesize
- Add _queue_tracker tracking to /stream, /stream/pcm, /ws endpoints
- Add _inference_watchdog: self-terminates via os._exit(1) on stuck jobs
- Robust sentinel propagation in streaming _run() finally block

## Test plan
- [ ] Unit tests pass: \`pytest server_test.py -v\`
- [ ] Manual: restart container, confirm synthesis works
- [ ] Manual: confirm health endpoint shows queue_depth for streaming requests"
```

**Step 4: Merge PR and update docs**

Update CHANGELOG.md with the fix entry under a new version. Update LEARNING_LOG.md with the deadlock root cause analysis. Update README.md if any environment variables or behavior changed (the `_WATCHDOG_GRACE` constant is internal, no new env vars).

**Step 5: Final commit and push**

```bash
git add CHANGELOG.md LEARNING_LOG.md
git commit -m "docs: update CHANGELOG and LEARNING_LOG for deadlock prevention fix"
git push
```
