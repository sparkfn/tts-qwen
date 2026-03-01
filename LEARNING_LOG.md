# Learning Log

Decisions, patterns, and lessons from building the Qwen3-TTS server. Each entry is written so someone new to the project can understand the reasoning without digging through commit history.

---

## Entry 0028 — Inference deadlock: why streaming must go through the queue
**Date**: 2026-03-01
**Type**: What just happened
**Related**: Issue #112 — Inference deadlock when streaming GPU thread stalls

The TTS server deadlocked in production: health checks passed but every synthesis request hung forever. Root cause: `_stream_synthesize()` bypassed the `PriorityInferQueue` and called `_infer_executor.submit()` directly. When a streaming GPU inference hung (CUDA OOM from a competing process using 4+ GiB), the executor's single thread was permanently occupied. Since both the queue worker and streaming shared the same `ThreadPoolExecutor(max_workers=1)`, all subsequent inference — batch and streaming — was blocked.

Three things made this unrecoverable: (1) `asyncio.Queue.get()` had no timeout, so the streaming consumer waited forever; (2) Python can't kill a running thread in a `ThreadPoolExecutor`, so the stuck thread couldn't be reclaimed; (3) no watchdog existed to detect the stuck state.

The fix routes all streaming through `PriorityInferQueue`, adds `REQUEST_TIMEOUT` to `queue.get()`, and adds an inference watchdog that calls `os._exit(1)` when a job exceeds `REQUEST_TIMEOUT + 30s`. The self-terminate approach is deliberate — once a GPU thread is stuck, the process is in an unrecoverable state. Trying to "recreate the executor" would leak the stuck thread (still holding GPU memory) and risk CUDA context corruption. A clean `os._exit(1)` lets Docker restart with a fresh process and clean GPU state.

The sentinel pattern in `_run()` also got a `finally` block. Previously, if `asyncio.run_coroutine_threadsafe(...).result()` itself failed (e.g., event loop already closed because the client disconnected), the chunk queue never received the termination sentinel, leaving the consumer hanging indefinitely.

---

## Entry 0027 — Per-token streaming: fork-based architecture for sub-sentence latency
**Date**: 2026-03-01
**Type**: Why this design
**Related**: Issue #110 — Per-token streaming

The sentence-chunked streaming endpoints (`/stream`, `/stream/pcm`, `/ws`) have a fundamental latency floor: the client hears nothing until the first sentence is fully synthesized. For short sentences that's ~400ms, but for long ones it can be seconds. Per-token streaming eliminates this by emitting audio fragments as the model generates tokens, before any sentence is complete.

This required a fork of the qwen-tts library (rekuenkdr/Qwen3-TTS-streaming) that exposes `stream_generate_voice_clone()` — a generator yielding audio frames during generation. The server wraps this in `_do_synthesize_streaming()` (sync generator) and `_stream_synthesize()` (async bridge using `asyncio.Queue`). Two env vars control emission cadence: `STREAM_EMIT_FRAMES` (frames between emissions, default 4) and `STREAM_FIRST_EMIT` (first-chunk interval, default 3 for faster initial audio).

The dual-mode architecture (`STREAM_TYPE=sentence` vs `token`) preserves backward compatibility. If the streaming fork isn't installed, the server detects this at import time (`_HAS_STREAMING` flag) and falls back to sentence mode with a warning log. This means the same Docker image works with or without the fork — useful during migration.

---

## Entry 0026 — Standardized error responses: why {detail} isn't enough
**Date**: 2026-03-01
**Type**: Why this design
**Related**: Issue #109 — Standardize error response shape

FastAPI's default error shape `{"detail": "..."}` gives callers a string. That's fine for humans reading logs but terrible for programmatic error handling — callers have to parse the string to figure out what went wrong. The standard shape `{code, message, context, statusCode}` gives each error a machine-readable code (`QUEUE_FULL`, `UNKNOWN_VOICE`, `SYNTHESIS_TIMEOUT`, etc.) that callers can switch on, a human-readable message, optional context dict for debugging, and the HTTP status code echoed in the body for clients that lose access to headers (proxies, message queues). The `APIError` exception class and FastAPI exception handlers ensure every error path — including unhandled exceptions — produces this shape consistently.

---

## Entry 0025 — Startup env validation: fail fast, fail clearly
**Date**: 2026-03-01
**Type**: What could go wrong
**Related**: Issue #108 — Startup env validation

Before `_validate_env()`, misconfigured environment variables caused cryptic runtime failures. Setting `QUANTIZE=fp16` (invalid — valid values are `int8`, `fp8`, or empty) would pass startup silently and fail during model load with an inscrutable torchao error. Setting `LOG_LEVEL=VERBOSE` (invalid) would silently default to INFO. The validation function checks every env var at startup and exits immediately with a clear error message listing what's wrong and what the valid values are. The `.env.example` file serves double duty: it documents every variable and provides safe defaults for `cp .env.example .env` workflows.

---

## Entry 0024 — Structured logging standardization: ISO 8601, ops levels, stdout
**Date**: 2026-03-01
**Type**: Why this design
**Related**: Issue #107 — Standardize structured logging output

Three small changes with outsized operational impact. (1) Full ISO 8601 timestamps with timezone and fractional seconds (`2026-03-01T14:30:00.123+09:00`) — log aggregators (Loki, CloudWatch, Datadog) parse this natively without custom timestamp formats. (2) Level names remapped to ops convention (`fatal/error/warn/info/debug/trace` instead of Python's `CRITICAL/ERROR/WARNING/INFO/DEBUG`) — consistent with every other service in a typical stack. (3) JSON output to stdout instead of stderr — Docker and container orchestrators capture stdout by default, and mixing stdout/stderr causes interleaving issues in log pipelines.

---

## Entry 0023 — Performance optimizations: CUDA streams, sentence pipelining, and FP8
**Date**: 2026-03-01
**Type**: What just happened
**Related**: Performance optimization work (no issue numbers)

Three categories of performance work, each targeting a different bottleneck:

**GPU utilization** — Separate CUDA streams for inference and host transfer let the GPU overlap model execution with copying results to CPU. Combined with `torch.compile(mode="max-autotune")` and CUDA graph capture via triton backend, this squeezes more throughput from the same hardware. The `max-autotune` mode is slower on first inference (kernel autotuning) but faster on every subsequent call — the right tradeoff for a long-running server.

**Streaming latency** — Sentence pipelining submits sentence N+1 to the GPU while sentence N's audio is being yielded to the client. The client perceives continuous audio with no inter-sentence gap, even though synthesis is sequential on the GPU.

**VRAM efficiency** — FP8 quantization via torchao (`Float8WeightOnlyConfig`) reduces model weights to 8-bit floats post-load, dropping VRAM from ~2600 MiB to ~2088 MiB. INT8 via bitsandbytes was attempted first but failed — `deepcopy` in the library's `get_keys_to_not_convert` can't pickle the Qwen3-TTS model. The FP8 path also required using the correct torchao API: `Float8WeightOnlyConfig` applied post-load, not `TorchAoConfig` at load time which doesn't exist in current torchao.

---

## Entry 0022 — Loguru migration: intercepting stdlib loggers
**Date**: 2026-02-24
**Type**: Why this design
**Related**: Loguru migration (v0.9.0)

Python's stdlib logging and loguru are two completely separate logging systems. FastAPI and uvicorn use stdlib logging internally, so simply switching the application code to loguru leaves framework logs flowing through a different pipeline — different formats, different outputs, sometimes even different destinations. The `_InterceptHandler` bridges this: it's a stdlib `logging.Handler` subclass that intercepts every stdlib log record and re-emits it through loguru. Installed on the root logger, it captures everything — uvicorn access logs, FastAPI error traces, third-party library warnings — and routes them all through loguru's formatter. This means one log format, one output, one configuration point, regardless of which library emitted the log.

---

## Entry 0021 — Why VAD doesn't belong in a TTS pipeline
**Date**: 2026-02-24
**Type**: What could go wrong
**Related**: Issue #100 — Remove VAD trim from TTS pipeline

Voice Activity Detection (VAD) finds speech in recordings that contain a mix of speech, silence, and background noise. TTS output is none of those things — it's synthesized audio that the model generated on purpose. Running VAD on TTS output is applying a recording-analysis tool to a signal-generation output. The failure mode is that VAD can aggressively trim valid speech content (soft consonants, trailing words) because it was designed to detect human speech patterns in noisy environments, not to quality-check a model's output. If the model generates silence padding, the right fix is at the model output level, not a post-hoc recording analyzer. Removed entirely — audio is returned as-is after speed adjustment and encoding.

---

## Entry 0020 — Multi-stage Docker builds with conda base images
**Date**: 2026-02-24
**Type**: What could go wrong
**Related**: Issue #94 — Docker build fails

The `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image uses conda, not system Python. Python lives at `/opt/conda/bin/python` and packages at `/opt/conda/lib/python3.11/site-packages/`. The standard multi-stage pattern of `pip install --prefix=/install` then `COPY --from=builder /install /usr/local` fails silently — packages land in `/usr/local/lib/python3.11/site-packages/` which conda's Python doesn't search. The fix is to COPY directly into the conda tree: `/opt/conda/lib/python3.11/site-packages/` for packages and `/opt/conda/bin/` for scripts. Additionally, `torchao` versions must match the base torch version exactly (0.5.x for torch 2.5.x), and even compatible torchao versions can break newer transformers due to missing quantization APIs.

---

## Entry 0019 — Queue depth counters must cover all exit paths
**Date**: 2026-02-24
**Type**: What could go wrong
**Related**: Issue #97 — Queue depth counter leaks on cache hits

The `_queue_depth += 1` counter was placed before both the cache-hit early-return and the `_ensure_model_loaded()` call, but only the `try/finally` around the inference block decremented it. Cache hits returned early without decrementing, and model-load failures raised exceptions that bypassed the finally block. After a few cache hits, `_queue_depth` permanently reached `MAX_QUEUE_DEPTH` and every subsequent request got 503. The fix: wrap everything after the increment in a single `try/finally`. The aha moment is that any counter increment must be immediately followed by its decrement guarantee — never separated by any code that might exit early or raise.

---

## Entry 0017 — Batch inference: draining the queue for free GPU utilization
**Date**: 2026-02-24
**Type**: Why this design
**Related**: Issue #84 — Add batch inference for concurrent synthesis requests

The priority queue (Issue #81) serializes GPU inference, but under concurrent load each request still waits in line for its own dedicated model call. Transformer-based TTS models like Qwen3-TTS support batched inputs natively via `generate_custom_voice(text=[...])`, where multiple texts are padded and processed in a single forward pass. The GPU's parallel compute units are underutilized when processing one short text at a time, so batching N requests into one call yields near-1x inference time instead of Nx.

The implementation drains the priority queue's heap: when the worker picks up a synthesis job and `MAX_BATCH_SIZE > 1`, it pops all pending synthesis jobs (up to `MAX_BATCH_SIZE`) in one lock pass, collects their texts/languages/speakers, and dispatches a single `_do_synthesize_batch()` call. Results fan out: each job's future gets its individual wav. If the batch call fails, all futures receive the same exception. `MAX_BATCH_SIZE` (env var, default 4) caps the drain to prevent excessive padding waste and memory spikes. Setting it to 1 disables batching entirely, falling back to the original single-dispatch path.

The `instruct` parameter is not supported in the batch path because it is per-request and the model's batch API does not support mixed instruct/non-instruct calls. Requests with `instruct` set fall back to single-job dispatch automatically. Voice clone requests are similarly excluded since they use a different model method (`generate_voice_clone`).

---

## Entry 0018 — Gateway/Worker mode: why two processes beat one idle process
**Date**: 2026-02-24
**Type**: Why this design
**Related**: Issue #85 — Add Gateway/Worker mode for minimal idle footprint

The idle RAM problem: even with `IDLE_TIMEOUT` unloading the model from VRAM, the Python process still holds ~1 GB RSS from PyTorch, CUDA runtime, and the loaded server modules. In shared GPU environments where the TTS service may sit idle for hours between requests, that memory is wasted.

The two-process split solves this by separating concerns. `gateway.py` is a minimal FastAPI proxy (~30 MB RSS) that knows nothing about models or inference. It spawns `worker.py` (the full TTS server) as a subprocess only when a request arrives, then kills it after `IDLE_TIMEOUT` seconds of inactivity. When the worker dies, all its memory (Python heap, CUDA context, model weights) is reclaimed by the OS — something that is impossible to achieve within a single Python process due to allocator fragmentation.

The gateway uses `aiohttp.ClientSession` to proxy all HTTP requests transparently to the worker on an internal port (`WORKER_PORT=8001`). A double-checked lock (`_worker_lock`) prevents concurrent request storms from spawning multiple workers. The idle watchdog runs every 30 seconds, checking `_last_used` against `IDLE_TIMEOUT`.

The main trade-off is cold start latency: the first request after an idle kill pays the full model load time (~10-15 seconds). This is acceptable for the target use case (shared GPU, infrequent usage) where memory savings outweigh startup latency.

Note: streaming (SSE) and WebSocket endpoints are not proxied correctly in this initial version — `gateway.py` buffers full responses. These are known limitations documented for a follow-up issue.

---

## Entry 0019 — Quantization: when to trade precision for VRAM
**Date**: 2026-02-24
**Type**: Why this design
**Related**: Issue #86 — Add quantization support (INT8/FP8)

The Qwen3-TTS-0.6B model consumes ~2.4 GB VRAM in bfloat16. On shared GPU environments where multiple services compete for a single card, that footprint matters. INT8 quantization via bitsandbytes (`QUANTIZE=int8`) cuts VRAM roughly in half to ~1.2 GB at the cost of 10-20% slower inference. FP8 via torchao (`QUANTIZE=fp8`) is more aggressive — ~0.8 GB — with minimal speed impact, but only works on Hopper (H100) and newer architectures. We keep bfloat16 as the default because maximum audio quality matters more than VRAM savings for most deployments.

Both `bitsandbytes` and `torchao` are large packages that most users don't need. They're listed in `requirements.txt` as optional and installed with `|| true` in the Dockerfile so builds don't break on CPU-only hosts. The `_resolve_quant_kwargs()` helper validates `QUANTIZE` at model load time — not at request time — so a misconfigured env var fails fast with a clear error message instead of silently producing degraded output or crashing on the first inference call.

---

## Entry 0016 — Priority inference queue: why a semaphore is not enough
**Date**: 2026-02-24
**Type**: What just happened
**Related**: Issue #81 — Replace inference semaphore with priority queue

The original server used `asyncio.Semaphore(1)` to serialize GPU inference. This was correct for preventing OOM, but it treated all requests equally. A 10-sentence batch REST request and a real-time WebSocket stream competing for the semaphore had equal chances of acquiring it. For a TTS server serving both interactive clients (WebSocket, SSE streaming, raw PCM) and bulk callers (REST `/v1/audio/speech`), this is wrong. A user on a phone call hearing silence while their stream waits behind a batch job is a perceptible quality regression.

The replacement is a `PriorityInferQueue` backed by `heapq` (stdlib min-heap). Jobs are `@dataclass(order=True)` with `(priority, submit_time)` as the sort key. Lower priority number means higher urgency: `PRIORITY_REALTIME=0` for streaming endpoints, `PRIORITY_BATCH=1` for REST. When multiple jobs are queued, the worker always pops the lowest-priority-number job first. Within the same priority level, `submit_time` (monotonic clock) ensures FIFO ordering.

The queue uses `asyncio.Lock` (not a threading lock) because all heap mutations happen in async code. An `asyncio.Event` wakes the worker when new jobs arrive, avoiding busy-waiting. The worker runs `loop.run_in_executor()` on the same single-thread `_infer_executor`, preserving the guarantee that only one GPU inference runs at a time.

The key design choice was keeping the queue as a module-level singleton (`_infer_queue`) started from `lifespan()`. This matches the existing pattern for `_idle_watchdog` and keeps the startup/shutdown lifecycle centralized. Each endpoint simply calls `await _infer_queue.submit(fn, priority=...)` instead of `async with _infer_semaphore:` followed by `loop.run_in_executor(...)`.

Five sites were updated: `synthesize_speech` (batch), `synthesize_speech_stream` (realtime), `clone_voice` (batch), `synthesize_speech_stream_pcm` (realtime), and `ws_synthesize` (realtime). The `loop = asyncio.get_running_loop()` lines at those sites were removed since the queue handles the executor call internally.

---

## Entry 0015 — Voice clone prompt cache: caching at the right layer
**Date**: 2026-02-24
**Type**: Why this design
**Related**: Issue #82 — Fix voice clone caching — use `create_voice_clone_prompt()`

The original `_voice_cache` (#15) stored `(np.ndarray, int)` — the decoded reference audio array and sample rate. This saved the cost of reading and decoding the uploaded WAV file (~1ms), but missed the actual bottleneck: computing the speaker embedding inside `model.generate_voice_clone()`. Every clone request, even with identical reference audio, ran the full encoder pass to extract the voice embedding. This pass dominates the preprocessing cost (50-200ms depending on audio length).

The fix is to cache at the embedding layer, not the audio layer. `model.create_voice_clone_prompt()` takes the decoded audio and reference text, runs the encoder once, and returns a reusable prompt object containing the speaker embedding. Subsequent `generate_voice_clone()` calls accept this prompt via `ref_prompt=` and skip the encoder entirely.

The cache structure is unchanged — `OrderedDict` with SHA-256 content hash keys, LRU eviction, configurable via `VOICE_CACHE_MAX`. What changed is *what* gets cached: opaque prompt objects instead of numpy arrays. The trade-off is that prompt objects may hold GPU tensors (consuming VRAM proportional to cache size), but at 32 entries default this is negligible compared to the 2.4 GB model weights.

The `/cache/clear` endpoint was also updated to clear both audio and voice prompt caches, returning separate counts for each. This is a minor API change (the response shape changed from `{"cleared": N}` to `{"audio_cleared": N, "voice_cleared": M}`).

Key observation: caching at the wrong layer is worse than no caching at all — it gives operators false confidence that "caching is working" while the expensive operation still runs on every request.

---

## Entry 0014 — Exposing generation parameters: temperature and top_p
**Date**: 2026-02-24
**Type**: What just happened
**Related**: Issue #83 — Expose temperature and top_p in TTSRequest

The Qwen3-TTS model's `generate_custom_voice()` and `generate_voice_clone()` accept standard HuggingFace generation kwargs (`temperature`, `top_p`, etc.) but the server hardcoded only `max_new_tokens`. Clients had no way to control generation diversity.

The fix adds two optional fields to `TTSRequest` (both default to `None`) and a `_build_gen_kwargs()` helper that conditionally includes them in the kwargs dict. When `None`, the key is omitted entirely — the model uses its own defaults, preserving exact backward compatibility. This matters because passing `temperature=None` explicitly to HuggingFace's `generate()` would be different from not passing it at all.

The DRY improvement is incidental but valuable: four endpoints had identical `gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(text)}` lines. The helper centralizes this pattern. Two endpoints (clone and WebSocket) build kwargs inline because they don't use `TTSRequest` — clone uses `Form` params and WebSocket uses raw JSON.

A pre-existing bug was fixed in the clone endpoint: `_adaptive_max_tokens(text)` was called before `text = input.strip()`, using an undefined variable. The reordering to assign `text` first was necessary for correctness regardless of the temperature/top_p feature.

---

## Entry 0012 — GPU memory pool pre-warming and CUDA allocator tuning
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #16 — Pre-allocate GPU memory pool to reduce allocation jitter

CUDA memory allocation is lazy by default. The first time a tensor of a given size is allocated, the CUDA allocator calls `cudaMalloc`, which involves a kernel-mode transition and device synchronization. This takes 1-5ms per allocation. For a TTS server handling its first request after model load, there are multiple novel allocation sizes (KV-cache, attention intermediates, audio output buffers), each paying this penalty. The cumulative cost can add 10-30ms to the first inference.

The fix is a dummy allocation: after warmup, allocate a 128 MB tensor (`torch.empty(64*1024*1024, dtype=bfloat16, device="cuda")`) and immediately delete it. This forces the CUDA allocator to reserve a contiguous 128 MB block in its free pool. Subsequent allocations that fit within this block are served from the pool without `cudaMalloc` calls.

The `max_split_size_mb:512` addition to `PYTORCH_CUDA_ALLOC_CONF` prevents the allocator from splitting large cached blocks into small fragments. Without this, the allocator might split a 128 MB cached block into many small pieces to serve a 1 MB request, then not be able to recombine them when a 64 MB request arrives.

---

## Entry 0008 — Why pitch-preserving time stretch matters for TTS
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #14 — Replace scipy speed adjustment with pitch-preserving pyrubberband

The original speed adjustment used `scipy.signal.resample()` to change the number of audio samples. This is a frequency-domain resampling that compresses or expands the waveform uniformly. The problem: when you compress samples to make audio play faster, the fundamental frequency of the voice shifts upward proportionally. At speed=1.5, the voice pitch rises by 50%, creating the classic "chipmunk" effect. At speed=0.75, the pitch drops, making the voice sound artificially deep.

For a TTS server, this is unacceptable. Speed adjustment is used for accessibility (slower speech for comprehension), time fitting (faster speech for constrained UIs), and prosody matching (adjusting pace to context). In all cases, the user expects the voice to sound like the same person speaking at a different pace, not a pitch-shifted version.

pyrubberband wraps the Rubber Band Library, which implements PSOLA (Pitch-Synchronous Overlap-Add) time stretching. PSOLA works by identifying pitch periods in the audio, duplicating or removing complete pitch cycles, and crossfading at zero-crossing points. The result is audio that plays faster or slower without any pitch change. The algorithm is well-established in audio processing and adds negligible latency (< 10ms for typical TTS output lengths).

The implementation uses the same graceful-fallback pattern as other optional dependencies: if `pyrubberband` is not importable, `_pyrubberband` is set to `None` and the function falls back to `scipy.signal.resample`. This keeps the server functional on systems without the rubberband-cli binary installed, at the cost of the pitch shift artifact.

The Dockerfile installs both `pyrubberband` (Python bindings) and `rubberband-cli` (the C++ binary that pyrubberband calls). The binary is available in Ubuntu's package manager, so no compilation is needed.

---

## Entry 0001 — Project baseline: current architecture
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The server is a single-file FastAPI application that wraps the Qwen3-TTS-0.6B model behind an OpenAI-compatible API. There are four design choices that define the baseline, and all four exist for the same reason: the event loop must never block.

First, model loading is lazy. The model does not load at startup. It loads on the first request and stays resident until an idle timeout fires. This matters because the container shares a GPU with other services on a Synology NAS. Holding 2.4 GB of VRAM permanently when the service might go hours without a request is wasteful. The `_idle_watchdog` background task checks every 30 seconds whether `_last_used` has exceeded `IDLE_TIMEOUT` (default 120 seconds), and if so, unloads the model and calls `torch.cuda.empty_cache()` plus `ipc_collect()` to return VRAM to the system.

Second, all GPU work runs through a dedicated single-thread `ThreadPoolExecutor` via `run_in_executor`. This is the single most important architectural decision. The Qwen3-TTS model's `generate_custom_voice` is a blocking synchronous call that holds the GIL and runs CUDA kernels for 400-2000ms. If this ran directly in an async handler, the entire event loop would freeze — health checks would hang, the idle watchdog would stall, and concurrent HTTP connections would time out. Offloading to a thread executor lets the event loop continue servicing other coroutines while GPU inference runs in a background thread.

Third, an `asyncio.Semaphore(1)` serializes GPU inference. Even though the executor has only one thread, the semaphore is still necessary — it prevents a second request from queuing inside the executor while a first is running. Without it, two requests could both enter `run_in_executor`, and the second would block its event loop coroutine waiting for the executor thread, which is functionally identical to blocking the loop. The semaphore makes the queueing explicit and visible to the async scheduler.

Fourth, an `asyncio.Lock` with double-checked locking protects model load and unload. Two requests arriving simultaneously on a cold server would both see `model is None` and both try to load. The lock ensures only one load happens. The double-check pattern (check before acquiring, check again after acquiring) avoids holding the lock on the hot path when the model is already loaded.

The critical baseline insight: every community alternative we analyzed (twolven, ValyrianTech) gets the event loop blocking wrong. They call synchronous model inference directly inside async handlers. Our server is the only one that correctly keeps the event loop responsive during inference. This is not an optimization — it is a correctness requirement.

---

## Entry 0002 — Why the bottleneck is not what you'd expect
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The intuition when looking at a TTS server is that the bottleneck is model inference speed. The model takes 400-600ms to generate a sentence of audio — that feels like the thing to optimize. But the actual user experience problem is different: the server returns zero bytes until the entire synthesis is complete.

Consider a three-sentence paragraph. Total inference time is roughly 1.5 seconds. But the client does not receive the first byte of audio until all three sentences are done. The user hears 1.5 seconds of silence, then the full audio plays. The perceived latency is the total time, not the per-sentence time.

Now consider what happens with streaming. If the server splits the text into sentences and sends each sentence's audio as soon as it is ready, the first audio arrives after roughly 500ms (one sentence of inference). The user starts hearing speech while the server is still synthesizing the remaining sentences. The total wall-clock time is the same, but the perceived latency drops by 60-70%.

This is why the improvement plan puts streaming (Phase 1) before any inference optimization (Phase 2). A 20% inference speedup on a non-streaming server saves maybe 300ms on a 1.5s total. Streaming saves 1000ms of perceived silence on the same input. The streaming architecture creates a low-latency shell — once it exists, inference speedups compound on top of it because each sentence chunk gets individually faster. But doing inference optimization without streaming means the speed gains are invisible to the user. They still wait for everything to finish before hearing anything.

The ordering is not about difficulty or risk. It is about which layer of the stack the user actually perceives.

---

## Entry 0003 — The max_new_tokens blind spot
**Date**: 2026-02-20
**Type**: What just happened (planning discovery)
**Related**: Planning — pre-issue

During the second pass over the improvement plan, we found something that the first review entirely missed: `max_new_tokens` is hardcoded to 2048 for every request. This line appears in both `synthesize_speech` and `clone_voice`:

```python
gen_kwargs = {"max_new_tokens": 2048}
```

The model's name includes "12Hz" — it generates 12 codec tokens per second of audio output. Average speech runs at about 150 words per minute, which works out to roughly 5 codec tokens per word. A 10-word sentence like "Please hold while I transfer your call" needs approximately 50 tokens. The server is allocating a budget of 2048 tokens — about 40 times what is actually needed.

The model still stops at the EOS token, so the output audio is correct. But the inference engine pre-allocates a KV-cache sized for 2048 tokens and manages attention over that full budget. For short texts, this wastes memory bandwidth and adds overhead to every attention computation. The fix is a simple function that scales the token budget with input length:

```python
def _adaptive_max_tokens(text: str) -> int:
    word_count = len(text.split())
    return max(128, min(2048, word_count * 8))
```

For short inputs (the vast majority of real-world TTS — greetings, IVR prompts, single sentences), the expected latency improvement is 30-60%. For long inputs the budget stays at 2048 and nothing changes.

The aha moment: when reviewing a model server, always read the `gen_kwargs`. The architecture diagram, the async patterns, the concurrency controls — those are what draw your attention during a code review. The generation parameters are one line buried in a handler, and they are easy to gloss over. But they directly control how hard the GPU works per request.

---

## Entry 0004 — Why three phases and in this order
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The improvement plan is organized into three phases, and the ordering is deliberate.

Phase 1 targets perceived latency. It starts with latency instrumentation (measure first — you cannot improve what you cannot see), then fixes the hardcoded `max_new_tokens` budget that wastes GPU cycles on short inputs, and finally delivers streaming — sentence-chunked SSE and raw PCM endpoints. The goal is to change what the user experiences. Even if the model runs at exactly the same speed, the user hears audio sooner because chunks arrive incrementally. The metric that matters here is time-to-first-audio, not total synthesis time. For the primary use case (phone calls), this is the difference between a natural conversation flow and an awkward 2-3 second pause.

Phase 2 targets actual latency. This is where inference gets faster: `flash_attention_2`, `torch.compile`, TF32 matmul, GPU clock locking. Each of these makes the model produce audio faster in wall-clock time. But notice — these gains only feel impactful because streaming is already in place. A 20% speedup on a 500ms sentence chunk means the user hears audio 100ms sooner. Without streaming, the same 20% speedup on a 1.5s total synthesis saves 300ms of wait time that the user still experiences as a single block of silence. The streaming shell amplifies the perceived impact of every inference optimization that follows.

Phase 3 targets operational correctness. Prometheus metrics, structured JSON logging, dependency pinning, lifecycle improvements. Phase 3 comes last not because it is least important, but because you need a working, optimized system to instrument meaningfully. Basic latency logging exists from Phase 1, but the production-grade observability stack (Prometheus, structured logs with per-request fields, request queue depth limits) needs a stable system to measure against. Instrumenting after streaming gives you time-to-first-chunk, per-sentence inference time, encode overhead, and queue wait — the metrics you actually need to find the next bottleneck.

The phases form a dependency chain: streaming creates the architecture that makes speed gains visible, and speed gains create the system worth measuring.

---

## Entry 0005 — What could go wrong with the streaming approach
**Date**: 2026-02-20
**Type**: What could go wrong
**Related**: Planning — pre-issue

Sentence-chunked streaming is the highest-leverage change in the plan, but it has at least five failure modes that are not obvious from the implementation sketch.

**Sentence splitting on abbreviations.** A naive regex split on `.!?` will break on "Dr. Smith called at 3 p.m. to confirm." — producing four fragments instead of one sentence. "U.S.A." becomes three splits. The sentence splitter needs an abbreviation-aware tokenizer, not a regex. This is solvable (libraries like `pysbd` handle it), but if you ship the naive version first and discover the bug in production, the audio will have unnatural micro-pauses between "Dr" and "Smith" that sound worse than no streaming at all.

**WAV format and the data_size header.** A WAV file begins with a RIFF header that includes the total data size. You cannot write a valid WAV header until you know how many bytes of audio follow. This means chunked WAV streaming requires either: (a) writing a placeholder header with size 0xFFFFFFFF and hoping the client tolerates it, (b) using raw PCM with no header and documenting the sample format separately, or (c) using a container format designed for streaming like OGG/Opus. The plan includes a raw PCM endpoint for this reason, but any client expecting a standard WAV file will reject a chunked stream.

**Idle timeout and streaming sessions.** The `_last_used` timestamp currently updates once per request. With streaming, a long text might take 10+ seconds to stream all sentence chunks. If `_last_used` is set at the start and the idle timeout is 120 seconds, this is fine. But if `_last_used` is only set once and the next request comes 115 seconds after the stream started, the watchdog sees 115 seconds of idle time and unloads the model. The fix is simple — update `_last_used` after each chunk — but forgetting this will cause the model to unload mid-stream, which is a request failure that only appears under specific timing conditions.

**Reverse proxy buffering.** Nginx, Cloudflare, and most load balancers buffer response bodies by default. A chunked HTTP response that the server sends in 500ms increments will arrive at the client as a single burst after the full response is buffered. The `X-Accel-Buffering: no` header disables this for Nginx, but other proxies need their own configuration. If the deployment sits behind any proxy layer, streaming will appear to not work even though the server is sending chunks correctly. This is an infrastructure problem, not a code problem, and it is invisible during local testing.

**Semaphore serialization between chunks.** The current `Semaphore(1)` serializes all GPU inference. In the streaming flow, each sentence is a separate inference call. Sentence N must complete and release the semaphore before sentence N+1 can acquire it. For a single streaming request this is correct — sentences must be sequential anyway. But if two clients are streaming simultaneously, their sentences interleave: client A sentence 1, client B sentence 1, client A sentence 2, and so on. This doubles the time-to-completion for both clients. The async pipeline optimization (encoding chunk N on CPU while synthesizing chunk N+1 on GPU) can partially hide this, but the fundamental issue is that `Semaphore(1)` means the GPU can only work on one sentence at a time regardless of how many clients are waiting.

---

## Entry 0006 — GPU system tuning: the invisible ms
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

There is a category of optimization that does not appear in any Python code review because it lives in the GPU driver layer. Three settings in particular have outsized impact on latency consistency, and all three are single-command changes.

**GPU persistence mode** (`nvidia-smi -pm 1`). By default, the NVIDIA driver powers down the GPU between workloads. When the next CUDA call arrives, the driver reinitializes the GPU context — this takes 200-500ms. With persistence mode enabled, the GPU stays initialized even when idle. The effect: the first inference after a long idle period is 200-500ms faster. Without it, there is a "double cold start" — the model loads into VRAM (5-10 seconds), then the first inference stalls while the GPU context reinitializes. Users report this as "the first request is always slow" and often attribute it to model loading, but the GPU context initialization is a separate penalty on top of model load time.

**GPU clock locking** (`nvidia-smi -lgc <max>,<max>`). GPUs dynamically scale their clock frequency based on temperature and power draw. For sustained workloads (training, video rendering) this is fine — the clock ramps up within milliseconds and stays there. But TTS inference is bursty: a single request runs for 400-600ms and then the GPU goes idle. The clock may not reach boost frequency before inference completes, meaning every request runs at a slower-than-maximum clock speed. Locking the clocks to the maximum boost frequency eliminates this ramp-up latency and, more importantly, eliminates variance. Without clock locking, two identical requests can have different latencies depending on whether the GPU was already boosted from a recent request. The tradeoff is higher idle power consumption and slightly higher GPU temperature.

**TF32 matmul** (`torch.backends.cuda.matmul.allow_tf32 = True`). On Ampere and newer GPUs (RTX 3000/4000 series, A100, H100), PyTorch defaults to full FP32 for matrix multiplication. TF32 uses the same 8-bit exponent but rounds the mantissa from 23 bits to 10 bits, allowing the operation to use Tensor Core hardware that runs 3x faster. Since the model already runs in bfloat16 (which has the same 8-bit exponent and only 7 bits of mantissa), enabling TF32 for intermediate operations has negligible quality impact. This is two lines of Python, but the effect is an 8-12% inference speedup on supported hardware. It is a no-op on older GPUs.

None of these changes appear in a typical code review. They are infrastructure-level settings. But together, they can reduce p99 latency by 20-30% and virtually eliminate tail latency variance. For a real-time application like phone calls, consistency matters as much as raw speed — a system that is usually fast but occasionally slow feels worse than one that is always moderately fast.

---

## Entry 0008 — Why fasttext over Unicode heuristic for language detection
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #13 — Replace Unicode language heuristic with fasttext detection

The original `detect_language()` function used a character-range heuristic: scan the input text for CJK, Hiragana/Katakana, or Hangul characters, and default to English for anything else. This works for scripts with unique Unicode ranges but fails completely for languages that share the Latin alphabet. French, German, Spanish, Italian, Portuguese, and Russian are all detected as "English" because their characters fall within ASCII or basic Latin ranges.

The Qwen3-TTS model supports all of these languages. A French user sending "Bonjour le monde" gets English prosody applied because the server cannot tell the difference. This is not a theoretical problem — it affects every European language user.

The fix uses `fasttext-langdetect`, which wraps Facebook's fasttext language identification model. It returns ISO 639-1 codes (e.g., "fr", "de", "es") with confidence scores. A mapping dict (`_LANG_MAP`) converts these to the Qwen-expected language names. The implementation is a graceful upgrade: if `fasttext-langdetect` is not installed, the function falls back to the original Unicode heuristic. This means the server works identically without the dependency — it just cannot detect Latin-script languages beyond English.

Key design decisions:
- **Lazy loading**: The fasttext model loads on first call to `detect_language()`, not at import time. This avoids slowing down startup for a model that might not be needed if every request provides an explicit `language` parameter.
- **False sentinel**: `_langdetect_model` uses `False` (not `None`) to distinguish "tried to import and failed" from "haven't tried yet". This prevents retrying the import on every request when the package is genuinely missing.
- **low_memory=False**: The fasttext model is small (~1MB). Loading it fully into memory is faster than the compressed low-memory mode, and the memory cost is negligible compared to the TTS model.
- **Default to English for unknown ISO codes**: If fasttext returns a language code not in `_LANG_MAP` (e.g., "tl" for Tagalog), the function returns "English" rather than passing through the raw code. This is because Qwen3-TTS has a fixed set of supported languages, and an unsupported language name would cause an inference error.

---

## Entry 0007 — The caching hierarchy
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The improvement plan includes three layers of caching, and the ordering from highest to lowest leverage is the opposite of what you might expect.

**Layer 1: Audio output cache** (text + voice + language + speed + format -> final audio bytes). This is an in-memory LRU dict keyed by a SHA-256 hash of the request parameters. A cache hit costs about 1ms (memory lookup plus hash computation). A cache miss costs 500-2000ms (full model inference plus audio encoding). The hit-to-miss ratio depends entirely on the workload. For an IVR system where a menu says "Press 1 for billing, press 2 for support" on every call, the hit rate is effectively 100% after the first call. For a voice assistant generating unique responses, the hit rate is near 0%. The plan defaults to 256 entries and allows disabling via `AUDIO_CACHE_MAX=0`.

**Layer 2: Voice prompt cache** (reference audio bytes -> processed voice embedding). This applies only to the `/clone` endpoint. Voice cloning requires processing the reference audio file on every request: reading the audio, converting to mono if stereo, and passing it to the model's voice cloning pipeline. If the same reference audio is used repeatedly (which is the common case — you pick a voice and use it for all requests), this processing is redundant. Caching the processed audio keyed by a content hash saves roughly 1 second per clone request. This is lower leverage than the output cache because it only saves the preprocessing step, not the inference itself.

**Layer 3: KV prefix cache** (future). If the model's KV-cache for common text prefixes (e.g., "Thank you for calling") could be pre-computed and reused, inference for texts sharing that prefix would skip the prefill phase. This is the most technically complex cache and depends on the model's internals exposing KV-cache manipulation. It is listed as future because it requires deeper integration with the `qwen-tts` library than the other two layers.

The ordering matters for implementation priority. The output cache collapses the entire pipeline for repeated requests — inference, audio encoding, everything. One dict lookup replaces all of it. The voice prompt cache only saves preprocessing. The KV cache only saves part of inference. In terms of implementation effort, the output cache is roughly 20 lines of code. The voice prompt cache is similar. The KV cache is an open research question.

For the phone call use case, the realistic expectation is that the output cache provides the majority of the benefit. IVR menus, hold messages, greeting phrases, and common system responses repeat constantly. A deployment serving 1000 calls per day with 20 unique system phrases would see cache hit rates above 90% after the first few calls. The per-request cost drops from 500ms of GPU inference to 1ms of memory lookup.

---

## Entry 0008 — Audio cache: key design and LRU eviction
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #17

The audio output cache is an OrderedDict keyed by SHA-256 of `text|voice|speed|format|instruct`. The key includes every parameter that affects the output — if any parameter changes, the cache key changes, and a new synthesis runs. The pipe delimiter prevents ambiguity between parameters (e.g., "hello|vivian" vs "hello|" + "vivian").

The cache stores the final encoded bytes and content type, not the raw audio array. This means a cache hit returns the exact HTTP response body — no format conversion, no speed adjustment, no GPU work at all. The cost of a cache hit is one SHA-256 hash (~1 microsecond) plus one OrderedDict lookup (~1 microsecond).

The cache check happens before `_ensure_model_loaded()`. This is deliberate: if every request for the next hour hits the cache, the model never loads, and VRAM stays free. The idle watchdog continues running, but since the model was never loaded, it has nothing to unload. This makes the cache especially valuable in shared GPU environments where VRAM is contended.

LRU eviction uses `OrderedDict.move_to_end()` on hit and `popitem(last=False)` when full. This is O(1) for both operations. The default capacity of 256 entries is sized for a typical IVR deployment where 20-50 unique system phrases repeat across thousands of calls. At roughly 100KB per WAV entry (1 second of 24kHz 16-bit audio), 256 entries consume about 25MB of RAM — negligible compared to the 2.4GB model.

Setting `AUDIO_CACHE_MAX=0` disables all cache operations: `_get_audio_cache` returns None immediately, `_set_audio_cache` is a no-op. This is the safe default for testing or debugging where deterministic behavior is needed.

---

## Entry 0009 — GPU persistence mode and the entrypoint pattern
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #6

We introduced `docker-entrypoint.sh` as the container's ENTRYPOINT, running GPU tuning commands before exec-ing into uvicorn. GPU settings like `nvidia-smi -pm 1` cannot be baked into the image at build time (no GPU during build). The entrypoint runs at container start when the GPU is available via NVIDIA runtime. The `|| echo` pattern ensures the service starts even without sufficient permissions.

---

## Entry 0009 — flash_attention_2: hardware requirements and fallback
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #8

Switched the model's attention implementation from PyTorch's native SDPA to Flash Attention 2. Flash Attention 2 uses fused CUDA kernels that are 15-20% faster and use less memory by avoiding materialization of the full attention matrix.

Hardware requirement: Flash Attention 2 requires Ampere or newer GPUs (compute capability >= 8.0). This means RTX 3000/4000 series, A100, H100. On older hardware (V100, RTX 2000), the `flash-attn` package either won't install or won't work at runtime.

The fallback pattern is a simple try/except on `import flash_attn`. If the import fails, we fall back to `sdpa` (PyTorch's built-in scaled dot product attention). This means the code works on any GPU — it just runs faster on newer ones. The check happens at model load time, not at import time, so the server starts correctly even without flash-attn installed.

---

## Entry 0011 — Voice prompt cache: hash bytes not filenames
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #15 — Add voice prompt cache for /clone endpoint

The voice cloning endpoint processes reference audio on every request: read bytes, decode with soundfile, convert stereo to mono. When the same reference audio is reused across requests (the common case — a user picks a voice and uses it repeatedly), this processing is redundant.

The cache key is a SHA-256 hash of the raw audio bytes, not the filename. Filenames are unreliable — the same file can be uploaded with different names, and different files can share a name. Content hashing guarantees that identical audio produces the same key regardless of how it was uploaded.

The cache uses `collections.OrderedDict` as an LRU. On hit, `move_to_end()` promotes the entry; on insert, `popitem(last=False)` evicts the oldest entry when capacity exceeds `VOICE_CACHE_MAX`. This is simpler than `functools.lru_cache` because the cache key is a hash string (not the raw bytes), and we need manual control over cache size via an env var that can be set to 0 to disable caching entirely.

The health endpoint exposes `voice_cache_size`, `voice_cache_max`, and `voice_cache_hits` so operators can monitor hit rates and tune capacity.

---

## Entry 0010 — torch.compile: the first-inference cost
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #9

Enabled `torch.compile(model.model, mode="reduce-overhead", fullgraph=False)` after model loading. This tells PyTorch to trace the model's forward pass and generate optimized CUDA kernels, eliminating Python overhead on subsequent calls.

The trade-off is first-inference latency. The first call after compilation triggers the tracing/compiling step, which can take 10-30 seconds depending on model size and GPU. After that, every subsequent inference is faster. For a TTS server that loads the model once and runs many requests, this is a clear win -- the compilation cost is amortized across all requests.

`mode="reduce-overhead"` uses CUDA graphs which are ideal for repeated inference with similar-shaped inputs (exactly the TTS use case). `fullgraph=False` allows partial compilation if some operations aren't compilable, avoiding hard failures.

The `TORCH_COMPILE` env var (default true) provides an escape hatch for environments where compilation causes issues (older PyTorch versions, unsupported ops).

---

## Entry 0012 — SSE streaming: base64 PCM over text/event-stream
**Date**: 2026-02-20
**Type**: What just happened
**Related**: Issue #3

The streaming endpoint sends audio as base64-encoded raw PCM inside SSE events. This was chosen over chunked WAV (which requires a RIFF header with total data size, impossible for streaming) and raw binary HTTP chunks (no framing protocol, client must guess byte boundaries). SSE gives us text-based framing with `data:` prefix and `\n\n` delimiters, plus built-in reconnection semantics. Base64 adds ~33% overhead but keeps the protocol clean — for zero-overhead binary streaming, issue #4 adds a separate raw PCM endpoint. The `_last_used` update per chunk prevents the idle watchdog from unloading the model mid-stream.

---

## Entry 0011 — TF32: why it is safe for this model
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #5

TF32 (TensorFloat-32) is a numeric format available on Ampere and newer NVIDIA GPUs. It uses the same 8-bit exponent as float32 but truncates the mantissa from 23 bits to 10 bits, allowing matrix multiplications to run on Tensor Core hardware at roughly 3x the throughput.

The safety argument for enabling TF32 on this model is straightforward: the model already runs in bfloat16, which has only 7 bits of mantissa. TF32 intermediate operations have 10 bits of mantissa — strictly more precision than the model's own weight format. Enabling TF32 cannot lose information that bfloat16 already discards.

Two separate flags are needed: `torch.backends.cuda.matmul.allow_tf32` controls general matrix multiplication, and `torch.backends.cudnn.allow_tf32` controls cuDNN convolution operations. Both default to False in PyTorch. On pre-Ampere GPUs these flags are no-ops — the hardware simply ignores them.

The test strategy uses mock-based reimport rather than `if torch.cuda.is_available()` guards. This ensures tests actually assert on non-CUDA CI machines instead of silently passing. The pattern: reset flags to False, mock `cuda.is_available` to return True, reimport the server module, verify flags became True.

---

## Entry 0013 — Why lifespan over @app.on_event
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #33

FastAPI deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` in version 0.93.0. The replacement is the lifespan context manager pattern: `@asynccontextmanager async def lifespan(app)`. The yield point separates startup from shutdown — everything before yield runs at startup, everything after runs at shutdown.

The practical advantage is not just suppressing deprecation warnings. The old pattern had separate startup and shutdown functions with no shared scope. If startup allocated a resource (like a background task handle), the shutdown function needed that handle stored in a global or on the app object. With lifespan, variables from the startup section are naturally in scope during teardown:

```python
@asynccontextmanager
async def lifespan(app):
    watchdog_task = asyncio.create_task(_idle_watchdog())  # startup
    yield
    watchdog_task.cancel()  # shutdown — same scope, no global needed
```

For our server, the immediate benefit is that model unload now runs on graceful shutdown. Previously, if the container was stopped with SIGTERM, the model was not explicitly unloaded — the process just died and the GPU driver reclaimed VRAM. With the lifespan teardown, we run `_unload_model_sync()` which calls `gc.collect()` and `torch.cuda.empty_cache()` before exit. This ensures clean VRAM release in shared GPU environments where another container might be waiting for memory.

---

## Entry 0009 — Prometheus instrumentator vs manual metrics
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #30

We use two layers of metrics: automatic HTTP instrumentation via `prometheus-fastapi-instrumentator` and custom TTS-specific metrics via `prometheus-client` directly.

The instrumentator auto-adds request count, latency histograms, and response size metrics for every endpoint. This covers the "web server" dimension -- you can alert on 5xx rate, p99 latency, and throughput without writing any code.

The custom metrics cover the "TTS engine" dimension that the instrumentator cannot see: `tts_inference_duration_seconds` measures only the model inference time (excluding queue wait and audio encoding), `tts_requests_total` breaks down by voice and format, and `tts_model_loaded` tracks whether the model is in VRAM. These are the metrics you actually need for capacity planning -- if inference duration is climbing, the GPU is under pressure; if model_loaded flaps between 0 and 1, the idle timeout is too aggressive.

The implementation is gated behind `PROMETHEUS_ENABLED` and falls back gracefully if the packages are not installed. This keeps Prometheus as a soft dependency -- the server works without it.

---

## Entry 0014 — jemalloc: why the default allocator causes RSS bloat
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #21

Python's default memory allocator (glibc ptmalloc2) uses per-thread arenas to reduce lock contention. Each arena independently allocates and frees memory from the OS. The problem: when a thread frees a block, the arena may not return the underlying pages to the OS if adjacent blocks are still allocated. Over hours of operation with many small allocations (tokenizer strings, audio buffers, numpy intermediates), the RSS grows 2-3x beyond actual usage.

jemalloc uses a different strategy: size-class segregated regions with background thread compaction. The `dirty_decay_ms=1000` setting tells jemalloc to return freed pages to the OS within 1 second. `muzzy_decay_ms=0` tells it to immediately decommit pages rather than keeping them as "muzzy" (mapped but uncommitted). `background_thread:true` enables a dedicated thread that handles the decay without blocking application threads.

The LD_PRELOAD approach is the least invasive: the application code is unchanged, the allocator swap happens at process startup, and removing the env var reverts to ptmalloc2. No server.py changes needed.

---

## Entry 0015 — CPU affinity: sched_setaffinity over taskset
**Date**: 2026-02-20
**Type**: What could go wrong
**Related**: Issue #22

The original implementation used `os.system(f"taskset -p -c {cores} {pid}")` which has two problems. First, it is a command injection vector — the `INFERENCE_CPU_CORES` env var is interpolated directly into a shell command. Setting it to `0-7; rm -rf /` would execute the destructive command. Second, `taskset -p` changes the affinity for the entire process (all threads), including the uvicorn event loop, which defeats the purpose of pinning only the inference thread. The fix uses `os.sched_setaffinity(0, cores)` which: (a) takes a set of integers, eliminating shell injection, and (b) is a direct syscall wrapper with no shell involved. Note that `os.sched_setaffinity(0, ...)` still sets affinity for the calling process (PID 0 = current), not just the calling thread. True per-thread affinity would require `pthread_setaffinity_np` via ctypes, which is too fragile. The process-level approach is acceptable because the inference thread pool has only one thread and the event loop is lightweight.

---

## Entry 0016 — Transparent huge pages: madvise over always
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #23

THP `madvise` mode is chosen over `always` because `always` can cause latency spikes from compaction pauses and memory bloat in allocations that do not benefit from large pages (e.g., small Python objects). With `madvise`, only memory regions explicitly marked with `madvise(MADV_HUGEPAGE)` — or large anonymous mappings like PyTorch model weights — get backed by 2MB pages. The model weights (~2.4GB) consist of thousands of 4KB pages; mapping them as 2MB pages reduces TLB entries from ~600K to ~1200, significantly reducing TLB miss overhead during inference. The `defrag: defer+madvise` setting tells the kernel to defer compaction to a background thread rather than stalling the allocating process. All THP writes in the entrypoint are non-fatal (`|| true`) because the sysfs paths may be read-only in containers without `--privileged` or SYS_ADMIN capability.

---

## Entry 0017 — Opus codec: bitrate choice for speech
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #18

Opus at 32kbps is near-transparent quality for mono speech. The codec was designed specifically for speech and handles it well at low bitrates. We chose 32kbps over the original 64kbps spec because: (1) doubling the bitrate provides no perceptible quality improvement for single-speaker TTS audio, (2) lower bitrate halves bandwidth for streaming use cases, and (3) Opus at 32kbps still comfortably exceeds the quality of telephone-grade audio (8kbps G.711). The encoding path goes through a WAV intermediate via pydub (same as the existing MP3 path), which adds ~20ms overhead. A future optimization could pipe raw PCM directly to ffmpeg's Opus encoder, bypassing the WAV round-trip. The Dockerfile installs `libopus-dev` for build-time compilation support; multi-stage builds should use `libopus0` in the runtime stage to avoid shipping unnecessary header files.

---

## Entry 0018 — torchaudio: GPU-accelerated audio processing
**Date**: 2026-02-20
**Type**: What just happened
**Related**: Issue #19

The torchaudio integration replaces soundfile for WAV encoding and scipy for speed-adjustment resampling. The key insight is that torchaudio's `resample()` operates on CUDA tensors, keeping the audio data on GPU and avoiding a CPU round-trip. On Ampere+ GPUs, the polyphase resampling kernel runs significantly faster than scipy's CPU-bound FFT-based resample. The implementation moves the audio tensor to CUDA before resampling and back to CPU only for the final encoding step. On CPU-only hosts, torchaudio still runs on CPU (faster than scipy due to optimized C++ kernels) with a graceful fallback to scipy if torchaudio is not installed. Important caveat: `torchaudio.functional.resample()` changes sample rate, which changes both duration AND pitch — identical behavior to scipy. It is NOT pitch-preserving. Issue #14 (pyrubberband) is the pitch-preserving solution. The base PyTorch Docker image may already include torchaudio; the Dockerfile pip install is a safeguard but could cause version mismatches — verify inside the container.

---

## Entry 0019 — Async encode pipeline: infrastructure before overlap
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #20

The async encode pipeline moves audio format conversion (WAV/MP3/FLAC/OGG encoding) from the main event loop thread into a dedicated 2-thread CPU executor. This is infrastructure, not the full optimization. The full optimization is: while encoding sentence N on a CPU thread, start synthesizing sentence N+1 on the GPU. That requires the streaming endpoints from Phase 1 (#3/#4). Without streaming, there is only one inference per request, so encoding overlap has limited benefit for single-sentence requests. For multi-sentence streaming, the pipeline overlap could reduce total latency by 20-40ms per sentence (the encoding time that would otherwise gate the next synthesis). The `_split_sentences` helper is included here because it is the natural boundary for chunking. The `import re` was moved to the top of the file per Python convention — inline imports are confusing.

---

## Entry 0020 — WebSocket streaming: sentence chunking and PCM safety
**Date**: 2026-02-20
**Type**: What could go wrong
**Related**: Issue #24

Two bugs were caught in review. First, the sentence-splitting regex `(?<=[.!?])\s+` incorrectly splits on abbreviations like "Dr. Smith" or "U.S.A.". The fix uses abbreviation-aware lookbehinds: `(?<!\w\.\w.)(?<![A-Z][a-z]\.)` and adds CJK full-width punctuation (U+3002, U+FF01, U+FF1F) so Chinese/Japanese text with `。！？` gets chunked correctly. Second, float audio values outside [-1.0, 1.0] cause int16 wraparound distortion when multiplied by 32767. The model can occasionally produce values slightly above 1.0. Adding `np.clip(audio_data, -1.0, 1.0)` before conversion prevents this. The existing PCM streaming endpoint already had this clip; the WebSocket endpoint missed it.

---

## Entry 0021 — HTTP/2: TLS requirement and practical benefits
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #25

HTTP/2 requires TLS in practice. While the spec defines h2c (cleartext HTTP/2), browsers and most HTTP clients do not support it. The `h2` package provides the protocol implementation, but uvicorn only negotiates HTTP/2 via ALPN during the TLS handshake. Without TLS certificates, the server runs plain HTTP/1.1 — the `h2` package sits unused but harmless. The practical benefits of HTTP/2 for a TTS server are modest: header compression saves a few bytes per request, and multiplexing allows concurrent requests on one connection. For most TTS workloads (single request, wait for audio, play), HTTP/1.1 is sufficient. The value comes in multi-tenant deployments where many clients connect through a load balancer — fewer TCP connections and faster header processing. The docker-entrypoint.sh appends conditional TLS flags when SSL env vars are set, keeping the ENTRYPOINT pattern from #23 intact.

---

## Entry 0022 — Unix domain sockets: when to bypass TCP
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #26

UDS removes the TCP/IP stack overhead for same-host communication: no three-way handshake, no Nagle buffering, no checksumming. For a TTS server called by a local proxy or application, this saves 1-2ms per request (meaningful when the goal is sub-100ms time-to-first-byte for cached responses). The tradeoff: UDS and TCP are mutually exclusive — when `UNIX_SOCKET_PATH` is set, uvicorn binds only to the socket file, not to a TCP port. This means external clients cannot reach the service over the network. The intended deployment is behind a reverse proxy (nginx, caddy) that listens on TCP and forwards to the UDS. The `docker-entrypoint.sh` checks `UNIX_SOCKET_PATH` first and execs uvicorn with `--uds`, consolidating all startup logic in the entrypoint script.
