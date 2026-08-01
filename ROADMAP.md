# Roadmap

Nine-phase plan to take Qwen3-TTS from a working prototype to a production-grade, real-time TTS server. Each phase targets a minor version and has a clear milestone vision.

---

## Phase 1 — Real-Time (v0.4.0)

**Vision**: First audio reaches the client in under 500 ms.

Before optimizing anything, we instrument. Then we remove the biggest bottleneck (fixed token budget), and finally deliver streaming so clients hear audio while synthesis continues.

- [x] #1 Add per-request latency breakdown logging
- [x] #2 Add adaptive `max_new_tokens` scaling with text length
- [x] #3 Add sentence-chunked SSE streaming endpoint
- [x] #4 Add raw PCM streaming endpoint

---

## Phase 2 — Speed & Quality (v0.5.0)

**Vision**: Every synthesis is measurably faster and the audio quality is noticeably better.

GPU tuning flags come first because they are low-risk and establish a faster baseline for benchmarking everything after. Inference optimizations build on that baseline. Audio quality improvements come last in the phase because they depend on a stable, fast inference path.

- [x] #5 Enable TF32 matmul mode
- [x] #6 Enable GPU persistence mode
- [x] #7 Lock GPU clocks to max boost
- [x] #8 Switch `attn_implementation` to `flash_attention_2`
- [x] #9 Enable `torch.compile` on model forward pass
- [x] #10 Deepen GPU warmup with multi-length synthesis calls
- [x] #11 Add VAD silence trimming (strip leading/trailing silence)
- [x] #12 Add text normalization for numbers, currency, abbreviations
- [x] #13 Replace Unicode language heuristic with fasttext detection
- [x] #14 Replace scipy speed adjustment with pitch-preserving pyrubberband
- [x] #15 Add voice prompt cache for `/clone` endpoint
- [x] #16 Pre-allocate GPU memory pool to reduce allocation jitter

---

## Phase 3 — Production Grade (v0.6.0)

**Vision**: The server is observable, resilient, and efficient under real production load.

Caching and codec work unlocks efficient streaming. System-level tuning reduces jitter. Protocol upgrades widen the deployment surface. Lifecycle and observability close the loop for production operations. Housekeeping at the end cleans up tech debt while the server is stable.

- [x] #17 Add full audio output LRU cache (text+voice to cached bytes)
- [x] #18 Add Opus codec support for streaming
- [x] #19 Add GPU-accelerated audio processing with torchaudio
- [x] #20 Add async audio encode pipeline (overlap encode N with synthesis N+1)
- [x] #21 Add jemalloc memory allocator via `LD_PRELOAD`
- [x] #22 Set CPU affinity for inference thread to GPU-adjacent cores
- [x] #23 Enable transparent huge pages for model weights
- [x] #24 Add WebSocket streaming endpoint
- [x] #25 Enable HTTP/2 support
- [x] #26 Add Unix domain socket support for same-host clients
- [x] #27 Add always-on mode (`IDLE_TIMEOUT=0` option, documented)
- [x] #28 Add eager model preload on startup (`PRELOAD_MODEL` env var)
- [x] #29 Add `ipc:host` to Docker compose for CUDA IPC
- [x] #30 Add Prometheus metrics endpoint
- [x] #31 Add structured JSON logging with per-request fields
- [x] #32 Add request queue depth limit with 503 early rejection
- [x] #33 Migrate `@app.on_event` to FastAPI lifespan context manager
- [x] #34 Pin all dependency versions in `requirements.txt`
- [x] #35 Convert to multi-stage Docker build
- [x] #36 Remove dead `VoiceCloneRequest` model

---

---

## Phase 4 — Intelligence (v0.7.0)

**Vision**: The server fully exploits what the Qwen3-TTS model can actually do — real-time clients are never starved, repeat voice-clone callers pay nothing, and generation parameters are in client hands.

Model-grounded: after research into the `qwen-tts` package API, three concrete gaps were identified between model capability and server implementation.

- [x] #81 Replace inference semaphore with priority queue (WS/SSE/PCM at priority 0, REST at priority 1)
- [x] #82 Fix voice clone caching — use `create_voice_clone_prompt()` instead of raw audio arrays
- [x] #83 Expose `temperature` and `top_p` in TTSRequest

---

## Phase 5 — Scale (v0.8.0)

**Vision**: The server handles concurrent load efficiently and runs lean in shared GPU environments.

- [x] #84 Add batch inference for concurrent synthesis requests (depends on #81)
- [x] #85 Add Gateway/Worker mode for minimal idle footprint (~30 MB idle vs ~1 GB)
- [x] #86 Add quantization support (INT8/FP8) via bitsandbytes and torchao

---

## Phase 6 — Performance (v0.10.1)

**Vision**: Squeeze maximum throughput from the GPU — lower latency, less VRAM, better streaming.

Issues are ad-hoc (no GitHub issue numbers) — performance work driven by profiling.

- [x] Switch torch.compile to max-autotune mode
- [x] Enable CUDA graphs via triton backend
- [x] Add CUDA inference and transfer streams
- [x] Add server-side sample rate conversion
- [x] Pipeline sentence synthesis in streaming endpoints
- [x] Add FP8 quantization via torchao
- [x] Fix torch.compile mode/options conflict
- [x] Release unused CUDA pool memory after model warmup

---

## Phase 7 — Observability (v0.9.0, v0.9.1)

**Vision**: Every meaningful event is logged with enough context to diagnose production issues without reproducing them.

Note: versions v0.9.0/v0.9.1 are chronologically before v0.10.x but were documented separately in the CHANGELOG. The logging work was foundational — migrating to loguru and adding comprehensive structured logging before the model switch.

- [x] Migrate logging from stdlib to loguru
- [x] Add comprehensive structured logging across all code paths

---

## Phase 8 — Foundation (v0.10.0, v0.10.2)

**Vision**: The server uses the right model variant and speaks a consistent error language.

- [x] Switch from CustomVoice to Base model with voice cloning support
- [x] #107 Standardize structured logging output (ISO 8601, ops level names, stdout)
- [x] #108 Add startup env validation and .env.example
- [x] #109 Standardize error response shape

---

## Phase 9 — Streaming (v0.10.3)

**Vision**: First audio byte reaches the client before the first sentence finishes synthesizing.

- [x] #110 Add per-token streaming via rekuenkdr/Qwen3-TTS-streaming fork

---

## Phase 10 — KBN Deployment Foundation (v0.10.6)

**Vision**: A strict, privacy-safe OpenAI-compatible TTS runtime can be operated behind a shared gateway on a single production GPU.

- [x] #116 Add request validation, OpenAI error envelopes, MP3 default and PCM output
- [x] #116 Add liveness/readiness probes and bounded queue/input behavior
- [x] #116 Pin the runtime dependency and streaming-fork build inputs
- [x] #116 Record target-GPU image and integration evidence before milestone closure

---

## Backlog

No unplaced items.

---

## Current Status

**v0.10.6** — Phase 10 complete. The KBN edge4 TTS vertical slice is deployed and proved through LiteLLM.
