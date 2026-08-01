# Qwen3-TTS

OpenAI-compatible Text-to-Speech API server powered by [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base), running in Docker with NVIDIA GPU acceleration.

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/audio/speech`
- **9 built-in voices** — vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee
- **OpenAI voice aliases** — alloy, echo, fable, onyx, nova, shimmer
- **Voice cloning** — generate speech from a reference audio sample
- **Multi-language** — English, Chinese, Japanese, Korean, German, Italian, Portuguese, Spanish, French, Russian, Beijing dialect, Sichuan dialect
- **Standard output formats** — MP3 (default), WAV, raw PCM, FLAC, Opus
- **Speed control** — adjustable playback speed via resampling
- **Server-side resampling** — optional `sample_rate` parameter for telephony or custom sample rates
- **Audio output cache** — LRU cache skips GPU entirely on repeated requests (~1ms vs 500ms+)
- **Streaming** — SSE, raw PCM, and WebSocket endpoints with per-token or per-sentence modes

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with CUDA 12.4+ support

## Quick Start

```bash
# Build and start the service
docker compose up -d --build

# Check health
curl http://localhost:8101/readyz
```

The model (~2.4 GB) downloads automatically on first startup.

## API Endpoints

### `POST /v1/audio/speech`

Generate speech from text.

```bash
curl -X POST http://localhost:8101/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "speech-tts", "input": "Hello, world!", "voice": "vivian"}' \
  -o speech.mp3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `speech-tts` | Model or gateway alias; must be non-empty |
| `input` | string | *required* | Non-empty text, bounded by `MAX_INPUT_CHARS` |
| `voice` | string | `vivian` | Voice name or OpenAI alias |
| `response_format` | string | `mp3` | Output format: `mp3`, `wav`, `pcm`, `flac`, `opus` |
| `speed` | float | `1.0` | Playback speed multiplier, 0.25–4.0 |
| `language` | string | *auto-detect* | Language override |
| `instructions` / `instruct` | string | *unsupported* | Rejected because the Base model cannot honor instructions |
| `temperature` | float | *optional* | Sampling temperature for generation |
| `top_p` | float | *optional* | Nucleus sampling threshold |
| `sample_rate` | int | *optional* | Server-side resampling (e.g. `8000` for telephony) |

Errors use the OpenAI envelope: `{"error":{"message":"...","type":"invalid_request_error","param":"...","code":"..."}}`. Unsupported values are rejected before inference; the service never silently substitutes another audio format.

### `POST /v1/audio/speech/stream`

Stream speech synthesis via Server-Sent Events. Each chunk is streamed as base64-encoded raw PCM audio (signed 16-bit, 24 kHz, mono).

```bash
curl -N -X POST http://localhost:8101/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "First sentence. Second sentence.", "voice": "vivian"}'
```

Each SSE event contains base64-encoded PCM data. The stream ends with `data: [DONE]`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `speech-tts` | Model or gateway alias; must be non-empty |
| `input` | string | *required* | Non-empty text, bounded by `MAX_INPUT_CHARS` |
| `voice` | string | `vivian` | Voice name or OpenAI alias |
| `speed` | float | `1.0` | Playback speed multiplier |
| `language` | string | *auto-detect* | Language override |
| `instructions` / `instruct` | string | *unsupported* | Rejected because the Base model cannot honor instructions |
| `temperature` | float | *optional* | Sampling temperature for generation |
| `top_p` | float | *optional* | Nucleus sampling threshold |
| `sample_rate` | int | *optional* | Server-side resampling (e.g. `8000` for telephony) |

### `POST /v1/audio/speech/stream/pcm`

Stream speech as raw PCM audio. Text is split into chunks and streamed as raw int16 bytes. Use the response headers to configure your audio player.

```bash
curl -X POST http://localhost:8101/v1/audio/speech/stream/pcm \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world! This is streaming PCM audio.", "voice": "vivian"}' \
  -o speech.pcm
# Play with: ffplay -f s16le -ar 24000 -ac 1 speech.pcm
```

| Header | Value | Description |
|--------|-------|-------------|
| `X-PCM-Sample-Rate` | `24000` | Sample rate in Hz |
| `X-PCM-Bit-Depth` | `16` | Bits per sample (signed int16) |
| `X-PCM-Channels` | `1` | Mono audio |

Request body parameters are the same as `/v1/audio/speech/stream`.

### `WS /v1/audio/speech/ws`

WebSocket endpoint for real-time streaming. Send JSON messages, receive binary PCM frames.

```python
import asyncio
import websockets
import json

async def stream_tts():
    async with websockets.connect("ws://localhost:8101/v1/audio/speech/ws") as ws:
        await ws.send(json.dumps({"input": "Hello world. How are you?", "voice": "vivian"}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("event") == "done":
                    break
            else:
                # Binary PCM data (16-bit signed, 24kHz, mono)
                process_audio(msg)

asyncio.run(stream_tts())
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `speech-tts` | Model or gateway alias; must be non-empty |
| `input` | string | *required* | Non-empty text, bounded by `MAX_INPUT_CHARS` |
| `voice` | string | `vivian` | Voice name or OpenAI alias |
| `language` | string | *auto-detect* | Language override |
| `speed` | float | `1.0` | Playback speed multiplier |
| `temperature` | float | *optional* | Sampling temperature for generation |
| `top_p` | float | *optional* | Nucleus sampling threshold |
| `sample_rate` | int | *optional* | Server-side resampling (e.g. `8000` for telephony) |

### `POST /v1/audio/speech/clone`

Generate speech using a cloned voice from a reference audio file.

```bash
curl -X POST http://localhost:8101/v1/audio/speech/clone \
  -F "file=@reference.wav" \
  -F "input=Hello, this is my cloned voice." \
  -F "ref_text=Original text spoken in the reference audio." \
  -o cloned_speech.wav
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | *required* | Reference audio file |
| `input` | string | *required* | Text to synthesize |
| `ref_text` | string | *optional* | Transcript of the reference audio |
| `language` | string | *auto-detect* | Language override |
| `response_format` | string | `wav` | Output format |
| `temperature` | float | *optional* | Sampling temperature for generation |
| `top_p` | float | *optional* | Nucleus sampling threshold |

### Health endpoints

- `GET /livez` returns 200 while the API process/event loop is alive.
- `GET /readyz` returns 200 only when CUDA is available and the configured model is loaded; otherwise it returns 503.
- `GET /health` returns detailed internal model, CUDA, queue, voice, and cache status.

### `POST /cache/clear`

Clear the audio output and voice prompt caches. Returns the number of entries cleared from each.

```bash
curl -X POST http://localhost:8101/cache/clear
# {"audio_cleared": 42, "voice_cleared": 5}
```

### `GET /metrics`

Prometheus metrics endpoint (when `PROMETHEUS_ENABLED=true`). Exposes request counts, inference duration histograms, and model load state.

## Configuration

Environment variables (set in `.env` file, referenced by `compose.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Hugging Face model ID |
| `IDLE_TIMEOUT` | `120` | Seconds of inactivity before unloading model from GPU (0 = disabled) |
| `REQUEST_TIMEOUT` | `300` | Maximum seconds per inference request |
| `PRELOAD_MODEL` | `false` | Load model at startup instead of on first request |
| `QUANTIZE` | *(empty)* | Quantization mode: `fp8` (torchao), `int8` (bitsandbytes), or empty to disable |
| `TORCH_COMPILE` | `true` | Enable torch.compile optimization |
| `TORCH_COMPILE_MODE` | `max-autotune` | torch.compile mode: `max-autotune`, `reduce-overhead`, `default` |
| `CUDA_GRAPHS` | `true` | Enable CUDA graph capture via triton backend |
| `AUDIO_CACHE_MAX` | `256` | Max LRU cache entries for audio output (0 = disabled) |
| `VOICE_CACHE_MAX` | `32` | LRU cache capacity for voice clone speaker embeddings (0 = disabled) |
| `TEXT_NORMALIZE` | `true` | Expand numbers, currency, and abbreviations before synthesis |
| `MAX_QUEUE_DEPTH` | `5` | Max queued requests before 429 + `Retry-After` (0 = unlimited) |
| `MAX_INPUT_CHARS` | `4000` | Maximum input characters accepted before GPU work |
| `STREAM_TYPE` | `sentence` | Streaming mode: `sentence` (one chunk per sentence) or `token` (per-token, sub-400ms TTFA) |
| `STREAM_EMIT_FRAMES` | `4` | Token-mode: emit audio every N codec frames |
| `STREAM_FIRST_EMIT` | `3` | Token-mode: first-chunk emit interval (0 = disable two-phase) |
| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus metrics at `GET /metrics` |
| `LOG_FORMAT` | `json` | Log format: `json` for structured output, `text` for human-readable |
| `LOG_LEVEL` | `INFO` | Minimum log level: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL` |
| `GATEWAY_MODE` | `false` | Use gateway/worker subprocess architecture for lower idle memory |
| `INFERENCE_CPU_CORES` | *(empty)* | Pin to specific CPU cores (e.g., `0-7`). Empty = no pinning |
| `UNIX_SOCKET_PATH` | *(empty)* | Path to Unix socket (e.g., `/tmp/tts.sock`). Replaces TCP when set |
| `SSL_KEYFILE` | *(empty)* | Path to TLS private key (enables HTTP/2) |
| `SSL_CERTFILE` | *(empty)* | Path to TLS certificate (enables HTTP/2) |

The model cache is persisted to `./models` via volume mount.

### Unix Domain Socket (optional)

For same-host clients, UDS bypasses the TCP stack (~0.1-0.5ms savings per request):

```yaml
# compose.yaml
environment:
  - UNIX_SOCKET_PATH=/tmp/tts.sock
volumes:
  - /tmp:/tmp  # share socket with host
```

Connect from Python:
```python
import httpx
transport = httpx.AsyncHTTPTransport(uds="/tmp/tts.sock")
async with httpx.AsyncClient(transport=transport) as client:
    resp = await client.post("http://localhost/v1/audio/speech", json={...})
```

### Transparent Huge Pages (optional)

For optimal model loading performance, enable THP on the Docker host:

```bash
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

The container attempts to set these at startup but may need `--privileged` or appropriate capabilities. If the host already has THP enabled, no container privileges are needed.

### GPU Memory Management

The model loads **on-demand** with the first request and automatically **unloads after idle timeout** to free VRAM for other services. This is ideal for shared GPU environments.

### Always-On Mode

For dedicated GPU servers where you want the model to stay loaded permanently, set `IDLE_TIMEOUT=0`:

```yaml
environment:
  - IDLE_TIMEOUT=0  # Never unload — model stays in VRAM
```

The idle watchdog still runs but skips the unload check. The model loads on first request and remains loaded until the server shuts down.

## Testing

```bash
# Unit tests (no running server needed)
pytest server_test.py -v

# E2E tests (requires running server + GPU)
pytest E2Etest/ -v

# Integration test suite (requires running server)
python test_tts.py

# Quick smoke test
bash test_quick.sh
```

## License

Uses the [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) model — see its license for usage terms.
