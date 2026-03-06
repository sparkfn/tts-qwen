from contextlib import asynccontextmanager, nullcontext
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, field_validator
from typing import Optional
import torch
import soundfile as sf
import hashlib
import io
import json
import os
import gc
import asyncio
import time
import re
import uuid
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import scipy.signal as scipy_signal
import logging
import sys
import base64
import heapq
from dataclasses import dataclass, field

from loguru import logger

try:
    import pyrubberband as _pyrubberband
except ImportError:
    _pyrubberband = None

# Prometheus metrics (optional — enabled by default)
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() in ("true", "1")

if PROMETHEUS_ENABLED:
    try:
        from prometheus_client import Counter, Histogram, Gauge
        from prometheus_fastapi_instrumentator import Instrumentator

        tts_requests_total = Counter(
            "tts_requests_total", "Total TTS requests", ["voice", "format"]
        )
        tts_inference_duration = Histogram(
            "tts_inference_duration_seconds", "Inference duration in seconds"
        )
        tts_model_loaded = Gauge(
            "tts_model_loaded", "Whether model is currently loaded (1=yes, 0=no)"
        )
        _prometheus_available = True
    except ImportError:
        _prometheus_available = False
else:
    _prometheus_available = False

# Structured logging via loguru
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


_LEVEL_REMAP = {
    "CRITICAL": "fatal",
    "ERROR": "error",
    "WARNING": "warn",
    "INFO": "info",
    "DEBUG": "debug",
    "TRACE": "trace",
    "SUCCESS": "info",
}


def _json_sink(message):
    """Flat JSON sink matching previous JsonFormatter schema."""
    record = message.record
    log_obj = {
        "timestamp": record["time"].isoformat(),
        "level": _LEVEL_REMAP.get(record["level"].name, record["level"].name.lower()),
        "message": record["message"],
        "logger": record["name"] or "qwen3-tts",
        "service": "qwen3-tts",
    }
    log_obj.update(record["extra"])
    print(json.dumps(log_obj, default=str), file=sys.stdout, flush=True)


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn/fastapi) → loguru."""
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _configure_logging():
    logger.remove()  # remove default stderr handler
    if LOG_FORMAT == "json":
        logger.add(_json_sink, level=LOG_LEVEL, format="{message}")
    else:
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format="<green>{time:YYYY-MM-DDTHH:mm:ss}</green> <level>{level:<8}</level> <cyan>qwen3-tts</cyan> {message}",
        )
    # Intercept uvicorn / fastapi stdlib logs
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # Silence noisy upstream loggers (INFO spam from HF config diffs)
    for noisy in (
        "qwen_tts.core.models.configuration_qwen3_tts",
        "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
        "torchao",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()

def _validate_env():
    """Validate environment variables at startup -- fail fast on bad config."""
    errors = []

    quantize = os.getenv("QUANTIZE", "").lower()
    if quantize not in ("", "fp8", "int8"):
        errors.append("QUANTIZE=%r. Valid: '', 'fp8', 'int8'" % quantize)

    log_fmt = os.getenv("LOG_FORMAT", "json")
    if log_fmt not in ("json", "text"):
        errors.append("LOG_FORMAT=%r. Valid: 'json', 'text'" % log_fmt)

    log_lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE", "SUCCESS")
    if log_lvl not in valid_levels:
        errors.append("LOG_LEVEL=%r. Valid: %s" % (log_lvl, ", ".join(valid_levels)))

    numeric_vars = {
        "IDLE_TIMEOUT": "120",
        "REQUEST_TIMEOUT": "300",
        "MAX_QUEUE_DEPTH": "5",
        "AUDIO_CACHE_MAX": "256",
        "VOICE_CACHE_MAX": "32",
        "MAX_BATCH_SIZE": "4",
    }
    for var, default in numeric_vars.items():
        raw = os.getenv(var, default)
        try:
            val = int(raw)
            if val < 0:
                errors.append("%s=%r. Must be a non-negative integer" % (var, raw))
        except ValueError:
            errors.append("%s=%r. Must be a non-negative integer" % (var, raw))

    stream_type = os.getenv("STREAM_TYPE", "token").lower()
    if stream_type not in ("sentence", "token"):
        errors.append("STREAM_TYPE=%r. Valid: 'sentence', 'token'" % stream_type)

    stream_emit = os.getenv("STREAM_EMIT_FRAMES", "4")
    try:
        val = int(stream_emit)
        if val < 1:
            errors.append("STREAM_EMIT_FRAMES=%r. Must be a positive integer" % stream_emit)
    except ValueError:
        errors.append("STREAM_EMIT_FRAMES=%r. Must be a positive integer" % stream_emit)

    stream_first = os.getenv("STREAM_FIRST_EMIT", "3")
    try:
        val = int(stream_first)
        if val < 0:
            errors.append("STREAM_FIRST_EMIT=%r. Must be a non-negative integer" % stream_first)
    except ValueError:
        errors.append("STREAM_FIRST_EMIT=%r. Must be a non-negative integer" % stream_first)

    if errors:
        for err in errors:
            logger.critical("Invalid %s" % err)
        sys.exit(1)


_validate_env()

try:
    from pydub import AudioSegment as _PydubAudioSegment
except ImportError:
    _PydubAudioSegment = None

try:
    import torchaudio
    import torchaudio.functional as torchaudio_F
    _TORCHAUDIO = True
except ImportError:
    _TORCHAUDIO = False

# Enable cudnn autotuner — finds fastest convolution algorithms for the GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # 3x faster matmul on Ampere+ GPUs
    torch.backends.cudnn.allow_tf32 = True           # enable TF32 for cuDNN ops
    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    logger.bind(gpu=_gpu_name, vram_total_mb=_gpu_total_mb).info("CUDA device detected")
else:
    logger.warning("CUDA not available — running on CPU, performance will be poor")

# CUDA streams — overlap inference compute with data transfer
_inference_stream: torch.cuda.Stream | None = None
_transfer_stream: torch.cuda.Stream | None = None

# Eager model preload on startup (default: false)
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").lower() in ("true", "1")

@asynccontextmanager
async def lifespan(app):
    # Startup
    _infer_queue.start()
    if _prometheus_available:
        Instrumentator().instrument(app).expose(app)
    _set_cpu_affinity()
    asyncio.create_task(_idle_watchdog())
    asyncio.create_task(_inference_watchdog())
    if PRELOAD_MODEL:
        logger.info("Loading model at startup (PRELOAD_MODEL=true)")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_infer_executor, _load_model_sync)
    logger.info("Server started")
    yield
    # Shutdown
    logger.info("Server shutting down")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_infer_executor, _unload_model_sync)

class ErrorResponse(BaseModel):
    """Standard error response shape."""
    code: str
    message: str
    context: dict | None = None
    statusCode: int


class APIError(Exception):
    """Structured API error that produces a standard JSON response."""
    def __init__(self, status_code: int, code: str, message: str,
                 context: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.context = context
        self.headers = headers


app = FastAPI(title="Qwen3-TTS API", lifespan=lifespan)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.code, message=exc.message,
            context=exc.context, statusCode=exc.status_code,
        ).model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code="INTERNAL_ERROR", message=str(exc.detail),
            statusCode=exc.status_code,
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


model = None
loaded_model_id = None

# Single-thread executor for GPU inference — avoids default pool overhead
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")

# CPU executor for audio encoding — runs in parallel with GPU inference
_encode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-encode")

# Priority constants for inference queue
PRIORITY_REALTIME = 0   # streaming clients — WS, SSE, PCM
PRIORITY_BATCH    = 1   # buffered clients — REST /speech, /clone


@dataclass(order=True)
class _InferJob:
    priority: int
    submit_time: float
    future: "asyncio.Future" = field(compare=False)
    fn: "callable" = field(compare=False)
    batch_key: str = field(default="single", compare=False)
    batch_args: dict = field(default_factory=dict, compare=False)


class PriorityInferQueue:
    """Min-heap inference queue. Lower priority number = runs sooner."""

    def __init__(self):
        self._heap: list = []
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._infer_executor = _infer_executor

    def start(self):
        self._task = asyncio.create_task(self._worker())

    async def _worker(self):
        global _infer_job_started_at
        while True:
            await self._event.wait()
            while True:
                async with self._lock:
                    if not self._heap:
                        self._event.clear()
                        break

                    top = self._heap[0]
                    if top.batch_key == "synthesis":
                        batch_jobs = []
                        while self._heap and self._heap[0].batch_key == "synthesis" and len(batch_jobs) < MAX_BATCH_SIZE:
                            batch_jobs.append(heapq.heappop(self._heap))
                    else:
                        batch_jobs = None
                        single_job = heapq.heappop(self._heap)

                loop = asyncio.get_running_loop()

                if batch_jobs:
                    texts = [j.batch_args["text"] for j in batch_jobs]
                    langs = [j.batch_args["language"] for j in batch_jobs]
                    voice_files = [j.batch_args["speaker"] for j in batch_jobs]
                    kwargs_list = [j.batch_args["gen_kwargs"] for j in batch_jobs]
                    logger.debug("Queue dispatching batch", batch_size=len(batch_jobs),
                                 remaining=len(self._heap))
                    try:
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
                else:
                    logger.debug("Queue dispatching single job", priority=single_job.priority,
                                 remaining=len(self._heap))
                    try:
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

    async def submit(self, fn: callable, priority: int = PRIORITY_BATCH) -> any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = _InferJob(priority=priority, submit_time=time.monotonic(), future=future, fn=fn)
        async with self._lock:
            heapq.heappush(self._heap, job)
        self._event.set()
        return await future

    async def submit_batch(self, text: str, language: str, speaker: str, gen_kwargs: dict) -> tuple:
        """Submit a batchable synthesis job. The worker collects concurrent peers."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = _InferJob(
            priority=PRIORITY_BATCH,
            submit_time=time.monotonic(),
            future=future,
            fn=None,
            batch_key="synthesis",
            batch_args={"text": text, "language": language, "speaker": speaker, "gen_kwargs": gen_kwargs},
        )
        async with self._lock:
            heapq.heappush(self._heap, job)
        self._event.set()
        return await future


_infer_queue = PriorityInferQueue()

# Timestamp when the current inference job started (None = idle)
_infer_job_started_at: float | None = None

# Lock to prevent concurrent load/unload
_model_lock = asyncio.Lock()

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

# Idle unload timeout in seconds (0 = disabled)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))

# Text normalization (expand numbers, currency, abbreviations)
TEXT_NORMALIZE = os.getenv("TEXT_NORMALIZE", "true").lower() in ("true", "1", "yes")

# Queue depth limit — 503 when exceeded (0 = unlimited)
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "5"))


class _QueueDepthTracker:
    """Asyncio-safe queue depth counter with context manager for clean inc/dec."""

    def __init__(self) -> None:
        self._depth = 0
        self._lock = asyncio.Lock()

    @property
    def depth(self) -> int:
        return self._depth

    async def acquire(self, request_id: str, endpoint: str) -> None:
        """Increment depth; raise APIError 503 if queue is full."""
        async with self._lock:
            if MAX_QUEUE_DEPTH > 0 and self._depth >= MAX_QUEUE_DEPTH:
                logger.bind(request_id=request_id, endpoint=endpoint,
                             queue_depth=self._depth).warning("Request rejected: queue full")
                raise APIError(503, "QUEUE_FULL",
                               f"Server busy: {self._depth} requests queued. Try again later.",
                               headers={"Retry-After": "5"})
            self._depth += 1

    async def release(self) -> None:
        async with self._lock:
            self._depth -= 1


_queue_tracker = _QueueDepthTracker()

# Quantization mode — "", "int8" (bitsandbytes), or "fp8" (torchao)
QUANTIZE = os.getenv("QUANTIZE", "").lower()

# torch.compile mode — "max-autotune" (slower warmup, faster steady-state),
# "reduce-overhead", or "default"
TORCH_COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", "max-autotune")

# CUDA graphs via triton backend — enables automatic graph capture/replay
CUDA_GRAPHS = os.getenv("CUDA_GRAPHS", "true").lower() in ("true", "1")

# Per-token streaming mode — "sentence" (default) or "token" (sub-400ms TTFA)
STREAM_TYPE = os.getenv("STREAM_TYPE", "token").lower()

# Token-mode tuning: frames between emissions, first-chunk emit interval
STREAM_EMIT_FRAMES = int(os.getenv("STREAM_EMIT_FRAMES", "4"))
STREAM_FIRST_EMIT = int(os.getenv("STREAM_FIRST_EMIT", "3"))

# Whether the qwen-tts fork with stream_generate_voice_clone() is available.
# Set during model load after probing the model instance.
_HAS_STREAMING = False

# Batch inference — max jobs per GPU dispatch (1 = disabled)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))

# Track last request time (initialised to now so the idle watchdog never
# computes a bogus idle_secs during the window before the first request)
_last_used = time.time()

# Audio output LRU cache — skips GPU entirely on cache hit
_AUDIO_CACHE_MAX = int(os.getenv("AUDIO_CACHE_MAX", "256"))
_audio_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()


def _audio_cache_key(text: str, voice: str, speed: float, fmt: str, language: str = "", instruct: str = "") -> str:
    raw = f"{text}|{voice}|{speed}|{fmt}|{language}|{instruct}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_audio_cache(key: str) -> tuple[bytes, str] | None:
    if _AUDIO_CACHE_MAX <= 0:
        return None
    if key in _audio_cache:
        _audio_cache.move_to_end(key)
        logger.debug("Audio cache hit", cache_key=key[:12], cache_size=len(_audio_cache))
        return _audio_cache[key]
    logger.debug("Audio cache miss", cache_key=key[:12], cache_size=len(_audio_cache))
    return None


def _set_audio_cache(key: str, data: bytes, content_type: str) -> None:
    if _AUDIO_CACHE_MAX <= 0:
        return
    evicted = len(_audio_cache) >= _AUDIO_CACHE_MAX
    if evicted:
        _audio_cache.popitem(last=False)
    _audio_cache[key] = (data, content_type)
    if evicted:
        logger.debug("Audio cache store (evicted LRU)", cache_key=key[:12], cache_size=len(_audio_cache))
    else:
        logger.debug("Audio cache store", cache_key=key[:12], cache_size=len(_audio_cache))

# Voice prompt cache — caches speaker embeddings by reference audio content hash
# Uses model.create_voice_clone_prompt() to precompute the embedding once,
# so repeat clone requests skip the encoder pass entirely.
VOICE_CACHE_MAX = int(os.getenv("VOICE_CACHE_MAX", "32"))
_voice_prompt_cache: OrderedDict = OrderedDict()
_voice_cache_hits = 0

logger.bind(
    PRELOAD_MODEL=PRELOAD_MODEL,
    IDLE_TIMEOUT=IDLE_TIMEOUT,
    REQUEST_TIMEOUT=REQUEST_TIMEOUT,
    TEXT_NORMALIZE=TEXT_NORMALIZE,
    MAX_QUEUE_DEPTH=MAX_QUEUE_DEPTH,
    QUANTIZE=QUANTIZE or "none",
    MAX_BATCH_SIZE=MAX_BATCH_SIZE,
    AUDIO_CACHE_MAX=_AUDIO_CACHE_MAX,
    VOICE_CACHE_MAX=VOICE_CACHE_MAX,
    TORCH_COMPILE_MODE=TORCH_COMPILE_MODE,
    CUDA_GRAPHS=CUDA_GRAPHS,
    STREAM_TYPE=STREAM_TYPE,
    STREAM_EMIT_FRAMES=STREAM_EMIT_FRAMES,
    STREAM_FIRST_EMIT=STREAM_FIRST_EMIT,
).info("Server configuration loaded")

# Reference audio directory — each voice is backed by a pre-generated WAV file
_VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")

# Map voice names to reference audio filenames (Base model uses voice cloning)
VOICE_MAP = {
    # Direct Qwen speaker names → reference WAV files
    "vivian": "vivian.wav",
    "serena": "serena.wav",
    "uncle_fu": "uncle_fu.wav",
    "dylan": "dylan.wav",
    "eric": "eric.wav",
    "ryan": "ryan.wav",
    "aiden": "aiden.wav",
    "ono_anna": "ono_anna.wav",
    "sohee": "sohee.wav",
    # OpenAI-compatible aliases
    "alloy": "ryan.wav",
    "echo": "aiden.wav",
    "fable": "dylan.wav",
    "onyx": "uncle_fu.wav",
    "nova": "vivian.wav",
    "shimmer": "serena.wav",
}

DEFAULT_VOICE = "vivian.wav"

# Pre-computed voice clone prompts — populated during model load
_voice_prompts: dict = {}  # voice_file → list[VoiceClonePromptItem]


class TTSRequest(BaseModel):
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0
    language: Optional[str] = None
    instruct: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    sample_rate: Optional[int] = None
    stream_mode: Optional[str] = None

    @field_validator("stream_mode")
    @classmethod
    def _validate_stream_mode(cls, v):
        if v is not None and v not in ("sentence", "token"):
            raise ValueError(f"stream_mode must be 'sentence' or 'token', got '{v}'")
        return v


def _effective_stream_mode(request: TTSRequest) -> str:
    return request.stream_mode or STREAM_TYPE


def _release_gpu_full():
    """Full GPU memory release — only used during model unload."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resolve_quant_kwargs() -> tuple[torch.dtype, dict]:
    """Return (dtype, extra_load_kwargs) based on QUANTIZE env var.

    int8: bitsandbytes LLM.int8() — ~50% VRAM reduction, ~10-20% slower
    fp8:  torchao FP8 dynamic activation — ~67% VRAM reduction, minimal speed impact on Hopper+
    """
    if not QUANTIZE:
        return torch.bfloat16, {}

    if QUANTIZE == "int8":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for QUANTIZE=int8. "
                "Install it: pip install bitsandbytes>=0.43.0"
            )
        return torch.float16, {"load_in_8bit": True}

    if QUANTIZE == "fp8":
        # FP8 is applied post-load via torchao — just validate the import here
        try:
            import torchao  # noqa: F401
        except ImportError:
            raise ImportError(
                "torchao is required for QUANTIZE=fp8. "
                "Install it: pip install torchao>=0.5.0"
            )
        return torch.bfloat16, {}

    raise ValueError(
        f"Unknown QUANTIZE value: {QUANTIZE!r}. Must be 'int8', 'fp8', or unset."
    )


def _apply_fp8_quantization() -> None:
    """Apply post-load FP8 weight quantization via torchao."""
    try:
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
        quantize_(model.model, Float8WeightOnlyConfig())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        logger.bind(gpu_allocated_mb=round(alloc_mb)).success(
            "FP8 weight quantization applied via torchao"
        )
    except Exception as e:
        logger.warning("FP8 quantization failed, continuing without: {}", e)


def _apply_torch_compile() -> None:
    """Compile model for faster inference (PyTorch 2.0+)."""
    if os.getenv("TORCH_COMPILE", "true").lower() != "true":
        return
    try:
        compile_mode = TORCH_COMPILE_MODE
        # PyTorch doesn't allow both mode and options simultaneously.
        # max-autotune already enables CUDA graphs internally, so only
        # pass explicit options when using a mode that doesn't (e.g. "default").
        if CUDA_GRAPHS and torch.cuda.is_available() and compile_mode not in ("max-autotune", "max-autotune-no-cudagraphs"):
            compile_options: dict | None = {"triton.cudagraphs": True}
        else:
            compile_options = None
        model.model = torch.compile(
            model.model, mode=compile_mode, fullgraph=False, options=compile_options,
        )
        logger.bind(compile_mode=compile_mode, cuda_graphs=CUDA_GRAPHS).success(
            "torch.compile enabled"
        )
    except Exception as e:
        logger.warning("torch.compile not available or failed: {}", e)


def _precompute_voice_prompts() -> None:
    """Pre-compute voice clone prompts from reference audio files."""
    logger.info("Pre-computing voice prompts from reference audio")
    for voice_file in sorted(set(VOICE_MAP.values())):
        path = os.path.join(_VOICES_DIR, voice_file)
        if os.path.exists(path):
            t_v = time.perf_counter()
            prompts = model.create_voice_clone_prompt(ref_audio=path, x_vector_only_mode=True)
            _voice_prompts[voice_file] = prompts
            logger.bind(voice=voice_file, ms=round((time.perf_counter() - t_v) * 1000)).debug("Voice prompt computed")
        else:
            logger.bind(voice=voice_file, path=path).warning("Reference audio not found, skipping")
    logger.bind(voices=len(_voice_prompts)).info("Voice prompts ready")


def _gpu_warmup() -> None:
    """Multi-length warmup to pre-cache CUDA kernels for different input sizes."""
    if not torch.cuda.is_available() or not _voice_prompts:
        return

    logger.info("Warming up GPU with multi-length synthesis")
    warmup_prompt = next(iter(_voice_prompts.values()))
    warmup_texts = [
        "Hello.",                                       # ~5 tokens — short prompt path
        "Hello, how are you doing today?",              # ~20 tokens — medium
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",  # ~50 tokens — longer
    ]
    t_warmup = time.perf_counter()
    for text in warmup_texts:
        try:
            with torch.inference_mode():
                model.generate_voice_clone(
                    text=text,
                    language="English",
                    voice_clone_prompt=warmup_prompt,
                    max_new_tokens=256,
                )
            logger.bind(chars=len(text)).debug("Warmup synthesis complete")
        except Exception as e:
            logger.bind(text_preview=text[:20]).warning("Warmup synthesis failed: {}", e)
    logger.bind(warmup_ms=round((time.perf_counter() - t_warmup) * 1000)).info("GPU warmup complete")
    # Clear warmup allocations so steady-state VRAM is clean
    _release_gpu_full()

    # Pre-warm CUDA memory pool — allocate and free a large tensor so the
    # allocator pre-reserves a contiguous block, reducing first-request jitter
    logger.info("Pre-warming CUDA memory pool")
    try:
        dummy = torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
        del dummy
        torch.cuda.empty_cache()
        logger.debug("CUDA pool pre-warmed (128 MB dummy tensor)")
    except Exception as e:
        logger.warning("CUDA pool pre-warm failed: {}", e)


def _load_model_sync() -> None:
    """Load model into GPU (blocking). Called from async context via lock."""
    global model, loaded_model_id, _last_used, _HAS_STREAMING, STREAM_TYPE
    from qwen_tts import Qwen3TTSModel

    if model is not None:
        return

    t_load_start = time.time()
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    loaded_model_id = model_id

    # Prefer flash_attention_2 on Ampere+ GPUs; fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        logger.warning("flash_attention_2 not available, falling back to sdpa")

    dtype, quant_kwargs = _resolve_quant_kwargs()
    if QUANTIZE:
        logger.bind(quantize=QUANTIZE).info("Loading model with quantization")

    logger.bind(model_id=model_id, attn=attn_impl, dtype=str(dtype)).info("Loading model")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        **quant_kwargs,
    )

    if QUANTIZE == "fp8":
        _apply_fp8_quantization()

    # Create dedicated CUDA streams for overlapping compute + transfer
    global _inference_stream, _transfer_stream
    if torch.cuda.is_available():
        _inference_stream = torch.cuda.Stream()
        _transfer_stream = torch.cuda.Stream()
        logger.info("CUDA inference and transfer streams created")

    # Compile model for faster inference (PyTorch 2.0+)
    if os.getenv("TORCH_COMPILE", "true").lower() == "true":
        try:
            compile_mode = TORCH_COMPILE_MODE
            # PyTorch doesn't allow both mode and options simultaneously.
            # max-autotune already enables CUDA graphs internally, so only
            # pass explicit options when using a mode that doesn't (e.g. "default").
            if CUDA_GRAPHS and torch.cuda.is_available() and compile_mode not in ("max-autotune", "max-autotune-no-cudagraphs"):
                compile_options: dict | None = {"triton.cudagraphs": True}
            else:
                compile_options = None
            model.model = torch.compile(
                model.model, mode=compile_mode, fullgraph=False, options=compile_options,
            )
            logger.bind(compile_mode=compile_mode, cuda_graphs=CUDA_GRAPHS).success(
                "torch.compile enabled"
            )
        except Exception as e:
            logger.warning("torch.compile not available or failed: {}", e)

    # Enable per-token streaming optimizations if fork is installed
    if STREAM_TYPE == "token" and hasattr(model, "enable_streaming_optimizations"):
        try:
            model.enable_streaming_optimizations(
                decode_window_frames=80,
                use_compile=(os.getenv("TORCH_COMPILE", "true").lower() == "true"),
                compile_mode=TORCH_COMPILE_MODE if TORCH_COMPILE_MODE != "false" else "reduce-overhead",
                use_cuda_graphs=CUDA_GRAPHS,
            )
            logger.info("streaming_optimizations_enabled")
        except Exception as e:
            logger.warning("Failed to enable streaming optimizations: {}", e)

    # Detect whether the streaming fork is installed
    _HAS_STREAMING = hasattr(model, "stream_generate_voice_clone")
    logger.bind(has_streaming=_HAS_STREAMING, stream_type=STREAM_TYPE).info(
        "Streaming fork detection complete"
    )

    # Pre-compute voice clone prompts from reference audio files
    logger.info("Pre-computing voice prompts from reference audio")
    for voice_file in sorted(set(VOICE_MAP.values())):
        path = os.path.join(_VOICES_DIR, voice_file)
        if os.path.exists(path):
            t_v = time.perf_counter()
            prompts = model.create_voice_clone_prompt(ref_audio=path, x_vector_only_mode=True)
            _voice_prompts[voice_file] = prompts
            logger.bind(voice=voice_file, ms=round((time.perf_counter() - t_v) * 1000)).debug("Voice prompt computed")
        else:
            logger.bind(voice=voice_file, path=path).warning("Reference audio not found, skipping")
    logger.bind(voices=len(_voice_prompts)).info("Voice prompts ready")

    # Multi-length warmup to pre-cache CUDA kernels for different input sizes
    if torch.cuda.is_available() and _voice_prompts:
        logger.info("Warming up GPU with multi-length synthesis")
        warmup_prompt = next(iter(_voice_prompts.values()))
        warmup_texts = [
            "Hello.",                                       # ~5 tokens — short prompt path
            "Hello, how are you doing today?",              # ~20 tokens — medium
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",  # ~50 tokens — longer
        ]
        t_warmup = time.perf_counter()
        for text in warmup_texts:
            try:
                with torch.inference_mode():
                    model.generate_voice_clone(
                        text=text,
                        language="English",
                        voice_clone_prompt=warmup_prompt,
                        max_new_tokens=256,
                    )
                logger.bind(chars=len(text)).debug("Warmup synthesis complete")
            except Exception as e:
                logger.bind(text_preview=text[:20]).warning("Warmup synthesis failed: {}", e)
        logger.bind(warmup_ms=round((time.perf_counter() - t_warmup) * 1000)).info("GPU warmup complete")
        # Clear warmup allocations so steady-state VRAM is clean
        _release_gpu_full()

        # Pre-warm CUDA memory pool — allocate and free a large tensor so the
        # allocator pre-reserves a contiguous block, reducing first-request jitter
        logger.info("Pre-warming CUDA memory pool")
        try:
            dummy = torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
            del dummy
            torch.cuda.empty_cache()
            logger.debug("CUDA pool pre-warmed (128 MB dummy tensor)")
        except Exception as e:
            logger.warning("CUDA pool pre-warm failed: {}", e)

    # Release all unused cached memory back to the driver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _last_used = time.time()
    if _prometheus_available:
        tts_model_loaded.set(1)
    logger.bind(model_id=model_id, load_ms=round((time.time() - t_load_start) * 1000)).success("Model loaded")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.bind(gpu_allocated_mb=round(allocated), gpu_reserved_mb=round(reserved)).info("GPU memory after load")

    # Graceful fallback: if token streaming requested but fork not installed
    if STREAM_TYPE == "token" and not hasattr(model, "stream_generate_voice_clone"):
        logger.warning("STREAM_TYPE=token but streaming fork not installed, falling back to sentence")
        STREAM_TYPE = "sentence"


def _unload_model_sync():
    """Unload model from GPU to free VRAM."""
    global model, _inference_stream, _transfer_stream, _HAS_STREAMING

    if model is None:
        return

    logger.info("Unloading model (idle timeout)")
    _voice_prompts.clear()
    _HAS_STREAMING = False
    _inference_stream = None
    _transfer_stream = None
    del model
    model = None
    if _prometheus_available:
        tts_model_loaded.set(0)
    _release_gpu_full()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.bind(gpu_allocated_mb=round(allocated), gpu_reserved_mb=round(reserved)).info("Model unloaded")


async def _ensure_model_loaded() -> None:
    """Load model if not already loaded. Thread-safe via lock."""
    global _last_used
    if model is not None:
        _last_used = time.time()
        return
    async with _model_lock:
        if model is not None:
            _last_used = time.time()
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_infer_executor, _load_model_sync)
        _last_used = time.time()


async def _idle_watchdog() -> None:
    """Background task that unloads model after IDLE_TIMEOUT seconds of inactivity."""
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0 or model is None:
            continue
        idle_secs = time.time() - _last_used
        if idle_secs > IDLE_TIMEOUT:
            logger.bind(idle_s=round(idle_secs), timeout_s=IDLE_TIMEOUT).info("Idle timeout reached, unloading model")
            async with _model_lock:
                if model is not None and time.time() - _last_used > IDLE_TIMEOUT:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(_infer_executor, _unload_model_sync)


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


def _parse_cpu_cores(spec: str) -> set[int]:
    """Parse CPU core spec like '0-3,6,8-11' into a set of ints."""
    cores = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.update(range(int(lo), int(hi) + 1))
        else:
            cores.add(int(part))
    return cores


def _set_cpu_affinity():
    """Pin process to GPU-adjacent CPU cores for better cache locality.

    Uses os.sched_setaffinity() instead of taskset to avoid command injection
    and to correctly set affinity for the calling process.
    """
    affinity_cores = os.getenv("INFERENCE_CPU_CORES", "")
    if not affinity_cores:
        return
    try:
        cores = _parse_cpu_cores(affinity_cores)
        os.sched_setaffinity(0, cores)
        logger.bind(cores=sorted(cores)).info("CPU affinity set")
    except Exception as e:
        logger.warning("Could not set CPU affinity: {}", e)


@app.get("/health")
async def health() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        }
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": torch.cuda.is_available(),
        "queue_depth": _queue_tracker.depth,
        "max_queue_depth": MAX_QUEUE_DEPTH,
        "voices": list(VOICE_MAP.keys()),
        "audio_cache_size": len(_audio_cache),
        "audio_cache_max": _AUDIO_CACHE_MAX,
        "voice_cache_size": len(_voice_prompt_cache),
        "voice_cache_max": VOICE_CACHE_MAX,
        "voice_cache_hits": _voice_cache_hits,
        **gpu_info,
    }


@app.get("/v1/info")
async def info() -> dict:
    """Lightweight capability endpoint for downstream service discovery."""
    return {
        "stream_type": STREAM_TYPE,
        "voices": list(VOICE_MAP.keys()),
        "model_id": loaded_model_id,
    }


@app.post("/cache/clear")
async def clear_cache() -> dict:
    """Clear the audio output cache and voice prompt cache."""
    audio_count = len(_audio_cache)
    voice_count = len(_voice_prompt_cache)
    _audio_cache.clear()
    _voice_prompt_cache.clear()
    logger.bind(audio_cleared=audio_count, voice_cleared=voice_count).info("Cache cleared")
    return {"audio_cleared": audio_count, "voice_cleared": voice_count}


def convert_audio_format(audio_data: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    """Convert audio data to the requested format."""
    buffer = io.BytesIO()

    if output_format in ("wav", "wave"):
        if _TORCHAUDIO:
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
        content_type = "audio/wav"
    elif output_format == "mp3":
        if _PydubAudioSegment is not None:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
            wav_buffer.seek(0)
            audio_segment = _PydubAudioSegment.from_wav(wav_buffer)
            audio_segment.export(buffer, format="mp3")
            content_type = "audio/mpeg"
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            content_type = "audio/wav"
    elif output_format == "flac":
        sf.write(buffer, audio_data, sample_rate, format="FLAC")
        content_type = "audio/flac"
    elif output_format == "ogg":
        sf.write(buffer, audio_data, sample_rate, format="OGG")
        content_type = "audio/ogg"
    elif output_format == "opus":
        if _PydubAudioSegment is not None:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
            wav_buffer.seek(0)
            audio_segment = _PydubAudioSegment.from_wav(wav_buffer)
            audio_segment.export(
                buffer, format="opus", codec="libopus",
                parameters=["-b:a", "32k"]
            )
            content_type = "audio/ogg"
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            content_type = "audio/wav"
    else:
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        content_type = "audio/wav"

    buffer.seek(0)
    return buffer.read(), content_type


def resolve_voice(voice: Optional[str]) -> str:
    """Resolve a voice name to a reference audio WAV filename.

    Raises APIError 400 if the voice is not in VOICE_MAP.
    """
    if not voice:
        return DEFAULT_VOICE
    voice_lower = voice.lower().replace(" ", "_")
    resolved = VOICE_MAP.get(voice_lower)
    if resolved is None:
        valid = sorted(VOICE_MAP.keys())
        logger.bind(voice=voice).warning("Unknown voice requested")
        raise APIError(400, "UNKNOWN_VOICE", f"Unknown voice '{voice}'",
                       context={"voice": voice, "valid_voices": valid})
    return resolved


_langdetect_model = None


def _get_langdetect():
    """Lazy-load fasttext language detector."""
    global _langdetect_model
    if _langdetect_model is None:
        try:
            from fasttext_langdetect import detect
            _langdetect_model = detect
        except ImportError:
            _langdetect_model = False
    return _langdetect_model


def _detect_language_unicode(text: str) -> str:
    """Fallback: language detection based on Unicode character ranges."""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "Chinese"
        if '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff':
            return "Japanese"
        if '\uac00' <= ch <= '\ud7af':
            return "Korean"
    return "English"


# Map fasttext ISO codes to Qwen language names
_LANG_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian",
}


def detect_language(text: str) -> str:
    """Detect language using fasttext if available, falling back to Unicode heuristic."""
    detector = _get_langdetect()
    if detector:
        try:
            result = detector(text, low_memory=False)
            lang = result.get("lang", "en")
            return _LANG_MAP.get(lang, "English")
        except Exception as e:
            logger.bind(text_preview=text[:40]).debug(
                "fasttext language detection failed, using Unicode heuristic: {}", e
            )
    return _detect_language_unicode(text)


def _adaptive_max_tokens(text: str) -> int:
    """Scale token budget with text length to avoid over-allocating KV cache."""
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    if cjk_chars > len(text) * 0.3:
        return max(128, min(2048, len(text) * 3))
    return max(128, min(2048, len(text.split()) * 8))


def _build_gen_kwargs(text: str, request: TTSRequest) -> dict:
    """Build model.generate() kwargs for a synthesis request."""
    kwargs = {"max_new_tokens": _adaptive_max_tokens(text)}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    return kwargs


def _expand_currency(amount: str, unit: str) -> str:
    """Expand currency amount to words."""
    parts = amount.split('.')
    result = f"{parts[0]} {unit}"
    if len(parts) > 1 and parts[1] != '00':
        result += f" and {parts[1]} cents"
    return result


def _normalize_text(text: str) -> str:
    """Normalize text for TTS: expand numbers, currency, abbreviations."""
    if not TEXT_NORMALIZE:
        return text
    # Currency
    text = re.sub(r'\$(\d+(?:\.\d{2})?)', lambda m: _expand_currency(m.group(1), 'dollars'), text)
    text = re.sub(r'€(\d+)', lambda m: f"{m.group(1)} euros", text)
    text = re.sub(r'£(\d+)', lambda m: f"{m.group(1)} pounds", text)
    # Common abbreviations
    abbrevs = {'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Missus', 'Prof.': 'Professor',
               'Jr.': 'Junior', 'Sr.': 'Senior', 'St.': 'Saint', 'Ave.': 'Avenue',
               'Blvd.': 'Boulevard', 'Dept.': 'Department', 'Est.': 'Established'}
    for abbr, expansion in abbrevs.items():
        text = text.replace(abbr, expansion)
    # Large numbers with commas: 1,000,000 -> 1000000
    while re.search(r'(\d),(\d{3})', text):
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    return text


def _adjust_speed(audio_data: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Adjust audio playback speed. Uses pyrubberband (pitch-preserving) if available,
    falling back to scipy resampling."""
    if speed == 1.0:
        return audio_data
    if _pyrubberband is not None:
        return _pyrubberband.time_stretch(audio_data, sample_rate, speed)
    new_length = int(len(audio_data) / speed)
    if new_length > 0:
        return scipy_signal.resample(audio_data, new_length)
    return audio_data


def _resample_audio(audio_data: np.ndarray, source_rate: int, target_rate: int | None) -> np.ndarray:
    """Resample audio to target sample rate. Returns original if rates match or target is None."""
    if target_rate is None or target_rate == source_rate:
        return audio_data
    num_samples = int(len(audio_data) * target_rate / source_rate)
    if num_samples <= 0:
        return audio_data
    return scipy_signal.resample(audio_data, num_samples).astype(np.float32)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, aware of common abbreviations and CJK punctuation."""
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\u3002|\uff01|\uff1f)\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _do_synthesize(text: str, language: str, voice_file: str, gen_kwargs: dict, instruct: str | None = None) -> tuple[list, int]:
    """Run TTS inference via voice cloning with pre-computed prompts.

    No per-request GC — let CUDA reuse cached allocations.
    ``instruct`` is accepted for API backwards-compat but ignored (Base model
    does not support it).
    """
    prompt = _voice_prompts[voice_file]
    stream = _inference_stream
    with torch.inference_mode():
        if stream is not None:
            with torch.cuda.stream(stream):
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                    **gen_kwargs,
                )
            stream.synchronize()
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                **gen_kwargs,
            )
    return wavs, sr


def _do_synthesize_batch(texts: list[str], languages: list[str], voice_files: list[str], gen_kwargs_list: list[dict]) -> tuple[list, list]:
    """Run batched TTS inference via voice cloning. All items share the max max_new_tokens."""
    max_tokens = max(k.get("max_new_tokens", 2048) for k in gen_kwargs_list)
    extra = {k: v for k, v in gen_kwargs_list[0].items() if k != "max_new_tokens"}
    # Flatten: each _voice_prompts[vf] is [VoiceClonePromptItem]; API expects a flat list
    prompts = [_voice_prompts[vf][0] for vf in voice_files]
    stream = _inference_stream
    with torch.inference_mode():
        if stream is not None:
            with torch.cuda.stream(stream):
                wavs, sr = model.generate_voice_clone(
                    text=texts,
                    language=languages,
                    voice_clone_prompt=prompts,
                    max_new_tokens=max_tokens,
                    **extra,
                )
            stream.synchronize()
        else:
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=prompts,
                max_new_tokens=max_tokens,
                **extra,
            )
    return wavs, sr


def _get_cached_voice_prompt(audio_bytes: bytes, ref_text: str | None):
    """Return a cached voice clone prompt, or compute and cache it.

    Uses model.create_voice_clone_prompt() to pre-compute the speaker embedding
    from reference audio. The returned prompt object can be reused across
    generate_voice_clone() calls, skipping the encoder pass entirely.
    """
    global _voice_cache_hits

    key_material = audio_bytes + (ref_text or "").encode()
    cache_key = hashlib.sha256(key_material).hexdigest()

    if VOICE_CACHE_MAX > 0 and cache_key in _voice_prompt_cache:
        _voice_cache_hits += 1
        _voice_prompt_cache.move_to_end(cache_key)
        logger.debug("Voice prompt cache hit", cache_key=cache_key[:12], hits=_voice_cache_hits)
        return _voice_prompt_cache[cache_key]

    # Decode audio to pass to create_voice_clone_prompt
    try:
        ref_audio_data, ref_sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        raise APIError(400, "INVALID_AUDIO", f"Cannot read reference audio: {e}")

    if len(ref_audio_data.shape) > 1:
        ref_audio_data = ref_audio_data.mean(axis=1)

    # x_vector_only_mode=True extracts speaker embedding without needing ref_text;
    # when ref_text is provided, use ICL mode (x_vector_only_mode=False) for better quality
    t_encode = time.perf_counter()
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=(ref_audio_data, ref_sr),
            ref_text=ref_text,
            x_vector_only_mode=(ref_text is None),
        )
    except Exception as e:
        logger.opt(exception=True).error("create_voice_clone_prompt failed")
        raise APIError(500, "CLONE_FAILED", f"Voice prompt creation failed: {e}")
    logger.bind(cache_key=cache_key[:12], encode_ms=round((time.perf_counter() - t_encode) * 1000)).debug(
        "Voice prompt encoded"
    )

    if VOICE_CACHE_MAX > 0:
        evicted = len(_voice_prompt_cache) >= VOICE_CACHE_MAX
        _voice_prompt_cache[cache_key] = prompt
        while len(_voice_prompt_cache) > VOICE_CACHE_MAX:
            _voice_prompt_cache.popitem(last=False)
        logger.debug("Voice prompt cached{}", " (evicted LRU)" if evicted else "",
                     cache_key=cache_key[:12], cache_size=len(_voice_prompt_cache))

    return prompt


def _do_voice_clone(text: str, language: str, ref_prompt: list, gen_kwargs: dict) -> tuple[list, int]:
    """Run voice clone inference using a pre-computed voice prompt.

    No per-request GC — let CUDA reuse cached allocations.
    """
    stream = _inference_stream
    with torch.inference_mode():
        if stream is not None:
            with torch.cuda.stream(stream):
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=ref_prompt,
                    **gen_kwargs,
                )
            stream.synchronize()
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=ref_prompt,
                **gen_kwargs,
            )
    return wavs, sr


def _do_synthesize_streaming(text, language, voice_file, gen_kwargs):
    """Yield (chunk_np, sr) tuples via per-token streaming."""
    prompt = _voice_prompts[voice_file]
    stream = _inference_stream
    with torch.inference_mode():
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            for chunk, sr in model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                emit_every_frames=STREAM_EMIT_FRAMES,
                decode_window_frames=80,
                overlap_samples=512,
                first_chunk_emit_every=STREAM_FIRST_EMIT if STREAM_FIRST_EMIT > 0 else None,
                first_chunk_decode_window=48,
                first_chunk_frames=48,
                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.05),
                max_new_tokens=gen_kwargs.get("max_new_tokens", 2048),
            ):
                yield chunk, sr


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
    # Fire-and-forget: we consume chunks from chunk_queue, not the submit future.
    asyncio.ensure_future(_infer_queue.submit(_run, priority=PRIORITY_REALTIME))

    while True:
        item = await asyncio.wait_for(chunk_queue.get(), timeout=REQUEST_TIMEOUT)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def _encode_audio_async(audio_data: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    """Run audio encoding in the CPU thread pool, overlapping with GPU work."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _encode_executor,
        lambda: convert_audio_format(audio_data, sample_rate, output_format)
    )


@app.post("/v1/audio/speech")
async def synthesize_speech(request: TTSRequest) -> Response:
    """OpenAI-compatible TTS endpoint using Base model with voice cloning."""
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    if not request.input or not request.input.strip():
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech").warning("Request rejected: empty input")
        raise APIError(400, "EMPTY_INPUT", "Input text is required")

    if request.instruct:
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech").warning(
            "instruct parameter ignored — Base model does not support instruct"
        )

    voice_file = resolve_voice(request.voice)
    text = request.input.strip()
    await _queue_tracker.acquire(request_id, "/v1/audio/speech")

    try:
        # Fast path: return cached audio without touching the GPU
        cache_key = _audio_cache_key(
            text, voice_file, request.speed, request.response_format, request.language or "", request.instruct or ""
        )
        cached = _get_audio_cache(cache_key)
        if cached is not None:
            logger.bind(request_id=request_id, endpoint="/v1/audio/speech",
                        voice=voice_file, chars=len(text), format=request.response_format,
                        total_ms=round((time.perf_counter() - t_start) * 1000),
                        ).info("synthesis_cache_hit")
            return Response(
                content=cached[0],
                media_type=cached[1],
                headers={
                    "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
                },
            )

        await _ensure_model_loaded()

        language = request.language or detect_language(request.input)
        text = _normalize_text(text)
        gen_kwargs = _build_gen_kwargs(text, request)

        t_infer_start = time.perf_counter()
        wavs, sr = await asyncio.wait_for(
            _infer_queue.submit_batch(
                text=text, language=language, speaker=voice_file, gen_kwargs=gen_kwargs,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        t_infer_done = time.perf_counter()

        audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        audio_data = _adjust_speed(audio_data, sr, request.speed)

        if request.sample_rate:
            audio_data = _resample_audio(audio_data, sr, request.sample_rate)
            sr = request.sample_rate

        audio_bytes, content_type = await _encode_audio_async(
            audio_data, sr, request.response_format
        )
        t_encode_done = time.perf_counter()

        logger.bind(
            request_id=request_id,
            endpoint="/v1/audio/speech",
            voice=voice_file,
            language=language,
            chars=len(text),
            format=request.response_format,
            infer_ms=round((t_infer_done - t_infer_start) * 1000),
            encode_ms=round((t_encode_done - t_infer_done) * 1000),
            total_ms=round((t_encode_done - t_start) * 1000),
        ).info("synthesis_complete")

        _set_audio_cache(cache_key, audio_bytes, content_type)

        if _prometheus_available:
            tts_requests_total.labels(voice=voice_file, format=request.response_format).inc()
            tts_inference_duration.observe(t_infer_done - t_infer_start)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            },
        )

    except asyncio.TimeoutError:
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech",
                     timeout_s=REQUEST_TIMEOUT).error("Synthesis timed out")
        raise APIError(504, "SYNTHESIS_TIMEOUT", "Synthesis timed out",
                       context={"timeout_s": REQUEST_TIMEOUT})
    except APIError:
        raise
    except Exception as e:
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech").opt(exception=True).error("Synthesis failed")
        raise APIError(500, "SYNTHESIS_FAILED", f"Synthesis failed: {str(e)}")
    finally:
        await _queue_tracker.release()


@app.post("/v1/audio/speech/stream")
async def synthesize_speech_stream(request: TTSRequest) -> StreamingResponse:
    """Sentence-chunked SSE streaming TTS endpoint."""
    global _last_used
    request_id = str(uuid.uuid4())[:8]
    await _ensure_model_loaded()
    await _queue_tracker.acquire(request_id, "/v1/audio/speech/stream")

    if not request.input or not request.input.strip():
        await _queue_tracker.release()
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream").warning("Request rejected: empty input")
        raise APIError(400, "EMPTY_INPUT", "Input text is required")

    voice_file = resolve_voice(request.voice)
    language = request.language or detect_language(request.input)
    text = request.input.strip()
    sentences = _split_sentences(text)

    if not sentences:
        await _queue_tracker.release()
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream").warning("Request rejected: no sentences")
        raise APIError(400, "NO_SENTENCES", "No sentences found in input")

    effective_mode = _effective_stream_mode(request)
    if effective_mode == "token" and not _HAS_STREAMING:
        await _queue_tracker.release()
        raise APIError(400, "TOKEN_STREAMING_UNAVAILABLE",
                       "stream_mode='token' requires the streaming TTS fork")

    async def generate():
        try:
            t_stream_start = time.perf_counter()
            chunks_sent = 0

            if effective_mode == "token":
                # Per-token streaming — full text, no sentence splitting
                gen_kwargs = _build_gen_kwargs(text, request)
                try:
                    async for chunk, sr_val in _stream_synthesize(text, language, voice_file, gen_kwargs):
                        audio_data = np.array(chunk, dtype=np.float32, copy=True)
                        if audio_data.ndim > 1:
                            audio_data = audio_data.squeeze()
                        audio_data = _adjust_speed(audio_data, sr_val, request.speed)
                        if request.sample_rate:
                            audio_data = _resample_audio(audio_data, sr_val, request.sample_rate)
                        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                        pcm_bytes = audio_int16.tobytes()
                        yield f"data: {base64.b64encode(pcm_bytes).decode()}\n\n"
                        t_now = time.perf_counter()
                        logger.bind(
                            request_id=request_id,
                            chunk_idx=chunks_sent,
                            ms=round((t_now - t_stream_start) * 1000) if chunks_sent == 0 else None,
                            ttfc_ms=round((t_now - t_stream_start) * 1000) if chunks_sent == 0 else None,
                        ).info("token_chunk_sent")
                        chunks_sent += 1
                        _last_used = time.time()
                except asyncio.TimeoutError:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream",
                                chunks_sent=chunks_sent).error("Token stream timed out")
                    yield "data: [ERROR] Synthesis timed out\n\n"
                    return
                except Exception as e:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream",
                                chunks_sent=chunks_sent).opt(exception=True).error("Token stream failed")
                    yield f"data: [ERROR] {str(e)}\n\n"
                    return

                yield "data: [DONE]\n\n"
                logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream", voice=voice_file,
                            language=language, chunks_sent=chunks_sent, mode=effective_mode,
                            chars=len(text), total_ms=round((time.perf_counter() - t_stream_start) * 1000),
                            ).info("stream_complete")
                return

            # else: sentence-level streaming (existing code below, unchanged)
            def _make_synth_fn(s, gk):
                return lambda: _do_synthesize(s, language, voice_file, gk)

            # Pre-fetch: submit first sentence immediately
            pending_future = None
            sentence_idx = 0

            if sentences:
                gk = _build_gen_kwargs(sentences[0], request)
                pending_future = asyncio.ensure_future(
                    asyncio.wait_for(
                        _infer_queue.submit(
                            _make_synth_fn(sentences[0], gk),
                            priority=PRIORITY_REALTIME,
                        ),
                        timeout=REQUEST_TIMEOUT,
                    )
                )

            while sentence_idx < len(sentences):
                try:
                    t_sent = time.perf_counter()
                    wavs, sr_val = await pending_future

                    # Pre-fetch next sentence while we process current
                    next_idx = sentence_idx + 1
                    if next_idx < len(sentences):
                        gk_next = _build_gen_kwargs(sentences[next_idx], request)
                        pending_future = asyncio.ensure_future(
                            asyncio.wait_for(
                                _infer_queue.submit(
                                    _make_synth_fn(sentences[next_idx], gk_next),
                                    priority=PRIORITY_REALTIME,
                                ),
                                timeout=REQUEST_TIMEOUT,
                            )
                        )

                    audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.squeeze()

                    audio_data = _adjust_speed(audio_data, sr_val, request.speed)

                    if request.sample_rate:
                        audio_data = _resample_audio(audio_data, sr_val, request.sample_rate)

                    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                    pcm_bytes = audio_int16.tobytes()
                    yield f"data: {base64.b64encode(pcm_bytes).decode()}\n\n"
                    sent_ms = round((time.perf_counter() - t_sent) * 1000)
                    logger.bind(
                        request_id=request_id,
                        sentence_idx=chunks_sent,
                        sentences_total=len(sentences),
                        chars=len(sentences[sentence_idx]),
                        ms=sent_ms,
                        ttfs_ms=round((time.perf_counter() - t_stream_start) * 1000) if chunks_sent == 0 else None,
                    ).info("sentence_synthesized")
                    chunks_sent += 1
                    sentence_idx += 1

                    _last_used = time.time()

                except asyncio.TimeoutError:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream", voice=voice_file,
                                 chunks_sent=chunks_sent, timeout_s=REQUEST_TIMEOUT).error("Stream synthesis timed out")
                    yield "data: [ERROR] Synthesis timed out\n\n"
                    return
                except Exception as e:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream", voice=voice_file,
                                 chunks_sent=chunks_sent).opt(exception=True).error("Stream synthesis failed")
                    yield f"data: [ERROR] {str(e)}\n\n"
                    return

            yield "data: [DONE]\n\n"
            logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream", voice=voice_file,
                         language=language, sentences=len(sentences), chunks_sent=chunks_sent,
                         chars=len(text), total_ms=round((time.perf_counter() - t_stream_start) * 1000),
                         ).info("stream_complete")
        finally:
            await _queue_tracker.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/v1/audio/speech/clone")
async def clone_voice(
    file: UploadFile = File(...),
    input: str = Form(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: str = Form("wav"),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
) -> Response:
    """Voice cloning endpoint - requires a reference audio file."""
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    if not input or not input.strip():
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/clone").warning("Request rejected: empty input")
        raise APIError(400, "EMPTY_INPUT", "Input text is required")

    await _queue_tracker.acquire(request_id, "/v1/audio/speech/clone")
    await _ensure_model_loaded()

    try:
        # Read reference audio and compute/cache speaker embedding
        audio_bytes = await file.read()
        text = input.strip()
        language = language or detect_language(text)
        text = _normalize_text(text)
        gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(text)}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        ref_prompt = _get_cached_voice_prompt(
            audio_bytes, ref_text.strip() if ref_text else None
        )

        t_infer_start = time.perf_counter()
        wavs, sr = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_voice_clone(
                    text,
                    language,
                    ref_prompt,
                    gen_kwargs,
                ),
                priority=PRIORITY_BATCH,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        t_infer_end = time.perf_counter()

        audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        audio_bytes_out, content_type = await _encode_audio_async(
            audio_data, sr, response_format
        )
        t_end = time.perf_counter()

        logger.bind(
            request_id=request_id,
            endpoint="/v1/audio/speech/clone",
            infer_ms=round((t_infer_end - t_infer_start) * 1000),
            encode_ms=round((t_end - t_infer_end) * 1000),
            total_ms=round((t_end - t_start) * 1000),
            chars=len(text),
            format=response_format,
            language=language,
        ).info("request_complete")

        return Response(
            content=audio_bytes_out,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"'
            },
        )

    except asyncio.TimeoutError:
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/clone",
                     timeout_s=REQUEST_TIMEOUT).error("Voice clone timed out")
        raise APIError(504, "SYNTHESIS_TIMEOUT", "Voice clone timed out",
                       context={"timeout_s": REQUEST_TIMEOUT})
    except APIError:
        raise
    except Exception as e:
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/clone").opt(exception=True).error("Voice clone failed")
        raise APIError(500, "CLONE_FAILED", f"Voice clone failed: {str(e)}")
    finally:
        await _queue_tracker.release()


@app.post("/v1/audio/speech/stream/pcm")
async def synthesize_speech_stream_pcm(request: TTSRequest) -> StreamingResponse:
    """Raw PCM streaming TTS endpoint — no SSE framing, no base64.

    Splits text into sentences and streams each as raw int16 PCM bytes.
    Headers report the audio format: 24000 Hz, 16-bit, mono.
    """
    global _last_used
    request_id = str(uuid.uuid4())[:8]
    await _ensure_model_loaded()
    await _queue_tracker.acquire(request_id, "/v1/audio/speech/stream/pcm")

    if not request.input or not request.input.strip():
        await _queue_tracker.release()
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm").warning("Request rejected: empty input")
        raise APIError(400, "EMPTY_INPUT", "Input text is required")

    voice_file = resolve_voice(request.voice)
    language = request.language or detect_language(request.input)
    text = request.input.strip()
    sentences = _split_sentences(text)
    if not sentences:
        await _queue_tracker.release()
        logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm").warning("Request rejected: no sentences")
        raise APIError(400, "NO_SENTENCES", "No sentences found in input")

    effective_mode = _effective_stream_mode(request)
    if effective_mode == "token" and not _HAS_STREAMING:
        await _queue_tracker.release()
        raise APIError(400, "TOKEN_STREAMING_UNAVAILABLE",
                       "stream_mode='token' requires the streaming TTS fork")

    async def pcm_generator():
        global _last_used
        try:
            t_pcm_start = time.perf_counter()
            chunks_sent = 0

            if effective_mode == "token":
                gen_kwargs = _build_gen_kwargs(text, request)
                try:
                    async for chunk, sr_val in _stream_synthesize(text, language, voice_file, gen_kwargs):
                        _last_used = time.time()
                        audio_data = np.array(chunk, dtype=np.float32, copy=True)
                        if audio_data.ndim > 1:
                            audio_data = audio_data.squeeze()
                        audio_data = _adjust_speed(audio_data, sr_val, request.speed)
                        if request.sample_rate:
                            audio_data = _resample_audio(audio_data, sr_val, request.sample_rate)
                        pcm_data = np.clip(audio_data, -1.0, 1.0)
                        pcm_bytes = (pcm_data * 32767).astype(np.int16).tobytes()
                        yield pcm_bytes
                        t_now = time.perf_counter()
                        logger.bind(
                            request_id=request_id,
                            chunk_idx=chunks_sent,
                            ms=round((t_now - t_pcm_start) * 1000) if chunks_sent == 0 else None,
                            ttfc_ms=round((t_now - t_pcm_start) * 1000) if chunks_sent == 0 else None,
                        ).info("token_chunk_sent")
                        chunks_sent += 1
                except asyncio.TimeoutError:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm",
                                chunks_sent=chunks_sent).error("Token PCM stream timed out")
                    return
                except Exception:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm",
                                chunks_sent=chunks_sent).opt(exception=True).error("Token PCM stream failed")
                    return

                logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                            language=language, chunks_sent=chunks_sent, mode=effective_mode,
                            chars=len(text), total_ms=round((time.perf_counter() - t_pcm_start) * 1000),
                            ).info("pcm_stream_complete")
                return

            # else: sentence-level (existing code below, unchanged)
            def _make_synth_fn(s, gk):
                return lambda: _do_synthesize(s, language, voice_file, gk)

            # Pre-fetch: submit first sentence immediately
            pending_future = None
            sentence_idx = 0

            if sentences:
                gk = _build_gen_kwargs(sentences[0], request)
                pending_future = asyncio.ensure_future(
                    asyncio.wait_for(
                        _infer_queue.submit(
                            _make_synth_fn(sentences[0], gk),
                            priority=PRIORITY_REALTIME,
                        ),
                        timeout=REQUEST_TIMEOUT,
                    )
                )

            while sentence_idx < len(sentences):
                try:
                    t_sent = time.perf_counter()
                    wavs, sr_val = await pending_future

                    # Pre-fetch next sentence while we process current
                    next_idx = sentence_idx + 1
                    if next_idx < len(sentences):
                        gk_next = _build_gen_kwargs(sentences[next_idx], request)
                        pending_future = asyncio.ensure_future(
                            asyncio.wait_for(
                                _infer_queue.submit(
                                    _make_synth_fn(sentences[next_idx], gk_next),
                                    priority=PRIORITY_REALTIME,
                                ),
                                timeout=REQUEST_TIMEOUT,
                            )
                        )

                    _last_used = time.time()
                    audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.squeeze()
                    audio_data = _adjust_speed(audio_data, sr_val, request.speed)
                    if request.sample_rate:
                        audio_data = _resample_audio(audio_data, sr_val, request.sample_rate)
                    pcm_data = np.clip(audio_data, -1.0, 1.0)
                    pcm_bytes = (pcm_data * 32767).astype(np.int16).tobytes()
                    yield pcm_bytes
                    sent_ms = round((time.perf_counter() - t_sent) * 1000)
                    logger.bind(
                        request_id=request_id,
                        sentence_idx=chunks_sent,
                        sentences_total=len(sentences),
                        chars=len(sentences[sentence_idx]),
                        ms=sent_ms,
                        ttfs_ms=round((time.perf_counter() - t_pcm_start) * 1000) if chunks_sent == 0 else None,
                    ).info("sentence_synthesized")
                    chunks_sent += 1
                    sentence_idx += 1

                except asyncio.TimeoutError:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                                 chunks_sent=chunks_sent, timeout_s=REQUEST_TIMEOUT).error("PCM stream timed out")
                    break
                except Exception:
                    logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                                 chunks_sent=chunks_sent).opt(exception=True).error("PCM stream failed")
                    break

            logger.bind(request_id=request_id, endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                         language=language, sentences=len(sentences), chunks_sent=chunks_sent,
                         chars=len(text), total_ms=round((time.perf_counter() - t_pcm_start) * 1000),
                         ).info("pcm_stream_complete")
        finally:
            await _queue_tracker.release()

    pcm_sr = request.sample_rate or 24000
    return StreamingResponse(
        pcm_generator(),
        media_type="application/octet-stream",
        headers={
            "X-PCM-Sample-Rate": str(pcm_sr),
            "X-PCM-Bit-Depth": "16",
            "X-PCM-Channels": "1",
            "Content-Disposition": 'attachment; filename="speech.pcm"',
        },
    )


@app.websocket("/v1/audio/speech/ws")
async def ws_synthesize(websocket: WebSocket) -> None:
    """WebSocket streaming endpoint. Send JSON, receive binary PCM per sentence."""
    global _last_used
    ws_id = str(uuid.uuid4())[:8]
    request_id = ws_id
    await websocket.accept()
    logger.bind(request_id=request_id, ws_id=ws_id, endpoint="/v1/audio/speech/ws").debug("WebSocket connected")
    messages_handled = 0
    t_ws_start = time.perf_counter()
    try:
        while True:
            data = await websocket.receive_json()
            text = (data.get("input") or "").strip()
            if not text:
                logger.bind(request_id=request_id, ws_id=ws_id).debug("WebSocket empty input, skipping")
                await websocket.send_json({"event": "error", "detail": "input is required"})
                continue

            voice_file = resolve_voice(data.get("voice"))
            language = data.get("language") or detect_language(text)
            speed = float(data.get("speed", 1.0))
            ws_sample_rate = data.get("sample_rate")
            ws_temperature = data.get("temperature")
            ws_top_p = data.get("top_p")
            ws_stream_mode = data.get("stream_mode")
            if ws_stream_mode is not None and ws_stream_mode not in ("sentence", "token"):
                await websocket.send_json({"event": "error", "detail": "stream_mode must be 'sentence' or 'token'"})
                continue

            await _ensure_model_loaded()
            await _queue_tracker.acquire(request_id, "/v1/audio/speech/ws")

            try:
                def _ws_build_gen_kwargs(s):
                    gk = {"max_new_tokens": _adaptive_max_tokens(s)}
                    if ws_temperature is not None:
                        gk["temperature"] = float(ws_temperature)
                    if ws_top_p is not None:
                        gk["top_p"] = float(ws_top_p)
                    return gk

                effective_mode = ws_stream_mode or STREAM_TYPE
                if effective_mode == "token" and not _HAS_STREAMING:
                    await websocket.send_json({"event": "error", "detail": "stream_mode='token' requires the streaming TTS fork"})
                    await _queue_tracker.release()
                    continue
                if effective_mode == "token":
                    # Per-token streaming — no sentence splitting
                    ws_gen_kwargs = _ws_build_gen_kwargs(text)
                    t_ws_token_start = time.perf_counter()
                    chunk_count = 0
                    async for chunk, sr_val in _stream_synthesize(text, language, voice_file, ws_gen_kwargs):
                        audio_data = np.array(chunk, dtype=np.float32, copy=True)
                        if audio_data.ndim > 1:
                            audio_data = audio_data.squeeze()
                        audio_data = _adjust_speed(audio_data, sr_val, speed)
                        if ws_sample_rate:
                            audio_data = _resample_audio(audio_data, sr_val, int(ws_sample_rate))
                        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                        await websocket.send_bytes(pcm)
                        _last_used = time.time()
                        logger.bind(
                            request_id=request_id, ws_id=ws_id,
                            chunk_idx=chunk_count,
                            ms=round((time.perf_counter() - t_ws_token_start) * 1000) if chunk_count == 0 else None,
                        ).info("ws_token_chunk_sent")
                        chunk_count += 1
                    await websocket.send_json({"event": "done"})
                    messages_handled += 1
                    continue  # skip sentence-level code below

                sentences = _split_sentences(text)
                if not sentences:
                    sentences = [text]

                def _ws_make_synth_fn(s, gk):
                    return lambda: _do_synthesize(s, language, voice_file, gk)

                # Pre-fetch: submit first sentence immediately
                pending_future = None
                sentence_idx = 0

                if sentences:
                    gk = _ws_build_gen_kwargs(sentences[0])
                    pending_future = asyncio.ensure_future(
                        asyncio.wait_for(
                            _infer_queue.submit(
                                _ws_make_synth_fn(sentences[0], gk),
                                priority=PRIORITY_REALTIME,
                            ),
                            timeout=REQUEST_TIMEOUT,
                        )
                    )

                while sentence_idx < len(sentences):
                    t_sent = time.perf_counter()
                    wavs, sr_val = await pending_future

                    # Pre-fetch next sentence while we process current
                    next_idx = sentence_idx + 1
                    if next_idx < len(sentences):
                        gk_next = _ws_build_gen_kwargs(sentences[next_idx])
                        pending_future = asyncio.ensure_future(
                            asyncio.wait_for(
                                _infer_queue.submit(
                                    _ws_make_synth_fn(sentences[next_idx], gk_next),
                                    priority=PRIORITY_REALTIME,
                                ),
                                timeout=REQUEST_TIMEOUT,
                            )
                        )

                    audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.squeeze()

                    audio_data = _adjust_speed(audio_data, sr_val, speed)

                    if ws_sample_rate:
                        audio_data = _resample_audio(audio_data, sr_val, int(ws_sample_rate))

                    pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    await websocket.send_bytes(pcm)
                    _last_used = time.time()
                    logger.bind(
                        request_id=request_id,
                        ws_id=ws_id,
                        sentence_idx=sentence_idx,
                        sentences_total=len(sentences),
                        chars=len(sentences[sentence_idx]),
                        ms=round((time.perf_counter() - t_sent) * 1000),
                    ).info("sentence_synthesized")
                    sentence_idx += 1

                await websocket.send_json({"event": "done"})
                messages_handled += 1
            finally:
                await _queue_tracker.release()
    except WebSocketDisconnect:
        logger.bind(request_id=request_id, ws_id=ws_id, messages=messages_handled,
                     duration_ms=round((time.perf_counter() - t_ws_start) * 1000)).info("WebSocket disconnected")
    except Exception as e:
        logger.bind(request_id=request_id, ws_id=ws_id, messages=messages_handled).opt(exception=True).error("WebSocket error")
        try:
            await websocket.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
