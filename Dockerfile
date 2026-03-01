# Single-stage build using devel image — ensures flash-attn compiles
# against the same PyTorch it runs with (no ABI mismatch).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

# Python runtime tuning
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA memory allocator tuning — reduce fragmentation for shared GPU
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
ENV TOKENIZERS_PARALLELISM=false

# Limit CPU thread spawning — GPU does the heavy work, excess threads just contend
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# jemalloc replaces ptmalloc2 to reduce RSS bloat from arena fragmentation
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg sox rubberband-cli libjemalloc2 libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# FlashAttention 2 — force source compilation matching PyTorch's CXX11 ABI=0.
# RTX 4060 = sm_89 (Ada Lovelace) — only build for this arch.
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="8.9" \
    MAX_JOBS=2 \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Optional quantization
RUN pip install --no-cache-dir "bitsandbytes>=0.43.0" || true
# torchao for FP8 quantization (transformers 4.57+ expects this version's API)
RUN pip install --no-cache-dir torchao || true

# Streaming TTS fork — adds stream_generate_voice_clone() to qwen-tts
# Installed AFTER flash-attn to preserve layer cache
RUN pip install --no-cache-dir --no-deps \
    "qwen-tts @ git+https://github.com/rekuenkdr/Qwen3-TTS-streaming.git" \
    || true

# Copy application
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY server.py /app/server.py
COPY gateway.py /app/gateway.py
COPY worker.py /app/worker.py
COPY voices/ /app/voices/

EXPOSE 8000

CMD if [ "${GATEWAY_MODE:-false}" = "true" ]; then \
      exec uvicorn gateway:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools --no-access-log; \
    else \
      exec /app/docker-entrypoint.sh uvicorn server:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools --no-access-log --timeout-keep-alive 65; \
    fi
