# Stream Mode Hot-Swap — Design

**Date:** 2026-03-07
**Status:** Approved

## Summary

Allow callers to switch between `"sentence"` and `"token"` streaming modes per-request via a `stream_mode` field in the JSON request body, instead of requiring a container restart with a different `STREAM_TYPE` env var.

## Changes

### 1. `TTSRequest` model — new field

```python
stream_mode: Optional[str] = None  # "sentence" or "token"; None = use STREAM_TYPE env default
```

Field-level validator returns 400 on invalid values.

### 2. Resolution helper

```python
def _effective_stream_mode(request: TTSRequest) -> str:
    return request.stream_mode or STREAM_TYPE
```

### 3. Streaming endpoints

All three (`/stream`, `/stream/pcm`, `/ws`): replace `if STREAM_TYPE == "token":` with `if _effective_stream_mode(request) == "token":`.

### 4. Fallback guard

If `stream_mode="token"` is requested but `_HAS_STREAMING` is `False` (streaming fork not installed), return 400 with a clear error. Explicit per-request mode should not silently degrade.

### 5. Health endpoint

No change — `stream_type` reports the server default.

### 6. Tests

Unit tests in `server_test.py`:
- `stream_mode="token"` overrides `STREAM_TYPE="sentence"` default
- `stream_mode="sentence"` overrides `STREAM_TYPE="token"` default
- `stream_mode=None` falls back to env default
- Invalid `stream_mode` returns 400

## Not changing

- `STREAM_TYPE` env var remains the server-wide default
- Non-streaming `/v1/audio/speech` ignores the field (like `instruct`)
- No new env vars
