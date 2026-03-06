# Stream Mode Hot-Swap — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow per-request switching between `"sentence"` and `"token"` streaming modes via a `stream_mode` field in the request body.

**Architecture:** Add `stream_mode` optional field to `TTSRequest`, add a `_effective_stream_mode()` helper that falls back to the `STREAM_TYPE` env var, and replace 3 global checks with per-request resolution. WS endpoint parses the field manually from the JSON dict.

**Tech Stack:** Python, FastAPI, Pydantic v2, pytest

---

### Task 1: Add `stream_mode` field to TTSRequest and validator

**Files:**
- Modify: `server.py:571-580` (TTSRequest class)

**Step 1: Write the failing test**

In `server_test.py`, add at the end:

```python
class TestStreamMode(unittest.TestCase):
    def test_valid_token_mode(self):
        req = server.TTSRequest(input="hello", stream_mode="token")
        assert req.stream_mode == "token"

    def test_valid_sentence_mode(self):
        req = server.TTSRequest(input="hello", stream_mode="sentence")
        assert req.stream_mode == "sentence"

    def test_none_default(self):
        req = server.TTSRequest(input="hello")
        assert req.stream_mode is None

    def test_invalid_mode_rejected(self):
        with pytest.raises(Exception):
            server.TTSRequest(input="hello", stream_mode="invalid")
```

**Step 2: Run test to verify it fails**

Run: `pytest server_test.py::TestStreamMode -v`
Expected: FAIL — `stream_mode` field doesn't exist yet

**Step 3: Write minimal implementation**

In `server.py`, add `stream_mode` field to `TTSRequest` (after line 580):

```python
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
```

Also add `field_validator` to the pydantic imports if not already present. Check existing imports:

```bash
grep "from pydantic" server.py
```

Add `field_validator` to the import if missing.

**Step 4: Run test to verify it passes**

Run: `pytest server_test.py::TestStreamMode -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: add stream_mode field to TTSRequest with validation"
```

---

### Task 2: Add `_effective_stream_mode()` helper and tests

**Files:**
- Modify: `server.py` (after TTSRequest class, around line 585)
- Modify: `server_test.py`

**Step 1: Write the failing test**

Add to `TestStreamMode` in `server_test.py`:

```python
    def test_effective_mode_uses_request_token(self):
        req = server.TTSRequest(input="hello", stream_mode="token")
        assert server._effective_stream_mode(req) == "token"

    def test_effective_mode_uses_request_sentence(self):
        req = server.TTSRequest(input="hello", stream_mode="sentence")
        assert server._effective_stream_mode(req) == "sentence"

    @patch.object(server, "STREAM_TYPE", "sentence")
    def test_effective_mode_falls_back_to_env_sentence(self):
        req = server.TTSRequest(input="hello")
        assert server._effective_stream_mode(req) == "sentence"

    @patch.object(server, "STREAM_TYPE", "token")
    def test_effective_mode_falls_back_to_env_token(self):
        req = server.TTSRequest(input="hello")
        assert server._effective_stream_mode(req) == "token"
```

Also add to the import block at top of `server_test.py` (line 22-30):

```python
    from server import (
        ...existing imports...,
        _effective_stream_mode,
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest server_test.py::TestStreamMode::test_effective_mode_uses_request_token -v`
Expected: FAIL — `_effective_stream_mode` not defined

**Step 3: Write minimal implementation**

In `server.py`, add after the `TTSRequest` class (around line 585):

```python
def _effective_stream_mode(request: TTSRequest) -> str:
    return request.stream_mode or STREAM_TYPE
```

**Step 4: Run test to verify it passes**

Run: `pytest server_test.py::TestStreamMode -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: add _effective_stream_mode() helper with env fallback"
```

---

### Task 3: Replace STREAM_TYPE checks in `/stream` endpoint

**Files:**
- Modify: `server.py:1571` (SSE stream endpoint)

**Step 1: Identify the change**

At line 1571, replace:
```python
if STREAM_TYPE == "token":
```
with:
```python
effective_mode = _effective_stream_mode(request)
if effective_mode == "token":
```

Also update the log context at the end of this endpoint to include `mode=effective_mode` instead of hard-coded `mode="token"`.

**Step 2: Make the change**

Edit `server.py:1571`. The `request` variable is already in scope (it's the endpoint parameter).

**Step 3: Run tests**

Run: `pytest server_test.py -v`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add server.py
git commit -m "feat: use per-request stream_mode in /stream endpoint"
```

---

### Task 4: Replace STREAM_TYPE checks in `/stream/pcm` endpoint

**Files:**
- Modify: `server.py:1832` (PCM stream endpoint)

**Step 1: Identify the change**

At line 1832, replace:
```python
if STREAM_TYPE == "token":
```
with:
```python
effective_mode = _effective_stream_mode(request)
if effective_mode == "token":
```

Update any log context referencing `mode="token"` to use `mode=effective_mode`.

**Step 2: Make the change**

Edit `server.py:1832`.

**Step 3: Run tests**

Run: `pytest server_test.py -v`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add server.py
git commit -m "feat: use per-request stream_mode in /stream/pcm endpoint"
```

---

### Task 5: Replace STREAM_TYPE check in `/ws` endpoint

**Files:**
- Modify: `server.py:1978-1997` (WebSocket endpoint)

**Step 1: Identify the change**

The WS endpoint parses JSON manually via `data.get(...)`. Add `stream_mode` parsing alongside existing fields (after line 1983):

```python
ws_stream_mode = data.get("stream_mode")
if ws_stream_mode is not None and ws_stream_mode not in ("sentence", "token"):
    await websocket.send_json({"event": "error", "detail": "stream_mode must be 'sentence' or 'token'"})
    await _queue_tracker.release()
    continue
```

Then at line 1997, replace:
```python
if STREAM_TYPE == "token":
```
with:
```python
effective_mode = ws_stream_mode or STREAM_TYPE
if effective_mode == "token":
```

**Step 2: Make the change**

Edit `server.py:1978-1997`.

**Step 3: Run tests**

Run: `pytest server_test.py -v`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add server.py
git commit -m "feat: use per-request stream_mode in /ws endpoint"
```

---

### Task 6: Add fallback guard for token mode without streaming fork

**Files:**
- Modify: `server.py` (in all 3 streaming endpoints, right after resolving effective_mode)

**Step 1: Write the guard**

In each of the 3 streaming endpoints, after `effective_mode = ...`, add the guard before entering the `if effective_mode == "token":` block. For `/stream` and `/stream/pcm`:

```python
if effective_mode == "token" and not _HAS_STREAMING:
    await _queue_tracker.release()
    raise APIError(400, "TOKEN_STREAMING_UNAVAILABLE",
                   "stream_mode='token' requires the streaming TTS fork")
```

For `/ws`, send a JSON error instead:

```python
if effective_mode == "token" and not _HAS_STREAMING:
    await websocket.send_json({"event": "error", "detail": "stream_mode='token' requires the streaming TTS fork"})
    await _queue_tracker.release()
    continue
```

**Step 2: Run tests**

Run: `pytest server_test.py -v`
Expected: All existing tests still pass

**Step 3: Commit**

```bash
git add server.py
git commit -m "feat: guard against token mode when streaming fork unavailable"
```

---

### Task 7: Update CLAUDE.md and run full test suite

**Files:**
- Modify: `CLAUDE.md` (TTSRequest docs, key env vars section)

**Step 1: Update CLAUDE.md**

In the `TTSRequest` / request body docs, mention the new `stream_mode` field. In the architecture section under streaming, note that mode is now per-request with env fallback.

**Step 2: Run full test suite**

```bash
pytest server_test.py -v
```

Expected: All tests pass, including the new `TestStreamMode` tests.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document stream_mode per-request parameter"
```
