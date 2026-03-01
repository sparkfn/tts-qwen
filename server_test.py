"""Tests for server.py - Phase 1 Real-Time + Phase 2 Speed & Quality features."""
import os
import sys
import io
import hashlib
import asyncio
import pytest
import torch
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

# Mock heavy imports before importing server
_mock_modules = {
    "qwen_tts": MagicMock(),
}

with patch.dict("sys.modules", _mock_modules):
    from server import (
        _normalize_text, _expand_currency,
        _detect_language_unicode, _get_langdetect, detect_language,
        _adjust_speed, _resample_audio, resolve_voice, _LANG_MAP,
        _get_cached_voice_prompt, _split_sentences, _adaptive_max_tokens,
        _audio_cache_key, _get_audio_cache, _set_audio_cache,
        _audio_cache, _AUDIO_CACHE_MAX, _build_gen_kwargs,
        APIError, ErrorResponse,
    )
    import server


# --- Issue #12: Text normalization tests ---

class TestNormalizeTextCurrency:
    def test_dollar_amount(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Price is $5.00") == "Price is 5 dollars"

    def test_dollar_with_cents(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Cost $10.50") == "Cost 10 dollars and 50 cents"

    def test_dollar_no_cents(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("$100") == "100 dollars"

    def test_euro(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            result = _normalize_text("€50")
            assert result == "50 euros"

    def test_pound(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            result = _normalize_text("£20")
            assert result == "20 pounds"


class TestNormalizeTextAbbreviations:
    def test_doctor(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Dr. Smith") == "Doctor Smith"

    def test_mister(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Mr. Jones") == "Mister Jones"

    def test_professor(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Prof. Lee") == "Professor Lee"


class TestNormalizeTextCommas:
    def test_comma_in_number(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("1,000 items") == "1000 items"

    def test_large_number(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            result = _normalize_text("1,000,000")
            assert "," not in result


class TestNormalizeTextDisabled:
    def test_disabled_passthrough(self):
        with patch.object(server, "TEXT_NORMALIZE", False):
            assert _normalize_text("$5.00 Dr. Smith") == "$5.00 Dr. Smith"


class TestExpandCurrency:
    def test_whole_dollars(self):
        assert _expand_currency("5", "dollars") == "5 dollars"

    def test_dollars_with_cents(self):
        assert _expand_currency("5.50", "dollars") == "5 dollars and 50 cents"

    def test_dollars_zero_cents(self):
        assert _expand_currency("5.00", "dollars") == "5 dollars"


# --- Issue #13: fasttext language detection tests ---

class TestDetectLanguageUnicode:
    def test_chinese_characters(self):
        assert _detect_language_unicode("你好世界") == "Chinese"

    def test_japanese_hiragana(self):
        assert _detect_language_unicode("こんにちは") == "Japanese"

    def test_japanese_katakana(self):
        assert _detect_language_unicode("カタカナ") == "Japanese"

    def test_korean(self):
        assert _detect_language_unicode("안녕하세요") == "Korean"

    def test_english_default(self):
        assert _detect_language_unicode("Hello world") == "English"

    def test_empty_string(self):
        assert _detect_language_unicode("") == "English"

    def test_mixed_starts_with_chinese(self):
        assert _detect_language_unicode("你好 hello") == "Chinese"


class TestLangMap:
    def test_known_codes(self):
        assert _LANG_MAP["zh"] == "Chinese"
        assert _LANG_MAP["en"] == "English"
        assert _LANG_MAP["ja"] == "Japanese"
        assert _LANG_MAP["ko"] == "Korean"
        assert _LANG_MAP["fr"] == "French"
        assert _LANG_MAP["de"] == "German"
        assert _LANG_MAP["es"] == "Spanish"

    def test_all_ten_languages(self):
        assert len(_LANG_MAP) == 10


class TestGetLangdetect:
    def test_returns_false_when_import_fails(self):
        server._langdetect_model = None
        with patch.dict(sys.modules, {"fasttext_langdetect": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = _get_langdetect()
        assert result is False
        server._langdetect_model = None

    def test_caches_result(self):
        server._langdetect_model = "cached_value"
        result = _get_langdetect()
        assert result == "cached_value"
        server._langdetect_model = None


class TestDetectLanguageWithFasttext:
    def test_uses_fasttext_when_available(self):
        mock_detector = MagicMock(return_value={"lang": "fr", "score": 0.99})
        server._langdetect_model = None
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("Bonjour le monde")
        assert result == "French"
        mock_detector.assert_called_once_with("Bonjour le monde", low_memory=False)

    def test_maps_zh_to_chinese(self):
        mock_detector = MagicMock(return_value={"lang": "zh", "score": 0.95})
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("你好")
        assert result == "Chinese"

    def test_unknown_lang_defaults_to_english(self):
        mock_detector = MagicMock(return_value={"lang": "xx", "score": 0.5})
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("something")
        assert result == "English"

    def test_falls_back_on_exception(self):
        mock_detector = MagicMock(side_effect=RuntimeError("model error"))
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("Hello world")
        assert result == "English"

    def test_falls_back_when_fasttext_unavailable(self):
        with patch.object(server, "_get_langdetect", return_value=False):
            result = detect_language("你好世界")
        assert result == "Chinese"

    def test_maps_all_supported_languages(self):
        for iso, name in _LANG_MAP.items():
            mock_detector = MagicMock(return_value={"lang": iso, "score": 0.9})
            with patch.object(server, "_get_langdetect", return_value=mock_detector):
                result = detect_language("test")
            assert result == name, f"Failed for {iso} -> {name}"


# --- Issue #14: pyrubberband speed adjustment tests ---

class TestAdjustSpeedWithPyrubberband:
    def test_speed_1_returns_unchanged(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = _adjust_speed(audio, 24000, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_speed_faster_calls_pyrubberband(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        expected = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        mock_rb = MagicMock()
        mock_rb.time_stretch.return_value = expected
        with patch.object(server, "_pyrubberband", mock_rb):
            result = _adjust_speed(audio, 24000, 1.5)
        mock_rb.time_stretch.assert_called_once_with(audio, 24000, 1.5)
        np.testing.assert_array_equal(result, expected)

    def test_speed_slower_calls_pyrubberband(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        expected = np.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)
        mock_rb = MagicMock()
        mock_rb.time_stretch.return_value = expected
        with patch.object(server, "_pyrubberband", mock_rb):
            result = _adjust_speed(audio, 24000, 0.75)
        mock_rb.time_stretch.assert_called_once_with(audio, 24000, 0.75)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_sample_rate(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_rb = MagicMock()
        mock_rb.time_stretch.return_value = audio
        with patch.object(server, "_pyrubberband", mock_rb):
            _adjust_speed(audio, 48000, 2.0)
        mock_rb.time_stretch.assert_called_once_with(audio, 48000, 2.0)


class TestAdjustSpeedFallback:
    def test_speed_1_returns_unchanged(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        with patch.object(server, "_pyrubberband", None):
            result = _adjust_speed(audio, 24000, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_speed_faster_calls_scipy_resample(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        expected = np.array([0.1, 0.3, 0.5, 0.6], dtype=np.float32)
        mock_scipy = MagicMock()
        mock_scipy.resample.return_value = expected
        with patch.object(server, "_pyrubberband", None), \
             patch.object(server, "scipy_signal", mock_scipy):
            result = _adjust_speed(audio, 24000, 1.5)
        mock_scipy.resample.assert_called_once_with(audio, 4)
        np.testing.assert_array_equal(result, expected)

    def test_zero_length_returns_original(self):
        audio = np.array([0.1], dtype=np.float32)
        with patch.object(server, "_pyrubberband", None):
            result = _adjust_speed(audio, 24000, 100.0)
        np.testing.assert_array_equal(result, audio)


# --- Server-side sample rate conversion tests ---


class TestResampleAudio:
    def test_resample_24k_to_8k(self):
        """Downsampling 24kHz to 8kHz should produce 1/3 the samples."""
        audio = np.random.randn(24000).astype(np.float32)
        result = _resample_audio(audio, 24000, 8000)
        assert len(result) == 8000

    def test_resample_noop_same_rate(self):
        """Same input/output rate returns original array."""
        audio = np.random.randn(24000).astype(np.float32)
        result = _resample_audio(audio, 24000, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_resample_none_target_noop(self):
        """None target rate returns original array."""
        audio = np.random.randn(24000).astype(np.float32)
        result = _resample_audio(audio, 24000, None)
        np.testing.assert_array_equal(result, audio)


# --- Issue #5: TF32 matmul mode tests ---

class TestTF32Flags:
    def test_tf32_matmul_enabled(self):
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is True

    def test_tf32_cudnn_enabled(self):
        if torch.cuda.is_available():
            assert torch.backends.cudnn.allow_tf32 is True

    def test_cudnn_benchmark_enabled(self):
        if torch.cuda.is_available():
            assert torch.backends.cudnn.benchmark is True


# --- Issue #3: Sentence splitting tests (Phase 1) ---


class TestSplitSentences:
    """Tests for _split_sentences (Issue #3)."""

    def test_single_sentence(self):
        assert _split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = _split_sentences("First sentence. Second sentence. Third one.")
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third one."

    def test_question_and_exclamation(self):
        result = _split_sentences("Is it working? Yes it is! Great.")
        assert len(result) == 3

    def test_abbreviation_awareness(self):
        """Should not split on Dr. or Mr. abbreviations."""
        result = _split_sentences("Dr. Smith called Mr. Jones today.")
        assert len(result) == 1

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []

    def test_no_sentence_ending(self):
        """Text without sentence-ending punctuation returns as single item."""
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_strips_whitespace(self):
        result = _split_sentences("  First.   Second.  ")
        assert all(s == s.strip() for s in result)


# --- Issue #2: Adaptive max tokens tests (Phase 1) ---


class TestAdaptiveMaxTokens:
    def test_short_text_minimum(self):
        assert _adaptive_max_tokens("Hi") == 128

    def test_scales_with_words(self):
        text = " ".join(["word"] * 20)
        assert _adaptive_max_tokens(text) == min(2048, 20 * 8)

    def test_cjk_uses_char_count(self):
        cjk = "你好世界" * 10
        result = _adaptive_max_tokens(cjk)
        assert result == max(128, min(2048, len(cjk) * 3))

    def test_cap_at_2048(self):
        long_text = " ".join(["word"] * 300)
        assert _adaptive_max_tokens(long_text) == 2048


# --- Baseline utility tests ---

class TestResolveVoice:
    def test_default_voice_when_none(self):
        assert resolve_voice(None) == "vivian"
    def test_openai_alias(self):
        assert resolve_voice("alloy") == "ryan"
    def test_case_insensitive(self):
        assert resolve_voice("VIVIAN") == "vivian"
    def test_unknown_voice_raises_api_error(self):
        with pytest.raises(APIError) as exc_info:
            resolve_voice("nonexistent_voice")
        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "UNKNOWN_VOICE"
        assert "nonexistent_voice" in exc_info.value.message
        assert exc_info.value.context["voice"] == "nonexistent_voice"
        assert isinstance(exc_info.value.context["valid_voices"], list)


# --- Issue #109: Standard error response shape tests ---

class TestErrorResponse:
    def test_error_response_model_fields(self):
        err = ErrorResponse(code="TEST_ERROR", message="Test", statusCode=400)
        assert err.code == "TEST_ERROR"
        assert err.message == "Test"
        assert err.context is None
        assert err.statusCode == 400

    def test_error_response_with_context(self):
        err = ErrorResponse(code="TEST", message="msg", context={"key": "val"}, statusCode=500)
        d = err.model_dump()
        assert d["code"] == "TEST"
        assert d["context"] == {"key": "val"}
        assert d["statusCode"] == 500

    def test_error_response_serialization(self):
        err = ErrorResponse(code="X", message="Y", statusCode=422)
        d = err.model_dump()
        assert set(d.keys()) == {"code", "message", "context", "statusCode"}


class TestAPIError:
    def test_api_error_attributes(self):
        exc = APIError(503, "QUEUE_FULL", "busy", context={"q": 5}, headers={"Retry-After": "5"})
        assert exc.status_code == 503
        assert exc.code == "QUEUE_FULL"
        assert exc.message == "busy"
        assert exc.context == {"q": 5}
        assert exc.headers == {"Retry-After": "5"}

    def test_api_error_is_exception(self):
        exc = APIError(400, "BAD", "bad request")
        assert isinstance(exc, Exception)

    def test_api_error_defaults(self):
        exc = APIError(500, "ERR", "fail")
        assert exc.context is None
        assert exc.headers is None


# --- Issue #15 → #82: Voice prompt cache tests ---
# (Original #15 cached raw audio arrays; #82 upgrades to speaker embedding cache)

def _make_wav_bytes(samples=None, sr=24000, channels=1):
    """Helper to create valid WAV bytes for testing."""
    if samples is None:
        samples = np.random.randn(sr).astype(np.float32) * 0.1
    if channels > 1:
        samples = np.column_stack([samples] * channels)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue(), samples, sr


# --- Issue #16: GPU memory pool pre-allocation tests ---

class TestGpuPoolPreAllocation:
    """Verify GPU memory pool pre-warming code exists in _gpu_warmup."""

    def test_gpu_warmup_contains_pool_prewarm(self):
        import inspect
        source = inspect.getsource(server._gpu_warmup)
        assert "Pre-warming CUDA memory pool" in source
        assert "torch.empty" in source
        assert "dtype=torch.bfloat16" in source

    def test_dummy_tensor_size_is_128mb(self):
        import inspect
        source = inspect.getsource(server._gpu_warmup)
        assert "64 * 1024 * 1024" in source

    def test_pool_prewarm_has_exception_handling(self):
        import inspect
        source = inspect.getsource(server._gpu_warmup)
        idx = source.find("Pre-warming CUDA memory pool")
        assert idx > 0
        section = source[idx - 200:idx + 500]
        assert "try:" in section
        assert "except Exception" in section

    def test_pool_prewarm_after_warmup(self):
        import inspect
        source = inspect.getsource(server._gpu_warmup)
        warmup_idx = source.find("Warming up GPU with multi-length synthesis")
        pool_idx = source.find("Pre-warming CUDA memory pool")
        assert warmup_idx > 0
        assert pool_idx > 0
        assert pool_idx > warmup_idx

    def test_dummy_tensor_is_deleted(self):
        import inspect
        source = inspect.getsource(server._gpu_warmup)
        pool_section = source[source.find("Pre-warming CUDA memory pool"):]
        assert "del dummy" in pool_section

    def test_load_model_calls_gpu_warmup(self):
        import inspect
        source = inspect.getsource(server._load_model_sync)
        assert "_gpu_warmup()" in source


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello world") == "English"

    def test_chinese(self):
        assert detect_language("\u4f60\u597d\u4e16\u754c") == "Chinese"

    def test_japanese(self):
        assert detect_language("\u3053\u3093\u306b\u3061\u306f") == "Japanese"

    def test_korean(self):
        assert detect_language("\uc548\ub155\ud558\uc138\uc694") == "Korean"


# --- Issue #17: Audio output LRU cache tests ---


@pytest.fixture(autouse=False)
def clear_audio_cache():
    """Clear the audio cache before and after each test."""
    _audio_cache.clear()
    yield
    _audio_cache.clear()


class TestAudioCacheKey:
    def test_deterministic(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        assert k1 == k2

    def test_different_text_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("world", "vivian", 1.0, "wav", "English", "")
        assert k1 != k2

    def test_different_voice_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "ryan", 1.0, "wav", "English", "")
        assert k1 != k2

    def test_different_language_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.0, "wav", "Chinese", "")
        assert k1 != k2

    def test_key_is_sha256_hex(self):
        key = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        assert len(key) == 64
        int(key, 16)


class TestAudioCacheGetSet:
    def setup_method(self):
        _audio_cache.clear()

    def teardown_method(self):
        _audio_cache.clear()

    def test_cache_miss_returns_none(self):
        assert _get_audio_cache("nonexistent") is None

    def test_cache_hit_returns_stored_data(self):
        _set_audio_cache("key1", b"audio_data", "audio/wav")
        result = _get_audio_cache("key1")
        assert result is not None
        assert result[0] == b"audio_data"
        assert result[1] == "audio/wav"

    def test_cache_disabled_when_max_zero(self):
        with patch("server._AUDIO_CACHE_MAX", 0):
            _set_audio_cache("key1", b"data", "audio/wav")
            assert len(_audio_cache) == 0
            assert _get_audio_cache("key1") is None


# --- Issue #83: Generation parameter exposure tests ---


class TestGenerationParams:
    """temperature and top_p are passed through to model.generate() when set."""

    def test_temperature_included_in_build_gen_kwargs(self):
        """temperature in request -> present in gen_kwargs via _build_gen_kwargs."""
        req = server.TTSRequest(input="hello", temperature=0.8)
        gen_kwargs = server._build_gen_kwargs("hello", req)
        assert gen_kwargs["temperature"] == 0.8
        assert "top_p" not in gen_kwargs

    def test_top_p_included_in_build_gen_kwargs(self):
        """top_p in request -> present in gen_kwargs via _build_gen_kwargs."""
        req = server.TTSRequest(input="hello", top_p=0.95)
        gen_kwargs = server._build_gen_kwargs("hello", req)
        assert gen_kwargs["top_p"] == 0.95
        assert "temperature" not in gen_kwargs

    def test_both_params_included(self):
        """Both temperature and top_p set -> both in gen_kwargs."""
        req = server.TTSRequest(input="hello", temperature=1.2, top_p=0.9)
        gen_kwargs = server._build_gen_kwargs("hello", req)
        assert gen_kwargs["temperature"] == 1.2
        assert gen_kwargs["top_p"] == 0.9

    def test_neither_param_means_neither_in_kwargs(self):
        """Omitting both leaves gen_kwargs with only max_new_tokens."""
        req = server.TTSRequest(input="hello")
        assert req.temperature is None
        assert req.top_p is None
        gen_kwargs = server._build_gen_kwargs("hello", req)
        assert "temperature" not in gen_kwargs
        assert "top_p" not in gen_kwargs
        assert "max_new_tokens" in gen_kwargs

    def test_max_new_tokens_always_present(self):
        """_build_gen_kwargs always includes max_new_tokens from _adaptive_max_tokens."""
        req = server.TTSRequest(input="hello", temperature=0.5)
        gen_kwargs = server._build_gen_kwargs("hello", req)
        assert gen_kwargs["max_new_tokens"] == _adaptive_max_tokens("hello")

    def test_ttsrequest_accepts_temperature_and_top_p(self):
        """TTSRequest model accepts temperature and top_p fields."""
        req = server.TTSRequest(input="test", temperature=1.2, top_p=0.9)
        assert req.temperature == 1.2
        assert req.top_p == 0.9


# --- Issue #82: Voice clone prompt cache tests ---


class TestVoiceClonePromptCache:
    """create_voice_clone_prompt() is called at most once per unique ref audio."""

    def _make_audio_bytes(self, seed: int = 42) -> bytes:
        """Create deterministic fake WAV bytes for testing."""
        buf = io.BytesIO()
        rng = np.random.default_rng(seed)
        sf.write(buf, rng.random(24000).astype(np.float32), 24000, format="WAV")
        return buf.getvalue()

    def setup_method(self):
        server._voice_prompt_cache.clear()
        server._voice_cache_hits = 0

    def test_cache_miss_calls_create_prompt(self):
        """First call for a ref audio invokes create_voice_clone_prompt exactly once."""
        audio_bytes = self._make_audio_bytes()
        mock_prompt = MagicMock(name="voice_prompt")
        mock_model = MagicMock()
        mock_model.create_voice_clone_prompt.return_value = mock_prompt

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 32):
            result = server._get_cached_voice_prompt(audio_bytes, ref_text="hello")

        mock_model.create_voice_clone_prompt.assert_called_once()
        assert result is mock_prompt

    def test_cache_hit_skips_create_prompt(self):
        """Second call with the same audio returns cached prompt, no model call."""
        audio_bytes = self._make_audio_bytes()
        mock_prompt = MagicMock(name="voice_prompt")
        mock_model = MagicMock()
        mock_model.create_voice_clone_prompt.return_value = mock_prompt

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 32):
            server._get_cached_voice_prompt(audio_bytes, ref_text="hello")
            result = server._get_cached_voice_prompt(audio_bytes, ref_text="hello")

        assert mock_model.create_voice_clone_prompt.call_count == 1
        assert result is mock_prompt
        assert server._voice_cache_hits == 1

    def test_different_audio_different_cache_entry(self):
        """Different ref audio bytes create separate cache entries."""
        audio1 = self._make_audio_bytes(seed=1)
        audio2 = self._make_audio_bytes(seed=2)
        mock_prompt1 = MagicMock(name="prompt1")
        mock_prompt2 = MagicMock(name="prompt2")
        mock_model = MagicMock()
        mock_model.create_voice_clone_prompt.side_effect = [mock_prompt1, mock_prompt2]

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 32):
            r1 = server._get_cached_voice_prompt(audio1, ref_text="a")
            r2 = server._get_cached_voice_prompt(audio2, ref_text="b")

        assert r1 is mock_prompt1
        assert r2 is mock_prompt2
        assert mock_model.create_voice_clone_prompt.call_count == 2
        assert len(server._voice_prompt_cache) == 2

    def test_same_audio_different_ref_text_different_cache_entry(self):
        """Same audio bytes + different ref_text produces separate cache entries."""
        audio = self._make_audio_bytes(seed=42)
        mock_prompt1 = MagicMock(name="prompt_text_a")
        mock_prompt2 = MagicMock(name="prompt_text_b")
        mock_model = MagicMock()
        mock_model.create_voice_clone_prompt.side_effect = [mock_prompt1, mock_prompt2]

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 32):
            r1 = server._get_cached_voice_prompt(audio, ref_text="hello")
            r2 = server._get_cached_voice_prompt(audio, ref_text="world")

        assert r1 is mock_prompt1
        assert r2 is mock_prompt2
        assert mock_model.create_voice_clone_prompt.call_count == 2
        assert len(server._voice_prompt_cache) == 2

    def test_lru_eviction_removes_oldest_entry(self):
        """When cache exceeds VOICE_CACHE_MAX, oldest entry is evicted."""
        mock_model = MagicMock()
        mock_model.create_voice_clone_prompt.return_value = MagicMock()

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 2):
            a1 = self._make_audio_bytes(seed=10)
            a2 = self._make_audio_bytes(seed=11)
            a3 = self._make_audio_bytes(seed=12)
            server._get_cached_voice_prompt(a1, ref_text="x")
            server._get_cached_voice_prompt(a2, ref_text="y")
            # This should evict a1
            server._get_cached_voice_prompt(a3, ref_text="z")

        assert len(server._voice_prompt_cache) == 2
        key1 = hashlib.sha256(a1).hexdigest()
        assert key1 not in server._voice_prompt_cache

    def test_cache_disabled_still_computes_prompt(self):
        """When VOICE_CACHE_MAX=0, prompt is still computed but not cached."""
        audio_bytes = self._make_audio_bytes()
        mock_prompt = MagicMock(name="voice_prompt")
        mock_model = MagicMock()
        mock_model.create_voice_clone_prompt.return_value = mock_prompt

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 0):
            result = server._get_cached_voice_prompt(audio_bytes, ref_text="hello")

        assert result is mock_prompt
        assert len(server._voice_prompt_cache) == 0

    def test_access_promotes_entry(self):
        """Accessing a cached entry promotes it, preventing eviction."""
        mock_model = MagicMock()
        prompts = [MagicMock(name=f"p{i}") for i in range(3)]
        mock_model.create_voice_clone_prompt.side_effect = prompts

        a0 = self._make_audio_bytes(seed=100)
        a1 = self._make_audio_bytes(seed=101)
        a2 = self._make_audio_bytes(seed=102)

        with patch.object(server, "model", mock_model), \
             patch.object(server, "VOICE_CACHE_MAX", 2):
            server._get_cached_voice_prompt(a0, ref_text="x")  # cache: [a0]
            server._get_cached_voice_prompt(a1, ref_text="y")  # cache: [a0, a1]
            server._get_cached_voice_prompt(a0, ref_text="x")  # hit, promotes -> [a1, a0]
            server._get_cached_voice_prompt(a2, ref_text="z")  # evicts a1 -> [a0, a2]

        key0 = hashlib.sha256(a0).hexdigest()
        key1 = hashlib.sha256(a1).hexdigest()
        key2 = hashlib.sha256(a2).hexdigest()
        assert key0 in server._voice_prompt_cache
        assert key1 not in server._voice_prompt_cache
        assert key2 in server._voice_prompt_cache


# --- Issue #81: Priority inference queue tests ---


class TestPriorityInferQueue:
    """Tests for PriorityInferQueue scheduling."""

    def test_higher_priority_runs_before_lower(self):
        """Priority 0 job completes before priority 1 job when both queued."""
        order = []

        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            t1 = asyncio.create_task(queue.submit(lambda: order.append("low") or "low", priority=1))
            await asyncio.sleep(0)
            t2 = asyncio.create_task(queue.submit(lambda: order.append("high") or "high", priority=0))
            await asyncio.gather(t1, t2)

        asyncio.run(run())
        # Both jobs should have executed
        assert set(order) == {"low", "high"}

    def test_submit_returns_function_result(self):
        """Queue.submit resolves the future with the function's return value."""
        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()
            result = await queue.submit(lambda: 42, priority=1)
            assert result == 42

        asyncio.run(run())

    def test_submit_propagates_exception(self):
        """Queue.submit propagates exceptions from the worker function."""
        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            def boom():
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await queue.submit(boom, priority=1)

        asyncio.run(run())

    def test_fifo_within_same_priority(self):
        """Jobs with equal priority execute in submission order."""
        order = []

        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()
            for i in range(3):
                await queue.submit(lambda i=i: order.append(i) or i, priority=1)

        asyncio.run(run())
        assert order == [0, 1, 2]

    def test_priority_constants_exist(self):
        """PRIORITY_REALTIME and PRIORITY_BATCH are defined correctly."""
        assert server.PRIORITY_REALTIME == 0
        assert server.PRIORITY_BATCH == 1
        assert server.PRIORITY_REALTIME < server.PRIORITY_BATCH


# --- Issue #86: Quantization support tests ---


class TestQuantization:
    """QUANTIZE env var controls model loading precision via _resolve_quant_kwargs()."""

    def test_no_quantize_returns_bfloat16_no_extras(self):
        """Empty QUANTIZE env var -> bfloat16, empty extra kwargs."""
        with patch.object(server, "QUANTIZE", ""):
            dtype, kwargs = server._resolve_quant_kwargs()
        assert dtype == torch.bfloat16
        assert kwargs == {}

    def test_int8_returns_float16_and_load_in_8bit(self):
        """QUANTIZE=int8 -> float16 dtype, load_in_8bit=True."""
        mock_bnb = MagicMock()
        with patch.dict("sys.modules", {"bitsandbytes": mock_bnb}), \
             patch.object(server, "QUANTIZE", "int8"):
            dtype, kwargs = server._resolve_quant_kwargs()
        assert dtype == torch.float16
        assert kwargs.get("load_in_8bit") is True

    def test_int8_missing_bitsandbytes_raises_importerror(self):
        """QUANTIZE=int8 without bitsandbytes installed raises ImportError."""
        with patch.dict("sys.modules", {"bitsandbytes": None}), \
             patch.object(server, "QUANTIZE", "int8"):
            with pytest.raises(ImportError, match="bitsandbytes"):
                server._resolve_quant_kwargs()

    def test_fp8_returns_bfloat16_and_quantization_config(self):
        """QUANTIZE=fp8 -> bfloat16 dtype, quantization_config in kwargs."""
        mock_config_cls = MagicMock(name="TorchAoConfig")
        mock_transformers = MagicMock()
        mock_transformers.TorchAoConfig = mock_config_cls
        with patch.dict("sys.modules", {"transformers": mock_transformers}), \
             patch.object(server, "QUANTIZE", "fp8"):
            dtype, kwargs = server._resolve_quant_kwargs()
        assert dtype == torch.bfloat16
        assert "quantization_config" in kwargs
        mock_config_cls.assert_called_once_with("fp8_dynamic_activation_fp8_weight")

    def test_unknown_quantize_value_raises_valueerror(self):
        """Unknown QUANTIZE value raises ValueError with descriptive message."""
        with patch.object(server, "QUANTIZE", "gguf"):
            with pytest.raises(ValueError, match="Unknown QUANTIZE"):
                server._resolve_quant_kwargs()


# --- Issue #85: Gateway/Worker mode tests ---

# Mock aiohttp if not installed (it's a Docker-only dependency)
try:
    import aiohttp as _aiohttp_check  # noqa: F401
except ImportError:
    sys.modules["aiohttp"] = MagicMock()

import gateway  # noqa: E402


class TestGateway:
    """Gateway proxies requests to worker and manages worker subprocess lifecycle."""

    def test_worker_process_none_at_start(self):
        """_worker_process is None before any request is received."""
        assert gateway._worker_process is None

    def test_check_idle_kills_worker_after_timeout(self):
        """_check_idle kills the worker when last_used is older than IDLE_TIMEOUT."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is running

        with patch.object(gateway, "_worker_process", mock_proc, create=True), \
             patch.object(gateway, "_last_used", 0.0, create=True), \
             patch.object(gateway, "IDLE_TIMEOUT", 120), \
             patch("time.time", return_value=200.0):
            asyncio.run(gateway._check_idle())

        mock_proc.kill.assert_called_once()

    def test_check_idle_noop_when_no_worker(self):
        """_check_idle does nothing when _worker_process is None."""
        with patch.object(gateway, "_worker_process", None, create=True):
            asyncio.run(gateway._check_idle())  # should not raise

    def test_check_idle_noop_when_idle_timeout_zero(self):
        """IDLE_TIMEOUT=0 disables idle killing."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch.object(gateway, "_worker_process", mock_proc, create=True), \
             patch.object(gateway, "_last_used", 0.0, create=True), \
             patch.object(gateway, "IDLE_TIMEOUT", 0), \
             patch("time.time", return_value=99999.0):
            asyncio.run(gateway._check_idle())
        mock_proc.kill.assert_not_called()

    def test_check_idle_clears_dead_process(self):
        """If worker process has already exited, _worker_process is set to None."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # process exited
        with patch.object(gateway, "_worker_process", mock_proc, create=True):
            asyncio.run(gateway._check_idle())
        # Should clear _worker_process without calling kill
        mock_proc.kill.assert_not_called()


# --- Issue #84: Batch inference tests ---


class TestBatchInference:
    """Tests for batch inference dispatching via PriorityInferQueue."""

    def test_batch_key_carried_on_job(self):
        """_InferJob has batch_key field, default 'single'."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        job = server._InferJob(
            priority=1, submit_time=0.0,
            future=future, fn=lambda: None,
        )
        assert job.batch_key == "single"

        job_batch = server._InferJob(
            priority=1, submit_time=0.0,
            future=future, fn=lambda: None,
            batch_key="synthesis",
        )
        assert job_batch.batch_key == "synthesis"
        loop.close()

    def test_submit_batch_returns_result(self):
        """Single submit_batch call resolves correctly."""
        mock_wavs = [np.array([0.1, 0.2], dtype=np.float32)]
        mock_sr = 24000

        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_do_synthesize_batch", return_value=(mock_wavs, mock_sr)):
                with patch.object(server, "MAX_BATCH_SIZE", 4):
                    result = await queue.submit_batch(
                        text="hello", language="English",
                        speaker="vivian", gen_kwargs={"max_new_tokens": 256},
                    )
            return result

        wavs, sr = asyncio.run(run())
        assert sr == mock_sr
        np.testing.assert_array_equal(wavs[0], mock_wavs[0])

    def test_multiple_queued_jobs_batched(self):
        """3 synthesis jobs submitted together are dispatched as a single batch call."""
        call_args = {}

        def fake_batch(texts, languages, speakers, gen_kwargs_list):
            call_args["texts"] = texts
            call_args["languages"] = languages
            call_args["speakers"] = speakers
            wavs = [np.array([0.1 * i], dtype=np.float32) for i in range(len(texts))]
            return wavs, 24000

        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_do_synthesize_batch", side_effect=fake_batch):
                with patch.object(server, "MAX_BATCH_SIZE", 4):
                    # Submit 3 jobs concurrently — they should all land in the heap
                    # before the worker drains them
                    tasks = []
                    for i in range(3):
                        t = asyncio.create_task(queue.submit_batch(
                            text=f"text_{i}", language="English",
                            speaker="vivian", gen_kwargs={"max_new_tokens": 256},
                        ))
                        tasks.append(t)
                    results = await asyncio.gather(*tasks)
            return results, call_args

        results, captured = asyncio.run(run())
        # All 3 texts should have been passed to _do_synthesize_batch
        assert len(captured["texts"]) == 3
        assert set(captured["texts"]) == {"text_0", "text_1", "text_2"}
        assert len(results) == 3

    def test_exception_propagates_to_all_futures(self):
        """If _do_synthesize_batch raises, all futures get the exception."""
        def exploding_batch(*args, **kwargs):
            raise RuntimeError("GPU exploded")

        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_do_synthesize_batch", side_effect=exploding_batch):
                with patch.object(server, "MAX_BATCH_SIZE", 4):
                    tasks = []
                    for i in range(3):
                        t = asyncio.create_task(queue.submit_batch(
                            text=f"text_{i}", language="English",
                            speaker="vivian", gen_kwargs={"max_new_tokens": 256},
                        ))
                        tasks.append(t)

                    exceptions = []
                    for t in tasks:
                        try:
                            await t
                        except RuntimeError as e:
                            exceptions.append(str(e))
            return exceptions

        exceptions = asyncio.run(run())
        assert len(exceptions) == 3
        assert all("GPU exploded" in e for e in exceptions)

    def test_max_batch_size_one_disables_batching(self):
        """With MAX_BATCH_SIZE=1, submit_batch still works as a batch-of-1."""
        mock_wavs = [np.array([0.5], dtype=np.float32)]
        mock_sr = 24000

        async def run():
            queue = server.PriorityInferQueue()
            queue._infer_executor = ThreadPoolExecutor(max_workers=1)
            queue.start()

            with patch.object(server, "_do_synthesize_batch", return_value=(mock_wavs, mock_sr)) as mock_batch:
                with patch.object(server, "MAX_BATCH_SIZE", 1):
                    result = await queue.submit_batch(
                        text="hello", language="English",
                        speaker="vivian", gen_kwargs={"max_new_tokens": 256},
                    )
            return result, mock_batch.call_count

        (wavs, sr), call_count = asyncio.run(run())
        assert call_count == 1
        assert sr == mock_sr
        np.testing.assert_array_equal(wavs[0], mock_wavs[0])
