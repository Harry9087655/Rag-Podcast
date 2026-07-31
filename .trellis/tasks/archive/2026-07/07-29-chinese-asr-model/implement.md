# Implementation Plan — Chinese ASR Routing

## Ordered Checklist

1. **Dependency**: add `funasr` to `transcription` optional-dependency group in `backend/pyproject.toml`; run `uv lock` (updates `backend/uv.lock`).
2. **Config**: add `funasr_model`, `funasr_vad_model`, `funasr_punc_model`, `funasr_device`, `lang_detect_window_seconds` to `Settings` in `backend/src/rag_podcast/config.py` (defaults per `design.md`).
3. **`transcriber.py` additions** (`backend/src/rag_podcast/transcription/transcriber.py`):
   - `random_window(audio, sample_rate, window_seconds, rng)` helper.
   - `detect_language(whisperx_model, clip, batch_size=8)` helper.
   - `_funasr_to_segments(result)` adapter (FunASR `sentence_info` → WhisperX-shaped `segments`).
   - `LocalFunASR` class (eager load in `__init__`, `async transcribe()` via `asyncio.to_thread`, `TranscribeError` on failure — mirror `LocalWhisperX`'s try/except shape).
   - `LanguageRoutingTranscriber` class (holds both transcribers, does load → detect → dispatch, wraps detection failures in `TranscribeError`).
   - Extend `create_transcriber(...)` to build and return a `LanguageRoutingTranscriber` from both sets of settings.
4. **`cli.py`**: update `_main()` to pass the new FunASR settings (and `lang_detect_window_seconds`) into `create_transcriber(...)`.
5. **Tests** (`backend/tests/`, new `test_transcriber.py` or similar — follow `test_cleaning.py`/`test_chunker.py`'s plain-pytest, no-fixtures style):
   - `_funasr_to_segments`: given a `sentence_info`-shaped input, asserts `segments[].text/start/end` (ms→s converted) and `words[]` are one-per-character with correct `start`/`end`.
   - `random_window`: audio shorter than the window returns the whole array unchanged; audio longer than the window returns a `window_seconds`-length slice starting within bounds (seed the `rng` for determinism).
   - `detect_language`: fake model object whose `.transcribe()` returns scripted `{"language": ...}` values across calls; assert (a) short-circuits and returns `"zh"` as soon as any window says `"zh"` (including when only the *second* window does, not just the first), (b) returns the last window's language when neither says `"zh"`, (c) only calls the model up to `num_windows` times.
   - `LanguageRoutingTranscriber.transcribe`: use fake `Transcriber`-shaped stand-ins for `_whisperx`/`_funasr` (simple async stubs recording calls) plus a monkeypatched `detect_language` to force `"zh"` and `"en"` paths; assert the correct delegate was called and its result returned, without loading real models.
   - No test should import/load real `whisperx`/`funasr` models (keep tests fast and independent of GPU/model downloads) — mirror how existing tests avoid touching real WhisperX.
6. **Manual smoke test** (optional, not part of automated suite): adapt `backend/scripts/test_whisperx.py`'s pattern — point at a known-Chinese episode id, run the full `LanguageRoutingTranscriber`, eyeball the resulting `segments`/`words` shape and that `language == "zh"`.

## Validation Commands

```bash
cd backend
uv run --extra transcription pytest tests/ -k "transcriber or cleaning or chunker"
uv run --extra transcription pytest tests/          # full backend suite
```

## Risky Files / Rollback Points

- `transcriber.py`, `config.py`, `cli.py`, `pyproject.toml` are the only touched files — no DB migration, no `worker.py` change. Revert these four files to roll back completely.
- Confirm `worker.py` and `models/episode.py` remain byte-for-byte unchanged as a design invariant check before marking this done.

## Follow-up Checks Before `task.py start`

- [ ] `prd.md` acceptance criteria map 1:1 onto steps above.
- [ ] No DB migration needed (confirmed in design.md) — skip Alembic step entirely.
- [ ] Confirm `funasr` installs cleanly alongside existing `torch`/`ctranslate2`/`faster-whisper` pins (version conflicts are the main integration risk — check during step 1).
- [ ] `transcription/__init__.py`'s existing `try/except ImportError` (wrapping the whole `from .transcriber import ...` line) already covers a new `import funasr` failure in `transcriber.py` — no code change needed there, just confirm during review (see `.trellis/spec/backend/quality-guidelines.md`).
