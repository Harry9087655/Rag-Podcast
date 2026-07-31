# Add Chinese ASR model (Qwen3-ASR / FunASR)

## Goal

WhisperX transcription quality is good for English but poor for Chinese podcasts. Add a second local ASR backend specialized for Chinese, without breaking the existing WhisperX path for English podcasts.

## Confirmed Facts (from codebase)

- `transcription/transcriber.py` already defines a `Transcriber` Protocol (`async def transcribe(audio_path) -> dict`) plus a `create_transcriber()` factory and a `LocalWhisperX` implementation. The archived design doc (`.trellis/tasks/archive/2026-07/07-22-whisperx-transcription/design.md:248`) explicitly names "Different ASR model" as a designed-for extension point: "Implement `Transcriber` protocol ... Worker loop unchanged."
- `worker.py`'s `transcribe_episode()` calls `transcriber.transcribe(audio_path)` and stores the raw dict on `episode.transcript_data` (JSONB) plus `episode.language = result.get("language")`. The worker is constructed with one `Transcriber` instance for its whole run (`run_transcription_worker.py` → `create_transcriber(...)` from `Settings`).
- Downstream code already has script-aware (CJK vs. Latin) handling keyed on `episode.language`, and expects a specific output shape:
  - `episode.transcript_data`: `{"language": ..., "segments": [{"text", "start", "end", "words": [{"word","start","end", "score"?}, ...]}, ...]}`.
  - `cleaning/filler_words.py`: `clean_words()` already branches on `language == "zh"` (own `ZH_FILLER_WORDS` set, no-separator join) vs. everything else (English filler list, space join).
  - `cleaning/text_join.py`: `join_words()` already treats `zh`/`ja` as no-space-between-words languages.
  - `indexing/chunker.py`: builds `ChunkSpan`s from segment-level `text`/`start`/`end` (verbatim, never rebuilt from words) plus a `words` list for the embedder; the oversized-segment fallback path rebuilds text via `join_words(buffer, language)` and requires per-word `start`/`end`.
  - **Constraint**: any new transcriber must return `segments[].words[]` with numeric `start`/`end` per word/character — the chunker's oversized-segment split and the embedder's `clean_words`/`join_words` both depend on it.
- `Episode.language` is only known *after* transcription runs (whisperx auto-detects it) — there is no pre-existing per-podcast or per-episode language field to route on. `Podcast` (`models/podcast.py`) has no language column.
- `pyproject.toml`'s `[project.optional-dependencies] transcription` group currently pins `whisperx, torch, torchaudio, ctranslate2, faster-whisper`.

## Research: Qwen3-ASR vs. FunASR fit

- **FunASR** (`modelscope/FunASR`, `funasr` PyPI package): `AutoModel(model="paraformer-zh", vad_model="fsmn-vad")` → single `.generate(input=audio)` call returns text + native **character-level timestamps** (`res[0]["timestamp"]`) in one pass. Mature, single dependency, VAD+punctuation restoration available in the same pipeline. Output shape maps directly onto the existing `segments[].words[]` contract with minimal glue code. [FunASR timestamps blog](https://www.funasr.com/en/blog/speech-to-text-timestamps-python.html), [FunASR GitHub](https://github.com/modelscope/FunASR)
- **Qwen3-ASR** (`QwenLM/Qwen3-ASR`, 0.6B/1.7B): the ASR model itself returns text + language ID but **not** word timestamps in the same call — timestamps require a separate companion model, `Qwen3-ForcedAligner-0.6B`, run as a second pass, and that aligner is capped at **5 minutes of audio per call** and 11 languages. Podcast episodes (30–90+ min) would need to be chunked into ≤5-minute windows for alignment and the results stitched back together — meaningfully more integration work than FunASR for equivalent output shape. [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR), [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)
- **Recommendation**: FunASR (Paraformer) — single-pass native character timestamps matching this project's existing contract; Qwen3-ASR's two-model/chunked-alignment pipeline is possible but adds real complexity (chunk-boundary stitching, an extra model, a 5-minute alignment ceiling) for a Chinese-quality gain that isn't yet demonstrated to be necessary.

## Decisions

- **Model**: FunASR (`paraformer-zh` + `fsmn-vad` + `ct-punc`), not Qwen3-ASR — single-pass native timestamps vs. Qwen3-ASR's two-model/≤5-min-chunked alignment pipeline (see Research above).
- **Routing**: fully automatic per-episode language detection, no manual per-podcast tagging. After download, before full transcription: extract **up to 2 independent random 30-second windows** from the episode audio, run WhisperX on each to detect language, then:
  - either window detects `"zh"` → transcribe the full episode with FunASR
  - neither window detects `"zh"` (including `"en"`) → transcribe the full episode with WhisperX (current behavior, unchanged)

  Random windows (rather than e.g. the first 30s) avoid landing on intro music/jingles that would give an unreliable language guess; a second window guards against the first one unluckily landing on such a stretch. The combine rule is OR (favor FunASR) rather than requiring both windows to agree, since requiring agreement would reintroduce the same single-bad-sample failure mode this is meant to fix.
- **Existing bad-quality Chinese transcripts**: out of scope. The user will delete those episodes (DB rows + audio files) manually; this task only changes the transcription path for episodes downloaded from now on.

## Requirements

- Add a `LocalFunASR` transcriber (mirrors `LocalWhisperX`'s shape: eager model load in `__init__`, `async transcribe(audio_path) -> dict` via `asyncio.to_thread`) that produces the same output contract as WhisperX: `{"language": "zh", "segments": [{"text", "start", "end", "words": [{"word", "start", "end"}, ...]}, ...]}`, with per-word entries being per-character (matching the char-level granularity `cleaning/text_join.py` and `cleaning/filler_words.py` already special-case for `"zh"`). FunASR's `sentence_info` (per-VAD-utterance `start`/`end`/`sentence`) maps to `segments`; each sentence's own character timestamp list maps to that segment's `words`. Millisecond timestamps convert to seconds to match WhisperX's convention.
- Add a language-detection step (random 30s window, or the whole clip if the episode is shorter) that reuses the already-loaded WhisperX model — no second model load just for detection.
- Add a `Transcriber` implementation (e.g. `LanguageRoutingTranscriber`) that performs detection + dispatches to `LocalWhisperX` or `LocalFunASR`, fully encapsulated behind the existing `Transcriber` Protocol so **`worker.py` requires no changes** (matches the extension point the archived design already anticipated).
- Update `create_transcriber()` and `cli.py` to build both underlying models and the router; add FunASR-related `Settings` fields (model name, vad/punc model names, device) following the existing `whisperx_*` naming convention.
- Add `funasr` (and its transitive deps) to the `transcription` optional-dependency group in `pyproject.toml`.

## Acceptance Criteria

- [ ] A newly downloaded Chinese-language episode ends up transcribed by FunASR; `episode.language == "zh"`; `episode.transcript_data.segments[].words[]` are per-character with numeric `start`/`end` in seconds.
- [ ] A newly downloaded English-language episode still ends up transcribed by WhisperX exactly as today (no behavior change on the English path).
- [ ] `worker.py` is unmodified (routing logic lives entirely behind the `Transcriber` Protocol).
- [ ] Chunking (`indexing/chunker.py`) and cleaning (`cleaning/filler_words.py`, `cleaning/text_join.py`) work unmodified against FunASR-produced transcripts, since the output shape matches WhisperX's.
- [ ] Unit tests cover: the FunASR-output → segments/words adapter, the random-window language-detection logic (short-audio fallback included), and the routing transcriber's dispatch decision — all without loading real ASR models (mocked).

## Out of Scope

- Re-transcribing/backfilling existing DONE episodes (handled manually by the user via deletion).
- Speaker diarization, non-Mandarin dialects, or any language beyond the existing `zh`/`en` handling.
- Qwen3-ASR (rejected — see Research).
