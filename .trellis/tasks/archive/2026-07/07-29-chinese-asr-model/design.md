# Chinese ASR Routing — Technical Design

## Architecture Overview

No new processes, no new DB tables/columns, no `worker.py` changes. The existing `Transcriber` Protocol (`transcription/transcriber.py:25`) is the seam: the worker only ever calls `transcriber.transcribe(audio_path)` and stores whatever dict comes back. This design adds one more `Transcriber` implementation that internally owns two models and a routing decision.

```
create_transcriber()
        │
        ▼
LanguageRoutingTranscriber            (implements Transcriber)
  ├─ _whisperx: LocalWhisperX          (existing, unchanged)
  └─ _funasr:   LocalFunASR            (new)

transcribe(audio_path):
  1. audio = whisperx.load_audio(audio_path)                    # 16kHz mono np.ndarray
  2. lang  = detect_language(self._whisperx, audio, ...)         # up to 2 random 30s windows, reuses loaded WhisperX model
  3. if lang == "zh": return await self._funasr.transcribe(audio_path)
     else:            return await self._whisperx.transcribe(audio_path)
```

`worker.py` and `models/episode.py` are untouched. `episode.language` continues to be read from whatever the chosen transcriber's result dict reports (`"zh"` for FunASR, WhisperX's own detected code otherwise) — same line as today (`worker.py:130`, unchanged).

## Module Boundaries

```
backend/src/rag_podcast/transcription/
├── transcriber.py
│   ├── Transcriber (Protocol)                 # unchanged
│   ├── LocalWhisperX                          # unchanged
│   ├── LocalFunASR                             # NEW
│   ├── LanguageRoutingTranscriber              # NEW
│   ├── detect_language(model, audio, ...)      # NEW helper
│   ├── random_window(audio, sample_rate, ...)  # NEW helper
│   └── create_transcriber(...)                 # signature extended
└── worker.py                                   # unchanged
```

## Contracts

### `LocalFunASR`

```python
class LocalFunASR:
    def __init__(self, model: str, vad_model: str, punc_model: str, device: str) -> None:
        # eager load, mirrors LocalWhisperX.__init__
        self._model = funasr.AutoModel(
            model=model, vad_model=vad_model, punc_model=punc_model, device=device,
            disable_update=True,
        )

    async def transcribe(self, audio_path: Path) -> dict:
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    def _transcribe_sync(self, audio_path: Path) -> dict:
        result = self._model.generate(input=str(audio_path))[0]
        return _funasr_to_segments(result)  # adapter, see below
```

### Output adapter — FunASR `sentence_info` → WhisperX-shaped `segments`

FunASR (with `vad_model` + `punc_model` configured) returns, per call:

```python
result = {
    "text": "...",
    "sentence_info": [
        {"text": "欢迎大家来体验", "start": 880, "end": 4200, "timestamp": [[880, 1120], [1120, 1360], ...]},
        ...
    ],
}
```

`start`/`end`/inner `timestamp` pairs are **milliseconds**; WhisperX's contract uses **seconds**. The adapter:

```python
def _funasr_to_segments(result: dict) -> dict:
    segments = []
    for sent in result.get("sentence_info", []):
        words = [
            {"word": ch, "start": s / 1000, "end": e / 1000}
            for ch, (s, e) in zip(sent["text"], sent["timestamp"])
        ]
        segments.append({
            "text": sent["text"],
            "start": sent["start"] / 1000,
            "end": sent["end"] / 1000,
            "words": words,
        })
    return {"language": "zh", "segments": segments}
```

`words[].word` is a single character each — this is exactly the granularity `cleaning/text_join.py:7-9` and `cleaning/filler_words.py:63-65` already special-case for `"zh"` (no-separator join, `ZH_FILLER_WORDS`), and what `indexing/chunker.py`'s oversized-segment split already expects (`_words_span`/`join_words` need per-word `start`/`end`, which every character now has). No downstream changes needed.

`language` is hardcoded to `"zh"` here rather than read from FunASR (which doesn't do language ID) — the routing decision that sent audio to FunASR *is* the language determination.

### Language detection

Uses **two independent random 30-second windows**, not one: a single window risks landing entirely on intro music/silence and returning an unreliable guess. Combine rule is **OR, biased toward not missing Chinese** — if *either* window detects `"zh"`, route to FunASR; only fall through to WhisperX if neither window says `"zh"`. (Requiring both windows to agree would be *more* conservative but defeats the purpose: if window A unluckily hits music and window B correctly says `"zh"`, "require both" would still misroute to WhisperX.) No forced non-overlap between the two draws — for a normal-length episode, two independent random 30s windows are very unlikely to collide anyway.

```python
def random_window(audio: np.ndarray, sample_rate: int, window_seconds: float, rng: random.Random) -> np.ndarray:
    n = len(audio)
    window_len = int(window_seconds * sample_rate)
    if n <= window_len:
        return audio
    start = rng.randrange(0, n - window_len)
    return audio[start : start + window_len]

def detect_language(
    whisperx_model, audio: np.ndarray, window_seconds: float, rng: random.Random,
    num_windows: int = 2, batch_size: int = 8,
) -> str:
    last_lang = "en"
    for _ in range(num_windows):
        clip = random_window(audio, 16000, window_seconds, rng)
        result = whisperx_model.transcribe(clip, batch_size=batch_size)
        last_lang = result.get("language", "en")
        if last_lang == "zh":
            return "zh"
    return last_lang  # neither window said zh — use the last window's guess
```

`detect_language` calls the raw underlying whisperx model (`LocalWhisperX._model`, exposed via a small accessor or by having `LanguageRoutingTranscriber` reach into `self._whisperx._model` since both live in the same module) — no alignment, no second model load. WhisperX's sample rate is fixed at 16000 Hz (existing `whisperx.load_audio` contract).

### `LanguageRoutingTranscriber`

```python
ZH_LANGUAGE_CODES = frozenset({"zh"})

class LanguageRoutingTranscriber:
    def __init__(
        self, whisperx: LocalWhisperX, funasr: LocalFunASR,
        window_seconds: float = 30.0, num_detect_windows: int = 2,
    ) -> None:
        self._whisperx = whisperx
        self._funasr = funasr
        self._window_seconds = window_seconds
        self._num_detect_windows = num_detect_windows
        self._rng = random.Random()

    async def transcribe(self, audio_path: Path) -> dict:
        audio = await asyncio.to_thread(whisperx.load_audio, str(audio_path))
        lang = await asyncio.to_thread(
            detect_language, self._whisperx._model, audio,
            self._window_seconds, self._rng, self._num_detect_windows,
        )
        if lang in ZH_LANGUAGE_CODES:
            return await self._funasr.transcribe(audio_path)
        return await self._whisperx.transcribe(audio_path)
```

Note: both branches re-read `audio_path` from disk inside the delegate (`LocalWhisperX.transcribe` / `LocalFunASR.transcribe` each call their own load). This duplicates one local disk read per episode versus threading the already-loaded array through — acceptable for MVP (local files, not network I/O) and keeps `LocalWhisperX`/`LocalFunASR` self-contained/independently testable rather than needing an internal audio-in variant.

### `create_transcriber()` — extended signature

```python
def create_transcriber(
    *,
    whisperx_model: str, whisperx_device: str, whisperx_compute_type: str, whisperx_batch_size: int = 8,
    funasr_model: str, funasr_vad_model: str, funasr_punc_model: str, funasr_device: str,
    lang_detect_window_seconds: float = 30.0,
) -> Transcriber:
    whisperx_t = LocalWhisperX(whisperx_model, whisperx_device, whisperx_compute_type, whisperx_batch_size)
    funasr_t = LocalFunASR(funasr_model, funasr_vad_model, funasr_punc_model, funasr_device)
    return LanguageRoutingTranscriber(whisperx_t, funasr_t, lang_detect_window_seconds)
```

Both models load eagerly at worker startup (same convention as today's single-model eager load) — `paraformer-zh`/`fsmn-vad`/`ct-punc` are small/fast models, consistent with keeping worker startup simple over micro-optimizing VRAM.

## Config (`config.py` additions)

```python
funasr_model: str = "paraformer-zh"
funasr_vad_model: str = "fsmn-vad"
funasr_punc_model: str = "ct-punc"
funasr_device: str = "cuda"
lang_detect_window_seconds: float = 30.0
```

`cli.py`'s `_main()` passes these plus the existing `whisperx_*` settings into `create_transcriber(...)`.

## Dependencies

`pyproject.toml`'s `transcription` extra gains `funasr` (pulls in `modelscope` for model auto-download, same on-first-run download convention WhisperX already uses).

## Error Handling

| Failure Mode | Behavior |
|---|---|
| FunASR model crash / OOM | `LocalFunASR.transcribe()` wraps in `TranscribeError`, same as `LocalWhisperX` — worker marks episode FAILED, continues (unchanged `worker.py` behavior) |
| Language detection raises | Propagates as `TranscribeError` from `LanguageRoutingTranscriber.transcribe()` (wrap in try/except) — episode marked FAILED rather than silently defaulting, since a broken detection step should surface, not fail open |
| Episode audio shorter than the detection window | `random_window()` returns the whole clip — no error |

## Compatibility & Rollback

- No DB migration. No `worker.py` change. Rollback = remove `funasr` extra, revert `transcriber.py`/`config.py`/`cli.py` to the pre-change versions; nothing else references the new classes.
- `Podcast`/`Episode` schemas unchanged — this design deliberately avoids adding a per-podcast language field per the routing decision (fully automatic detection instead).
