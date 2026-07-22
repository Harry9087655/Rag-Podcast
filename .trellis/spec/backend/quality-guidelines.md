# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

<!--
Document your project's quality standards here.

Questions to answer:
- What patterns are forbidden?
- What linting rules do you enforce?
- What are your testing requirements?
- What code review standards apply?
-->

(To be filled by the team)

---

## Forbidden Patterns

<!-- Patterns that should never be used and why -->

(To be filled by the team)

---

## Required Patterns

### Guarded Imports for Optional GPU Dependencies

**Problem**: Modules with optional extras (e.g., `whisperx`, `torch`) must be importable in environments that don't have those packages installed (e.g., the API container in Docker, which has no GPU).

**Solution**: Use `try/except ImportError` in the package `__init__.py` to guard re-exports:

```python
# transcription/__init__.py
from .transcriber import TranscribeError as TranscribeError

try:
    from .transcriber import LocalWhisperX as LocalWhisperX
except ImportError:
    LocalWhisperX = None  # type: ignore[assignment]

try:
    from .worker import run_worker as run_worker, transcribe_episode as transcribe_episode
except ImportError:
    run_worker = None  # type: ignore[assignment]
    transcribe_episode = None  # type: ignore[assignment]
```

**Why this works**: `TranscribeError` and `Transcriber` have no heavy dependencies, so they're always safe to import. `LocalWhisperX` imports `whisperx`/`torch` — guarded. `run_worker` transitively imports `LocalWhisperX` through type annotations (deferred by `from __future__ import annotations`) — also guarded.

**Anti-pattern**: Eager `from .transcriber import LocalWhisperX` at module level without a guard. This crashes `import rag_podcast.transcription` in any environment without whisperx, even if the import is only for type-checking.

---

## Testing Requirements

<!-- What level of testing is expected -->

(To be filled by the team)

---

## Code Review Checklist

<!-- What reviewers should check -->

(To be filled by the team)
