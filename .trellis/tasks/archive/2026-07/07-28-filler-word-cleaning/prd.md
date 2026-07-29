# Filler-word cleaning for word-level transcript segments

## Goal

Implement `clean_words` in `backend/src/rag_podcast/cleaning/filler_words.py` to strip filler
words (e.g. "um", "uh") from WhisperX segment-level transcript output, operating right after
transcription (pre-chunking) so segment/pause boundaries are preserved. Chunker integration
(making `build_chunks` consume the cleaned segments) is explicitly **out of scope** — deferred to
a separate future task/session.

## Role note

The user is implementing this one themselves as practice. Claude's role for the Execute phase is
review/guidance on request, not authoring the implementation or tests.

## Requirements

1. **Signature & shape**: `clean_words(transcript: list[dict], language: str | None = None) ->
   list[dict]`, where `transcript` is a list of WhisperX segments:
   `{"start": float, "end": float, "text": str, "words": [{"word", "start", "end", "score"}, ...],
   "avg_logprob": float}`. Output is the same shape, with filler words removed from both `text`
   and `words` per segment.
2. **Word filtering**: for each word in a segment's `words`, use the existing `_normalize()` +
   `_filler_words_for(language)` to decide if it's a filler; drop it from `words` if so.
3. **Fix existing bug**: `_filler_words_for` currently calls `language.lower()` unconditionally and
   crashes when `language=None` (the default, and how most callers will invoke this). `None` must
   resolve to the English filler list.
4. **Text reconstruction — do not rejoin kept tokens.** Rebuilding `text` via
   `" ".join(surviving words)` reintroduces the CJK-spacing bug already documented in
   `text_join.py` (whisperX aligns CJK content character-by-character, so naive joins produce
   wrong spacing). Instead, remove exactly the filler substrings from the segment's *original*
   `text`, using the `words` list only to decide which substrings are fillers — preserving the
   original spacing/punctuation of everything that survives.
   - Walk `text` left-to-right with a cursor that advances past each removed span before locating
     the next filler, so a repeated filler token (e.g. two "um"s) doesn't get matched out of order.
   - After deleting a filler substring, collapse the resulting whitespace so no double spaces or
     stray leading/trailing spaces are left at the deletion site.
5. **Pass-through fields**: `start`, `end`, `avg_logprob` are copied unchanged from the segment —
   never recomputed from the surviving words.
6. **Fully-emptied segments**: if a segment's `words` becomes empty after filtering (e.g. a
   segment that was just "um"), drop that segment entirely from the returned list rather than
   keeping an empty placeholder.
7. `cleaning/__init__.py`'s broken `DEFAULT_FILLER_WORDS` import is already fixed (now exports
   `EN_FILLER_WORDS` + `clean_words`) — no work needed there.
8. **Tests**: the existing `clean_words` tests in `backend/tests/test_cleaning.py` encode the old
   flat-word-list contract and must be rewritten for the segment-level input/output shape.

## Acceptance Criteria

- [ ] `clean_words` accepts a list of WhisperX segment dicts + optional `language`, returns the
      same-shaped list with filler words removed from both `text` and `words` per segment.
- [ ] `_filler_words_for(None)` resolves to the English filler list instead of raising.
- [ ] Filler removal from `text` preserves original spacing/punctuation of surviving content (no
      naive rejoin-based reconstruction).
- [ ] No double spaces or stray leading/trailing whitespace left behind at deletion sites.
- [ ] A segment fully emptied of words after cleaning is dropped from the returned list.
- [ ] `avg_logprob`/`start`/`end` are passed through unchanged.
- [ ] `test_cleaning.py`'s `clean_words` tests are rewritten to match the segment-level contract.
- [ ] `pytest backend/tests/test_cleaning.py` passes.

## Notes

- Chunker integration (`build_chunks` consuming cleaned segments instead of raw ones, and updating
  its "operates on ORIGINAL uncleaned segments" docstring) is a known follow-up, tracked as a
  separate future task — not part of this one.
