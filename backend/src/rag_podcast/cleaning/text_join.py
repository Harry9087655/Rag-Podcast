"""Rejoin a word-level token list into a display/embedding string.

Script-aware: CJK scripts (Chinese, Japanese) are written without spaces
between words, so joining them with a naive " ".join(...) inserts spaces
that were never there — including between individual Latin letters when
WhisperX aligns a whole "zh"-detected episode at character granularity
(e.g. "Hello" embedded in Chinese comes back as separate {"word": "H"},
{"word": "e"}, ... entries). Keying the join on the episode's detected
language (not per-character script sniffing) fixes both cases at once.
"""

from __future__ import annotations

# Languages conventionally written without spaces between words. Korean is
# deliberately excluded: Hangul is block-syllable-based but Korean text is
# conventionally space-delimited between words, unlike Chinese/Japanese.
NO_SPACE_LANGUAGES: frozenset[str] = frozenset({"zh", "ja"})


def join_words(words: list[dict], language: str | None = None) -> str:
    """Join word dicts' ``"word"`` values into a single string.

    Entries missing a ``"word"`` key are skipped. Uses no separator for
    ``language`` in ``NO_SPACE_LANGUAGES``, a single space otherwise
    (including when *language* is ``None``).
    """
    tokens = [w["word"] for w in words if "word" in w]
    separator = "" if (language is not None and language.lower() in NO_SPACE_LANGUAGES) else " "
    return separator.join(tokens)
