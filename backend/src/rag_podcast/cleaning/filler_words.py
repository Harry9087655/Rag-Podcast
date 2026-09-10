
from __future__ import annotations
import re
import string

# Default English filler-word list. Module-level constant so it's easy to
# extend later (and so a future per-language list has an obvious sibling to
# follow the shape of).
EN_FILLER_WORDS: frozenset[str] = frozenset({"um", "uh", "ah", "hmm", "erm", "er", "hm"})

ZH_FILLER_WORDS: frozenset[str] = frozenset({"额", "呃", "噢", "嗯", "呢"})

# `string.punctuation` is ASCII-only. FunASR word tokens carry the mark that
# follows them, and for Chinese that mark is full-width — without these,
# `_normalize("嗯，")` returns "嗯，", which matches nothing in ZH_FILLER_WORDS
# and silently disables Chinese filler removal.
_CJK_PUNCTUATION = "，。！？；：、…“”‘’（）《》【】—～·"
_STRIP_CHARS = string.punctuation + string.whitespace + _CJK_PUNCTUATION


def _normalize(word: str) -> str:
    """Lowercase and strip surrounding punctuation/whitespace for comparison.

    WhisperX word text can carry leading/trailing punctuation (e.g. "Um,",
    "uh-"), so a naive lowercase-only compare would miss those.
    """
    return word.strip(_STRIP_CHARS).lower()


def clean_words(transcript: list[dict], language: str | None = 'en') -> list[dict]:
    """
    Input: Raw whisperX transcript segments, in the form of [{"start": ,
    "end": , "text": , "words": [{"word": , "start": , "end": , "score": }, ],
    "avg_logprob": }, ]. Runs right after transcription, before chunking, so
    segment/pause boundaries are still intact.
    Output: Same structure, with filler words removed from both "text" and
    "words" per segment. "start"/"end"/"avg_logprob" are passed through
    unchanged. A segment left with no words after filtering is dropped
    entirely from the returned list. Only "zh" and "en" are supported
    languages; "zh" joins surviving word tokens with no separator (WhisperX
    aligns Chinese at character granularity), anything else — including
    None/unset — joins with a single space and uses the English filler list.
    Strategy:
    1. Iterate through each segment dict in the list.
    2. Filter "words": drop entries whose normalized word is a filler
       (via `_normalize` against the language's filler-word set).
    3. Rebuild "text" by rejoining the surviving word tokens with the
       language-specific joiner. Note this does not preserve the original
       text's exact whitespace/punctuation layout (e.g. a leading space
       before the first word is lost) — it's a fresh join of the surviving
       tokens, not an edit of the original string.

       Example:
       Input:
       [{"start": 0.291, "end": 4.035, "text": " Hello, um yes, welcome.",
       "words": [{"word": "Hello,", "start": 0.291, "end": 0.712, "score": 0.477},
       {"word": "um", "start": 0.752, "end": 1.012, "score": 0.974}, {"word": "yes,",
       "start": 1.272, "end": 1.432, "score": 0.905}, {"word": "welcome.", "start": 1.472,
       "end": 1.632, "score": 0.753}], "avg_logprob": -0.1619944914298899}]
       Output:
       [{"start": 0.291, "end": 4.035, "text": "Hello, yes, welcome.",
       "words": [{"word": "Hello,", "start": 0.291, "end": 0.712, "score": 0.477},
       {"word": "yes,", "start": 1.272, "end": 1.432, "score": 0.905},
       {"word": "welcome.", "start": 1.472, "end": 1.632, "score": 0.753}], "avg_logprob": -0.1619944914298899}]


    """
    if language is not None and language.lower() == 'zh':
        filler_words = ZH_FILLER_WORDS
        joined_by = ""
    else:
        filler_words = EN_FILLER_WORDS
        joined_by = " "

    cleaned_transcript = []
    for text_dict in transcript:
        text_dict = text_dict.copy()
        cleaned_words = [w for w in text_dict['words'] if _normalize(w.get('word','')) not in filler_words] 
        if not cleaned_words:
            continue
        text_dict['words'] = cleaned_words
        text_dict['text'] = joined_by.join([w['word'] for w in cleaned_words])
        cleaned_transcript.append(text_dict)

    return cleaned_transcript

if __name__ == "__main__":
    test_1 = [{"start": 0.291, "end": 4.035, "text": " Um, hello yes, welcome.",
       "words": [{"word": "Um,", "start": 0.291, "end": 0.712, "score": 0.477},
       {"word": "hello", "start": 0.752, "end": 1.012, "score": 0.974}, {"word": "yes,",
       "start": 1.272, "end": 1.432, "score": 0.905}, {"word": "welcome.", "start": 1.472,
       "end": 1.632, "score": 0.753}], "avg_logprob": -0.1}]

    test_2 = [{"start": 0.291, "end": 4.035, "text": " Um.",
       "words": [{"word": "Um.", "start": 0.291, "end":4.035, "score": 0.477}], "avg_logprob": -0.1}]

    test_3 = [{"start": 0.291, "end": 4.035, "text": " Um, hmm.",
       "words": [{"word": "Um,", "start": 0.291, "end":1.100, "score": 0.477}, 
        {"word": "hmm.", "start": 1.100, "end":4.035, "score": 0.477}], "avg_logprob": -0.1}]
    # print([w['word'] for ws in test for w in ws['words']])

    from pathlib import Path 
    import json
    with open(Path(__file__).resolve().parent.parent.parent.parent.parent / "funasr_aligned_test_segments.json",
              encoding='utf-8') as f:
        data = json.load(f)['segments']
    res = clean_words(data, language='zh')
    print(res[:5])
    print(f"Initial number of words: {sum(len(s['words']) for s in data)}")
    print(f"Number of words after cleaning: {sum(len(s['words']) for s in clean_words(data, language='zh'))}")
    