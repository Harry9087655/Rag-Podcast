from __future__ import annotations

from rag_podcast.cleaning import clean_words, join_words


def test_clean_words_removes_fillers_and_preserves_survivor_words():
    segments = [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "So um today we're uh talking",
            "words": [
                {"word": "So", "start": 0.0, "end": 0.2, "score": 0.9},
                {"word": "um", "start": 0.2, "end": 0.4, "score": 0.9},
                {"word": "today", "start": 0.4, "end": 0.8, "score": 0.9},
                {"word": "we're", "start": 0.8, "end": 1.0, "score": 0.9},
                {"word": "uh", "start": 1.0, "end": 1.1, "score": 0.9},
                {"word": "talking", "start": 1.1, "end": 1.5, "score": 0.9},
            ],
            "avg_logprob": -0.2,
        }
    ]

    result = clean_words(segments)

    assert result == [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "So today we're talking",
            "words": [
                {"word": "So", "start": 0.0, "end": 0.2, "score": 0.9},
                {"word": "today", "start": 0.4, "end": 0.8, "score": 0.9},
                {"word": "we're", "start": 0.8, "end": 1.0, "score": 0.9},
                {"word": "talking", "start": 1.1, "end": 1.5, "score": 0.9},
            ],
            "avg_logprob": -0.2,
        }
    ]


def test_clean_words_matches_fillers_with_punctuation_case_insensitively():
    segments = [
        {
            "start": 0.0,
            "end": 1.4,
            "text": "Um, Hello uh- AH. world",
            "words": [
                {"word": "Um,", "start": 0.0, "end": 0.3, "score": 0.9},
                {"word": "Hello", "start": 0.3, "end": 0.6, "score": 0.9},
                {"word": "uh-", "start": 0.6, "end": 0.8, "score": 0.9},
                {"word": "AH.", "start": 0.8, "end": 1.0, "score": 0.9},
                {"word": "world", "start": 1.0, "end": 1.4, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]

    result = clean_words(segments)

    assert result == [
        {
            "start": 0.0,
            "end": 1.4,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.3, "end": 0.6, "score": 0.9},
                {"word": "world", "start": 1.0, "end": 1.4, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]


def test_clean_words_no_fillers_leaves_words_and_avg_logprob_unchanged():
    segments = [
        {
            "start": 0.0,
            "end": 0.6,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.3, "score": 0.9},
                {"word": "world", "start": 0.3, "end": 0.6, "score": 0.9},
            ],
            "avg_logprob": -0.05,
        }
    ]

    result = clean_words(segments)

    assert result == segments


def test_clean_words_empty_input_returns_empty_list():
    assert clean_words([]) == []


def test_clean_words_segment_fully_emptied_by_fillers_is_dropped():
    segments = [
        {
            "start": 0.0,
            "end": 0.3,
            "text": "Um",
            "words": [{"word": "Um", "start": 0.0, "end": 0.3, "score": 0.9}],
            "avg_logprob": -0.1,
        },
        {
            "start": 0.3,
            "end": 0.8,
            "text": "Hello",
            "words": [{"word": "Hello", "start": 0.3, "end": 0.8, "score": 0.9}],
            "avg_logprob": -0.1,
        },
    ]

    result = clean_words(segments)

    assert result == [
        {
            "start": 0.3,
            "end": 0.8,
            "text": "Hello",
            "words": [{"word": "Hello", "start": 0.3, "end": 0.8, "score": 0.9}],
            "avg_logprob": -0.1,
        },
    ]


def test_clean_words_language_none_falls_back_to_english_without_crashing():
    segments = [
        {
            "start": 0.0,
            "end": 0.5,
            "text": "um hola",
            "words": [
                {"word": "um", "start": 0.0, "end": 0.2, "score": 0.9},
                {"word": "hola", "start": 0.2, "end": 0.5, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]

    result = clean_words(segments, language=None)

    assert result == [
        {
            "start": 0.0,
            "end": 0.5,
            "text": "hola",
            "words": [{"word": "hola", "start": 0.2, "end": 0.5, "score": 0.9}],
            "avg_logprob": -0.1,
        }
    ]


def test_clean_words_non_zh_language_falls_back_to_english_list():
    segments = [
        {
            "start": 0.0,
            "end": 0.5,
            "text": "um hola",
            "words": [
                {"word": "um", "start": 0.0, "end": 0.2, "score": 0.9},
                {"word": "hola", "start": 0.2, "end": 0.5, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]

    result = clean_words(segments, language="es")

    assert result == [
        {
            "start": 0.0,
            "end": 0.5,
            "text": "hola",
            "words": [{"word": "hola", "start": 0.2, "end": 0.5, "score": 0.9}],
            "avg_logprob": -0.1,
        }
    ]


def test_clean_words_chinese_language_uses_zh_filler_list_and_no_space_join():
    # Real shape from WhisperX: Chinese is aligned per-character, so each
    # character (and the filler token itself) is its own word entry.
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "大家好呃你好",
            "words": [
                {"word": "大", "start": 0.0, "end": 0.1, "score": 0.9},
                {"word": "家", "start": 0.1, "end": 0.2, "score": 0.9},
                {"word": "好", "start": 0.2, "end": 0.3, "score": 0.9},
                {"word": "呃", "start": 0.3, "end": 0.4, "score": 0.9},
                {"word": "你", "start": 0.4, "end": 0.5, "score": 0.9},
                {"word": "好", "start": 0.5, "end": 0.6, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]

    result = clean_words(segments, language="zh")

    assert result == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "大家好你好",
            "words": [
                {"word": "大", "start": 0.0, "end": 0.1, "score": 0.9},
                {"word": "家", "start": 0.1, "end": 0.2, "score": 0.9},
                {"word": "好", "start": 0.2, "end": 0.3, "score": 0.9},
                {"word": "你", "start": 0.4, "end": 0.5, "score": 0.9},
                {"word": "好", "start": 0.5, "end": 0.6, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]


def test_clean_words_word_missing_start_end_is_kept():
    segments = [
        {
            "start": 0.0,
            "end": 0.6,
            "text": "hello um world",
            "words": [
                {"word": "hello"},
                {"word": "um", "start": 0.2, "end": 0.4, "score": 0.9},
                {"word": "world", "start": 0.4, "end": 0.6, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]

    result = clean_words(segments)

    assert result == [
        {
            "start": 0.0,
            "end": 0.6,
            "text": "hello world",
            "words": [
                {"word": "hello"},
                {"word": "world", "start": 0.4, "end": 0.6, "score": 0.9},
            ],
            "avg_logprob": -0.1,
        }
    ]


def test_clean_words_text_is_rejoined_not_edited_in_place():
    # Documents current behavior: "text" is rebuilt via a fresh join of the
    # surviving word tokens rather than by editing the original string, so
    # layout not captured by the tokens themselves — e.g. a leading space
    # before the first word — is not preserved. See clean_words's docstring.
    segments = [
        {
            "start": 0.291,
            "end": 4.035,
            "text": " Hello, um yes, welcome.",
            "words": [
                {"word": "Hello,", "start": 0.291, "end": 0.712, "score": 0.477},
                {"word": "um", "start": 0.752, "end": 1.012, "score": 0.974},
                {"word": "yes,", "start": 1.272, "end": 1.432, "score": 0.905},
                {"word": "welcome.", "start": 1.472, "end": 1.632, "score": 0.753},
            ],
            "avg_logprob": -0.1619944914298899,
        }
    ]

    result = clean_words(segments)

    assert result[0]["text"] == "Hello, yes, welcome."


def test_join_words_no_language_defaults_to_space_join():
    words = [{"word": "Hello"}, {"word": "world"}]
    assert join_words(words) == "Hello world"


def test_join_words_english_language_uses_space_join():
    words = [{"word": "Hello"}, {"word": "world"}]
    assert join_words(words, language="en") == "Hello world"


def test_join_words_chinese_language_uses_no_space_join():
    # Real shape from episode 1's transcript_data: WhisperX aligns Chinese
    # per-character, each character its own word entry.
    words = [{"word": "大"}, {"word": "家"}, {"word": "好"}]
    assert join_words(words, language="zh") == "大家好"


def test_join_words_japanese_language_uses_no_space_join():
    words = [{"word": "こ"}, {"word": "ん"}, {"word": "に"}]
    assert join_words(words, language="ja") == "こんに"


def test_join_words_chinese_language_glues_embedded_latin_tokens_too():
    # Regression test + quality improvement: episode 1's real segment mixes
    # "Hello" into Chinese, and WhisperX (having detected the whole episode
    # as "zh") aligns even the Latin letters at character granularity:
    # {"word":"H"}, {"word":"e"}, {"word":"l"}, ... A naive " ".join(...)
    # over this produces "H e l l o 大 家 好". The old per-character CJK-range
    # detection approach only fixed the CJK portion, still leaving
    # "H e l l o大家好" (spaces *within* "Hello", since individual Latin
    # letters aren't themselves CJK characters). Keying the join on
    # episode.language instead fixes this properly: since the whole episode
    # was aligned as one "zh" file, joining everything with no space
    # correctly re-glues "Hello" back together too.
    words = [
        {"word": "H"}, {"word": "e"}, {"word": "l"}, {"word": "l"}, {"word": "o"},
        {"word": "大"}, {"word": "家"}, {"word": "好"},
    ]
    assert join_words(words, language="zh") == "Hello大家好"


def test_join_words_korean_language_uses_space_join():
    # Korean is deliberately NOT in the no-space language set — Hangul is
    # visually block-syllable-based but Korean conventionally spaces
    # between words, unlike Chinese/Japanese.
    words = [{"word": "안녕"}, {"word": "하세요"}]
    assert join_words(words, language="ko") == "안녕 하세요"


def test_join_words_skips_words_missing_word_key():
    words = [{"word": "Hello"}, {"start": 0.1, "end": 0.2}, {"word": "world"}]
    assert join_words(words) == "Hello world"


def test_join_words_empty_input_returns_empty_string():
    assert join_words([]) == ""
