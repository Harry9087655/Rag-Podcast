"""One-off: prove where FunASR's sentence_info punctuation drift comes from.

Monkeypatches the three functions on the punctuation path so we can see the
exact strings/arrays that ``AutoModel.inference_with_vad`` feeds them:

  _join_vad_texts        -> builds ``punc_input_text`` from per-VAD-chunk ASR text
  timestamp_sentence     -> legacy splitter, tokenizes with ``text.split()``
  _timestamp_sentences_from_surface / _punctuate_surface_text -> surface path

Usage (inside the worker container):
    python scripts/funasr_punc_probe.py /data/_probe60.wav
"""

from __future__ import annotations

import sys

from funasr import AutoModel
from funasr.auto import auto_model as am
from funasr.utils import timestamp_tools
from funasr.models.ct_transformer.utils import split_words

captured: dict = {}

_orig_join = am._join_vad_texts
_orig_ts_sentence = am.timestamp_sentence
_orig_surface_sentences = am._timestamp_sentences_from_surface
_orig_punct_surface = am._punctuate_surface_text


def join_spy(texts):
    chunks = list(texts)
    out = _orig_join(chunks)
    captured["vad_chunks"] = chunks
    captured["punc_input_text"] = out
    return out


def ts_sentence_spy(punc_id_list, timestamp_postprocessed, text_postprocessed, **kw):
    captured["legacy_path_used"] = True
    captured["punc_array_len"] = len(punc_id_list) if punc_id_list is not None else None
    captured["timestamps_len"] = len(timestamp_postprocessed or [])
    captured["texts_split"] = text_postprocessed.split()
    return _orig_ts_sentence(punc_id_list, timestamp_postprocessed, text_postprocessed, **kw)


def surface_sentences_spy(*a, **kw):
    out = _orig_surface_sentences(*a, **kw)
    captured["surface_sentences_returned"] = out is not None
    return out


def punct_surface_spy(*a, **kw):
    out = _orig_punct_surface(*a, **kw)
    captured["punctuated_surface_returned"] = out is not None
    return out


am._join_vad_texts = join_spy
am.timestamp_sentence = ts_sentence_spy
am._timestamp_sentences_from_surface = surface_sentences_spy
am._punctuate_surface_text = punct_surface_spy
timestamp_tools.timestamp_sentence = ts_sentence_spy


def main() -> None:
    audio = sys.argv[1]
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cuda",
        disable_update=True,
    )
    result = model.generate(
        input=audio, batch_size_s=300, sentence_timestamp=True, pred_timestamp=True
    )
    res = result[0]

    print("\n" + "=" * 70)
    print("PATH TAKEN")
    print("=" * 70)
    print("result has 'words' key            :", "words" in res)
    print("punctuated_surface returned       :", captured.get("punctuated_surface_returned"))
    print("surface_sentences returned        :", captured.get("surface_sentences_returned"))
    print("legacy timestamp_sentence() used  :", captured.get("legacy_path_used", False))

    chunks = captured.get("vad_chunks", [])
    punc_text = captured.get("punc_input_text", "")
    texts = captured.get("texts_split", [])
    punc_tokens = split_words(punc_text) if punc_text else []

    print("\n" + "=" * 70)
    print("TOKENIZATION MISMATCH")
    print("=" * 70)
    print("VAD chunks                        :", len(chunks))
    print("len(punc_array)                   :", captured.get("punc_array_len"))
    print("len(result['timestamp'])          :", captured.get("timestamps_len"))
    print("len(split_words(punc_input_text)) :", len(punc_tokens),
          "  <- what punc_array indexes")
    print("len(punc_input_text.split())      :", len(texts),
          "  <- what timestamp_sentence() indexes")
    print("DEFICIT                           :", len(punc_tokens) - len(texts))

    print("\nfirst 3 VAD chunk texts (repr):")
    for c in chunks[:3]:
        print("   ", repr(c))
    print("\npunc_input_text[:120]:", repr(punc_text[:120]))
    print("\nwhitespace tokens holding >1 CJK char (the glued VAD boundaries):")
    shown = 0
    for i, tok in enumerate(texts):
        cjk = [c for c in tok if not c.isascii()]
        if len(cjk) > 1:
            print(f"    texts[{i}] = {tok!r}")
            shown += 1
            if shown >= 8:
                break
    if not shown:
        print("    (none)")

    print("\nresult['text'][:100]      :", res["text"][:100])
    print("sentence_info joined[:100]:",
          "".join(s["text"] for s in res.get("sentence_info", []))[:100])


if __name__ == "__main__":
    main()
