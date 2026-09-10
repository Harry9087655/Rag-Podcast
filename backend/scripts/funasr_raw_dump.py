"""One-off: run FunASR on a single audio file and dump its NATIVE output.

Deliberately does NOT go through ``LocalFunASR._funasr_to_segments`` — the point
is to inspect where ``ct-punc`` inserts punctuation in FunASR's own
``text`` / ``sentence_info`` shape, before any WhisperX-shaped adaptation.

Usage (inside the worker container, which has the GPU + models):
    python scripts/funasr_raw_dump.py /data/<podcast>/<id>/<id>.m4a <out_prefix>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from funasr import AutoModel

from rag_podcast.config import settings


def main() -> None:
    audio_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "funasr_raw"

    print(f"Loading model={settings.funasr_model} vad={settings.funasr_vad_model} "
          f"punc={settings.funasr_punc_model} device={settings.funasr_device} …",
          flush=True)
    model = AutoModel(
        model=settings.funasr_model,
        vad_model=settings.funasr_vad_model,
        punc_model=settings.funasr_punc_model,
        device=settings.funasr_device,
        disable_update=True,
    )

    print(f"Transcribing {audio_path} …", flush=True)
    started = time.time()
    result = model.generate(
        input=audio_path,
        batch_size_s=300,
        sentence_timestamp=True,
        pred_timestamp=True,
    )
    print(f"Done in {time.time() - started:.1f}s", flush=True)

    # No rich_transcription_postprocess() — keep the model's text verbatim.
    json_path = Path(f"{out_prefix}.json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    item = result[0]
    sentences = item.get("sentence_info", [])

    # Human-readable preview: one sentence per line, so punctuation placement
    # (and the fact that ct-punc emits sentence-final marks that split the
    # sentence stream) is visible at a glance.
    lines = [
        f"# audio: {audio_path}",
        f"# sentences: {len(sentences)}   text chars: {len(item.get('text', ''))}",
        f"# timestamp entries (top-level): {len(item.get('timestamp', []))}",
        "",
        "=== full text (single string, as FunASR returns it) ===",
        item.get("text", ""),
        "",
        "=== sentence_info, one per line: [idx] start-end  text ===",
    ]
    for i, s in enumerate(sentences):
        text = s.get("text", "")
        n_ts = len(s.get("timestamp", []))
        lines.append(
            f"[{i:04d}] {s['start'] / 1000:8.2f}-{s['end'] / 1000:8.2f}s  "
            f"chars={len(text):3d} ts={n_ts:3d}  {text}"
        )

    txt_path = Path(f"{out_prefix}_preview.txt")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path} and {txt_path}", flush=True)


if __name__ == "__main__":
    main()
