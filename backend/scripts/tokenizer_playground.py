"""Play with BGE-M3's tokenizer — ad-hoc string tokenization, or per-segment
token-count stats over a real WhisperX transcript (e.g. en_aligned_test.json).

Loading approach matches indexing/chunker.py's planned design (see this
task's design.md, "BGE-M3 tokenizer loading & caching"): fetch tokenizer.json
via huggingface_hub, cached under settings.data_dir so it doesn't touch the
global ~/.cache/huggingface, then load it as a plain tokenizers.Tokenizer
(vocab only — no torch, no model weights).

Usage:
    uv run python scripts/tokenizer_playground.py --text "Hello world"
    uv run python scripts/tokenizer_playground.py --transcript ../en_aligned_test.json
    uv run python scripts/tokenizer_playground.py   # defaults to ../en_aligned_test.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import token

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from rag_podcast.config import settings

REPO_ID = "BAAI/bge-m3"
DEFAULT_TRANSCRIPT = Path(__file__).resolve().parent.parent.parent / "en_aligned_test.json"


def load_tokenizer() -> Tokenizer:
    cache_dir = Path(__file__).resolve().parent.parent.parent / "hf_cache"
    print(f"Loading {REPO_ID}'s tokenizer (cache: {cache_dir})...")
    path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", cache_dir=str(cache_dir))
    return Tokenizer.from_file(path)


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p25": statistics.quantiles(sorted_values, n=4)[0] if len(sorted_values) >= 2 else sorted_values[0],
        "median": statistics.median(sorted_values),
        "p75": statistics.quantiles(sorted_values, n=4)[2] if len(sorted_values) >= 2 else sorted_values[-1],
        "max": sorted_values[-1],
    }


def show_text(tokenizer: Tokenizer, text: str) -> None:
    encoding = tokenizer.encode(text)
    print(f"\nText ({len(text)} chars): {text!r}")
    print(f"Token count: {len(encoding.ids)}")
    print(f"Tokens: {encoding.tokens[:40]}{' ...' if len(encoding.tokens) > 40 else ''}")


def show_transcript(tokenizer: Tokenizer, transcript_path: Path) -> None:
    with open(transcript_path, encoding="utf-8") as f:
        segments = json.load(f) # ["segments"]

    total_duration = segments[-1]["end"] - segments[0]["start"]
    token_counts = [count_tokens(tokenizer, seg["text"]) for seg in segments]
    total_token_counts = len(tokenizer.encode("".join(seg['text'] for seg in segments)).ids)
    total_tokens = sum(token_counts)
    densities = [
        tc / (seg["end"] - seg["start"])
        for tc, seg in zip(token_counts, segments)
        if seg["end"] > seg["start"]
    ]

    print(f"\nTranscript: {transcript_path}")
    print(f"Segments: {len(segments)}, total duration: {total_duration:.1f}s")
    print(f"Total tokens: {total_tokens}  (~{total_tokens / total_duration * 60:.1f} tokens/min)")
    print(f"Per-segment token count percentiles: {_percentiles(token_counts)}")
    print(f"Per-segment token density (tokens/sec) percentiles: {_percentiles(densities)}")

    print(f"Total tokens (all segments): {total_token_counts}  (~{total_token_counts / total_duration * 60:.1f} tokens/min)")
    # Quick reference: how many tokens accumulate over a few candidate
    # duration windows, useful for sanity-checking tier thresholds.
    for window in (20, 90, 300, 600):
        acc = 0
        count = 0
        window_start = segments[0]["start"]
        for seg, tc in zip(segments, token_counts):
            if seg["start"] - window_start > window:
                break
            acc += tc
            count += 1
        print(f"  First ~{window}s window: {count} segments, {acc} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Tokenize an ad-hoc string")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help=f"Path to a WhisperX-segment-shaped JSON file (default: {DEFAULT_TRANSCRIPT})",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    if args.text:
        show_text(tokenizer, args.text)
        return

    transcript_path = args.transcript or DEFAULT_TRANSCRIPT
    show_transcript(tokenizer, transcript_path)


if __name__ == "__main__":
    t = load_tokenizer()
    text = " Hello world"
    word_start  = [0, 6]  
    enc = t.encode(text, add_special_tokens=False)
    print(word_start)
    print(enc.offsets)

    

