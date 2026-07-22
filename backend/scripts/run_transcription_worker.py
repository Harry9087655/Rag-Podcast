"""Thin wrapper around ``python -m rag_podcast.transcription.cli``.

Run on the host (not inside Docker) to access GPU/CUDA::

    uv run --extra transcription python backend/scripts/run_transcription_worker.py
    uv run --extra transcription python -m rag_podcast.transcription.cli   # equivalent

The CLI logic lives in ``src/rag_podcast/transcription/cli.py``.
"""

from rag_podcast.transcription.cli import main

if __name__ == "__main__":
    main()
