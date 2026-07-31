"""Thin wrapper around ``python -m rag_podcast.transcription.cli``.

The transcription worker normally runs as the ``worker`` service in
``docker-compose.yml`` (containerized, with GPU access via compose's
``deploy.resources.reservations.devices``) — see ``backend/Dockerfile.worker``.
This script is a manual/local-debugging fallback for running it directly on
the host instead::

    uv run --extra transcription python backend/scripts/run_transcription_worker.py
    uv run --extra transcription python -m rag_podcast.transcription.cli   # equivalent

The CLI logic lives in ``src/rag_podcast/transcription/cli.py``.
"""

from rag_podcast.transcription.cli import main

if __name__ == "__main__":
    main()
