# Re-exports for convenience.  The whisperx library is an optional
# [transcription] extra and is only installed on the host (GPU access).
# Guard the import so the API container (which has no whisperx) doesn't
# crash if something references ``rag_podcast.transcription`` at import
# time.
try:
    from .transcriber import create_transcriber, LocalWhisperX, TranscribeError, Transcriber  # noqa: F401
    from .worker import run_worker, transcribe_episode  # noqa: F401
except ImportError:
    create_transcriber = None  # type: ignore[assignment]
    LocalWhisperX = None  # type: ignore[assignment]
    TranscribeError = Exception  # type: ignore[assignment,misc]
    Transcriber = None  # type: ignore[assignment]
    run_worker = None  # type: ignore[assignment]
    transcribe_episode = None  # type: ignore[assignment]
