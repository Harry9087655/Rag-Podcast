from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # extra="ignore": .env holds infra-only keys the app doesn't consume
    # (e.g. HOST_CACHE_DIR, read directly by docker-compose for the worker's
    # model-cache bind mounts) alongside app settings. Without this,
    # pydantic-settings' default extra="forbid" makes Settings() raise on
    # any key in .env that isn't a declared field below.
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), extra="ignore"
    )

    data_dir: str

    @field_validator("data_dir", "hf_cache_dir")
    @classmethod
    def resolve_data_dir(cls, v: str) -> str:
        path = Path(v)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)
    postgres_user: str
    postgres_password: str
    postgres_db: str
    database_url: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    # WhisperX transcription settings
    whisperx_model: str = "small"
    whisperx_device: str = "cuda"
    whisperx_compute_type: str = "float16"
    worker_poll_interval: int = 10

    # FunASR (Chinese ASR) transcription settings
    funasr_model: str = "paraformer-zh"
    funasr_vad_model: str = "fsmn-vad"
    funasr_punc_model: str = "ct-punc"
    funasr_device: str = "cuda"
    lang_detect_window_seconds: float = 30.0

    # HuggingFace Hub settings
    hf_cache_dir: str = "./.cache/hf"


    # Indexing / chunking settings
    chunk_min_duration_seconds: float = 20.0
    chunk_max_duration_seconds: float = 360.0

    # Audio tier settings
    audio_tier: dict[str, dict[str, float]] = {"Short": {"max_episode_duration_seconds": 900,
                                                         "min_chunk_tokens": 100,
                                                         "max_chunk_tokens": 250,
                                                         "desired_chunk_tokens": 175},
                                                "Medium": {"max_episode_duration_seconds": 2700,
                                                           "min_chunk_tokens": 150,
                                                           "max_chunk_tokens": 400,
                                                           "desired_chunk_tokens": 275},
                                                "Long": {"max_episode_duration_seconds": float("inf"),  # no limit
                                                         "min_chunk_tokens": 250,
                                                         "max_chunk_tokens": 600,
                                                         "desired_chunk_tokens": 425}
                                                }

    embedding_batch_size: int = 32
    embedding_timeout_seconds: float = 60.0


settings = Settings()

if __name__ == "__main__":
    print(settings.hf_cache_dir)
