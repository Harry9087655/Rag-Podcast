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

    @field_validator("data_dir")
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


settings = Settings()
