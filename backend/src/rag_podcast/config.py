from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"))

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
