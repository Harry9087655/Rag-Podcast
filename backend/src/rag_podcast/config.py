from pathlib import Path
from typing import Literal
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
    evaluator_model: str | None = None

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
    chunk_min_duration_seconds: float = 10.0
    chunk_max_duration_seconds: float = 360.0

    # Audio tier settings: per-duration-tier token budgets, split into leaf
    # (the only embedded/retrieved unit) and parent (time-contiguous leaf merge)
    # boundaries. Values are tunable defaults (see 09-17 design §3).
    audio_tier: dict[str, dict[str, float | dict[str, float]]] = {
        "Short": {  # ≤900s
            "max_episode_duration_seconds": 900,
            "leaf": {"min_tokens": 30, "desired_tokens": 45, "max_tokens": 60},
            "parent": {"desired_tokens": 300, "max_tokens": 500},
        },
        "Medium": {  # ≤2700s
            "max_episode_duration_seconds": 2700,
            "leaf": {"min_tokens": 60, "desired_tokens": 100, "max_tokens": 150},
            "parent": {"desired_tokens": 400, "max_tokens": 650},
        },
        "Long": {  # >2700s
            "max_episode_duration_seconds": float("inf"),
            "leaf": {"min_tokens": 150, "desired_tokens": 200, "max_tokens": 250},
            "parent": {"desired_tokens": 520, "max_tokens": 800},
        },
    }

    embedding_batch_size: int = 32
    embedding_timeout_seconds: float = 60.0

    retrieval_top_k: int = 5
    retrieval_score_mode: Literal["distance", "similarity"] = "distance"
    retrieval_candidate_multiplier: int = 3
    auto_merge_threshold: int = 2

    llm_max_context_tokens: int = 1000000
    llm_response_reserve_tokens: int = 2048
    llm_timeout_seconds: float = 120.0
    qa_max_citations: int = 8  # Maximum number of transcript segments referenced by the LLM


settings = Settings()

if __name__ == "__main__":
    print(settings.hf_cache_dir)
