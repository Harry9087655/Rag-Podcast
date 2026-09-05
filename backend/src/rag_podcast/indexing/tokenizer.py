from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from pathlib import Path
from rag_podcast.config import settings
import logging
BGE_M3_TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


logger = logging.getLogger(__name__)

def get_tokenizer() -> Tokenizer:
    cache_dir = Path(settings.hf_cache_dir)
    logger.info(f"Loading {settings.embedding_model}'s tokenizer (cache: {cache_dir})...")
    path = hf_hub_download(repo_id=settings.embedding_model, filename="tokenizer.json", 
                           revision=BGE_M3_TOKENIZER_REVISION,
                           cache_dir=str(cache_dir))
    return Tokenizer.from_file(path)
    