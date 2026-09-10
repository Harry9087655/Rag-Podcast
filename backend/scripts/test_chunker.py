import pytest 
from rag_podcast.indexing import SizeBasedChunker, tokenizer
from pathlib import Path 
import json 
import logging 
from rag_podcast.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent.parent.parent

with open(root/"funasr_aligned_test_segments.json", encoding="utf-8") as f:
    data = json.load(f)['segments']

logger.info(f"Loaded {len(data)} segments")

logger.info(f"First segment: {data[0]}")


chunker = SizeBasedChunker.SizeBasedChunker(tokenizer=tokenizer.get_tokenizer())

chunks = chunker.build_chunks(data, 150, 400, 275, 10, 300, 'zh')
logger.info(f"Built {len(chunks)} chunks")
logger.info(f"First chunk: {chunks[0].text}")