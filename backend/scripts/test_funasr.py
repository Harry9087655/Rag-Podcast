import os 
import torch 
from pathlib import Path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from rag_podcast.transcription.LocalFunASR import LocalFunASR
import asyncio
from sqlalchemy import select
from rag_podcast.db import async_session, get_session
from rag_podcast.models.episode import Episode
from rag_podcast.config import settings
import json


async def get_audio_path(episode_id: int) -> str:
    async with async_session() as session:
        episode = await session.get(Episode, episode_id)
        if episode is None:
            raise ValueError(f"Episode with id {episode_id} not found")
        return os.path.join(settings.data_dir, episode.audio_local_path)



audio = asyncio.run(get_audio_path(2))
print(audio)
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", device="cuda")
result = model.generate(input=str(audio), batch_size_s=300, sentence_timestamp=True, pred_timestamp=True)

print("Transribe successfully, writing to funasr_aligned_test.json")
with open("./funasr_aligned_test.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print("Transcription result written")
print("Length of result:", len(result))
print("Type of result:", type(result))
for item in result:
    print("Type of item:", type(item))

with open("./funasr_aligned_test.json", "r", encoding="utf-8") as f:
    result = json.load(f)
    mock = object.__new__(LocalFunASR)
    segments = mock._funasr_to_segments(result[0])
with open("./funasr_aligned_test_segments.json", "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False)
# print(len(result))