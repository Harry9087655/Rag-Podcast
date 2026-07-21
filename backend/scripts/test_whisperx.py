import os 
import torch 
from pathlib import Path
import whisperx
from rag_podcast.config import settings
import asyncio
from sqlalchemy import select
from rag_podcast.db import async_session, get_session
from rag_podcast.models.episode import Episode
from rag_podcast.config import settings


device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16"
# audio_path = Path(__file__).resolve().parent / "probe_sample.wav"


async def get_audio_path(episode_id: int) -> str:
    async with async_session() as session:
        episode = await session.get(Episode, episode_id)
        if episode is None:
            raise ValueError(f"Episode with id {episode_id} not found")
        return os.path.join(settings.data_dir, episode.audio_local_path)
    


def main() -> None:
    if device == "cuda":
        print(f"available GPU {torch.cuda.get_device_name()}")
    print(f"Loading WhisperX mode.......")
    model = whisperx.load_model("small", device, compute_type=compute_type)
    print("-------- Model is loaded -------")
    audio_path = asyncio.run(get_audio_path(1))  # Replace 1 with the actual episode ID
    print(f"Loading audio from {audio_path}.......")
    audio = whisperx.load_audio(audio_path)
    print(f"Audio {audio_path} is loaded")
    print("Transcribing.......")
    result = model.transcribe(audio, batch_size=8)
    try:
        print("Raw segments:", result['segments'][:50])
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        with open("aligned_segments.txt", "w") as f:
            f.write(str(result["segments"]))
        print("Transcription result written")
    except Exception as e:
        print(f"Error occurred while aligning segments: {e}")

if __name__ == "__main__":
    main()
    