import sys 
import torch 
from pathlib import Path
import whisperx
import gc 
from whisperx.diarize import DiarizationPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16"
audio_path = Path(__file__).resolve().parent / "probe_sample.wav"
def main() -> None:
    if device == "cuda":
        print(f"available GPU {torch.cuda.get_device_name()}")
    print(f"Loading WhisperX mode.......")
    model = whisperx.load_model("small", device, compute_type=compute_type)
    print("-------- Model is loaded -------")
    audio = whisperx.load_audio(audio_path)
    print(f"Audio {audio_path} is loaded")
    print("Transcribing.......")
    result = model.transcribe(audio, batch_size=8)
    print("Raw segments:", result['segments'])
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("Alligned segments:", result["segments"])

if __name__ == "__main__":
    main()
    