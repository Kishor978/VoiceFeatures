from pydub import AudioSegment
import librosa
import numpy as np

def convert_m4a_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio.export(output_path, format="wav")

def load_audio(file_path):
    # Convert .m4a to .wav if needed
    if file_path.endswith(".m4a"):
        wav_path = file_path.replace(".m4a", ".wav")
        convert_m4a_to_wav(file_path, wav_path)
        file_path = wav_path

    y, sr = librosa.load(file_path, sr=16000)
    return y, sr

# Example usage
file_path = "Recording.m4a"  # Change to your actual file
y, sr = load_audio(file_path)
print(f"Loaded audio with sample rate {sr}")
