import librosa
import numpy as np
# import parselmouth  # For Praat-based analysis
import pyworld as pw
import soundfile as sf
import matplotlib.pyplot as plt

# Load audio file
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # Convert to 16kHz mono
    return y, sr

# Extract spectral features
def extract_spectral_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    return {
        "MFCCs": mfccs.mean(axis=1),
        "Spectral Centroid": np.mean(spectral_centroid),
        "Spectral Bandwidth": np.mean(spectral_bandwidth),
        "Spectral Rolloff": np.mean(spectral_rolloff),
        "Spectral Contrast": np.mean(spectral_contrast)
    }

# Extract speaking volume features
def extract_volume_features(y):
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    return {
        "RMS Energy": np.mean(rms),
        "Zero Crossing Rate": np.mean(zcr)
    }

# # Extract pitch and harmonic features using Parselmouth (Praat)
# def extract_pitch_harmonics(file_path):
#     snd = parselmouth.Sound(file_path)
#     pitch = snd.to_pitch()
    
#     # Extract pitch contour (F0)
#     f0_values = pitch.selected_array['frequency']
#     f0_values = f0_values[f0_values > 0]  # Remove unvoiced parts

#     # Extract jitter, shimmer, and HNR
#     jitter = snd.get_jitter_relative()
#     shimmer = snd.get_shimmer_local()
#     hnr = snd.to_harmonicity().get_mean()
    
#     return {
#         "Mean Pitch (F0)": np.mean(f0_values) if len(f0_values) > 0 else 0,
#         "Jitter": jitter,
#         "Shimmer": shimmer,
#         "Harmonic-to-Noise Ratio (HNR)": hnr
#     }

# Extract fundamental frequency (F0) using pyWorld
def extract_f0_pyworld(file_path):
    y, sr = sf.read(file_path)
    
    # WORLD expects mono audio
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    _f0, timeaxis = pw.dio(y, sr)  # Raw pitch extraction
    f0 = pw.stonemask(y, _f0, timeaxis, sr)  # Refined pitch estimation
    
    return {
        "WORLD Mean Pitch (F0)": np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
    }

# Main function
def analyze_voice(file_path):
    print(f"Analyzing: {file_path}\n")
    
    y, sr = load_audio(file_path)
    
    spectral_features = extract_spectral_features(y, sr)
    volume_features = extract_volume_features(y)
    # pitch_features = extract_pitch_harmonics(file_path)
    f0_pyworld = extract_f0_pyworld(file_path)

    # Combine all features
    all_features = {**spectral_features, **volume_features, **f0_pyworld}

    for key, value in all_features.items():
        print(f"{key}: {value}")

    return all_features

# Run analysis on an example file
audio_file = "Recording.m4a"  # Replace with your actual file
features = analyze_voice(audio_file)
