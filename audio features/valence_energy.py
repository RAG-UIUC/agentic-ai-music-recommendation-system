"""
valence_energy.py
Extracts valence and energy features from audio files using librosa
"""

import librosa
import numpy as np
import os
import json

def extract_valence_energy(audio_path):
    """
    Extract valence and energy from audio.
    
    Valence (0.0-1.0): Musical positiveness. High = happy/cheerful, low = sad/angry
    Energy (0.0-1.0): Intensity and activity. High = fast/loud/noisy, low = calm
    """
    y, sr = librosa.load(audio_path, duration=30)
    
    # ENERGY FEATURES
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Dynamic range
    dynamic_range = np.max(rms) - np.min(rms)
    
    # Onset rate (number of note attacks per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = len(onset_frames) / librosa.get_duration(y=y, sr=sr)
    
    # VALENCE FEATURES
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Major vs minor
    major_score = chroma_mean[4]
    minor_score = chroma_mean[3]
    mode_score = major_score - minor_score
    
    # Spectral contrast (timbral texture)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # Harmonic vs percussive ratio
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y)) + 1e-6)
    
    # CALCULATE ENERGY (0-1)
    energy = (
        (rms * 2.5) +
        (spectral_centroid / 4000) +
        (dynamic_range * 2) +
        (onset_rate / 10)
    ) / 4
    energy = np.clip(energy, 0, 1)
    
    # CALCULATE VALENCE (0-1)
    valence = (
        ((mode_score + 1) / 2) * 0.6 +
        (spectral_contrast / 50) * 0.2 +
        (harmonic_ratio) * 0.2
    )
    valence = np.clip(valence, 0, 1)
    
    return {
        'energy': float(energy),
        'valence': float(valence)
    }


def process_single_file(audio_path):
    """Process single audio file"""
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return None
    
    features = extract_valence_energy(audio_path)
    print(f"\n{os.path.basename(audio_path)}")
    print(f"Energy: {features['energy']:.3f}")
    print(f"Valence: {features['valence']:.3f}")
    return features


def process_directory(directory_path, output_file=None):
    """Process all audio files in directory"""
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    results = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.splitext(filename)[1].lower() in audio_extensions:
            features = extract_valence_energy(file_path)
            features['filename'] = filename
            results.append(features)
    
    if output_file and results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Example usage
    # features = process_single_file("song.mp3")
    # results = process_directory("music_folder", "features.json")
    pass