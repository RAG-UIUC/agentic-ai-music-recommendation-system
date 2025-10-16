#!/usr/bin/env python3
"""
analyze_track.py

Usage:
    python analyze_track.py path/to/song.mp3

Outputs a JSON-like print of extracted traits:
- tempo_bpm
- tempo_confidence (from librosa)
- tempo_var_bpm (instantaneous tempo variability)
- onset_rate (onsets per second)
- rms_mean, rms_std
- spectral_centroid_mean
- zcr_mean (zero-crossing rate)
- liveliness_score (0.0 - 1.0)
"""

import sys
import json
import numpy as np
import librosa


def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def estimate_tempo(y, sr):
    # Use librosa beat tracker to get a global tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    # Compute beat times and instantaneous BPMs from inter-beat intervals
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    ibi = np.diff(beat_times)  # inter-beat intervals in seconds
    with np.errstate(divide='ignore', invalid='ignore'):
        inst_bpm = 60.0 / ibi
    inst_bpm = inst_bpm[np.isfinite(inst_bpm)]
    tempo_var = float(np.std(inst_bpm)) if inst_bpm.size > 0 else 0.0
    tempo_median = float(np.median(inst_bpm)) if inst_bpm.size > 0 else float(tempo)
    return {
        "tempo_bpm": float(tempo),
        "tempo_median_bpm": tempo_median,
        "tempo_var_bpm": tempo_var,
        "beat_count": int(max(0, len(beat_frames)))
    }


def onset_and_rate(y, sr):
    # onset strength envelope and onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # onset frames
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    duration = len(y) / sr
    onset_rate = float(len(onset_times) / duration) if duration > 0 else 0.0
    onset_env_mean = float(np.mean(onset_env)) if onset_env.size else 0.0
    onset_env_std = float(np.std(onset_env)) if onset_env.size else 0.0
    return {
        "onset_count": int(len(onset_times)),
        "onset_rate_per_sec": onset_rate,
        "onset_env_mean": onset_env_mean,
        "onset_env_std": onset_env_std
    }


def spectral_and_energy(y, sr, hop_length=512):
    # RMS (energy), spectral centroid, zcr
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr))
    }


def normalize(x, lo, hi):
    """Clamp and scale to 0..1 given expected lo..hi"""
    if hi == lo:
        return 0.0
    x = float(x)
    s = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, s)))


def compute_liveliness(features):
    """
    Heuristic liveliness score (0..1).
    Weighted combination of:
     - tempo (mapped from 40-200 BPM)
     - onset rate (mapped from 0-10 onsets/sec)
     - energy (rms_mean)
     - spectral centroid (brighter = more lively)
    Tune weights as needed.
    """
    tempo = features.get("tempo_bpm", 0.0)
    onset_rate = features.get("onset_rate_per_sec", 0.0)
    rms_mean = features.get("rms_mean", 0.0)
    centroid = features.get("spectral_centroid_mean", 0.0)

    # Normalizations: choose reasonable empirical ranges
    tempo_n = normalize(tempo, 40.0, 200.0)          # slow -> fast
    onset_n = normalize(onset_rate, 0.0, 10.0)      # quiet -> packed
    rms_n = normalize(rms_mean, 0.001, 0.1)         # adjust if your recordings are loud/quiet
    centroid_n = normalize(centroid, 500.0, 6000.0) # low -> bright

    # Weights (feel free to change)
    w_tempo = 0.35
    w_onset = 0.30
    w_rms = 0.20
    w_centroid = 0.15

    score = (w_tempo*tempo_n + w_onset*onset_n + w_rms*rms_n + w_centroid*centroid_n)
    return {
        "liveliness_score": float(max(0.0, min(1.0, score))),
        "components": {
            "tempo_norm": tempo_n,
            "onset_norm": onset_n,
            "rms_norm": rms_n,
            "centroid_norm": centroid_n
        },
        "weights": {
            "tempo": w_tempo,
            "onset_rate": w_onset,
            "rms": w_rms,
            "spectral_centroid": w_centroid
        }
    }


def extract_traits(path):
    y, sr = load_audio(path)
    tempo_info = estimate_tempo(y, sr)
    onset_info = onset_and_rate(y, sr)
    spectral_info = spectral_and_energy(y, sr)

    features = {}
    features.update(tempo_info)
    features.update(onset_info)
    features.update(spectral_info)

    liveliness = compute_liveliness(features)
    features.update(liveliness)

    # Add some human-friendly derived items
    duration = float(len(y) / sr)
    features["duration_sec"] = duration

    # Tempo confidence approximation: if tempo_var small => confident
    tempo_var = features.get("tempo_var_bpm", 0.0)
    tempo_conf = float(max(0.0, min(1.0, 1.0 - normalize(tempo_var, 0.0, 15.0))))
    features["tempo_confidence"] = tempo_conf

    return features


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_track.py path/to/song.mp3")
        sys.exit(1)

    path = sys.argv[1]
    print("Analyzing:", path)
    try:
        features = extract_traits(path)
    except Exception as e:
        print("Error analyzing file:", e)
        sys.exit(2)

    # Pretty print
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
