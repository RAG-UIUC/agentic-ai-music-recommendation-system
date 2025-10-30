"""
Test valence and energy extraction
"""
from valence_energy import extract_valence_energy
import sys

def test_extraction():
    audio_file = input("Enter path to audio file: ").strip().strip('"').strip("'")
    
    try:
        features = extract_valence_energy(audio_file)
        print(f"\nEnergy: {features['energy']:.3f}")
        print(f"Valence: {features['valence']:.3f}")
        print("\nSuccess!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_extraction()