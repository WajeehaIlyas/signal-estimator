import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

#1. Load the audio file
original_audio_path = 'original_audio.wav'

signal, sr = librosa.load('original_audio.wav', sr=None)

def load_audio():
    signal, sr = librosa.load('original_audio.wav', sr=None)
    return signal, sr

def plot_signal(signal):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.show()

def main():
    signal, sr = load_audio()
    plot_signal(signal)
    
if __name__ == '__main__':
    main()


#3. Modify magnitude or phase

#4. Initialize signal estimation

#5. Compute inverse STFT

#6. Griffin-lim's algo for iterative phase and magnitude estimation

#7. Check convergence

#8. Plot final output