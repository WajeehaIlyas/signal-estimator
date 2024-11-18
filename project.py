import librosa
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt


audio_path = "input_audio.wav"
audio, sr = librosa.load(audio_path, sr=None)  # Loading with original sampling rate

# STFT parameters
n_fft = 2048 
hop_length = 512 


# Computing the STFT of input audio
D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
magnitude = np.abs(D)  # Magnitude spectrogram




def griffin_lim(magnitude, n_iterations, hop_length, win_length, fft_size):

    # Initialize a random phase
    phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, magnitude.shape))
    spectrogram = magnitude * phase
    
    for i in range(n_iterations):
        # Reconstruct the time-domain signal
        signal = librosa.istft(spectrogram, hop_length=hop_length, win_length=win_length, window='hann')
        
        # Compute STFT of the signal
        spectrogram = librosa.stft(signal, n_fft=fft_size, hop_length=hop_length, win_length=win_length, window='hann')
        
        # Replace magnitude while keeping phase
        spectrogram = magnitude * np.exp(1j * np.angle(spectrogram))
        
        # Debug: Check if phase changes
        if i % 10 == 0:
            print(f"Iteration {i}: Phase diff = {np.max(np.abs(np.angle(spectrogram) - np.angle(phase)))}")
        phase = np.angle(spectrogram)

    
    # Final reconstruction
    reconstructed_signal = librosa.istft(spectrogram, hop_length=hop_length, win_length=win_length, window='hann')
    return reconstructed_signal

# Parameters for Griffin-Lim
n_iterations = 50  # Number of iterations for Griffin-Lim
reconstructed_audio = griffin_lim(magnitude, n_iterations, hop_length, win_length=n_fft, fft_size=n_fft)

# Save the reconstructed audio
sf.write("reconstructed_audio.wav", reconstructed_audio, sr)

# Compute STFT of the reconstructed audio for visualization
reconstructed_D = librosa.stft(reconstructed_audio, n_fft=n_fft, hop_length=hop_length)

# Plot both original and reconstructed spectrograms
plt.figure(figsize=(14, 8))

# Original spectrogram
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title("Original Magnitude Spectrogram")
plt.colorbar(format='%+2.0f dB')

# Reconstructed spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(reconstructed_D), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title("Reconstructed Magnitude Spectrogram")
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()
