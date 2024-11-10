import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Compute STFT
def compute_stft(signal, frame_size=2048, hop_size=512):
    D = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size)
    return D

# Compute ISTFT (Inverse STFT)
def compute_istft(stft, hop_size=512):
    signal = librosa.istft(stft, hop_length=hop_size)
    return signal

def plot_stft(stft):
    magnitude = np.abs(stft)
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), x_axis='time', y_axis='log')
    plt.colorbar(label='Amplitude (dB)')
    plt.title('STFT Magnitude')
    plt.show()

def modify_stft_magnitude(stft, stretch_factor):
    # Extract magnitude and phase from STFT
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # The number of time frames (columns in the magnitude matrix)
    num_frames = magnitude.shape[1]
    
    # Compute new number of time frames after stretching
    new_num_frames = int(num_frames * stretch_factor)
    
    # Create a new magnitude array with interpolated values
    new_magnitude = np.zeros((magnitude.shape[0], new_num_frames))
    
    for freq_bin in range(magnitude.shape[0]):
        # Interpolate magnitude for each frequency bin independently
        new_magnitude[freq_bin, :] = np.interp(
            np.linspace(0, num_frames, new_num_frames),
            np.arange(num_frames),
            magnitude[freq_bin, :]
        )
    
    # Reconstruct the modified STFT by combining the new magnitude with the original phase
    modified_stft = new_magnitude * np.exp(1j * phase)
    
    return modified_stft



def lsee_mstftm(signal, target_stft, max_iter=100, learning_rate=0.01):
    # Initial STFT of the signal
    stft = compute_stft(signal)
    
    # Optimization loop (this is a basic example using gradient descent)
    for i in range(max_iter):
        # Compute the difference between the current and target STFT
        diff = np.abs(stft) - np.abs(target_stft)
        
        # Objective function: sum of squared differences
        loss = np.sum(np.square(diff))
        
        # Compute gradients (this is a simplified example)
        gradient = np.sign(diff)  # Simple gradient, use a more sophisticated one in practice
        
        # Update the STFT (here we're using a simple gradient step)
        stft -= learning_rate * gradient
        
        # Ensure the STFT stays in the correct domain (positive magnitudes)
        stft = np.maximum(stft, 0)
        
        print(f"Iteration {i}, Loss: {loss}")
        
    return stft

def overlap_add_reconstruction(stft, hop_size=512):
    # Reconstruct the time-domain signal from the modified STFT
    reconstructed_signal = compute_istft(stft, hop_size=hop_size)
    return reconstructed_signal

def time_scale_modify(signal, stretch_factor=1.5, frame_size=2048, hop_size=512):
    # Compute the STFT of the signal
    stft = compute_stft(signal, frame_size=frame_size, hop_size=hop_size)
    
    # Modify the STFT (for example, changing the magnitude)
    modified_stft = modify_stft_magnitude(stft, stretch_factor)
    
    # Optionally, apply LSEE-MSTFTM for better optimization
    # target_stft = some_target_stft # this could be a target signal's STFT
    # refined_stft = lsee_mstftm(signal, target_stft)  # refine the STFT iteratively
    
    # Reconstruct the signal from the modified STFT
    reconstructed_signal = overlap_add_reconstruction(modified_stft, hop_size=hop_size)
    
    return reconstructed_signal

def plot_signals(original_signal, modified_signal, sr):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original_signal, sr=sr)
    plt.title("Original Signal")
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(modified_signal, sr=sr)
    plt.title("Modified Signal (Time-Scaled)")
    
    plt.tight_layout()
    plt.show()


# Load an example audio signal
filename = "new.wav"
signal, sr = librosa.load(filename, sr=None)

# Apply time-scale modification
stretch_factor = 1.5  # Stretch by 1.5x (change as needed)
modified_signal = time_scale_modify(signal, stretch_factor=stretch_factor)

# Plot original vs modified signal
plot_signals(signal, modified_signal, sr)
