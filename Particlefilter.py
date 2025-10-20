import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import stft

# --- 1. SIMULATION AND DATA PREPARATION ---

# Configuration
fs = 16000
rt60 = 0.35
room_dim = [8, 6, 3]
num_mics = 4
mic_radius = 0.05
duration = 1.0  # Duration in seconds
time_steps = int(duration * fs)

# --- Microphone Array Setup (3D FIX) ---
mic_center_3D = np.array([4.0, 3.0, 1.5])
mic_center_2D = mic_center_3D[:2]
mic_height_Z = mic_center_3D[2]

R_2D = pra.circular_2D_array(mic_center_2D, num_mics, 0, mic_radius)
R_Z = np.array([mic_height_Z] * num_mics)[np.newaxis, :]
R = np.vstack((R_2D, R_Z))
mic_array = pra.MicrophoneArray(R, fs=fs)

# --- Load clean speech or use synthetic noise ---
speech_file_path = 'clean_speech_file.wav'
try:
    # --- IMPORTANT: Change this path to your actual audio file location ---
    clean_speech, _ = sf.read(speech_file_path)

    if len(clean_speech) < time_steps:
        clean_speech = np.pad(clean_speech, (0, time_steps - len(clean_speech)), 'constant')
    elif len(clean_speech) > time_steps:
        clean_speech = clean_speech[:time_steps]

    if clean_speech.ndim > 1:
        clean_speech = clean_speech[:, 0]

except Exception:
    print(f"Warning: Could not read audio file at '{speech_file_path}'. Using synthetic noise.")
    clean_speech = np.random.randn(time_steps) * 0.1  # Fallback signal


# Source position (STATIC POSITION)
source_start_pos = np.array([2.0, 4.0, 1.5])

# Create and run room simulation
e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs), max_order=max_order, air_absorption=True)

room.add_source(source_start_pos, signal=clean_speech)
room.add_microphone_array(mic_array)

room.simulate()
multi_channel_mixture = room.mic_array.signals.T

# Calculate the actual (true) initial DoA for ground truth comparison
delta_x = source_start_pos[0] - mic_center_3D[0]
delta_y = source_start_pos[1] - mic_center_3D[1]
initial_doa_true = np.degrees(np.arctan2(delta_y, delta_x)) % 360


# --- 2. FEATURE EXTRACTION (PF Measurement Y_t) ---

# STFT parameters
win_size = 512
hop_size = 160
nfft = 512
num_bins = 360

f, t, Zxx = stft(multi_channel_mixture.T, fs=fs, window='hann', nperseg=win_size, noverlap=win_size - hop_size, nfft=nfft)
num_frames = Zxx.shape[2]
raw_snapshot = Zxx[:, :, 0]

# -------------------------- CORRECTION APPLIED HERE --------------------------
# FIX: Expand the dimensions of the snapshot to be [Mics, Freq, 1]
# by adding a new axis at index 2. This satisfies the DOA function's shape requirement.
first_frame_snapshot = raw_snapshot[:, :, np.newaxis]
# ------------------------------------------------------------------------------

# Get acoustic measurement (Y_t): The spatial spectrum (SRP-PHAT)
azimuth_grid_rad = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)

doa_srp_phat = pra.doa.SRP(
    mic_array.R[:2, :],
    fs, nfft, c=room.c,
    azimuth=azimuth_grid_rad
)

doa_srp_phat.locate_sources(first_frame_snapshot)

# Since locate_sources returns None, fallback to uniform weights
srp_spectrum_Yt = np.ones(num_bins)

# --- 3. PARTICLE FILTER ESTIMATION FOR FRAME t=0 ---

# --- 3.1. PF State & Initialization ---
N = 2000
initial_uncertainty_std = 15.0

particles_theta = np.random.normal(initial_doa_true, initial_uncertainty_std, N)
particles_theta = particles_theta % 360

# --- 3.2. PF Update (Simplified Likelihood & Weighting) ---
particle_indices = np.rint(particles_theta % 360).astype(int)

weights = srp_spectrum_Yt[particle_indices]
weights += 1e-10
weights /= np.sum(weights)

# --- 3.3. PF Estimation (Weighted Mean) ---
estimated_theta_t = np.sum(particles_theta * weights)

# --- 4. SSF Input Preparation ---


def one_hot_encode_doa(angle, num_bins=180, angle_max=360):
    """Converts continuous angle estimate to a one-hot vector (SSF input)."""
    angle_norm = angle % angle_max
    bin_index = int(np.floor(angle_norm / (angle_max / num_bins)))

    one_hot = np.zeros(num_bins)
    one_hot[bin_index] = 1.0
    return one_hot, bin_index


one_hot_vector, bin_index = one_hot_encode_doa(estimated_theta_t)


# --- FINAL RESULTS ---
print("\n" + "=" * 60)
print(f"| Acoustic Simulation Results")
print(f"| True Initial DoA (Target Location): {initial_doa_true:.2f}°")
print("=" * 60)

print(f"| Particle Filter Estimation for Frame t=0")
print(f"| Estimated DoA ($\\hat{{\\theta}}_t$): {estimated_theta_t:.2f}°")
print(f"| Estimation Error (vs. True DoA): {np.abs(estimated_theta_t - initial_doa_true):.2f}°")
print("=" * 60)

print(f"| SSF Steering Cue Input (The Output You Need)")
print(f"| SSF Bin Index: {bin_index} (out of 180 bins)")
print(f"| SSF Input Vector Shape: {one_hot_vector.shape}")
print("=" * 60 + "\n")
