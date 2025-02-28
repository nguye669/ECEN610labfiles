import numpy as np
import matplotlib.pyplot as plt

Fs      = 5e6       
f_sig   = 2e6       
A       = 1.0       
N       = 4096      
SNR_dB  = 50        
SNR_lin = 10**(SNR_dB / 10)
n = np.arange(N)
t = n / Fs
signal = A * np.sin(2 * np.pi * f_sig * t)
P_signal = 0.5
P_noise = P_signal / SNR_lin   
sigma   = np.sqrt(P_noise)
noise = sigma * np.random.randn(N)
x = signal + noise

def compute_windowed_psd_and_snr(x, window, Fs, guard_bins=10):
    x_w = x * window
    X_w = np.fft.fft(x_w)
    N   = len(x_w)
    U = np.sum(window**2)
    raw_psd_2sided = (1.0 / (U * Fs)) * np.abs(X_w)**2
    if N % 2 == 0:
        N_half = N//2 + 1
        psd_1sided = raw_psd_2sided[:N_half].copy()
        psd_1sided[1:-1] *= 2.0
    else:
        N_half = (N+1)//2
        psd_1sided = raw_psd_2sided[:N_half].copy()
        psd_1sided[1:] *= 2.0
    freqs_1sided = np.linspace(0, Fs/2, N_half)
    k_peak = np.argmax(psd_1sided)
    peak_power = psd_1sided[k_peak]
    low_guard  = max(0, k_peak - guard_bins)
    high_guard = min(N_half, k_peak + guard_bins + 1)
    all_indices = np.arange(N_half)
    noise_indices = np.concatenate((all_indices[:low_guard], all_indices[high_guard:]))

    noise_floor_est = np.mean(psd_1sided[noise_indices])
    SNR_lin_est = peak_power / noise_floor_est
    SNR_est_dB  = 10.0 * np.log10(SNR_lin_est)
    
    return freqs_1sided, psd_1sided, SNR_est_dB
windows = [
    ("Hanning", np.hanning(N)),
    ("Hamming", np.hamming(N)),
    ("Blackman", np.blackman(N)),
]

guard_bins = 1024
plt.figure(figsize=(8, 10))

for i, (win_name, w) in enumerate(windows, start=1):
    freqs, psd_1sided, snr_meas_dB = compute_windowed_psd_and_snr(
        x, w, Fs, guard_bins=guard_bins
    )
    plt.subplot(len(windows), 1, i)
    plt.semilogy(freqs, psd_1sided)
    plt.title(f"{win_name} Window - PSD (V^2/Hz)\nMeasured SNR = {snr_meas_dB:.2f} dB")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V^2/Hz]")
    plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Target SNR: {SNR_dB} dB")



