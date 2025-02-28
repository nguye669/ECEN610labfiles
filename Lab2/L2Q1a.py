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
P_noise = P_signal / SNR_lin  # e.g. 0.5 / 1e5 = 5e-6
sigma   = np.sqrt(P_noise)
noise = sigma * np.random.randn(N)
x = signal + noise
Xf = np.fft.fft(x)
N_half = N // 2 + 1  
raw_psd_2sided = (1.0 / (N * Fs)) * np.abs(Xf)**2
psd_1sided = np.copy(raw_psd_2sided[:N_half])
psd_1sided[1:N_half-1] *= 2.0
freqs_pos = np.linspace(0, Fs/2, N_half)

plt.figure(figsize=(8,5))
plt.semilogy(freqs_pos, psd_1sided, label='Noisy Sine PSD (1-sided)')
plt.title("PSD in V^2/Hz (One-Sided)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V^2/Hz]")
plt.grid(True)
plt.legend()
plt.show()

k_peak = np.argmax(psd_1sided)
peak_power = psd_1sided[k_peak]
guard_bins = 15  # <--- Adjust this to get an SNR close to 50 dB
all_indices = np.arange(N_half)
low_guard   = max(0, k_peak - guard_bins)
high_guard  = min(N_half, k_peak + guard_bins + 1)
noise_indices = np.concatenate((all_indices[:low_guard], all_indices[high_guard:]))

noise_floor_est = np.mean(psd_1sided[noise_indices])
SNR_est_lin = peak_power / noise_floor_est
SNR_est_dB  = 10 * np.log10(SNR_est_lin)

print(f"Requested SNR: {SNR_dB:.2f} dB")
print(f"Estimated SNR from FFT: {SNR_est_dB:.2f} dB (guard_bins={guard_bins})")


