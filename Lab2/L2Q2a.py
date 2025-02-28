import numpy as np
import matplotlib.pyplot as plt

bits = 6
amplitude = 1           
sampling_rate = 450e6   
frequency = 200e6       
num_cycles = 30         


duration = num_cycles / frequency

time = np.arange(0, duration, 1/sampling_rate)
sine_wave = amplitude * np.cos(2 * np.pi * frequency * time+np.pi/8)
quantization_levels = 2 ** bits
quantized_wave = np.round(sine_wave * (quantization_levels / 2)) / (quantization_levels / 2)
error = quantized_wave - sine_wave
signal_power = np.mean(sine_wave**2)
noise_power = np.mean(error**2)
snr_simulated = 10 * np.log10(signal_power / noise_power)
snr_theoretical = 6.02 * bits + 1.76

print(f"Simulated SNR: {snr_simulated:.2f} dB")
print(f"Theoretical SNR: {snr_theoretical:.2f} dB")

fft_result = np.fft.fft(quantized_wave)
N = len(fft_result)
frequencies = np.fft.fftfreq(N, d=1/sampling_rate)
psd = (np.abs(fft_result)**2) / (sampling_rate * N)
psd_db = 10 * np.log10(psd + 1e-20)

plt.figure(figsize=(8, 6))
plt.stem(frequencies, psd_db, basefmt=" ")
plt.title("Power Spectral Density (PSD)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB/Hz)")
plt.grid()
plt.tight_layout()
plt.show()
