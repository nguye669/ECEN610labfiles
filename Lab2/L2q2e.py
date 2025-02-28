import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
amplitude = 1           
sampling_rate = 400e6   
frequency = 200e6       
num_cycles = 30         
duration = num_cycles / frequency
time = np.arange(0, duration, 1/sampling_rate)

sine_wave = amplitude * np.cos(2 * np.pi * frequency * time + np.pi/6)
signal_power = np.mean(sine_wave**2) 
N_target = 6
quantization_levels_target = 2 ** N_target
Delta_target = 2 / quantization_levels_target  
quant_noise_power_target = (Delta_target ** 2) / 12
snr_target_db = 38
desired_total_noise_power = signal_power / (10**(snr_target_db/10))
gauss_noise_var = desired_total_noise_power - quant_noise_power_target
gauss_noise_std = np.sqrt(gauss_noise_var)

print(f"Using Gaussian noise with std = {gauss_noise_std:.4f} so that for N=6 (no window) SNR is ~{snr_target_db} dB\n")
for bits in [6, 12]:
    quantization_levels = 2 ** bits
    noise = np.random.normal(0, gauss_noise_std, size=time.shape)
    noisy_signal = sine_wave + noise
    quantized_signal = np.round(noisy_signal * (quantization_levels / 2)) / (quantization_levels / 2)
    error = quantized_signal - sine_wave
    noise_power = np.mean(error**2)
    snr_no_window = 10 * np.log10(signal_power / noise_power)
    window = np.hanning(len(sine_wave))
    noisy_signal_win = (sine_wave + noise) * window
    quantized_signal_win = np.round(noisy_signal_win * (quantization_levels / 2)) / (quantization_levels / 2)
    reference_win = sine_wave * window
    error_win = quantized_signal_win - reference_win
    signal_power_win = np.mean(reference_win**2)
    noise_power_win = np.mean(error_win**2)
    snr_window = 10 * np.log10(signal_power_win / noise_power_win)
    print(f"Quantizer Resolution N = {bits} bits")
    print(f"Without windowing: Simulated SNR = {snr_no_window:.2f} dB")
    print(f"With Hanning window: Simulated SNR = {snr_window:.2f} dB\n")
