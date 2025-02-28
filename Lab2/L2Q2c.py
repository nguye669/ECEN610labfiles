import numpy as np
import matplotlib.pyplot as plt

for resolution in [6,12]:
    bits = resolution
    amplitude = 1           
    sampling_rate = 400e6   
    frequency = 200e6       
    num_cycles = 30         
    duration = num_cycles / frequency
    time = np.arange(0, duration, 1/sampling_rate)
    sine_wave = amplitude * np.cos(2 * np.pi * frequency * time+np.pi/6)
    quantization_levels = 2 ** bits
    quantized_wave = np.round(sine_wave * (quantization_levels / 2)) / (quantization_levels / 2)
    error = quantized_wave - sine_wave
    signal_power = np.mean(sine_wave**2)
    noise_power = np.mean(error**2)
    snr_simulated = 10 * np.log10(signal_power / noise_power)
    snr_theoretical = 6.02 * bits + 1.76
    print(f"for quantizer resolution N = {bits}")
    print(f"Simulated SNR for: {snr_simulated:.2f} dB")
    print(f"Theoretical SNR: {snr_theoretical:.2f} dB")
