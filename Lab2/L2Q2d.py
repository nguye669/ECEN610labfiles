import numpy as np
import matplotlib.pyplot as plt

for resolution in [6, 12]:
    bits = resolution
    amplitude = 1           
    sampling_rate = 900e6   
    frequency = 200e6       
    num_cycles = 30         
    duration = num_cycles / frequency
    time = np.arange(0, duration, 1/sampling_rate)
    
    # Generate the sine wave
    sine_wave = amplitude * np.cos(2 * np.pi * frequency * time)
    
    # Quantization without windowing
    quantization_levels = 2 ** bits
    quantized_wave = np.round(sine_wave * (quantization_levels / 2)) / (quantization_levels / 2)
    error = quantized_wave - sine_wave
    signal_power = np.mean(sine_wave**2)
    noise_power = np.mean(error**2)
    snr_no_window = 10 * np.log10(signal_power / noise_power)
    snr_theoretical = 6.02 * bits + 1.76
    
    # Apply a Hanning window
    window = np.hanning(len(sine_wave))
    sine_wave_windowed = sine_wave * window
    
    # Quantization with windowing
    quantized_wave_windowed = np.round(sine_wave_windowed * (quantization_levels / 2)) / (quantization_levels / 2)
    error_windowed = quantized_wave_windowed - sine_wave_windowed
    signal_power_windowed = np.mean(sine_wave_windowed**2)
    noise_power_windowed = np.mean(error_windowed**2)
    snr_window = 10 * np.log10(signal_power_windowed / noise_power_windowed)
    
    # Print the results
    print(f"Resolution N = {bits}")
    print(f"Without windowing: Simulated SNR = {snr_no_window:.2f} dB, Theoretical SNR = {snr_theoretical:.2f} dB")
    print(f"With Hanning window: Simulated SNR = {snr_window:.2f} dB\n")
