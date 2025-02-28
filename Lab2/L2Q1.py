import numpy as np
import matplotlib.pyplot as plt

# Define signal parameters
f_tone = 2e6       # Tone frequency: 2 MHz
amplitude = 1.0    # Amplitude: 1 V
Fs = 5e6           # Sampling frequency: 5 MHz
duration = 5e-6    # Duration in seconds (5 microseconds)

# Generate the continuous sine wave (using fine resolution)
t_continuous = np.linspace(0, duration, 1000)
continuous_tone = amplitude * np.sin(2 * np.pi * f_tone * t_continuous)

# Generate the sampled points (discrete samples)
t_sampled = np.arange(0, duration, 1/Fs)
sampled_tone = amplitude * np.sin(2 * np.pi * f_tone * t_sampled)

# Create the plot with both signals
plt.figure(figsize=(8, 4))

# Plot the continuous sine wave
plt.plot(t_continuous * 1e6, continuous_tone, label="Continuous Sine Wave", color="blue")

# Overlay the sampled points (displayed as red markers only)
plt.plot(t_sampled * 1e6, sampled_tone, 'ro', label="Sampled Points")

plt.xlabel("Time (µs)")
plt.ylabel("Amplitude (V)")
plt.title("2 MHz Sine Wave: Continuous Signal with Sampled Points at 5 MHz")
plt.legend()
plt.grid(True)
plt.show()





