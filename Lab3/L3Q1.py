import numpy as np
import matplotlib.pyplot as plt
N = 8
f_clk = 2.4e9         
T_clk = 1.0 / f_clk   
T = N * T_clk         
C = 15.925e-12        
f = np.logspace(6, 10, 1000) 
omega = 2.0 * np.pi * f
H_a = (1.0 / C) * (1.0 - np.exp(-1j * omega * T)) / (1j * omega)
H_b = 1.0 / (1j * omega * C)
H_a_mag_dB = 20.0 * np.log10(np.abs(H_a))
H_b_mag_dB = 20.0 * np.log10(np.abs(H_b))

plt.figure()
plt.title("Magnitude of Transfer Functions (Case a vs. Case b)")
plt.xscale("log")
plt.plot(f, H_a_mag_dB, label="|H_a(f)|, reset each time")
plt.plot(f, H_b_mag_dB, label="|H_b(f)|, never reset")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.legend()
plt.grid(True)
plt.show()
