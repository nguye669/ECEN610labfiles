import numpy as np
import matplotlib.pyplot as plt
N = 8
f_clk = 2.4e9         
T_clk = 1.0 / f_clk   
T = N * T_clk        
C_H = 15.425e-12      
C_R = 0.5e-12        
f_min = 1e6           
f_max = 1e10          
num_points = 1000
f = np.logspace(np.log10(f_min), np.log10(f_max), num_points)
omega = 2 * np.pi * f
#    H(f) = [ 1 / ( j·2πf·(CH+CR) ) ] * [ 1 - e^(-jωT) ] / [ 1 - (CH/(CH+CR))· e^(-jωT) ]
num = (1 - np.exp(-1j * omega * T)) / (1j * omega * (C_H + C_R))
den = 1 - (C_H / (C_H + C_R)) * np.exp(-1j * omega * T)
H = num / den
H_mag_dB = 20 * np.log10(np.abs(H))
H_phase_deg = np.angle(H, deg=True)

plt.figure()
plt.title("Magnitude of H(f) [dB]")
plt.xscale('log')
plt.plot(f, H_mag_dB)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()
plt.figure()
plt.title("Phase of H(f) [degrees]")
plt.xscale('log')
plt.plot(f, H_phase_deg)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)
plt.show()
