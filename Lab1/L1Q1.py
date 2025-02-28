import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter
b_FIR = [1, 1, 1, 1, 1]
a_FIR = [1]
b_IIR = [1, 1]    
a_IIR = [1, -1]   
w = np.linspace(1e-6, np.pi, 8000)
w_FIR, h_FIR = freqz(b_FIR, a_FIR, worN=w)
w_IIR, h_IIR = freqz(b_IIR, a_IIR, worN=w)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
ax1 = axs[0]
ax1.plot(w_FIR / np.pi, 20 * np.log10(np.abs(h_FIR)), label='FIR: $H_{FIR}(z)$')
ax1.plot(w_IIR / np.pi, 20 * np.log10(np.abs(h_IIR)), label='IIR: $H_{IIR}(z)$', linestyle='--')
zeros_FIR = np.roots(b_FIR)
fir_zero_angles_FIR = np.abs(np.angle(zeros_FIR))
fir_zero_angles_FIR = fir_zero_angles_FIR[fir_zero_angles_FIR <= np.pi]
fir_zero_angles_FIR = np.unique(fir_zero_angles_FIR)

for angle in fir_zero_angles_FIR:
    idx = np.argmin(np.abs(w_FIR - angle))
    mag_db = 20 * np.log10(np.abs(h_FIR[idx]))
    ax1.plot(w_FIR[idx] / np.pi, mag_db, 'ko', markersize=8, markerfacecolor='yellow')
    ax1.annotate(f'FIR Zero\nω={angle:.2f} rad',
                 xy=(w_FIR[idx] / np.pi, mag_db),
                 xytext=(w_FIR[idx] / np.pi, mag_db + 8),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9)

iir_pole_angle = 0.0
idx = np.argmin(np.abs(w_IIR - iir_pole_angle))
mag_db = 20 * np.log10(np.abs(h_IIR[idx]))
ax1.plot(w_IIR[idx] / np.pi, mag_db, 'ks', markersize=8, markerfacecolor='red')
ax1.annotate(f'IIR Pole\nω={iir_pole_angle:.2f} rad',
             xy=(w_IIR[idx] / np.pi, mag_db),
             xytext=(w_IIR[idx] / np.pi, mag_db - 15),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9)

iir_zero_angle = np.pi
idx = np.argmin(np.abs(w_IIR - iir_zero_angle))
mag_db = 20 * np.log10(np.abs(h_IIR[idx]))
ax1.plot(w_IIR[idx] / np.pi, mag_db, 'kd', markersize=8, markerfacecolor='cyan')
ax1.annotate(f'IIR Zero\nω={iir_zero_angle:.2f} rad',
             xy=(w_IIR[idx] / np.pi, mag_db),
             xytext=(w_IIR[idx] / np.pi, mag_db + 8),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=9)

ax1.set_xlabel('Normalized Frequency (×π rad/sample)')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Frequency Response with Pole/Zero Locations')
ax1.legend()
ax1.grid(True)
ax2 = axs[1]
theta = np.linspace(0, 2 * np.pi, 400)
unit_circle = np.exp(1j * theta)
ax2.plot(np.real(unit_circle), np.imag(unit_circle), 'k--', label='Unit Circle')

ax2.plot(np.real(zeros_FIR), np.imag(zeros_FIR), 'bo', markersize=8, label='FIR Zeros')
ax2.plot(0, 0, 'rx', markersize=10, label='FIR Delay Pole')
zeros_IIR = np.roots(b_IIR)
poles_IIR = np.roots(a_IIR)
ax2.plot(np.real(zeros_IIR), np.imag(zeros_IIR), 'cs', markersize=8, label='IIR Zero')
ax2.plot(np.real(poles_IIR), np.imag(poles_IIR), 'mx', markersize=10, label='IIR Pole')
ax2.set_xlabel('Real Part')
ax2.set_ylabel('Imaginary Part')
ax2.set_title('Pole–Zero Plot')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')
plt.tight_layout()
plt.show()

N = 50
impulse = np.zeros(N)
impulse[0] = 1
h_FIR_impulse = lfilter(b_FIR, a_FIR, impulse)
h_IIR_impulse = lfilter(b_IIR, a_IIR, impulse)
plt.figure(figsize=(10, 4))
plt.stem(np.arange(N), h_FIR_impulse, basefmt=" ", label="FIR Impulse Response")
plt.stem(np.arange(N), h_IIR_impulse, basefmt=" ", linefmt='r-', markerfmt='ro', label="IIR Impulse Response")
plt.xlabel('n (samples)')
plt.ylabel('Amplitude')
plt.title('Impulse Responses of FIR and IIR Filters')
plt.legend()
plt.grid(True)
plt.show()

