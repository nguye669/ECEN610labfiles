import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

def rc_filter(x, tau, Ts):
    a = np.exp(-Ts / tau)
    b = 1 - a
    y = signal.lfilter([b], [1, -a], x)
    return y

def fractional_delay(x, frac_delay):
    n = np.arange(len(x))
    n_shifted = n - frac_delay
    x_delayed = np.interp(n_shifted, n, x, left=x[0], right=x[-1])
    return x_delayed

def compute_sndr(x, fs, f_sig):
    N = len(x)
    window = np.hanning(N)
    xw = x * window
    X = fft(xw)
    X = X[:N//2]  
    X_mag = np.abs(X)**2
    freqs = fftfreq(N, 1/fs)[:N//2]

    idx = np.argmin(np.abs(freqs - f_sig))
    signal_power = X_mag[idx]
    noise_power = np.sum(X_mag) - signal_power
    sndr = 10 * np.log10(signal_power / noise_power)
    return sndr, freqs, X_mag
fs_total = 10e9              
Ts_total = 1 / fs_total      
N_total = 4096               
t_total = np.arange(N_total) * Ts_total  
fs_channel = fs_total / 2   
Ts_channel = 1 / fs_channel
f_in = 1e9
A_in = 0.5
x_cont = A_in * np.sin(2 * np.pi * f_in * t_total)
tau_A = 10e-12             
t_A = t_total[0::2]
x_A_samples = A_in * np.sin(2 * np.pi * f_in * t_A)
chanA = rc_filter(x_A_samples, tau_A, Ts_channel)
tau_B = 40e-12             
dt_error = 30e-12          
offset_error = 0.2         
t_B = t_total[1::2] + dt_error
x_B_samples = A_in * np.sin(2 * np.pi * f_in * t_B)
chanB = rc_filter(x_B_samples, tau_B, Ts_channel)
chanB = chanB + offset_error
adc_pre = np.empty(N_total)
adc_pre[0::2] = chanA
adc_pre[1::2] = chanB
LSB = 1 / 128.0
adc_pre_quant = np.round(adc_pre / LSB) * LSB
sndr_pre, freqs_pre, Xmag_pre = compute_sndr(adc_pre_quant, fs_total, f_in)
print(f"SNDR before calibration: {sndr_pre:.2f} dB")

offset_est = np.mean(chanB)  
chanB_cal = chanB - offset_est
frac_delay = dt_error / Ts_channel  
chanB_cal = fractional_delay(chanB_cal, frac_delay)
num_taps = 21
f_grid = np.linspace(0, fs_channel/2, 512)
H_A = 1 / (1 + 1j * 2 * np.pi * f_grid * tau_A)
H_B = 1 / (1 + 1j * 2 * np.pi * f_grid * tau_B)
H_eq_desired = np.abs(H_A / H_B)
f_norm = f_grid / (fs_channel/2)
eq_filter = signal.firwin2(num_taps, f_norm, H_eq_desired)
chanB_cal_eq = signal.lfilter(eq_filter, [1.0], chanB_cal)
adc_post = np.empty(N_total)
adc_post[0::2] = chanA
adc_post[1::2] = chanB_cal_eq
adc_post_quant = np.round(adc_post / LSB) * LSB
sndr_post, freqs_post, Xmag_post = compute_sndr(adc_post_quant, fs_total, f_in)
print(f"SNDR after calibration: {sndr_post:.2f} dB")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(freqs_pre/1e9, 10*np.log10(Xmag_pre+1e-12))
plt.title('Spectrum Before Calibration')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(freqs_post/1e9, 10*np.log10(Xmag_post+1e-12))
plt.title('Spectrum After Calibration')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 4))
plt.plot(t_total*1e9, adc_pre_quant, 'o-', label='Pre-calibration')
plt.plot(t_total*1e9, adc_post_quant, 'o-', label='Post-calibration')
plt.title('TI-ADC Output Waveforms')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)
plt.show()
