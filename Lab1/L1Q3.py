import numpy as np
import matplotlib.pyplot as plt


F = 2e6         
Fs = 5e6        
N = 64          
Ts = 1 / Fs     
t = np.arange(N) * Ts
x = np.cos(2 * np.pi * F * t)
X = np.fft.fft(x, N)
freq = np.fft.fftfreq(N, d=Ts)
X_shifted = np.fft.fftshift(X)
freq_shifted = np.fft.fftshift(freq)
plt.figure(figsize=(10, 6))
plt.stem(freq_shifted/1e6, np.abs(X_shifted), basefmt=" ")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.title('64-Point DFT of x(t) = cos(2π·2MHz·t) sampled at 5MHz')
plt.grid(True)
plt.show()



F1 = 200e6        
F2 = 400e6        
Fs = 1e9          
N = 64            
Ts = 1 / Fs       
t = np.arange(N) * Ts
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)
Y = np.fft.fft(y, N)
freq = np.fft.fftfreq(N, d=Ts)
Y_shifted = np.fft.fftshift(Y)
freq_shifted = np.fft.fftshift(freq)
plt.figure(figsize=(10, 6))
plt.stem(freq_shifted/1e6, np.abs(Y_shifted), basefmt=" ")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.title('64-Point DFT of $y(t)=\cos(2\pi\cdot200MHz\cdot t)+\cos(2\pi\cdot400MHz\cdot t)$\nSampled at 1 GHz')
plt.grid(True)
plt.show()


F1 = 200e6        
F2 = 400e6       
Fs = 500e6        
N = 64            
Ts = 1 / Fs       
t = np.arange(N) * Ts
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)
Y = np.fft.fft(y, N)
freq = np.fft.fftfreq(N, d=Ts)
Y_shifted = np.fft.fftshift(Y)
freq_shifted = np.fft.fftshift(freq)

plt.figure(figsize=(10, 6))
plt.stem(freq_shifted/1e6, np.abs(Y_shifted), basefmt=" ")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.title('64-Point DFT of $y(t)=\cos(2\pi\cdot200MHz\,t)+\cos(2\pi\cdot400MHz\,t)$ sampled at 500 MHz')
plt.grid(True)
plt.show()


F1 = 200e6     
F2 = 400e6      
Fs = 1e9        
N = 64          
Ts = 1 / Fs     
t = np.arange(N) * Ts
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)
Y = np.fft.fft(y, N)
freq = np.fft.fftfreq(N, Ts)
Y_shifted = np.fft.fftshift(Y)
freq_shifted = np.fft.fftshift(freq)
window = np.blackman(N)
y_windowed = y * window
Yw = np.fft.fft(y_windowed, N)
Yw_shifted = np.fft.fftshift(Yw)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.stem(freq_shifted/1e6, np.abs(Y_shifted), basefmt=" ")
plt.title("64-point DFT of y(t) (Rectangular Window) at Fs = 1 GHz")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.stem(freq_shifted/1e6, np.abs(Yw_shifted), basefmt=" ")
plt.title("64-point DFT of y(t) with Blackman Window at Fs = 1 GHz")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

F1 = 200e6      
F2 = 400e6      
Fs = 500e6      
N = 64          
Ts = 1 / Fs     
t = np.arange(N) * Ts
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)
Y = np.fft.fft(y, N)
freq = np.fft.fftfreq(N, Ts)
Y_shifted = np.fft.fftshift(Y)
freq_shifted = np.fft.fftshift(freq)
window = np.blackman(N)
y_windowed = y * window
Yw = np.fft.fft(y_windowed, N)
Yw_shifted = np.fft.fftshift(Yw)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.stem(freq_shifted/1e6, np.abs(Y_shifted), basefmt=" ")
plt.title("64-point DFT of y(t) (Rectangular Window) at Fs = 500 MHz")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.stem(freq_shifted/1e6, np.abs(Yw_shifted), basefmt=" ")
plt.title("64-point DFT of y(t) with Blackman Window at Fs = 500 MHz")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()
