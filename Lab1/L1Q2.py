import numpy as np
import matplotlib.pyplot as plt

F1 = 300e6  
F2 = 800e6  
Fs = 500e6  
t_cont = np.linspace(0, 20e-9, 2000)  
x1_cont = np.cos(2 * np.pi * F1 * t_cont)
x2_cont = np.cos(2 * np.pi * F2 * t_cont)
n = np.arange(0, 11)  
t_samples = n / Fs 
x1_disc = np.cos(2 * np.pi * F1 * t_samples)
x2_disc = np.cos(2 * np.pi * F2 * t_samples)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t_cont * 1e9, x1_cont, label=r'$x_1(t)=\cos(2\pi\cdot300\mathrm{MHz}\,t)$', color='blue')
axs[0].plot(t_cont * 1e9, x2_cont, label=r'$x_2(t)=\cos(2\pi\cdot800\mathrm{MHz}\,t)$', color='green')
axs[0].set_title('Continuous-Time Signals')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[0].grid(True)

axs[1].stem(t_samples * 1e9, x1_disc, linefmt='b-', markerfmt='bo', basefmt='r-', label=r'$x_1(n)$')
axs[1].stem(t_samples * 1e9, x2_disc, linefmt='g--', markerfmt='go', basefmt='r-', label=r'$x_2(n)$')
axs[1].set_title('Discrete-Time Sampled Signals')
axs[1].set_xlabel('Time (ns)')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()
plt.show()

F1 = 300e6          
Fs = 800e6          
Ts = 1 / Fs        
T = 10 / F1        
t_dense = np.linspace(0, T, 1000)
x_original = np.cos(2 * np.pi * F1 * t_dense)
t_samples = np.arange(0, T, Ts)
x_samples = np.cos(2 * np.pi * F1 * t_samples)
xr_nonshift = np.zeros_like(t_dense)
for n, t_n in enumerate(t_samples):
    xr_nonshift += x_samples[n] * np.sinc((t_dense - t_n) / Ts)
mse_nonshift = np.mean((xr_nonshift - x_original)**2)
t_samples_shift = np.arange(Ts/2, T, Ts)
x_samples_shift = np.cos(2 * np.pi * F1 * t_samples_shift)
xr_shift = np.zeros_like(t_dense)
for n, t_n in enumerate(t_samples_shift):
    xr_shift += x_samples_shift[n] * np.sinc((t_dense - t_n) / Ts)
mse_shift = np.mean((xr_shift - x_original)**2)

print("MSE for non-shifted sampling:", mse_nonshift)
print("MSE for shifted sampling:", mse_shift)

fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axs[0].plot(t_dense * 1e9, x_original, 'b', label="Original x1(t)")
axs[0].set_title('Original Continuous-Time Signal $x_1(t)$')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_dense * 1e9, xr_nonshift, 'g--', label="Reconstructed (non-shifted)")
axs[1].stem(t_samples * 1e9, x_samples, linefmt='r-', markerfmt='ro', basefmt='r-', label="Samples")
axs[1].set_title('Non-Shifted Sampling Reconstruction')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t_dense * 1e9, xr_shift, 'm--', label="Reconstructed (shifted)")
axs[2].stem(t_samples_shift * 1e9, x_samples_shift, linefmt='k-', markerfmt='ko', basefmt='k-', label="Shifted Samples")
axs[2].set_title('Shifted Sampling Reconstruction')
axs[2].set_xlabel('Time (ns)')
axs[2].set_ylabel('Amplitude')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()


F1 = 300e6           
T = 10 / F1          
t_dense = np.linspace(0, T, 1000)  
x_original = np.cos(2 * np.pi * F1 * t_dense)  

sampling_freqs = [1e9, 500e6]

for Fs in sampling_freqs:
    Ts = 1 / Fs  # Sampling interval
    t_samples = np.arange(0, T, Ts)
    x_samples = np.cos(2 * np.pi * F1 * t_samples)
    xr_nonshift = np.zeros_like(t_dense)
    for n, t_n in enumerate(t_samples):
        xr_nonshift += x_samples[n] * np.sinc((t_dense - t_n) / Ts)
    mse_nonshift = np.mean((xr_nonshift - x_original)**2)
    t_samples_shift = np.arange(Ts/2, T, Ts)
    x_samples_shift = np.cos(2 * np.pi * F1 * t_samples_shift)
    xr_shift = np.zeros_like(t_dense)
    for n, t_n in enumerate(t_samples_shift):
        xr_shift += x_samples_shift[n] * np.sinc((t_dense - t_n) / Ts)
    mse_shift = np.mean((xr_shift - x_original)**2)
    
    print("Sampling frequency Fs = {:.0f} MHz".format(Fs/1e6))
    print("MSE for non-shifted sampling:", mse_nonshift)
    print("MSE for shifted sampling:", mse_shift)
    print("-" * 50)
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Reconstruction for Fs = {:.0f} MHz".format(Fs/1e6), fontsize=16)
    axs[0].plot(t_dense * 1e9, x_original, 'b', label="x1(t)")
    axs[0].set_title('Original Continuous-Time Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(t_dense * 1e9, xr_nonshift, 'g--', label="Reconstructed (non-shifted)")
    axs[1].stem(t_samples * 1e9, x_samples, linefmt='r-', markerfmt='ro', basefmt='r-', label="Samples")
    axs[1].set_title('Non-Shifted Sampling Reconstruction\nMSE = {:.3e}'.format(mse_nonshift))
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(t_dense * 1e9, xr_shift, 'm--', label="Reconstructed (shifted)")
    axs[2].stem(t_samples_shift * 1e9, x_samples_shift, linefmt='k-', markerfmt='ko', basefmt='k-', label="Shifted Samples")
    axs[2].set_title('Shifted Sampling Reconstruction\nMSE = {:.3e}'.format(mse_shift))
    axs[2].set_xlabel('Time (ns)')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



