import numpy as np
import matplotlib.pyplot as plt
f_in = 1e9        
f_s  = 10e9       
tau  = 10e-12     

t_stop = 5e-9     
dt = 0.1e-12      
t = np.arange(0, t_stop, dt)
vin = np.sin(2.0 * np.pi * f_in * t)
T_s = 1.0 / f_s
clock_on_fraction = 0.5
switch = ( (t % T_s) < (clock_on_fraction * T_s) ).astype(float)
vout = np.zeros_like(t)
vout[0] = 0.0 
alpha = dt / tau
for i in range(len(t) - 1):
    if switch[i] == 1:
        vout[i+1] = vout[i] + alpha * (vin[i] - vout[i])
    else:
        vout[i+1] = vout[i]
plt.figure()
plt.plot(t*1e9, vin, label="Input (1 GHz sine)")
plt.plot(t*1e9, vout, label="Sampled Output")
plt.xlabel("Time [ns]")
plt.ylabel("Voltage [V]")
plt.title("RC Sampler Output (f_in=1GHz, f_s=10GHz, τ=10ps)")
plt.legend()
plt.show()
