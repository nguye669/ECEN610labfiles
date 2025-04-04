import numpy as np
import matplotlib.pyplot as plt

Fs = 10e9          
Ts = 1 / Fs        
N = 10000          
t = np.arange(N) * Ts

freqs = np.array([0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9])

x_ideal = 0.5 + 0.1 * np.sum([np.cos(2 * np.pi * f * t) for f in freqs], axis=0)
tau = 40e-12       
T_on = Ts / 2      
alpha = 1 - np.exp(-T_on / tau)
y = np.zeros_like(x_ideal)
y[0] = x_ideal[0]  
for i in range(1, N):
    y[i] = y[i - 1] * np.exp(-T_on / tau) + alpha * x_ideal[i]
N_bits = 7
levels = 2 ** N_bits        
Delta = 1.0 / levels        
x_ADC = np.round(y / Delta) * Delta
x_ADC = np.clip(x_ADC, 0, 1)
E = x_ADC - x_ideal
var_E_initial = np.var(E)
var_quant = Delta ** 2 / 12

print("Initial ADC error variance: {:.3e}".format(var_E_initial))
print("Uniform quantization noise variance: {:.3e}".format(var_quant))
print("Initial variance ratio: {:.3f}".format(var_E_initial / var_quant))
M_values = np.arange(2, 11)  
ratio_list = []
fir_weights = {} 

for M in M_values:
    X = np.vstack([x_ADC[i : N - M + i + 1] for i in range(M)]).T
    target = E[M - 1:]
    h, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    fir_weights[M] = h
    E_est = X @ h
    x_ADC_corr = x_ADC[M - 1:] + E_est
    E_new = x_ADC_corr - x_ideal[M - 1:]
    var_E_new = np.var(E_new)
    ratio = var_E_new / var_quant
    ratio_list.append(ratio) 
    print("FIR filter length M = {}:".format(M))
    print("  FIR Weights: ", h)
    print("  Corrected variance ratio = {:.3f}".format(ratio))
    print("-" * 50)
plt.figure(figsize=(8, 5))
plt.plot(M_values, ratio_list, marker='o', linestyle='-')
plt.xlabel('FIR Filter Tap Length (M)')
plt.ylabel('Variance Ratio (Var(E_new) / Var(Quantization Noise))')
plt.title('Effect of FIR Filter Length on ADC Error Variance Ratio')
plt.grid(True)
plt.show()

