import numpy as np
import matplotlib.pyplot as plt

class MDAC25:
    def __init__(self, Vr: float,
                 ota_gain: float = np.inf,
                 ota_offset: float = 0.0,
                 cap_mismatch: float = 0.0,
                 comp_offset: float = 0.0):
        self.Vr_nom      = Vr
        self.ota_gain    = ota_gain
        self.ota_offset  = ota_offset
        self.cap_mismatch= cap_mismatch
        self.comp_offset = comp_offset
        
        self.levels = 7
        self.offset = (self.levels - 1) / 2
        self.G_nom  = 4.0
        self.Vr     = self.Vr_nom * (1 + self.cap_mismatch)

    def step(self, x: np.ndarray):
        x1 = x + self.ota_offset
        G_eff = self.G_nom if np.isinf(self.ota_gain) else self.G_nom * (self.ota_gain/(self.ota_gain+1))
        x_cmp = x1 + self.comp_offset
        b = np.floor(G_eff * x_cmp / self.Vr + self.offset + 0.5)
        b = np.clip(b, 0, self.levels-1).astype(int)
        residue = G_eff * x1 - (b - self.offset)*self.Vr
        return b, residue
num_stages = 6
N_cal = 4                 
mu = 1e-3                  
max_iters = 10000          
fs, fin, VFS = 500e6, 200e6, 1.0
N = 2**12                 
t = np.arange(N) / fs
x = VFS * np.sin(2 * np.pi * fin * t)
error_params = {
    'ota_gain':    200,
    'cap_mismatch':0.4,
    'ota_offset':  0.1,
    'comp_offset': 0.1
}


phimat_err = np.zeros((num_stages, N))
res = x.copy()
mdacs_err = [MDAC25(VFS, **error_params) for _ in range(num_stages)]
for i, stage in enumerate(mdacs_err):
    b, res = stage.step(res)
    phimat_err[i, :] = b - stage.offset
phimat_ideal = np.zeros_like(phimat_err)
res = x.copy()
mdacs_ideal = [MDAC25(VFS) for _ in range(num_stages)]
for i, stage in enumerate(mdacs_ideal):
    b, res = stage.step(res)
    phimat_ideal[i, :] = b - stage.offset
M = 7
ideal_w = np.array([M**(num_stages-1-i) for i in range(num_stages)], dtype=float)
raw_ideal = ideal_w @ phimat_ideal
w = ideal_w.copy()
errs   = []
w_traj = []   
snrs   = []

def compute_snr(w):
    raw_est = w @ phimat_err
    maxr = M**num_stages - 1
    code13 = np.round((raw_est + maxr/2)*(2**13-1)/maxr)
    analog = (code13/(2**13-1)*2 - 1)*VFS
    return 10*np.log10(np.mean(x**2)/np.mean((analog-x)**2))


for k in range(max_iters):
    n = k % N
    raw_est = w @ phimat_err[:, n]
    e = raw_ideal[n] - raw_est
    errs.append(np.abs(e))
    w[:N_cal] += mu * e * phimat_err[:N_cal, n]
    
    if k % 100 == 0:
        w_traj.append(w[:N_cal].copy())
        snrs.append(compute_snr(w))

w_traj = np.array(w_traj)
snrs   = np.array(snrs)

plt.figure(figsize=(6,4))
its = np.arange(len(w_traj))*100
for i in range(N_cal):
    plt.plot(its, w_traj[:, i], label=f'w{i}')
plt.title("Weight Convergence (First 4 Stages)")
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.legend()
plt.grid(True)

plt.figure(figsize=(6,4))
plt.plot(errs[:2000])
plt.title("Instantaneous Error |e[n]|")
plt.xlabel("Iteration")
plt.ylabel("|e|")
plt.grid(True)

plt.figure(figsize=(6,4))
plt.plot(its, snrs, marker='o')
plt.title("SNR During Calibration")
plt.xlabel("Iteration")
plt.ylabel("SNR (dB)")
plt.grid(True)

mu_list = [1e-5, 1e-4, 1e-3, 1e-2,0.1]
snr_mult = {}
for mu_test in mu_list:
    w_tmp = ideal_w.copy()
    snr_trace = []
    for k in range(max_iters):
        n = k % N
        e = raw_ideal[n] - (w_tmp @ phimat_err[:, n])
        w_tmp[:N_cal] += mu_test * e * phimat_err[:N_cal, n]
        if k % 100 == 0:
            snr_trace.append(compute_snr(w_tmp))
    snr_mult[mu_test] = snr_trace

plt.figure(figsize=(6,4))
for mu_test, trace in snr_mult.items():
    plt.plot(its, trace, label=f"μ={mu_test}")
plt.title("SNR vs Iteration for Various Step Sizes")
plt.xlabel("Iteration")
plt.ylabel("SNR (dB)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
