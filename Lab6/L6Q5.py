import numpy as np
import matplotlib.pyplot as plt

class MDAC25:
    def __init__(self, Vr: float,
                 ota_gain: float = np.inf,
                 ota_offset: float = 0.0,
                 cap_mismatch: float = 0.0,
                 comp_offset: float = 0.0,
                 gbw_factor: float = 1.0,
                 nl2: float = 0.0,
                 nl3: float = 0.0,
                 nl4: float = 0.0,
                 nl5: float = 0.0):
        self.Vr_nom       = Vr
        self.ota_gain     = ota_gain
        self.ota_offset   = ota_offset
        self.cap_mismatch = cap_mismatch
        self.comp_offset  = comp_offset
        self.k2 = nl2
        self.k3 = nl3
        self.k4 = nl4
        self.k5 = nl5
        self.levels = 7
        self.offset = (self.levels - 1) / 2
        self.G_nom  = 4.0
        self.Vr     = self.Vr_nom * (1 + self.cap_mismatch)

    def step(self, x: np.ndarray):
        x1 = x + self.ota_offset
        G_eff = self.G_nom if np.isinf(self.ota_gain) else self.G_nom * (self.ota_gain / (self.ota_gain + self.G_nom))
        v_lin = G_eff * x1
        v_cmp = v_lin + self.comp_offset
        b = np.floor(v_cmp / self.Vr + self.offset + 0.5)
        b = np.clip(b, 0, self.levels - 1).astype(int)
        residue_ideal = v_lin - (b - self.offset) * self.Vr
        r_n = residue_ideal / self.Vr
        r_nl = (r_n + self.k2*r_n**2 + self.k3*r_n**3 + self.k4*r_n**4 + self.k5*r_n**5)
        residue = r_nl * self.Vr
        return b, residue
num_stages = 6
N_cal      = 4
mu         = 1e-3
max_iters  = 100000
fs, fin, VFS = 500e6, 200e6, 1.0
N = 2**12

t = np.arange(N) / fs
signal = VFS * np.sin(2 * np.pi * fin * t)
noise_power = np.mean(signal**2) / 10**(80/10)
x = signal + np.sqrt(noise_power) * np.random.randn(N)

error_params = {'ota_gain':200, 'cap_mismatch':0.4, 'ota_offset':0.1, 'comp_offset':0.1}
nl = {'nl2':0.10,'nl3':0.20,'nl4':0.15,'nl5':0.10}

phimat_err = np.zeros((num_stages, N))
res = x.copy()
mdacs_err = [MDAC25(VFS, **error_params, **nl) for _ in range(num_stages)]
for i, stage in enumerate(mdacs_err): b, res = stage.step(res); phimat_err[i] = b - stage.offset

phimat_ideal = np.zeros_like(phimat_err)
res = x.copy()
mdacs_ideal = [MDAC25(VFS) for _ in range(num_stages)]
for i, stage in enumerate(mdacs_ideal): b, res = stage.step(res); phimat_ideal[i] = b - stage.offset

M = 7
ideal_w   = np.array([M**(num_stages-1-i) for i in range(num_stages)], float)
raw_ideal = ideal_w @ phimat_ideal

w      = ideal_w.copy(); errs = []; w_traj = []; snrs = []
maxr   = M**num_stages - 1

def compute_snr(w):
    raw  = w @ phimat_err
    code = np.round((raw + maxr/2)*(2**13-1)/maxr)
    ana  = (code/(2**13-1)*2 - 1)*VFS
    return 10*np.log10(np.mean(signal**2)/np.mean((ana-signal)**2))

for k in range(max_iters):
    n = k % N
    e = raw_ideal[n] - (w @ phimat_err[:, n])
    errs.append(abs(e))
    w[:N_cal] += mu * e * phimat_err[:N_cal, n]
    if k % 100 == 0:
        w_traj.append(w[:N_cal].copy()); snrs.append(compute_snr(w))
w_traj = np.array(w_traj); snrs = np.array(snrs)
its = np.arange(len(snrs))*100

plt.figure(figsize=(6,4))
plt.plot(its, snrs, marker='o')
plt.title('Baseline SNR vs Iteration')
plt.xlabel('Iteration'); plt.ylabel('SNR (dB)'); plt.grid(True)

dec_factors = [10, 100, 1000, 10000]
snr_decim = {}
for dec in dec_factors:
    w_tmp = ideal_w.copy()
    trace = []
    for k in range(max_iters):
        n = k % N
        e = raw_ideal[n] - (w_tmp @ phimat_err[:, n])
        w_tmp[:N_cal] += mu * e * phimat_err[:N_cal, n]
        if k % dec == 0:
            trace.append(compute_snr(w_tmp))
    snr_decim[dec] = trace

plt.figure(figsize=(6,4))
for dec, trace in snr_decim.items():
    iters = np.arange(len(trace)) * dec
    plt.plot(iters, trace, label=f'n={dec}')
plt.title('SNR vs Iteration for Decimation Factors')
plt.xlabel('Iteration'); plt.ylabel('SNR (dB)')
plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()

