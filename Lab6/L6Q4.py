import numpy as np
import matplotlib.pyplot as plt

fs, fin, VFS = 500e6, 200e6, 1.0
N           = 2**12
t           = np.arange(N) / fs
x           = VFS * np.sin(2*np.pi*fin*t)

num_stages  = 6
N_cal       = 4
M           = 7

error_params = dict(
    ota_gain     = 200,
    cap_mismatch = 0.4,
    ota_offset   = 0.1,
    comp_offset  = 0.1,
)

class MDAC25:
    def __init__(self, Vr, ota_gain=np.inf, ota_offset=0.0,
                 cap_mismatch=0.0, comp_offset=0.0):
        self.Vr_nom      = Vr
        self.ota_gain    = ota_gain
        self.ota_offset  = ota_offset
        self.cap_mismatch= cap_mismatch
        self.comp_offset = comp_offset
        
        self.levels = 7
        self.offset = (self.levels-1)/2
        self.G_nom  = 4.0
        self.Vr     = self.Vr_nom * (1 + self.cap_mismatch)
    
    def step(self, u):
        u1      = u + self.ota_offset
        G_eff   = self.G_nom if np.isinf(self.ota_gain) else self.G_nom * (self.ota_gain/(self.ota_gain+1))
        u_cmp   = u1 + self.comp_offset
        b       = np.floor(G_eff*u_cmp/self.Vr + self.offset + 0.5)
        b       = np.clip(b, 0, self.levels-1).astype(int)
        residue = G_eff*u1 - (b - self.offset)*self.Vr
        return b, residue

def build_phi(mdacs):
    phi, res = np.zeros((num_stages, N)), x.copy()
    for i, stg in enumerate(mdacs):
        b, res = stg.step(res)
        phi[i, :] = b - stg.offset
    return phi

phi_err   = build_phi([MDAC25(VFS, **error_params) for _ in range(num_stages)])
phi_ideal = build_phi([MDAC25(VFS)                  for _ in range(num_stages)])

ideal_w   = np.array([M**(num_stages-1-i) for i in range(num_stages)], dtype=float)
raw_ideal = ideal_w @ phi_ideal

def compute_snr(w):
    raw_est = w @ phi_err
    maxr    = M**num_stages - 1
    code13  = np.round((raw_est + maxr/2)*(2**13-1)/maxr)
    analog  = (code13/(2**13-1)*2 - 1)*VFS
    return 10*np.log10(np.mean(x**2)/np.mean((analog - x)**2))

mu       = 0.001
eps      = 1e-6
max_iters= 30000
record   = 100

w        = ideal_w.copy()
its, snr = [], []

for k in range(max_iters):
    n        = k % N
    phi_vec  = phi_err[:N_cal, n]
    e        = raw_ideal[n] - (w @ phi_err[:, n])
    norm2    = np.sum(phi_vec**2) + eps
    w[:N_cal]+= mu * e * phi_vec / norm2
    
    if k % record == 0:
        its.append(k)
        snr.append(compute_snr(w))

its = np.array(its); snr = np.array(snr)

plt.figure(figsize=(7,4))
plt.plot(its, snr)
plt.title("SNR vs. Iteration – Normalised LMS")
plt.xlabel("Iteration")
plt.ylabel("SNR (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

