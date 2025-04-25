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
        self.k2, self.k3, self.k4, self.k5 = nl2, nl3, nl4, nl5
        self.levels = 7
        self.offset = (self.levels - 1)/2
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
        r_n_safe = np.clip(r_n, -1.5, 1.5)
        coeffs = [0.0, 1.0, self.k2, self.k3, self.k4, self.k5]  
        r_nl = np.polynomial.polynomial.polyval(r_n_safe, coeffs)
        return b, r_nl * self.Vr
    
def generate_multitone_bpsk(fs: float, VFS: float, K: int = 128, BW: float = 200e6, n_rep: int = 32):
    f0 = BW / K
    N_block = int(round(fs / f0))
    f0 = fs / N_block  
    t_block = np.arange(N_block) / fs

    bits = np.random.choice([-1, 1], size=K)
    A_tone = VFS / np.sqrt(2*K)

    s_block = np.zeros_like(t_block)
    for k in range(K):
        fi = (k+1) * f0
        s_block += bits[k] * A_tone * np.cos(2*np.pi*fi*t_block)

    signal = np.tile(s_block, n_rep)
    return signal, bits, N_block, np.arange(1, K+1)


def ber_mse(y_block: np.ndarray, bits: np.ndarray, N_block: int, k_idx):
    Y = np.fft.rfft(y_block, n=N_block)
    amps = (2/N_block) * np.real(Y[k_idx])  
    est_bits = np.sign(amps)
    return np.mean(est_bits != bits), np.mean((amps - bits)**2)

def run_calibration(dec_factor: int, include_nl: bool = False, max_iters: int = 100000):
    fs, VFS = 500e6, 1.0
    signal, bits, N_block, k_idx = generate_multitone_bpsk(fs, VFS)
    N = len(signal)

    noise_power = np.mean(signal**2) / 10**(80/10)
    x = signal + np.sqrt(noise_power) * np.random.randn(N)

    num_stages = 6
    err_lin = {'ota_gain': 200, 'cap_mismatch': 0.4, 'ota_offset': 0.1, 'comp_offset': 0.1}
    nl_kwargs = {'nl2': 0.10, 'nl3': 0.20, 'nl4': 0.15, 'nl5': 0.10} if include_nl else {}

    mdacs_err   = [MDAC25(VFS, **err_lin, **nl_kwargs) for _ in range(num_stages)]
    mdacs_ideal = [MDAC25(VFS) for _ in range(num_stages)]

    phimat_err = np.zeros((num_stages, N))
    res = x.copy()
    for i, st in enumerate(mdacs_err):
        b, res = st.step(res)
        phimat_err[i] = b - st.offset

    phimat_ideal = np.zeros_like(phimat_err)
    res = x.copy()
    for i, st in enumerate(mdacs_ideal):
        b, res = st.step(res)
        phimat_ideal[i] = b - st.offset

    M = 7
    w_ideal = np.array([M**(num_stages-1-i) for i in range(num_stages)], float)
    raw_ideal = w_ideal @ phimat_ideal

    N_cal, mu = 4, 1e-3
    w = w_ideal.copy()
    maxr = M**num_stages - 1

    blk_idx     = np.arange(N_block)
    phimat_blk  = phimat_err[:, blk_idx]

    snr_list, ber_list, mse_list, iters = [], [], [], []

    def compute_snr(curr_w):
        raw  = curr_w @ phimat_err
        code = np.round((raw + maxr/2)*(2**13-1)/maxr)
        ana  = (code/(2**13-1)*2 - 1) * VFS
        return 10*np.log10(np.mean(signal**2) / np.mean((ana - signal)**2))

    for k in range(max_iters):
        n = k % N
        e = raw_ideal[n] - (w @ phimat_err[:, n])
        w[:N_cal] += mu * e * phimat_err[:N_cal, n]

        if k % dec_factor == 0:
            iters.append(k)
            snr_list.append(compute_snr(w))
            raw_blk  = w @ phimat_blk
            code_blk = np.round((raw_blk + maxr/2)*(2**13-1)/maxr)
            ana_blk  = (code_blk/(2**13-1)*2 - 1) * VFS
            ber, mse = ber_mse(ana_blk, bits, N_block, k_idx)
            ber_list.append(ber)
            mse_list.append(mse)

    return {'iters': np.array(iters),
            'snr':  np.array(snr_list),
            'ber':  np.array(ber_list),
            'mse':  np.array(mse_list)}

if __name__ == "__main__":
    dec_factors = [10, 100, 1000, 10000]

    results_lin = {d: run_calibration(d, include_nl=False) for d in dec_factors}
    results_nl  = {d: run_calibration(d, include_nl=True)  for d in dec_factors}

    plt.figure(figsize=(7,4))
    for d, res in results_lin.items():
        plt.plot(res['iters'], res['snr'], label=f'Linear, n={d}')
    for d, res in results_nl.items():
        plt.plot(res['iters'], res['snr'], '--', label=f'Non‑linear, n={d}')
    plt.xlabel('Iteration'); plt.ylabel('SNR (dB)'); plt.grid(True)
    plt.legend(); plt.title('Problem 7: SNR Convergence'); plt.tight_layout()

    plt.figure(figsize=(7,4))
    for d, res in results_lin.items():
        plt.semilogy(res['iters'], res['ber'] + 1e-12, label=f'Linear, n={d}')
    for d, res in results_nl.items():
        plt.semilogy(res['iters'], res['ber'] + 1e-12, '--', label=f'Non‑linear, n={d}')
    plt.xlabel('Iteration'); plt.ylabel('BER'); plt.grid(True, which='both')
    plt.legend(); plt.title('Problem 7: BER Convergence'); plt.tight_layout()

    plt.figure(figsize=(7,4))
    for d, res in results_lin.items():
        plt.plot(res['iters'], res['mse'] + 1e-15, label=f'Linear, n={d}')
    for d, res in results_nl.items():
        plt.plot(res['iters'], res['mse'] + 1e-15, '--', label=f'Non‑linear, n={d}')
    plt.yscale('log')
    plt.xlabel('Iteration'); plt.ylabel('MSE'); plt.grid(True, which='both')
    plt.legend(); plt.title('Problem 7: MSE Convergence'); plt.tight_layout()

    plt.show()
