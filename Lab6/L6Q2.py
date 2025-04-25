import numpy as np
import matplotlib.pyplot as plt

class MDAC25:
    def __init__(self, Vr: float,
                 ota_gain: float = np.inf,
                 ota_offset: float = 0.0,
                 cap_mismatch: float = 0.0,
                 comp_offset: float = 0.0):
        self.Vr_nom     = Vr
        self.ota_gain   = ota_gain
        self.ota_offset = ota_offset
        self.cap_mismatch = cap_mismatch
        self.comp_offset  = comp_offset
        
        self.levels = 7
        self.offset = (self.levels - 1) / 2
        self.G_nom  = 4.0
        self.Vr = self.Vr_nom * (1 + self.cap_mismatch)

    def step(self, x: np.ndarray):
        x1 = x + self.ota_offset
        if np.isinf(self.ota_gain):
            G_eff = self.G_nom
        else:
            G_eff = self.G_nom * (self.ota_gain / (self.ota_gain + 1))
        x_cmp = x1 + self.comp_offset
        b = np.floor(G_eff * x_cmp / self.Vr + self.offset + 0.5)
        b = np.clip(b, 0, self.levels - 1).astype(int)
        residue = G_eff * x1 - (b - self.offset) * self.Vr
        return b, residue

class PipelineADC:
    def __init__(self, num_stages: int, Vr: float, **err_kwargs):
        self.stages = [MDAC25(Vr, **err_kwargs) for _ in range(num_stages)]
        self.M      = 7
        self.offset = (self.M - 1) / 2

    def convert(self, x: np.ndarray):
        residue = x.copy()
        codes = []
        for stage in self.stages:
            b, residue = stage.step(residue)
            codes.append(b)
        raw = np.zeros_like(x)
        N = len(self.stages)
        for i, b in enumerate(codes):
            raw += (b - self.offset) * (self.M ** (N - 1 - i))
        max_raw = self.M**N - 1
        code13 = np.round((raw + max_raw/2) * (2**13 - 1) / max_raw)
        return code13.astype(int)

def compute_snr(code13, x, VFS):
    analog = (code13 / (2**13 - 1) * 2 - 1) * VFS
    sig_p = np.mean(x**2)
    err_p = np.mean((analog - x)**2)
    return 10 * np.log10(sig_p / err_p)

fs, fin, VFS = 500e6, 200e6, 1.0
N = 2**14
t = np.arange(N) / fs
x = VFS * np.sin(2 * np.pi * fin * t)

error_params = {
    'ota_gain': 200,
    'cap_mismatch': 0.4,
    'ota_offset': 0.1,
    'comp_offset': 0.1
}

adc_err = PipelineADC(num_stages=6, Vr=VFS, **error_params)
code13_err = adc_err.convert(x)
snr_err = compute_snr(code13_err, x, VFS)

print(f"SNR with errors (OTA gain=500, cap mismatch=0.22, ota offset=5 mV, comp offset=0.04 V): {snr_err:.2f} dB")
