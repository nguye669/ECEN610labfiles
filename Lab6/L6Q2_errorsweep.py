import numpy as np
import matplotlib.pyplot as plt

class MDAC25:
    """
    2.5-bit MDAC stage with static error parameters:
      Vo = G_eff*(Vi + ota_offset) - (b - offset)*Vr*(1+cap_mismatch)
      Comparator uses comp_offset on input.
    G_eff = G_nom * (ota_gain / (ota_gain + 1)), with infinite gain handled.
    """
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

        # effective Vr including capacitor mismatch
        self.Vr = self.Vr_nom * (1 + self.cap_mismatch)

    def step(self, x: np.ndarray):
        # 1) add OTA offset
        x1 = x + self.ota_offset
        # 2) effective gain with infinite check
        if np.isinf(self.ota_gain):
            G_eff = self.G_nom
        else:
            G_eff = self.G_nom * (self.ota_gain / (self.ota_gain + 1))
        # 3) comparator offset
        x_cmp = x1 + self.comp_offset
        # 4) decision
        b = np.floor(G_eff * x_cmp / self.Vr + self.offset + 0.5)
        b = np.clip(b, 0, self.levels - 1).astype(int)
        # 5) residue
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
        # digital combine to 13-bit
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

# Simulation setup
fs, fin, VFS = 500e6, 200e6, 1.0
N = 2**14
t = np.arange(N) / fs
x = VFS * np.sin(2 * np.pi * fin * t)

# Define sweep ranges
param_sweeps = {
    'cap_mismatch': np.linspace(0, 0.3, 301),
    'ota_gain': np.logspace(1, 6, 301),          # 10^1 to 10^6
    'ota_offset': np.linspace(0, 20e-3, 301),    # 0 to 20 mV
    'comp_offset': np.linspace(-50e-3, 50e-3, 301)  # ±50 mV
}

# Sweep and record SNR
snr_results = {}
for param, values in param_sweeps.items():
    snrs = []
    for v in values:
        adc = PipelineADC(num_stages=6, Vr=VFS, **{param: v})
        code13 = adc.convert(x)
        snrs.append(compute_snr(code13, x, VFS))
    snr_results[param] = np.array(snrs)

# Find parameter value closest to 10 dB SNR
threshold = 10.0
exact_vals = {}
for param, values in param_sweeps.items():
    diffs = np.abs(snr_results[param] - threshold)
    idx = np.nanargmin(diffs)
    exact_vals[param] = (values[idx], snr_results[param][idx])

# Print exact error values for 10 dB
for param, (val, snr_val) in exact_vals.items():
    unit = {
        'cap_mismatch': '',
        'ota_gain': '',
        'ota_offset': ' V',
        'comp_offset': ' V',
    }[param]
    print(f"{param}: {val:.6g}{unit} → SNR = {snr_val:.2f} dB")

# Plot all SNR curves with marker at 10 dB crossing
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, (param, values) in zip(axes, param_sweeps.items()):
    snrs = snr_results[param]
    val10, _ = exact_vals[param]
    if param == 'ota_gain':
        ax.semilogx(values, snrs, '-b')
    else:
        ax.plot(values, snrs, '-b')
    ax.axhline(threshold, color='r', linestyle='--')
    ax.axvline(val10, color='k', linestyle=':')
    ax.scatter([val10], [threshold], color='k', zorder=5)
    ax.set_title(f"SNR vs {param.replace('_',' ').title()}")
    ax.set_xlabel(param.replace('_', ' ').title())
    ax.set_ylabel("SNR (dB)")
    ax.grid(True)

plt.tight_layout()
plt.show()
