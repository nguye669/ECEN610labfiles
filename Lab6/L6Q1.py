import numpy as np
import matplotlib.pyplot as plt

class MDAC25:
    """
    2.5-bit MDAC stage (7-level quantizer, interstage gain=4):
    Vo = 4*Vi - (b - 3)*Vr
    """
    def __init__(self, Vr: float):
        self.Vr     = Vr
        self.gain   = 4
        self.levels = 7
        self.offset = (self.levels - 1) / 2

    def step(self, x: np.ndarray):
        b = np.floor(self.gain * x / self.Vr + self.offset + 0.5)
        b = np.clip(b, 0, self.levels - 1).astype(int)
        residue = self.gain * x - (b - self.offset) * self.Vr
        return b, residue

class PipelineADC:
    """
    Pipeline ADC built from multiple MDAC25 stages, reconstructing a 13-bit output.
    """
    def __init__(self, num_stages: int, Vr: float):
        self.stages = [MDAC25(Vr) for _ in range(num_stages)]
        self.M      = 7
        self.offset = (self.M - 1) / 2

    def convert(self, x: np.ndarray):
        stage_codes = []
        residue = x.copy()
        for stage in self.stages:
            b, residue = stage.step(residue)
            stage_codes.append(b)
        raw = np.zeros_like(x)
        N_stages = len(self.stages)
        for i, b in enumerate(stage_codes):
            raw += (b - self.offset) * (self.M ** (N_stages - 1 - i))
        max_raw = self.M**N_stages - 1
        code13 = np.round((raw + max_raw/2) * (2**13 - 1) / max_raw)
        return code13.astype(int)

fs  = 500e6      
fin = 200e6      
VFS = 1.0        
N   = 2**14      
t = np.arange(N) / fs
x = VFS * np.sin(2 * np.pi * fin * t)
adc = PipelineADC(num_stages=6, Vr=VFS)
code13 = adc.convert(x)
analog_recon = (code13 / (2**13 - 1) * 2 - 1) * VFS
signal_power = np.mean(x**2)
noise_power  = np.mean((analog_recon - x)**2)
snr_pipeline = 10 * np.log10(signal_power / noise_power)
print(f"Pipeline ADC SNR: {snr_pipeline:.2f} dB")
Y = np.fft.fft(analog_recon)
freqs = np.fft.fftfreq(N, 1/fs)
pos = freqs >= 0
plt.figure()
plt.plot(freqs[pos] / 1e6, 20 * np.log10(np.abs(Y[pos]) / N))
plt.title("Spectrum of Pipeline ADC Output")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.xlim(0, (fs/2/1e6))
plt.show()

