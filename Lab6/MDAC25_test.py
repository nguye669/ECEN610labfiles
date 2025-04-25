import numpy as np
import matplotlib.pyplot as plt

class MDAC25:
    def __init__(self, vfs: float):
        self.vr = vfs
        self.gain = 4
        self.levels = 7            
        self.offset = (self.levels - 1) / 2 

    def transfer(self, x: np.ndarray) -> np.ndarray:
        b = np.floor((self.gain * x / self.vr) + self.offset + 0.5)
        b = np.clip(b, 0, self.levels - 1)
        return self.gain * x - (b - self.offset) * self.vr
VFS = 1.0
mdac = MDAC25(vfs=VFS)

x = np.linspace(-VFS, VFS, 10001)
y = mdac.transfer(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2)
plt.axhline(VFS/2, linestyle='--')
plt.axhline(-VFS/2, linestyle='--')
plt.title("2.5-bit MDAC Transfer (Vo = 4·Vi - (b-3)·Vr)")
plt.xlabel("Input Voltage Vi (V)")
plt.ylabel("Output Residue Vo (V)")
plt.grid(True)
plt.show()
