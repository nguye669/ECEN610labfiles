import numpy as np
import matplotlib.pyplot as plt
f_clk = 2.4e9
T_clk = 1.0 / f_clk
N = 8
C_equal = 15.925e-12  # example single capacitor from the lab
C0 = 10e-12
C1 = 12e-12
C2 = 15e-12
C3 = 20e-12
Csum = C0 + C1 + C2 + C3
offsets = [0, 2, 4, 6]
num_points = 20001
fmax = 1.0e9
fvals = np.linspace(0, fmax, num_points)
# z = e^{-j 2π f T_clk}
j = 1j
zvals = np.exp(-j * 2.0 * np.pi * fvals * T_clk)

def finite_sum_8(z, offset):
    # sum_{n=0..7} z^{-(n + offset)} = z^{-offset} * sum_{n=0..7} z^{-n}
    return np.sum([z**-(n + offset) for n in range(8)], axis=0)

#  H_a = (1/(4*C)) * sum_{k=0..3} sum_{n=0..7} z^{-(n + 2k)}
def H_a_discharge_equal(z_array):
    total_sum = 0
    for k in offsets:
        total_sum += finite_sum_8(z_array, k)
    return total_sum / (4.0 * C_equal)

#  H_b = (1/(4*C)) * [ sum_{k=0..3} sum_{n=0..7} z^{-(n + 2k)} ] / (1 - z^{-16})
def H_b_no_discharge_equal(z_array):
    sum_8caps = 0
    for k in offsets:
        sum_8caps += finite_sum_8(z_array, k)
    denom = 1.0 - z_array**16  #z^{-16} = (z^16)^{-1}
    # Avoid division by zero near f=0
    return sum_8caps / (4.0 * C_equal * (denom + 1e-30))


#  H_c = 1/(C0+C1+C2+C3) * sum_{k=0..3} sum_{n=0..7} z^{-(n + offset_k)}
def H_c_discharge_diff(z_array):
    total_sum = 0
    for k in offsets:
        total_sum += finite_sum_8(z_array, k)
    return total_sum / (Csum)

#  H_c2 = 1/(C0+C1+C2+C3) * [ sum_{k=0..3} sum_{n=0..7} z^{-(n + offset_k)} ] / (1 - z^{-16})
def H_c2_no_discharge_diff(z_array):
    sum_8caps = 0
    for k in offsets:
        sum_8caps += finite_sum_8(z_array, k)
    denom = 1.0 - z_array**16
    return sum_8caps / (Csum * (denom + 1e-30))

H_a_vals = H_a_discharge_equal(zvals)
mag_a = np.abs(H_a_vals)

plt.figure()
plt.plot(fvals*1e-9, 20*np.log10(mag_a+1e-30))
plt.title("Scenario (a): Discharged, Equal Caps")
plt.xlabel("Frequency [GHz]")
plt.ylabel("|H_a| [dB]")
plt.grid(True)
plt.show()

H_b_vals = H_b_no_discharge_equal(zvals)
mag_b = np.abs(H_b_vals)

plt.figure()
plt.plot(fvals*1e-9, 20*np.log10(mag_b+1e-30))
plt.title("Scenario (b): Never Discharged, Equal Caps")
plt.xlabel("Frequency [GHz]")
plt.ylabel("|H_b| [dB]")
plt.grid(True)
plt.show()


H_c_vals = H_c_discharge_diff(zvals)
mag_c = np.abs(H_c_vals)

plt.figure()
plt.plot(fvals*1e-9, 20*np.log10(mag_c+1e-30))
plt.title("Scenario (c): Discharged, Different Caps")
plt.xlabel("Frequency [GHz]")
plt.ylabel("|H_c| [dB]")
plt.grid(True)
plt.show()


H_c2_vals = H_c2_no_discharge_diff(zvals)
mag_c2 = np.abs(H_c2_vals)

plt.figure()
plt.plot(fvals*1e-9, 20*np.log10(mag_c2+1e-30))
plt.title("Scenario (c2): Never Discharged, Different Caps [Optional]")
plt.xlabel("Frequency [GHz]")
plt.ylabel("|H_c2| [dB]")
plt.grid(True)
plt.show()
