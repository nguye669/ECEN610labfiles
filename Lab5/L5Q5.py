import numpy as np
import matplotlib.pyplot as plt

# Given data
N_BITS = 3
dnl = [0, -0.5, 0, +0.5, -1, +0.5, +0.5, 0]
offset_error = 0.5
fullscale_error = 0.5  # for reference only

# 1) Compute the code boundaries
B = np.zeros(len(dnl)+1)
B[0] = offset_error  # boundary for code 0
for k in range(len(dnl)):
    Wk = 1.0 + dnl[k]   # code width
    B[k+1] = B[k] + Wk

# 2) Midpoints, code center
M = 0.5*(B[:-1] + B[1:])  # midpoints for codes 0..7

# 3) Ideal end-point line => from B0=0.5 to B8=8.5 => slope=1 => midpoint for code k = k+1
k_arr = np.arange(8)  # codes 0..7
M_ideal = k_arr + 1.0
INL = M - M_ideal

# 4) Print a table of results
print("Code | Boundary(k)  Boundary(k+1)  Midpoint  IdealMid  INL")
for k in range(8):
    print(f"{k:3d} | {B[k]:8.2f}    {B[k+1]:8.2f}    {M[k]:8.2f}   {M_ideal[k]:8.2f}  {INL[k]:6.2f}")

print("\nPeak INL (in LSB):", np.max(np.abs(INL)) )

# 5) Plot the transfer function
# We'll do a step plot from boundary B[k]..B[k+1] => code k
x_vals = []
y_vals = []
for k in range(8):
    # For a step:
    x_vals.extend([B[k], B[k+1]])
    y_vals.extend([k, k])

plt.figure()
plt.plot(x_vals, y_vals, drawstyle='steps-post')
plt.title("ADC Transfer Characteristic (3-bit with given DNL)")
plt.xlabel("Input Level (LSB)")
plt.ylabel("Digital Output Code")
plt.grid(True)
plt.ylim(-0.5, 7.5)
plt.show()
