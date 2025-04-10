import numpy as np

measured = np.array([-0.01, 0.105, 0.195, 0.28, 
                     0.37,  0.48,  0.60,  0.75])
M0 = measured[0]           
meas_FS = measured[7] - M0  
ideal_FS = 0.70
alpha = ideal_FS/meas_FS    
LSB = 0.1
corrected = (measured - M0)*alpha

print("Code | Measured | Corrected")
for k in range(8):
    print(f"{k:4d} | {measured[k]:8.3f} | {corrected[k]:8.3f}")
dnl = np.zeros(8)
inl = np.zeros(8)
for k in range(1, 8):
    step_k = corrected[k] - corrected[k-1]
    dnl[k] = (step_k / LSB) - 1.0
for k in range(8):
    ideal_k = k * LSB
    inl[k]  = (corrected[k] - ideal_k) / LSB

print("\nCode | Corrected(V) |  Ideal(V) |  Step(V)  |  DNL(LSB)  |  INL(LSB)")
print("-----+--------------+-----------+-----------+------------+----------")

for k in range(8):
    if k == 0:
        step_str = "  --   "
        dnl_str  = "   --   "
    else:
        step_k = corrected[k] - corrected[k-1]
        step_str = f"{step_k:8.4f}"
        dnl_str  = f"{dnl[k]:8.3f}"

    ideal_k = k * LSB
    print(f"{k:4d} | {corrected[k]:12.4f} | {ideal_k:9.4f} | "
          f"{step_str} | {dnl_str} | {inl[k]:8.3f}")

max_dnl = np.max(np.abs(dnl))
max_inl = np.max(np.abs(inl))

print(f"\nMaximum DNL = {max_dnl:.3f} LSB")
print(f"Maximum INL = {max_inl:.3f} LSB")
