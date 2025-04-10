import sympy
import math

N = sympy.Symbol('N', positive=True)  # number of bits
SQNR_dB_expr = 6.02*N + 1.76

print("Part (a) --------------------------------------")
print("SQNR(dB) = 6.02 * N + 1.76")
print("Symbolic expression:", SQNR_dB_expr, "dB")

L = sympy.Symbol('L', positive=True)  # DFT size
difference_dB_expr = SQNR_dB_expr + 10*sympy.log(L/sympy.Integer(2), 10)

print("\nPart (b) --------------------------------------")
print("Difference in dB between signal bin and noise floor = SQNR(dB) + 10 * log10(L/2)")
print("Symbolic expression:", difference_dB_expr, "dB")

print("\nPart (c) --------------------------------------")
print("Necessary conditions for flat noise in the DFT:")
print("1) The sinusoid must be placed so that we have coherent sampling (integer number")
print("   of cycles in the window), or we apply an appropriate window function.")
print("2) The quantization noise must act as a pseudo-random/dithered process.")
print("These conditions are necessary for an approximately flat spectrum;")
print("they are not strictly sufficient for perfect flatness in hardware.")


OSR = sympy.Symbol('OSR', positive=True)  # Oversampling or noise-reduction ratio = Fs/(2B)
SNR_filtered_expr = SQNR_dB_expr + 10 * sympy.log(OSR, 10)

print("\nPart (d) --------------------------------------")
print("With OSR = Fs/(2B), the SNR after filtering = SQNR(dB) + 10 * log10(OSR)")
print("Symbolic expression:", SNR_filtered_expr, "dB")

print("\nDONE.")
