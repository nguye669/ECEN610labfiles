import math

def db(x):
    """Helper function to convert linear ratio to dB."""
    return 10 * math.log10(x)

# 1(a) --------------------------------------------------------
N = 12
full_scale_pp = 1.2  # V peak-to-peak
full_scale_peak = full_scale_pp / 2.0  # 0.6 V
adc_step = full_scale_pp / (2**N)      # quantization step Δ
adc_noise_var = (adc_step**2)/12

# Input sinusoid has 200 mV RMS
input_rms_a = 0.2  # V
input_peak_a = input_rms_a * math.sqrt(2.0)

# (i) SNR using the standard formula for a sine wave of amplitude A vs. full-scale amplitude
FS_ideal_SNR = 6.02 * N + 1.76  # dB, for full-scale sine
amplitude_ratio = input_peak_a / full_scale_peak
SNR_a_db = FS_ideal_SNR + 20*math.log10(amplitude_ratio)

print("1(a) -------------------------------------------------")
print(f"Full-scale SNR (ideal 12-bit, full-range): {FS_ideal_SNR:.2f} dB")
print(f"Amplitude ratio (input vs full-scale)    : {amplitude_ratio:.4f}")
print(f"SNR for 200mV_rms input                  : {SNR_a_db:.2f} dB\n")

# 1(b) --------------------------------------------------------
# Full-range sinusoid => RMS = 0.6 / sqrt(2) = 0.4243 V
full_range_rms = full_scale_peak / math.sqrt(2.0)
signal_power = full_range_rms**2  # ~0.180 V^2

# (b1) Gaussian noise with sigma=0.5 => noise RMS=0.5 => noise power=0.25
noise_rms_b = 0.5
noise_power_b = noise_rms_b**2

snr_in_b_linear = signal_power / noise_power_b
snr_in_b_db = db(snr_in_b_linear)

# (b2) The total noise = input noise + quantization noise (but quant noise is tiny)
total_noise_power_b = noise_power_b + adc_noise_var
snr_out_b_linear = signal_power / total_noise_power_b
snr_out_b_db = db(snr_out_b_linear)

print("1(b) -------------------------------------------------")
print(f"Input Signal RMS (full-range sine): {full_range_rms:.4f} V")
print(f"Input Noise (Gaussian) RMS        : 0.50 V")
print(f"Input SNR                          : {snr_in_b_db:.2f} dB")
print(f"ADC Output SNR                     : {snr_out_b_db:.2f} dB\n")

# 1(c) --------------------------------------------------------
# Now noise is uniform from -0.5 to +0.5 => peak-to-peak =1 => var=1/12 => ~0.0833 => RMS=~0.2887
noise_var_c = 1.0/12.0
noise_rms_c = math.sqrt(noise_var_c)  # ~0.2887
noise_power_c = noise_var_c

snr_in_c_linear = signal_power / noise_power_c
snr_in_c_db = db(snr_in_c_linear)

total_noise_power_c = noise_power_c + adc_noise_var
snr_out_c_linear = signal_power / total_noise_power_c
snr_out_c_db = db(snr_out_c_linear)

print("1(c) -------------------------------------------------")
print(f"Uniform Noise Variance          : {noise_var_c:.4f} V^2")
print(f"Uniform Noise RMS               : {noise_rms_c:.4f} V")
print(f"Input SNR (uniform noise)       : {snr_in_c_db:.2f} dB")
print(f"ADC Output SNR (uniform noise)  : {snr_out_c_db:.2f} dB")
