import numpy as np
hist_counts = np.array([43, 115, 85, 101, 122, 170, 75, 146, 
                        125, 60, 95, 95, 115, 40, 120, 242])
N = np.sum(hist_counts)
T = np.zeros(len(hist_counts)+1)
for i in range(len(hist_counts)):
    T[i+1] = T[i] + hist_counts[i]
ideal_width = N / 16
DNL = []
for i in range(len(hist_counts)):
    actual_width = T[i+1] - T[i]
    DNL_i = (actual_width - ideal_width) / ideal_width
    DNL.append(DNL_i)
INL = []
for i in range(len(T)):
    ideal_boundary = i * ideal_width
    INL_i = (T[i] - ideal_boundary) / ideal_width
    INL.append(INL_i)
DNL = np.array(DNL)
INL = np.array(INL)
peak_DNL = np.max(np.abs(DNL))
peak_INL = np.max(np.abs(INL))
monotonic = np.all(np.diff(T) >= 0)

print("Total samples N =", N)
print("Ideal code width =", ideal_width)
print("\nCode | HistCount | DNL")
for i in range(len(hist_counts)):
    print(f"{i:2d}   {hist_counts[i]:8d}   {DNL[i]:8.4f}")

print("\nPeak DNL =", peak_DNL)
print("Peak INL =", peak_INL)
print("\nIs the ADC monotonic?", monotonic)
