import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import entropy

cosine_sim_last_last = torch.load("./results_pecs/interstellar_propulsion_review/cosine_sim_last_last.pt")

threshold = 0.3

# Move everything to numpy
cosine_sim_np = cosine_sim_last_last.numpy()
distances_np = 1 - cosine_sim_np
recurrence_np = distances_np < threshold
np.fill_diagonal(recurrence_np, 0)


# --- Recurrence Quantification Analysis ---

def analyze_recurrence(recurrence_binary):
    recurrence_binary = recurrence_binary.astype(np.uint8)

    N = recurrence_binary.shape[0]
    assert recurrence_binary.shape[0] == recurrence_binary.shape[1], "Recurrence matrix must be square."

    # --- Recurrence Rate ---
    RR = np.sum(recurrence_binary) / (N * N)

    # --- Diagonal lines ---
    def get_diagonal_line_lengths(recurrence):
        lengths = []
        for offset in range(-N + 1, N):
            diag = np.diagonal(recurrence, offset=offset)
            if diag.size == 0:
                continue
            counts = count_lines(diag)
            lengths.extend(counts)
        return np.array(lengths)

    # --- Vertical lines ---
    def get_vertical_line_lengths(recurrence):
        lengths = []
        for col in recurrence.T:
            counts = count_lines(col)
            lengths.extend(counts)
        return np.array(lengths)

    # --- Helper: Count continuous 1's ---
    def count_lines(array):
        counts = []
        count = 0
        for v in array:
            if v == 1:
                count += 1
            elif count > 0:
                counts.append(count)
                count = 0
        if count > 0:
            counts.append(count)
        return counts

    diag_lengths = get_diagonal_line_lengths(recurrence_binary)
    vert_lengths = get_vertical_line_lengths(recurrence_binary)

    # Only consider lines of length >= 2
    diag_lengths = diag_lengths[diag_lengths >= 2]
    vert_lengths = vert_lengths[vert_lengths >= 2]

    # --- Determinism ---
    if np.sum(recurrence_binary) == 0:
        DET = 0.0
    else:
        DET = np.sum(diag_lengths) / np.sum(recurrence_binary)

    # --- Average Line Length (diagonals) ---
    if diag_lengths.size == 0:
        L = 0.0
    else:
        L = np.mean(diag_lengths)

    # --- Longest Diagonal Line ---
    if diag_lengths.size == 0:
        Lmax = 0
    else:
        Lmax = np.max(diag_lengths)

    # --- Entropy of Diagonal Lines ---
    if diag_lengths.size == 0:
        Entr_diag = 0.0
    else:
        hist, _ = np.histogram(diag_lengths, bins=np.arange(1, np.max(diag_lengths) + 2))
        p = hist / np.sum(hist)
        Entr_diag = entropy(p, base=np.e)

    # --- Laminarity (vertical lines) ---
    if np.sum(recurrence_binary) == 0:
        LAM = 0.0
    else:
        LAM = np.sum(vert_lengths) / np.sum(recurrence_binary)

    # --- Trapping Time (vertical lines) ---
    if vert_lengths.size == 0:
        TT = 0.0
    else:
        TT = np.mean(vert_lengths)

    # --- Longest Vertical Line ---
    if vert_lengths.size == 0:
        Vmax = 0
    else:
        Vmax = np.max(vert_lengths)

    # --- Entropy of Vertical Lines ---
    if vert_lengths.size == 0:
        Entr_vert = 0.0
    else:
        hist_v, _ = np.histogram(vert_lengths, bins=np.arange(1, np.max(vert_lengths) + 2))
        p_v = hist_v / np.sum(hist_v)
        Entr_vert = entropy(p_v, base=np.e)

    return np.array([RR, DET, L, Lmax, Entr_diag, LAM, TT, Vmax, Entr_vert])


# Single threshold analysis
metrics = analyze_recurrence(recurrence_np)

# Display in table
metrics_names = [
    'Recurrence Rate',
    'Determinism',
    'Average Line Length',
    'Longest Diagonal Line',
    'Entropy of Diagonal Lines',
    'Laminarity',
    'Trapping Time',
    'Longest Vertical Line',
    'Entropy of Vertical Lines'
]

df_metrics = pd.DataFrame({'Metric': metrics_names, 'Value': metrics})
print("\n==== Recurrence Quantification Analysis (RQA) Metrics ====\n")
print(df_metrics.to_string(index=False))

# --- Analyze Across Different Thresholds ---
thresholds = np.linspace(0.04, 0.4, 10)  # example thresholds from 0.05 to 0.5
all_metrics = []

for thresh in thresholds:
    recurrence_new = (distances_np < thresh)
    np.fill_diagonal(recurrence_new, 0)
    m = analyze_recurrence(recurrence_new)
    all_metrics.append(m)

all_metrics = np.array(all_metrics)

# Make one figure per RQA metric evolution
for i, name in enumerate(metrics_names):
    plt.figure()
    plt.plot(thresholds, all_metrics[:, i], marker='o')
    plt.title(f"{name} vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel(name)
    plt.grid(True)
    plt.show()

# Make a summary table
df_summary = pd.DataFrame(all_metrics, columns=metrics_names)
df_summary.insert(0, 'Threshold', thresholds)
print("\n==== RQA Metrics Across Thresholds ====\n")
print(df_summary.to_string(index=False))




# ------------------- plot recurrence plot
plt.figure(figsize=(10, 8))
plt.imshow(recurrence_np, cmap='binary', origin='lower')
plt.title(f"Recurrence Plot (childhood development)\n Threshold: {threshold}")
plt.xlabel("Time Index")
plt.ylabel("Time Index")
plt.colorbar(label="Recurrence")
plt.show()
plt.close()

# ------------------- fit RR vs threshold metrics_names[0] is RR

# Define models
def power_law(x, a, b):
    return a * x**b

def exponential(x, a, b):
    return a * np.exp(b * x)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# R^2 calculation
def r2(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Make one figure per RQA metric evolution
for i, name in enumerate([metrics_names[0]]):
    plt.figure()
    # plt.plot(thresholds, all_metrics[:, i], marker='o')

    x = thresholds
    y = all_metrics[:, 0]

    # Fit models
    popt_power, _ = curve_fit(power_law, x, y, maxfev=10000)
    popt_exp, _ = curve_fit(exponential, x, y, maxfev=10000)

    # Plot
    plt.plot(x, y, 'o', label='Data')
    plt.plot(x, power_law(x, *popt_power), label='Power law fit')
    plt.plot(x, exponential(x, *popt_exp), label='Exponential fit')

    # Predictions
    y_fit_power = power_law(x, *popt_power)
    y_fit_exp = exponential(x, *popt_exp)
    r2_power = r2(y, y_fit_power)
    r2_exp = r2(y, y_fit_exp)

    print(f'Power law R^2: {r2_power:.4f}')
    print(f'Exponential R^2: {r2_exp:.4f}')


    plt.title(f"{name} vs Threshold. Fit")
    plt.xlabel("Threshold")
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.show()

