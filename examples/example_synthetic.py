import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/content/spatial-efficiency-eeg')

from spatial_efficiency import SpatialEfficiencyPipeline

np.random.seed(42)
N_CH  = 32
SFREQ = 256.0
DUR_S = 30.0
N_T   = int(DUR_S * SFREQ)
T     = np.arange(N_T) / SFREQ

def make_epoch(alpha_amp, delta_amp, gamma_amp,
               alpha_coherence, delta_coherence, noise_amp=8.0):
    rng_a = np.random.RandomState(1)
    rng_d = np.random.RandomState(2)
    rng_g = np.random.RandomState(3)
    alpha_spatial = np.abs(rng_a.randn(N_CH)); alpha_spatial /= alpha_spatial.sum()
    delta_spatial = np.abs(rng_d.randn(N_CH)); delta_spatial /= delta_spatial.sum()
    gamma_spatial = np.abs(rng_g.randn(N_CH)); gamma_spatial /= gamma_spatial.sum()
    data = np.random.randn(N_CH, N_T) * noise_amp
    for ch in range(N_CH):
        aw = alpha_spatial[ch]*N_CH if alpha_coherence > 0.5 else np.random.uniform(0,2)
        dw = delta_spatial[ch]*N_CH if delta_coherence > 0.5 else np.random.uniform(0,2)
        gw = gamma_spatial[ch]*N_CH
        data[ch] += alpha_amp * aw * np.sin(2*np.pi*10*T  + np.random.uniform(0,0.3))
        data[ch] += delta_amp * dw * np.sin(2*np.pi*1.5*T + np.random.uniform(0,0.3))
        data[ch] += gamma_amp * gw * np.sin(2*np.pi*40*T)
    data -= data.mean(axis=0, keepdims=True)
    return data

print("Simulating EEG states...")

baseline_epochs = [
    make_epoch(3.0, 0.3, 0.5, 1.0, 1.0, noise_amp=6.0)
    for _ in range(8)
]

states = {
    "Awake":   make_epoch(3.0, 0.3, 0.5, 1.0, 1.0, noise_amp=6.0),
    "Sedated": make_epoch(0.3, 4.0, 0.05, 0.0, 0.0, noise_amp=8.0),
    "REM":     make_epoch(1.5, 0.2, 0.4, 1.0, 1.0, noise_amp=6.0),
    "N3":      make_epoch(0.2, 5.0, 0.05, 0.0, 0.0, noise_amp=5.0),
}

pipe = SpatialEfficiencyPipeline(sfreq=SFREQ, n_ch_top=20)
pipe.fit_reference_from_epochs(baseline_epochs)
print(pipe)
print()
print("=" * 58)
print(f"{'State':<12} {'eta':>8} {'||Delta|| (uV)':>16} {'Psi_c (Hz)':>12}")
print("-" * 58)

results = {}
for name, epoch in states.items():
    m = pipe.compute(epoch)
    results[name] = m
    print(f"{name:<12} {m['eta']:>8.4f} {m['norm_delta']:>16.3f} {m['psi_c']:>12.2f}")

print("=" * 58)
print()
print("Checking theoretical predictions:")
checks = [
    ("Awake > Sedated", results["Awake"]["eta"]   > results["Sedated"]["eta"]),
    ("REM   > N3",      results["REM"]["eta"]     > results["N3"]["eta"]),
    ("REM   > Sedated", results["REM"]["eta"]     > results["Sedated"]["eta"]),
    ("Awake > REM",     results["Awake"]["eta"]   > results["REM"]["eta"]),
]
for label, passed in checks:
    print(f"  {label}: {'PASS' if passed else 'FAIL'}")

labels   = list(states.keys())
eta_vals = [results[s]["eta"]        for s in labels]
nd_vals  = [results[s]["norm_delta"] for s in labels]
colors   = ["#2c7bb6", "#d7191c", "#1a6b3c", "#e67e22"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Spatial Efficiency (eta) - Synthetic EEG Demo", fontweight="bold")

ax = axes[0]
bars = ax.bar(range(4), eta_vals, color=colors, alpha=0.85)
ax.set_xticks(range(4)); ax.set_xticklabels(labels)
ax.set_ylabel("eta"); ax.set_ylim(0, 1.05)
ax.set_title("A. eta by state")
for bar, val in zip(bars, eta_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

ax = axes[1]
bars = ax.bar(range(4), nd_vals, color=colors, alpha=0.85)
ax.set_xticks(range(4)); ax.set_xticklabels(labels)
ax.set_ylabel("||Delta|| (uV)"); ax.set_title("B. Dissipation norm")
for bar, val in zip(bars, nd_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.2,
            f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

ax = axes[2]
for i,(s,c) in enumerate(zip(labels, colors)):
    ax.scatter(eta_vals[i], nd_vals[i], s=180, color=c, alpha=0.9,
               zorder=5, label=s, edgecolors="white", linewidths=0.8)
    ax.annotate(s, (eta_vals[i], nd_vals[i]),
                textcoords="offset points", xytext=(8,5), fontsize=9, color=c)
ax.set_xlabel("eta"); ax.set_ylabel("||Delta|| (uV)")
ax.set_title("C. eta-||Delta|| state space")
ax.set_xlim(-0.05, 1.05); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("example_synthetic_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: example_synthetic_results.png")
