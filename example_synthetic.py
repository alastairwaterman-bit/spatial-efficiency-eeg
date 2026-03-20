"""
examples/example_synthetic.py

Demonstration of spatial efficiency metrics on synthetic EEG data.
No dataset download required.

This example simulates:
    1. An "awake" state: spatially coherent alpha oscillation
    2. A "sedated" state: high-amplitude delta, spatially disorganised
    3. A "REM-like" state: low-amplitude, spatially coherent

Expected output:
    Awake:   η ~ 0.85,  ||Δ|| low
    Sedated: η ~ 0.15,  ||Δ|| high
    REM:     η ~ 0.70,  ||Δ|| low

This illustrates why η correctly ranks REM > sedated
even when sedated amplitude is higher.
"""

import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

from spatial_efficiency import SpatialEfficiencyPipeline

# ── Parameters ────────────────────────────────────────────────
np.random.seed(42)
N_CH   = 32
SFREQ  = 256.0
DUR_S  = 30.0          # 30-second epochs
N_T    = int(DUR_S * SFREQ)
T      = np.arange(N_T) / SFREQ

NOISE  = 5.0           # background noise (μV)

def make_epoch(
    alpha_amp: float,
    delta_amp: float,
    alpha_coherent: bool,
    delta_coherent: bool,
    noise: float = NOISE,
) -> np.ndarray:
    """
    Simulate a single EEG epoch.

    alpha_coherent=True  → channels share alpha phase (organised)
    delta_coherent=False → channels have random delta phase (disorganised)
    """
    data = np.random.randn(N_CH, N_T) * noise

    for ch in range(N_CH):
        # Alpha (10 Hz)
        alpha_phase = 0.0 if alpha_coherent else np.random.uniform(0, 2*np.pi)
        data[ch] += alpha_amp * np.sin(2 * np.pi * 10 * T + alpha_phase)

        # Delta (1 Hz)
        delta_phase = 0.0 if delta_coherent else np.random.uniform(0, 2*np.pi)
        data[ch] += delta_amp * np.sin(2 * np.pi * 1 * T + delta_phase)

    # Average reference
    data -= data.mean(axis=0, keepdims=True)
    return data


# ── Simulate states ───────────────────────────────────────────
print("Simulating EEG states...")

# Awake baseline (reference): coherent alpha, low delta
n_baseline = 5
baseline_epochs = [
    make_epoch(alpha_amp=15, delta_amp=2,
               alpha_coherent=True, delta_coherent=True)
    for _ in range(n_baseline)
]

# Test states
states = {
    "Awake": make_epoch(
        alpha_amp=15, delta_amp=2,
        alpha_coherent=True, delta_coherent=True
    ),
    "Sedated
(high amp,
disorganised)": make_epoch(
        alpha_amp=3, delta_amp=35,
        alpha_coherent=False, delta_coherent=False
    ),
    "REM-like
(low amp,
organised)": make_epoch(
        alpha_amp=8, delta_amp=2,
        alpha_coherent=True, delta_coherent=True
    ),
    "N3-like
(high amp,
partly organised)": make_epoch(
        alpha_amp=2, delta_amp=30,
        alpha_coherent=False, delta_coherent=True
    ),
}

# ── Pipeline ──────────────────────────────────────────────────
pipe = SpatialEfficiencyPipeline(sfreq=SFREQ, n_ch_top=16)
pipe.fit_reference_from_epochs(baseline_epochs)
print(f"Pipeline: {pipe}")

# ── Compute metrics ───────────────────────────────────────────
print("\nResults:")
print("=" * 60)
print(f"{'State':<28} {'η':>8} {'||Δ|| (μV)':>12} {'Ψ_c (Hz)':>10}")
print("-" * 60)

results = {}
for state_name, epoch in states.items():
    m = pipe.compute(epoch)
    results[state_name] = m
    label = state_name.replace("\n", " ")
    print(f"{label:<28} {m['eta']:>8.4f} {m['norm_delta']:>12.3f} "
          f"{m['psi_c']:>10.2f}")

print("=" * 60)
print("\nKey finding:")
print(f"  REM-like η ({results['REM-like\n(low amp,\norganised)']['eta']:.3f})"
      f" > Sedated η ({results['Sedated\n(high amp,\ndisorganised)']['eta']:.3f})")
print("  despite Sedated having much higher amplitude.")
print("  This is why η outperforms LZC for sedation detection.")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Spatial Efficiency (η) — Synthetic EEG Demo\n"
    "No dataset required",
    fontsize=11, fontweight="bold"
)

state_labels  = list(states.keys())
eta_vals      = [results[s]["eta"]        for s in state_labels]
nd_vals       = [results[s]["norm_delta"] for s in state_labels]
colors        = ["#2c7bb6", "#d7191c", "#1a6b3c", "#e67e22"]

# A: η bar chart
ax = axes[0]
bars = ax.bar(range(len(state_labels)), eta_vals,
              color=colors, alpha=0.85)
ax.set_xticks(range(len(state_labels)))
ax.set_xticklabels([s.replace("\n", "\n") for s in state_labels],
                   fontsize=8)
ax.set_ylabel("η (spatial efficiency)")
ax.set_ylim(0, 1.05)
ax.set_title("A. η by simulated state", fontsize=10)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
for bar, val in zip(bars, eta_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.02, f"{val:.3f}",
            ha="center", fontsize=9, fontweight="bold")

# B: ||Δ|| bar chart
ax = axes[1]
bars = ax.bar(range(len(state_labels)), nd_vals,
              color=colors, alpha=0.85)
ax.set_xticks(range(len(state_labels)))
ax.set_xticklabels([s.replace("\n", "\n") for s in state_labels],
                   fontsize=8)
ax.set_ylabel("||Δ|| (μV)")
ax.set_title("B. ||Δ|| by simulated state", fontsize=10)
for bar, val in zip(bars, nd_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.2, f"{val:.1f}",
            ha="center", fontsize=9, fontweight="bold")

# C: η–||Δ|| state space
ax = axes[2]
for i, (state, c) in enumerate(zip(state_labels, colors)):
    label = state.replace("\n", " ")
    ax.scatter(eta_vals[i], nd_vals[i],
               s=150, color=c, alpha=0.9,
               zorder=5, label=label,
               edgecolors="white", linewidths=0.8)
    ax.annotate(
        label, (eta_vals[i], nd_vals[i]),
        textcoords="offset points",
        xytext=(8, 5), fontsize=7, color=c
    )
ax.set_xlabel("η (spatial efficiency)")
ax.set_ylabel("||Δ|| (μV)")
ax.set_title("C. η–||Δ|| state space", fontsize=10)
ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("example_synthetic_results.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: example_synthetic_results.png")
