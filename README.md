# spatial-efficiency-eeg

**Spatial Efficiency of Neural Oscillations (η) — EEG Analysis Pipeline**

[![Tests](https://github.com/alastairwaterman-bit/spatial-efficiency-eeg/actions/workflows/test.yml/badge.svg)](https://github.com/alastairwaterman-bit/spatial-efficiency-eeg/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19139967.svg)](https://doi.org/10.5281/zenodo.19139967)

## Overview

This repository contains the complete analysis pipeline for computing
**η (spatial efficiency)** and **||Δ|| (dissipation norm)** from EEG data.

**η** measures the fraction of total oscillatory energy converted into
organised spatial patterns relative to a stable reference:
```
η = ||Ψ|| / ||A||  ∈ [0, 1]
```

where **Ψ** is the five-band Effective Power Vector (amplitude weighted
by spatial pattern stability Sγ) and **A** is the total amplitude vector.

This work is part of the **research series** ([Author] 2026),
preprints available at Zenodo

### Key properties

- **Amplitude-independent**: immune to the LZC paradox under propofol sedation
- **No TMS required**: computed from continuous spontaneous EEG
- **Validated across four paradigms**: propofol sedation (×2 independent datasets), sleep staging, reversal learning
- **AUC = 1.000** for awake vs. moderate propofol sedation (Chennu dataset, N=20)
- **r = −1.000** in all subjects across NREM sleep stages (N=7)
- **Correctly ranks REM > N3** despite both being behaviourally unresponsive

---

## Key Results

### Propofol Sedation

| Dataset | Metric | Awake | Sedated | d | AUC |
|---------|--------|-------|---------|---|-----|
| Bajwa 2024 (N=21) | **η** | 0.893 | 0.330 | −2.15 | **0.988** |
| Bajwa 2024 (N=21) | LZC | 0.022 | 0.036 | +0.68 | 0.755 |
| Chennu 2016 (N=20) | **η** | 0.200 | 0.098 | −3.25 | **1.000** |
| Chennu 2016 (N=20) | LZC | 0.573 | 0.585 | +0.25 ns | 0.612 |

η dramatically outperforms LZC in both datasets.
LZC increases paradoxically under propofol; η is immune.

### Sleep Staging (ANPHY-Sleep, N=7)

| Stage | η | ||Δ|| (μV) | d vs Wake | p |
|-------|---|-----------|-----------|---|
| Wake  | 0.722 ± 0.040 | 3.85 | — | — |
| N1    | 0.464 ± 0.028 | 5.24 | −3.25 | 0.016 |
| N2    | 0.353 ± 0.025 | 7.97 | −5.29 | 0.016 |
| N3    | 0.265 ± 0.020 | 15.06 | −3.75 | 0.016 |
| **REM** | **0.507 ± 0.046** | 4.31 | −4.41 | 0.016 |

**REM > N3** (d=+1.78, p=0.016) despite both states being behaviourally
unresponsive. Spearman r = −1.000 in all 7 subjects across Wake→N1→N2→N3.

### Reversal Learning (ds004295, N=22)

η falls transiently at rule reversal (d=−2.58, p<0.001) in fully conscious
subjects — demonstrating sensitivity to intra-conscious phenomenal
reorganisation that temporal complexity metrics cannot detect.

---

## Installation
```bash
git clone https://github.com/[username]/spatial-efficiency-eeg.git
cd spatial-efficiency-eeg
pip install -r requirements.txt
```

### Requirements
```
numpy>=1.21
scipy>=1.7
mne>=1.0
scikit-learn>=1.0
antropy>=0.1.4
matplotlib>=3.4
pandas>=1.3
```

---

## Quick Start
```python
import numpy as np
from spatial_efficiency import SpatialEfficiencyPipeline

# EEG data: shape (n_channels, n_timepoints), in microvolts, average reference
data = np.random.randn(64, 76800)  # 64 ch, 5 min at 256 Hz
sfreq = 256.0

# Build reference from baseline (e.g. first 60 seconds)
pipe = SpatialEfficiencyPipeline(sfreq=sfreq, n_ch_top=20)
pipe.fit_reference(data[:, :int(60 * sfreq)])

# Compute metrics for any 30-second epoch
metrics = pipe.compute(data[:, int(60*sfreq):int(90*sfreq)])

print(f"η       = {metrics['eta']:.4f}")
print(f"||Δ||   = {metrics['norm_delta']:.3f} μV")
print(f"||Ψ||   = {metrics['norm_psi']:.3f} μV")
print(f"Ψ_c     = {metrics['psi_c']:.2f} Hz")
print(f"Sγ      = {[round(s,3) for s in metrics['stab']]}")
```

---

## Core Metrics

| Symbol | Name | Formula | Range | Interpretation |
|--------|------|---------|-------|----------------|
| Sγ(b) | Pattern stability | Pearson_r(pat, ref) | [−1, +1] | Spatial similarity to reference |
| Amp(b) | Amplitude | mean Hilbert envelope | [0, ∞) μV | Total oscillatory energy in band |
| Eff(b) | Effective Power | Amp × max(Sγ, 0) | [0, Amp] μV | Organised energy |
| Δ(b) | Dissipative Power | Amp − Eff | [0, Amp] μV | Disorganised energy |
| **η** | **Spatial efficiency** | **‖Ψ‖ / ‖A‖** | **[0, 1]** | **Fraction of energy organised** |
| ‖Δ‖ | Dissipation norm | √(Σ Δ(b)²) | [0, ∞) μV | Total disorganised energy |
| ‖Ψ‖ | Organised power norm | √(Σ Eff(b)²) | [0, ∞) μV | Total organised energy |
| Ψ_c | Frequency centroid | Σ(f_b·Eff(b)) / Σ(Eff(b)) | [0, 80] Hz | Dominant frequency of organised activity |

**Five frequency bands:**
δ (0.5–4 Hz), θ (4–8 Hz), α (8–13 Hz), β (13–30 Hz), γ (30–80 Hz)

---

## Repository Structure
```
spatial-efficiency-eeg/
├── README.md
├── requirements.txt
├── LICENSE
│
├── spatial_efficiency/
│   ├── __init__.py
│   ├── core.py          # All metric computations (η, ‖Δ‖, Sγ, Ψ)
│   ├── pipeline.py      # High-level SpatialEfficiencyPipeline class
│   └── utils.py         # Statistical utilities (AUC, Cohen's d, Wilcoxon)
│
├── examples/
│   ├── example_synthetic.py    # Works without any dataset download
│   ├── example_sedation.py     # Bajwa + Chennu propofol datasets
│   ├── example_sleep.py        # ANPHY-Sleep polysomnography
│   └── example_reversal.py     # ds004295 reversal learning
│
└── tests/
    └── test_core.py
```

---

## Reproducing Paper Results

All four datasets used in the analysis are openly available:
```bash
# Dataset 1: Bajwa et al. 2024 (OpenNeuro ds005620)
aws s3 sync --no-sign-request s3://openneuro.org/ds005620 ./ds005620/
python examples/example_sedation.py --dataset bajwa --data_path ./ds005620

# Dataset 2: Chennu et al. 2016
# Download from: https://doi.org/10.17863/CAM.68959
python examples/example_sedation.py --dataset chennu --data_path ./Sedation-RestingState

# Dataset 3: ANPHY-Sleep (Wei et al. 2024)
# Download from: https://osf.io/r26fh/
python examples/example_sleep.py --data_path ./ds_sleep_anphy

# Dataset 4: ds004295 reversal learning (OpenNeuro)
aws s3 sync --no-sign-request s3://openneuro.org/ds004295 ./ds004295/
python examples/example_reversal.py --data_path ./ds004295
```

---

## Theoretical Background

η operationalises the spatial organisation of neural oscillatory activity —
a property predicted to covary with conscious state by multiple theoretical
frameworks:

- **Global Workspace Theory** (Dehaene & Changeux 2011):
  high η reflects spatially coordinated workspace ignition
- **Integrated Information Theory** (Tononi et al. 2016):
  high η reflects spatial integration of oscillatory activity across channels
- **Thermodynamic accounts of neural computation**:
  η = efficiency of converting metabolic energy into organised
  representational structure; ‖Δ‖ = energy expended without organised output

For full theoretical discussion, see the preprint series:
[Author] [DOI]

---

## Citation

If you use this code, please cite the preprint series:
```bibtex
@software{[Author]2026code,
  title     = {spatial-efficiency-eeg: EEG Analysis Pipeline for
               Spatial Efficiency Metrics (η, ||Δ||)},
  author    = {[Author]},
  year      = {2026},
  doi       = {[DOI]},
  url       = {https://github.com/[username]/spatial-efficiency-eeg}
}

@misc{[Author]2026preprints,
  title     = {Effective Neural Power and the Phenomenal Vector —
               Preprint Series},
  author    = {[Author]},
  year      = {2026},
  doi       = {[DOI]},
  note      = {Preprint series, Zenodo}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Free to use, modify, and distribute with attribution.

---

## Contact

[Author]

