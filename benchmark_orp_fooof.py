"""
benchmark_orp_fooof.py
======================
Benchmark spatial efficiency (η) against Odd Ratio Product (ORP) and
Lempel-Ziv Complexity (LZC) on the Bajwa et al. (2024) propofol sedation
dataset (ds005620, OpenNeuro).

Also computes aperiodic slope (log-log Welch PSD fit, 2-40 Hz) to verify
that η is independent of the 1/f EEG component.

Requires:
    - ds005620 raw EEG downloaded to DS_PATH (awake_EC and sed_acq-rest_run-1
      files only; ~15 GB total)
    - spatial_efficiency package (core.py + pipeline.py) on sys.path
    - eff_sedation.pkl and lzc_sedation.pkl in RESULTS_DIR (pre-computed)
    - Google Drive mounted at /content/gdrive (if running in Colab)

Usage (Colab):
    # Mount Drive, then run this script cell by cell or as:
    # exec(open('benchmark_orp_fooof.py').read())

Usage (local):
    python benchmark_orp_fooof.py

Outputs:
    RESULTS_DIR/benchmark_pipeline_ORP_FOOOF.pkl   -- per-subject results
    RESULTS_DIR/benchmark_summary.txt              -- printed summary table

References:
    Hartmann S et al. (2019). Odds ratio product of sleep EEG as a
        continuous measure of sleep depth. J Clin Sleep Med 15(3):369-375.
    Waterman A (2026). Organised and Dissipated Neural Energy: A
        Two-Dimensional Marker of Phenomenal State from Resting EEG.
        Zenodo. https://doi.org/10.5281/zenodo.19233202
"""

import os
import sys
import re
import pickle
import numpy as np
from scipy.signal import welch
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import roc_auc_score

# ── Paths (edit as needed) ────────────────────────────────────────────────────
BASE        = '/content/gdrive/MyDrive'          # Google Drive root (Colab)
DS_PATH     = os.path.join(BASE, 'ds005620')     # raw EEG dataset
RESULTS_DIR = os.path.join(BASE, 'results_eff_sedation')
PKG_PATH    = os.path.join(BASE, 'spatial_efficiency_eeg')  # package root

sys.path.insert(0, PKG_PATH)
from spatial_efficiency.pipeline import SpatialEfficiencyPipeline
from spatial_efficiency.core import BAND_ORDER

# ── Subjects (excluding sub-1037 per dataset exclusion criterion) ─────────────
SUBJECTS = [
    '1010', '1016', '1017', '1022', '1024', '1033',
    '1036', '1045', '1046', '1054', '1055', '1057',
    '1060', '1061', '1062', '1064', '1067', '1068', '1071', '1074',
]


# ── BrainVision reader ────────────────────────────────────────────────────────
def parse_vhdr(vhdr_path):
    config, channels, section = {}, {}, None
    with open(vhdr_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1].lower()
                continue
            if '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip()
            if section == 'common infos':
                config[key.lower()] = val
            elif section == 'binary infos':
                config[key.lower()] = val
            elif section == 'channel infos':
                idx = int(re.sub(r'[^0-9]', '', key)) - 1
                parts = val.split(',')
                res = float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else 1.0
                channels[idx] = {'name': parts[0].strip(), 'resolution': res}
    sfreq = 1e6 / float(config.get('samplinginterval', 200))
    n_ch  = len(channels)
    return {
        'sfreq':      sfreq,
        'ch_names':   [channels[i]['name'] for i in range(n_ch)],
        'ch_res':     [channels[i]['resolution'] for i in range(n_ch)],
        'data_file':  config.get('datafile', '').strip(),
        'binary_fmt': config.get('binaryformat', 'int_16').lower(),
        'n_ch':       n_ch,
    }


def read_eeg(vhdr_path, max_sec=300, target_hz=256):
    """Read BrainVision EEG, apply average reference, downsample."""
    info  = parse_vhdr(vhdr_path)
    base  = os.path.dirname(vhdr_path)
    eeg_f = os.path.join(base, info['data_file']) if info['data_file'] \
            else vhdr_path.replace('.vhdr', '.eeg')
    n_ch  = info['n_ch']
    sfreq = info['sfreq']
    dtype = (np.int16  if 'int_16'  in info['binary_fmt'] else
             np.int32  if 'int_32'  in info['binary_fmt'] else np.float32)
    max_samples = int(max_sec * sfreq) * n_ch
    raw  = np.fromfile(eeg_f, dtype=dtype, count=max_samples)
    if len(raw) % n_ch != 0:
        raw = raw[:-(len(raw) % n_ch)]
    data = raw.reshape(-1, n_ch).T.astype(np.float64)
    for i, r in enumerate(info['ch_res']):
        data[i] *= r
    # Average reference
    data -= data.mean(axis=0, keepdims=True)
    # Downsample
    if sfreq > target_hz:
        factor = int(round(sfreq / target_hz))
        data   = data[:, ::factor]
        sfreq  = sfreq / factor
    return data, sfreq, info['ch_names']


# ── ORP ───────────────────────────────────────────────────────────────────────
def compute_orp(data, sfreq, ch_names=None):
    """
    Odd Ratio Product (Hartmann et al. 2019).

    Geometric mean of per-bin odds ratios p/(1-p) across 0.5-32 Hz,
    where p = fractional PSD in each Welch bin.
    Higher ORP = more awake.
    """
    preferred = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'F3', 'F4', 'FCz']
    if ch_names:
        idx = [i for i, c in enumerate(ch_names) if c in preferred]
        if not idx:
            idx = list(range(min(10, data.shape[0])))
    else:
        idx = list(range(min(10, data.shape[0])))

    eeg     = data[idx].mean(axis=0)
    nperseg = int(4 * sfreq)
    freqs, psd = welch(eeg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)

    mask = (freqs >= 0.5) & (freqs < 32.0)
    if mask.sum() == 0:
        return np.nan
    p = psd[mask] / (psd[mask].sum() + 1e-30)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(np.exp(np.mean(np.log(p / (1 - p)))))


# ── Aperiodic slope (FOOOF proxy) ─────────────────────────────────────────────
def compute_aperiodic_slope(data, sfreq, ch_names=None):
    """
    Aperiodic exponent from log-log linear fit of Welch PSD (2-40 Hz).
    More negative = steeper 1/f slope.
    """
    preferred = ['Fz', 'Cz', 'Pz', 'C3', 'C4']
    if ch_names:
        idx = [i for i, c in enumerate(ch_names) if c in preferred]
        if not idx:
            idx = list(range(min(5, data.shape[0])))
    else:
        idx = list(range(min(5, data.shape[0])))

    eeg     = data[idx].mean(axis=0)
    nperseg = int(4 * sfreq)
    freqs, psd = welch(eeg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)

    mask = (freqs >= 2) & (freqs <= 40)
    if mask.sum() < 5:
        return np.nan
    slope, _ = np.polyfit(np.log10(freqs[mask]),
                          np.log10(psd[mask] + 1e-30), 1)
    return float(slope)


# ── η via pipeline.py (sliding window) ───────────────────────────────────────
def compute_eta_sliding(data, sfreq, ref_sec=60):
    """
    Compute η using SpatialEfficiencyPipeline with 500 ms sliding windows.
    Reference = first ref_sec seconds of the recording.
    Returns mean η across all windows.
    """
    ref_samples  = min(int(ref_sec * sfreq), data.shape[1])
    ref_patterns = __import__('spatial_efficiency.core', fromlist=['build_all_references']) \
                       .build_all_references(data[:, :ref_samples], sfreq)
    pipe = SpatialEfficiencyPipeline(sfreq=sfreq)
    pipe._ref_patterns = ref_patterns
    pipe._n_channels   = data.shape[0]
    _, win_results = pipe.compute_sliding_window(
        data, window_sec=0.5, step_sec=0.1, average_reference=False
    )
    return float(np.nanmean([r['eta'] for r in win_results]))


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for sub in SUBJECTS:
        row = {'sub': sub}
        for cond, task in [
            ('awake', 'awake_acq-EC'),
            ('sed',   'sed_acq-rest_run-1'),
        ]:
            vhdr = os.path.join(
                DS_PATH,
                f"sub-{sub}/eeg/sub-{sub}_task-{task}_eeg.vhdr"
            )
            if not os.path.exists(vhdr):
                print(f"  ✗ sub-{sub} {cond}: file not found")
                for k in ['eta', 'orp', 'slope']:
                    row[f'{k}_{cond}'] = np.nan
                continue
            try:
                data, sfreq, ch_names = read_eeg(vhdr, max_sec=300)

                # η (reference built from awake_EC; same ref used for sedation)
                if cond == 'awake':
                    ref_samples  = min(int(60 * sfreq), data.shape[1])
                    from spatial_efficiency.core import build_all_references
                    _ref = build_all_references(data[:, :ref_samples], sfreq)
                    pipe = SpatialEfficiencyPipeline(sfreq=sfreq)
                    pipe._ref_patterns = _ref
                    pipe._n_channels   = data.shape[0]
                    row['_pipe']       = pipe   # reuse for sedation

                pipe = row.get('_pipe')
                if pipe is None:
                    row[f'eta_{cond}'] = np.nan
                else:
                    _, wins = pipe.compute_sliding_window(
                        data, window_sec=0.5, step_sec=0.1,
                        average_reference=False
                    )
                    row[f'eta_{cond}'] = float(np.nanmean(
                        [r['eta'] for r in wins]
                    ))

                row[f'orp_{cond}']   = compute_orp(data, sfreq, ch_names)
                row[f'slope_{cond}'] = compute_aperiodic_slope(
                    data, sfreq, ch_names
                )
                print(f"  ✓ sub-{sub} {cond}: "
                      f"η={row[f'eta_{cond}']:.3f}, "
                      f"ORP={row[f'orp_{cond}']:.5f}, "
                      f"slope={row[f'slope_{cond}']:.3f}")
            except Exception as e:
                print(f"  ✗ sub-{sub} {cond}: {e}")
                for k in ['eta', 'orp', 'slope']:
                    row[f'{k}_{cond}'] = np.nan

        # Remove internal pipeline object before saving
        row.pop('_pipe', None)
        results.append(row)

    # ── Load LZC from pre-computed pkl ────────────────────────────────────────
    lzc_path = os.path.join(RESULTS_DIR, 'lzc_sedation.pkl')
    if os.path.exists(lzc_path):
        with open(lzc_path, 'rb') as f:
            lzc_pkl = pickle.load(f)
        for r in results:
            sk = f"sub-{r['sub']}"
            r['lzc_awake'] = lzc_pkl.get(sk, {}).get('awake_EC', np.nan)
            r['lzc_sed']   = np.nanmean([
                lzc_pkl.get(sk, {}).get(c, np.nan)
                for c in ['sed_run1', 'sed_run2', 'sed_run3']
            ])

    # ── Statistics ────────────────────────────────────────────────────────────
    def auc_score(aw, sd, higher=True):
        v = np.concatenate([aw, sd])
        l = np.concatenate([np.ones(len(aw)), np.zeros(len(sd))])
        m = ~(np.isnan(v) | np.isnan(l))
        if m.sum() < 4:
            return np.nan
        a = roc_auc_score(l[m], v[m])
        return a if higher else 1 - a

    def paired(aw, sd, higher=True):
        m = ~(np.isnan(aw) | np.isnan(sd))
        a, s = aw[m], sd[m]
        d = (np.mean(a) - np.mean(s)) / np.std(a - s, ddof=1)
        if not higher:
            d = -d
        _, p = wilcoxon(a, s)
        return float(np.mean(a)), float(np.mean(s)), float(d), float(p)

    eta_aw  = np.array([r.get('eta_awake', np.nan) for r in results])
    eta_sd  = np.array([r.get('eta_sed',   np.nan) for r in results])
    orp_aw  = np.array([r.get('orp_awake', np.nan) for r in results])
    orp_sd  = np.array([r.get('orp_sed',   np.nan) for r in results])
    lzc_aw  = np.array([r.get('lzc_awake', np.nan) for r in results])
    lzc_sd  = np.array([r.get('lzc_sed',   np.nan) for r in results])
    sl_aw   = np.array([r.get('slope_awake', np.nan) for r in results])
    sl_sd   = np.array([r.get('slope_sed',   np.nan) for r in results])

    summary_lines = [
        f"{'='*68}",
        f"BENCHMARK TABLE — Bajwa ds005620, N={len(results)}",
        f"{'='*68}",
        f"{'Metric':<22} {'Awake':>7} {'Sed':>7} {'d':>7} {'AUC':>7} {'p':>9}",
        f"{'-'*68}",
    ]
    for name, aw, sd, higher in [
        ('η (pipeline.py)',   eta_aw, eta_sd, True),
        ('ORP',               orp_aw, orp_sd, True),
        ('LZC (pkl)',         lzc_aw, lzc_sd, False),
    ]:
        m_aw, m_sd, d, p = paired(aw, sd, higher)
        a = auc_score(aw, sd, higher)
        summary_lines.append(
            f"{name:<22} {m_aw:>7.3f} {m_sd:>7.3f} {d:>7.3f} {a:>7.3f} {p:>9.4f}"
        )
    summary_lines.append(f"{'-'*68}")

    # FOOOF robustness
    m_aw = ~(np.isnan(eta_aw) | np.isnan(sl_aw))
    r_aw, p_aw = pearsonr(eta_aw[m_aw], sl_aw[m_aw])
    m_sd = ~(np.isnan(eta_sd) | np.isnan(sl_sd))
    r_sd, p_sd = pearsonr(eta_sd[m_sd], sl_sd[m_sd])
    d_eta = eta_aw - eta_sd
    d_sl  = sl_aw  - sl_sd
    m_d   = ~(np.isnan(d_eta) | np.isnan(d_sl))
    r_d,  p_d  = pearsonr(d_eta[m_d], d_sl[m_d])

    summary_lines += [
        "",
        "FOOOF Robustness (aperiodic slope, Welch log-log 2-40 Hz):",
        f"  r(η, slope) awake:   r={r_aw:.3f}, p={p_aw:.3f}",
        f"  r(η, slope) sed:     r={r_sd:.3f}, p={p_sd:.3f}",
        f"  r(Δη, Δslope):       r={r_d:.3f},  p={p_d:.3f}",
        "  ✓ η independent of aperiodic slope" if (
            abs(r_aw) < 0.35 and p_aw > 0.1
        ) else f"  ⚠ r={r_aw:.3f} — report transparently",
    ]

    summary = "\n".join(summary_lines)
    print(summary)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_pkl = os.path.join(RESULTS_DIR, 'benchmark_pipeline_ORP_FOOOF.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'N': len(results),
            'per_subject': results,
            'auc': {
                'eta': float(auc_score(eta_aw, eta_sd, True)),
                'orp': float(auc_score(orp_aw, orp_sd, True)),
                'lzc': float(auc_score(lzc_aw, lzc_sd, False)),
            },
            'fooof': {
                'r_awake': float(r_aw), 'p_awake': float(p_aw),
                'r_sed':   float(r_sd), 'p_sed':   float(p_sd),
                'r_delta': float(r_d),  'p_delta': float(p_d),
            },
        }, f)
    print(f"\n✓ Saved: {out_pkl}")

    out_txt = os.path.join(RESULTS_DIR, 'benchmark_summary.txt')
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"✓ Saved: {out_txt}")


if __name__ == '__main__':
    main()
