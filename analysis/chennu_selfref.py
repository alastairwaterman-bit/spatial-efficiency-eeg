"""
chennu_selfref.py
=================
Recompute η and angle(Ψ,Δ) for the Chennu et al. (2016) propofol sedation
dataset using a within-subject self-reference (first 60 s of baseline
recording), replacing the earlier population-reference computation that
yielded artificially low Sγ values.

Requires:
    - Sedation-RestingState/ folder on Drive with .set/.fdt files
      (MATLAB v7.3 / HDF5 format, EEGLAB)
    - eff_chennu.pkl (original computation, for subject/file mapping)
    - spatial_efficiency package on sys.path
    - h5py

Usage (Colab):
    exec(open('chennu_selfref.py').read())

Outputs:
    RESULTS_DIR/chennu_selfref_recalc.pkl   -- per-subject results

References:
    Chennu S et al. (2016). Brain connectivity dissociates responsiveness
        from drug exposure during propofol-induced transitions of
        consciousness. PLOS Comput Biol 12(1):e1004669.
    Waterman A (2026). Organised and Dissipated Neural Energy.
        Zenodo. https://doi.org/10.5281/zenodo.19233202
"""

import os
import sys
import pickle
import numpy as np
import h5py
from scipy.stats import wilcoxon, spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = '/content/gdrive/MyDrive'
SEDATION_DIR = os.path.join(BASE, 'Sedation-RestingState')
RESULTS_DIR  = os.path.join(BASE, 'results_eff_chennu')
PKG_PATH     = os.path.join(BASE, 'spatial_efficiency_eeg')

sys.path.insert(0, PKG_PATH)
from spatial_efficiency.core import build_all_references, BAND_ORDER
from spatial_efficiency.pipeline import SpatialEfficiencyPipeline


# ── angle(Ψ,Δ) ────────────────────────────────────────────────────────────────
def angle_psi_delta(eff_arr, amp_arr):
    """
    Angle between the Effective Power Vector Ψ and the Dissipative
    Vector Δ = max(A - Ψ, 0), in degrees.

    High angle = organised and dissipated energy distributed differently
    across the frequency hierarchy (structural tension).
    """
    Psi   = np.array(eff_arr, dtype=float)
    Delta = np.maximum(np.array(amp_arr, dtype=float) - Psi, 0.0)
    n1    = np.linalg.norm(Psi)
    n2    = np.linalg.norm(Delta)
    if n1 < 1e-10 or n2 < 1e-10:
        return np.nan
    cos_sim = np.dot(Psi, Delta) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))


# ── EEGLAB .set reader (MATLAB v7.3 / HDF5) ───────────────────────────────────
def read_set_hdf5(set_path, max_sec=None):
    """
    Read EEGLAB .set file saved in MATLAB v7.3 (HDF5) format.
    Data is stored in the companion .fdt file as float32, Fortran order.

    Returns:
        data      : ndarray (n_channels, n_times), average-referenced, μV
        sfreq     : float
        ch_names  : list of str
    """
    with h5py.File(set_path, 'r') as f:
        eeg    = f['EEG']
        sfreq  = float(np.array(eeg['srate']).squeeze())
        n_ch   = int(np.array(eeg['nbchan']).squeeze())
        n_pts  = int(np.array(eeg['pnts']).squeeze())
        n_tri  = int(np.array(eeg['trials']).squeeze())

        ch_names = [f'ch{i}' for i in range(n_ch)]
        try:
            chanlocs = eeg['chanlocs']
            if hasattr(chanlocs, 'dtype') and 'labels' in chanlocs.dtype.names:
                for i, row in enumerate(chanlocs):
                    name = row['labels']
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    if i < n_ch:
                        ch_names[i] = str(name).strip()
        except Exception:
            pass

    fdt_path = set_path.replace('.set', '.fdt')
    raw  = np.fromfile(fdt_path, dtype=np.float32,
                       count=n_ch * n_pts * n_tri)
    data = raw.reshape(n_ch, n_pts * n_tri, order='F').astype(np.float64)

    # Average reference
    data -= data.mean(axis=0, keepdims=True)

    if max_sec is not None:
        data = data[:, :int(max_sec * sfreq)]

    return data, sfreq, ch_names


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load original pkl for subject/file mapping
    orig_pkl = os.path.join(RESULTS_DIR, 'eff_chennu.pkl')
    with open(orig_pkl, 'rb') as f:
        chennu_orig = pickle.load(f)

    subjects = sorted(chennu_orig.keys())
    print(f"Processing {len(subjects)} subjects with self-reference...")

    results = []

    for sub in subjects:
        sub_data = chennu_orig[sub]

        # Baseline file (level 1)
        baseline_cond = f"{sub}_level1"
        if baseline_cond not in sub_data:
            print(f"  ✗ {sub}: no baseline condition"); continue

        fname_base   = sub_data[baseline_cond].get('fname', '')
        set_path_base = os.path.join(SEDATION_DIR, fname_base + '.set')
        if not os.path.exists(set_path_base):
            print(f"  ✗ {sub}: baseline file not found: {fname_base}"); continue

        try:
            data_base, sfreq, ch_names = read_set_hdf5(set_path_base)
            # Reference = first 60 s of baseline
            ref_samples  = min(int(60 * sfreq), data_base.shape[1])
            ref_patterns = build_all_references(
                data_base[:, :ref_samples], sfreq
            )
        except Exception as e:
            print(f"  ✗ {sub} baseline load error: {e}"); continue

        pipe = SpatialEfficiencyPipeline(sfreq=sfreq)
        pipe._ref_patterns = ref_patterns
        pipe._n_channels   = data_base.shape[0]

        sub_result = {'sub': sub}

        for level in [1, 2, 3, 4]:
            cond_key = f"{sub}_level{level}"
            if cond_key not in sub_data:
                continue
            fname    = sub_data[cond_key].get('fname', '')
            set_path = os.path.join(SEDATION_DIR, fname + '.set')
            if not os.path.exists(set_path):
                print(f"    ✗ {sub} level{level}: {fname} not found")
                continue
            try:
                data_lv, _, _ = read_set_hdf5(set_path)
                _, win_res = pipe.compute_sliding_window(
                    data_lv, window_sec=0.5, step_sec=0.1,
                    average_reference=False
                )
                etas  = [r['eta']        for r in win_res]
                angs  = [angle_psi_delta(r['eff'], r['amp']) for r in win_res]
                norms = [r['norm_delta'] for r in win_res]

                sub_result[f'level{level}'] = {
                    'eta':        float(np.nanmean(etas)),
                    'angle':      float(np.nanmean(angs)),
                    'norm_delta': float(np.nanmean(norms)),
                    'level_name': sub_data[cond_key].get('level_name', ''),
                    'n_windows':  len(etas),
                }
                print(f"  {sub} L{level} "
                      f"({sub_data[cond_key].get('level_name',''):<10}): "
                      f"η={np.nanmean(etas):.3f}, "
                      f"angle={np.nanmean(angs):.1f}°")
            except Exception as e:
                print(f"    ✗ {sub} level{level}: {e}")

        results.append(sub_result)

    # ── Summary ───────────────────────────────────────────────────────────────
    level_names = {1: 'Baseline', 2: 'Mild', 3: 'Moderate', 4: 'Recovery'}
    print(f"\n{'='*60}")
    print("CHENNU — self-reference recomputation")
    print(f"{'='*60}")
    print(f"{'Level':<12} {'η':>7} {'η SEM':>7} "
          f"{'angle':>8} {'angle SEM':>10} {'N':>4}")
    print("-"*52)

    for lvl in [1, 2, 3, 4]:
        etas = [r[f'level{lvl}']['eta']   for r in results
                if f'level{lvl}' in r]
        angs = [r[f'level{lvl}']['angle'] for r in results
                if f'level{lvl}' in r]
        n = len(etas)
        if n == 0:
            continue
        print(f"{level_names[lvl]:<12} {np.mean(etas):>7.3f} "
              f"{np.std(etas)/np.sqrt(n):>7.3f} "
              f"{np.mean(angs):>7.1f}° "
              f"{np.std(angs)/np.sqrt(n):>9.1f}° {n:>4}")

    # Spearman r(consciousness level, angle/η)
    all_lvls, all_angs, all_etas = [], [], []
    for r in results:
        for lvl in [1, 2, 3, 4]:
            if f'level{lvl}' in r:
                all_lvls.append(lvl)
                all_angs.append(r[f'level{lvl}']['angle'])
                all_etas.append(r[f'level{lvl}']['eta'])

    r_eta, p_eta = spearmanr(all_lvls, all_etas)
    r_ang, p_ang = spearmanr(all_lvls, all_angs)
    print(f"\nr(consciousness level, η)     = {r_eta:.3f}, p={p_eta:.4f}")
    print(f"r(consciousness level, angle) = {r_ang:.3f}, p={p_ang:.4f}")
    print(f"N = {len(all_lvls)} recordings")

    # Paired: baseline vs moderate
    pairs = [(r['level1']['eta'],   r['level3']['eta'],
              r['level1']['angle'], r['level3']['angle'])
             for r in results if 'level1' in r and 'level3' in r]
    if pairs:
        be, me, ba, ma = zip(*pairs)
        _, p_e = wilcoxon(be, me)
        d_e = (np.mean(be) - np.mean(me)) / np.std(
            np.array(be) - np.array(me), ddof=1)
        _, p_a = wilcoxon(ba, ma)
        d_a = (np.mean(ba) - np.mean(ma)) / np.std(
            np.array(ba) - np.array(ma), ddof=1)
        print(f"\nBaseline vs Moderate (paired, N={len(pairs)}):")
        print(f"  η:     {np.mean(be):.3f} vs {np.mean(me):.3f}, "
              f"d={d_e:.3f}, p={p_e:.4f}")
        print(f"  angle: {np.mean(ba):.1f}° vs {np.mean(ma):.1f}°, "
              f"d={d_a:.3f}, p={p_a:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, 'chennu_selfref_recalc.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved: {out_path}")


if __name__ == '__main__':
    main()
