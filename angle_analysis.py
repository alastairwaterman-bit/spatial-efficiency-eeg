"""
angle_analysis.py
=================
Compute angle(Ψ,Δ) across all four EEG datasets and produce the summary
table reported in Waterman (2026).

angle(Ψ,Δ) = arccos(Ψ·Δ / (||Ψ|| ||Δ||))

where Ψ = Effective Power Vector [Eff(δ), Eff(θ), Eff(α), Eff(β), Eff(γ)]
and   Δ = max(A - Ψ, 0) component-wise.

A high angle indicates that organised and dissipated energy are distributed
differently across the frequency hierarchy — maximum when one component
is concentrated in fast bands and the other in slow bands (e.g. N2 sleep
spindles: fast bands organised, slow bands dissipated).

Requires pre-computed pkl files:
    results_eff_sedation/eff_sedation.pkl
    results_eff_chennu/chennu_selfref_recalc.pkl
    results_eff_sleep/*_eta_sleep_v2.pkl
    results_nse_reversal/nse_eff_allbands.pkl

Usage:
    exec(open('angle_analysis.py').read())   # Colab
    python angle_analysis.py                 # local

Outputs:
    results_eff_sedation/angle_summary.pkl
    results_eff_sedation/angle_summary.txt
"""

import os
import sys
import glob
import pickle
import numpy as np
from scipy.stats import wilcoxon, spearmanr

BASE = '/content/gdrive/MyDrive'

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']


# ── Core function ─────────────────────────────────────────────────────────────
def angle_psi_delta(eff_arr, amp_arr):
    """
    Angle between Ψ (Effective Power Vector) and Δ (Dissipative Vector)
    in degrees. Returns nan if either vector has zero norm.
    """
    Psi   = np.array(eff_arr, dtype=float)
    Delta = np.maximum(np.array(amp_arr, dtype=float) - Psi, 0.0)
    n1    = np.linalg.norm(Psi)
    n2    = np.linalg.norm(Delta)
    if n1 < 1e-10 or n2 < 1e-10:
        return np.nan
    cos_sim = np.dot(Psi, Delta) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))


def angle_from_band_dict(band_data):
    """Compute angle from a per-band dict with 'eff' and 'amp' keys."""
    eff, amp = [], []
    for b in BANDS:
        if b not in band_data:
            continue
        e = band_data[b]['eff']
        a = band_data[b]['amp']
        if hasattr(e, '__len__'):
            e = np.nanmean(e)
        if hasattr(a, '__len__'):
            a = np.nanmean(a)
        eff.append(e)
        amp.append(a)
    if len(eff) < 5:
        return np.nan
    return angle_psi_delta(eff, amp)


def eta_from_band_dict(band_data):
    """Compute η from a per-band dict."""
    psi_sq = amp_sq = 0.0
    for b in BANDS:
        if b not in band_data:
            continue
        e = band_data[b]['eff']
        a = band_data[b]['amp']
        if hasattr(e, '__len__'):
            e = np.nanmean(e)
        if hasattr(a, '__len__'):
            a = np.nanmean(a)
        psi_sq += e ** 2
        amp_sq += a ** 2
    return float(np.sqrt(psi_sq) / np.sqrt(amp_sq)) if amp_sq > 0 else np.nan


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / np.std(diff, ddof=1))


# ── 1. Bajwa sedation ─────────────────────────────────────────────────────────
def analyse_bajwa():
    path = os.path.join(BASE, 'results_eff_sedation', 'eff_sedation.pkl')
    with open(path, 'rb') as f:
        sed = pickle.load(f)

    results = []
    for sub, sub_data in sed.items():
        if 'awake_EC' not in sub_data:
            continue
        sd_etas, sd_angs = [], []
        for cond in ['sed_run1', 'sed_run2', 'sed_run3']:
            if cond in sub_data:
                sd_etas.append(eta_from_band_dict(sub_data[cond]))
                sd_angs.append(angle_from_band_dict(sub_data[cond]))
        if not sd_etas:
            continue
        results.append({
            'sub':      sub,
            'eta_aw':   eta_from_band_dict(sub_data['awake_EC']),
            'angle_aw': angle_from_band_dict(sub_data['awake_EC']),
            'eta_sed':  np.nanmean(sd_etas),
            'angle_sed': np.nanmean(sd_angs),
        })

    aw_ang = np.array([r['angle_aw']  for r in results])
    sd_ang = np.array([r['angle_sed'] for r in results])
    _, p   = wilcoxon(aw_ang, sd_ang)
    d      = cohens_d(aw_ang, sd_ang)

    print(f"\n=== BAJWA SEDATION (N={len(results)}) ===")
    print(f"  angle awake:   {np.mean(aw_ang):.1f}° ± {np.std(aw_ang):.1f}°")
    print(f"  angle sedated: {np.mean(sd_ang):.1f}° ± {np.std(sd_ang):.1f}°")
    print(f"  d={d:.3f}, p={p:.4f}")

    return results


# ── 2. Chennu graded sedation ─────────────────────────────────────────────────
def analyse_chennu():
    path = os.path.join(BASE, 'results_eff_chennu', 'chennu_selfref_recalc.pkl')
    with open(path, 'rb') as f:
        chennu = pickle.load(f)

    level_names = {1: 'Baseline', 2: 'Mild', 3: 'Moderate', 4: 'Recovery'}
    all_lvls, all_angs, all_etas = [], [], []

    print(f"\n=== CHENNU SEDATION (N={len(chennu)}) ===")
    for lvl in [1, 2, 3, 4]:
        etas = [r[f'level{lvl}']['eta']   for r in chennu if f'level{lvl}' in r]
        angs = [r[f'level{lvl}']['angle'] for r in chennu if f'level{lvl}' in r]
        if not etas:
            continue
        print(f"  {level_names[lvl]:<12}: "
              f"η={np.mean(etas):.3f}, angle={np.mean(angs):.1f}°, "
              f"N={len(etas)}")
        all_lvls.extend([lvl] * len(angs))
        all_angs.extend(angs)
        all_etas.extend(etas)

    r_ang, p_ang = spearmanr(all_lvls, all_angs)
    r_eta, p_eta = spearmanr(all_lvls, all_etas)
    print(f"  r(level, angle) = {r_ang:.3f}, p={p_ang:.4f}")
    print(f"  r(level, η)     = {r_eta:.3f}, p={p_eta:.4f}")

    return chennu


# ── 3. Sleep staging ──────────────────────────────────────────────────────────
def analyse_sleep():
    sleep_files = sorted(glob.glob(
        os.path.join(BASE, 'results_eff_sleep', '*_eta_sleep_v2.pkl')
    ))
    print(f"\n=== SLEEP ({len(sleep_files)} subjects) ===")

    stage_data = {s: {'eta': [], 'angle': []}
                  for s in ['W', 'R', 'N1', 'N2', 'N3']}

    for fpath in sleep_files:
        with open(fpath, 'rb') as f:
            slp = pickle.load(f)

        sub_stage = {}
        for ep in slp.get('results', []):
            stage = ep.get('stage', '')
            if stage not in stage_data:
                continue
            eff = ep.get('eff', [])
            amp = ep.get('amp', [])
            if len(eff) != 5 or len(amp) != 5:
                continue
            ang = angle_psi_delta(eff, amp)
            eta = ep.get('eta', np.nan)
            sub_stage.setdefault(stage, {'eta': [], 'angle': []})
            sub_stage[stage]['eta'].append(eta)
            sub_stage[stage]['angle'].append(ang)

        for stage in sub_stage:
            stage_data[stage]['eta'].append(
                np.nanmean(sub_stage[stage]['eta'])
            )
            stage_data[stage]['angle'].append(
                np.nanmean(sub_stage[stage]['angle'])
            )

    stage_labels = {'W': 'Wake', 'N1': 'N1', 'N2': 'N2',
                    'N3': 'N3', 'R': 'REM'}
    for s in ['W', 'N1', 'N2', 'N3', 'R']:
        d = stage_data[s]
        if not d['eta']:
            continue
        n = len(d['eta'])
        print(f"  {stage_labels[s]:<5}: "
              f"η={np.nanmean(d['eta']):.3f} ± "
              f"{np.nanstd(d['eta'])/np.sqrt(n):.3f}, "
              f"angle={np.nanmean(d['angle']):.1f}° ± "
              f"{np.nanstd(d['angle'])/np.sqrt(n):.1f}°, "
              f"N={n}")

    # Key comparisons
    print("\n  Key comparisons (Wilcoxon, subject-level):")
    for (s1, s2, lbl) in [
        ('N2', 'W',  'N2 vs Wake (angle — N2 paradox)'),
        ('N2', 'N3', 'N2 vs N3 (angle)'),
        ('N2', 'R',  'N2 vs REM (angle)'),
        ('R',  'N3', 'REM vs N3 (η)'),
    ]:
        key = 'angle' if 'angle' in lbl.lower() else 'eta'
        d1  = stage_data[s1][key]
        d2  = stage_data[s2][key]
        n   = min(len(d1), len(d2))
        if n < 3:
            continue
        _, p = wilcoxon(d1[:n], d2[:n])
        d    = cohens_d(d1[:n], d2[:n])
        print(f"    {lbl}: "
              f"{np.mean(d1[:n]):.1f} vs {np.mean(d2[:n]):.1f}, "
              f"d={d:.3f}, p={p:.4f}")

    return stage_data


# ── 4. Reversal learning ──────────────────────────────────────────────────────
def analyse_reversal():
    path = os.path.join(BASE, 'results_nse_reversal', 'nse_eff_allbands.pkl')
    with open(path, 'rb') as f:
        rev = pickle.load(f)

    t_len = 56
    subs  = [s for s in rev if len(rev[s].get('t_grid', [])) == t_len]
    t_grid = np.array(rev[subs[0]]['t_grid'])

    print(f"\n=== REVERSAL LEARNING (N={len(subs)}) ===")

    base_ang, post_ang = [], []
    base_eta, post_eta = [], []

    for sub in subs:
        sub_data = rev[sub]
        t   = np.array(sub_data['t_grid'])
        try:
            eff_all = np.array([sub_data[b]['eff'] for b in BANDS])
            amp_all = np.array([sub_data[b]['amp'] for b in BANDS])
        except Exception:
            continue

        ang_ts = np.array([
            angle_psi_delta(eff_all[:, i], amp_all[:, i])
            for i in range(eff_all.shape[1])
        ])
        eta_ts = np.array([
            float(np.linalg.norm(eff_all[:, i]) /
                  (np.linalg.norm(amp_all[:, i]) + 1e-10))
            for i in range(eff_all.shape[1])
        ])

        base_mask = t < -1.5
        post_mask = t > 0.5

        if base_mask.sum() > 0 and post_mask.sum() > 0:
            base_ang.append(np.nanmean(ang_ts[base_mask]))
            post_ang.append(np.nanmean(ang_ts[post_mask]))
            base_eta.append(np.nanmean(eta_ts[base_mask]))
            post_eta.append(np.nanmean(eta_ts[post_mask]))

    n = min(len(base_ang), len(post_ang))

    print(f"  angle baseline (t<-1.5s): {np.nanmean(base_ang):.1f}°")
    print(f"  angle post-rev (t>+0.5s): {np.nanmean(post_ang):.1f}°")
    if n > 2:
        _, p_ang = wilcoxon(base_ang[:n], post_ang[:n])
        d_ang    = cohens_d(post_ang[:n], base_ang[:n])
        print(f"  d={d_ang:.3f}, p={p_ang:.4f}")

    print(f"  η baseline:    {np.nanmean(base_eta):.3f}")
    print(f"  η post-rev:    {np.nanmean(post_eta):.3f}")
    if n > 2:
        _, p_eta = wilcoxon(base_eta[:n], post_eta[:n])
        d_eta    = cohens_d(post_eta[:n], base_eta[:n])
        print(f"  η d={d_eta:.3f}, p={p_eta:.4f}")

    return {'base_ang': base_ang, 'post_ang': post_ang,
            'base_eta': base_eta, 'post_eta': post_eta,
            't_grid': t_grid.tolist()}


# ── Run all & save ─────────────────────────────────────────────────────────────
def main():
    bajwa   = analyse_bajwa()
    chennu  = analyse_chennu()
    sleep   = analyse_sleep()
    reversal = analyse_reversal()

    # ── Cross-dataset summary table ───────────────────────────────────────────
    print(f"\n{'='*62}")
    print("ANGLE(Ψ,Δ) SUMMARY ACROSS DATASETS")
    print(f"{'='*62}")
    print(f"{'State':<25} {'η':>7} {'angle':>8}")
    print("-"*44)

    # Bajwa
    aw_etas = [r['eta_aw']   for r in bajwa]
    aw_angs = [r['angle_aw'] for r in bajwa]
    sd_etas = [r['eta_sed']  for r in bajwa]
    sd_angs = [r['angle_sed'] for r in bajwa]
    print(f"  {'Awake (Bajwa)':<23} {np.mean(aw_etas):>7.3f} "
          f"{np.mean(aw_angs):>7.1f}°")
    print(f"  {'Sedated (Bajwa)':<23} {np.mean(sd_etas):>7.3f} "
          f"{np.mean(sd_angs):>7.1f}°")

    # Chennu
    for lvl, name in [(1, 'Baseline (Chennu)'), (3, 'Moderate (Chennu)')]:
        etas = [r[f'level{lvl}']['eta']   for r in chennu if f'level{lvl}' in r]
        angs = [r[f'level{lvl}']['angle'] for r in chennu if f'level{lvl}' in r]
        print(f"  {name:<23} {np.mean(etas):>7.3f} {np.mean(angs):>7.1f}°")

    # Sleep
    stage_labels = {'W': 'Sleep Wake', 'N1': 'Sleep N1', 'N2': 'Sleep N2',
                    'N3': 'Sleep N3', 'R': 'Sleep REM'}
    for s in ['W', 'N1', 'N2', 'N3', 'R']:
        if sleep[s]['eta']:
            print(f"  {stage_labels[s]:<23} "
                  f"{np.nanmean(sleep[s]['eta']):>7.3f} "
                  f"{np.nanmean(sleep[s]['angle']):>7.1f}°")

    print("-"*44)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.join(BASE, 'results_eff_sedation')
    os.makedirs(out_dir, exist_ok=True)

    out_pkl = os.path.join(out_dir, 'angle_summary.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'bajwa':    bajwa,
            'chennu':   chennu,
            'sleep':    sleep,
            'reversal': reversal,
        }, f)
    print(f"\n✓ Saved: {out_pkl}")


if __name__ == '__main__':
    main()
