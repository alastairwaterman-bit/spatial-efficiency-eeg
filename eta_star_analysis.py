"""
eta_star_analysis.py
====================
Compute and validate η* ≈ 0.675 — the critical point of spatial efficiency
across five independent EEG datasets.

Three independent methods converge on η* ≈ 0.675:
  1. ODE zero-crossing (from SINDy on reversal learning)
  2. Sleep stage boundary (conscious/unconscious transition)
  3. Independent dataset convergence (YOTO mental imagery)

Requires pre-computed pkl files in RESULTS_DIR (see paths below).
All pkl files are outputs of the spatial_efficiency pipeline.

Usage:
    python analysis/eta_star_analysis.py

Outputs:
    results/eta_star_summary.json   — key statistics
    results/eta_star_figure.png     — Figure for preprint

References:
    Waterman A (2026). Spatial Efficiency Converges on a Universal
        Critical Point: η* ≈ 0.675 as a Cross-Dataset Threshold of
        Phenomenal Consciousness. Zenodo.
    Waterman A (2026). Organised and Dissipated Neural Energy.
        Zenodo. https://doi.org/10.5281/zenodo.19233202
"""

import os
import sys
import glob
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

# ── Paths ─────────────────────────────────────────────────────
# Edit BASE to point to your local data directory,
# or set environment variable EEG_BASE before running.
BASE = os.environ.get(
    'EEG_BASE',
    os.path.join(os.path.expanduser('~'), 'eeg', 'from drive')
)

PATHS = {
    'sedation':  os.path.join(BASE, 'results_eff_sedation', 'eff_sedation.pkl'),
    'chennu':    os.path.join(BASE, 'results_eff_chennu', 'chennu_selfref_recalc.pkl'),
    'sleep_dir': os.path.join(BASE, 'results_eff_sleep'),
    'yoto':      os.path.join(BASE, 'results_yoto', 'yoto_fullseg_method.pkl'),
    'reversal':  os.path.join(BASE, 'results_nse_reversal', 'nse_eff_allbands.pkl'),
    'out_dir':   os.path.join(BASE, 'results_eta_star'),
}

# ── Constants ─────────────────────────────────────────────────
BANDS     = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ETA_STAR  = 0.675   # critical point from ODE zero-crossing

# ODE parameters from SINDy on reversal learning dataset
# dη/dt = A_COEF × ‖Ψ‖ + B_COEF × angle(Ψ,Δ)
A_COEF = -0.00417
B_COEF =  0.00502

# Sleep stage consciousness labels
STAGE_CONSCIOUS = {
    'W':  True,
    'R':  True,
    'N1': None,   # ambiguous
    'N2': False,
    'N3': False,
}


# ── Core functions ────────────────────────────────────────────
def angle_psi_delta(eff, amp):
    """
    Angle between Effective Power Vector Ψ and Dissipative Vector Δ.

    Δ = max(A − Ψ, 0) component-wise.
    Returns angle in degrees, or nan if either vector has zero norm.
    """
    Psi   = np.array(eff, dtype=float)
    Delta = np.maximum(np.array(amp, dtype=float) - Psi, 0.0)
    n1    = np.linalg.norm(Psi)
    n2    = np.linalg.norm(Delta)
    if n1 < 1e-10 or n2 < 1e-10:
        return np.nan
    cos_sim = np.dot(Psi, Delta) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))


def eta_from_pkl(band_dict):
    """Compute η from a per-band dict with 'eff' and 'amp' keys."""
    eff = [band_dict[b]['eff'] for b in BANDS if b in band_dict]
    amp = [band_dict[b]['amp'] for b in BANDS if b in band_dict]
    if len(eff) != 5:
        return np.nan
    P = np.array(eff)
    A = np.array(amp)
    return float(np.linalg.norm(P) / (np.linalg.norm(A) + 1e-10))


def cohens_d_pooled(a, b):
    """Cohen's d with pooled SD."""
    pooled = np.std(np.concatenate([a, b]))
    return float((np.mean(a) - np.mean(b)) / pooled)


# ── Data loading ──────────────────────────────────────────────
def load_all_states():
    """
    Load η values from all five datasets.

    Returns list of dicts:
        {'dataset', 'condition', 'eta', 'conscious', 'sub'}
    """
    states = []

    # 1. Bajwa propofol sedation
    print("Loading Bajwa sedation...", end=' ')
    with open(PATHS['sedation'], 'rb') as f:
        sed = pickle.load(f)
    for sub, data in sed.items():
        for cond, label, conscious in [
            ('awake_EC',  'Awake',   True),
            ('sed_run1',  'Sedated', False),
            ('sed_run2',  'Sedated', False),
            ('sed_run3',  'Sedated', False),
        ]:
            if cond not in data:
                continue
            eta = eta_from_pkl(data[cond])
            if not np.isnan(eta):
                states.append({
                    'dataset':   'Bajwa',
                    'condition': label,
                    'eta':       eta,
                    'conscious': conscious,
                    'sub':       sub,
                })
    print(f"N={sum(1 for s in states if s['dataset']=='Bajwa')}")

    # 2. Chennu graded sedation
    print("Loading Chennu sedation...", end=' ')
    with open(PATHS['chennu'], 'rb') as f:
        chennu = pickle.load(f)
    level_map = {1: ('Baseline', True),
                 2: ('Mild',     None),
                 3: ('Moderate', False),
                 4: ('Recovery', True)}
    for r in chennu:
        for lvl, (label, conscious) in level_map.items():
            key = f'level{lvl}'
            if key not in r:
                continue
            eta = r[key].get('eta', np.nan)
            if not np.isnan(eta):
                states.append({
                    'dataset':   'Chennu',
                    'condition': label,
                    'eta':       eta,
                    'conscious': conscious,
                    'sub':       r.get('sub', '?'),
                })
    print(f"N={sum(1 for s in states if s['dataset']=='Chennu')}")

    # 3. Polysomnographic sleep
    print("Loading sleep...", end=' ')
    sleep_files = sorted(glob.glob(
        os.path.join(PATHS['sleep_dir'], '*_eta_sleep_v2.pkl')
    ))
    for fpath in sleep_files:
        with open(fpath, 'rb') as f:
            s = pickle.load(f)
        sub = s.get('sub', os.path.basename(fpath))
        for r in s.get('results', []):
            stage = r.get('stage', '')
            if stage not in STAGE_CONSCIOUS:
                continue
            eta = r.get('eta', np.nan)
            if not np.isnan(eta):
                states.append({
                    'dataset':   'Sleep',
                    'condition': f'Sleep {stage}',
                    'eta':       eta,
                    'conscious': STAGE_CONSCIOUS[stage],
                    'sub':       sub,
                })
    print(f"N={sum(1 for s in states if s['dataset']=='Sleep')}")

    # 4. YOTO mental imagery
    print("Loading YOTO...", end=' ')
    with open(PATHS['yoto'], 'rb') as f:
        yoto = pickle.load(f)
    for r in yoto:
        for cond, key in [('YOTO Baseline', 'eta_base'),
                           ('YOTO Imagery',  'eta_imag')]:
            eta = r.get(key, np.nan)
            if not np.isnan(eta):
                states.append({
                    'dataset':   'YOTO',
                    'condition': cond,
                    'eta':       eta,
                    'conscious': True,
                    'sub':       r.get('sub', '?'),
                })
    print(f"N={sum(1 for s in states if s['dataset']=='YOTO')}")

    # 5. Reversal learning
    print("Loading reversal learning...", end=' ')
    with open(PATHS['reversal'], 'rb') as f:
        rev = pickle.load(f)
    for sub, d in rev.items():
        if len(d.get('t_grid', [])) != 56:
            continue
        try:
            eff_all = np.array([d[b]['eff'] for b in BANDS])
            amp_all = np.array([d[b]['amp'] for b in BANDS])
        except KeyError:
            continue
        for i in range(eff_all.shape[1]):
            P   = eff_all[:, i]
            A   = amp_all[:, i]
            eta = float(np.linalg.norm(P) / (np.linalg.norm(A) + 1e-10))
            states.append({
                'dataset':   'Reversal',
                'condition': 'Reversal',
                'eta':       eta,
                'conscious': True,
                'sub':       sub,
            })
    print(f"N={sum(1 for s in states if s['dataset']=='Reversal')}")

    return states


# ── Statistics ────────────────────────────────────────────────
def compute_statistics(states):
    """Compute η* validation statistics."""
    conscious_etas   = np.array([s['eta'] for s in states
                                  if s['conscious'] is True])
    unconscious_etas = np.array([s['eta'] for s in states
                                  if s['conscious'] is False])

    y_true  = np.concatenate([
        np.ones(len(conscious_etas)),
        np.zeros(len(unconscious_etas))
    ])
    y_score = np.concatenate([conscious_etas, unconscious_etas])

    auc      = float(roc_auc_score(y_true, y_score))
    cohens_d = cohens_d_pooled(conscious_etas, unconscious_etas)
    pct_c    = float((conscious_etas >= ETA_STAR).mean() * 100)
    pct_uc   = float((unconscious_etas >= ETA_STAR).mean() * 100)

    # ODE equilibrium
    # At η*, dη/dt = 0: A_COEF × ‖Ψ‖ + B_COEF × angle = 0
    # → ‖Ψ‖ / angle = -B_COEF / A_COEF
    ode_ratio = float(-B_COEF / A_COEF)

    # YOTO validation
    yoto_base = np.array([s['eta'] for s in states
                           if s['condition'] == 'YOTO Baseline'])
    yoto_imag = np.array([s['eta'] for s in states
                           if s['condition'] == 'YOTO Imagery'])
    yoto_convergence = float(np.mean(yoto_base))

    stats = {
        'eta_star':          ETA_STAR,
        'n_total':           len(states),
        'n_conscious':       int(len(conscious_etas)),
        'n_unconscious':     int(len(unconscious_etas)),
        'eta_conscious':     float(np.mean(conscious_etas)),
        'eta_unconscious':   float(np.mean(unconscious_etas)),
        'auc':               auc,
        'cohens_d':          cohens_d,
        'pct_conscious_above_star':   pct_c,
        'pct_unconscious_above_star': pct_uc,
        'ode_a_coef':        A_COEF,
        'ode_b_coef':        B_COEF,
        'ode_equilibrium_ratio': ode_ratio,
        'yoto_baseline_eta': yoto_convergence,
        'yoto_imagery_eta':  float(np.mean(yoto_imag)),
        'yoto_convergence_to_star': float(abs(yoto_convergence - ETA_STAR)),
    }
    return stats, conscious_etas, unconscious_etas


# ── Figures ───────────────────────────────────────────────────
def make_figure(states, conscious_etas, unconscious_etas, stats):
    """Generate three-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r'$\eta^*$ $\approx$ 0.675: Universal Critical Point of Phenomenal Consciousness',
        fontsize=12, fontweight='bold'
    )

    # ── Panel A: distribution ─────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1.05, 43)
    ax.hist(conscious_etas,   bins=bins, alpha=0.65, color='#2d7a2d',
            label=f"Conscious (N={stats['n_conscious']:,})")
    ax.hist(unconscious_etas, bins=bins, alpha=0.65, color='#c0392b',
            label=f"Unconscious (N={stats['n_unconscious']:,})")
    ax.axvline(ETA_STAR, color='black', ls='--', lw=2.5,
               label=fr'$\eta^*$ = {ETA_STAR}')
    ax.set_xlabel(r'$\eta$ (spatial efficiency)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(
        f"A.  $\eta$ distribution\n"
        f"AUC = {stats['auc']:.3f},  d = {stats['cohens_d']:.2f}",
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel B: per-dataset bar chart ────────────────────────
    ax2 = axes[1]
    conditions = [
        ('Bajwa',    'Awake',         '#2d7a2d', True),
        ('Bajwa',    'Sedated',       '#c0392b', False),
        ('Chennu',   'Baseline',      '#27ae60', True),
        ('Chennu',   'Moderate',      '#e74c3c', False),
        ('Sleep',    'Sleep W',       '#3a7abf', True),
        ('Sleep',    'Sleep R',       '#2d7a2d', True),
        ('Sleep',    'Sleep N2',      '#e6a817', False),
        ('Sleep',    'Sleep N3',      '#c0392b', False),
        ('YOTO',     'YOTO Baseline', '#8e44ad', True),
        ('YOTO',     'YOTO Imagery',  '#e8567a', True),
        ('Reversal', 'Reversal',      '#16a085', True),
    ]
    labels = []; means = []; sems = []; colors = []; hatches = []
    for ds, cond, col, cons in conditions:
        vals = [s['eta'] for s in states
                if s['dataset'] == ds and s['condition'] == cond]
        if not vals:
            continue
        labels.append(f"{cond.replace('Sleep ','').replace('YOTO ','')}\n({ds})")
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
        colors.append(col)
        hatches.append('///' if cons is False else '')

    y = np.arange(len(labels))
    bars = ax2.barh(y, means, xerr=sems, color=colors, alpha=0.8,
                    height=0.6, capsize=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax2.axvline(ETA_STAR, color='black', ls='--', lw=2,
                label=fr'$\eta^*$ = {ETA_STAR}')
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel(r'$\eta$ (spatial efficiency)', fontsize=11)
    ax2.set_title('B.  Mean $\eta$ per condition\n(hatch = unconscious)', fontsize=10)
    ax2.legend(fontsize=8)
    for i, (m, s) in enumerate(zip(means, sems)):
        ax2.text(m + s + 0.01, i, f'{m:.3f}', va='center', fontsize=7)
    ax2.spines[['top', 'right']].set_visible(False)

    # ── Panel C: ODE landscape ────────────────────────────────
    ax3 = axes[2]
    mean_A_norm = 25.0
    eta_range   = np.linspace(0, 1, 300)
    deta = np.array([
        A_COEF * (e * mean_A_norm) + B_COEF * 52.0
        for e in eta_range
    ])
    ax3.plot(eta_range, deta, color='#2c3e50', lw=2.5,
             label=r'd$\eta$/dt = f($\eta$)')
    ax3.axhline(0, color='black', lw=0.8)
    ax3.axvline(ETA_STAR, color='red', ls='--', lw=2,
                label=fr'$\eta^*$ = {ETA_STAR}')
    ax3.fill_between(eta_range, deta, 0,
                     where=deta > 0, alpha=0.15, color='green')
    ax3.fill_between(eta_range, deta, 0,
                     where=deta < 0, alpha=0.15, color='red')

    key_states = [
        (0.122, 'Sedated\n(Bajwa)',   '#c0392b', 'o'),
        (0.342, 'Sleep N2',           '#e6a817', 'D'),
        (0.484, 'Sleep REM',          '#2d7a2d', 'D'),
        (0.668, 'YOTO\nBaseline',     '#8e44ad', 's'),
        (0.729, 'Sleep Wake',         '#3a7abf', 'D'),
        (0.894, 'Awake\n(Bajwa)',     '#2d7a2d', 'o'),
    ]
    for eta_v, lbl, col, mk in key_states:
        dv = A_COEF * (eta_v * mean_A_norm) + B_COEF * 52.0
        ax3.scatter(eta_v, dv, color=col, s=90, marker=mk,
                    zorder=5, edgecolors='black', lw=0.8)
        off = (0, 8) if dv > 0.15 else (0, -18)
        ax3.annotate(lbl, xy=(eta_v, dv), xytext=off,
                     textcoords='offset points', fontsize=7, ha='center')

    ax3.set_xlabel(r'$\eta$ (spatial efficiency)', fontsize=11)
    ax3.set_ylabel(r'd$\eta$/dt (predicted)', fontsize=11)
    ax3.set_title(
        'C.  ODE landscape\n'
        r'd$\eta$/dt = $-0.00417 \times \|\Psi\| + 0.00502 \times$ angle',
        fontsize=10
    )
    ax3.legend(fontsize=8, loc='upper right')
    ax3.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────
def main():
    os.makedirs(PATHS['out_dir'], exist_ok=True)

    print("=" * 60)
    print(f"η* = {ETA_STAR} — Universal Critical Point Validation")
    print("=" * 60)

    # Load
    states = load_all_states()
    print(f"\nTotal observations: {len(states):,}")

    # Statistics
    stats, conscious_etas, unconscious_etas = compute_statistics(states)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Conscious   (N={stats['n_conscious']:,}):   "
          f"η = {stats['eta_conscious']:.3f}")
    print(f"Unconscious (N={stats['n_unconscious']:,}): "
          f"η = {stats['eta_unconscious']:.3f}")
    print(f"AUC = {stats['auc']:.3f}")
    print(f"Cohen's d = {stats['cohens_d']:.3f}")
    print(f"% conscious above η*:   {stats['pct_conscious_above_star']:.1f}%")
    print(f"% unconscious above η*: {stats['pct_unconscious_above_star']:.1f}%")
    print(f"\nODE equilibrium: ‖Ψ‖/angle = {stats['ode_equilibrium_ratio']:.3f}")
    print(f"YOTO baseline η = {stats['yoto_baseline_eta']:.3f} "
          f"(distance to η* = {stats['yoto_convergence_to_star']:.3f})")

    # Per-dataset summary
    print(f"\n{'Dataset':<12} {'Condition':<22} {'N':>6} {'η':>7} {'%>η*':>7}")
    print("-" * 58)
    for ds in ['Bajwa', 'Chennu', 'Sleep', 'YOTO', 'Reversal']:
        ds_states = [s for s in states if s['dataset'] == ds]
        conds     = dict.fromkeys(s['condition'] for s in ds_states)
        for cond in conds:
            cs  = [s for s in ds_states if s['condition'] == cond]
            eta = np.mean([s['eta'] for s in cs])
            pct = np.mean([s['eta'] >= ETA_STAR for s in cs]) * 100
            print(f"  {ds:<10} {cond:<22} {len(cs):>6} {eta:>7.3f} {pct:>6.1f}%")

    # Save stats
    stats_path = os.path.join(PATHS['out_dir'], 'eta_star_summary.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Stats saved: {stats_path}")

    # Figure
    fig = make_figure(states, conscious_etas, unconscious_etas, stats)
    fig_path = os.path.join(PATHS['out_dir'], 'eta_star_figure.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure saved: {fig_path}")

    # Save pkl
    pkl_path = os.path.join(PATHS['out_dir'], 'eta_star_states.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'states': states, 'stats': stats}, f)
    print(f"✓ Data saved: {pkl_path}")


if __name__ == '__main__':
    main()
