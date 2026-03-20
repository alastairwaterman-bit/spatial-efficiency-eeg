"""
spatial_efficiency/core.py

Core computation functions for spatial efficiency metrics.

Metrics:
    - Sγ (pattern stability): Pearson correlation with reference pattern
    - Amp (amplitude): mean Hilbert envelope
    - Eff (Effective Power): Amp × max(Sγ, 0)
    - Δ (Dissipative Power): Amp × (1 − max(Sγ, 0))
    - η (spatial efficiency): ||Ψ|| / ||A||
    - ||Δ|| (dissipation norm): Euclidean norm of dissipative vector
    - ||Ψ|| (organised power norm): Euclidean norm of Effective Power Vector
    - Ψ_c (frequency centroid): weighted mean frequency of organised power

Author: Alastair Waterman
License: MIT
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from typing import Dict, List, Optional, Tuple

# ── Band definitions ──────────────────────────────────────────────────────────

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

BAND_ORDER: List[str] = ["delta", "theta", "alpha", "beta", "gamma"]

BAND_CENTRES: Dict[str, float] = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.5,
    "beta":  21.5,
    "gamma": 55.0,
}

FREQ_VALS: np.ndarray = np.array(
    [BAND_CENTRES[b] for b in BAND_ORDER]
)


# ── Filtering ─────────────────────────────────────────────────────────────────

def bandpass_sos(
    data: np.ndarray,
    sfreq: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Apply zero-phase 4th-order Butterworth bandpass filter.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        EEG data in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    lo : float
        Low cutoff frequency in Hz.
    hi : float
        High cutoff frequency in Hz.

    Returns
    -------
    filtered : ndarray, shape (n_channels, n_times)
    """
    nyq = sfreq / 2.0
    lo_norm = max(lo / nyq, 1e-4)
    hi_norm = min(hi / nyq, 0.999)
    sos = butter(4, [lo_norm, hi_norm], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=1)


def hilbert_envelope(filtered: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous amplitude via Hilbert transform.

    Parameters
    ----------
    filtered : ndarray, shape (n_channels, n_times)

    Returns
    -------
    envelope : ndarray, shape (n_channels, n_times)
    """
    return np.abs(hilbert(filtered, axis=1))


# ── Top-channel selection ─────────────────────────────────────────────────────

def select_top_channels(
    envelope: np.ndarray,
    n_top: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top n_top channels by envelope variance.

    Parameters
    ----------
    envelope : ndarray, shape (n_channels, n_times)
    n_top : int
        Number of channels to select.

    Returns
    -------
    var_idx : ndarray, shape (n_top,)
        Indices of selected channels.
    env_top : ndarray, shape (n_top, n_times)
        Envelope of selected channels.
    """
    n_ch = envelope.shape[0]
    n_top = min(n_top, n_ch)
    variances = envelope.var(axis=1)
    var_idx = np.argsort(variances)[-n_top:]
    return var_idx, envelope[var_idx, :]


# ── Reference pattern ─────────────────────────────────────────────────────────

def build_reference(
    data: np.ndarray,
    sfreq: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Build spatial reference pattern from a data segment.

    The reference is the mean spatial Hilbert envelope across
    all channels and time points, for a given frequency band.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Baseline EEG segment in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    lo, hi : float
        Band limits in Hz.

    Returns
    -------
    ref_pattern : ndarray, shape (n_channels,)
        Mean spatial envelope pattern.
    """
    filt = bandpass_sos(data, sfreq, lo, hi)
    env  = hilbert_envelope(filt)
    return env.mean(axis=1)


def build_reference_from_epochs(
    epochs: List[np.ndarray],
    sfreq: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Build reference pattern from a list of epochs.

    Parameters
    ----------
    epochs : list of ndarray, each shape (n_channels, n_times)
    sfreq : float
    lo, hi : float

    Returns
    -------
    ref_pattern : ndarray, shape (n_channels,)
    """
    patterns = []
    for ep in epochs:
        filt = bandpass_sos(ep, sfreq, lo, hi)
        env  = hilbert_envelope(filt)
        patterns.append(env.mean(axis=1))
    return np.array(patterns).mean(axis=0)


def build_all_references(
    data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build reference patterns for all frequency bands at once.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Baseline EEG in microvolts.
    sfreq : float
    bands : dict, optional
        Band definitions. Defaults to BANDS.

    Returns
    -------
    ref_patterns : dict {band_name: ndarray(n_channels,)}
    """
    if bands is None:
        bands = BANDS
    return {
        band: build_reference(data, sfreq, lo, hi)
        for band, (lo, hi) in bands.items()
    }


# ── Pattern stability Sγ ─────────────────────────────────────────────────────

def compute_sgamma(
    env_top: np.ndarray,
    ref_pattern: np.ndarray,
    var_idx: np.ndarray,
) -> float:
    """
    Compute pattern stability Sγ.

    Sγ is the Pearson correlation between the current spatial
    envelope pattern (mean over time) and the reference pattern,
    restricted to the top channels.

    Parameters
    ----------
    env_top : ndarray, shape (n_top, n_times)
        Hilbert envelope of top channels.
    ref_pattern : ndarray, shape (n_channels,)
        Full reference pattern.
    var_idx : ndarray, shape (n_top,)
        Indices of top channels.

    Returns
    -------
    sgamma : float in [−1, +1]
    """
    pat   = env_top.mean(axis=1)
    pat_c = pat - pat.mean()

    ref   = ref_pattern[var_idx]
    ref_c = ref - ref.mean()

    denom = np.std(ref_c) * np.std(pat_c) + 1e-10
    return float(np.dot(ref_c, pat_c) / (len(ref_c) * denom))


# ── Per-band metrics ──────────────────────────────────────────────────────────

def compute_band_metrics(
    data: np.ndarray,
    sfreq: float,
    ref_patterns: Dict[str, np.ndarray],
    n_ch_top: int = 20,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute Amp, Sγ, Eff, and Δ for all frequency bands.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        EEG segment in microvolts (average reference).
    sfreq : float
        Sampling frequency in Hz.
    ref_patterns : dict {band: ndarray(n_channels,)}
        Reference spatial patterns per band.
    n_ch_top : int
        Number of top channels to use.
    bands : dict, optional
        Band definitions. Defaults to BANDS.

    Returns
    -------
    results : dict {band: {'amp', 'sgamma', 'eff', 'delta'}}
        Per-band scalar metrics.
    """
    if bands is None:
        bands = BANDS

    results = {}
    for band, (lo, hi) in bands.items():
        filt    = bandpass_sos(data, sfreq, lo, hi)
        env     = hilbert_envelope(filt)
        var_idx, env_top = select_top_channels(env, n_ch_top)

        amp   = float(env_top.mean())
        sgamma = compute_sgamma(env_top, ref_patterns[band], var_idx)
        eff   = amp * max(sgamma, 0.0)
        delta = amp * (1.0 - max(sgamma, 0.0))

        results[band] = {
            "amp":    amp,
            "sgamma": sgamma,
            "eff":    eff,
            "delta":  delta,
        }

    return results


# ── Five-band vectors ─────────────────────────────────────────────────────────

def assemble_vectors(
    band_results: Dict[str, Dict[str, float]],
    band_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble five-band vectors Ψ, A, Δ_vec from per-band results.

    Parameters
    ----------
    band_results : dict
        Output of compute_band_metrics().
    band_order : list, optional
        Order of bands. Defaults to BAND_ORDER.

    Returns
    -------
    A : ndarray (5,)
        Total amplitude vector [Amp(δ), ..., Amp(γ)].
    Psi : ndarray (5,)
        Effective Power Vector [Eff(δ), ..., Eff(γ)].
    Delta_vec : ndarray (5,)
        Dissipative Vector [Δ(δ), ..., Δ(γ)].

    Notes
    -----
    Exact decomposition: A = Psi + Delta_vec
    """
    if band_order is None:
        band_order = BAND_ORDER

    A         = np.array([band_results[b]["amp"]   for b in band_order])
    Psi       = np.array([band_results[b]["eff"]   for b in band_order])
    Delta_vec = np.array([band_results[b]["delta"] for b in band_order])

    return A, Psi, Delta_vec


# ── Scalar metrics ────────────────────────────────────────────────────────────

def compute_eta(Psi: np.ndarray, A: np.ndarray) -> float:
    """
    Compute spatial efficiency η = ||Ψ|| / ||A||.

    Parameters
    ----------
    Psi : ndarray (5,)
        Effective Power Vector.
    A : ndarray (5,)
        Amplitude Vector.

    Returns
    -------
    eta : float in [0, 1]

    Notes
    -----
    η = 1 when all energy is spatially organised (Sγ = 1 everywhere).
    η = 0 when no energy is spatially organised (Sγ ≤ 0 everywhere).
    η is amplitude-independent: proportional amplitude increase
    leaves η unchanged.
    """
    norm_psi = np.sqrt(np.sum(Psi ** 2))
    norm_a   = np.sqrt(np.sum(A ** 2))
    return float(norm_psi / (norm_a + 1e-10))


def compute_norm_delta(Delta_vec: np.ndarray) -> float:
    """
    Compute dissipation norm ||Δ|| = √(Σ Δ(b)²).

    Parameters
    ----------
    Delta_vec : ndarray (5,)
        Dissipative Vector.

    Returns
    -------
    norm_delta : float (μV)
    """
    return float(np.sqrt(np.sum(Delta_vec ** 2)))


def compute_norm_psi(Psi: np.ndarray) -> float:
    """
    Compute organised power norm ||Ψ|| = √(Σ Eff(b)²).

    Parameters
    ----------
    Psi : ndarray (5,)
        Effective Power Vector.

    Returns
    -------
    norm_psi : float (μV)
    """
    return float(np.sqrt(np.sum(Psi ** 2)))


def compute_psi_centroid(
    Psi: np.ndarray,
    freq_vals: Optional[np.ndarray] = None,
) -> float:
    """
    Compute frequency centroid of the Effective Power Vector.

    Ψ_c = Σ(f_b × Eff(b)) / Σ(Eff(b))

    Parameters
    ----------
    Psi : ndarray (5,)
        Effective Power Vector.
    freq_vals : ndarray (5,), optional
        Band centre frequencies in Hz. Defaults to FREQ_VALS.

    Returns
    -------
    psi_c : float (Hz)
    """
    if freq_vals is None:
        freq_vals = FREQ_VALS
    total = Psi.sum() + 1e-10
    return float(np.dot(freq_vals, Psi) / total)


def psi_angle(Psi1: np.ndarray, Psi2: np.ndarray) -> float:
    """
    Compute angle between two Effective Power Vectors in degrees.

    θ = arccos( Ψ₁·Ψ₂ / (||Ψ₁|| × ||Ψ₂||) )

    Parameters
    ----------
    Psi1, Psi2 : ndarray (5,)

    Returns
    -------
    angle : float in [0, 180] degrees
    """
    cos_sim = (
        np.dot(Psi1, Psi2)
        / (np.linalg.norm(Psi1) * np.linalg.norm(Psi2) + 1e-10)
    )
    return float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))


# ── Full pipeline (single segment) ───────────────────────────────────────────

def compute_all_metrics(
    data: np.ndarray,
    sfreq: float,
    ref_patterns: Dict[str, np.ndarray],
    n_ch_top: int = 20,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """
    Compute all spatial efficiency metrics for a single EEG segment.

    This is the main entry point for single-segment analysis.
    For continuous or epoched data, use SpatialEfficiencyPipeline
    from pipeline.py.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        EEG segment in microvolts (average reference applied).
    sfreq : float
        Sampling frequency in Hz.
    ref_patterns : dict {band: ndarray(n_channels,)}
        Reference patterns. Build with build_all_references().
    n_ch_top : int
        Number of top channels for spatial pattern computation.
    bands : dict, optional
        Band definitions. Defaults to BANDS.

    Returns
    -------
    metrics : dict with keys:
        'band_results' : dict — per-band {amp, sgamma, eff, delta}
        'A'           : ndarray (5,) — amplitude vector
        'Psi'         : ndarray (5,) — Effective Power Vector
        'Delta_vec'   : ndarray (5,) — Dissipative Vector
        'eta'         : float — spatial efficiency η ∈ [0,1]
        'norm_delta'  : float — dissipation norm ||Δ|| (μV)
        'norm_psi'    : float — organised power norm ||Ψ|| (μV)
        'psi_c'       : float — frequency centroid Ψ_c (Hz)
        'stab'        : list[float] — Sγ per band
        'amp'         : list[float] — Amp per band (μV)
        'eff'         : list[float] — Eff per band (μV)
    """
    if bands is None:
        bands = BANDS

    # Per-band computation
    band_results = compute_band_metrics(
        data, sfreq, ref_patterns, n_ch_top, bands
    )

    # Assemble vectors
    A, Psi, Delta_vec = assemble_vectors(band_results)

    # Scalar metrics
    metrics = {
        "band_results": band_results,
        "A":            A,
        "Psi":          Psi,
        "Delta_vec":    Delta_vec,
        "eta":          compute_eta(Psi, A),
        "norm_delta":   compute_norm_delta(Delta_vec),
        "norm_psi":     compute_norm_psi(Psi),
        "psi_c":        compute_psi_centroid(Psi),
        "stab": [band_results[b]["sgamma"] for b in BAND_ORDER],
        "amp":  [band_results[b]["amp"]    for b in BAND_ORDER],
        "eff":  [band_results[b]["eff"]    for b in BAND_ORDER],
    }

    return metrics


# ── Mirror criterion ──────────────────────────────────────────────────────────

def mirror_criterion(
    d_eff: Dict[str, float],
    d_delta: Dict[str, float],
    threshold: float = 0.2,
) -> List[str]:
    """
    Identify bands satisfying the mirror criterion for halt detection.

    A band satisfies the mirror criterion when:
    - Eff decreases (d_eff < 0)
    - Δ increases (d_delta > 0)
    - Both effect sizes exceed threshold

    This pattern identifies the frequency level at which
    spatial organisation is specifically disrupted.

    Parameters
    ----------
    d_eff : dict {band: Cohen's d for Eff change}
    d_delta : dict {band: Cohen's d for Δ change}
    threshold : float
        Minimum |d| for criterion (default 0.2, small effect).

    Returns
    -------
    mirror_bands : list of band names satisfying the criterion

    Examples
    --------
    >>> d_eff   = {'alpha': -0.64, 'beta': -0.10, ...}
    >>> d_delta = {'alpha': +0.63, 'beta': +0.05, ...}
    >>> mirror_criterion(d_eff, d_delta)
    ['alpha']   # alpha mirror — localised halt
    """
    mirror_bands = []
    for band in BAND_ORDER:
        de = d_eff.get(band, 0.0)
        dd = d_delta.get(band, 0.0)
        if (de < -threshold) and (dd > threshold):
            mirror_bands.append(band)
    return mirror_bands
