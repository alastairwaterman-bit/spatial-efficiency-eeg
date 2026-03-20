"""
spatial_efficiency/pipeline.py

High-level pipeline class for computing spatial efficiency metrics.

Usage
-----
    from spatial_efficiency import SpatialEfficiencyPipeline

    pipe = SpatialEfficiencyPipeline(sfreq=256.0)
    pipe.fit_reference(baseline_data)           # build reference
    metrics = pipe.compute(epoch)               # single epoch
    results = pipe.compute_epochs(epochs_list)  # list of epochs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .core import (
    BANDS, BAND_ORDER,
    build_all_references,
    build_reference_from_epochs,
    compute_all_metrics,
)


class SpatialEfficiencyPipeline:
    """
    Pipeline for computing spatial efficiency metrics from EEG data.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz. Must be >= 160 Hz for gamma band.
    n_ch_top : int
        Number of top channels selected by envelope variance.
        Default: 20. Reduce for datasets with fewer channels.
    bands : dict, optional
        Frequency band definitions {name: (lo_hz, hi_hz)}.
        Defaults to standard 5-band decomposition.

    Examples
    --------
    Basic usage with continuous data:

    >>> pipe = SpatialEfficiencyPipeline(sfreq=256.0)
    >>> pipe.fit_reference(baseline_data)       # (n_ch, n_t) in μV
    >>> metrics = pipe.compute(epoch)           # (n_ch, n_t) in μV
    >>> print(f"η = {metrics['eta']:.4f}")

    Sleep staging with 30-second epochs:

    >>> pipe = SpatialEfficiencyPipeline(sfreq=256.0)
    >>> pipe.fit_reference(wake_data)
    >>> results = pipe.compute_epochs(nrem_epochs)
    >>> eta_per_epoch = [r['eta'] for r in results]
    """

    def __init__(
        self,
        sfreq: float,
        n_ch_top: int = 20,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.sfreq    = float(sfreq)
        self.n_ch_top = int(n_ch_top)
        self.bands    = bands if bands is not None else BANDS
        self._ref_patterns: Optional[Dict[str, np.ndarray]] = None

        if sfreq < 160:
            import warnings
            warnings.warn(
                f"sfreq={sfreq} Hz is below 160 Hz. "
                "Gamma band (30-80 Hz) cannot be computed correctly. "
                "Consider resampling to at least 256 Hz.",
                UserWarning,
            )

    # ── Reference fitting ─────────────────────────────────────

    def fit_reference(
        self,
        data: np.ndarray,
        average_reference: bool = True,
    ) -> "SpatialEfficiencyPipeline":
        """
        Build reference spatial patterns from baseline data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Baseline EEG in microvolts.
            Recommended: wakefulness, pre-event baseline,
            or stable resting state.
        average_reference : bool
            Apply average reference before building patterns.
            Default: True.

        Returns
        -------
        self : SpatialEfficiencyPipeline
            For method chaining.

        Notes
        -----
        The reference encodes the "organised" spatial template
        of the baseline condition. Sγ measures how closely each
        subsequent epoch matches this template.
        """
        if average_reference:
            data = data - data.mean(axis=0, keepdims=True)

        self._ref_patterns = build_all_references(
            data, self.sfreq, self.bands
        )
        self._n_channels = data.shape[0]
        return self

    def fit_reference_from_epochs(
        self,
        epochs: List[np.ndarray],
        average_reference: bool = True,
    ) -> "SpatialEfficiencyPipeline":
        """
        Build reference from a list of baseline epochs.

        Parameters
        ----------
        epochs : list of ndarray, each shape (n_channels, n_times)
        average_reference : bool

        Returns
        -------
        self
        """
        if average_reference:
            epochs = [ep - ep.mean(axis=0, keepdims=True)
                      for ep in epochs]

        self._ref_patterns = {
            band: build_reference_from_epochs(
                epochs, self.sfreq, lo, hi
            )
            for band, (lo, hi) in self.bands.items()
        }
        self._n_channels = epochs[0].shape[0]
        return self

    def is_fitted(self) -> bool:
        """Return True if reference has been built."""
        return self._ref_patterns is not None

    # ── Computing metrics ─────────────────────────────────────

    def compute(
        self,
        data: np.ndarray,
        average_reference: bool = True,
    ) -> Dict:
        """
        Compute all metrics for a single EEG segment.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            EEG segment in microvolts.
        average_reference : bool
            Apply average reference. Default: True.

        Returns
        -------
        metrics : dict with keys:
            'eta'         : float  — spatial efficiency η ∈ [0,1]
            'norm_delta'  : float  — dissipation norm ||Δ|| (μV)
            'norm_psi'    : float  — organised power norm ||Ψ|| (μV)
            'psi_c'       : float  — frequency centroid Ψ_c (Hz)
            'stab'        : list[float] — Sγ per band
            'amp'         : list[float] — Amp per band (μV)
            'eff'         : list[float] — Eff per band (μV)
            'Psi'         : ndarray(5,) — Effective Power Vector
            'A'           : ndarray(5,) — Amplitude Vector
            'Delta_vec'   : ndarray(5,) — Dissipative Vector
            'band_results': dict — full per-band results

        Raises
        ------
        RuntimeError
            If fit_reference() has not been called.
        """
        self._check_fitted()
        if average_reference:
            data = data - data.mean(axis=0, keepdims=True)

        return compute_all_metrics(
            data,
            self.sfreq,
            self._ref_patterns,
            self.n_ch_top,
            self.bands,
        )

    def compute_epochs(
        self,
        epochs: List[np.ndarray],
        average_reference: bool = True,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Compute metrics for a list of EEG epochs.

        Parameters
        ----------
        epochs : list of ndarray, each shape (n_channels, n_times)
        average_reference : bool
        verbose : bool
            Print progress every 50 epochs.

        Returns
        -------
        results : list of dict
            One metrics dict per epoch (same format as compute()).
        """
        self._check_fitted()
        results = []
        for i, ep in enumerate(epochs):
            results.append(self.compute(ep, average_reference))
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(epochs)} epochs computed")
        return results

    def compute_sliding_window(
        self,
        data: np.ndarray,
        window_sec: float = 0.5,
        step_sec: float = 0.1,
        average_reference: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute metrics in a sliding window over continuous data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        window_sec : float
            Window length in seconds. Default: 0.5s.
        step_sec : float
            Step size in seconds. Default: 0.1s.
        average_reference : bool
        verbose : bool

        Returns
        -------
        t_centres : ndarray
            Centre time of each window in seconds.
        results : list of dict
            Metrics per window.
        """
        self._check_fitted()
        win_s  = int(window_sec * self.sfreq)
        step_s = int(step_sec   * self.sfreq)
        n_t    = data.shape[1]

        t_centres = []
        results   = []
        pos = 0
        while pos + win_s <= n_t:
            t_centres.append((pos + win_s / 2) / self.sfreq)
            seg = data[:, pos:pos + win_s]
            results.append(self.compute(seg, average_reference))
            pos += step_s
            if verbose and len(results) % 100 == 0:
                print(f"  {len(results)} windows computed "
                      f"({t_centres[-1]:.1f}s)")

        return np.array(t_centres), results

    # ── Convenience extractors ────────────────────────────────

    @staticmethod
    def extract_eta(results: List[Dict]) -> np.ndarray:
        """Extract η values from a list of metric dicts."""
        return np.array([r["eta"] for r in results])

    @staticmethod
    def extract_norm_delta(results: List[Dict]) -> np.ndarray:
        """Extract ||Δ|| values from a list of metric dicts."""
        return np.array([r["norm_delta"] for r in results])

    @staticmethod
    def extract_stab(results: List[Dict]) -> np.ndarray:
        """Extract Sγ matrix (n_epochs × 5) from results."""
        return np.array([r["stab"] for r in results])

    @staticmethod
    def extract_eff(results: List[Dict]) -> np.ndarray:
        """Extract Eff matrix (n_epochs × 5) from results."""
        return np.array([r["eff"] for r in results])

    # ── Internal ──────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted():
            raise RuntimeError(
                "Reference not built. Call fit_reference() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"SpatialEfficiencyPipeline("
            f"sfreq={self.sfreq}, "
            f"n_ch_top={self.n_ch_top}, "
            f"status={status})"
        )
