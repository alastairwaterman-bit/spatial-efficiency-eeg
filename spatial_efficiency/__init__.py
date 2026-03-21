"""
spatial_efficiency
==================

EEG metrics for spatial efficiency of neural oscillations.

Main classes
------------
SpatialEfficiencyPipeline
    High-level pipeline for computing η and related metrics.

Main functions
--------------
compute_all_metrics
    Compute all metrics for a single EEG segment.
build_all_references
    Build reference patterns from baseline data.

Preprint series
---------------
[Author] (2026). Reaserch series.
Zenodo DOI: 10.5281/zenodo.19068027
"""

from .core import (
    # Band definitions
    BANDS,
    BAND_ORDER,
    BAND_CENTRES,
    FREQ_VALS,

    # Building blocks
    bandpass_sos,
    hilbert_envelope,
    select_top_channels,

    # Reference
    build_reference,
    build_reference_from_epochs,
    build_all_references,

    # Per-band metrics
    compute_sgamma,
    compute_band_metrics,

    # Vectors
    assemble_vectors,

    # Scalar metrics
    compute_eta,
    compute_norm_delta,
    compute_norm_psi,
    compute_psi_centroid,
    psi_angle,

    # Full pipeline
    compute_all_metrics,

    # Mirror criterion
    mirror_criterion,
)

from .pipeline import SpatialEfficiencyPipeline

__version__ = "1.0.0"
__author__  = "[Author]"
__license__ = "MIT"
__doi__     = "10.5281/zenodo.19068027"

__all__ = [
    "SpatialEfficiencyPipeline",
    "compute_all_metrics",
    "build_all_references",
    "build_reference",
    "build_reference_from_epochs",
    "compute_eta",
    "compute_norm_delta",
    "compute_norm_psi",
    "compute_psi_centroid",
    "psi_angle",
    "mirror_criterion",
    "BANDS",
    "BAND_ORDER",
    "BAND_CENTRES",
    "FREQ_VALS",
]
