
import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from spatial_efficiency.core import (
    BANDS, BAND_ORDER, FREQ_VALS,
    bandpass_sos,
    hilbert_envelope,
    select_top_channels,
    build_reference,
    build_all_references,
    compute_sgamma,
    compute_band_metrics,
    assemble_vectors,
    compute_eta,
    compute_norm_delta,
    compute_norm_psi,
    compute_psi_centroid,
    compute_all_metrics,
    mirror_criterion,
)
from spatial_efficiency.pipeline import SpatialEfficiencyPipeline

# ── Fixtures ──────────────────────────────────────────────────

SFREQ  = 256.0
N_CH   = 32
N_T    = int(10 * SFREQ)  # 10 seconds

@pytest.fixture
def rng():
    return np.random.RandomState(42)

@pytest.fixture
def baseline_data(rng):
    """Organised baseline: coherent alpha oscillation."""
    t    = np.arange(N_T) / SFREQ
    data = rng.randn(N_CH, N_T) * 5.0
    spatial = np.abs(rng.randn(N_CH))
    for ch in range(N_CH):
        data[ch] += 20.0 * spatial[ch] * np.sin(2 * np.pi * 10 * t)
    data -= data.mean(axis=0, keepdims=True)
    return data

@pytest.fixture
def organised_epoch(rng):
    """Epoch resembling baseline: high eta expected."""
    t    = np.arange(N_T) / SFREQ
    data = rng.randn(N_CH, N_T) * 5.0
    spatial = np.abs(np.random.RandomState(42).randn(N_CH))
    for ch in range(N_CH):
        data[ch] += 18.0 * spatial[ch] * np.sin(2 * np.pi * 10 * t)
    data -= data.mean(axis=0, keepdims=True)
    return data

@pytest.fixture
def disorganised_epoch(rng):
    """Epoch with random spatial structure: low eta expected."""
    t    = np.arange(N_T) / SFREQ
    data = rng.randn(N_CH, N_T) * 5.0
    for ch in range(N_CH):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] += 25.0 * rng.uniform(0, 1) * np.sin(2*np.pi*1.5*t + phase)
    data -= data.mean(axis=0, keepdims=True)
    return data

@pytest.fixture
def ref_patterns(baseline_data):
    return build_all_references(baseline_data, SFREQ)

@pytest.fixture
def pipeline(baseline_data):
    pipe = SpatialEfficiencyPipeline(sfreq=SFREQ, n_ch_top=16)
    pipe.fit_reference(baseline_data)
    return pipe


# ── Tests: bandpass_sos ───────────────────────────────────────

class TestBandpassSos:
    def test_output_shape(self, baseline_data):
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        assert filtered.shape == baseline_data.shape

    def test_attenuates_out_of_band(self, rng):
        t    = np.arange(N_T) / SFREQ
        data = np.zeros((1, N_T))
        data[0] = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz — outside alpha band
        filtered = bandpass_sos(data, SFREQ, 8.0, 13.0)
        assert filtered.var() < data.var() * 0.01

    def test_passes_in_band(self, rng):
        t    = np.arange(N_T) / SFREQ
        data = np.zeros((1, N_T))
        data[0] = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz — inside alpha band
        filtered = bandpass_sos(data, SFREQ, 8.0, 13.0)
        assert filtered.var() > 0.5 * data.var()


# ── Tests: hilbert_envelope ───────────────────────────────────

class TestHilbertEnvelope:
    def test_output_shape(self, baseline_data):
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        env = hilbert_envelope(filtered)
        assert env.shape == filtered.shape

    def test_non_negative(self, baseline_data):
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        env = hilbert_envelope(filtered)
        assert (env >= 0).all()


# ── Tests: select_top_channels ────────────────────────────────

class TestSelectTopChannels:
    def test_returns_n_top(self, baseline_data):
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        env = hilbert_envelope(filtered)
        idx, env_top = select_top_channels(env, n_top=10)
        assert len(idx) == 10
        assert env_top.shape == (10, N_T)

    def test_caps_at_n_channels(self, baseline_data):
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        env = hilbert_envelope(filtered)
        idx, env_top = select_top_channels(env, n_top=1000)
        assert len(idx) == N_CH

    def test_highest_variance_selected(self, rng):
        env = np.ones((10, N_T))
        env[0] *= 10.0  # channel 0 has highest variance
        env[0, :N_T//2] = 0.0
        idx, _ = select_top_channels(env, n_top=1)
        assert 0 in idx


# ── Tests: build_reference ────────────────────────────────────

class TestBuildReference:
    def test_output_shape(self, baseline_data):
        ref = build_reference(baseline_data, SFREQ, 8.0, 13.0)
        assert ref.shape == (N_CH,)

    def test_non_negative(self, baseline_data):
        ref = build_reference(baseline_data, SFREQ, 8.0, 13.0)
        assert (ref >= 0).all()

    def test_build_all_references(self, baseline_data):
        refs = build_all_references(baseline_data, SFREQ)
        assert set(refs.keys()) == set(BAND_ORDER)
        for band in BAND_ORDER:
            assert refs[band].shape == (N_CH,)


# ── Tests: compute_sgamma ─────────────────────────────────────

class TestComputeSgamma:
    def test_range(self, baseline_data, ref_patterns):
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        env = hilbert_envelope(filtered)
        idx, env_top = select_top_channels(env, n_top=16)
        sg = compute_sgamma(env_top, ref_patterns["alpha"], idx)
        assert -1.0 <= sg <= 1.0

    def test_self_similarity_high(self, baseline_data):
        """Computing Sγ of baseline against its own reference → high."""
        ref = build_all_references(baseline_data, SFREQ)
        filtered = bandpass_sos(baseline_data, SFREQ, 8.0, 13.0)
        env = hilbert_envelope(filtered)
        idx, env_top = select_top_channels(env, n_top=16)
        sg = compute_sgamma(env_top, ref["alpha"], idx)
        assert sg > 0.5


# ── Tests: compute_eta ────────────────────────────────────────

class TestComputeEta:
    def test_range(self):
        Psi = np.array([1.0, 0.5, 2.0, 0.3, 0.1])
        A   = np.array([2.0, 1.0, 3.0, 1.0, 0.5])
        eta = compute_eta(Psi, A)
        assert 0.0 <= eta <= 1.0

    def test_zero_when_psi_zero(self):
        Psi = np.zeros(5)
        A   = np.array([1.0, 2.0, 3.0, 1.0, 0.5])
        assert compute_eta(Psi, A) == pytest.approx(0.0, abs=1e-6)

    def test_one_when_psi_equals_A(self):
        A   = np.array([1.0, 2.0, 3.0, 1.0, 0.5])
        eta = compute_eta(A, A)
        assert eta == pytest.approx(1.0, abs=1e-6)

    def test_amplitude_independent(self):
        """Scaling A and Psi by same factor should not change eta."""
        Psi = np.array([1.0, 0.5, 2.0, 0.3, 0.1])
        A   = np.array([2.0, 1.0, 3.0, 1.0, 0.5])
        eta1 = compute_eta(Psi, A)
        eta2 = compute_eta(Psi * 10, A * 10)
        assert eta1 == pytest.approx(eta2, abs=1e-6)


# ── Tests: vector decomposition ───────────────────────────────

class TestVectorDecomposition:
    def test_exact_decomposition(self, baseline_data, ref_patterns):
        """A = Psi + Delta_vec must hold exactly."""
        band_results = compute_band_metrics(
            baseline_data, SFREQ, ref_patterns, n_ch_top=16
        )
        A, Psi, Delta_vec = assemble_vectors(band_results)
        np.testing.assert_allclose(A, Psi + Delta_vec, rtol=1e-6)

    def test_delta_non_negative(self, baseline_data, ref_patterns):
        band_results = compute_band_metrics(
            baseline_data, SFREQ, ref_patterns, n_ch_top=16
        )
        A, Psi, Delta_vec = assemble_vectors(band_results)
        assert (Delta_vec >= 0).all()

    def test_psi_non_negative(self, baseline_data, ref_patterns):
        band_results = compute_band_metrics(
            baseline_data, SFREQ, ref_patterns, n_ch_top=16
        )
        A, Psi, Delta_vec = assemble_vectors(band_results)
        assert (Psi >= 0).all()


# ── Tests: compute_psi_centroid ───────────────────────────────

class TestPsiCentroid:
    def test_range(self):
        Psi = np.array([1.0, 0.5, 2.0, 0.3, 0.1])
        psi_c = compute_psi_centroid(Psi)
        assert FREQ_VALS.min() <= psi_c <= FREQ_VALS.max()

    def test_single_band(self):
        """If only alpha active, centroid should be near alpha centre."""
        Psi = np.array([0.0, 0.0, 5.0, 0.0, 0.0])  # only alpha
        psi_c = compute_psi_centroid(Psi)
        assert psi_c == pytest.approx(10.5, abs=0.1)


# ── Tests: mirror_criterion ───────────────────────────────────

class TestMirrorCriterion:
    def test_alpha_mirror(self):
        d_eff   = {"delta": 0.1, "theta": 0.1,
                   "alpha": -0.64, "beta": -0.1, "gamma": 0.0}
        d_delta = {"delta": 0.1, "theta": 0.1,
                   "alpha": +0.63, "beta":  0.1, "gamma": 0.0}
        bands = mirror_criterion(d_eff, d_delta, threshold=0.2)
        assert "alpha" in bands
        assert "delta" not in bands

    def test_no_mirror(self):
        d_eff   = {b: 0.0 for b in BAND_ORDER}
        d_delta = {b: 0.0 for b in BAND_ORDER}
        bands = mirror_criterion(d_eff, d_delta)
        assert bands == []


# ── Tests: SpatialEfficiencyPipeline ─────────────────────────

class TestPipeline:
    def test_not_fitted_raises(self, baseline_data):
        pipe = SpatialEfficiencyPipeline(sfreq=SFREQ)
        with pytest.raises(RuntimeError):
            pipe.compute(baseline_data)

    def test_fit_and_compute(self, pipeline, organised_epoch):
        m = pipeline.compute(organised_epoch)
        assert "eta" in m
        assert "norm_delta" in m
        assert "norm_psi" in m
        assert "psi_c" in m
        assert 0.0 <= m["eta"] <= 1.0

    def test_organised_higher_eta(self, pipeline,
                                   organised_epoch,
                                   disorganised_epoch):
        """Organised epoch should have higher eta than disorganised."""
        m_org  = pipeline.compute(organised_epoch)
        m_dis  = pipeline.compute(disorganised_epoch)
        assert m_org["eta"] > m_dis["eta"]

    def test_compute_epochs(self, pipeline, organised_epoch):
        epochs  = [organised_epoch] * 5
        results = pipeline.compute_epochs(epochs)
        assert len(results) == 5
        assert all("eta" in r for r in results)

    def test_sliding_window(self, pipeline, baseline_data):
        t, results = pipeline.compute_sliding_window(
            baseline_data, window_sec=1.0, step_sec=0.5
        )
        assert len(t) == len(results)
        assert len(results) > 0

    def test_extract_eta(self, pipeline, organised_epoch):
        epochs  = [organised_epoch] * 3
        results = pipeline.compute_epochs(epochs)
        eta_arr = SpatialEfficiencyPipeline.extract_eta(results)
        assert eta_arr.shape == (3,)
        assert (eta_arr >= 0).all()
        assert (eta_arr <= 1).all()

    def test_repr(self, pipeline):
        r = repr(pipeline)
        assert "fitted" in r
        assert "256" in r

    def test_low_sfreq_warning(self):
        with pytest.warns(UserWarning, match="sfreq"):
            SpatialEfficiencyPipeline(sfreq=100.0)


# ── Tests: compute_all_metrics ────────────────────────────────

class TestComputeAllMetrics:
    def test_output_keys(self, baseline_data, ref_patterns):
        m = compute_all_metrics(baseline_data, SFREQ, ref_patterns)
        for key in ["eta", "norm_delta", "norm_psi", "psi_c",
                    "stab", "amp", "eff", "Psi", "A", "Delta_vec"]:
            assert key in m

    def test_stab_length(self, baseline_data, ref_patterns):
        m = compute_all_metrics(baseline_data, SFREQ, ref_patterns)
        assert len(m["stab"]) == 5
        assert len(m["amp"])  == 5
        assert len(m["eff"])  == 5

    def test_eta_in_range(self, baseline_data, ref_patterns):
        m = compute_all_metrics(baseline_data, SFREQ, ref_patterns)
        assert 0.0 <= m["eta"] <= 1.0

    def test_decomposition_holds(self, baseline_data, ref_patterns):
        m = compute_all_metrics(baseline_data, SFREQ, ref_patterns)
        np.testing.assert_allclose(
            m["A"], m["Psi"] + m["Delta_vec"], rtol=1e-6
        )
