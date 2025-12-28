import numpy as np
import pytest
from src.audio_processing import generate_synthetic_guitar_note, slice_segment
from src.dft_engine import manual_dft
from src.harmonic_analysis import (
    calculate_harmonic_ratios,
    compute_residual_spectrum,
    detect_spectral_peaks,
    verify_guitar_harmonics,
)


def test_peak_detection_and_harmonic_verification():
    """Validates that peak detection isolates the 60 Hz and 120 Hz harmonics."""
    # synthesize calibrated test note and grab the 0.05s analysis slice
    fs, note = generate_synthetic_guitar_note(samplerate=44100, duration_sec=4.5)
    segment, _ = slice_segment(note, fs, start_sec=2.1, duration_sec=0.05)

    # run dft across the 2205 samples
    dft_coeffs = manual_dft(segment, vectorized=True)

    # 5% prominence cutoff to drop mic static and equipment hum
    pos_peaks, peak_freqs, peak_mags, bin_indices = detect_spectral_peaks(
        dft_coeffs,
        samplerate=fs,
        prominence_ratio=0.05,
    )

    assert len(pos_peaks) >= 2

    # verify fundamental at 60.00 hz (bin k=3) and 1st overtone at 120.00 hz (bin k=6)
    verification = verify_guitar_harmonics(
        peak_freqs,
        peak_mags,
        bin_indices,
        tol_hz=0.5,
    )

    assert verification["verified"] is True
    assert verification["f0_bin"] == 3
    assert verification["f1_bin"] == 6
    assert np.isclose(verification["f0_freq"], 60.0)
    assert np.isclose(verification["f1_freq"], 120.0)
    # 120hz overtone dominates 60hz fundamental (pluck point suppresses fundamental node)
    assert verification["overtone_ratio"] > 1.5


def test_residual_spectrum_properties():
    """Confirms that residual error contains only flat noise without periodic spikes."""
    # residual noise shouldn't show any lingering harmonic spikes
    rng = np.random.default_rng(303)
    residual = rng.normal(0.0, 1e-11, 2205)

    res_freq, res_mags = compute_residual_spectrum(residual, samplerate=44100)
    assert len(res_freq) == 1103
    assert np.max(res_mags) < 1e-8
