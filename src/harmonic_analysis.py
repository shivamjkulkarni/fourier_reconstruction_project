from typing import Dict, Tuple, Union
import numpy as np
from scipy.signal import find_peaks


def detect_spectral_peaks(
    dft_coeffs: np.ndarray,
    samplerate: int,
    prominence_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finds prominent frequency peaks in the positive DFT magnitude spectrum."""
    n_samples = len(dft_coeffs)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / samplerate)
    mags = np.abs(dft_coeffs)

    # 5% prominence cutoff to drop mic static and equipment hum
    prominence_threshold = np.max(mags) * prominence_ratio
    peaks, _ = find_peaks(mags, prominence=prominence_threshold)

    # keep only positive physical frequencies up to the nyquist limit
    pos_peaks = peaks[freqs[peaks] >= 0]
    peak_freqs = freqs[pos_peaks]
    peak_mags = mags[pos_peaks]

    # grab corresponding bin indices k (where freq = k * 20 hz)
    bin_indices = pos_peaks

    return pos_peaks, peak_freqs, peak_mags, bin_indices


def verify_guitar_harmonics(
    peak_freqs: np.ndarray,
    peak_mags: np.ndarray,
    bin_indices: np.ndarray,
    tol_hz: float = 1e-3,
) -> Dict[str, Union[bool, float, int]]:
    """Checks that the fundamental is at 60 Hz and first overtone is at 120 Hz."""
    # verify fundamental at 60.00 hz (bin k=3) and 1st overtone at 120.00 hz (bin k=6)
    f0_match = False
    f1_match = False
    f0_freq, f0_mag, f0_bin = 0.0, 0.0, -1
    f1_freq, f1_mag, f1_bin = 0.0, 0.0, -1

    for freq, mag, k in zip(peak_freqs, peak_mags, bin_indices):
        if np.isclose(freq, 60.0, atol=tol_hz):
            f0_match = (k == 3)
            f0_freq, f0_mag, f0_bin = float(freq), float(mag), int(k)
        elif np.isclose(freq, 120.0, atol=tol_hz):
            f1_match = (k == 6)
            f1_freq, f1_mag, f1_bin = float(freq), float(mag), int(k)

    # 120hz overtone dominates 60hz fundamental (pluck point suppresses fundamental node)
    overtone_ratio = float(f1_mag / f0_mag) if f0_mag > 0 else 0.0

    return {
        "verified": bool(f0_match and f1_match),
        "f0_freq": f0_freq,
        "f0_mag": f0_mag,
        "f0_bin": f0_bin,
        "f1_freq": f1_freq,
        "f1_mag": f1_mag,
        "f1_bin": f1_bin,
        "overtone_ratio": overtone_ratio,
    }


def calculate_harmonic_ratios(peak_freqs: np.ndarray, peak_mags: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculates frequency multiples and amplitude ratios relative to the fundamental."""
    # compare harmonic peak heights relative to fundamental to get timbre recipe
    if len(peak_freqs) == 0:
        return {"freq_multiples": np.array([]), "amplitude_ratios": np.array([])}

    f0 = peak_freqs[0]
    a0 = peak_mags[0]

    freq_multiples = peak_freqs / f0
    amplitude_ratios = peak_mags / a0

    return {
        "fundamental_freq": float(f0),
        "fundamental_mag": float(a0),
        "freq_multiples": freq_multiples,
        "amplitude_ratios": amplitude_ratios,
    }


def compute_residual_spectrum(residual: np.ndarray, samplerate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the frequency spectrum of the residual error signal."""
    # take fft of residual error to prove there are no hidden sine waves left over
    n_samples = len(residual)
    res_fft = np.fft.fft(residual)
    res_freq = np.fft.fftfreq(n_samples, d=1.0 / samplerate)
    res_mags = np.abs(res_fft)

    # look only at positive frequencies
    pos_mask = res_freq >= 0
    return res_freq[pos_mask], res_mags[pos_mask]
