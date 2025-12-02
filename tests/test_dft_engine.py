import numpy as np
import pytest
from src.dft_engine import (
    check_basis_orthogonality,
    compute_residual,
    compute_rmse,
    manual_dft,
    manual_idft,
)


def test_toy_four_point_dft():
    """Checks the 4-point toy calculation solved by hand in the paper."""
    # hand-calculated 4-point toy wave [1, 0, -1, 0] from section 1.3.4 of the paper
    x = np.array([1.0, 0.0, -1.0, 0.0])
    coeffs = manual_dft(x, vectorized=False)

    # dc bin cancels out to 0, matching frequency constructively adds up to 2
    assert np.isclose(coeffs[0], 0.0 + 0.0j, atol=1e-12)
    assert np.isclose(coeffs[1], 2.0 + 0.0j, atol=1e-12)
    assert np.isclose(coeffs[2], 0.0 + 0.0j, atol=1e-12)
    assert np.isclose(coeffs[3], 2.0 + 0.0j, atol=1e-12)


def test_dft_matches_numpy_fft():
    """Verifies that our manual summation gives the exact same result as np.fft.fft."""
    # test our manual loop and matrix code against numpy's trusted fft
    rng = np.random.default_rng(101)
    signal = rng.standard_normal(128)

    manual_res = manual_dft(signal, vectorized=False)
    vec_res = manual_dft(signal, vectorized=True)
    numpy_res = np.fft.fft(signal)

    # should match within floating point rounding
    assert np.allclose(manual_res, numpy_res, atol=1e-10)
    assert np.allclose(vec_res, numpy_res, atol=1e-10)


def test_lossless_reconstruction_and_rmse():
    """Confirms that IDFT reconstruction perfectly restores the original signal."""
    # idft reconstruction should perfectly undo the dft with error near machine float precision
    rng = np.random.default_rng(202)
    signal = rng.standard_normal(256)

    dft_coeffs = manual_dft(signal, vectorized=True)
    reconstructed = manual_idft(dft_coeffs, vectorized=True)

    # rmse check: leftover difference is just float precision (~1e-11)
    rmse = compute_rmse(signal, reconstructed)
    residual = compute_residual(signal, reconstructed)

    assert rmse < 1e-11
    assert np.max(np.abs(residual)) < 1e-10


def test_basis_orthogonality():
    """Checks that basis waves cancel out to zero when multiplied together."""
    # discrete orthogonality test: off-diagonals must sum to 0, diagonals to N
    is_ortho_4, err_4 = check_basis_orthogonality(4)
    assert is_ortho_4
    assert err_4 < 1e-12

    is_ortho_16, err_16 = check_basis_orthogonality(16)
    assert is_ortho_16
    assert err_16 < 1e-12
