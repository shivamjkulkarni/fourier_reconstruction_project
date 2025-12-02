import numpy as np


def manual_dft(signal: np.ndarray, vectorized: bool = False) -> np.ndarray:
    """Computes the Discrete Fourier Transform directly using the summation formula."""
    # manual o(n^2) loop to see the actual summation instead of hiding behind np.fft
    x = np.asarray(signal, dtype=np.float64)
    n_samples = len(x)

    if vectorized:
        # vectorized matrix multiply when the nested python loops get too sluggish
        n = np.arange(n_samples)
        k = n.reshape((n_samples, 1))
        basis_matrix = np.exp(-2j * np.pi * k * n / n_samples)
        return np.dot(basis_matrix, x)

    # nested loops matching appendix a to show the exact math on paper
    dft_coeffs = np.zeros(n_samples, dtype=complex)
    for k in range(n_samples):
        accumulator = 0.0 + 0.0j
        for n in range(n_samples):
            # rotate sample n by -2*pi*k*n/N radians
            angle = -2.0 * np.pi * k * n / n_samples
            accumulator += x[n] * np.exp(1j * angle)
        dft_coeffs[k] = accumulator

    return dft_coeffs


def manual_idft(dft_coeffs: np.ndarray, vectorized: bool = False, return_real: bool = True) -> np.ndarray:
    """Reconstructs the original time-domain signal by summing complex frequency components."""
    # idft adds all frequency bins back together with positive rotation angles
    coeffs = np.asarray(dft_coeffs, dtype=np.complex128)
    n_samples = len(coeffs)

    if vectorized:
        # matrix form of the synthesis formula scaled by 1/N
        n = np.arange(n_samples)
        k = n.reshape((n_samples, 1))
        inv_basis_matrix = np.exp(2j * np.pi * k * n / n_samples)
        reconstructed = np.dot(inv_basis_matrix, coeffs) / n_samples
        return reconstructed.real if return_real else reconstructed

    # manual synthesis loop over all frequencies
    reconstructed = np.zeros(n_samples, dtype=complex)
    for n in range(n_samples):
        accumulator = 0.0 + 0.0j
        for k in range(n_samples):
            angle = 2.0 * np.pi * k * n / n_samples
            accumulator += coeffs[k] * np.exp(1j * angle)
        # remember the 1/N scaling factor from continuous 1/T integral
        reconstructed[n] = accumulator / n_samples

    return reconstructed.real if return_real else reconstructed


def check_basis_orthogonality(n_points: int) -> tuple[bool, float]:
    """Tests whether discrete basis waves cancel out to zero when multiplied together."""
    # check orthogonality: sum exp(i*2*pi*(k-j)*n/N) should be N if k==j else 0
    n = np.arange(n_points)
    k = n.reshape((n_points, 1))
    basis_matrix = np.exp(2j * np.pi * k * n / n_points)

    # multiply basis matrix by its conjugate transpose to get dot products of all pairs
    gram_matrix = np.dot(basis_matrix, basis_matrix.conj().T)
    expected = n_points * np.eye(n_points, dtype=complex)
    max_error = float(np.max(np.abs(gram_matrix - expected)))
    # off-diagonals cancel to zero within float precision
    is_orthogonal = bool(max_error < 1e-10)

    return is_orthogonal, max_error


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates the root mean square error between the original and reconstructed waveforms."""
    # rmse check: leftover difference is just float precision (~1e-11)
    orig = np.asarray(original, dtype=np.float64)
    recon = np.asarray(reconstructed, dtype=np.float64)
    return float(np.sqrt(np.mean((orig - recon) ** 2)))


def compute_residual(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Extracts the sample-by-sample difference left over after reconstruction."""
    # residual = original - reconstructed; should look like pure random noise
    orig = np.asarray(original, dtype=np.float64)
    recon = np.asarray(reconstructed, dtype=np.float64)
    return orig - recon
