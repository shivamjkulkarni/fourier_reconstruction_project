from .audio_processing import (
    downmix_mono,
    generate_synthetic_guitar_note,
    load_audio,
    slice_note,
    slice_segment,
)
from .dft_engine import (
    check_basis_orthogonality,
    compute_residual,
    compute_rmse,
    manual_dft,
    manual_idft,
)
from .harmonic_analysis import (
    calculate_harmonic_ratios,
    compute_residual_spectrum,
    detect_spectral_peaks,
    verify_guitar_harmonics,
)
from .visualization import (
    plot_aliasing,
    plot_analysis_segment,
    plot_full_waveform,
    plot_magnitude_spectrum,
    plot_note_event,
    plot_reconstruction_validation,
    plot_residual_error,
    plot_residual_spectrum,
    plot_riemann_sum,
    plot_unit_circle,
)

__all__ = [
    "manual_dft",
    "manual_idft",
    "check_basis_orthogonality",
    "compute_rmse",
    "compute_residual",
    "downmix_mono",
    "load_audio",
    "slice_note",
    "slice_segment",
    "generate_synthetic_guitar_note",
    "detect_spectral_peaks",
    "verify_guitar_harmonics",
    "calculate_harmonic_ratios",
    "compute_residual_spectrum",
    "plot_riemann_sum",
    "plot_aliasing",
    "plot_unit_circle",
    "plot_full_waveform",
    "plot_note_event",
    "plot_analysis_segment",
    "plot_magnitude_spectrum",
    "plot_reconstruction_validation",
    "plot_residual_error",
    "plot_residual_spectrum",
]

