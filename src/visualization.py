from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_riemann_sum(output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plots a continuous function alongside 10 discrete rectangular slices."""
    fig = plt.figure(figsize=(8, 5))

    # draw smooth test curve f(t) = sin(2*pi*t)*e^(-t) + 1.5
    t_continuous = np.linspace(0.0, 1.0, 200)
    f_continuous = np.sin(2.0 * np.pi * t_continuous) * np.exp(-t_continuous) + 1.5

    # cut into n=10 slices of width dt=0.1 to show how the integral becomes a sum
    n_slices = 10
    t_discrete = np.linspace(0.0, 1.0, n_slices, endpoint=False)
    f_discrete = np.sin(2.0 * np.pi * t_discrete) * np.exp(-t_discrete) + 1.5
    dt = 1.0 / n_slices

    plt.plot(
        t_continuous,
        f_continuous,
        "b-",
        linewidth=2,
        label=r"Continuous function $f(t)e^{-ik\omega_0t}$",
    )
    # rectangular slices matching the riemann sum diagram in the paper
    plt.bar(
        t_discrete,
        f_discrete,
        width=dt,
        alpha=0.4,
        color="orange",
        align="edge",
        edgecolor="black",
        label="Discrete samples (Riemann slices)",
    )

    plt.title("Riemann Sum Approximation from Continuous to Discrete")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig


def plot_aliasing(output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plots a 9 Hz wave sampled at 10 Hz to demonstrate Nyquist aliasing."""
    fig = plt.figure(figsize=(10, 4))

    # smooth 9 hz wave oscillates too fast for a 10 hz sample clock
    t_continuous = np.linspace(0.0, 1.0, 500)
    true_signal = np.sin(2.0 * np.pi * 9.0 * t_continuous)

    # sampling at 10 hz gives only one point per cycle plus a slow drift
    fs = 10
    t_discrete = np.linspace(0.0, 1.0, fs + 1)
    discrete_samples = np.sin(2.0 * np.pi * 9.0 * t_discrete)

    # the discrete points falsely look like a smooth 1 hz wave (9 - 10 = -1 hz alias)
    aliased_signal = np.sin(2.0 * np.pi * -1.0 * t_continuous)

    plt.plot(t_continuous, true_signal, "b-", alpha=0.4, label="True High-Frequency Signal (9 Hz)")
    plt.plot(t_continuous, aliased_signal, "r--", linewidth=2, label="Perceived Aliased Signal (-1 Hz)")
    plt.plot(t_discrete, discrete_samples, "ko", markersize=8, label=f"Discrete Samples (sampled at {fs} Hz)")

    plt.title("Visualizing the Nyquist Limit: Signal Aliasing")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.ylim(-1.2, 2.0)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_unit_circle(output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plots the four roots of unity on the complex plane for N=4."""
    fig = plt.figure(figsize=(6, 6))

    # unit circle showing where the basis vectors point for a 4-point dft
    angles = np.linspace(0.0, 2.0 * np.pi, 100)
    plt.plot(np.cos(angles), np.sin(angles), "k--", alpha=0.5)

    n_values = [0, 1, 2, 3]
    labels = [r"$e^0 = 1$", r"$e^{-i\pi/2} = -i$", r"$e^{-i\pi} = -1$", r"$e^{-i3\pi/2} = i$"]
    colors = ["red", "blue", "green", "purple"]

    for n, label, color in zip(n_values, labels, colors):
        # each step rotates by 90 degrees clockwise: 1, -i, -1, i
        angle = -np.pi * n / 2.0
        x, y = np.cos(angle), np.sin(angle)
        plt.quiver(0, 0, x, y, angles="xy", scale_units="xy", scale=1, color=color, width=0.008)
        plt.plot(x, y, "ko")
        plt.text(
            x * 1.2,
            y * 1.2,
            label,
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title(r"DFT Complex Exponentials ($N=4$)")
    plt.xlabel("Real Axis (Re)")
    plt.ylabel("Imaginary Axis (Im)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.gca().set_aspect("equal", adjustable="box")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig


def plot_full_waveform(
    time_axis: np.ndarray,
    data: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots the entire audio recording across its full duration."""
    # entire recording showing several guitar notes separated by pauses
    fig = plt.figure(figsize=(12, 3))
    plt.plot(time_axis, data, color="#1f77b4", linewidth=0.8)
    plt.title("Full Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_note_event(
    time_axis: np.ndarray,
    note_data: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots a close-up of a single plucked note event."""
    # closeup of one pluck: sharp attack at 2.0s followed by clean decay
    fig = plt.figure(figsize=(8, 3.5))
    plt.plot(time_axis, note_data, color="#2ca02c", linewidth=0.8)
    plt.title("Note Event")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_analysis_segment(
    time_axis: np.ndarray,
    segment: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots the short, stable 0.05-second slice used for Fourier analysis."""
    # 0.05s stable slice -> fixes frequency resolution at exactly 20 hz (n=2205 samples)
    fig = plt.figure(figsize=(8, 3.5))
    plt.plot(time_axis, segment, color="#d62728", linewidth=1.0)
    plt.title("Segment for Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_magnitude_spectrum(
    freqs: np.ndarray,
    mags: np.ndarray,
    peak_freqs: Optional[Sequence[float]] = None,
    peak_mags: Optional[Sequence[float]] = None,
    xlim: Tuple[float, float] = (0.0, 3000.0),
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots the DFT magnitude spectrum with detected harmonic peaks."""
    # magnitude |X_k| vs physical frequency k * 20 hz
    fig = plt.figure(figsize=(9, 4))
    plt.plot(freqs, mags, color="#1f77b4", linewidth=1.0, label="DFT Spectrum")

    # mark detected peaks (60 hz fundamental and 120 hz overtone)
    if peak_freqs is not None and peak_mags is not None and len(peak_freqs) > 0:
        plt.scatter(
            peak_freqs,
            peak_mags,
            color="red",
            s=45,
            zorder=5,
            label="Detected Peaks (5% prominence)",
        )
        for pf, pm in zip(peak_freqs, peak_mags):
            plt.annotate(
                f"{pf:.1f} Hz",
                (pf, pm),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    plt.xlim(*xlim)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_reconstruction_validation(
    time_axis: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots the original signal and the reconstructed IDFT wave together."""
    # overlay original and idft wave: dashed orange line lands right on blue
    fig = plt.figure(figsize=(9, 4))
    plt.plot(time_axis, original, color="#1f77b4", linewidth=1.2, label="Original")
    plt.plot(
        time_axis,
        reconstructed,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label="Reconstructed (IDFT)",
    )
    plt.title("Original vs Reconstructed Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_residual_error(
    time_axis: np.ndarray,
    residual: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots the point-by-point residual error between original and reconstruction."""
    # leftover error sits at tiny ~1e-10 scale and looks like pure static
    fig = plt.figure(figsize=(9, 3.5))
    plt.plot(time_axis, residual, color="#9467bd", linewidth=0.8)
    plt.title("Residual Error (Original - Reconstructed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude Error")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig


def plot_residual_spectrum(
    res_freq: np.ndarray,
    res_mags: np.ndarray,
    max_freq: float = 22050.0,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots the frequency spectrum of the reconstruction residual."""
    # fft of residual shows a flat floor with zero harmonic spikes left over
    fig = plt.figure(figsize=(9, 3.5))
    plt.plot(res_freq, res_mags, color="#8c564b", linewidth=0.8)
    plt.xlim(0.0, max_freq)
    plt.title("Residual Error Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300)

    return fig
