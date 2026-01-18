import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.audio_processing import (
    generate_synthetic_guitar_note,
    slice_note,
    slice_segment,
)
from src.dft_engine import (
    compute_residual,
    compute_rmse,
    manual_dft,
    manual_idft,
)
from src.harmonic_analysis import (
    compute_residual_spectrum,
    detect_spectral_peaks,
    verify_guitar_harmonics,
)
from src.visualization import (
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


def main():
    repo_root = Path(__file__).resolve().parent.parent
    assets_dir = repo_root / "assets"
    data_raw_dir = repo_root / "data" / "raw"
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. theory diagrams from appendix a (riemann slices, nyquist aliasing, n=4 roots of unity)
    plot_riemann_sum(assets_dir / "riemann_sum.png")
    plt.close("all")

    plot_aliasing(assets_dir / "aliasing_diagram.png")
    plt.close("all")

    plot_unit_circle(assets_dir / "unit_circle_roots.png")
    plt.close("all")

    # 2. copy theory flowchart if available in workspace
    workspace_flowchart = repo_root.parent / "flowchart.png"
    if workspace_flowchart.exists():
        shutil.copyfile(workspace_flowchart, assets_dir / "theory_flowchart.png")
        shutil.copyfile(workspace_flowchart, assets_dir / "flowchart.png")

    # 3. reference audio: 4.5s guitar note with attack and decay
    wav_path = data_raw_dir / "Guitar_sample.wav"
    fs, data = generate_synthetic_guitar_note(
        samplerate=44100,
        duration_sec=4.5,
        save_path=wav_path,
    )

    duration = len(data) / fs
    time_axis_full = np.linspace(0.0, duration, len(data), endpoint=False)

    plot_full_waveform(time_axis_full, data, assets_dir / "full_waveform.png")
    plt.close("all")

    # 4. slice single note between 2.0s and 4.0s
    note_data, time_note = slice_note(data, fs, start_sec=2.0, end_sec=4.0)
    plot_note_event(time_note, note_data, assets_dir / "note_closeup.png")
    plt.close("all")

    # 5. 0.05s stable slice -> fixes frequency resolution at exactly 20 hz
    segment, time_seg = slice_segment(data, fs, start_sec=2.1, duration_sec=0.05)
    plot_analysis_segment(time_seg, segment, assets_dir / "isolated_segment.png")
    plt.close("all")

    # 6. manual dft and 5% prominence peak finding
    dft_coeffs = manual_dft(segment, vectorized=True)
    n_samples = len(segment)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / fs)
    mags = np.abs(dft_coeffs)

    # 5% prominence cutoff to drop mic static and equipment hum
    pos_peaks, peak_freqs, peak_mags, bin_indices = detect_spectral_peaks(
        dft_coeffs,
        samplerate=fs,
        prominence_ratio=0.05,
    )

    # verify fundamental at 60.00 hz (bin k=3) and 1st overtone at 120.00 hz (bin k=6)
    verification = verify_guitar_harmonics(peak_freqs, peak_mags, bin_indices)
    print(f"harmonic verification status: {verification}")

    plot_magnitude_spectrum(
        freqs,
        mags,
        peak_freqs=peak_freqs,
        peak_mags=peak_mags,
        xlim=(0.0, 3000.0),
        output_path=assets_dir / "spectrum.png",
    )
    plt.close("all")

    # 7. idft reconstruction: check that overlay matches to float precision
    reconstructed = manual_idft(dft_coeffs, vectorized=True, return_real=True)
    rmse = compute_rmse(segment, reconstructed)
    residual = compute_residual(segment, reconstructed)
    # rmse check: leftover difference is just float precision (~1e-11)
    print(f"reconstruction rmse: {rmse:.6e}")

    plot_reconstruction_validation(
        time_seg,
        segment,
        reconstructed,
        output_path=assets_dir / "validation.png",
    )
    plt.close("all")

    # 8. residual error and its fft (should just look like flat noise floor)
    plot_residual_error(
        time_seg,
        residual,
        output_path=assets_dir / "residual_error.png",
    )
    plt.close("all")

    res_freq, res_mags = compute_residual_spectrum(residual, samplerate=fs)
    plot_residual_spectrum(
        res_freq,
        res_mags,
        max_freq=fs / 2.0,
        output_path=assets_dir / "residual_spectrum.png",
    )
    plt.close("all")

    print("all figures successfully generated into assets/")


if __name__ == "__main__":
    main()
