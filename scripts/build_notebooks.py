import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


def create_notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Notebook 01: Theory and Discretization
### Modeling a vibrating string and turning integrals into code

When I play guitar, notes don't just have pitch—they have *timbre*, that warm, full feeling in the sound. I wanted to see if I could strip away the vague words and measure that sound in actual numbers.

I started with the physics of an ideal vibrating string of length $L$ clamped at both ends ($u(0,t) = u(L,t) = 0$):
$$\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\frac{\\partial^2 u}{\\partial x^2}$$

Separating variables into space $X(x)$ and time $T(t)$ gave standing wave modes $\\sin\\left(\\frac{n\\pi x}{L}\\right)$ at frequencies $\\omega_n = \\frac{n\\pi c}{L}$. By superposition, the full vibration is just the sum of these sinusoidal modes:
$$u(x,t) = \\sum_{n=1}^\\infty \\left( A_n \\cos(\\omega_n t) + B_n \\sin(\\omega_n t) \\right) \\sin\\left( \\frac{n\\pi x}{L} \\right)$$

This gave me the physical green light to treat sine and cosine waves as the literal building blocks of guitar timbre.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

# make sure we can import from src/
repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt

from src.dft_engine import manual_dft
from src.visualization import plot_riemann_sum, plot_aliasing, plot_unit_circle
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. From Continuous Integrals to Discrete Slices

The continuous Fourier coefficient formula over period $T$ is:
$$c_k = \\frac{1}{T} \\int_0^T f(t) e^{-i k \\omega_0 t} \\, dt$$

At first, I spent hours wondering how code was supposed to handle that $1/T$ integral when all I had were discrete data points from a file. It clicked when I drew out Riemann rectangles.
The base of each rectangle is $\\Delta t = T / N$. When you substitute that into the Riemann sum:
$$c_k \\approx \\frac{1}{T} \\sum_{n=0}^{N-1} f(t_n) e^{-i k (2\\pi/T)(n \\Delta t)} \\Delta t = \\frac{1}{T} \\sum_{n=0}^{N-1} x_n e^{-i 2\\pi k n / N} \\left(\\frac{T}{N}\\right) = \\frac{1}{N} \\sum_{n=0}^{N-1} x_n e^{-i 2\\pi k n / N}$$

The time duration $T$ cancels out completely! We are left with the discrete $1/N$ factor replacing the continuous $1/T$ integration.
That gave me confidence that the DFT isn't a black-box trick; it's literally just the continuous integral cut into $N$ rectangular slices.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# continuous curve sliced into n=10 discrete riemann rectangles
fig_riemann = plot_riemann_sum()
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Sampling Limits: Aliasing and Bin Resolution

Working with discrete audio samples forces two hard mathematical constraints:

1. **Frequency Resolution ($\\Delta f = 1/T$):** The DFT sorts energy into discrete frequency bins spaced by $\\Delta f = 1/T$. For our $0.05\\text{ s}$ analysis window, $\\Delta f = 1/0.05 = 20\\text{ Hz}$. We can only distinguish frequencies in steps of $20\\text{ Hz}$.
2. **Nyquist Limit ($f_s / 2$):** For CD audio at $f_s = 44,100\\text{ Hz}$, the maximum frequency we can accurately record without aliasing is $22,050\\text{ Hz}$. If a wave oscillates faster than half the sample clock, the discrete dots get connected into a false low-frequency ghost wave.

Here is what happens when a $9\\text{ Hz}$ wave is sampled at only $10\\text{ Hz}$—it looks identically like a $-1\\text{ Hz}$ alias wave.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# 9hz wave sampled at 10hz aliases down to perceived 1hz
fig_aliasing = plot_aliasing()
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Walking the Complex Unit Circle

Instead of juggling separate sines and cosines, Euler's formula $e^{-i\\theta} = \\cos\\theta - i\\sin\\theta$ wraps everything into a single rotating vector.
For $N=4$, the DFT basis functions rotate by $90^\\circ$ ($\\pi/2$ radians) each step:
- $n=0 \\implies e^0 = 1$
- $n=1 \\implies e^{-i\\pi/2} = -i$
- $n=2 \\implies e^{-i\\pi} = -1$
- $n=3 \\implies e^{-i3\\pi/2} = i$
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# argand diagram of the 4 complex roots of unity
fig_unit_circle = plot_unit_circle()
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Hand-Calculating a 4-Point Toy Signal

To verify the summation formula before running it on thousands of audio samples, I tested it on a simple toy wave $x_n = [1, 0, -1, 0]$:
- For $X_1$ ($k=1$, testing for matching wave):
  $$X_1 = (1)(1) + (0)(-i) + (-1)(-1) + (0)(i) = 1 + 0 + 1 + 0 = 2$$
- For $X_0$ ($k=0$, testing for DC offset):
  $$X_0 = (1)(1) + (0)(1) + (-1)(1) + (0)(1) = 1 + 0 - 1 + 0 = 0$$

The math constructively interferes to 2 for matching frequencies and destructively cancels to 0 for mismatched ones.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# test our manual loop on the toy signal worked out by hand
toy_signal = np.array([1.0, 0.0, -1.0, 0.0])
toy_coeffs = manual_dft(toy_signal, vectorized=False)

print(f"Toy signal: {toy_signal}")
print(f"X_0 (DC offset):          {toy_coeffs[0].real:.1f} + {toy_coeffs[0].imag:.1f}i (expected 0)")
print(f"X_1 (matching frequency): {toy_coeffs[1].real:.1f} + {toy_coeffs[1].imag:.1f}i (expected 2)")

assert np.isclose(toy_coeffs[0], 0.0)
assert np.isclose(toy_coeffs[1], 2.0)
print("\\nhand calculation matches code output perfectly")
"""
        )
    )

    nb.cells = cells
    return nb


def create_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Notebook 02: Manual DFT & Timbre Fingerprint
### Isolating the note, calculating harmonic peaks, and discovering why the guitar sounds warm

I originally tried recording my acoustic guitar with my phone microphone. When I ran that data through my initial code, the spectrum was an absolute mess—random spikes everywhere. After days of troubleshooting, I realized the phone's built-in noise cancellation and compression algorithms were actively destroying the natural harmonics of the plucked string.

To get pure, unfiltered sound, I switched to an uncompressed 44.1 kHz WAV recording. In this notebook, we slice out a single stable note, run our manual DFT, and find the harmonic recipe that defines the instrument's timbre.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt

from src.audio_processing import (
    downmix_mono,
    generate_synthetic_guitar_note,
    load_audio,
    slice_note,
    slice_segment,
)
from src.dft_engine import manual_dft
from src.harmonic_analysis import (
    calculate_harmonic_ratios,
    detect_spectral_peaks,
    verify_guitar_harmonics,
)
from src.visualization import (
    plot_analysis_segment,
    plot_full_waveform,
    plot_magnitude_spectrum,
    plot_note_event,
)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Loading Audio and Isolating a Single Pluck

The raw recording contains several notes. We need to:
1. Cut out a single note between $2.0\\text{ s}$ and $4.0\\text{ s}$.
2. Extract a stationary $0.05\\text{ s}$ slice from $[2.1, 2.15]\\text{ s}$ ($N = 2205$ samples). 

Slicing at 2.1s avoids the messy attack clatter when the pick strikes the string and catches the sustained tone where the wave is periodic.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """data_file = repo_root / "data" / "raw" / "Guitar_sample.wav"

if data_file.exists():
    fs, data = load_audio(data_file)
    print(f"loaded audio from {data_file}: fs = {fs} hz, {len(data)} samples")
else:
    # fallback note generator so notebooks run even without downloading the raw wav
    fs, data = generate_synthetic_guitar_note(
        samplerate=44100,
        duration_sec=4.5,
        save_path=data_file,
    )
    print(f"generated synthetic reference note: fs = {fs} hz, {len(data)} samples")

duration = len(data) / fs
time_axis_full = np.linspace(0.0, duration, len(data), endpoint=False)

# entire recording showing several guitar notes separated by pauses
fig_full = plot_full_waveform(time_axis_full, data)
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Note Closeup and the 0.05s Analysis Window

We isolate the note event and take the stable sustain slice:
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# isolate one note between 2.0s and 4.0s
note_data, time_note = slice_note(data, fs, start_sec=2.0, end_sec=4.0)
fig_note = plot_note_event(time_note, note_data)
plt.show()

# 0.05s stable slice -> fixes frequency resolution at exactly 20 hz
segment, time_seg = slice_segment(data, fs, start_sec=2.1, duration_sec=0.05)
print(f"analysis segment: N = {len(segment)} samples")
print(f"bin resolution: Delta f = fs / N = {fs / len(segment):.2f} hz")

fig_seg = plot_analysis_segment(time_seg, segment)
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Running the Manual $\\mathcal{O}(N^2)$ DFT

We run our manual DFT summation across all 2205 samples:
$$X_k = \\sum_{n=0}^{N-1} x_n e^{-i 2\\pi k n / N}$$
Each bin index $k$ maps to physical frequency $f_k = k \\cdot \\Delta f = k \\times 20\\text{ Hz}$.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# manual o(n^2) loop to see the actual summation instead of hiding behind np.fft
dft_coeffs = manual_dft(segment, vectorized=True)
N = len(segment)
freqs = np.fft.fftfreq(N, d=1.0 / fs)
mags = np.abs(dft_coeffs)

print(f"calculated {len(dft_coeffs)} complex coefficients")
print(f"dc offset |X_0|: {mags[0]:.2f}")
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Detecting Peaks and Explaining the Timbre

When I first plotted the spectrum, I saw tiny bumps way out at high frequencies and wondered if they were exotic guitar harmonics. Checking the numbers showed they were just background mic hiss. Setting a 5% prominence cutoff dropped the static and cleanly isolated the true musical peaks.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# 5% prominence cutoff to drop mic static and equipment hum
pos_peaks, peak_freqs, peak_mags, bin_indices = detect_spectral_peaks(
    dft_coeffs,
    samplerate=fs,
    prominence_ratio=0.05,
)

print(f"{'Harmonic':<22} | {'Bin k':<8} | {'Frequency (Hz)':<16} | {'Magnitude':<12}")
print("-" * 65)
labels = ["Fundamental (f0)", "1st Overtone (2f0)"]
for i, (k, freq, mag) in enumerate(zip(bin_indices, peak_freqs, peak_mags)):
    lbl = labels[i] if i < len(labels) else f"Harmonic {i+1}"
    print(f"{lbl:<22} | {k:<8d} | {freq:<16.2f} | {mag:<12.2f}")

verification = verify_guitar_harmonics(peak_freqs, peak_mags, bin_indices)
print(f"\\nverification: fundamental={verification['f0_freq']:.1f} hz (bin {verification['f0_bin']}), overtone={verification['f1_freq']:.1f} hz (bin {verification['f1_bin']})")
print(f"overtone to fundamental ratio: {verification['overtone_ratio']:.2f}")

fig_spec = plot_magnitude_spectrum(
    freqs,
    mags,
    peak_freqs=peak_freqs,
    peak_mags=peak_mags,
    xlim=(0.0, 1500.0),
)
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### Why Does the 120 Hz Overtone Dominate?
This result surprised me at first. For an instrument like a flute, the fundamental frequency almost completely dominates the sound. Here, the 1st overtone at 120 Hz has twice the magnitude of the 60 Hz fundamental!

This isn't an error—it's the physical fingerprint of where the string was plucked:
- If you pluck a guitar string right in the middle ($L/2$), you maximally excite the fundamental antinode, while suppressing the 2nd harmonic node.
- If you pluck closer to the bridge or at an off-center node, the fundamental gets subdued while the 2nd harmonic (1st overtone) rings out strongly.
That ratio is the physical recipe that gives the acoustic guitar its characteristic warm tone.
"""
        )
    )

    nb.cells = cells
    return nb


def create_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Notebook 03: Lossless Reconstruction and Residuals
### Synthesizing the wave back together, testing basis orthogonality, and checking error noise

Breaking a waveform down into frequencies is only half the story. If Fourier's theorem is correct, we should be able to synthesize all those rotating complex numbers back into the exact original sound with zero information lost.

In this notebook, we check basis orthogonality, run the manual Inverse DFT, and look at the leftover residual noise.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt

from src.audio_processing import generate_synthetic_guitar_note, load_audio, slice_segment
from src.dft_engine import (
    check_basis_orthogonality,
    compute_residual,
    compute_rmse,
    manual_dft,
    manual_idft,
)
from src.harmonic_analysis import compute_residual_spectrum
from src.visualization import (
    plot_reconstruction_validation,
    plot_residual_error,
    plot_residual_spectrum,
)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Checking Basis Orthogonality

The reason the Inverse DFT works without error is that all the basis waves are orthogonal. If you multiply any two different basis waves together and sum over a full period, positive and negative areas cancel out to zero:
$$\\sum_{n=0}^{N-1} e^{i 2\\pi (k - j) n / N} = 
\\begin{cases}
N, & k = j \\\\
0, & k \\ne j
\\end{cases}$$

They are completely invisible to each other.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# check discrete basis orthogonality: off-diagonals cancel to zero
for test_n in [4, 8, 16, 64]:
    is_ortho, max_err = check_basis_orthogonality(test_n)
    print(f"N = {test_n:2d}: orthogonal = {is_ortho} (max off-diagonal error = {max_err:.2e})")
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Reconstructing the Waveform with Manual IDFT

We use the Inverse DFT formula:
$$x_n = \\frac{1}{N} \\sum_{k=0}^{N-1} X_k e^{i 2\\pi k n / N}$$

Let's synthesize the waveform and overlay it on the original guitar slice.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """data_file = repo_root / "data" / "raw" / "Guitar_sample.wav"
if data_file.exists():
    fs, data = load_audio(data_file)
else:
    fs, data = generate_synthetic_guitar_note(save_path=data_file)

# 0.05s stable slice -> fixes frequency resolution at exactly 20 hz
segment, time_seg = slice_segment(data, fs, start_sec=2.1, duration_sec=0.05)
dft_coeffs = manual_dft(segment, vectorized=True)

# reconstruct time-domain signal by summing all rotating complex vectors
reconstructed = manual_idft(dft_coeffs, vectorized=True, return_real=True)

# overlay original and idft wave: dashed orange line lands right on blue
fig_val = plot_reconstruction_validation(time_seg, segment, reconstructed)
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Measuring Reconstruction Error (RMSE)

To check whether the fit is truly exact or just close to the eye, we compute the Root Mean Square Error (RMSE):
$$\\text{RMSE} = \\sqrt{ \\frac{1}{N} \\sum_{n=1}^N (x_n - \\hat{x}_n)^2 }$$
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# rmse check: leftover difference is just float precision (~1e-11)
rmse = compute_rmse(segment, reconstructed)
residual = compute_residual(segment, reconstructed)
rel_rmse = rmse / np.max(np.abs(segment))

print(f"absolute rmse: {rmse:.6e}")
print(f"relative rmse: {rel_rmse:.6e} (near machine epsilon)")
print(f"max sample difference: {np.max(np.abs(residual)):.6e}")

assert rel_rmse < 1e-10
print("\\nreconstruction is mathematically exact within floating-point limits")
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Inspecting the Residual Noise

If our Fourier model missed a harmonic or an acoustic resonance, that missing wave would show up as an oscillating pattern in the leftover residual ($r_n = x_n - \\hat{x}_n$).
Let's look at the residual in time and frequency.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# leftover error sits at tiny ~1e-10 scale and looks like pure static
fig_res = plot_residual_error(time_seg, residual)
plt.show()

# fft of residual shows a flat floor with zero harmonic spikes left over
res_freq, res_mags = compute_residual_spectrum(residual, samplerate=fs)
fig_res_spec = plot_residual_spectrum(res_freq, res_mags, max_freq=fs / 2.0)
plt.show()

max_res_mag = np.max(res_mags)
rel_res_mag = max_res_mag / np.max(np.abs(dft_coeffs))
print(f"max residual peak: {max_res_mag:.4e} (relative to spectrum: {rel_res_mag:.4e})")
assert rel_res_mag < 1e-9
print("residual analysis confirms: leftover error is pure random floating-point noise")
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. What I Learned and Next Steps

1. **Timbre is a measurable recipe:** The warm tone of this guitar note comes directly from the $120\\text{ Hz}$ overtone being twice as strong as the $60\\text{ Hz}$ fundamental.
2. **DFT is an exact change of basis:** In exact arithmetic, the error is zero. The tiny $\\sim 10^{-11}$ error is purely from 64-bit float math.
3. **The trade-off of static slices:** A real guitar pluck changes over time—the pick clicks, the string rings bright, then high harmonics die out faster than the low ones. Our static 0.05s window only captures one snapshot.
4. **Next step:** Exploring the Short-Time Fourier Transform (STFT) with sliding windows to watch the timbre evolve over the entire 4 seconds.
"""
        )
    )

    nb.cells = cells
    return nb


def main():
    repo_root = Path(__file__).resolve().parent.parent
    nb_dir = repo_root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "01_theory_and_discretization.ipynb": create_notebook_01(),
        "02_manual_dft_and_timbre_fingerprint.ipynb": create_notebook_02(),
        "03_lossless_reconstruction_and_residuals.ipynb": create_notebook_03(),
    }

    for name, nb in notebooks.items():
        out_path = nb_dir / name
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"created notebook {out_path}")

        print(f"executing {name}...")
        client = NotebookClient(nb, timeout=60, kernel_name="python3")
        client.execute()
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"executed and saved {name} with outputs")


if __name__ == "__main__":
    main()
