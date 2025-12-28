from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from scipy.io import wavfile


def downmix_mono(data: np.ndarray) -> np.ndarray:
    """Averages multi-channel audio arrays down to a single mono track."""
    # average left and right channels together if recorded in stereo
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr.mean(axis=1)
    return arr


def load_audio(filepath: Union[str, Path]) -> Tuple[int, np.ndarray]:
    """Reads a WAV file and returns the sample rate and mono signal."""
    # load clean uncompressed wav data (phone mics applied aggressive noise filters that ruined harmonics)
    samplerate, data = wavfile.read(str(filepath))
    mono = downmix_mono(data).astype(np.float64)
    return samplerate, mono


def slice_note(
    data: np.ndarray,
    samplerate: int,
    start_sec: float = 2.0,
    end_sec: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cuts out a single guitar note event across a specified time interval."""
    # cut one specific note out of the full recording from 2.0s to 4.0s
    note_start = int(start_sec * samplerate)
    note_end = int(end_sec * samplerate)
    single_note = data[note_start:note_end]
    time_note = np.linspace(start_sec, end_sec, len(single_note), endpoint=False)
    return single_note, time_note


def slice_segment(
    data: np.ndarray,
    samplerate: int,
    start_sec: float = 2.1,
    duration_sec: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts a short, stationary slice of audio for discrete Fourier analysis."""
    # 0.05s stable slice -> fixes frequency resolution at exactly 20 hz (n = 2205 samples at 44100 hz)
    seg_start = int(start_sec * samplerate)
    seg_end = int((start_sec + duration_sec) * samplerate)
    segment = data[seg_start:seg_end]
    # start time axis at zero for the dft window
    time_seg = np.linspace(0.0, duration_sec, len(segment), endpoint=False)
    return segment, time_seg


def generate_synthetic_guitar_note(
    samplerate: int = 44100,
    duration_sec: float = 4.5,
    f0: float = 60.0,
    f0_mag: float = 45391.5,
    f1_mag: float = 91096.0,
    noise_std: float = 1.2,
    random_seed: int = 42,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[int, np.ndarray]:
    """Synthesizes a realistic guitar note with 60 Hz fundamental and 120 Hz overtone."""
    # fallback note generator so notebooks and tests run immediately even without downloading the raw wav
    rng = np.random.default_rng(random_seed)
    total_samples = int(duration_sec * samplerate)
    time = np.linspace(0.0, duration_sec, total_samples, endpoint=False)

    # calibrate sinusoid heights so the 0.05s slice hits the exact dft peak magnitudes in table 1
    n_slice = int(0.05 * samplerate)
    amp_f0 = (2.0 * f0_mag) / n_slice
    amp_f1 = (2.0 * f1_mag) / n_slice

    # model the physical pluck: silent before 2s, sharp pick attack up to 2.1s, then exponential decay
    envelope = np.zeros(total_samples, dtype=np.float64)
    attack_idx = int(2.0 * samplerate)
    sustain_idx = int(2.1 * samplerate)

    # quiet before the pick strikes
    envelope[:attack_idx] = 0.0

    # quick 0.1s pluck attack transient
    envelope[attack_idx:sustain_idx] = np.linspace(0.0, 1.0, sustain_idx - attack_idx)

    # string vibration rings out and decays slowly
    decay_samples = total_samples - sustain_idx
    tau = 1.8 * samplerate
    envelope[sustain_idx:] = np.exp(-np.arange(decay_samples) / tau)

    # 120hz overtone dominates 60hz fundamental (pluck point suppresses fundamental node)
    harmonic_signal = (
        amp_f0 * np.cos(2.0 * np.pi * f0 * time + 0.35)
        + amp_f1 * np.cos(2.0 * np.pi * 2.0 * f0 * time - 0.22)
    )

    # add subtle background hiss and body resonance
    signal = envelope * harmonic_signal + rng.normal(0.0, noise_std, total_samples)

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # save as 16-bit pcm wav for standard audio players
        scaled = np.int16(signal / np.max(np.abs(signal)) * 32767)
        wavfile.write(str(out_path), samplerate, scaled)

    return samplerate, signal
