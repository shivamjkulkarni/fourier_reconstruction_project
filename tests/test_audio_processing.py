import numpy as np
import pytest
from src.audio_processing import (
    downmix_mono,
    generate_synthetic_guitar_note,
    slice_note,
    slice_segment,
)


def test_downmix_mono():
    """Checks that stereo audio channels are properly averaged into mono."""
    # stereo tracks get averaged into a single mono wave
    stereo = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    mono = downmix_mono(stereo)

    assert mono.ndim == 1
    assert np.allclose(mono, np.array([2.0, 3.0, 4.0]))

    # already mono data should pass straight through untouched
    single_channel = np.array([1.0, 2.0, 3.0])
    assert np.allclose(downmix_mono(single_channel), single_channel)


def test_slice_note_and_segment_dimensions():
    """Validates sample counts for the note window and 0.05-second analysis slice."""
    # test slicing on a 5-second dummy audio buffer at 44100 hz
    fs = 44100
    duration = 5.0
    dummy_data = np.ones(int(duration * fs))

    # note event is sliced from 2.0s to 4.0s (2 full seconds of sound)
    note_data, time_note = slice_note(dummy_data, fs, start_sec=2.0, end_sec=4.0)
    assert len(note_data) == int(2.0 * fs)
    assert len(time_note) == len(note_data)

    # 0.05s stable slice -> fixes frequency resolution at exactly 20 hz
    segment, time_seg = slice_segment(dummy_data, fs, start_sec=2.1, duration_sec=0.05)
    # n = 2205 samples gives exactly delta f = 44100 / 2205 = 20 hz bin width
    assert len(segment) == 2205
    assert len(time_seg) == 2205


def test_generate_synthetic_guitar_note(tmp_path):
    """Verifies that the synthetic guitar fallback generates and saves properly."""
    # test synthetic fallback note generator so tests run without external audio files
    wav_path = tmp_path / "synthetic_guitar.wav"
    fs, signal = generate_synthetic_guitar_note(
        samplerate=44100,
        duration_sec=4.5,
        save_path=wav_path,
    )

    assert fs == 44100
    assert len(signal) == int(4.5 * 44100)
    assert wav_path.exists()
