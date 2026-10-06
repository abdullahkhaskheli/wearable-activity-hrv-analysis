import numpy as np
import pytest
from wearable_hrv.signal_processing import (
    detect_peaks,
    rr_intervals,
    valid_rr_mask,
    heart_rate,
    peak_window_intervals,
)


@pytest.mark.parametrize("polarity", [1, -1])
def test_controlled_ecg_peaks(polarity):
    fs = 100
    t = np.arange(1000) / fs
    peaks = np.arange(1, 9)
    x = sum(polarity * np.exp(-(((t - p) / 0.015) ** 2)) for p in peaks)
    found = detect_peaks(x, fs)
    np.testing.assert_array_equal(found, peaks * fs)


def test_controlled_pulse():
    fs = 64
    t = np.arange(640) / fs
    x = np.sin(2 * np.pi * t)
    found = detect_peaks(x, fs, prominence_mad=0.5, ecg=False)
    np.testing.assert_array_equal(found, np.arange(16, 640, 64))


def test_rr_hr():
    rr = rr_intervals([0, 100, 200, 300], 100)
    np.testing.assert_array_equal(rr, [1000, 1000, 1000])
    assert heart_rate(rr) == (60.0, 1.0)
    np.testing.assert_array_equal(
        peak_window_intervals([0, 100, 200, 300], 100, 1, 3), [1000]
    )


def test_invalid_rr():
    mask = valid_rr_mask([1000, 1000, 5000, 1000, 1000])
    np.testing.assert_array_equal(mask, [True, True, False, True, True])
    with pytest.raises(ValueError):
        rr_intervals([1, 1], 100)
    assert len(detect_peaks(np.zeros(100), 100)) == 0
