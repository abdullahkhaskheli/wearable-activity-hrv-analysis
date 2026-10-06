import numpy as np
import pytest
from wearable_hrv.preprocessing import (
    bandpass,
    windows,
    slice_time,
    activity_for_window,
    vector_magnitude,
    movement_intensity,
)


def test_filter_attenuation():
    fs = 100
    t = np.arange(2000) / fs
    wanted = np.sin(2 * np.pi * 10 * t)
    noise = np.sin(2 * np.pi * 35 * t)
    output = bandpass(wanted + noise, fs, 5, 15)
    assert output.shape == t.shape
    assert np.sqrt(np.mean((output[200:-200] - wanted[200:-200]) ** 2)) < 0.1


@pytest.mark.parametrize("fs,low,high", [(0, 1, 2), (10, 1, 6), (100, 10, 5)])
def test_bad_filter(fs, low, high):
    with pytest.raises(ValueError):
        bandpass(np.ones(100), fs, low, high)


def test_window_boundaries():
    assert list(windows(10, 4, 2)) == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert list(windows(3, 4, 2)) == []
    np.testing.assert_array_equal(slice_time(np.arange(20), 2, 2, 4), [4, 5, 6, 7])
    with pytest.raises(ValueError):
        slice_time(np.arange(20), 2, 0, 11)


def test_activity_purity():
    labels = np.array([1, 1, 2, 2])
    assert activity_for_window(labels, 1, 0, 4) is None
    assert activity_for_window(labels, 1, 0, 2) == 1
    assert activity_for_window(np.zeros(4), 1, 0, 4) is None


def test_acceleration():
    acc = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
    np.testing.assert_allclose(vector_magnitude(acc), [5, 5])
    assert movement_intensity(acc) == 0
    assert movement_intensity(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])) == 1
