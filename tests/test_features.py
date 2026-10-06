import numpy as np
import pytest
from wearable_hrv.feature_extraction import extract_features


def test_dimensions_values():
    result = extract_features(np.tile([1.0, 2.0, 2.0], (128, 1)), 32)
    assert len(result) == 30
    assert result["x_mean"] == 1 and result["magnitude_mean"] == 3
    assert result["y_variance"] == 0 and result["magnitude_rms"] == 3
    assert result["dynamic_dominant_hz"] == 0


def test_nan_rejected():
    x = np.zeros((128, 3))
    x[1, 2] = np.nan
    with pytest.raises(ValueError):
        extract_features(x, 32)


def test_spectral_feature():
    t = np.arange(256) / 32
    acc = np.column_stack([np.sin(2 * np.pi * 2 * t), np.zeros((256, 2))])
    # Rectified dynamic magnitude has twice the sinusoid frequency.
    assert extract_features(acc, 32)["dynamic_dominant_hz"] == pytest.approx(4.0)
