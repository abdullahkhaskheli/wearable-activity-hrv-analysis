import numpy as np
import pytest
from wearable_hrv.hrv_analysis import hrv_metrics, screened_hrv


def test_exact_mathematics():
    rr = np.array([800.0, 900.0, 1000.0, 900.0, 800.0])
    result = hrv_metrics(rr)
    mean = 880.0
    sample_variance = sum((x - mean) ** 2 for x in rr) / 4
    assert result["mean_nn_ms"] == mean
    assert result["sdnn_ms"] == pytest.approx(sample_variance**0.5)
    assert result["rmssd_ms"] == 100.0
    assert result["pnn50_percent"] == 100.0
    assert result["sd1_ms"] == pytest.approx((40000 / 3 / 2) ** 0.5)


def test_rejected_interval_does_not_bridge():
    rr = np.array([800.0, 850.0, 9999.0, 1000.0, 1050.0, 1100.0])
    metrics = hrv_metrics(rr, [True, True, False, True, True, True])
    assert metrics["rmssd_ms"] == 50 and metrics["n_adjacent_pairs"] == 3
    assert metrics["pnn50_percent"] == 0


def test_insufficient():
    with pytest.raises(ValueError):
        hrv_metrics([800, 900])
    with pytest.raises(ValueError):
        hrv_metrics([800, -1, 900], [True, True, True])


def test_screening():
    assert screened_hrv(np.full(100, 1000.0))["accepted"]
    rr = np.full(100, 1000.0)
    rr[::3] = 5000
    assert not screened_hrv(rr)["accepted"]
    assert not screened_hrv(np.r_[np.full(50, 800.0), np.full(50, 1000.0)])["accepted"]
