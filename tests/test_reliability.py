import numpy as np
import pytest
from wearable_hrv.reliability import agreement, bland_altman_points


def test_exact_agreement():
    result = agreement([60, 70, 80], [62, 68, 84])
    assert result["mae_bpm"] == pytest.approx(8 / 3)
    assert result["rmse_bpm"] == pytest.approx(8**0.5)
    assert result["bias_bpm"] == pytest.approx(4 / 3)
    sample_sd = (((2 - 4 / 3) ** 2 + (-2 - 4 / 3) ** 2 + (4 - 4 / 3) ** 2) / 2) ** 0.5
    assert result["loa_low_bpm"] == pytest.approx(4 / 3 - 1.96 * sample_sd)
    assert result["loa_high_bpm"] == pytest.approx(4 / 3 + 1.96 * sample_sd)
    means, diff = bland_altman_points([60, 70, 80], [62, 68, 84])
    np.testing.assert_array_equal(means, [61, 69, 82])
    np.testing.assert_array_equal(diff, [2, -2, 4])


def test_finite_pairs():
    assert agreement([60, np.nan], [62, 70])["n_pairs"] == 1
    assert np.isnan(agreement([60, 60], [61, 62])["pearson_r"])
    with pytest.raises(ValueError):
        agreement([1], [1, 2])
    with pytest.raises(ValueError):
        agreement([np.nan], [1])
