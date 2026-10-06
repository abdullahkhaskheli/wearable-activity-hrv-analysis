"""Explicit optional integration check; never replaced by a fixture."""

import pytest
from wearable_hrv.config import load_config
from wearable_hrv.validation import validate_dataset


@pytest.mark.real_data
def test_installed_dataset():
    root = load_config().path("dataset")
    if not root.exists():
        pytest.skip("PPG-DaLiA unavailable; real-data integration not run")
    table, errors = validate_dataset(root)
    assert not errors and not table.empty
