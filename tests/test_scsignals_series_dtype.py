# tests/test_scsignals_series_dtype.py
import pytest
import pandas as pd
import numpy as np

mod = pytest.importorskip("indicators.scsignal.scsignals", reason="scsignals belum diimplementasi penuh")


def test_numeric_series_outputs():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="1min"),
        "open": np.linspace(1, 2, 50),
        "high": np.linspace(1.1, 2.1, 50),
        "low": np.linspace(0.9, 1.9, 50),
        "close": np.linspace(1, 2, 50),
        "volume": np.ones(50),
    }).set_index("timestamp")
    out = mod.compute_all(df, {})
    for k, v in out.items():
        if hasattr(v, "dtype"):
            assert str(v.dtype).startswith(("float", "int"))

