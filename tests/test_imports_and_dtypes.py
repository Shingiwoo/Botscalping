# tests/test_imports_and_dtypes.py
import importlib
from importlib.util import find_spec
import pandas as pd
import numpy as np


def test_package_paths_do_not_crash():
    assert find_spec("signal_engine") is not None


def test_regime_metrics_return_float():
    from signal_engine.regime import compute_vol_metrics

    ts = pd.date_range("2024-01-01", periods=40, freq="5min")
    price = pd.Series(np.linspace(100, 110, 40), dtype="float64")
    df = pd.DataFrame({"timestamp": ts, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0})
    m = compute_vol_metrics(df, 20)
    assert isinstance(m["atr_pct"], float)
    assert isinstance(m["bb_width"], float)


def test_adapters_loaded_without_real_modules():
    from signal_engine import adapters

    assert hasattr(adapters, "SC_API")
    assert hasattr(adapters, "SD_API")

