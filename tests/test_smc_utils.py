import pandas as pd, numpy as np
from indicators.smartmoney.smc.utils import atr, resample_ohlcv

def test_atr_series_and_rolling():
    idx = pd.date_range("2025-01-01", periods=10, freq="1min")
    base = pd.Series(np.linspace(100, 101, len(idx)), index=idx)
    df = pd.DataFrame({
        "open": base, "high": base+1, "low": base-1, "close": base, "volume": 100
    })
    a = atr(df, period=3)
    assert isinstance(a, pd.Series)
    assert a.iloc[2] > 0

def test_resample_ohlcv_daily_mapping_ok():
    idx = pd.date_range("2025-01-01", periods=48, freq="30min")
    df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10
    }, index=idx)
    out = resample_ohlcv(df, "1D")
    assert set(out.columns) == {"open","high","low","close","volume"}
    assert len(out) >= 1
