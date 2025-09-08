import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indicators.smartmoney.smc.fvg_luxalgo import detect_fvg

def _mk_df():
    n = 10
    ts = pd.date_range("2024-01-01", periods=n, freq="5T")
    base = 100.0
    df = pd.DataFrame({
        "timestamp": ts,
        "open": base,
        "high": base + 1.0,
        "low":  base - 1.0,
        "close": base,
        "volume": 1.0
    })
    return df.copy()

def test_bullish_fvg_detected():
    df = _mk_df()
    # bentuk pola bullish FVG pada i=5:
    # i=3: referensi (high)
    df.loc[3, ["high","low","close"]] = [100.0, 99.0, 99.5]
    # i=4: close di atas high[i-1=3]
    df.loc[4, "close"] = 100.5
    # i=5: low > high[3]
    df.loc[5, ["low","high","close"]] = [101.0, 103.0, 102.0]
    out = detect_fvg(df, thresholdPer=0.0, auto=False, extend=10, dynamic=False)
    assert out["bull_count"] >= 1
    assert out["bullish_unfilled"] is True

def test_bearish_fvg_detected():
    df = _mk_df()
    # bentuk pola bearish FVG pada i=6:
    # i=4: referensi (low)
    df.loc[4, ["high","low","close"]] = [100.5, 99.0, 99.2]
    # i=5: close di bawah low[i-1=4]
    df.loc[5, "close"] = 98.8
    # i=6: high < low[4]
    df.loc[6, ["high","low","close"]] = [98.7, 97.5, 98.0]
    # Pastikan close terakhir tidak memitigasi gap (tetap di bawah 99.0)
    df.loc[9, "close"] = 98.0
    out = detect_fvg(df, thresholdPer=0.0, auto=False)
    assert out["bear_count"] >= 1
    assert out["bearish_unfilled"] is True

def test_mitigation_updates_flags():
    df = _mk_df()
    # Bullish FVG di i=5 (min = 100.0)
    df.loc[3, ["high","low","close"]] = [100.0, 99.0, 99.5]
    df.loc[4, "close"] = 100.5
    df.loc[5, ["low","high","close"]] = [101.0, 103.0, 102.0]
    # Sekarang bar terakhir ditutup di bawah min → mitigated
    df.loc[9, "close"] = 99.0
    out = detect_fvg(df, thresholdPer=0.0, auto=False)
    assert out["bull_mitigated"] >= 1
    # Tidak ada lagi bull unfilled
    assert out["bullish_unfilled"] is False
