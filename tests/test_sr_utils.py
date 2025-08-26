import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from indicators.sr_utils import compute_sr_levels, near_level, build_sr_cache, ltf_momentum_ok, htf_trend_ok_multi

def _mkdf(n=350):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = pd.Series(np.linspace(1, 1.2, n)) + np.random.normal(0, 0.005, n)
    high = close + np.abs(np.random.normal(0.005, 0.003, n))
    low  = close - np.abs(np.random.normal(0.005, 0.003, n))
    return pd.DataFrame({"timestamp":idx, "open":close, "high":high, "low":low, "close":close, "volume":1.0})

def test_sr_levels_and_near_level_are_bool():
    df = _mkdf()
    RES, SUP = compute_sr_levels(df, lb=3, window=300, k=6)
    assert isinstance(near_level(float(df['close'].iloc[-1]), RES, 1.0), bool)
    assert isinstance(near_level(float(df['close'].iloc[-1]), SUP, 1.0), bool)

def test_build_sr_cache_sparse():
    df = _mkdf()
    cache = build_sr_cache(df, lb=3, window=300, k=6, recalc_every=10)
    assert len(cache) > 0
    # kunci teratas selalu ada
    assert (len(df)-1) in cache

def test_ltf_momentum_and_htf_return_bool():
    df = _mkdf(500)
    l_ok, s_ok = ltf_momentum_ok(df, lookback=5)
    assert isinstance(l_ok, bool) and isinstance(s_ok, bool)
    assert isinstance(htf_trend_ok_multi("LONG", df, rules=("1H","4H")), bool)
