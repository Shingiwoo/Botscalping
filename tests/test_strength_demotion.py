import pandas as pd
import numpy as np
from signal_engine.aggregator import aggregate


def _df(n=220):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = pd.Series(np.linspace(100, 110, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": np.ones(n) * 10,
    })


def test_confirmation_demotion():
    df = _df()
    weights = {"sc_trend_htf": 0.35, "sr_breakout": 0.20, "sd_proximity": 0.20, "fvg_confirm": 0.10, "vol_confirm": 0.10}
    thresholds = {"min_confirms": 2, "score_gate": 0.0}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {}
    r = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"sr": {}})
    assert r["strength"] in {"netral", "lemah"}

