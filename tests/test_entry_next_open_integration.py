import pandas as pd
import numpy as np
from signal_engine.aggregator import aggregate


def test_entry_next_open_marker():
    ts = pd.date_range("2024-01-01", periods=12, freq="15min")
    price = pd.Series(np.linspace(100, 102, len(ts)))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": price.shift(1).fillna(price.iloc[0]),
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 10,
    })
    weights = {"sc_trend_htf": 1.0}
    thresholds = {"score_gate": 0.0}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {}
    r = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"sr": {}})
    assert "score" in r and "strength" in r

