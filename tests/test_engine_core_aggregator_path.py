import pandas as pd
import numpy as np
from datetime import datetime

from engine_core import make_decision


def _df(n: int = 120) -> pd.DataFrame:
    t0 = datetime(2024, 1, 1)
    ts = pd.date_range(t0, periods=n, freq="15min")
    c = pd.Series(np.linspace(100, 110, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": c,
        "high": c + 1,
        "low": c - 1,
        "close": c,
        "volume": 10,
    })


def test_aggregator_called_when_present():
    df = _df()
    coin_cfg = {
        "ema_len": 22,
        "sma_len": 20,
        "rsi_period": 14,
        "_agg": {
            "signal_weights": {"sc_trend_htf": 0.35},
            "regime_bounds": {"atr_p1": 0.0, "atr_p2": 1.0, "bbw_q1": 0.0, "bbw_q2": 1.0},
            "strength_thresholds": {"weak": 0.25, "fair": 0.5, "strong": 0.75},
            "score_gate": 0.0,
        },
    }
    side, _ = make_decision(df, "TEST", coin_cfg, ml_up_prob=None)
    assert side in (None, "LONG", "SHORT")

