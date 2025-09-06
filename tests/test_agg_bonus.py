import pandas as pd
import numpy as np
from signal_engine.aggregator import aggregate


def _df():
    # minimal synthetic DF
    n = 120
    ts = pd.date_range("2025-01-01", periods=n, freq="15min")
    base = pd.DataFrame({
        "timestamp": ts,
        "open": np.linspace(100, 105, n),
        "high": np.linspace(100.5, 105.5, n),
        "low": np.linspace(99.5, 104.5, n),
        "close": np.linspace(100.2, 105.2, n),
        "volume": np.linspace(1000, 2000, n),
    })
    return base


def test_confirm_bonus_clamped():
    df = _df()
    weights = {
        # zero-out base weights to isolate confirm bonus
        "sc_trend_htf": 0.0, "sc_no_htf": 0.0, "adx": 0.0, "body_atr": 0.0, "width_atr": 0.0, "rsi": 0.0,
        # confirmators
        "sr_breakout": 0.10, "sd_proximity": 0.05, "vol_confirm": 0.05, "fvg_confirm": 0.05
    }
    thresholds = {
        "strength_thresholds": {"weak": 0.25, "fair": 0.50, "strong": 0.75},
        "score_gate": 0.40,
        "confirm_bonus_per": 0.05,
        "confirm_bonus_max": 0.10,
        "vol_lookback": 20,
        "vol_z_thr": 10.0,  # avoid unintended vol boost
        "sd_tol_pct": 5.0,
    }
    regime_bounds = {"atr_p1": 0.01, "atr_p2": 0.05, "bbw_q1": 0.01, "bbw_q2": 0.05}
    sr_penalty = {"base_pct": 0.6, "k_atr": 0.5}
    # craft 3 confirmators via features
    feats = {
        "sr": {"breakout_same_dir": True},
        "sd": {"demand_wavg": float(df["close"].iloc[-1])},
        "fvg": {"has_same_dir": True, "has_contra": False},
    }
    r = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features=feats)
    # Without bonus the raw breakdown sums to <= 0.20 (approx, after clamp01)
    # With 3 confirms and per=0.05, the naive bonus would be 0.15, but max=0.10 must cap it.
    assert r["score"] <= 0.20 + 0.10 + 1e-6
