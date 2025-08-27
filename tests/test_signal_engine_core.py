# tests/test_signal_engine_core.py
import os, sys, numpy as np, pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from signal_engine.aggregator import (
    aggregate, volume_spike_factor, sr_tolerance_pct, build_features_from_modules
)
from signal_engine.regime import scale_weights


def _df(n=300):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = pd.Series(np.linspace(100, 110, n), dtype="float64")
    return pd.DataFrame({
        "timestamp": ts,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": np.ones(n, dtype="float64") * 10,
    })


def test_aggregator_clamp_and_label():
    df = _df()
    weights = {"sc_trend_htf": 1.0, "sr_breakout": 1.0}
    thresholds = {
        "vol_lookback": 20,
        "strength_thresholds": {"weak": 0.25, "fair": 0.5, "strong": 0.75},
        "score_gate": 0.5,
    }
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {"base_pct": 0.6, "k_atr": 0.5}
    features = {"sr": {"breakout_same_dir": True}}
    r = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features=features)
    assert 0.0 <= r["score"] <= 1.0
    assert r["strength"] in {"netral", "lemah", "cukup", "kuat"}


def test_dynamic_weights_scaling():
    df_low = _df()
    df_high = _df()
    df_high["high"] += 5
    df_high["low"] -= 5
    weights = {"sc_trend_htf": 0.0, "adx": 1.0}
    thresholds = {"vol_lookback": 20, "weight_scale": {"HIGH": {"adx": 0.5}, "LOW": {"adx": 2.0}}}
    regime_bounds = {"atr_p1": 0.02, "atr_p2": 0.05, "bbw_q1": 0.01, "bbw_q2": 0.05}
    r_low = aggregate(df_low, "LONG", weights, thresholds, regime_bounds, {}, features={"sr": {}})
    r_high = aggregate(df_high, "LONG", weights, thresholds, regime_bounds, {}, features={"sr": {}})
    assert r_low["breakdown"].get("adx", 0) > r_high["breakdown"].get("adx", 0)


def test_sr_tolerance_pct():
    assert sr_tolerance_pct(0.01, 0.6, 0.5) > 0.6


def test_fvg_bonuses_and_penalties():
    df = _df()
    weights = {"sc_trend_htf": 0.0, "fvg_confirm": 0.5, "fvg_contra": 0.5}
    thresholds = {}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    r_pos = aggregate(df, "LONG", weights, thresholds, regime_bounds, {}, features={"fvg": {"has_same_dir": True}})
    r_neg = aggregate(df, "LONG", weights, thresholds, regime_bounds, {}, features={"fvg": {"has_contra": True}})
    assert r_pos["breakdown"]["fvg_confirm"] > 0
    assert r_neg["breakdown"]["fvg_confirm"] < 0


def test_volume_spike_factor_zscore():
    vol = pd.Series([1] * 19 + [100], dtype="float64")
    assert volume_spike_factor(vol, lookback=20, z_thr=2.0, max_boost=0.5) > 0


def test_htf_fallback_discount_flag_affects_breakdown():
    df = _df()
    r = aggregate(df, "LONG", {"sc_trend_htf": 0.0}, {}, {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}, {}, features={"sr": {}, "htf_fallback": "D"})
    assert "htf_fallback_discount" in r["breakdown"]


def test_entry_next_open_simulation():
    df = _df(6)
    weights = {"sc_trend_htf": 1.0}
    thresholds = {}
    bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sig_i = ent_i = None
    for i in range(len(df)):
        r = aggregate(df.iloc[: i + 1], "LONG", weights, thresholds, bounds, {}, features={"sr": {}})
        if r["ok"] and sig_i is None:
            sig_i = i
        if sig_i is not None and i == sig_i + 1:
            ent_i = i
            break
    assert ent_i == sig_i + 1

