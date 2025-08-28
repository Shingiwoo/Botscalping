import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from signal_engine.aggregator import aggregate, volume_spike_factor, sr_tolerance_pct, build_features_from_modules
from signal_engine.regime import scale_weights
from indicators.rsi.ultimate_rsi import URSIAdapter, URSIConfig


def _df(n=300):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = pd.Series(np.linspace(100, 110, n))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": np.ones(n) * 10,
    })
    return df


def test_aggregator_clamp_and_label():
    df = _df()
    weights = {"sc_trend_htf": 1.0, "sr_breakout": 1.0}
    thresholds = {"vol_lookback": 20, "strength_thresholds": {"weak": 0.25, "fair": 0.5, "strong": 0.75}, "score_gate": 0.5}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {"base_pct": 0.6, "k_atr": 0.5}
    features = {"sr": {"breakout_same_dir": True}}
    r = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features=features)
    assert 0 <= r["score"] <= 1
    assert r["strength"] in {"lemah", "cukup", "kuat"}


def test_dynamic_weights_scaling():
    df_low = _df()
    df_high = _df()
    df_high["high"] += 5
    df_high["low"] -= 5
    weights = {"sc_trend_htf": 0.0, "adx": 1.0}
    thresholds = {"vol_lookback": 20, "weight_scale": {"HIGH": {"adx": 0.5}, "LOW": {"adx": 2.0}}}
    regime_bounds = {"atr_p1": 0.02, "atr_p2": 0.05, "bbw_q1": 0.01, "bbw_q2": 0.05}
    sr_penalty = {}
    r_low = aggregate(df_low, "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"sr": {}})
    r_high = aggregate(df_high, "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"sr": {}})
    assert r_low["breakdown"].get("adx", 0) > r_high["breakdown"].get("adx", 0)


def test_sr_tolerance_pct():
    assert sr_tolerance_pct(0.01, 0.6, 0.5) > 0.6


def test_fvg_bonuses():
    df = _df()
    weights = {"sc_trend_htf": 0.0, "fvg_confirm": 0.5, "fvg_contra": 0.5}
    thresholds = {}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {}
    r_pos = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"fvg": {"has_same_dir": True}})
    r_neg = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"fvg": {"has_contra": True}})
    assert r_pos["breakdown"]["fvg_confirm"] > 0
    assert r_neg["breakdown"]["fvg_confirm"] < 0


def test_volume_spike_factor():
    vol = pd.Series([1]*19 + [100])
    assert volume_spike_factor(vol, lookback=20, z_thr=2.0, max_boost=0.5) > 0


def test_htf_fallback_discount_flag():
    df = _df()
    weights = {"sc_trend_htf": 0.0}
    thresholds = {}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {}
    features = {"sr": {}, "htf_fallback": "D"}
    r = aggregate(df, "LONG", weights, thresholds, regime_bounds, sr_penalty, features=features)
    assert "htf_fallback_discount" in r["breakdown"]


def test_entry_next_open():
    df = _df(5)
    weights = {"sc_trend_htf": 1.0}
    thresholds = {}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {}
    signal_idx = None
    entry_idx = None
    for i in range(len(df)):
        r = aggregate(df.iloc[: i + 1], "LONG", weights, thresholds, regime_bounds, sr_penalty, features={"sr": {}})
        if r["ok"] and signal_idx is None:
            signal_idx = i
        if signal_idx is not None and i == signal_idx + 1:
            entry_idx = i
            break
    assert entry_idx == signal_idx + 1


def test_ursi_adapter_to_aggregator():
    df = _df()
    adp = URSIAdapter("BTCUSDT", URSIConfig(source="ohlc4"))
    evt = None
    for _, r in df.iterrows():
        evt = adp.on_price((r.open, r.high, r.low, r.close), r.timestamp)
    assert evt is not None
    features = {"sr": {}, "ursi": {"arsi": evt["arsi"], "signal": evt["signal"]}}
    side = "LONG" if evt["arsi"] >= evt["signal"] else "SHORT"
    weights = {"sc_trend_htf": 0.0, "ursi": 1.0}
    thresholds = {"vol_lookback": 20}
    regime_bounds = {"atr_p1": 0, "atr_p2": 1, "bbw_q1": 0, "bbw_q2": 1}
    sr_penalty = {}
    r = aggregate(df, side, weights, thresholds, regime_bounds, sr_penalty, features=features)
    assert r["breakdown"].get("ursi", 0) > 0
