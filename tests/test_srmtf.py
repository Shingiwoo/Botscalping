import numpy as np
import pandas as pd
import pytest

from indicators.srmtf.support_resistance_mtf import (
    SupportResistanceMTF, resample_ohlcv, Zone
)

def make_df_1m(n=600, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-08-01", periods=n, freq="1min")
    base = np.cumsum(rng.normal(0, 0.05, size=n)) + 10.0
    high = base + rng.random(n) * 0.05
    low = base - rng.random(n) * 0.05
    vol = rng.integers(100, 1500, size=n)
    return pd.DataFrame(
        {"open": base, "high": high, "low": low, "close": base, "volume": vol},
        index=idx
    )

def test_resample_namedagg_and_no_nan():
    df1m = make_df_1m(300)
    out = resample_ohlcv(df1m, "5m")
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}
    assert not out.isna().any().any()
    assert out.index.is_monotonic_increasing
    assert isinstance(out.index[0], pd.Timestamp)

def test_zone_created_at_timestamp():
    df1m = make_df_1m(400)
    chart_tf = "1m"
    srm = SupportResistanceMTF(detection_length=3)
    levels, _ = srm.compute(
        chart_df=df1m,
        chart_tf=chart_tf,
        base_1m=df1m,
        use_presets=True
    )
    any_group = next(iter(levels))
    zones = levels[any_group]
    assert all(isinstance(z.created_at, pd.Timestamp) for z in zones)

def test_mtf_length_scaling_sanity():
    df1m = make_df_1m(500)
    srm = SupportResistanceMTF(detection_length=10)
    df_1h = resample_ohlcv(df1m, "1h")
    z = srm.compute_levels_for_tf(df_1h, "1h", chart_tf="15m")
    assert isinstance(z, list)

def test_breakout_non_repaint_then_valid():
    n = 250
    df1m = make_df_1m(n)
    i_spike = n - 3
    df1m.iloc[i_spike, df1m.columns.get_loc("high")] += 0.8
    df1m.iloc[i_spike, df1m.columns.get_loc("close")] -= 0.3
    df1m.iloc[i_spike + 1, df1m.columns.get_loc("open")] += 0.6
    df1m.iloc[i_spike + 1, df1m.columns.get_loc("close")] += 0.8

    srm = SupportResistanceMTF(detection_length=3, filter_false_breakouts=False)
    levels, signals = srm.compute(
        chart_df=df1m,
        chart_tf="1m",
        base_1m=df1m,
        use_presets=True
    )
    all_kinds = [s.kind for g in signals.values() for s in g]
    assert "BULL_BREAKOUT" in all_kinds

def test_rejection_detection_exists():
    df1m = make_df_1m(320)
    i = len(df1m) - 1
    o = df1m.iloc[i]["open"]
    df1m.iloc[i, df1m.columns.get_loc("open")] = o
    df1m.iloc[i, df1m.columns.get_loc("close")] = o - 0.02
    df1m.iloc[i, df1m.columns.get_loc("high")] = o + 0.8
    df1m.iloc[i, df1m.columns.get_loc("low")] = o - 0.03
    df1m.iloc[i, df1m.columns.get_loc("volume")] = int(df1m["volume"].mean() * 5)

    srm = SupportResistanceMTF(detection_length=3, rejection_shadow_mult=1.2)
    levels, signals = srm.compute(
        chart_df=df1m, chart_tf="1m", base_1m=df1m, use_presets=True
    )
    kinds = [s.kind for g in signals.values() for s in g]
    assert ("REJECT_UP" in kinds) or ("REJECT_DOWN" in kinds)

def test_presets_keys_and_types():
    df1m = make_df_1m(360)
    srm = SupportResistanceMTF(detection_length=3)
    levels, signals = srm.compute(df1m, "1m", base_1m=df1m, use_presets=True)
    assert "INTRADAY" in levels and "HTF" in levels
    assert "INTRADAY" in signals and "HTF" in signals
    assert all(isinstance(z, Zone) for z in levels["INTRADAY"])

def test_no_historical_levels_mode():
    df1m = make_df_1m(360)
    srm = SupportResistanceMTF(detection_length=3, use_prev_historical_levels=False)
    levels, signals = srm.compute(df1m, "1m", base_1m=df1m, use_presets=True)
    for zs in levels.values():
        assert isinstance(zs, list)
        assert all(isinstance(z, Zone) for z in zs)
