import numpy as np
import pandas as pd
from indicators.srmtf.support_resistance_mtf import Zone, Signal
from aggregators.sr_features import sr_features_from_signals, sr_reason_weights_default

def _make_chart(n=120, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-08-10", periods=n, freq="15min")
    base = np.cumsum(rng.normal(0, 0.2, size=n)) + 10.0
    df = pd.DataFrame({
        "open": base,
        "high": base + rng.random(n) * 0.2,
        "low":  base - rng.random(n) * 0.2,
        "close": base,
        "volume": rng.integers(100, 2000, size=n),
    }, index=idx)
    return df

def test_features_mapping_basic():
    chart = _make_chart()
    ts = chart.index[-1]
    last = float(chart["close"].iloc[-1])
    z = Zone(kind="R", top=last*1.005, bottom=last*0.999, level=last, tf="4h",
             created_at=ts, margin_factor=0.1)
    s = Signal(ts=ts, tf_chart="15m", kind="REJECT_UP", price=last, ref_zone=z, volume=500, volume_sma=200, atr=0.1)
    feats = sr_features_from_signals(chart, {"HTF":[s]}, {"HTF":[z]}, {"atr_len":17, "near_zone_atr_mult": 10.0})
    assert feats.get("sr_reject_up", 0.0) > 0.0
    assert feats.get("sr_near_resistance", 0.0) >= 0.0
    weights = sr_reason_weights_default()
    assert "sr_reject_up" in weights
