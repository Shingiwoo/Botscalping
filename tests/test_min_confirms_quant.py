import pandas as pd
import numpy as np
import types
import engine_core as ec


def _df():
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = pd.DataFrame({
        "timestamp": ts,
        "open": np.linspace(100, 101, n),
        "high": np.linspace(100.5, 101.5, n),
        "low": np.linspace(99.5, 100.5, n),
        "close": np.linspace(100.2, 101.2, n),
        "volume": np.linspace(1000, 2000, n),
    })
    return base


def test_min_confirms_gate_blocks_when_zero(monkeypatch):
    df = _df()
    # Stub agg_signal to return ok with zero confirms and empty breakdown
    def fake_agg(df_, side, w, thresholds, regime_bounds, sr_penalty, htf_rules=("1h","4h"), features=None):
        return {
            "ok": True,
            "side": side,
            "score": 0.80,
            "strength": "kuat",
            "reasons": [],
            "breakdown": {},
            "context": {"confirms": 0}
        }

    monkeypatch.setattr(ec, "agg_signal", fake_agg, raising=True)
    coin_cfg = {
        "sr_mtf": {"chart_tf": "5m"},
        "_agg": {"score_gate": 0.60, "score_gate_no_confirms": 0.70,
                  "min_strength": "cukup", "min_strength_no_confirms": "kuat",
                  "no_confirms_require": ["adx","width_atr","body_atr"]}
    }
    side, details = ec.make_decision(df, "ADAUSDT", coin_cfg, ml_up_prob=None)
    # Aggregator path should not return (blocked by min_confirms), so either None or base-rule
    if details:
        assert details[0].get("source") != "aggregator"

