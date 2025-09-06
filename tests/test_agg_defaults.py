from engine_core import _read_agg_from_cfg


def test_agg_defaults_merge_minimal():
    # Minimal aggregator block should yield defaults filled
    coin_cfg = {"_agg": {"score_gate": 0.6}}
    agg = _read_agg_from_cfg(coin_cfg)
    assert isinstance(agg, dict)
    assert agg.get("score_gate") == 0.6
    # Defaults present
    assert "strength_thresholds" in agg
    assert "min_strength" in agg
    assert "score_gate_no_confirms" in agg
    assert "min_strength_no_confirms" in agg
    assert "no_confirms_require" in agg

