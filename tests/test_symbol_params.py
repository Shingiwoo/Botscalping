# tests/test_symbol_params.py
from indicators.scsignal.scsignals import SCConfig


def test_scconfig_from_dict_defaults():
    c = SCConfig.from_dict({})
    assert isinstance(c.atr_mult, float)
    assert c.base_tf == "1m"


def test_merge_like_adapter_logic():
    default_cfg = dict(length=20, atr_mult=0.6, htf="1m", ema_fast_len=20, ema_slow_len=60)
    override = dict(length=22, htf="5m")
    base = SCConfig.from_dict(default_cfg)
    merged = {**base.__dict__, **override}
    final = SCConfig.from_dict(merged)
    assert final.length == 22
    assert final.atr_mult == 0.6
    assert final.htf == "5m"


def test_partial_override_keeps_other_fields():
    default_cfg = dict(min_body_atr=0.38, min_adx=18.0)
    override = dict(min_body_atr=0.30)
    base = SCConfig.from_dict(default_cfg)
    final = SCConfig.from_dict({**base.__dict__, **override})
    assert final.min_body_atr == 0.30
    assert final.min_adx == 18.0

