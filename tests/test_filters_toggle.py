import pandas as pd, numpy as np
from engine_core import apply_filters


def test_filters_can_be_disabled():
    ind = pd.Series({"atr_pct": 0.10, "body_to_atr": 9.99, "close": 1.0})
    cfg = {"symbol":"TEST","filters":{"atr": False, "body": False}, "max_body_atr": 1.0, "min_atr_pct":0.02, "max_atr_pct":0.03}
    atr_ok, body_ok, _ = apply_filters(ind, cfg)
    assert atr_ok is True and body_ok is True

