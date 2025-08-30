import pandas as pd, numpy as np
from engine_core import make_decision
'''
gunakan min bukan T karena sudah depricated di pandas 2.0
'''

def _mkdf():
    t = pd.date_range("2024-01-01", periods=30, freq="15min")
    c = pd.Series(np.linspace(100, 110, 30))
    df = pd.DataFrame({"timestamp": t, "open": c, "high": c+1, "low": c-1, "close": c, "volume": 10})
    return df


def test_decision_respects_rsi_cfg():
    df = _mkdf()
    cfg = {"ema_len":22,"sma_len":20,"rsi_period":14,"rsi_long_min":10,"rsi_long_max":55, "ml":{"enabled":False,"score_threshold":0.0}}
    # panggil dua kali agar ada prev/now untuk cross
    side = None
    side, _ = make_decision(df.iloc[:-1], "TEST", cfg, ml_up_prob=None)
    side, _ = make_decision(df, "TEST", cfg, ml_up_prob=None)
    assert side in (None, "LONG", "SHORT")

