import pandas as pd, numpy as np
from indicators.smartmoney.smc.orderblock import OrderBlockEngine
from indicators.smartmoney.smc.types import SMCConfig, SMCEvent

def _df(n=60):
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    o = pd.Series(np.linspace(100, 101, n), index=idx)
    c = o.copy()
    c.iloc[10] = o.iloc[10] - 0.5  # bearish candle
    c.iloc[20] = o.iloc[20] + 0.5  # bullish candle
    h = pd.concat([o, c], axis=1).max(axis=1) + 0.2
    l = pd.concat([o, c], axis=1).min(axis=1) - 0.2
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":100})

def test_internal_ob_new_typing_safe():
    df = _df()
    eng = OrderBlockEngine(SMCConfig())
    i = 21
    ev = SMCEvent("internal_bos","BTCUSDT","1m",i,df.index[i], {"level": float(df["high"].iloc[i-1])})
    out = eng.on_structure_event(ev, df)
    for e in out:
        if e.type.endswith("_ob_new"):
            ob = e.payload["ob"]
            assert isinstance(ob["start_idx"], int)
            assert "start_time" in ob

def test_duplicate_index_does_not_break_start_idx():
    df = _df()
    dup_row = df.iloc[[15]].copy(); dup_row.index = [df.index[15]]
    df = pd.concat([df.iloc[:16], dup_row, df.iloc[16:]]).sort_index()
    eng = OrderBlockEngine(SMCConfig())
    i = 25
    ev = SMCEvent("internal_bos","BTCUSDT","1m",i,df.index[i], {"level": float(df["high"].iloc[i-1])})
    out = eng.on_structure_event(ev, df)
    for e in out:
        if e.type.endswith("_ob_new"):
            assert isinstance(e.payload["ob"]["start_idx"], int)
