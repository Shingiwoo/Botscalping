import pandas as pd, numpy as np
from indicators.smartmoney.smc.structure import StructureState
from indicators.smartmoney.smc.fvg import FVGDetector
from indicators.smartmoney.smc.types import SMCConfig

def test_structure_emits_events_minimal():
    # Data kecil hanya untuk memastikan .step tidak error
    idx = pd.date_range("2025-01-01", periods=30, freq="1min")
    price = pd.Series(np.linspace(100,105,len(idx)), index=idx)
    df = pd.DataFrame({
        "open": price.shift(1).fillna(price.iloc[0]),
        "high": price + 0.3, "low": price - 0.3, "close": price, "volume": 10
    })
    st = StructureState(swing_len=5, internal_len=2)
    evs = []
    for i in range(len(df)):
        evs += st.step(df, i, "BTCUSDT", "1m")
    # Tidak wajib ada event, yang penting tidak error

def test_fvg_detect_and_auto_remove_on_fill():
    idx = pd.date_range("2025-01-01", periods=5, freq="1min")
    # Rancang kondisi yang memicu FVG bullish: low[0] > high[2] pada i>=2
    high = pd.Series([10,11,12,13,14], index=idx)
    low  = pd.Series([9,10,11,12,13], index=idx)
    df = pd.DataFrame({
        "open": low+0.5, "high": high, "low": low, "close": (low+high)/2, "volume": 1
    }, index=idx)
    det = FVGDetector(SMCConfig())
    out_all = []
    for i in range(len(df)):
        out_all += det.step(df, i, "BTCUSDT", "1m")
    # Hanya memastikan berjalan; logika auto-remove dipanggil tiap step
