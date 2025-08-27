import pandas as pd, numpy as np
from indicators.smartmoney.smc.indicator import SMCIndicator
from indicators.smartmoney.smc.types import SMCConfig

def test_indicator_process_dataframe_end_to_end():
    idx = pd.date_range("2025-01-01", periods=120, freq="1min")
    price = pd.Series(np.cumsum(np.random.randn(len(idx))*0.1)+100, index=idx)
    df = pd.DataFrame({
        "open": price.shift(1).fillna(price.iloc[0]),
        "high": price + 0.2, "low": price - 0.2, "close": price, "volume": 100
    })
    ind = SMCIndicator("BTCUSDT","1m", SMCConfig(show_daily=True, show_weekly=True))
    events = ind.process_dataframe(df)
    assert isinstance(events, list)
