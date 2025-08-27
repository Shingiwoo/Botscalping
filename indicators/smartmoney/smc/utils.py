from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Dict

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # TR tetap Series agar .rolling() valid (hindari ndarray)
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def rolling_pivot_high(high: pd.Series, length: int) -> pd.Series:
    # pivot high pada bar t jika high[t] == rolling max & merupakan puncak lokal
    return (high == high.rolling(window=length*2+1, center=True).max())

def rolling_pivot_low(low: pd.Series, length: int) -> pd.Series:
    return (low == low.rolling(window=length*2+1, center=True).min())

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Ketatkan typing utk memuaskan Pylance; runtime Pandas menerima mapping ini
    agg: Dict[str, Any] = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }
    out = df.resample(rule).agg(agg)  # type: ignore[arg-type]
    return out.dropna()

def cumulative_mean_range(df: pd.DataFrame) -> float:
    # Approx untuk 'Cumulative Mean Range' di Pine
    rng = (df['high'] - df['low']).abs()
    return float(rng.cumsum().iloc[-1] / max(1, len(rng)))
