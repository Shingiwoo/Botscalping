from __future__ import annotations
import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    return tr.rolling(period, min_periods=1).mean()

def rolling_pivot_high(high: pd.Series, length: int) -> pd.Series:
    # pivot high pada bar t jika high[t] == rolling max & merupakan puncak lokal
    return (high == high.rolling(window=length*2+1, center=True).max())

def rolling_pivot_low(low: pd.Series, length: int) -> pd.Series:
    return (low == low.rolling(window=length*2+1, center=True).min())

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }
    out = df.resample(rule).agg(agg).dropna()
    return out

def cumulative_mean_range(df: pd.DataFrame) -> float:
    # Approx untuk 'Cumulative Mean Range' di Pine
    rng = (df['high'] - df['low']).abs()
    return float(rng.cumsum().iloc[-1] / max(1, len(rng)))
