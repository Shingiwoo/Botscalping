
# ultimate_rsi.py
# ------------------------------------------------------------
# Modular Ultimate RSI (URSI) indicator for trading applications
# Spec-aligned with LuxAlgo's "Ultimate RSI" (https://www.tradingview.com/script/17Jj7Vcg-Ultimate-RSI-LuxAlgo/)
# Author: ChatGPT (Indikator Trading)
# License: MIT
# ------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

MAType = Literal["EMA", "SMA", "RMA", "TMA"]
SourceType = Literal["close", "hl2", "hlc3", "ohlc4"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _select_source_from_df(df: pd.DataFrame, src: SourceType, price_col: str = "close") -> pd.Series:
    src = src.lower()
    if src == "close":
        return df[price_col].astype(float)
    if src == "hl2":
        return ((df["high"] + df["low"]) / 2.0).astype(float)
    if src == "hlc3":
        return ((df["high"] + df["low"] + df[price_col]) / 3.0).astype(float)
    if src == "ohlc4":
        return ((df["open"] + df["high"] + df["low"] + df[price_col]) / 4.0).astype(float)
    raise ValueError(f"Unsupported source type: {src}")

def _select_source_from_tuple(ohlc: Tuple[float, float, float, float], src: SourceType) -> float:
    o, h, l, c = map(float, ohlc)
    if src == "close":
        return c
    if src == "hl2":
        return (h + l) / 2.0
    if src == "hlc3":
        return (h + l + c) / 3.0
    if src == "ohlc4":
        return (o + h + l + c) / 4.0
    raise ValueError(f"Unsupported source type: {src}")

def _ma_series(series: pd.Series, length: int, ma_type: MAType) -> pd.Series:
    if length <= 0:
        raise ValueError("length must be > 0")
    if ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    if ma_type == "RMA":
        return series.ewm(alpha=1.0/length, adjust=False).mean()
    if ma_type == "SMA":
        return series.rolling(length, min_periods=length).mean()
    if ma_type == "TMA":
        return series.rolling(length, min_periods=length).mean()                     .rolling(length, min_periods=length).mean()
    raise ValueError(f"Unsupported ma_type: {ma_type}")

# ------------------------------------------------------------
# Vectorized URSI
# ------------------------------------------------------------
def ursi_vectorized(
    src: Union[pd.Series, Iterable[float], pd.DataFrame],
    length: int = 14,
    method1: MAType = "RMA",   # core smoothing for num/den
    smooth: int = 14,
    method2: MAType = "EMA",   # signal smoothing
    # Source selection (if src is DataFrame)
    source: SourceType = "close",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute Ultimate RSI (URSI) vectorized over a full series.

    Parameters
    ----------
    src : Series/iterable OR DataFrame with columns [open,high,low,close]
    length : int
        Lookback window for HH/LL & RSI core.
    method1 : MAType
        Smoothing for numerator/denominator (EMA/SMA/RMA/TMA).
    smooth : int
        Signal smoothing length.
    method2 : MAType
        Smoothing for signal line.
    source : SourceType
        Input source if DataFrame given (close/hl2/hlc3/ohlc4).
    price_col : str
        Column name for close if DataFrame.

    Returns
    -------
    DataFrame columns: ['arsi', 'signal', 'ob', 'os', 'mid']
    """
    if isinstance(src, pd.DataFrame):
        s = _select_source_from_df(src, source, price_col=price_col)
    else:
        s = pd.Series(src).astype(float)

    upper = s.rolling(length, min_periods=length).max()
    lower = s.rolling(length, min_periods=length).min()
    r = upper - lower
    d = s.diff(1)

    # LuxAlgo spec: diff = upper > upper[1] ? r : lower < lower[1] ? -r : d
    cond_up = (upper > upper.shift(1))
    cond_dn = (lower < lower.shift(1))
    diff = pd.Series(np.where(cond_up, r, np.where(cond_dn, -r, d)), index=s.index)

    num = _ma_series(diff, length, method1)
    den = _ma_series(diff.abs(), length, method1)

    with np.errstate(divide='ignore', invalid='ignore'):
        arsi = (num / den) * 50.0 + 50.0
    arsi = arsi.clip(lower=0.0, upper=100.0)

    signal = _ma_series(arsi, smooth, method2)

    out = pd.DataFrame({
        "arsi": arsi,
        "signal": signal,
        "ob": 80.0,
        "os": 20.0,
        "mid": 50.0,
    }, index=s.index)
    return out

# ------------------------------------------------------------
# Incremental MAs for streaming
# ------------------------------------------------------------
class _EMAInc:
    def __init__(self, length: int, alpha: Optional[float] = None):
        self.length = int(length)
        self.alpha = (2.0/(length+1.0)) if alpha is None else float(alpha)
        self.value: Optional[float] = None
        self.ready = False
    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
            self.ready = False
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
            self.ready = True
        return self.value

class _SMAInc:
    def __init__(self, length: int):
        self.length = int(length)
        self.buf: Deque[float] = deque(maxlen=self.length)
        self.sum = 0.0
        self.value: Optional[float] = None
        self.ready = False
    def update(self, x: float) -> float:
        if len(self.buf) == self.length:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        if len(self.buf) == self.length:
            self.value = self.sum / self.length
            self.ready = True
        else:
            self.value = self.sum / len(self.buf)
            self.ready = False
        return self.value

class _RMAInc(_EMAInc):
    # Wilder's alpha = 1/length
    def __init__(self, length: int):
        super().__init__(length, alpha=1.0/length)

class _TMAInc:
    # TMA = SMA(SMA(x, len), len)
    def __init__(self, length: int):
        self.sma1 = _SMAInc(length)
        self.sma2 = _SMAInc(length)
        self.value: Optional[float] = None
        self.ready = False
    def update(self, x: float) -> float:
        v1 = self.sma1.update(x)
        v2 = self.sma2.update(v1)
        self.value = v2
        self.ready = self.sma2.ready
        return v2

def _make_inc_ma(ma_type: MAType, length: int):
    if ma_type == "EMA": return _EMAInc(length)
    if ma_type == "RMA": return _RMAInc(length)
    if ma_type == "SMA": return _SMAInc(length)
    if ma_type == "TMA": return _TMAInc(length)
    raise ValueError(f"Unsupported MA type: {ma_type}")

# ------------------------------------------------------------
# Streaming URSI
# ------------------------------------------------------------
@dataclass
class URSIConfig:
    length: int = 14
    method1: MAType = "RMA"
    smooth: int = 14
    method2: MAType = "EMA"
    ob: float = 80.0
    os: float = 20.0
    source: SourceType = "close"  # input source (close/hl2/hlc3/ohlc4)

@dataclass
class URSISnapshot:
    arsi: Optional[float]
    signal: Optional[float]
    upper: Optional[float]
    lower: Optional[float]
    r: Optional[float]
    num_ma: Optional[float]
    den_ma: Optional[float]
    ready: bool

class UltimateRSI:
    """
    Streaming-friendly URSI.

    update() can accept either:
    - a single float (treated as 'close' per config.source)
    - a tuple (o, h, l, c) to honor config.source precisely
    """
    def __init__(self, cfg: URSIConfig = URSIConfig()):
        self.cfg = cfg
        L = self.cfg.length
        self.window: Deque[float] = deque(maxlen=L)
        self.prev_upper: Optional[float] = None
        self.prev_lower: Optional[float] = None
        self.prev_src: Optional[float] = None
        self.ma_num = _make_inc_ma(self.cfg.method1, L)
        self.ma_den = _make_inc_ma(self.cfg.method1, L)
        self.ma_sig = _make_inc_ma(self.cfg.method2, self.cfg.smooth)
        self.arsi: Optional[float] = None
        self.signal: Optional[float] = None
        self._prev_arsi: Optional[float] = None
        self._prev_signal: Optional[float] = None

    def _rolling_high(self) -> float:
        return max(self.window) if self.window else float('nan')
    def _rolling_low(self) -> float:
        return min(self.window) if self.window else float('nan')

    def _prep_src_value(self, x: Union[float, Tuple[float, float, float, float]]) -> float:
        if isinstance(x, tuple) or isinstance(x, list):
            o, h, l, c = map(float, x)
            return _select_source_from_tuple((o, h, l, c), self.cfg.source)
        else:
            return float(x)

    def update(self, x: Union[float, Tuple[float, float, float, float]]) -> URSISnapshot:
        src = self._prep_src_value(x)
        self.window.append(float(src))
        upper = self._rolling_high()
        lower = self._rolling_low()
        r = upper - lower if (upper == upper and lower == lower) else float('nan')

        d = (src - self.prev_src) if (self.prev_src is not None) else 0.0
        if self.prev_upper is None or self.prev_lower is None:
            diff = d
        else:
            if upper > self.prev_upper:
                diff = r
            elif lower < self.prev_lower:
                diff = -r
            else:
                diff = d

        num = self.ma_num.update(diff)
        den = self.ma_den.update(abs(diff))

        if den is None or den == 0:
            arsi = None
        else:
            arsi = (num / den) * 50.0 + 50.0
            arsi = max(0.0, min(100.0, arsi))

        sig = self.ma_sig.update(arsi if arsi is not None else 50.0)

        self._prev_arsi, self._prev_signal = self.arsi, self.signal
        self.arsi, self.signal = arsi, sig

        self.prev_upper, self.prev_lower, self.prev_src = upper, lower, src

        ready = self.ma_sig.ready and self.ma_den.ready and self.ma_num.ready and (len(self.window) >= self.cfg.length)
        return URSISnapshot(
            arsi=self.arsi,
            signal=self.signal,
            upper=upper,
            lower=lower,
            r=r,
            num_ma=num,
            den_ma=den,
            ready=ready,
        )

    def crosses_above(self, a_prev: float, a_cur: float, b_prev: float, b_cur: float) -> bool:
        return a_prev is not None and b_prev is not None and a_prev <= b_prev and a_cur > b_cur
    def crosses_below(self, a_prev: float, a_cur: float, b_prev: float, b_cur: float) -> bool:
        return a_prev is not None and b_prev is not None and a_prev >= b_prev and a_cur < b_cur

    def has_signal_cross(self) -> Tuple[bool, Optional[str]]:
        if self._prev_arsi is None or self._prev_signal is None or self.arsi is None or self.signal is None:
            return (False, None)
        if self.crosses_above(self._prev_arsi, self.arsi, self._prev_signal, self.signal):
            return (True, "ARSI_CROSS_UP_SIGNAL")
        if self.crosses_below(self._prev_arsi, self.arsi, self._prev_signal, self.signal):
            return (True, "ARSI_CROSS_DOWN_SIGNAL")
        if self.crosses_above(self._prev_arsi, self.arsi, self.cfg.os, self.cfg.os):
            return (True, "ARSI_CROSS_UP_OS")
        if self.crosses_below(self._prev_arsi, self.arsi, self.cfg.ob, self.cfg.ob):
            return (True, "ARSI_CROSS_DOWN_OB")
        return (False, None)

    def generate_event(self, symbol: str, timestamp: Union[pd.Timestamp, float, int, None] = None) -> Optional[Dict]:
        if self.arsi is None or self.signal is None:
            return None
        crossed, reason = self.has_signal_cross()
        side: Optional[str] = None
        if reason in ("ARSI_CROSS_UP_SIGNAL", "ARSI_CROSS_UP_OS"):
            side = "BUY"
        elif reason in ("ARSI_CROSS_DOWN_SIGNAL", "ARSI_CROSS_DOWN_OB"):
            side = "SELL"
        return {
            "type": "indicator_signal",
            "name": "URSI",
            "symbol": symbol,
            "time": pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.utcnow(),
            "arsi": float(self.arsi) if self.arsi is not None else None,
            "signal": float(self.signal) if self.signal is not None else None,
            "reason": reason,
            "side": side,
            "config": vars(self.cfg),
        }

# ------------------------------------------------------------
# Convenience: full-DF compute (backtesting)
# ------------------------------------------------------------
def compute_ursi_df(
    df: pd.DataFrame,
    length: int = 14,
    method1: MAType = "RMA",
    smooth: int = 14,
    method2: MAType = "EMA",
    source: SourceType = "close",
    price_col: str = "close",
) -> pd.DataFrame:
    out = ursi_vectorized(df, length, method1, smooth, method2, source=source, price_col=price_col)
    df2 = df.copy()
    df2["URSI"] = out["arsi"].values
    df2["URSI_signal"] = out["signal"].values
    df2["URSI_ob"] = 80.0
    df2["URSI_os"] = 20.0
    df2["URSI_mid"] = 50.0
    return df2

# ------------------------------------------------------------
# RajaDollar adapter
# ------------------------------------------------------------
class URSIAdapter:
    def __init__(self, symbol: str, cfg: URSIConfig = URSIConfig(), queue=None):
        self.symbol = symbol
        self.ind = UltimateRSI(cfg)
        self.queue = queue
    def on_price(self, x: Union[float, Tuple[float, float, float, float]], timestamp: Union[pd.Timestamp, float, int, None] = None):
        self.ind.update(x)
        evt = self.ind.generate_event(self.symbol, timestamp)
        if evt and self.queue is not None:
            put = getattr(self.queue, "put_nowait", None) or getattr(self.queue, "put", None)
            if put is not None:
                try:
                    put(evt)
                except TypeError:
                    self.queue.put(evt)
        return evt

# ------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    close = pd.Series(np.cumsum(rng.normal(0, 1, 300)) + 100.0,
                      index=pd.date_range("2024-01-01", periods=300, freq="1min"))
    df = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]),
                       "high": close + 0.5,
                       "low": close - 0.5,
                       "close": close})

    out = compute_ursi_df(df, source="hlc3")
    print(out.tail())

    ursi = UltimateRSI(URSIConfig(source="ohlc4"))
    for t, row in df.tail(20).iterrows():
        ohlc = (row.open, row.high, row.low, row.close)
        snap = ursi.update(ohlc)
        evt = ursi.generate_event(symbol="BTCUSDT", timestamp=t)
        if evt and evt["reason"]:
            print(t, evt)
