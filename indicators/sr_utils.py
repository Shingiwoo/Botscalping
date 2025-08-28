from __future__ import annotations
from typing import Dict, Tuple, List, Union, Iterable
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
import re


def _normalize_resample_rule(rule: str) -> str:
    """
    Ubah alias jam kapital ('1H','4H',...) menjadi huruf kecil ('1h','4h')
    agar kompatibel dengan Pandas terbaru.
    Tidak menyentuh alias lain (hindari mengubah 'MS', dsb).
    """
    r = str(rule)
    # match "<digit(s)>H" saja
    return re.sub(r'(?<=\d)H\b', 'h', r)

def _resample_close(df: pd.DataFrame, rule: str) -> pd.Series:
    tmp = df.set_index('timestamp')[['close']].copy()
    use_rule = _normalize_resample_rule(rule)
    return tmp['close'].resample(use_rule).last().dropna()

def htf_trend_ok_multi(side: str, base_df: pd.DataFrame, rules: Tuple[str, ...] = ("1h","4h")) -> bool:
    """
    Valid jika EMA50>=EMA200 di SEMUA HTF untuk LONG (sebaliknya untuk SHORT).
    Mengembalikan bool murni (bukan numpy.bool_).
    """
    try:
        for rule in rules:
            htf = _resample_close(base_df, rule)
            if len(htf) < 220:
                continue
            ema50  = htf.ewm(span=50, adjust=False).mean().iloc[-1]
            ema200 = htf.ewm(span=200, adjust=False).mean().iloc[-1]
            cond = (ema50 >= ema200) if side == 'LONG' else (ema50 <= ema200)
            if not bool(cond):
                return False
        return True
    except Exception:
        return True

def ltf_momentum_ok(df: pd.DataFrame, lookback: int = 5, rsi_thr_long: float = 52, rsi_thr_short: float = 48) -> Tuple[bool,bool]:
    """
    Proxy momentum TF rendah: micro-ROC & RSI pendek (3–7).
    Return tuple(bool,bool) murni.
    """
    try:
        roc = (df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1.0) if len(df) >= lookback+1 else 0.0
        rsi_m = RSIIndicator(df['close'], max(3, min(7, lookback))).rsi().iloc[-1]
        long_ok  = bool((roc > 0) and (rsi_m >= rsi_thr_long))
        short_ok = bool((roc < 0) and (rsi_m <= rsi_thr_short))
        return long_ok, short_ok
    except Exception:
        return True, True

def _swing_points(df: pd.DataFrame, lb:int=3) -> Tuple[pd.Series, pd.Series]:
    h, l = df['high'], df['low']
    # gunakan rolling untuk robust; hasilkan Series boolean
    hh = (h.shift(1).rolling(lb).max() < h) & (h > h.shift(-1).rolling(lb).max())
    ll = (l.shift(1).rolling(lb).min() > l) & (l < l.shift(-1).rolling(lb).min())
    return hh.fillna(False), ll.fillna(False)

def compute_sr_levels(df: pd.DataFrame, lb:int=3, window:int=300, k:int=6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ambil k level resistance & support terkuat dari swing terbaru dalam window.
    Return ndarray float (bisa panjang < k jika data terbatas).
    """
    seg = df.tail(window)
    hh, ll = _swing_points(seg, lb=lb)
    res_lvls = seg['high'][hh].nlargest(k).values if hh.any() else np.array([], dtype=float)
    sup_lvls = seg['low'][ll].nsmallest(k).values if ll.any() else np.array([], dtype=float)
    return np.array(res_lvls, dtype=float), np.array(sup_lvls, dtype=float)

def atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR sederhana (bukan %) pada bar terakhir."""
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period, min_periods=period).mean().iloc[-1]) if len(df) >= period else float(tr.mean() if len(tr) else 0.0)

def near_level(
    price: float,
    levels: Union[np.ndarray, Iterable[float]],
    tol_or_atr: float,
    regime: str | None = None,
    mult_low: float = 1.5,
    mult_mid: float = 2.0,
    mult_high: float = 2.5,
) -> bool:
    """
    Kompatibel belakang:
    - Mode lama (persentase): near_level(price, levels, pct)
      → true bila |level-price|/price <= pct/100
    - Mode ATR-adaptif: near_level(price, levels, atr_value, regime, ...)
      → true bila |level-price| <= ATR * mult(regime)
    """
    if levels is None:
        return False
    lv = np.array(list(levels), dtype=float)
    if lv.size == 0:
        return False
    if regime is None:
        pct = float(tol_or_atr)
        return bool(np.any(np.abs((lv - float(price)) / float(price)) <= pct / 100.0))
    mult = mult_low if regime == "LOW" else (mult_high if regime == "HIGH" else mult_mid)
    safe = float(tol_or_atr) * float(mult)
    return bool(np.any(np.abs(lv - float(price)) <= safe))

def build_sr_cache(df: pd.DataFrame, lb:int=3, window:int=300, k:int=6, recalc_every:int=10) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Bangun cache index→(RES,SUP). Hitung ulang setiap 'recalc_every' bar saja.
    Index yang tidak dihitung ulang akan memakai level terakhir (caller harus fallback).
    """
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    last: Tuple[np.ndarray, np.ndarray] = (np.array([], dtype=float), np.array([], dtype=float))
    for i in range(len(df)):
        if i < window:
            continue
        if i % max(1, int(recalc_every)) == 0:
            last = compute_sr_levels(df.iloc[:i+1], lb=lb, window=window, k=k)
            cache[i] = last
    if (len(df)-1) not in cache:
        cache[len(df)-1] = last
    return cache
