from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

"""
LuxAlgo - Fair Value Gap (FVG) port ke Python
================================================
Mereplikasi logika Pine v5:
- Bullish FVG: low[i] > high[i-2] & close[i-1] > high[i-2] & (low[i]-high[i-2])/high[i-2] > threshold
- Bearish FVG: high[i] < low[i-2] & close[i-1] < low[i-2] & (low[i-2]-high[i])/high[i] > threshold
- Auto threshold (opsional) = rata-rata kumulatif (high-low)/low
- Mitigasi: Bullish dianggap mitigated ketika close < min (batas bawah gap),
            Bearish mitigated ketika close > max (batas atas gap).
Opsi style (extend/dynamic/mitigationLevels/showLast) disiapkan sebagai metadata; visualisasi box/line tidak digambar di sini.

API utama:
  detect_fvg(df: DataFrame, **opts) -> Dict[str, Any]
Mengembalikan ringkasan termasuk flag:
  - 'bullish_unfilled' (bool): ada FVG bullish aktif (belum mitigasi) di bar terakhir
  - 'bearish_unfilled' (bool): ada FVG bearish aktif (belum mitigasi) di bar terakhir
Sesuai expectations aggregator (signal_engine.adapters.SMC_FVG).
"""

@dataclass
class FVGRecord:
    max: float
    min: float
    isbull: bool
    t: pd.Timestamp           # waktu bar pembentukan (bar i)
    i: int                    # index bar i (pada df yang dipakai deteksi)
    start_i: int              # i-2 (awal box)
    end_i: int                # i+extend (akhir box - untuk keperluan plotting eksternal)
    mitigated_at: Optional[pd.Timestamp] = None

def _normalize_tf(tf: Optional[str]) -> Optional[str]:
    if not tf or str(tf).strip() == "" or str(tf).lower() == "chart":
        return None
    s = str(tf).lower().strip()
    s = s.replace(" ", "")
    # map contoh umum
    if s in {"15", "15m", "15min", "15t"}: return "15T"
    if s in {"5", "5m", "5min", "5t"}: return "5T"
    if s in {"1h", "60", "60m"}: return "1H"
    if s in {"4h", "240"}: return "4H"
    if s in {"1d", "d"}: return "1D"
    if s in {"1w", "w"}: return "1W"
    # fallback: biarkan pandas coba parse
    return s

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample anti-lookahead: label='right', closed='right' (seperti security TradingView)."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame harus memiliki kolom 'timestamp'")
    ohlcv = df.set_index("timestamp")[ ["open","high","low","close","volume"] ]
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = ohlcv.resample(rule, label="right", closed="right").agg(agg).dropna()
    out = out.reset_index()
    return out

def _auto_threshold(df: pd.DataFrame) -> float:
    # threshold = ta.cum((high-low)/low)/bar_index  → nilai rata-rata kumulatif di bar terakhir
    rng = (df["high"] - df["low"]) / df["low"].replace(0, np.nan)
    cum = rng.cumsum()
    idx = np.arange(1, len(df) + 1, dtype=float)
    thr = float((cum / idx).iloc[-1]) if len(df) else 0.0
    if not np.isfinite(thr): thr = 0.0
    return max(0.0, thr)

def _compute_threshold(df: pd.DataFrame, threshold_per: float, auto: bool) -> float:
    if auto:
        return _auto_threshold(df)
    return float(threshold_per) / 100.0

def _iter_detect(df: pd.DataFrame, threshold: float, extend: int) -> List[FVGRecord]:
    recs: List[FVGRecord] = []
    # butuh i >= 2
    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low  = df["low"].to_numpy(dtype=float)
    close= df["close"].to_numpy(dtype=float)
    ts   = df["timestamp"].to_numpy()
    for i in range(2, n):
        # Bullish FVG
        bull = (low[i] > high[i-2]) and (close[i-1] > high[i-2])
        if bull:
            gap_per = (low[i] - high[i-2]) / max(high[i-2], 1e-12)
            if gap_per > threshold:
                recs.append(FVGRecord(
                    max=float(low[i]),
                    min=float(high[i-2]),
                    isbull=True,
                    t=pd.Timestamp(ts[i]),
                    i=i,
                    start_i=i-2,
                    end_i=i+extend
                ))
                continue
        # Bearish FVG
        bear = (high[i] < low[i-2]) and (close[i-1] < low[i-2])
        if bear:
            gap_per = (low[i-2] - high[i]) / max(high[i], 1e-12)
            if gap_per > threshold:
                recs.append(FVGRecord(
                    max=float(low[i-2]),
                    min=float(high[i]),
                    isbull=False,
                    t=pd.Timestamp(ts[i]),
                    i=i,
                    start_i=i-2,
                    end_i=i+extend
                ))
    return recs

def _apply_mitigation(df: pd.DataFrame, recs: List[FVGRecord]) -> Tuple[List[FVGRecord], int, int]:
    """Tandai mitigasi berdasarkan close terakhir; hitung kumulatif mitigated sejak awal."""
    bull_mit = 0
    bear_mit = 0
    if not len(df):
        return recs, bull_mit, bear_mit
    last_close = float(df["close"].iloc[-1])
    # iterasi dari paling tua agar konsisten
    for r in recs:
        if r.mitigated_at is not None:
            if r.isbull:
                bull_mit += 1
            else:
                bear_mit += 1
            continue
        if r.isbull and last_close < r.min:
            r.mitigated_at = pd.Timestamp(df["timestamp"].iloc[-1])
            bull_mit += 1
        elif (not r.isbull) and last_close > r.max:
            r.mitigated_at = pd.Timestamp(df["timestamp"].iloc[-1])
            bear_mit += 1
    return recs, bull_mit, bear_mit

def detect_fvg(
    df: pd.DataFrame,
    *,
    thresholdPer: float = 0.0,
    auto: bool = False,
    showLast: int = 0,
    mitigationLevels: bool = False,  # disimpan sebagai metadata
    tf: Optional[str] = None,
    extend: int = 20,
    dynamic: bool = False
) -> Dict[str, Any]:
    """
    Deteksi FVG ala LuxAlgo. Kembalikan ringkasan + flag untuk aggregator.
    Kolom wajib: timestamp, open, high, low, close, volume.
    """
    if not {"timestamp","open","high","low","close"}.issubset(df.columns):
        raise ValueError("DataFrame harus mengandung kolom: timestamp, open, high, low, close")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Timeframe override (opsional)
    rule = _normalize_tf(tf)
    src = _resample_ohlcv(df, rule) if rule else df.copy()
    # Hitung threshold
    thr = _compute_threshold(src, thresholdPer, auto)
    # Deteksi
    recs = _iter_detect(src, thr, extend=extend)
    # Mitigasi berdasarkan close TERAKHIR
    recs, bull_mit, bear_mit = _apply_mitigation(src, recs)
    # Filter unmitigated untuk status saat ini
    unmitigated = [r for r in recs if r.mitigated_at is None]
    bull_unfilled = any(r.isbull for r in unmitigated)
    bear_unfilled = any((not r.isbull) for r in unmitigated)

    # Dynamic mode (opsional): kembalikan batas dinamis untuk plotter eksternal
    max_bull = min_bull = max_bear = min_bear = None
    if dynamic and len(unmitigated):
        # Ambil FVG terakhir masing-masing arah, lalu "geser" dengan clamp ke harga close (mimik script)
        last_close = float(src["close"].iloc[-1])
        # Bullish
        bulls = [r for r in unmitigated if r.isbull]
        if bulls:
            b = sorted(bulls, key=lambda x: x.i, reverse=True)[0]
            max_bull = max(min(last_close, b.max), b.min)
            min_bull = b.min
        # Bearish
        bears = [r for r in unmitigated if not r.isbull]
        if bears:
            b = sorted(bears, key=lambda x: x.i, reverse=True)[0]
            min_bear = min(max(last_close, b.min), b.max)
            max_bear = b.max

    # Siapkan output
    out: Dict[str, Any] = {
        "bull_count": int(sum(1 for r in recs if r.isbull)),
        "bear_count": int(sum(1 for r in recs if not r.isbull)),
        "bull_mitigated": int(bull_mit),
        "bear_mitigated": int(bear_mit),
        "bullish_unfilled": bool(bull_unfilled),
        "bearish_unfilled": bool(bear_unfilled),
        "threshold_used": float(thr),
        "mitigationLevels": bool(mitigationLevels),
        "tf_used": rule or "chart",
        "extend": int(extend),
        "dynamic": bool(dynamic),
    }
    if dynamic:
        out.update({
            "max_bull_fvg": max_bull,
            "min_bull_fvg": min_bull,
            "max_bear_fvg": max_bear,
            "min_bear_fvg": min_bear,
        })
    # showLast: kembalikan N unmitigated terakhir (untuk konsistensi fitur TV)
    if showLast and showLast > 0:
        last_lines: List[Dict[str, Any]] = []
        # urutkan dari terbaru
        for r in sorted(unmitigated, key=lambda x: x.i, reverse=True)[:showLast]:
            price = r.min if r.isbull else r.max
            last_lines.append({
                "isbull": r.isbull,
                "price": float(price),
                "t_start": pd.Timestamp(src["timestamp"].iloc[r.start_i]),
                "t_end":   pd.Timestamp(src["timestamp"].iloc[-1]),
            })
        out["unmitigated_lines"] = last_lines
    # (opsional) kembalikan records mentah untuk debug
    out["records"] = [asdict(r) for r in recs]
    return out

# Kelas opsional untuk pemakaian incremental (live)
class FVGDetectorLuxAlgo:
    def __init__(self, **kwargs) -> None:
        self.kw = dict(kwargs)
    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        return detect_fvg(df, **self.kw)
    # alias agar kompatibel dengan adapter
    __call__ = process_dataframe
