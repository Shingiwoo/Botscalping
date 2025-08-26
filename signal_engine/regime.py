from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def compute_vol_metrics(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    atr_pct = (df["high"] - df["low"]).rolling(lookback).mean() / df["close"].rolling(lookback).mean()
    bbw = (df["close"].rolling(lookback).std() * 2.0) / df["close"].rolling(lookback).mean()
    return {
        "atr_pct": float(atr_pct.iloc[-1]) if len(atr_pct) else 0.0,
        "bb_width": float(bbw.iloc[-1]) if len(bbw) else 0.0,
    }

def classify_regime(atr_pct: float, bb_width: float, bounds: Dict[str, float]) -> str:
    low = (atr_pct < bounds["atr_p1"]) and (bb_width < bounds["bbw_q1"])
    high = (atr_pct > bounds["atr_p2"]) and (bb_width > bounds["bbw_q2"])
    return "HIGH" if high else "LOW" if low else "MID"

def scale_weights(regime: str, w: Dict[str, float], scale: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out = dict(w)
    for k, v in scale.get(regime, {}).items():
        if k in out:
            out[k] = float(out[k]) * float(v)
    return out
