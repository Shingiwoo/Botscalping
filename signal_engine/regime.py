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

def scale_weights_nonlinear(
    w: Dict[str, float],
    vol_metrics: Dict[str, float],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Non-linear dynamic scaling:
    cfg example:
      {
        "atr_pct": {"k": 3.0, "thr": 0.02, "effects": {"adx": -0.5, "sr_breakout": 0.5}},
        "bb_width": {"k": 3.0, "thr": 0.02, "effects": {"width_atr": -0.7, "sd_proximity": 0.3}}
      }
    Each effect applies: w'[key] = w[key] * (1 + effect * tanh(k * (metric - thr)))
    Positive effect means increase when metric > thr; negative reduces.
    Missing keys are ignored. Returns new dict.
    """
    out = dict(w)
    atr = float(vol_metrics.get("atr_pct", 0.0))
    bbw = float(vol_metrics.get("bb_width", 0.0))
    metrics = {"atr_pct": atr, "bb_width": bbw}
    for m_name, m_cfg in (cfg or {}).items():
        try:
            if m_name not in metrics:
                continue
            val = float(metrics[m_name])
            k = float(m_cfg.get("k", 3.0))
            thr = float(m_cfg.get("thr", 0.0))
            eff: Dict[str, float] = dict(m_cfg.get("effects", {}))
            scale_val = float(np.tanh(k * (val - thr)))
            for key, e in eff.items():
                if key in out:
                    out[key] = float(out[key]) * (1.0 + float(e) * scale_val)
        except Exception:
            continue
    return out
