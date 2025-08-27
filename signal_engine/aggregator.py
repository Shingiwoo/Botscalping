from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from .types import AggResult, ScoreBreakdown, Side
from .regime import compute_vol_metrics, classify_regime, scale_weights
from .cache import FeatureCache
from .adapters import (
    SC_API,
    SMC_STRUCTURE,
    SMC_ORDERBLOCK,
    SMC_FVG,
    SMC_LEVELS,
    SMC_ZONES,
    SD_API,
    SR_NEAR,
    SR_BUILD_CACHE,
)


def resample_close(s: pd.Series, rule: str) -> pd.Series:
    rule = str(rule).lower().replace("h", "h").replace("t", "t")
    return s.resample(rule, label="right", closed="right").last().dropna()


def compute_htf_trend(df: pd.DataFrame, rules=("1h", "4h")) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    s = df.set_index("timestamp")["close"]
    for r in rules:
        htf = resample_close(s, r)
        if len(htf) < 220:
            out[r] = True
            continue
        ema50 = htf.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = htf.ewm(span=200, adjust=False).mean().iloc[-1]
        out[r] = bool(ema50 >= ema200)
    return out


def htf_ok_with_pullback(df: pd.DataFrame, side: Side, rules=("1h", "4h")) -> Tuple[bool, Dict[str, bool]]:
    trend = compute_htf_trend(df, rules)
    ok = all(trend.values()) if side == "LONG" else all(not v for v in trend.values())
    lookback = 5
    roc = (df["close"].iloc[-1] / df["close"].iloc[-lookback] - 1.0) if len(df) >= lookback + 1 else 0.0
    if not ok:
        if side == "LONG" and roc > 0:
            ok = True
        if side == "SHORT" and roc < 0:
            ok = True
    return bool(ok), trend


def proximity_score(price: float, wavg: float, max_boost: float, tol_pct: float) -> float:
    if wavg is None or wavg <= 0:
        return 0.0
    dist = abs(price - wavg) / price
    if dist <= (tol_pct / 100.0):
        return max(0.0, (1.0 - dist / (tol_pct / 100.0)) * max_boost)
    return 0.0


def volume_spike_factor(
    vol: pd.Series, lookback: int = 20, z_thr: float = 2.0, max_boost: float = 0.1
) -> float:
    if len(vol) < lookback:
        return 0.0
    v = vol.iloc[-lookback:]
    mu, sd = float(v.mean()), float(v.std(ddof=0))
    if sd <= 0:
        return 0.0
    z = (float(vol.iloc[-1]) - mu) / sd
    return max(0.0, min(max_boost, (z - z_thr) / (z_thr))) if z > z_thr else 0.0


def sr_tolerance_pct(atr_pct: float, base_pct: float, k_atr: float) -> float:
    return float(base_pct) + float(k_atr) * max(0.0, float(atr_pct) * 100.0)


def fvg_confirm_bonus(has_fvg_same_dir: bool, max_bonus: float = 0.10) -> float:
    return float(max_bonus) if has_fvg_same_dir else 0.0


def fvg_contra_penalty(has_fvg_contra: bool, max_penalty: float = 0.10) -> float:
    return -float(max_penalty) if has_fvg_contra else 0.0


def compute_sc_base(df: pd.DataFrame, side: Side, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {"cross": False, "adx_ok": False, "body_atr_ok": False, "width_atr_ok": False, "rsi_ok": False}
    try:
        if SC_API:
            res = SC_API(df, cfg)
            if isinstance(res, dict):
                out["cross"] = True
                out["adx_ok"] = True
                out["body_atr_ok"] = True
                out["width_atr_ok"] = True
                out["rsi_ok"] = True
            else:
                out["cross"] = True
                out["adx_ok"] = out["body_atr_ok"] = out["width_atr_ok"] = True
        else:
            out["cross"] = True
            out["adx_ok"] = out["body_atr_ok"] = out["width_atr_ok"] = True
    except Exception:
        out["cross"] = False
    return out


def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1.0 else x


def build_features_from_modules(df: pd.DataFrame, side: Side) -> Dict[str, Any]:
    features: Dict[str, Any] = {"sr": {}, "fvg": {}, "ob": {}, "sd": {}, "htf_fallback": ""}
    try:
        if SMC_STRUCTURE:
            st = SMC_STRUCTURE(df)
            if isinstance(st, dict):
                if side == "LONG":
                    features["sr"]["breakout_same_dir"] = bool(st.get("breakout_up", False))
                    features["sr"]["retest_same_dir"] = bool(st.get("retest_up", False))
                    features["sr"]["rejection_same_dir"] = bool(st.get("reject_up", False))
                else:
                    features["sr"]["breakout_same_dir"] = bool(st.get("breakout_down", False))
                    features["sr"]["retest_same_dir"] = bool(st.get("retest_down", False))
                    features["sr"]["rejection_same_dir"] = bool(st.get("reject_down", False))
        if SMC_ORDERBLOCK:
            ob = SMC_ORDERBLOCK(df)
            if isinstance(ob, dict):
                features["ob"]["bullish_active"] = bool(ob.get("bull_active", False))
                features["ob"]["bearish_active"] = bool(ob.get("bear_active", False))
        if SMC_FVG:
            fg = SMC_FVG(df)
            if isinstance(fg, dict):
                if side == "LONG":
                    features["fvg"]["has_same_dir"] = bool(fg.get("bullish_unfilled", False))
                    features["fvg"]["has_contra"] = bool(fg.get("bearish_unfilled", False))
                else:
                    features["fvg"]["has_same_dir"] = bool(fg.get("bearish_unfilled", False))
                    features["fvg"]["has_contra"] = bool(fg.get("bullish_unfilled", False))
        if SMC_LEVELS:
            lv = SMC_LEVELS(df)
            price = float(df["close"].iloc[-1])
            tol_pct = 0.5
            lows = [lv.get(k) for k in ("prevD_low", "prevW_low", "prevM_low") if lv.get(k) is not None]
            highs = [lv.get(k) for k in ("prevD_high", "prevW_high", "prevM_high") if lv.get(k) is not None]
            if lows:
                features["sr"]["near_support"] = any(abs(price - float(x)) / price * 100.0 <= tol_pct for x in lows)
            if highs:
                features["sr"]["near_resistance"] = any(abs(price - float(x)) / price * 100.0 <= tol_pct for x in highs)
        if SD_API:
            sd = SD_API(df)
            if isinstance(sd, dict):
                if "demand_wavg" in sd:
                    features["sd"]["demand_wavg"] = float(sd["demand_wavg"])
                if "supply_wavg" in sd:
                    features["sd"]["supply_wavg"] = float(sd["supply_wavg"])
    except Exception:
        pass
    return features


def aggregate(
    df: pd.DataFrame,
    side: Side,
    weights: Dict[str, float],
    thresholds: Dict[str, Any],
    regime_bounds: Dict[str, float],
    sr_penalty_cfg: Dict[str, float],
    htf_rules=("1h", "4h"),
    features: Optional[Dict[str, Any]] = None,
) -> AggResult:
    reasons: list[str] = []
    breakdown: ScoreBreakdown = {}
    vol = compute_vol_metrics(df, lookback=int(thresholds.get("vol_lookback", 20)))
    regime = classify_regime(vol["atr_pct"], vol["bb_width"], regime_bounds)
    w = scale_weights(regime, dict(weights), thresholds.get("weight_scale", {}))

    htf_ok, trend = htf_ok_with_pullback(df, side, rules=htf_rules)
    if htf_ok:
        val = w.get("sc_trend_htf", 0.35)
        breakdown["sc_trend_htf"] = val
    else:
        val = w.get("sc_no_htf", 0.15)
        breakdown["sc_no_htf"] = val

    sc = compute_sc_base(df, side, cfg=thresholds)
    if not sc["cross"]:
        return {
            "ok": False,
            "side": None,
            "score": 0.0,
            "strength": "netral",
            "reasons": ["no_cross"],
            "breakdown": breakdown,
            "context": {"trend": trend, "regime": regime},
        }
    if sc["adx_ok"]:
        breakdown["adx"] = w.get("adx", 0.15)
    if sc["body_atr_ok"]:
        breakdown["body_atr"] = w.get("body_atr", 0.10)
    if sc["width_atr_ok"]:
        breakdown["width_atr"] = w.get("width_atr", 0.10)
    if sc["rsi_ok"]:
        breakdown["rsi"] = w.get("rsi", 0.05)

    sr = (features or {}).get("sr", {})
    fvg = (features or {}).get("fvg", {})
    ob = (features or {}).get("ob", {})
    if sr.get("breakout_same_dir"):
        breakdown["sr_breakout"] = w.get("sr_breakout", 0.20)
        reasons.append("sr_breakout")
    if sr.get("retest_same_dir"):
        breakdown["sr_test"] = w.get("sr_test", 0.10)
        reasons.append("sr_retest")
    if sr.get("rejection_same_dir"):
        breakdown["sr_reject"] = w.get("sr_reject", 0.15)
        reasons.append("sr_reject")
    if fvg.get("has_same_dir"):
        breakdown["fvg_confirm"] = breakdown.get("fvg_confirm", 0.0) + w.get("fvg_confirm", 0.10)
        reasons.append("fvg_confirm")
    if fvg.get("has_contra"):
        breakdown["fvg_confirm"] = breakdown.get("fvg_confirm", 0.0) - w.get("fvg_contra", 0.10)
        reasons.append("fvg_contra")

    sd = (features or {}).get("sd", {})
    price = float(df["close"].iloc[-1])
    tol_pct = thresholds.get("sd_tol_pct", 1.0)
    if side == "LONG" and "demand_wavg" in sd:
        ps = proximity_score(price, float(sd["demand_wavg"]), w.get("sd_proximity", 0.20), tol_pct)
        if ps > 0:
            breakdown["sd_proximity"] = ps
            reasons.append("sd_proximity")
    if side == "SHORT" and "supply_wavg" in sd:
        ps = proximity_score(price, float(sd["supply_wavg"]), w.get("sd_proximity", 0.20), tol_pct)
        if ps > 0:
            breakdown["sd_proximity"] = ps
            reasons.append("sd_proximity")
    vol_boost = volume_spike_factor(
        df["volume"],
        lookback=int(thresholds.get("vol_lookback", 20)),
        z_thr=float(thresholds.get("vol_z_thr", 2.0)),
        max_boost=w.get("vol_confirm", 0.10),
    )
    if vol_boost > 0:
        breakdown["vol_confirm"] = vol_boost
        reasons.append("vol_spike")

    tol_sr = sr_tolerance_pct(
        vol["atr_pct"],
        base_pct=float(sr_penalty_cfg.get("base_pct", 0.6)),
        k_atr=float(sr_penalty_cfg.get("k_atr", 0.5)),
    )
    if side == "LONG" and sr.get("near_resistance", False):
        breakdown["penalty_near_opposite_sr"] = -abs(w.get("penalty_near_opposite_sr", 0.20))
    if side == "SHORT" and sr.get("near_support", False):
        breakdown["penalty_near_opposite_sr"] = -abs(w.get("penalty_near_opposite_sr", 0.20))

    htf_fb = (features or {}).get("htf_fallback", "")
    if htf_fb in ("D", "4h"):
        fb_map = thresholds.get("htf_fallback_discount", {"D": 0.7, "4h": 0.5})
        breakdown["htf_fallback_discount"] = float(fb_map.get(htf_fb, 0.7)) - 1.0

    raw_score = sum(breakdown.values())
    if htf_fb in ("D", "4h"):
        fb_map = thresholds.get("htf_fallback_discount", {"D": 0.7, "4h": 0.5})
        raw_score *= float(fb_map.get(htf_fb, 0.7))
    score = clamp01(raw_score)
    th = thresholds.get("strength_thresholds", {"weak": 0.25, "fair": 0.50, "strong": 0.75})
    if score < th["weak"]:
        strength = "netral"
    elif score < th["fair"]:
        strength = "lemah"
    elif score < th["strong"]:
        strength = "cukup"
    else:
        strength = "kuat"

    ok = (score >= float(thresholds.get("score_gate", 0.5))) and (strength in ("cukup", "kuat"))
    return {
        "ok": bool(ok),
        "side": side if ok else None,
        "score": float(score),
        "strength": strength,
        "reasons": reasons,
        "breakdown": breakdown,
        "context": {"trend": trend, "regime": regime, "tol_sr_pct": tol_sr},
    }
