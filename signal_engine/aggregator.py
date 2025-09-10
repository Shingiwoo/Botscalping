from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from .types import AggResult, ScoreBreakdown, Side
from .regime import compute_vol_metrics, classify_regime, scale_weights, scale_weights_nonlinear
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
    URSI_COMPUTE_DF,
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
    """
    Fallback yang benar-benar menghitung metrik dasar jika SC_API tidak tersedia.
    - ADX via ta.trend.ADXIndicator bila ada (fallback sederhana bila tidak),
    - ATR & body/ATR & ATR% window terakhir,
    - RSI 14 vs band dari cfg (rsi_long_min/max, rsi_short_min/max).
    """
    out = {"cross": True, "adx_ok": False, "body_atr_ok": False, "width_atr_ok": False, "rsi_ok": False}
    # 1) Jika ada SC_API, gunakan—tetapi JANGAN auto-true semua flag
    if SC_API:
        try:
            _ = SC_API(df, cfg)
            # Anggap SC_API sudah melakukan validasi internal;
            # namun tetap siapkan fallback lokal agar distribusi skor tidak datar
        except Exception:
            pass
    # 2) Hitung fallback lokal
    close = float(df["close"].iloc[-1])
    open_ = float(df["open"].iloc[-1])
    # high/low digunakan untuk ATR & width
    body = abs(close - open_)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_win = int(cfg.get("adx_period", 14))
    atr_series = tr.rolling(atr_win, min_periods=atr_win).mean()
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else float(tr.tail(atr_win).mean())
    atr_pct = atr_val / max(1e-9, close)

    # Params dari cfg/preset (fallback ke default wajar)
    min_atr_pct = float(cfg.get("min_atr_pct", 0.003))
    max_atr_pct = float(cfg.get("max_atr_pct", 0.03))
    max_body_atr = float(cfg.get("max_body_atr", 1.6))
    rsi_p = int(cfg.get("rsi_period", 14))
    rsi_lmin = float(cfg.get("rsi_long_min", 10))
    rsi_lmax = float(cfg.get("rsi_long_max", 65))
    rsi_smin = float(cfg.get("rsi_short_min", 70))
    rsi_smax = float(cfg.get("rsi_short_max", 90))

    # ADX
    adx_val = 0.0
    try:
        from ta.trend import ADXIndicator
        adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=atr_win)
        adx_val = float(adx_ind.adx().iloc[-1])
    except Exception:
        # Fallback adx-like: trend strength relatif terhadap ATR
        ema_fast = df["close"].ewm(span=max(5, atr_win//2), adjust=False).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=max(10, atr_win), adjust=False).mean().iloc[-1]
        adx_val = min(50.0, abs(float(ema_fast - ema_slow)) / max(1e-9, atr_val) * 25.0)
    adx_thr = float(cfg.get("adx_thresh", 18.0))
    out["adx_ok"] = bool(adx_val >= adx_thr)

    # Body/ATR & ATR%
    body_over_atr = body / max(1e-9, atr_val)
    out["body_atr_ok"] = bool(body_over_atr <= max_body_atr)
    out["width_atr_ok"] = bool((atr_pct >= min_atr_pct) and (atr_pct <= max_atr_pct))

    # RSI
    try:
        from ta.momentum import RSIIndicator
        rsi_now = float(RSIIndicator(df["close"], window=max(3, rsi_p)).rsi().iloc[-1])
    except Exception:
        rsi_now = 50.0
    if side == "LONG":
        out["rsi_ok"] = bool(rsi_lmin <= rsi_now <= rsi_lmax)
    else:
        out["rsi_ok"] = bool(rsi_smin <= rsi_now <= rsi_smax)
    return out


def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1.0 else x


def build_features_from_modules(df: pd.DataFrame, side: Side) -> Dict[str, Any]:
    features: Dict[str, Any] = {"sr": {}, "fvg": {}, "ob": {}, "sd": {}, "ursi": {}, "htf_fallback": ""}
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
        if URSI_COMPUTE_DF:
            try:
                u = URSI_COMPUTE_DF(df)
                features["ursi"] = {
                    "arsi": float(u["URSI"].iloc[-1]),
                    "signal": float(u["URSI_signal"].iloc[-1]),
                    "ob": float(u["URSI_ob"].iloc[-1]),
                    "os": float(u["URSI_os"].iloc[-1]),
                }
            except Exception:
                pass
    except Exception:
        pass
    # ---- Fallback SR ketika modul SMC/* tidak tersedia ----
    try:
        price = float(df["close"].iloc[-1])
        if ("near_support" not in features["sr"]) or ("near_resistance" not in features["sr"]):
            # bangun level swing
            if SR_BUILD_CACHE:
                cache = SR_BUILD_CACHE(df, lb=3, window=300, k=6, recalc_every=10)
                try:
                    res_lvls, sup_lvls = cache.get(len(df)-1, ([], []))
                except Exception:
                    res_lvls, sup_lvls = ([], [])
            else:
                res_lvls, sup_lvls = ([], [])
            # toleransi ATR adaptif berdasar regime
            from .regime import compute_vol_metrics, classify_regime
            vol = compute_vol_metrics(df, lookback=20)
            _ = classify_regime(vol.get("atr_pct", 0.0), vol.get("bb_width", 0.0), {"atr_p1":0.01,"atr_p2":0.05,"bbw_q1":0.01,"bbw_q2":0.05})
            atr_val = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1]) if len(df) >= 14 else float((df["high"] - df["low"]).mean())
            if SR_NEAR and atr_val > 0:
                if "near_support" not in features["sr"]:
                    try:
                        features["sr"]["near_support"] = bool(SR_NEAR(price, sup_lvls, float(atr_val), _))
                    except Exception:
                        pass
                if "near_resistance" not in features["sr"]:
                    try:
                        features["sr"]["near_resistance"] = bool(SR_NEAR(price, res_lvls, float(atr_val), _))
                    except Exception:
                        pass
        # breakout/retest sederhana (swing 20 bar)
        if len(df) >= 2:
            rh = float(df["high"].rolling(20).max().iloc[-2]) if len(df) > 20 else float(df["high"].iloc[:-1].max())
            rl = float(df["low"].rolling(20).min().iloc[-2])  if len(df) > 20 else float(df["low"].iloc[:-1].min())
            if side == "LONG" and price > rh:
                features["sr"]["breakout_same_dir"] = True
            if side == "SHORT" and price < rl:
                features["sr"]["breakout_same_dir"] = True
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
    # Dynamic weights: first apply discrete regime scaling, then optional tanh-based scaling
    w = scale_weights(regime, dict(weights), thresholds.get("weight_scale", {}))
    w = scale_weights_nonlinear(w, vol, thresholds.get("weight_scale_nl", {}))

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
    ursi_feat = (features or {}).get("ursi", {})
    if ursi_feat.get("arsi") is not None and ursi_feat.get("signal") is not None:
        a = float(ursi_feat["arsi"])
        s = float(ursi_feat["signal"])
        if side == "LONG" and a > s:
            breakdown["ursi"] = w.get("ursi", 0.05)
        if side == "SHORT" and a < s:
            breakdown["ursi"] = w.get("ursi", 0.05)

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

    values_array = np.array(list(breakdown.values()), dtype=float)
    raw_score = np.sum(values_array)

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

    # Confirmation consolidation/demotion
    # Count active confirmators across categories: {SMC, SD, Volume, FVG, OB}
    # - SMC: any of sr_breakout/sr_test/sr_reject present
    # - SD: sd_proximity present
    # - Volume: vol_confirm present
    # - FVG: fvg_confirm present (positive or negative still counts as info)
    # - OB: ob active aligns with side
    confirms = 0
    if any(k in breakdown for k in ("sr_breakout", "sr_test", "sr_reject")):
        confirms += 1
    if "sd_proximity" in breakdown:
        confirms += 1
    if "vol_confirm" in breakdown:
        confirms += 1
    if "fvg_confirm" in breakdown:
        confirms += 1
    ob_feat = (features or {}).get("ob", {})
    if side == "LONG" and ob_feat.get("bullish_active"):
        confirms += 1
    if side == "SHORT" and ob_feat.get("bearish_active"):
        confirms += 1
    # Terapkan demotion hanya jika min_confirms ditentukan eksplisit pada thresholds
    _min_c = thresholds.get("min_confirms", None)
    if _min_c is not None:
        try:
            min_confirms = int(_min_c)
        except Exception:
            min_confirms = None
        # Demote bila confirms < min_confirms (termasuk kasus 0 konfirmasi)
        if isinstance(min_confirms, int) and (confirms < min_confirms):
            # Turunkan strength sejumlah kekurangan konfirmasi
            deficit = int(min_confirms - confirms)
            def _rank_of(s: str) -> int:
                return {"netral": 0, "lemah": 1, "cukup": 2, "kuat": 3}.get(s, 0)
            def _label_of(r: int) -> str:
                return {0: "netral", 1: "lemah", 2: "cukup", 3: "kuat"}.get(max(0, min(3, r)), "netral")
            new_rank = max(0, _rank_of(strength) - max(1, deficit))
            strength = _label_of(new_rank)
            # Clamp score ke bawah batas fair (dan bila deficit>1, dorong di bawah weak)
            th_map = thresholds.get("strength_thresholds", {"weak": 0.25, "fair": 0.50, "strong": 0.75})
            fair = float(th_map.get("fair", 0.50))
            weak = float(th_map.get("weak", 0.25))
            if deficit > 1 and score >= weak:
                score = max(0.0, weak - 1e-6)
            elif score >= fair:
                score = max(0.0, fair - 1e-6)

    # Bonus skor kecil per konfirmator (opsional, untuk profil agresif)
    c_bonus = float(thresholds.get("confirm_bonus_per", 0.0))
    c_bonus_max = float(thresholds.get("confirm_bonus_max", 0.0))
    if c_bonus > 0.0 and confirms > 0:
        score = clamp01(score + min(confirms * c_bonus, c_bonus_max))

    # Gate adaptif: lebih ketat bila TANPA konfirmasi
    def _rank(s: str) -> int:
        return {"netral": 0, "lemah": 1, "cukup": 2, "kuat": 3}.get(s, 0)

    base_gate = float(thresholds.get("score_gate", 0.55))
    gate_no_conf = thresholds.get("score_gate_no_confirms", None)
    min_str_base = thresholds.get("min_strength", "cukup")
    min_str_no_conf = thresholds.get("min_strength_no_confirms", min_str_base)
    req_feats = thresholds.get("no_confirms_require", [])

    used_gate = base_gate
    min_str_used = min_str_base
    req_ok = True
    if confirms == 0:
        if gate_no_conf is not None:
            used_gate = float(gate_no_conf)
        min_str_used = min_str_no_conf
        if isinstance(req_feats, (list, tuple)) and len(req_feats) > 0:
            # Wajib: fitur-fitur dasar ini harus muncul di breakdown
            req_ok = all(k in breakdown for k in req_feats)

    ok = (score >= used_gate) and (_rank(strength) >= _rank(min_str_used)) and bool(req_ok)
    return {
        "ok": bool(ok),
        "side": side if ok else None,
        "score": float(score),
        "strength": strength,
        "reasons": reasons,
        "breakdown": breakdown,
        "context": {"trend": trend, "regime": regime, "tol_sr_pct": tol_sr, "confirms": confirms},
    }
