import os, json, logging
from typing import Dict, Any, Tuple, Optional, Union, Any, List, cast

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from indicators.srmtf.sr_service import SRMTFService
from aggregators.sr_features import sr_features_from_signals, sr_reason_weights_default
from signal_engine.aggregator import aggregate as agg_signal, build_features_from_modules
from signal_engine.types import Side

# --- helpers (top-level util) ---
_SR_SERVICE: Optional[SRMTFService] = None
Numeric = Union[float, int, np.floating, np.integer]
ArrayLike = Union[pd.Series, np.ndarray, list, tuple]
ScalarOrArray = Union[Numeric, ArrayLike]

def as_scalar(x: Any) -> float:
    """Ambil angka scalar dari berbagai tipe.
    - pd.Series -> nilai terakhir
    - np.ndarray/list/tuple -> elemen terakhir
    - float/int -> cast float
    """
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return 0.0
        return float(x.iloc[-1])
    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0:
            return 0.0
        return float(x[-1])
    try:
        return float(x)
    except Exception:
        return 0.0

SAFE_EPS = float(os.getenv("SAFE_EPS", "1e-9"))


def to_scalar(x: Optional[Any], *, how: str = "last", default: float = 0.0) -> float:
    return as_scalar(x) if x is not None else default


def to_bool(x: Union[bool, ArrayLike]) -> bool:
    if isinstance(x, (pd.Series, np.ndarray)):
        if len(x) == 0:
            return False
        return bool(np.asarray(x).reshape(-1)[-1])
    return bool(x)


def safe_div(a: Any, b: Any, default: float = 0.0) -> float:
    aa, bb = as_scalar(a), as_scalar(b)
    if bb == 0.0 or np.isnan(bb):
        return default
    return aa / bb


def floor_to_step(x: Any, step: Any) -> float:
    xf, sf = as_scalar(x), max(as_scalar(step), 1e-18)
    return float(np.floor(xf / sf) * sf)


def ceil_to_step(x: Any, step: Any) -> float:
    xf, sf = as_scalar(x), max(as_scalar(step), 1e-18)
    return float(np.ceil(xf / sf) * sf)


def clamp_scalar(x: ScalarOrArray, lo: ScalarOrArray, hi: ScalarOrArray) -> float:
    xv = to_scalar(x)
    lov = to_scalar(lo)
    hiv = to_scalar(hi)
    return float(min(max(xv, lov), hiv))


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default

# -----------------------------------------------------
# Konversi tipe sederhana
# -----------------------------------------------------
def _to_float(v: Any, d: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)

def _to_int(v: Any, d: Any) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)

def _to_bool(v: Any, d: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1","true","y","yes","on"}: return True
        if s in {"0","false","n","no","off"}: return False
    return bool(d)

# -----------------------------------------------------
# Precision helpers
# -----------------------------------------------------
def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(round(x / step) * step)


def round_to_tick(x: Any, tick: Any) -> float:
    xf, tf = as_scalar(x), max(as_scalar(tick), 1e-18)
    return float(np.round(xf / tf) * tf)

def enforce_precision(sym_cfg: Dict[str, Any], price: float, qty: float) -> Tuple[float, float]:
    p_step = _to_float(sym_cfg.get("tickSize", 0), 0)
    q_step = _to_float(sym_cfg.get("stepSize", 0), 0)
    price = round_to_step(price, p_step) if p_step > 0 else price
    qty = floor_to_step(qty, q_step) if q_step > 0 else qty
    return price, qty

def meets_min_notional(sym_cfg: Dict[str, Any], price: float, qty: float) -> bool:
    min_not = _to_float(sym_cfg.get("minNotional", 0), 0)
    return (price * qty) >= min_not if min_not > 0 else True

# -----------------------------------------------------
# Config loader & merger
# -----------------------------------------------------
_LAST_COIN_CONFIG: Dict[str, Any] = {}

def load_coin_config(path: str) -> Dict[str, Any]:
    global _LAST_COIN_CONFIG
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
            _LAST_COIN_CONFIG = cfg if isinstance(cfg, dict) else {}
            return cfg
    except Exception:
        _LAST_COIN_CONFIG = {}
        return {}

def merge_config(symbol: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sym_cfg = base_cfg.get(symbol, {}) if isinstance(symbol, str) else {}
    return dict(sym_cfg)

# -----------------------------------------------------
# Indikator & sinyal sederhana
# -----------------------------------------------------
def compute_indicators(df: pd.DataFrame, heikin: bool = False) -> pd.DataFrame:
    d = df.copy()

    # (opsional) Heikin Ashi
    if heikin:
        ha = d.copy()
        ha['ha_close'] = (pd.to_numeric(ha['open'], errors='coerce') + 
                          pd.to_numeric(ha['high'], errors='coerce') + 
                          pd.to_numeric(ha['low'], errors='coerce') + 
                          pd.to_numeric(ha['close'], errors='coerce')) / 4
        ha['ha_open'] = pd.to_numeric(ha['open'], errors='coerce').shift(1)
        ha.loc[ha.index[0], 'ha_open'] = (pd.to_numeric(ha.loc[ha.index[0], 'open'], errors='coerce') + 
                                          pd.to_numeric(ha.loc[ha.index[0], 'close'], errors='coerce')) / 2
        ha['ha_high']  = ha[['high','ha_open','ha_close']].max(axis=1)
        ha['ha_low']   = ha[['low','ha_open','ha_close']].min(axis=1)
        d[['open','high','low','close']] = ha[['ha_open','ha_high','ha_low','ha_close']]

    # EMA/MA, MACD, RSI
    # Sesuaikan periode dengan penamaan kolom agar tidak membingungkan downstream:
    # - ema_20 = EMA 20
    # - ma_22  = SMA 22
    d['ema_20'] = EMAIndicator(d['close'], 20).ema_indicator()
    d['ma_22'] = SMAIndicator(d['close'], 22).sma_indicator()
    macd = MACD(d['close'])
    d['macd'] = macd.macd()
    d['macd_signal'] = macd.macd_signal()
    d['rsi'] = RSIIndicator(d['close'], 25).rsi()

    # ATR (EMA Wilder-ish) & normalisasi
    prev_close = d['close'].shift(1)
    tr = pd.concat([(d['high']-d['low']).abs(), (d['high']-prev_close).abs(), (d['low']-prev_close).abs()], axis=1).max(axis=1)
    d['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().fillna(0.0)
    d['atr_pct'] = (d['atr'] / d['close']).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # body/ATR + alias
    d['body'] = (d['close'] - d['open']).abs()
    d['body_to_atr'] = (d['body'] / d['atr']).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    d['body_atr']    = d['body_to_atr']  # alias untuk backward compat
    return d


# Legacy base-signal logic dihapus. Sumber kebenaran sinyal sekarang dari aggregator.

# ===================== AGGREGATOR INTEGRATION =====================
# Defaults are conservative but aligned with requested baseline
DEFAULT_AGG: Dict[str, Any] = {
    "signal_weights": {},
    "strength_thresholds": {"weak": 0.28, "fair": 0.55, "strong": 0.78},
    "score_gate": 0.56,
    "min_strength": "cukup",
    "score_gate_no_confirms": 0.66,
    "min_strength_no_confirms": "kuat",
    "no_confirms_require": ["adx", "width_atr", "body_atr"],
    "confirm_bonus_per": 0.0,
    "confirm_bonus_max": 0.0,
    "vol_lookback": 20,
    "vol_z_thr": 1.7,
    "sd_tol_pct": 2.0,
    "htf_rules": ["4h", "1d"],
}


def _merge_agg_thresholds(partial: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(DEFAULT_AGG)
    if isinstance(partial, dict):
        d.update(partial)
    return d


def _read_agg_from_cfg(coin_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ambil blok aggregator dari coin_cfg['_agg'] bila ada.
    Terima partial config; isi default otomatis.
    """
    agg = (coin_cfg or {}).get("_agg")
    if not isinstance(agg, dict) or not agg:
        return None
    return cast(Dict[str, Any], _merge_agg_thresholds(agg))


# -----------------------------------------------------
# Ensure base indicators exist per coin_cfg
# -----------------------------------------------------
def ensure_base_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ema_len = int(cfg.get("ema_len", 22))
    sma_len = int(cfg.get("sma_len", 20))
    if f"ema_{ema_len}" not in df.columns:
        df[f"ema_{ema_len}"] = df["close"].ewm(span=ema_len, adjust=False).mean()
    if f"sma_{sma_len}" not in df.columns:
        df[f"sma_{sma_len}"] = df["close"].rolling(sma_len).mean()
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        macd_fast, macd_slow, macd_sig = 12, 26, 9
        macd = df["close"].ewm(span=macd_fast, adjust=False).mean() - df["close"].ewm(span=macd_slow, adjust=False).mean()
        df["macd"] = macd
        df["macd_signal"] = macd.ewm(span=macd_sig, adjust=False).mean()
    if "rsi" not in df.columns:
        rsi_len = int(cfg.get("rsi_period", 14))
        delta = df["close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/rsi_len, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
    return df


# === Grading kekuatan sinyal (BASE & ML) + fusi & pemilih mode ===

def base_strength(ind: dict, side_hint: str | None) -> float:
    """
    Skor 0..1 berdasarkan komponen base. Bobot:
    +0.40 trend EMA vs MA searah side
    +0.30 MACD > Signal (long) / < (short)
    +0.20 RSI di zona sehat (long: 40–70, short: 30–60)
    +0.10 ATR% & body/ATR di koridor sehat (sudah difilter)
    """
    if not ind:
        return 0.0
    s = 0.0
    up = (side_hint == "LONG")
    ema = as_float(ind.get("ema"), 0.0)
    ma = as_float(ind.get("ma"), 0.0)
    macd = as_float(ind.get("macd"), 0.0)
    macd_sig = as_float(ind.get("macd_signal"), 0.0)
    rsi = as_float(ind.get("rsi"), 50.0)
    atr_pct = as_float(ind.get("atr_pct"), 0.0)
    body_atr = as_float(ind.get("body_to_atr"), 0.0)

    if up and ema > ma:
        s += 0.40
    if not up and ema < ma:
        s += 0.40
    if up and macd > macd_sig:
        s += 0.30
    if not up and macd < macd_sig:
        s += 0.30
    if up and 40.0 <= rsi <= 70.0:
        s += 0.20
    if not up and 30.0 <= rsi <= 60.0:
        s += 0.20
    if 0.0005 <= atr_pct <= 0.03 and body_atr <= 2.0:
        s += 0.10

    return max(0.0, min(1.0, s))


def ml_strength(up_prob: float | None, side_hint: str | None) -> float:
    """
    Skor 0..1 dari probabilitas ML.
    Ambang:
      LONG: 0.50–0.55 lemah, 0.55–0.60 netral, 0.60–0.65 cukup, >0.65 kuat
      SHORT: mirror kebawah (0.50..0.35..)
    """
    if up_prob is None or side_hint is None:
        return 0.0
    if side_hint == "LONG":
        p = up_prob
        if p < 0.50:
            return 0.0
        if p < 0.55:
            return 0.25
        if p < 0.60:
            return 0.50
        if p < 0.65:
            return 0.75
        return 1.0
    else:
        p = up_prob
        if p > 0.50:
            return 0.0
        if p > 0.45:
            return 0.25
        if p > 0.40:
            return 0.50
        if p > 0.35:
            return 0.75
        return 1.0


def fuse_strength(base_s: float, ml_s: float, w_base: float = 0.6, w_ml: float = 0.4) -> float:
    return max(0.0, min(1.0, w_base * as_float(base_s, 0.0) + w_ml * as_float(ml_s, 0.0)))


def pick_mode(combined: float, base_s: float, ml_s: float) -> str:
    """
    Kembalikan salah satu:
      - "SKIP"     : combined < 0.45
      - "TP10"     : 0.45–<0.60 (TP ROI 10%, BE aktif, trailing OFF)
      - "BE_TRAIL" : 0.60–<0.80 (BE+Trailing)
      - "SWING"    : >=0.80 DAN (base kuat & ml kuat)
    """
    if combined < 0.45:
        return "SKIP"
    if combined < 0.60:
        return "TP10"
    if combined < 0.80:
        return "BE_TRAIL"
    if base_s >= 0.75 and ml_s >= 0.75:
        return "SWING"
    return "BE_TRAIL"


ML_WEIGHT = float(os.getenv("ML_WEIGHT", "1.2"))


def get_coin_ml_params(symbol: str, coin_config: dict) -> dict:
    d = coin_config.get(symbol, {}).get("ml", {})
    return {
        "enabled": bool(d.get("enabled", True)),
        "strict": bool(d.get("strict", False)),
        "up_prob": float(d.get("up_prob_long", 0.55)),
        "down_prob": float(d.get("down_prob_short", 0.45)),
        "score_threshold": float(d.get("score_threshold", 1.0)),
        "weight": float(d.get("weight", ML_WEIGHT)),
    }


def _default_signal_params() -> Dict[str, Any]:
    return {
        "weights": {
            "sc_trend_htf": 0.35,
            "sc_no_htf": 0.15,
            "adx": 0.15,
            "body_atr": 0.10,
            "width_atr": 0.10,
            "rsi": 0.05,
            "sr_breakout": 0.20,
            "sr_test": 0.10,
            "sr_reject": 0.15,
            "sd_proximity": 0.20,
            "vol_confirm": 0.10,
            "fvg_confirm": 0.10,
        },
        "thresholds": {
            "vol_lookback": 20,
            "vol_z_thr": 2.0,
            "sd_tol_pct": 1.0,
            "strength_thresholds": {"weak": 0.25, "fair": 0.50, "strong": 0.75},
            "score_gate": 0.50,
            "htf_fallback_discount": {"D": 0.7, "4h": 0.5},
            "min_confirms": 2,
        },
        "regime_bounds": {"atr_p1": 0.02, "atr_p2": 0.05, "bbw_q1": 0.01, "bbw_q2": 0.05},
        "sr_penalty": {"base_pct": 0.6, "k_atr": 0.5},
    }

def _merge_signal_params(base: Dict[str, Any], coin_cfg: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, float], Dict[str, float]]:
    # coin_config keys mapping per spesifikasi
    scfg = coin_cfg.get("signal_engine", {})
    weights = dict(base["weights"]) | dict(scfg.get("signal_weights", coin_cfg.get("signal_weights", {})))
    thresholds = dict(base["thresholds"]) | dict({
        "strength_thresholds": scfg.get("strength_thresholds", coin_cfg.get("strength_thresholds", base["thresholds"]["strength_thresholds"])),
        "score_gate": scfg.get("score_gate", coin_cfg.get("score_gate", base["thresholds"]["score_gate"])),
        "sd_tol_pct": scfg.get("sd_tol_pct", coin_cfg.get("sd_tol_pct", base["thresholds"].get("sd_tol_pct", 1.0))),
        "vol_lookback": scfg.get("vol_lookback", coin_cfg.get("vol_lookback", base["thresholds"]["vol_lookback"])),
        "vol_z_thr": scfg.get("vol_z_thr", coin_cfg.get("vol_z_thr", base["thresholds"]["vol_z_thr"])),
        "htf_rules": scfg.get("htf_rules", coin_cfg.get("htf_rules", ("1h","4h"))),
        "htf_fallback_discount": scfg.get("htf_fallback_discount", coin_cfg.get("htf_fallback_discount", base["thresholds"]["htf_fallback_discount"])),
        "weight_scale": scfg.get("weight_scale", coin_cfg.get("weight_scale", {})),
        "weight_scale_nl": scfg.get("weight_scale_nl", coin_cfg.get("weight_scale_nl", {})),
        "min_confirms": scfg.get("min_confirms", coin_cfg.get("min_confirms", base["thresholds"]["min_confirms"]))
    })
    regime_bounds = dict(base["regime_bounds"]) | dict(scfg.get("regime_bounds", coin_cfg.get("regime_bounds", {})))
    sr_penalty = dict(base["sr_penalty"]) | dict(scfg.get("sr_penalty", coin_cfg.get("sr_penalty", {})))
    return weights, thresholds, regime_bounds, sr_penalty

_WARNED_EMPTY_AGG: set[str] = set()

def make_decision(df: pd.DataFrame, symbol: str, coin_cfg: dict, ml_up_prob: float | None) -> Tuple[Optional[str], List[dict]]:
    # Pastikan indikator dasar tersedia sesuai parameter
    df = ensure_base_indicators(df.copy(), coin_cfg or {})

    # 1) Jalur AGGREGATOR (utama jika preset tersedia)
    agg_cfg = _read_agg_from_cfg(coin_cfg)
    if agg_cfg is None:
        # Guard: fallback to SYMBOL_DEFAULTS._agg if available
        base_def: Dict[str, Any] = {}
        try:
            base_def = (_LAST_COIN_CONFIG or {}).get("SYMBOL_DEFAULTS", {}) or {}
            agg_cfg = _read_agg_from_cfg(base_def)
        except Exception:
            agg_cfg = None
        # If still missing, use conservative DEFAULT_AGG
        if agg_cfg is None:
            agg_cfg = dict(DEFAULT_AGG)
        # Warn once for auditability
        if symbol not in _WARNED_EMPTY_AGG:
            _WARNED_EMPTY_AGG.add(symbol)
            src = "SYMBOL_DEFAULTS" if isinstance(base_def, dict) and isinstance(base_def.get("_agg"), dict) else "DEFAULT_AGG"
            try:
                sym_txt = str(symbol)
            except Exception:
                sym_txt = "?"
            print(f"[WARN] {sym_txt}: _agg kosong; fallback ke {src}")
    if agg_cfg is not None:
        try:
            w = dict(agg_cfg.get("signal_weights", {}))
            th = dict(agg_cfg.get("strength_thresholds", {}))
            thresholds = {
                "strength_thresholds": th or {"weak": 0.25, "fair": 0.50, "strong": 0.75},
                "weight_scale": agg_cfg.get("weight_scale", {}),
                "vol_lookback": agg_cfg.get("vol_lookback", 20),
                "vol_z_thr": agg_cfg.get("vol_z_thr", 2.0),
                "sd_tol_pct": agg_cfg.get("sd_tol_pct", 1.0),
                "score_gate": agg_cfg.get("score_gate", 0.5),
                # allow optional knobs
                "htf_fallback_discount": agg_cfg.get("htf_fallback_discount", {"D": 0.7, "4h": 0.5}),
                "weight_scale_nl": agg_cfg.get("weight_scale_nl", {}),
                "min_confirms": agg_cfg.get("min_confirms", None),
                # adaptive gate & bonus confirms (new)
                "score_gate_no_confirms": agg_cfg.get("score_gate_no_confirms", None),
                "min_strength": agg_cfg.get("min_strength", "cukup"),
                "min_strength_no_confirms": agg_cfg.get("min_strength_no_confirms", agg_cfg.get("min_strength", "cukup")),
                "no_confirms_require": agg_cfg.get("no_confirms_require", []),
                "confirm_bonus_per": agg_cfg.get("confirm_bonus_per", 0.0),
                "confirm_bonus_max": agg_cfg.get("confirm_bonus_max", 0.0),
            }
            regime_bounds = dict(agg_cfg.get("regime_bounds", {"atr_p1": 0.01, "atr_p2": 0.05, "bbw_q1": 0.01, "bbw_q2": 0.05}))
            sr_penalty = dict(agg_cfg.get("sr_penalty", {"base_pct": 0.6, "k_atr": 0.5}))
            htf_rules = tuple(agg_cfg.get("htf_rules", ("1h", "4h")))

            f_long = build_features_from_modules(df, "LONG")
            f_short = build_features_from_modules(df, "SHORT")
            rL = agg_signal(df, cast(Side, "LONG"), w, thresholds, regime_bounds, sr_penalty, htf_rules=htf_rules, features=f_long)
            rS = agg_signal(df, cast(Side, "SHORT"), w, thresholds, regime_bounds, sr_penalty, htf_rules=htf_rules, features=f_short)
            import os as _os
            if _os.getenv("DEBUG_AGG") == "1":
                try:
                    print(f"[AGG] rL ok={rL.get('ok')} score={rL.get('score'):.3f} strength={rL.get('strength')} reasons={rL.get('reasons')}")
                    print(f"[AGG] rS ok={rS.get('ok')} score={rS.get('score'):.3f} strength={rS.get('strength')} reasons={rS.get('reasons')}")
                except Exception:
                    print(f"[AGG] rL={rL}")
                    print(f"[AGG] rS={rS}")
            candidates = [r for r in (rL, rS) if r.get("ok")]
            if candidates:
                best = max(candidates, key=lambda r: r.get("score", 0.0))
                side = cast(Optional[str], best.get("side"))
                logging.getLogger(__name__).info(
                    f"[{symbol}] AGG OK side={side} score={best['score']:.3f} strength={best['strength']} reasons={best.get('reasons')}"
                )
                return side, [{"source": "aggregator", "score": float(best.get("score", 0.0)), "strength": best.get("strength")}]
            else:
                logging.getLogger(__name__).info(
                    f"[{symbol}] AGG NO-ENTRY rL={rL['score']:.3f}/{rL['strength']} rS={rS['score']:.3f}/{rS['strength']}"
                )
        except Exception as e:
            logging.getLogger(__name__).warning(f"[{symbol}] aggregator path error: {e}. Fallback to base rule.")

    # 2) Fallback BASE-RULE (sederhana)
    ema_len = int(coin_cfg.get("ema_len", 22))
    sma_len = int(coin_cfg.get("sma_len", 20))
    rsi_min_long = float(coin_cfg.get("rsi_long_min", 10))
    rsi_max_long = float(coin_cfg.get("rsi_long_max", 55))
    rsi_min_short = float(coin_cfg.get("rsi_short_min", 70))
    rsi_max_short = float(coin_cfg.get("rsi_short_max", 90))

    ema_col, sma_col = f"ema_{ema_len}", f"sma_{sma_len}"
    if len(df) < 2:
        return None, []
    ema_now, ema_prev = df[ema_col].iloc[-1], df[ema_col].iloc[-2]
    sma_now, sma_prev = df[sma_col].iloc[-1], df[sma_col].iloc[-2]
    macd_now, macd_sig = df["macd"].iloc[-1], df["macd_signal"].iloc[-1]
    rsi_now = df["rsi"].iloc[-1]

    long_base = (ema_prev <= sma_prev) and (ema_now > sma_now) and (macd_now > macd_sig) and (rsi_min_long <= rsi_now <= rsi_max_long)
    short_base = (ema_prev >= sma_prev) and (ema_now < sma_now) and (macd_now < macd_sig) and (rsi_min_short <= rsi_now <= rsi_max_short)

    decision: Optional[str] = None
    if long_base and not short_base:
        decision = "LONG"
    elif short_base and not long_base:
        decision = "SHORT"

    # ML gate: gunakan threshold dari coin_cfg['ml']['score_threshold'] bila ada
    ml_cfg = (coin_cfg or {}).get("ml", {}) or {}
    ml_thr = float(ml_cfg.get("score_threshold", 1.0))
    ml_enabled = bool(ml_cfg.get("enabled", True))
    if decision is not None and ml_enabled:
        ok = apply_ml_gate(ml_up_prob, ml_thr)
        if not ok:
            decision = None

    reasons: List[dict] = []
    if decision is None:
        logging.getLogger(__name__).info(
            f"[{symbol}] NO-ENTRY base(L={long_base},S={short_base}) macd={macd_now:.4f}/{macd_sig:.4f} "
            f"rsi={rsi_now:.2f} ema={ema_prev:.4f}->{ema_now:.4f} sma={sma_prev:.4f}->{sma_now:.4f}"
        )
    return decision, reasons

def htf_trend_ok(side: str, base_df: pd.DataFrame, htf: str = '1h') -> bool:
    try:
        tmp = base_df.set_index('timestamp')[['close']].copy()
        res = str(htf).upper()
        htf_close = tmp['close'].resample(res).last().dropna()
        # Butuh ~200 bar untuk EMA200 yang stabil
        if len(htf_close) < 210:
            return True
        # Konsisten dengan istilah: gunakan EMA 50 vs EMA 200 untuk HTF trend
        ema50 = htf_close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = htf_close.ewm(span=200, adjust=False).mean().iloc[-1]
        return (ema50 >= ema200) if side == 'LONG' else (ema50 <= ema200)
    except Exception:
        return True
def apply_filters(ind: pd.Series, coin_cfg: Dict[str, Any]) -> Tuple[bool, bool, Dict[str, Any]]:
    flt = coin_cfg.get("filters", {}) or {}
    use_atr = bool(flt.get("atr", True))
    use_body = bool(flt.get("body", True))

    min_atr = _to_float(coin_cfg.get('min_atr_pct', 0.0), 0.0)
    max_atr = _to_float(coin_cfg.get('max_atr_pct', 1.0), 1.0)
    max_body = _to_float(coin_cfg.get('max_body_atr', 999.0), 999.0)

    atr_val = float(ind.get('atr_pct', 0.0))
    body_val = float(ind.get('body_to_atr', ind.get('body_atr', np.nan)))

    atr_ok = (atr_val >= min_atr) and (atr_val <= max_atr) if use_atr else True
    body_ok = (body_val <= max_body) if (use_body and not np.isnan(body_val)) else True

    if not atr_ok or not body_ok:
        logging.getLogger(__name__).info(
            f"[{coin_cfg.get('symbol','?')}] FILTER INFO atr_ok={atr_ok} body_ok={body_ok} "
            f"price={ind.get('close')} pos={coin_cfg.get('tf','?')}"
        )
    return atr_ok, body_ok, {'atr_pct': atr_val, 'body_to_atr': body_val}

def decide_base(ind: pd.Series, coin_cfg: Dict[str, Any]) -> Dict[str, bool]:
    return {'L': bool(ind.get('long_base', False)), 'S': bool(ind.get('short_base', False))}

def confirm_htf(htf_ind: pd.DataFrame, coin_cfg: Dict[str, Any]) -> bool:
    # placeholder konfirmasi HTF: true jika ema20>= ema22
    try:
        ema50 = htf_ind['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema200 = htf_ind['close'].ewm(span=22, adjust=False).mean().iloc[-1]
        side = coin_cfg.get('side','LONG')
        return ema50 >= ema200 if side=='LONG' else ema50 <= ema200
    except Exception:
        return True

def apply_ml_gate(up_prob: Optional[float], ml_thr: float, eps: float = SAFE_EPS) -> bool:
    if up_prob is None:
        return True
    p = min(max(up_prob, eps), 1 - eps)
    up_odds = safe_div(p, 1 - p)
    return up_odds >= ml_thr

# -----------------------------------------------------
# Money management & PnL
# -----------------------------------------------------
def risk_size(balance: float, risk_pct: float, entry: float, stop: float,
              fee_bps: float, slip_bps: float, sym_cfg: Dict[str, Any]) -> float:
    lev = _to_int(sym_cfg.get('leverage', 1), 1)
    risk_val = balance * risk_pct
    diff = abs(entry - stop)
    if diff <= 0:
        return 0.0
    qty = safe_div((risk_val * lev), diff)
    step = _to_float(sym_cfg.get('stepSize', 0), 0)
    return floor_to_step(qty, step) if step>0 else to_scalar(qty)

def pnl_net(side: str, entry: float, exit: float, qty: float,
            fee_bps: float, slip_bps: float) -> Tuple[float, float]:
    fee = (entry + exit) * qty * fee_bps/10000.0
    slip = (entry + exit) * qty * slip_bps/10000.0
    gross = (exit - entry) * qty if side=='LONG' else (entry - exit) * qty
    pnl = gross - fee - slip
    roi = safe_div(pnl, (entry*qty)) if entry*qty>0 else 0.0
    return pnl, roi*100.0

def r_multiple(entry: float, sl: float, price: float) -> float:
    r = abs(entry - sl)
    return safe_div(abs(price - entry), r)

def r_multiple_signed(entry: float, sl: float, price: float, side: str) -> float:
    """R multiple bertanda; positif hanya jika bergerak sesuai arah profit."""
    R = abs(entry - sl)
    if R <= 0:
        return 0.0
    move = (price - entry) if side == 'LONG' else (entry - price)
    return safe_div(move, R)

def apply_breakeven_sl(side: str,
                       entry: float,
                       price: float,
                       sl: float | None,
                       tick_size: float = 0.0,
                       min_gap_pct: float = 0.0001,
                       be_trigger_r: float = 0.0,
                       be_trigger_pct: float = 0.0) -> float | None:
    """
    Hitung SL BE baru bila trigger terpenuhi. Mengembalikan SL baru atau sl lama.
    - BE R-multiple: aktif hanya jika posisi sudah profit sesuai arah.
    - BE %: fallback jika R=0 atau be_trigger_r=0.
    - SL tidak ditempatkan menempel pada harga: beri gap min(max(tick, price*min_gap_pct)).
    """
    if entry is None or side not in ('LONG', 'SHORT'):
        return sl
    gap = max(tick_size or 0.0, abs(price) * (min_gap_pct or 0.0))

    # 1) R-based (prioritas jika > 0)
    if be_trigger_r and be_trigger_r > 0:
        rnow = r_multiple_signed(entry, sl if sl is not None else entry, price, side)
        if side == 'LONG':
            if price > entry and rnow >= be_trigger_r:
                target = max(sl or 0.0, entry)
                return min(target, price - gap)
        else:  # SHORT
            if price < entry and rnow >= be_trigger_r:
                target = min(sl or 1e18, entry)
                return max(target, price + gap)

    # 2) % fallback
    if be_trigger_pct and be_trigger_pct > 0:
        if side == 'LONG':
            if safe_div((price - entry), entry) >= be_trigger_pct:
                target = max(sl or 0.0, entry)
                return min(target, price - gap)
        else:
            if safe_div((entry - price), entry) >= be_trigger_pct:
                target = min(sl or 1e18, entry)
                return max(target, price + gap)

    return sl

def roi_frac_now(side: str, entry: Optional[float], price: float, qty: float, leverage: int) -> float:
    if entry is None or not qty or leverage <= 0:
        return 0.0
    entry = float(entry)
    price = float(price)
    init_margin = safe_div((entry * abs(qty)), leverage)
    if init_margin <= 0:
        return 0.0
    if side == 'LONG':
        pnl = (price - entry) * qty
    else:
        pnl = (entry - price) * qty
    return safe_div(pnl, init_margin)

def base_supports_side(base_long: bool, base_short: bool, side: str) -> bool:
    return (side == 'LONG' and base_long and not base_short) or (side == 'SHORT' and base_short and not base_long)

# -----------------------------------------------------
# Journaling
# -----------------------------------------------------
def journal_row(ts: str, symbol: str, side: str, entry: float, qty: float,
                sl_init: Optional[float], tsl_init: Optional[float],
                exit_price: float, exit_reason: str,
                pnl_usdt: float, roi_pct: float, balance_after: float) -> Dict[str, Any]:
    return {
        'timestamp': ts,
        'symbol': symbol,
        'side': side,
        'entry': f"{entry:.6f}",
        'qty': f"{qty:.6f}",
        'sl_init': "" if sl_init is None else f"{sl_init:.6f}",
        'tsl_init': "" if tsl_init is None else f"{tsl_init:.6f}",
        'exit_price': f"{exit_price:.6f}",
        'exit_reason': exit_reason,
        'pnl_usdt': f"{pnl_usdt:.4f}",
        'roi_pct': f"{roi_pct:.2f}",
        'balance_after': f"{balance_after:.4f}",
    }

def init_stops(side: str, entry: float, ind: pd.Series, coin_cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    atr = float(ind.get('atr', 0.0))
    sl = entry * (1 - _to_float(coin_cfg.get('sl_pct', 0.01), 0.01)) if side=='LONG' else entry * (1 + _to_float(coin_cfg.get('sl_pct', 0.01), 0.01))
    return {'sl': sl, 'tsl': None}

def step_trailing(side: str, bar: pd.Series, prev_state: Dict[str, Optional[float]], ind: pd.Series, coin_cfg: Dict[str, Any]) -> Optional[float]:
    trigger = _to_float(coin_cfg.get('trailing_trigger', 0.7), 0.7)
    step = _to_float(coin_cfg.get('trailing_step', 0.45), 0.45)
    entry = prev_state.get('entry', 0.0)
    price = float(bar['close'])
    if entry is None or price is None or entry == 0:
        profit_pct = 0.0
    else:
        profit_pct = safe_div((price - entry), entry) * 100 if side == 'LONG' else safe_div((entry - price), entry) * 100
    if profit_pct < trigger:
        return prev_state.get('tsl')
    if side=='LONG':
        new_tsl = price*(1-step/100)
        return max(prev_state.get('tsl') or prev_state.get('sl') or 0.0, new_tsl)
    else:
        new_tsl = price*(1+step/100)
        return min(prev_state.get('tsl') or prev_state.get('sl') or 1e18, new_tsl)

def maybe_move_to_BE(side: str, entry: float, tsl: Optional[float], rule: Dict[str, Any]) -> Optional[float]:
    be = _to_float(rule.get('be_trigger_pct', 0.0), 0.0)
    price = rule.get('price', entry)
    if be <= 0:
        return tsl
    if side=='LONG' and safe_div((price-entry), entry) >= be:
        return max(tsl or 0.0, entry)
    if side=='SHORT' and safe_div((entry-price), entry) >= be:
        return min(tsl or 1e18, entry)
    return tsl

def check_time_stop(entry_time: float, now: float, roi: float, rule: Dict[str, Any]) -> bool:
    max_secs = _to_int(rule.get('time_stop_secs', 0), 0)
    return (max_secs>0) and ((now-entry_time) >= max_secs)

def cooldown_until(now: float, rule: Dict[str, Any]) -> float:
    secs = _to_int(rule.get('cooldown_secs', 0), 0)
    return now + max(0, secs)

def simulate_fill_on_candle(side: str, state: Dict[str, float], bar: pd.Series, sym_cfg: Dict[str, Any], fee_bps: float, slip_bps: float) -> Optional[Tuple[float, str]]:
    o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
    sl = state.get('sl'); tsl = state.get('tsl')
    if side=='LONG':
        if tsl is not None and l <= tsl:
            return tsl, 'Hit Trailing SL'
        if sl is not None and l <= sl:
            return sl, 'Hit Hard SL'
    else:
        if tsl is not None and h >= tsl:
            return tsl, 'Hit Trailing SL'
        if sl is not None and h >= sl:
            return sl, 'Hit Hard SL'
    return None
