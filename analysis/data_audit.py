from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def _ensure_reports_dir(path: str = "reports") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Cast numeric cols if present
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _count_gaps(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0
    # infer timeframe from first two rows
    dt = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).to_pytimedelta()
    if dt.total_seconds() <= 0:
        return 0
    exp = pd.date_range(start=df["timestamp"].iloc[0], end=df["timestamp"].iloc[-1], freq=pd.Timedelta(seconds=dt.total_seconds()))
    # align to expected schedule
    actual = pd.DatetimeIndex(df["timestamp"])  # type: ignore[arg-type]
    missing = exp.difference(actual)
    return int(len(missing))


def _calc_price_decimals(df: pd.DataFrame) -> int:
    # Estimate decimal precision from close column
    if "close" not in df.columns or df["close"].empty:
        return 0
    vals = df["close"].dropna().astype(float).astype(str)
    decs = [len(x.split(".")[-1]) if "." in x else 0 for x in vals.iloc[-min(500, len(vals)):]]
    return int(np.median(decs)) if decs else 0


def _volume_outliers(df: pd.DataFrame, z_thr: float = 5.0) -> int:
    if "volume" not in df.columns or len(df) < 10:
        return 0
    v = df["volume"].astype(float)
    mu, sd = float(v.mean()), float(v.std(ddof=0))
    if sd <= 0:
        return 0
    z = (v - mu) / (sd + 1e-12)
    return int((np.abs(z) > z_thr).sum())


def audit_csv(path: str, symbol: str) -> Dict[str, Any]:
    df = _load_df(path)
    rows = int(len(df))
    dupes = int(df["timestamp"].duplicated().sum()) if rows else 0
    gaps = _count_gaps(df)
    tz_ok = bool(getattr(df["timestamp"].dt, "tz", None) is not None or True)  # naive treated as OK
    price_decimals = _calc_price_decimals(df)
    vol_outliers = _volume_outliers(df)
    return {
        "symbol": symbol,
        "path": path,
        "rows": rows,
        "gaps": gaps,
        "dupes": dupes,
        "tz_ok": int(tz_ok),
        "price_decimals": price_decimals,
        "vol_outliers": vol_outliers,
    }


def _ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = pd.concat([(d["high"]-d["low"]).abs(), (d["high"]-prev_close).abs(), (d["low"]-prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().fillna(0.0)
    d["atr_pct"] = (d["atr"] / d["close"]).replace([np.inf,-np.inf], 0.0).fillna(0.0)
    d["body"] = (d["close"] - d["open"]).abs()
    d["body_atr"] = (d["body"] / d["atr"]).replace([np.inf,-np.inf], 0.0).fillna(0.0)
    # simple volume z
    lv = 50
    v = d["volume"].astype(float)
    mu = v.rolling(lv).mean()
    sd = v.rolling(lv).std(ddof=0)
    d["vol_z"] = (v - mu) / (sd + 1e-12)
    d["vol_z"] = d["vol_z"].replace([np.inf,-np.inf], 0.0).fillna(0.0)
    return d


def describe_volatility(df: pd.DataFrame) -> Dict[str, Any]:
    d = _ensure_indicators(df)
    def q(x: pd.Series, qs=(0.25,0.5,0.75)) -> Tuple[float,float,float]:
        if len(x) == 0:
            return (0.0,0.0,0.0)
        v = x.replace([np.inf,-np.inf], np.nan).dropna()
        if len(v) == 0:
            return (0.0,0.0,0.0)
        arr = v.quantile(qs)
        return tuple(float(arr.loc[q]) for q in qs)  # type: ignore[index]

    q1,q2,q3 = q(d["atr_pct"])  # already fraction
    b1,b2,b3 = q(d["body_atr"])  # ratio
    vz1,vz2,vz3 = q(d["vol_z"], qs=(0.25,0.5,0.75))
    return {
        "atr_pct_q1": q1,
        "atr_pct_q2": q2,
        "atr_pct_q3": q3,
        "body_atr_q1": b1,
        "body_atr_q2": b2,
        "body_atr_q3": b3,
        "vol_z_q1": vz1,
        "vol_z_q2": vz2,
        "vol_z_q3": vz3,
    }


def write_audit_summary(rows: List[Dict[str, Any]], out_csv: str = "reports/data_audit.csv") -> None:
    _ensure_reports_dir(os.path.dirname(out_csv) or ".")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


def validate_config_against_audit(symbol: str, coin_cfg: Dict[str, Any], vol_desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    try:
        min_atr_pct = float(coin_cfg.get("min_atr_pct", 0.0))
        max_body_atr = float(coin_cfg.get("max_body_atr", 1e9))
    except Exception:
        min_atr_pct, max_body_atr = 0.0, 1e9
    q3_atr = float(vol_desc.get("atr_pct_q3", 0.0))
    q2_body = float(vol_desc.get("body_atr_q2", 0.0))
    if min_atr_pct > q3_atr and q3_atr > 0:
        warnings.append({
            "symbol": symbol,
            "type": "min_atr_pct_gt_q3_atr",
            "min_atr_pct": min_atr_pct,
            "q3_atr_pct": q3_atr,
        })
    if max_body_atr < q2_body and q2_body > 0:
        warnings.append({
            "symbol": symbol,
            "type": "max_body_lt_median_body",
            "max_body_atr": max_body_atr,
            "median_body_atr": q2_body,
        })
    return warnings


def write_config_warnings(rows: List[Dict[str, Any]], out_csv: str = "reports/config_warnings.csv") -> None:
    if not rows:
        # still write an empty file to indicate success
        _ensure_reports_dir(os.path.dirname(out_csv) or ".")
        pd.DataFrame([], columns=["symbol","type","min_atr_pct","q3_atr_pct","max_body_atr","median_body_atr"]).to_csv(out_csv, index=False)
        return
    _ensure_reports_dir(os.path.dirname(out_csv) or ".")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

