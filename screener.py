#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screener koin (SCALPING vs SPOT/LONG-only) berbasis MTF + S/R + Volume + (opsional) SMC.
 - Mode:
   1) scalping  : 15m, Futures, evaluasi LONG & SHORT -> pilih skor tertinggi
   2) spot_long : Spot (default) atau Futures LONG-only (opsional)
 - Output: tabel ranking, CSV opsional.
Ketergantungan:
  - python-binance==1.0.20 (public endpoint cukup, tanpa API key)
  - pandas, numpy
Integrasi skor via aggregator.aggregate (modular).
"""
from __future__ import annotations
import os, sys, json, argparse, time
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# pastikan modul lokal bisa diimport
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregator import aggregate, build_features_from_modules  # type: ignore
from regime import compute_vol_metrics  # type: ignore

# === DataFetcher (Binance) ===
def _df_from_klines(raw: list) -> pd.DataFrame:
    """
    raw bar: [open_time, open, high, low, close, volume, close_time, qav, trades, tb_base, tb_quote, ignore]
    """
    if not raw:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","tb_base","tb_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp","open","high","low","close","volume"]].dropna().reset_index(drop=True)
    return df

def fetch_klines_binance(symbol: str, interval: str = "15m", limit: int = 720, market: str = "spot") -> pd.DataFrame:
    """
    market: 'spot' | 'futures'
    """
    try:
        if market == "futures":
            from binance.um_futures import UMFutures  # type: ignore
            cli = UMFutures()  # public endpoint cukup
            raw = cli.klines(symbol=symbol.upper(), interval=interval, limit=int(limit))
        else:
            from binance.client import Client  # type: ignore
            cli = Client()  # public endpoint cukup
            raw = cli.get_klines(symbol=symbol.upper(), interval=interval, limit=int(limit))
        return _df_from_klines(raw)
    except Exception as e:
        print(f"[WARN] fetch_klines_binance gagal {symbol} {interval} {market}: {e}")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# === Preset / Parameter Aggregator ===
def load_scalping_preset(params_json: str = "scalping_params.json",
                         preset_key: str = "ADAUSDT_15m") -> Dict[str, Any]:
    """
    Ambil blok parameter agregator dari scalping_params.json (fleksibel).
    Wajib ada minimal: signal_weights, regime_bounds, strength_thresholds, sr_penalty.
    """
    try:
        with open(params_json, "r") as f:
            root = json.load(f)
        node = root.get(preset_key, root)
        # fallback aman
        return {
            "signal_weights": node.get("signal_weights", {}),
            "strength_thresholds": node.get("strength_thresholds", {"weak":0.25,"fair":0.5,"strong":0.75}),
            "regime_bounds": node.get("regime_bounds", {"atr_p1":0.01,"atr_p2":0.05,"bbw_q1":0.01,"bbw_q2":0.05}),
            "sr_penalty": node.get("sr_penalty", {"base_pct":0.6,"k_atr":0.5}),
            "sd_tol_pct": node.get("sd_tol_pct", 1.5),
            "vol_lookback": node.get("vol_lookback", 20),
            "vol_z_thr": node.get("vol_z_thr", 2.0),
            "score_gate": node.get("score_gate", 0.50),
            "htf_rules": tuple(node.get("htf_rules", ["1h","4h"])),
            "weight_scale": node.get("weight_scale", {}),
            "weight_scale_nl": node.get("weight_scale_nl", {}),
            "min_confirms": node.get("min_confirms", None),
        }
    except Exception as e:
        print(f"[WARN] load_scalping_preset gagal: {e}")
        # default sangat konservatif
        return {
            "signal_weights": {"sc_trend_htf":0.35,"adx":0.15,"body_atr":0.1,"width_atr":0.1,"rsi":0.05,
                               "sr_breakout":0.2,"sr_test":0.1,"sr_reject":0.15,"sd_proximity":0.2,
                               "vol_confirm":0.1,"fvg_confirm":0.1,"penalty_near_opposite_sr":0.2},
            "strength_thresholds": {"weak":0.25,"fair":0.5,"strong":0.75},
            "regime_bounds": {"atr_p1":0.01,"atr_p2":0.05,"bbw_q1":0.01,"bbw_q2":0.05},
            "sr_penalty": {"base_pct":0.6,"k_atr":0.5},
            "sd_tol_pct": 1.5,
            "vol_lookback": 20,
            "vol_z_thr": 2.0,
            "score_gate": 0.50,
            "htf_rules": ("1h","4h"),
            "weight_scale": {},
            "weight_scale_nl": {},
            "min_confirms": None,
        }

# === Core Scoring ===
def score_side(df_15m: pd.DataFrame, side: str, agg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hitung skor untuk 1 sisi (LONG/SHORT) pada 15m.
    """
    if df_15m.empty or len(df_15m) < 260:
        return {"ok": False, "score": 0.0, "strength": "netral", "reasons": [], "context": {}}
    features = build_features_from_modules(df_15m, side)  # aman walau modul SMC tidak ada
    res = aggregate(
        df_15m,
        side=side,
        weights=dict(agg_cfg["signal_weights"]),
        thresholds=agg_cfg,
        regime_bounds=dict(agg_cfg["regime_bounds"]),
        sr_penalty_cfg=dict(agg_cfg["sr_penalty"]),
        htf_rules=tuple(agg_cfg.get("htf_rules", ("1h","4h"))),
        features=features,
    )
    return res

def screen_symbol(symbol: str,
                  mode: str,
                  market: str,
                  interval: str,
                  limit: int,
                  agg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    mode: 'scalping' | 'spot_long'
    market default: 'futures' untuk scalping, 'spot' untuk spot_long (bisa di-override CLI)
    """
    df = fetch_klines_binance(symbol, interval=interval, limit=limit, market=market)
    if df.empty:
        return {"symbol": symbol, "market": market, "mode": mode, "status": "NO DATA"}
    out: Dict[str, Any] = {"symbol": symbol, "market": market, "mode": mode}
    if mode == "scalping":
        long_r = score_side(df, "LONG", agg_cfg)
        short_r = score_side(df, "SHORT", agg_cfg)
        pick = max([("LONG", long_r), ("SHORT", short_r)], key=lambda x: x[1].get("score", 0.0))
        side, res = pick
        status = "ENTRY" if res.get("ok") else ("WATCHLIST" if res.get("score",0.0) >= 0.45 else "AVOID")
        out.update({
            "side": side,
            "score": round(float(res.get("score",0.0)), 3),
            "strength": res.get("strength","netral"),
            "reasons": ",".join(res.get("reasons",[])),
            "regime": (res.get("context") or {}).get("regime"),
            "htf": (res.get("context") or {}).get("trend"),
            "confirms": (res.get("context") or {}).get("confirms"),
            "status": status,
        })
        return out
    else:
        # LONG-only screening (Spot atau Futures LONG-only)
        long_r = score_side(df, "LONG", agg_cfg)
        status = "ENTRY" if long_r.get("ok") else ("WATCHLIST" if long_r.get("score",0.0) >= 0.45 else "AVOID")
        out.update({
            "side": "LONG",
            "score": round(float(long_r.get("score",0.0)), 3),
            "strength": long_r.get("strength","netral"),
            "reasons": ",".join(long_r.get("reasons",[])),
            "regime": (long_r.get("context") or {}).get("regime"),
            "htf": (long_r.get("context") or {}).get("trend"),
            "confirms": (long_r.get("context") or {}).get("confirms"),
            "status": status,
        })
        return out

def run_screener(symbols: List[str],
                 mode: str = "scalping",
                 interval: str = "15m",
                 market: Optional[str] = None,
                 limit: int = 720,
                 params_json: str = "scalping_params.json",
                 preset_key: str = "ADAUSDT_15m",
                 out_csv: Optional[str] = None) -> pd.DataFrame:
    # default market per mode
    if market is None:
        market = "futures" if mode == "scalping" else "spot"
    agg_cfg = load_scalping_preset(params_json=params_json, preset_key=preset_key)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for s in symbols:
        try:
            rows.append(screen_symbol(s.strip().upper(), mode, market, interval, limit, agg_cfg))
        except Exception as e:
            rows.append({"symbol": s, "market": market, "mode": mode, "status": f"ERROR: {e}"})
    df = pd.DataFrame(rows)
    # Ranking: ENTRY dulu, kemudian score desc
    if not df.empty:
        df["rank_key"] = df["status"].map({"ENTRY":2,"WATCHLIST":1,"AVOID":0}).fillna(0)
        df = df.sort_values(["rank_key","score"], ascending=[False, False]).drop(columns=["rank_key"])
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Screener saved -> {out_csv}")
    print(f"[INFO] Done in {time.time()-t0:.2f}s")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated, e.g. ADAUSDT,DOGEUSDT,XRPUSDT")
    ap.add_argument("--mode", choices=["scalping","spot_long"], default="scalping")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--market", choices=["spot","futures"], default=None, help="Override default market by mode")
    ap.add_argument("--limit", type=int, default=720, help="jumlah bar (>= 260 disarankan)")
    ap.add_argument("--params-json", default="scalping_params.json")
    ap.add_argument("--preset-key", default="ADAUSDT_15m")
    ap.add_argument("--out", default=None, help="CSV output (opsional)")
    args = ap.parse_args()
    syms = [x for x in args.symbols.split(",") if x.strip()]
    run_screener(
        symbols=syms,
        mode=args.mode,
        interval=args.interval,
        market=args.market,
        limit=args.limit,
        params_json=args.params_json,
        preset_key=args.preset_key,
        out_csv=args.out,
    )

if __name__ == "__main__":
    main()

