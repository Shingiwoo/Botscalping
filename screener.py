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
import os, sys, json, argparse, time, logging
from typing import Dict, Any, List, Tuple, Optional, cast, TYPE_CHECKING
from typing import Literal, TypedDict

# --- typing support (stable for Pylance) ---
if TYPE_CHECKING:
    from signal_engine.types import Side as SideType, AggResult as AggResultType
else:
    try:
        from signal_engine.types import Side as SideType, AggResult as AggResultType  # type: ignore
    except Exception:
        SideType = Literal["LONG", "SHORT"]
        class AggResultType(TypedDict, total=False):
            ok: bool
            side: Optional[SideType]
            score: float
            strength: Literal["netral", "lemah", "cukup", "kuat"]
            reasons: List[str]
            breakdown: Dict[str, float]
            context: Dict[str, Any]
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# pastikan modul lokal bisa diimport
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- aggregator fallback (prefer package, else module lokal) ---
try:
    from signal_engine.aggregator import aggregate, build_features_from_modules  # type: ignore
except Exception:
    from aggregator import aggregate, build_features_from_modules  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
log = logging.getLogger("screener")

# === Mode penggunaan python-binance (opsional) ===
# Default: HTTP-only agar tidak butuh modul 'binance'.
USE_PYBINANCE = os.getenv("USE_PYBINANCE", "0") == "1"

# ========= helpers: load coin_config & deep merge =========
def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dict b into a (non-mutating)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[index]
        else:
            out[k] = v
    return out

def load_coin_config(path: str = "coin_config.json") -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"[WARN] coin_config load gagal: {e}")
        return {}

def overrides_from_coin_config(symbol: str, ccfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ambil subset key dari coin_config yang relevan untuk screener/aggregator.
    Mapping kunci yang dipakai screener:
      - min_atr_pct, max_atr_pct, max_body_atr
      - filters.atr/body/max_body_over_atr
      - sl_atr_mult, sl_pct
      - be_trigger_pct, trailing_trigger, trailing_step
      - use_sr_filter, sr_near_pct
      - score_gate
      - use_htf_filter -> kosongkan htf_rules jika 0
    """
    node = dict(ccfg.get(symbol, {}))
    if not node:
        return {}
    out: Dict[str, Any] = {}
    for k in (
        "min_atr_pct",
        "max_atr_pct",
        "max_body_atr",
        "sl_atr_mult",
        "sl_pct",
        "be_trigger_pct",
        "trailing_trigger",
        "trailing_step",
        "use_sr_filter",
        "sr_near_pct",
        "score_gate",
    ):
        if k in node:
            out[k] = node[k]
    # filters
    f = node.get("filters", {})
    if isinstance(f, dict) and f:
        out["filters"] = {
            "atr": bool(f.get("atr", True)),
            "body": bool(f.get("body", True)),
            "max_body_over_atr": float(f.get("max_body_over_atr", out.get("max_body_atr", 1.25))),
        }
        # sinkronkan ke max_body_atr jika ada
        out["max_body_atr"] = float(out["filters"]["max_body_over_atr"])  # type: ignore[index]
    # HTF
    try:
        if int(node.get("use_htf_filter", 1)) == 0:
            out["htf_rules"] = tuple()  # matikan HTF gating
    except Exception:
        pass
    return out

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

def _http_get_klines(symbol: str, interval: str, limit: int, market: str,
                     timeout: float = 10.0, retries: int = 2, backoff: float = 0.8) -> pd.DataFrame:
    """
    HTTP klines resmi Binance.
    - market='futures' -> fapi
    - market='spot'    -> api
    Retry ringan bila status >= 500 / network error.
    """
    base = "https://fapi.binance.com" if market == "futures" else "https://api.binance.com"
    endpoint = "/fapi/v1/klines" if market == "futures" else "/api/v3/klines"
    url = f"{base}{endpoint}"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Botscalping-Screener/1.0"})
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} server error")
            if r.status_code == 400:
                # kemungkinan symbol tidak tersedia di market tsb (misal tak listed di futures)
                # kita biarkan caller yang memutuskan fallback
                r.raise_for_status()
            r.raise_for_status()
            raw = r.json()
            return _df_from_klines(cast(List[Any], raw))
        except Exception:
            if i < retries:
                time.sleep(backoff * (i + 1))
                continue
            raise
    # Fallback for type checkers; in practice we either returned or raised above
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

def fetch_klines_binance(symbol: str, interval: str = "15m", limit: int = 720, market: str = "spot") -> pd.DataFrame:
    """
    market: 'spot' | 'futures'
    - Default: HTTP-only (stabil, tanpa dependency binance)
    - Jika USE_PYBINANCE=1, akan coba python-binance lalu fallback HTTP.
    - Bila market=futures tidak tersedia untuk symbol tsb, otomatis fallback ke spot **sekali**.
    """
    # 1) HTTP-first (default)
    if not USE_PYBINANCE:
        try:
            return _http_get_klines(symbol, interval, limit, market)
        except requests.HTTPError as e:
            # simbol tak ada di futures? fallback ke spot
            if market == "futures":
                try:
                    log.warning(f"[WARN] {symbol} tidak tersedia di FUTURES, fallback ke SPOT.")
                    return _http_get_klines(symbol, interval, limit, "spot")
                except Exception:
                    pass
            log.warning(f"[WARN] HTTP klines gagal {symbol} {interval} {market}: {e}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        except Exception as e:
            log.warning(f"[WARN] HTTP klines error {symbol} {interval} {market}: {e}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    # 2) Optional: python-binance lalu HTTP fallback
    try:
        if market == "futures":
            from binance.um_futures import UMFutures  # type: ignore
            cli = UMFutures()
            raw = cli.klines(symbol=symbol.upper(), interval=interval, limit=int(limit))
        else:
            from binance.client import Client  # type: ignore
            cli = Client()
            raw = cli.get_klines(symbol=symbol.upper(), interval=interval, limit=int(limit))
        return _df_from_klines(cast(List[Any], raw))
    except Exception as e:
        log.warning(f"[WARN] python-binance gagal {symbol} {interval} {market}: {e} -> fallback HTTP")
        try:
            return _http_get_klines(symbol, interval, limit, market)
        except requests.HTTPError as e2:
            if market == "futures":
                try:
                    log.warning(f"[WARN] {symbol} tidak tersedia di FUTURES, fallback ke SPOT.")
                    return _http_get_klines(symbol, interval, limit, "spot")
                except Exception:
                    pass
            log.warning(f"[WARN] HTTP klines gagal {symbol} {interval} {market}: {e2}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        except Exception as e2:
            log.warning(f"[WARN] HTTP klines error {symbol} {interval} {market}: {e2}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# === Preset / Parameter Aggregator ===
def load_scalping_preset(params_json: str = "presets/scalping_params.json",
                         preset_key: str = "ADAUSDT_15m") -> Dict[str, Any]:
    """
    Ambil blok parameter agregator dari scalping_params.json (fleksibel).
    Wajib ada minimal: signal_weights, regime_bounds, strength_thresholds, sr_penalty.
    """
    # fallback path bila file default tidak ada
    if not os.path.exists(params_json):
        alt = "scalping_params.json"
        if os.path.exists(alt):
            params_json = alt
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
def score_side(df_15m: pd.DataFrame, side: SideType, agg_cfg: Dict[str, Any]) -> AggResultType:
    """
    Hitung skor untuk 1 sisi (LONG/SHORT) pada 15m.
    """
    if df_15m.empty or len(df_15m) < 260:
        # Return a fully-typed AggResult shape when data is insufficient
        return {
            "ok": False,
            "side": None,
            "score": 0.0,
            "strength": "netral",
            "reasons": [],
            "breakdown": {},
            "context": {},
        }
    features = build_features_from_modules(df_15m, side)  # aman walau modul SMC tidak ada
    res = cast(AggResultType, aggregate(
        df_15m,
        side=side,
        weights=dict(agg_cfg["signal_weights"]),
        thresholds=agg_cfg,
        regime_bounds=dict(agg_cfg["regime_bounds"]),
        sr_penalty_cfg=dict(agg_cfg["sr_penalty"]),
        htf_rules=tuple(agg_cfg.get("htf_rules", ("1h","4h"))),
        features=features,
    ))
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
        return {
            "symbol": symbol,
            "market": market,
            "mode": mode,
            "status": "NO DATA",
            "side": None,
            "score": 0.0,
            "strength": "netral",
            "reasons": "",
            "regime": None,
            "htf": None,
            "confirms": None,
        }
    out: Dict[str, Any] = {"symbol": symbol, "market": market, "mode": mode}
    if mode == "scalping":
        long_r = score_side(df, cast(SideType, "LONG"), agg_cfg)
        short_r = score_side(df, cast(SideType, "SHORT"), agg_cfg)
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
        long_r = score_side(df, cast(SideType, "LONG"), agg_cfg)
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
                 params_json: str = "presets/scalping_params.json",
                 preset_key: str = "ADAUSDT_15m",
                 out_csv: Optional[str] = None,
                 coin_config_path: str = "coin_config.json",
                 use_coin_cfg: bool = True) -> pd.DataFrame:
    # default market per mode
    if market is None:
        market = "futures" if mode == "scalping" else "spot"
    agg_cfg = load_scalping_preset(params_json=params_json, preset_key=preset_key)
    coin_cfg = load_coin_config(coin_config_path) if use_coin_cfg else {}
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    # === Parallel fetch untuk responsif ===
    max_workers = min(8, max(2, len(symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for s in symbols:
            s_up = s.strip().upper()
            sym_cfg = agg_cfg
            if use_coin_cfg and coin_cfg:
                try:
                    sym_cfg = _deep_merge(agg_cfg, overrides_from_coin_config(s_up, coin_cfg))
                except Exception as e:
                    log.warning(f"[WARN] overrides {s_up} diabaikan: {e}")
            futs[ex.submit(screen_symbol, s_up, mode, market, interval, limit, sym_cfg)] = s_up
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"symbol": s, "market": market, "mode": mode, "status": f"ERROR: {e}"})
    df = pd.DataFrame(rows)
    # Ranking: ENTRY dulu, kemudian score desc
    if not df.empty:
        if "score" not in df.columns:
            df["score"] = 0.0
        df["rank_key"] = df["status"].map({"ENTRY":2,"WATCHLIST":1,"AVOID":0}).fillna(0)
        df = df.sort_values(["rank_key","score"], ascending=[False, False]).drop(columns=["rank_key"])
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Screener saved -> {out_csv}")
    log.info(f"[INFO] Done in {time.time()-t0:.2f}s")
    return df

def print_rich_table(df: pd.DataFrame) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except Exception:
        print(df.to_string(index=False))
        return
    c = Console()
    t = Table(title="Screener Ranking", box=box.SIMPLE_HEAVY)
    for col in ["symbol","market","mode","side","status","score","strength","htf","regime","confirms","reasons"]:
        t.add_column(col.upper(), justify="left" if col != "score" else "right")
    def _color_status(s: str) -> str:
        return {"ENTRY": "[bold green]ENTRY[/]", "WATCHLIST": "[yellow]WATCHLIST[/]", "AVOID": "[dim]AVOID[/]"}.get(s, s or "-")
    def _color_side(s: str) -> str:
        return {"LONG": "[bold cyan]LONG[/]", "SHORT": "[bold magenta]SHORT[/]"}.get(s, s or "-")
    for _, r in df.iterrows():
        t.add_row(
            str(r.get("symbol", "-")),
            str(r.get("market", "-")),
            str(r.get("mode", "-")),
            _color_side(str(r.get("side", "-"))),
            _color_status(str(r.get("status", "-"))),
            f"{float(r.get('score', 0.0)):.3f}",
            str(r.get("strength", "-")),
            str(r.get("htf", "-")),
            str(r.get("regime", "-")),
            str(r.get("confirms", "-")),
            str(r.get("reasons", ""))[:120],
        )
    c.print(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated, e.g. ADAUSDT,DOGEUSDT,XRPUSDT")
    ap.add_argument("--mode", choices=["scalping","spot_long"], default="scalping")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--market", choices=["spot","futures"], default=None, help="Override default market by mode")
    ap.add_argument("--limit", type=int, default=720, help="jumlah bar (>= 260 disarankan)")
    ap.add_argument("--params-json", default="presets/scalping_params.json", help="Path preset; fallback ke scalping_params.json bila tidak ada")
    ap.add_argument("--preset-key", default="ADAUSDT_15m")
    ap.add_argument("--out", default=None, help="CSV output (opsional)")
    ap.add_argument("--rich", action="store_true", help="Tampilkan tabel berwarna di terminal")
    ap.add_argument("--coin-config", default="coin_config.json", help="Path coin_config.json")
    ap.add_argument("--no-coincfg", action="store_true", help="Abaikan override dari coin_config")
    args = ap.parse_args()
    syms = [x for x in args.symbols.split(",") if x.strip()]
    df = run_screener(
        symbols=syms,
        mode=args.mode,
        interval=args.interval,
        market=args.market,
        limit=args.limit,
        params_json=args.params_json,
        preset_key=args.preset_key,
        out_csv=args.out,
        coin_config_path=args.coin_config,
        use_coin_cfg=not args.no_coincfg,
    )
    if args.rich:
        print_rich_table(df)

if __name__ == "__main__":
    main()
