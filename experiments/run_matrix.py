#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List
import glob
import pandas as pd

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools_dryrun_summary import run_dry


def _ensure_reports():
    os.makedirs("reports", exist_ok=True)


def _inject_preset(cfg_path: str, params_path: str, preset_key: str, symbol: str) -> str:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    sym_cfg = cfg.get(symbol, cfg.get("SYMBOL_DEFAULTS", {}))
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            preset_all = json.load(f)
        preset = preset_all.get(preset_key, {})
    except Exception:
        preset = {}
    # aggregator block
    if isinstance(preset.get("_agg"), dict):
        sym_cfg["_agg"] = dict(preset["_agg"])  # type: ignore[index]
    elif isinstance(preset.get("aggregator"), dict):
        sym_cfg["_agg"] = dict(preset["aggregator"])  # type: ignore[index]
    # profile aggressive overlay
    if isinstance(preset.get("profiles"), dict) and isinstance(preset["profiles"].get("aggressive"), dict):
        sym_cfg.setdefault("profiles", {}).setdefault("aggressive", {}).update(preset["profiles"]["aggressive"])  # type: ignore[index]
    cfg[symbol] = sym_cfg
    tmp = f"_tmp_{symbol}_matrix.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return tmp


def main():
    ap = argparse.ArgumentParser(description="Run matrix for multiple symbols and export summary_{tf}.csv")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols or 'all' to infer from data files")
    ap.add_argument("--tf", required=True, choices=["5m","15m"], help="Only used in filename and inferring CSVs")
    ap.add_argument("--csv-glob", default=None, help="Glob to locate CSVs; if None, auto: data/*_{tf}_*.csv")
    ap.add_argument("--coin-config", default="coin_config.json")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--balance", type=float, default=50.0)
    ap.add_argument("--use-ml", type=int, choices=[0,1], default=0)
    args = ap.parse_args()

    if args.use_ml is not None:
        os.environ["USE_ML"] = str(int(args.use_ml))

    csv_glob = args.csv_glob or f"data/*_{args.tf}_*.csv"
    files = sorted(glob.glob(csv_glob))
    if args.symbols.lower() == "all":
        symbols = sorted({os.path.basename(p).split("_")[0].upper() for p in files})
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        # pick first csv matching
        csv_path = None
        for p in files:
            if os.path.basename(p).startswith(sym + "_"):
                csv_path = p
                break
        if not csv_path:
            print(f"[WARN] No CSV for {sym} under {csv_glob}")
            continue
        cfg_path = args.coin_config
        tmp = None
        if args.params_json and args.preset:
            tmp = _inject_preset(cfg_path, args.params_json, args.preset, sym)
            cfg_path = tmp
        summary, _ = run_dry(sym, csv_path, cfg_path, args.steps, args.balance)
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        rows.append({
            "symbol": sym,
            "wr_pct": summary.get("win_rate_pct", 0.0),
            "pf": summary.get("profit_factor", None),
            "trades": summary.get("trades", 0),
        })

    df = pd.DataFrame(rows)
    _ensure_reports()
    out = f"reports/summary_{args.tf}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
