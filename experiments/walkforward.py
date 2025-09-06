#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List
import pandas as pd
from tools_dryrun_summary import simulate_dryrun


def _ensure_reports():
    os.makedirs("reports", exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Rolling-origin walk-forward evaluation; outputs walkforward_{tf}.csv")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin-config", default="coin_config.json")
    ap.add_argument("--window", type=int, default=500, help="Bars per window for walkforward")
    ap.add_argument("--steps", type=int, default=1500, help="Max bars per run")
    ap.add_argument("--balance", type=float, default=50.0)
    ap.add_argument("--tf", required=True, choices=["5m","15m"], help="Only used for output filename")
    args = ap.parse_args()

    df_full = pd.read_csv(args.csv)
    if "timestamp" not in df_full.columns:
        if "open_time" in df_full.columns:
            df_full["timestamp"] = pd.to_datetime(df_full["open_time"], unit="ms", errors="coerce")
        elif "date" in df_full.columns:
            df_full["timestamp"] = pd.to_datetime(df_full["date"], errors="coerce")
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], errors="coerce")
    df_full = df_full.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df_full)
    if n == 0:
        print("[ERR] Empty CSV")
        return

    # Windows: use sequential non-overlapping blocks of size 'window'
    rows: List[Dict[str, Any]] = []
    for start in range(0, n, args.window):
        end = min(n, start + args.window)
        block = df_full.iloc[start:end].copy()
        if len(block) < 50:
            continue
        summary, trades = simulate_dryrun(block, args.symbol.upper(), args.coin_config, min(args.steps, len(block)), args.balance)
        rows.append({
            "symbol": args.symbol.upper(),
            "start": str(block["timestamp"].iloc[0]),
            "end": str(block["timestamp"].iloc[-1]),
            "trades": summary.get("trades", 0),
            "wr_pct": summary.get("win_rate_pct", 0.0),
            "pf": summary.get("profit_factor", None),
        })

    out_df = pd.DataFrame(rows)
    _ensure_reports()
    out = f"reports/walkforward_{args.tf}.csv"
    out_df.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()

