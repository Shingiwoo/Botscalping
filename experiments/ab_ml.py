#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools_dryrun_summary import run_dry


def _ensure_reports():
    os.makedirs("reports", exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Sweep ML score threshold and compare PF/WR/trades")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin-config", default="coin_config.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--balance", type=float, default=50.0)
    ap.add_argument("--thr", nargs="*", type=float, default=[0.70,0.75,0.80,0.85,0.90])
    args = ap.parse_args()

    rows = []
    # Enable ML explicitly for the sweep
    os.environ["USE_ML"] = "1"
    for t in args.thr:
        os.environ["SCORE_THRESHOLD"] = str(float(t))
        summary, _ = run_dry(args.symbol.upper(), args.csv, args.coin_config, args.steps, args.balance)
        rows.append({
            "symbol": args.symbol.upper(),
            "ml_score_thr": float(t),
            "trades": summary.get("trades", 0),
            "wr_pct": summary.get("win_rate_pct", 0.0),
            "pf": summary.get("profit_factor", None),
        })

    df = pd.DataFrame(rows)
    _ensure_reports()
    out = f"reports/ab_ml_{args.symbol.upper()}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
