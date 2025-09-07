#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import pandas as pd

# Ensure project root on sys.path to import top-level modules when run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Use in-process function to avoid repeated CLI setup
from tools_dryrun_summary import run_dry


def _ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def try_runs(symbol: str, csv_path: str, base_cfg_path: str, steps: int, balance: float,
             grid: List[Tuple[float, float, float]]) -> pd.DataFrame:
    # Load coin_config and inject per-run overrides
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)
    res_rows: List[Dict[str, Any]] = []
    for score_gate, score_gate_no_confirms, confirm_bonus_per in grid:
        cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON
        sym_cfg = cfg.get(symbol.upper(), cfg.get("SYMBOL_DEFAULTS", {}))
        sym_cfg.setdefault("_agg", {})
        sym_cfg["_agg"]["score_gate"] = float(score_gate)
        sym_cfg["_agg"]["score_gate_no_confirms"] = float(score_gate_no_confirms)
        sym_cfg["_agg"]["confirm_bonus_per"] = float(confirm_bonus_per)
        cfg[symbol.upper()] = sym_cfg
        tmp = f"_tmp_{symbol.upper()}_ab_gates.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        summary, _ = run_dry(symbol.upper(), csv_path, tmp, steps, balance)
        os.remove(tmp)
        res_rows.append({
            "symbol": symbol.upper(),
            "score_gate": score_gate,
            "score_gate_no_confirms": score_gate_no_confirms,
            "confirm_bonus_per": confirm_bonus_per,
            "trades": summary.get("trades", 0),
            "wr_pct": summary.get("win_rate_pct", 0.0),
            "pf": summary.get("profit_factor", None),
        })
    return pd.DataFrame(res_rows)


def main():
    ap = argparse.ArgumentParser(description="Small grid A/B for gates and confirm bonus")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin-config", default="coin_config.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--balance", type=float, default=50.0)
    args = ap.parse_args()
    grid = [
        (0.56, 0.66, 0.03),
        (0.56, 0.70, 0.04),
        (0.56, 0.74, 0.05),
        (0.58, 0.66, 0.03),
        (0.58, 0.70, 0.04),
        (0.58, 0.74, 0.05),
        (0.60, 0.66, 0.03),
        (0.60, 0.70, 0.04),
        (0.60, 0.74, 0.05),
    ]
    df = try_runs(args.symbol, args.csv, args.coin_config, args.steps, args.balance, grid)
    _ensure_reports_dir()
    out = f"reports/ab_gates_{args.symbol.upper()}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
