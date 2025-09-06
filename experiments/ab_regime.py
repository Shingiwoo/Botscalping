#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict
import pandas as pd

from tools_dryrun_summary import run_dry


def _ensure_reports():
    os.makedirs("reports", exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="A/B regime-aware vs regime-agnostic aggregator")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin-config", default="coin_config.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--balance", type=float, default=50.0)
    args = ap.parse_args()

    with open(args.coin_config, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    rows = []
    for label in ("agnostic", "regime_aware"):
        cfg = json.loads(json.dumps(base_cfg))
        sym = args.symbol.upper()
        sc = cfg.get(sym, cfg.get("SYMBOL_DEFAULTS", {}))
        sc.setdefault("_agg", {})
        if label == "agnostic":
            sc["_agg"]["weight_scale"] = {}
            sc["_agg"]["weight_scale_nl"] = {}
        else:
            # Basic regime scaling: boost SR and SD proximity in trend, reduce in sideway
            sc["_agg"]["weight_scale"] = {
                "trend": {"sr_breakout": 1.2, "sd_proximity": 1.1},
                "sideway": {"sr_breakout": 0.9, "sd_proximity": 0.95}
            }
            sc["_agg"]["weight_scale_nl"] = {"bbw_k_tanh": 1.0, "bbw_lo": 0.01, "bbw_hi": 0.08}
        cfg[sym] = sc
        tmp = f"_tmp_{sym}_ab_regime.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        summary, _ = run_dry(sym, args.csv, tmp, args.steps, args.balance)
        os.remove(tmp)
        rows.append({
            "symbol": sym, "mode": label,
            "trades": summary.get("trades", 0),
            "wr_pct": summary.get("win_rate_pct", 0.0),
            "pf": summary.get("profit_factor", None),
        })

    df = pd.DataFrame(rows)
    _ensure_reports()
    out = f"reports/ab_regime_{args.symbol.upper()}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()

