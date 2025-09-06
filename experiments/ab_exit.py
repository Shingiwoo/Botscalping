#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict
import pandas as pd

from tools_dryrun_summary import run_dry


EXIT_KEYS = {
    "weak_tp_roi_pct","sl_atr_mult","be_trigger_pct","trailing_trigger","trailing_step",
    "be_min_gap_pct","early_stop_enabled","early_stop_bars"
}


def _ensure_reports():
    os.makedirs("reports", exist_ok=True)


def load_preset(path: str, key: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    p = root.get(key)
    if not isinstance(p, dict):
        raise SystemExit(f"Preset not found: {key}")
    return p


def overlay_exit(sym_cfg: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    for k in EXIT_KEYS:
        if k in preset:
            sym_cfg[k] = preset[k]
    return sym_cfg


def main():
    ap = argparse.ArgumentParser(description="Compare exit presets (A vs B) and summarize PF/WR/trades")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin-config", default="coin_config.json")
    ap.add_argument("--params-json", default="presets/scalping_params.json")
    ap.add_argument("--preset-a", default="AGGRESSIVE_5m")
    ap.add_argument("--preset-b", default="AGGRESSIVE_15m")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--balance", type=float, default=50.0)
    args = ap.parse_args()

    with open(args.coin_config, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    A = load_preset(args.params_json, args.preset_a)
    B = load_preset(args.params_json, args.preset_b)

    rows = []
    for label, P in ("A", A), ("B", B):
        cfg = json.loads(json.dumps(base_cfg))
        sym_cfg = cfg.get(args.symbol.upper(), cfg.get("SYMBOL_DEFAULTS", {}))
        sym_cfg = overlay_exit(sym_cfg, P)
        cfg[args.symbol.upper()] = sym_cfg
        tmp = f"_tmp_{args.symbol.upper()}_ab_exit.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        summary, _ = run_dry(args.symbol.upper(), args.csv, tmp, args.steps, args.balance)
        os.remove(tmp)
        rows.append({
            "symbol": args.symbol.upper(),
            "preset": label,
            "trades": summary.get("trades", 0),
            "wr_pct": summary.get("win_rate_pct", 0.0),
            "pf": summary.get("profit_factor", None),
        })

    out_df = pd.DataFrame(rows)
    _ensure_reports()
    out = f"reports/ab_exit_{args.symbol.upper()}.csv"
    out_df.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()

