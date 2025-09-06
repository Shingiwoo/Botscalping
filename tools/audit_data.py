#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os
from typing import Any, Dict, List
import pandas as pd

from analysis.data_audit import audit_csv, describe_volatility, write_audit_summary, validate_config_against_audit, write_config_warnings


def _load_coin_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="Audit CSV quality and volatility; validate config vs distribution.")
    ap.add_argument("--symbols", required=True, help="all or comma-separated symbols (e.g., ADAUSDT,ETHUSDT)")
    ap.add_argument("--tf", required=True, choices=["5m","15m"], help="Timeframe selector for matching files")
    ap.add_argument("--glob", required=True, help="Glob pattern for CSV files, e.g. data/*_15m_*.csv")
    ap.add_argument("--coin-config", default="coin_config.json", help="Path to coin_config.json for validation")
    args = ap.parse_args()

    symbols: List[str]
    if args.symbols.strip().lower() == "all":
        # infer from files
        symbols = []
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    coin_cfg = _load_coin_config(args.coin_config)
    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[WARN] No files matched: {args.glob}")
    results: List[Dict[str, Any]] = []
    warns: List[Dict[str, Any]] = []

    for p in files:
        base = os.path.basename(p)
        # extract symbol prefix before first underscore
        sym = base.split("_")[0].upper()
        if symbols and sym not in symbols:
            continue
        try:
            aud = audit_csv(p, sym)
            # load DF once for volatility description
            df = pd.read_csv(p)
            if "timestamp" not in df.columns:
                if "open_time" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
                elif "date" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            vol = describe_volatility(df)
            row = {**aud, **vol}
            results.append(row)
            # validation vs config
            sym_cfg = coin_cfg.get(sym, coin_cfg.get("SYMBOL_DEFAULTS", {})) if isinstance(coin_cfg, dict) else {}
            warns.extend(validate_config_against_audit(sym, sym_cfg or {}, vol))
        except Exception as e:
            print(f"[WARN] audit failed for {p}: {e}")

    write_audit_summary(results, out_csv="reports/data_audit.csv")
    write_config_warnings(warns, out_csv="reports/config_warnings.csv")
    print(f"[OK] Wrote reports/data_audit.csv and reports/config_warnings.csv")


if __name__ == "__main__":
    main()

