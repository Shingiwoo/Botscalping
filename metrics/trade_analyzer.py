from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict
import pandas as pd
import numpy as np


def _ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def _load_coin_cfg(path: str, symbol: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    return cfg.get(symbol.upper(), cfg.get("SYMBOL_DEFAULTS", {})) or {}


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-prev_close).abs(), (df["low"]-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().fillna(0.0)
    return atr


def analyze(trades_csv: str, ohlcv_csv: str, symbol: str, coin_config: str) -> pd.DataFrame:
    trades = pd.read_csv(trades_csv)
    if trades.empty:
        return trades
    df = pd.read_csv(ohlcv_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["atr"] = _compute_atr(df)
    cfg = _load_coin_cfg(coin_config, symbol)
    sl_atr_mult = float(cfg.get("sl_atr_mult", 1.5))

    # Align entry_time to bar timestamp granularity
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    # Map ATR at nearest prior bar
    df = df.set_index("timestamp")
    atr_at_entry = []
    for t in trades["entry_time"].tolist():
        if pd.isna(t):
            atr_at_entry.append(np.nan)
        else:
            # nearest prior timestamp
            try:
                idx = df.index.get_indexer([t], method="pad")[0]
                atr_at_entry.append(float(df["atr"].iloc[idx]))
            except Exception:
                atr_at_entry.append(np.nan)
    trades["atr_entry"] = atr_at_entry
    # Compute R distance in price and R-multiple signed
    def r_mult(row: pd.Series) -> float:
        entry = float(row.get("entry_price", 0.0))
        exitp = float(row.get("exit_price", 0.0))
        side = str(row.get("side", "LONG"))
        atr_e = float(row.get("atr_entry", 0.0))
        R = abs(sl_atr_mult * atr_e)
        if R <= 0:
            return 0.0
        move = (exitp - entry) if side == "LONG" else (entry - exitp)
        return float(move / R)

    trades["R"] = trades.apply(r_mult, axis=1)
    trades["is_win"] = trades["R"] > 0
    # Compute aggregate stats per-symbol
    wr = trades["is_win"].mean() * 100.0 if len(trades) > 0 else 0.0
    exp = float(trades["R"].mean()) if len(trades) > 0 else 0.0
    mae = float(trades["R"].min()) if len(trades) > 0 else 0.0
    mfe = float(trades["R"].max()) if len(trades) > 0 else 0.0
    summ = pd.DataFrame([
        {"symbol": symbol.upper(), "trades": int(len(trades)), "wr_pct": round(wr,2), "expectancy_R": round(exp,4), "MAE_R": mae, "MFE_R": mfe}
    ])
    _ensure_reports_dir()
    out = f"reports/trade_stats_{symbol.upper()}.csv"
    summ.to_csv(out, index=False)
    print(f"[OK] Saved {out}")
    return trades


def main():
    ap = argparse.ArgumentParser(description="Analyze trades to compute R-multiple distribution and expectancy.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--trades", required=True, help="Path to trades CSV (from tools_dryrun_summary --out)")
    ap.add_argument("--csv", required=True, help="Path to OHLCV CSV used for the run")
    ap.add_argument("--coin-config", default="coin_config.json")
    args = ap.parse_args()
    analyze(args.trades, args.csv, args.symbol, args.coin_config)


if __name__ == "__main__":
    main()

