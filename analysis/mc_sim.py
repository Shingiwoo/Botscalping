from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd


def _ensure_reports():
    os.makedirs("reports", exist_ok=True)


def mc_equity(pnls: np.ndarray, runs: int = 10000, horizon: int = 200, start: float = 100.0) -> pd.DataFrame:
    if pnls.size == 0:
        return pd.DataFrame()
    # bootstrap sample with replacement per step
    out = np.zeros((runs, horizon), dtype=float)
    for r in range(runs):
        bal = start
        max_peak = start
        dd_max = 0.0
        for t in range(horizon):
            pnl = float(np.random.choice(pnls))
            bal += pnl
            max_peak = max(max_peak, bal)
            dd = (max_peak - bal) / max_peak if max_peak > 0 else 0.0
            dd_max = max(dd_max, dd)
            out[r, t] = dd
    return pd.DataFrame({
        "dd_p50": [float(np.percentile(out[:, -1], 50))],
        "dd_p95": [float(np.percentile(out[:, -1], 95))]
    })


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo DD simulation from trades PnL distribution")
    ap.add_argument("--trades", required=True)
    ap.add_argument("--runs", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=200)
    args = ap.parse_args()
    df = pd.read_csv(args.trades)
    if "pnl" not in df.columns or df.empty:
        print("[WARN] No pnl column or empty trades")
        return
    pnls = df["pnl"].astype(float).values
    out = mc_equity(pnls, runs=args.runs, horizon=args.horizon)
    _ensure_reports()
    out_csv = "reports/sizing_sweep.csv"
    out.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")


if __name__ == "__main__":
    main()

