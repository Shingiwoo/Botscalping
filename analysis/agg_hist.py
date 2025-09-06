from __future__ import annotations
import argparse
import glob
import os
import pandas as pd
import numpy as np

def _ensure_reports_dir(path: str = "reports") -> None:
    os.makedirs(path, exist_ok=True)


def plot_hist(symbol: str, logs_glob: str) -> str | None:
    files = sorted(glob.glob(logs_glob))
    if not files:
        print(f"[WARN] No reject logs matched: {logs_glob}")
        return None
    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if "score" in df.columns:
                dfs.append(df[["score"]].copy())
        except Exception:
            pass
    if not dfs:
        print(f"[WARN] No valid 'score' column found in logs: {logs_glob}")
        return None
    d = pd.concat(dfs, ignore_index=True)
    # Plot
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(d["score"].astype(float).values, bins=np.linspace(0,1,26), color="#4F81BD", alpha=0.9, edgecolor="#333333")
    ax.set_title(f"Aggregator Score Histogram — {symbol.upper()}")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    _ensure_reports_dir("reports")
    out = f"reports/agg_hist_{symbol.upper()}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[OK] Saved histogram → {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot histogram of aggregator scores from reject logs.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--logs-glob", default=None, help="Override glob for reading logs, default logs/rejects_{symbol}.csv")
    args = ap.parse_args()
    sym = args.symbol.upper()
    g = args.logs_glob or f"logs/rejects_{sym}.csv"
    plot_hist(sym, g)


if __name__ == "__main__":
    main()

