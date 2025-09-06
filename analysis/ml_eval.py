from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import os


def reliability(df: pd.DataFrame, prob_col: str, label_col: str, bins: int = 10) -> pd.DataFrame:
    df = df[[prob_col, label_col]].dropna().copy()
    df["bin"] = pd.cut(df[prob_col].astype(float), bins=bins, labels=False, include_lowest=True)
    grp = df.groupby("bin")
    out = grp.agg({prob_col: "mean", label_col: "mean", prob_col: "count"}).rename(columns={prob_col: "p_avg", label_col: "hit_rate", "count": "n"})
    out = out.reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate ML calibration and metrics from a CSV with prob and label columns")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prob-col", default="up_prob")
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    rel = reliability(df, args.prob_col, args.label_col, bins=10)
    os.makedirs("reports", exist_ok=True)
    out = "reports/ml_reliability.csv"
    rel.to_csv(out, index=False)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()

