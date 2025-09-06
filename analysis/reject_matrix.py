from __future__ import annotations
import argparse, glob, os
import pandas as pd


def _ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def compute_contrib(df: pd.DataFrame) -> pd.Series:
    # Identify boolean-ish columns for reasons (exclude known base columns)
    base = {"ts","symbol","step","score","strength","extras"}
    cols = [c for c in df.columns if c not in base]
    denom = len(df)
    if denom <= 0:
        return pd.Series(dtype=float)
    out = {}
    for c in cols:
        try:
            v = pd.to_numeric(df[c], errors="coerce")
            rate = float((v.fillna(0) != 0).sum()) / float(denom)
            out[c] = rate
        except Exception:
            continue
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser(description="Compute reject contribution matrix per coin from logs/rejects_*.csv")
    ap.add_argument("--tf", required=True, choices=["5m","15m"], help="Only used in filename for output")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob("logs/rejects_*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        sym = os.path.basename(p).split("_")[1].split(".")[0].upper()
        s = compute_contrib(df)
        s.name = sym
        rows.append(s)
    if not rows:
        print("[WARN] No reject logs found under logs/rejects_*.csv")
        return
    mat = pd.DataFrame(rows).fillna(0.0)
    mat.index.name = "symbol"
    _ensure_reports_dir()
    out = f"reports/reject_matrix_{args.tf}.csv"
    mat.to_csv(out)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()

