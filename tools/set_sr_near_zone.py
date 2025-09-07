#!/usr/bin/env python3
# tools/set_sr_near_zone.py
import json, argparse, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="coin_config_agg_*.json")
    ap.add_argument("--tf", choices=["5m","15m"], required=True)
    ap.add_argument("--near", type=float, required=True, help="nilai near_zone_atr_mult")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    changed = 0
    for sym, obj in cfg.items():
        if not isinstance(obj, dict):  # skip non-symbol keys
            continue
        sr = obj.get("sr_mtf")
        if not isinstance(sr, dict):
            continue
        # hanya update chart_tf yang cocok
        if str(sr.get("chart_tf","")) .lower() == args.tf:
            try:
                cur = float(sr.get("near_zone_atr_mult", -1))
            except Exception:
                cur = -1
            if cur != args.near or args.force:
                sr["near_zone_atr_mult"] = args.near
                changed += 1

    with open(args.file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[OK] Updated near_zone_atr_mult={args.near} for TF={args.tf} on {changed} symbols in {args.file}")


if __name__ == "__main__":
    main()

