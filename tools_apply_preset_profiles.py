#!/usr/bin/env python3
import json, argparse, sys

AGG_KEYS = {
    "signal_weights","strength_thresholds","regime_bounds","weight_scale",
    "sr_penalty","sd_tol_pct","vol_lookback","vol_z_thr","score_gate",
    "htf_rules","htf_fallback_discount","weight_scale_nl","min_confirms",
    "score_gate_no_confirms","min_strength","min_strength_no_confirms",
    "no_confirms_require","confirm_bonus_per","confirm_bonus_max"
}

def load_agg(params_path: str, preset_key: str) -> dict:
    with open(params_path, "r") as f:
        root = json.load(f)
    preset = root.get(preset_key, {})
    if not isinstance(preset, dict) or not preset:
        raise SystemExit(f"[ERR] Preset '{preset_key}' tidak ditemukan di {params_path}")
    return {k: preset[k] for k in AGG_KEYS if k in preset}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin-config", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--preset", required=True, help="AGGRESSIVE_5m atau AGGRESSIVE_15m, dll.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="Timpa _agg jika sudah ada.")
    args = ap.parse_args()

    agg = load_agg(args.params, args.preset)

    with open(args.coin_config, "r") as f:
        cfg = json.load(f)

    updated = 0
    for sym, sym_cfg in list(cfg.items()):
        if sym.startswith("_"):
            continue
        if not isinstance(sym_cfg, dict):
            continue
        if (not args.force) and ("_agg" in sym_cfg):
            # lewati yang sudah punya _agg jika tidak force
            continue
        sym_cfg["_agg"] = dict(agg)  # copy
        cfg[sym] = sym_cfg
        updated += 1

    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[OK] Disuntik preset '{args.preset}' ke {updated} coin → {args.out}")

if __name__ == "__main__":
    main()

