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
    ap = argparse.ArgumentParser(description="Inject _agg preset and optionally tune aggressive profile thresholds.")
    ap.add_argument("--coin-config", required=True, help="Path input coin_config.json")
    # Accept aliases for robustness
    ap.add_argument("--params", "--params-json", dest="params", required=True, help="Path presets JSON (e.g. presets/scalping_params.json)")
    ap.add_argument("--preset", "--preset-key", dest="preset", required=True, help="Preset name (e.g. AGGRESSIVE_5m / AGGRESSIVE_15m)")
    ap.add_argument("--out", required=True, help="Output coin_config path")
    ap.add_argument("--force", action="store_true", help="Timpa _agg jika sudah ada.")
    ap.add_argument("--no-profile-tune", action="store_true", help="Jangan ubah profiles.aggressive thresholds")
    args = ap.parse_args()

    agg = load_agg(args.params, args.preset)

    with open(args.coin_config, "r") as f:
        cfg = json.load(f)

    updated = 0
    # Determine profile tuning target based on preset name
    p = str(args.preset).strip().lower()
    is_5m = "5m" in p
    is_15m = "15m" in p
    do_tune = not bool(args.no_profile_tune)

    for sym, sym_cfg in list(cfg.items()):
        if sym.startswith("_"):
            continue
        if not isinstance(sym_cfg, dict):
            continue
        if (not args.force) and ("_agg" in sym_cfg):
            # lewati yang sudah punya _agg jika tidak force
            continue
        sym_cfg["_agg"] = dict(agg)  # copy
        # Optional: tune aggressive profile thresholds (body/ATR and min_atr_pct) for throughput
        if do_tune and (is_5m or is_15m):
            profiles = sym_cfg.get("profiles") or {}
            if not isinstance(profiles, dict):
                profiles = {}
            aggr = profiles.get("aggressive") or {}
            if not isinstance(aggr, dict):
                aggr = {}
            flt = aggr.get("filters") or {}
            if not isinstance(flt, dict):
                flt = {}
            # Ensure filters toggles stay ON
            flt["atr"], flt["body"] = True, True
            if is_15m:
                aggr["max_body_atr"] = 1.55
                aggr["min_atr_pct"] = 0.0032
                flt["max_body_over_atr"] = 1.55
            elif is_5m:
                aggr["max_body_atr"] = 1.50
                aggr["min_atr_pct"] = 0.0022
                flt["max_body_over_atr"] = 1.50
            aggr["filters"] = flt
            profiles["aggressive"] = aggr
            sym_cfg["profiles"] = profiles
        cfg[sym] = sym_cfg
        updated += 1

    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[OK] Disuntik preset '{args.preset}' ke {updated} coin → {args.out}")
    if do_tune and (is_5m or is_15m):
        print(f"[OK] Tuning profiles.aggressive diterapkan untuk {'5m' if is_5m else '15m'} ke {updated} coin")

if __name__ == "__main__":
    main()
