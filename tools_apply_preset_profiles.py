#!/usr/bin/env python3
import json, argparse, os, sys, random
from copy import deepcopy
from typing import Any, Dict


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst


AGG_KEYS = {
    "signal_weights","strength_thresholds","regime_bounds","weight_scale",
    "sr_penalty","sd_tol_pct","vol_lookback","vol_z_thr","score_gate",
    "htf_rules","htf_fallback_discount","weight_scale_nl","min_confirms",
    "score_gate_no_confirms","min_strength","min_strength_no_confirms",
    "no_confirms_require","confirm_bonus_per","confirm_bonus_max"
}


def _load_preset_blocks(params_path: str, preset_key: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    with open(params_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    preset = root.get(preset_key)
    if not isinstance(preset, dict):
        raise SystemExit(f"[ERR] Preset '{preset_key}' tidak ditemukan di {params_path}")
    # Aggregator block
    agg_block = {}
    if isinstance(preset.get("_agg"), dict):
        agg_block = preset["_agg"]
    elif isinstance(preset.get("aggregator"), dict):
        agg_block = preset["aggregator"]
    else:
        # Fallback: collect known keys at this level (legacy style)
        agg_block = {k: preset[k] for k in AGG_KEYS if k in preset}
    # profiles.aggressive overlay
    profile_agg = {}
    if isinstance(preset.get("profiles"), dict) and isinstance(preset["profiles"].get("aggressive"), dict):
        profile_agg = preset["profiles"]["aggressive"]
    elif isinstance(preset.get("profiles.aggressive"), dict):
        profile_agg = preset["profiles.aggressive"]
    return agg_block or {}, profile_agg or {}


def main():
    ap = argparse.ArgumentParser(description="Apply preset aggregator + profiles.aggressive overlay to all symbols in coin_config.")
    ap.add_argument("--coin-config", required=True, help="Path input coin_config.json")
    # Aliases for backward compatibility
    ap.add_argument("--params", "--params-json", dest="params", required=True, help="Path presets JSON (e.g. presets/scalping_params.json)")
    ap.add_argument("--preset", "--preset-key", dest="preset", required=True, help="Preset name (e.g. AGGRESSIVE_5m / AGGRESSIVE_15m)")
    ap.add_argument("--out", required=True, help="Output coin_config path")
    ap.add_argument("--force", action="store_true", help="Overwrite/merge into all symbols and allow overwrite of output file if exists.")
    args = ap.parse_args()
    # Seed determinism if provided
    seed_txt = os.getenv("BOT_SEED", "").strip()
    if seed_txt:
        try:
            seed = int(seed_txt)
            try:
                import numpy as _np
                _np.random.seed(seed)
            except Exception:
                pass
            random.seed(seed)
        except Exception:
            pass

    # Safety: prevent accidental overwrite unless --force
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"[ERR] Output exists: {args.out}. Gunakan --force untuk menimpa.")

    try:
        with open(args.coin_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        raise SystemExit(f"[ERR] Gagal membaca coin_config: {e}")
    # Validate coin_config against schema if available
    try:
        import jsonschema  # type: ignore
        with open(os.path.join("schema","coin_config.schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=cfg, schema=schema)  # type: ignore
    except Exception:
        pass

    agg_cfg, profile_agg = _load_preset_blocks(args.params, args.preset)
    # Validate presets if possible
    try:
        import jsonschema  # type: ignore
        with open(os.path.join("schema","presets.schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(args.params, "r", encoding="utf-8") as f:
            root = json.load(f)
        jsonschema.validate(instance=root, schema=schema)  # type: ignore
    except Exception:
        pass
    if not isinstance(agg_cfg, dict):
        agg_cfg = {}
    if not isinstance(profile_agg, dict):
        profile_agg = {}

    updated = 0
    for sym, sym_cfg in list(cfg.items()):
        # Skip meta keys except SYMBOL_DEFAULTS (should be processed)
        if sym.startswith("_") and sym != "SYMBOL_DEFAULTS":
            continue
        if not isinstance(sym_cfg, dict):
            continue
        # Determine whether to apply when not forcing
        should_apply = True
        if not args.force:
            has_agg = isinstance(sym_cfg.get("_agg"), dict) and bool(sym_cfg.get("_agg"))
            has_prof = isinstance(((sym_cfg.get("profiles") or {}).get("aggressive") if isinstance(sym_cfg.get("profiles"), dict) else None), dict)
            should_apply = (not has_agg) or (not has_prof)
        if not should_apply:
            continue
        # Ensure and deep-merge _agg
        sym_cfg.setdefault("_agg", {})
        if isinstance(agg_cfg, dict) and agg_cfg:
            _deep_update(sym_cfg["_agg"], agg_cfg)
        # Ensure and deep-merge profiles.aggressive
        if profile_agg:
            profiles = sym_cfg.setdefault("profiles", {}) if isinstance(sym_cfg.get("profiles"), dict) else sym_cfg.setdefault("profiles", {})
            aggr = profiles.setdefault("aggressive", {})
            _deep_update(aggr, profile_agg)
            profiles["aggressive"] = aggr
            sym_cfg["profiles"] = profiles
        cfg[sym] = sym_cfg
        updated += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[OK] Preset '{args.preset}' merged into {updated} symbols → {args.out}")

if __name__ == "__main__":
    main()
