import json
import os
import subprocess


def test_apply_preset_merge(tmp_path):
    # minimal coin_config with two symbols
    cfg = {
        "SYMBOL_DEFAULTS": {"min_atr_pct": 0.003, "max_body_atr": 1.5},
        "ADAUSDT": {"min_atr_pct": 0.0035, "profiles": {"aggressive": {"sl_atr_mult": 1.5}}},
        "XRPUSDT": {}
    }
    cfg_path = tmp_path / "coin_config.json"
    cfg_path.write_text(json.dumps(cfg))
    # use existing presets/scalping_params.json
    params = os.path.join("presets", "scalping_params.json")
    out = tmp_path / "out.json"
    # run script
    cmd = ["python", "tools_apply_preset_profiles.py", "--coin-config", str(cfg_path), "--params", params, "--preset", "AGGRESSIVE_15m", "--out", str(out), "--force"]
    subprocess.check_call(cmd)
    merged = json.loads(out.read_text())
    assert isinstance(merged.get("ADAUSDT", {}).get("_agg"), dict)
    assert isinstance(merged.get("XRPUSDT", {}).get("_agg"), dict)
    # aggressive profile overlay exists
    assert isinstance(merged.get("ADAUSDT", {}).get("profiles", {}).get("aggressive"), dict)

