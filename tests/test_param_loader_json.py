import os, json, tempfile, shutil
from optimizer.param_loader import load_params_from_json


def test_load_params_from_json_basic():
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "presets.json")
        data = {
            "ADAUSDT_15m": {
                "ema_len": 34, "sma_len": 20, "rsi_period": 25,
                "rsi_long_min": 18, "rsi_long_max": 54,
                "rsi_short_min": 46, "rsi_short_max": 82,
                "min_atr_pct": 0.006, "max_atr_pct": 0.03, "max_body_atr": 0.9,
                "sl_atr_mult": 1.6, "sl_pct": 0.008,
                "trailing_trigger": 0.7, "trailing_step": 0.45,
                "be_trigger_pct": 0.45, "score_threshold": 1.2,
                "use_sr_filter": True, "sr_near_pct": 0.8, "use_mtf_plus": True
            }
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        params = load_params_from_json(p, "ADAUSDT_15m")
        assert params["ema_len"] == 34 and params["sma_len"] == 20
        assert abs(params["min_atr_pct"] - 0.006) < 1e-9
        assert params["use_sr_filter"] is True and params["use_mtf_plus"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
