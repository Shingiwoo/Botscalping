import json, os, tempfile, shutil
from optimizer.param_loader import load_params_from_csv
from optimizer.exporter import export_params_to_coin_config, SAFE_STRATEGY_KEY

def _mk_coin_config(tmpdir: str) -> str:
    p = os.path.join(tmpdir, "coin_config.json")
    cfg = {
        "_defaults": {},
        "SYMBOL_DEFAULTS": {"use_breakeven": 1, "SLIPPAGE_PCT": 0.02},
        "ADAUSDT": {"use_breakeven": 1, "SLIPPAGE_PCT": 0.02}
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return p

def test_export_params_to_coin_config_basic():
    tmp = tempfile.mkdtemp()
    try:
        path = _mk_coin_config(tmp)
        params = {
            "ema_len": 34, "sma_len": 20, "rsi_period": 25,
            "min_atr_pct": 0.006, "max_atr_pct": 0.03,
            "score_threshold": 1.2, "use_sr_filter": True, "sr_near_pct": 0.8,
            "use_mtf_plus": True,
            "be_trigger_pct": 0.45,
            "trailing_trigger": 0.7, "trailing_step": 0.45,
            "slippage_pct": 0.02,
        }
        cfg = export_params_to_coin_config(path, "ADAUSDT", params, also_update_legacy=True)
        assert "ADAUSDT" in cfg
        assert SAFE_STRATEGY_KEY in cfg["ADAUSDT"]
        snap = cfg["ADAUSDT"][SAFE_STRATEGY_KEY]
        assert snap["ema_len"] == 34 and snap["sma_len"] == 20
        # legacy keys updated
        assert cfg["ADAUSDT"]["be_trigger_pct"] == 0.45
        assert cfg["ADAUSDT"]["trailing_step"] == 0.45
        assert cfg["ADAUSDT"]["trailing_trigger"] == 0.7
        assert cfg["ADAUSDT"]["SLIPPAGE_PCT"] == 0.02
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
