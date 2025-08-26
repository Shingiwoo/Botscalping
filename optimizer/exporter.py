from __future__ import annotations
import json, os
from typing import Dict, Any

SAFE_STRATEGY_KEY = "strategy_scalping"

# peta param strategi -> kunci coin_config (legacy) yang aman di-overwrite
LEGACY_MAP = {
    "be_trigger_pct": "be_trigger_pct",
    "trailing_step": "trailing_step",
    "trailing_trigger": "trailing_trigger",
    "use_breakeven": "use_breakeven",
    "slippage_pct": "SLIPPAGE_PCT",
    # tambahkan map lain bila dibutuhkan
}

def export_params_to_coin_config(
    coin_config_path: str,
    symbol: str,
    params: Dict[str, Any],
    also_update_legacy: bool = True
) -> Dict[str, Any]:
    """
    Simpan param aktif ke coin_config.json tanpa merusak struktur lama.
    - Menyimpan snapshot lengkap param strategi ke <symbol>["strategy_scalping"]
    - (opsional) Mengisi beberapa kunci legacy yang umum dipakai runner realtrade
    Return: dict coin_config terbaru (yang sudah disimpan ke file).
    """
    if not os.path.exists(coin_config_path):
        raise FileNotFoundError(f"coin_config.json tidak ditemukan: {coin_config_path}")
    with open(coin_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if symbol not in cfg:
        # jika simbol belum ada, tiru SYMBOL_DEFAULTS kemudian tempelkan
        base = cfg.get("SYMBOL_DEFAULTS", {}).copy()
        cfg[symbol] = base
    # simpan snapshot strategi di namespace aman
    sym = cfg[symbol]
    sym[SAFE_STRATEGY_KEY] = dict(params)  # shallow copy cukup
    # opsional: isi beberapa kunci legacy agar runner lama tetap sinkron
    if also_update_legacy:
        # normalisasi beberapa nama
        if "SLIPPAGE_PCT" not in params and "slippage_pct" in params:
            params["SLIPPAGE_PCT"] = params["slippage_pct"]
        for k_src, k_dst in LEGACY_MAP.items():
            if k_src in params and params[k_src] is not None:
                sym[k_dst] = params[k_src]
    # tulis balik ke file
    with open(coin_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg
