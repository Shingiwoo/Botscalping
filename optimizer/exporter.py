from __future__ import annotations
import json, os, hashlib
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


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def export_params_to_preset_json(
    preset_path: str,
    preset_key: str,
    params: Dict[str, Any],
    merge: bool = True
) -> Dict[str, Any]:
    """
    Simpan snapshot parameter ke file preset JSON.
    - preset_key contoh: "ADAUSDT_15m"
    - jika merge=True, gabungkan dengan konten lama (kalau ada)
    Return: dict hasil akhir yang ditulis.
    """
    _ensure_parent_dir(preset_path)
    data: Dict[str, Any] = {}
    if os.path.exists(preset_path):
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    # sematkan version hash 7-char (berdasar konten params)
    try:
        h = hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:7]
    except Exception:
        h = "unknown"
    out = dict(params)
    out["version"] = h
    if merge:
        data[preset_key] = out
    else:
        data = {preset_key: out}
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data
