from __future__ import annotations
import json
import pandas as pd
from typing import Dict, Any, Optional

# mapping kolom CSV -> nama param internal
PARAM_MAP = {
    # indikator
    "ema": "ema_len",
    "sma": "sma_len",
    "rsi_period": "rsi_period",
    "rsi_long_min": "rsi_long_min",
    "rsi_long_max": "rsi_long_max",
    "rsi_short_min": "rsi_short_min",
    "rsi_short_max": "rsi_short_max",
    # regime / kualitas candle
    "min_atr_pct": "min_atr_pct",
    "max_atr_pct": "max_atr_pct",
    "max_body_atr": "max_body_atr",  # bila ada di CSV
    # risk/exit
    "sl_atr_mult": "sl_atr_mult",
    "sl_pct": "sl_pct",
    "trailing_trigger": "trailing_trigger",
    "trailing_step": "trailing_step",
    "be_trigger_pct": "be_trigger_pct",  # bila ada di CSV
    # scoring
    "score_threshold": "score_threshold",
    # filter S/R dan MTF
    "sr_near_pct": "sr_near_pct",
    "use_sr_filter": "use_sr_filter",
    "use_mtf_plus": "use_mtf_plus",
}

def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool): return val
    if val is None: return False
    s = str(val).strip().lower()
    return s in ("1","true","yes","y","on")

def load_params_from_csv(
    path: str,
    min_wr: float = 70.0,
    min_pf: float = 2.0,
    min_trades: int = 20,
    prefer: str = "pf_then_wr",  # alternatif: "wr_then_pf"
    rank: int = 1,               # ambil kandidat ke-n setelah disaring & diurutkan
) -> Dict[str, Any]:
    """
    Baca opt_results-<SYMBOL>.csv, pilih kandidat terbaik sesuai constraint & preferensi.
    Return: dict param internal {ema_len, sma_len, ...}
    """
    df = pd.read_csv(path)
    # normalisasi nama kolom penting agar aman huruf besar/kecil
    cols = {c.lower(): c for c in df.columns}
    wr_col = cols.get("win_rate") or cols.get("win_rate_pct")
    pf_col = cols.get("profit_factor") or cols.get("pf")
    tr_col = cols.get("trades") or cols.get("trade_count")

    if wr_col is None or pf_col is None:
        raise ValueError("CSV tidak punya kolom WinRate / ProfitFactor yang dikenali.")

    # filter constraint
    if tr_col and tr_col in df.columns:
        cands = df[(df[wr_col] >= min_wr) & (df[pf_col] >= min_pf) & (df[tr_col] >= min_trades)].copy()
    else:
        cands = df[(df[wr_col] >= min_wr) & (df[pf_col] >= min_pf)].copy()

    # jika kosong, longgarkan constraint secara bertahap
    if cands.empty:
        cands = df.sort_values([pf_col, wr_col], ascending=[False, False]).copy()
    else:
        # urutkan sesuai preferensi
        if prefer == "wr_then_pf":
            cands = cands.sort_values([wr_col, pf_col], ascending=[False, False])
        else:
            cands = cands.sort_values([pf_col, wr_col], ascending=[False, False])

    # ambil kandidat ke-rank
    idx = min(max(rank, 1), len(cands)) - 1
    row = cands.iloc[idx].to_dict()

    # map ke param internal
    params: Dict[str, Any] = {}
    for csv_key, internal in PARAM_MAP.items():
        if csv_key in df.columns:
            params[internal] = row.get(csv_key)
    # koersi tipe
    # ints
    for k in ("ema_len","sma_len","rsi_period","rsi_long_min","rsi_long_max","rsi_short_min","rsi_short_max"):
        if k in params and params[k] is not None:
            params[k] = int(round(float(params[k])))
    # floats
    for k in ("min_atr_pct","max_atr_pct","max_body_atr","sl_atr_mult","sl_pct",
              "trailing_trigger","trailing_step","be_trigger_pct","score_threshold","sr_near_pct"):
        if k in params and params[k] is not None:
            params[k] = float(params[k])
    # bools
    for k in ("use_sr_filter","use_mtf_plus"):
        if k in params:
            params[k] = _coerce_bool(params[k])

    return params

def load_params_from_json(
    preset_path: str,
    preset_key: str
) -> Dict[str, Any]:
    """
    Baca JSON preset (hasil export Streamlit) dan kembalikan dict param siap pakai.
    Diharapkan preset menyimpan kunci seperti 'ema_len','sma_len','rsi_period', dst.
    """
    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or preset_key not in data:
        raise KeyError(f"Preset key '{preset_key}' tidak ditemukan di {preset_path}")
    raw = data[preset_key]
    params: Dict[str, Any] = dict(raw)  # shallow copy
    # cast tipe yang umum
    for k in ("ema_len","sma_len","rsi_period","rsi_long_min","rsi_long_max","rsi_short_min","rsi_short_max"):
        if k in params and params[k] is not None:
            params[k] = int(round(float(params[k])))
    for k in ("min_atr_pct","max_atr_pct","max_body_atr","sl_atr_mult","sl_pct",
              "trailing_trigger","trailing_step","be_trigger_pct","score_threshold","sr_near_pct"):
        if k in params and params[k] is not None:
            params[k] = float(params[k])
    for k in ("use_sr_filter","use_mtf_plus"):
        if k in params:
            params[k] = _coerce_bool(params[k])
    return params
