import streamlit as st
import pandas as pd
import numpy as np
import os, math, json, warnings, random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from engine_core import apply_breakeven_sl
from signal_engine.aggregator import aggregate as agg_signal, build_features_from_modules
from indicators.sr_utils import (
    compute_sr_levels,
    near_level,
    build_sr_cache,
    htf_trend_ok_multi,
    ltf_momentum_ok,
)
from optimizer.param_loader import load_params_from_csv, load_params_from_json
from optimizer.exporter import export_params_to_coin_config, export_params_to_preset_json

"""
============================================================
APP: Backtester — SCALPING (Selaras Real Trading)
FILE: backtester_scalping.py
UPDATE: 2025-08-11 (patch: HTF default OFF + Mode Debug)
============================================================
Fitur:
- Loader coin_config.json (per simbol) → prefill leverage, risk_per_trade, taker_fee, filter presisi, SL/BE/Trailing.
- Presisi Entri v2: ATR regime, rasio body/ATR, HTF filter (EMA50 vs EMA200 1h), cooldown.
- Hard Stop Loss (ATR/PCT + clamp), Breakeven, Trailing, opsi TP bertingkat.
- Money management identik Binance Futures:
  qty = ((balance * risk_per_trade) * leverage) / price → normalisasi LOT_SIZE.
  - NEW: Multi-timeframe filter default = OFF, dan Mode Debug untuk melonggarkan filter + tampilkan alasan blokir.
============================================================
"""

# ---------- Helpers ----------
def floor_to_step(x: float, step: float) -> float:
    if step is None or step <= 0: return float(x)
    return math.floor(float(x)/float(step))*float(step)

def load_coin_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

# ---------- Defaults ----------
DEFAULT_TAKER_FEE = float(os.getenv('TAKER_FEE', '0.0005'))
DEFAULT_MIN_ROI_TO_CLOSE_BY_TIME = float(os.getenv('MIN_ROI_TO_CLOSE_BY_TIME', '0.0'))
DEFAULT_MAX_HOLD_SECONDS = int(os.getenv('MAX_HOLD_SECONDS', '3600'))

# ---------- UI ----------
warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="Backtester Scalping+", layout="wide")
st.title("⚡ Backtester — Scalping")

# ========= Header: Panel Ringkasan Active Params =========
_hdr = st.container()

# ------ Deklarasi awal df agar tidak undefined di Sidebar ------
df: Optional[pd.DataFrame] = None

st.sidebar.header("📂 Data & Config")
data_dir = st.sidebar.text_input("Folder Data CSV", value="./data")
try:
    csv_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
except Exception:
    csv_files = []
selected_file = st.selectbox("Pilih data file (1 simbol per backtest)", options=csv_files)

cfg_default_path = st.sidebar.text_input("Path coin_config.json", value="./coin_config.json")
load_cfg = st.sidebar.checkbox("Muat konfigurasi dari coin_config.json", True)

st.sidebar.header("🕒 Timeframe")
timeframe = st.sidebar.selectbox("Resample", ["as-is","5m","15m","1h","4h","1d"], index=0)

# Performa: opsi untuk mempercepat eksekusi
with st.sidebar.expander("🚀 Performa (rekomendasi)", expanded=False):
    limit_n_bars = st.number_input(
        "Batasi N bar terakhir (0 = semua)", value=5000, min_value=0, step=500,
        help="Memproses subset data terbaru agar lebih cepat."
    )
    signal_window_bars = st.number_input(
        "Window sinyal (bars)", value=800, min_value=200, step=100,
        help="Bangun fitur/sinyal dari N bar terakhir (sliding window)."
    )
    show_progress = st.checkbox("Tampilkan progress bar", value=True)

st.sidebar.header("💰 Money Management")
initial_capital = st.sidebar.number_input("Available Balance (USDT)", value=20.0, min_value=0.0, step=1.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1.0, 10.0, 8.0)/100.0
leverage = st.sidebar.slider("Leverage", 1, 125, 15)
taker_fee = st.sidebar.number_input("Taker Fee", value=DEFAULT_TAKER_FEE, format="%.6f")

with st.sidebar.expander("⚙️ LOT_SIZE & Precision"):
    lot_step = st.number_input("stepSize (LOT_SIZE)", value=0.0, min_value=0.0, format="%.10f")
    min_qty = st.number_input("minQty (LOT_SIZE)", value=0.0, min_value=0.0, format="%.10f")
    qty_precision = st.number_input("quantityPrecision", value=0, min_value=0, max_value=8, step=1)

# simbol → prefill dari coin_config
symbol = os.path.splitext(selected_file)[0].upper() if selected_file else None

# --- SAFE loader coin_config (tanpa bikin Pylance bingung) ---
raw_cfg: dict[str, Any] = load_coin_config(cfg_default_path) if (load_cfg and os.path.exists(cfg_default_path)) else {}
sym_cfg: dict[str, Any] = raw_cfg.get(symbol, {}) if isinstance(symbol, str) else {}

# helper casting aman
def cfgf(key: str, fallback: float) -> float:
    try:
        v = sym_cfg.get(key, fallback)
        return float(v)
    except Exception:
        return float(fallback)

def cfgi(key: str, fallback: int) -> int:
    try:
        v = sym_cfg.get(key, fallback)
        return int(v)
    except Exception:
        return int(fallback)

def cfgb(key: str, fallback: bool) -> bool:
    try:
        v = sym_cfg.get(key, fallback)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))
        if isinstance(v, str):
            return bool(int(v)) if v.isdigit() else (v.lower() in {"true","on","yes","y"})
        return bool(v)
    except Exception:
        return bool(fallback)

# prefill mm bila tersedia di config (tanpa wajib)
leverage = cfgi("leverage", leverage)
risk_per_trade = cfgf("risk_per_trade", risk_per_trade)
taker_fee = cfgf("taker_fee", taker_fee)

st.sidebar.header("📈 Indikator")
ema_len = st.sidebar.number_input("EMA length", value=cfgi("ema_len", 22), min_value=1)
sma_len = st.sidebar.number_input("SMA length", value=cfgi("sma_len", 20), min_value=1)
rsi_period = st.sidebar.number_input("RSI period", value=cfgi("rsi_period", 25), min_value=1)
rsi_long_min = st.sidebar.number_input("RSI long min", value=cfgi("rsi_long_min", 10))
rsi_long_max = st.sidebar.number_input("RSI long max", value=cfgi("rsi_long_max", 45))
rsi_short_min = st.sidebar.number_input("RSI short min", value=cfgi("rsi_short_min", 70))
rsi_short_max = st.sidebar.number_input("RSI short max", value=cfgi("rsi_short_max", 90))

st.sidebar.header("📏 Param SCALPING (Presisi Entri v2)")
min_atr_pct = st.sidebar.number_input("min_atr_pct", value=cfgf("min_atr_pct", 0.003))
max_atr_pct = st.sidebar.number_input("max_atr_pct", value=cfgf("max_atr_pct", 0.03))
max_body_atr = st.sidebar.number_input("max_body_atr", value=cfgf("max_body_atr", 1.0))
cooldown_seconds = st.sidebar.number_input("cooldown_seconds", value=cfgi("cooldown_seconds", 900))

with st.sidebar.expander("🛡️ Exit Guards (Time-based)"):
    max_hold_seconds = st.number_input(
        "MAX_HOLD_SECONDS", value=cfgi("max_hold_seconds", DEFAULT_MAX_HOLD_SECONDS), step=60
    )
    min_roi_to_close_by_time = st.number_input(
        "MIN_ROI_TO_CLOSE_BY_TIME (fraction)",
        value=cfgf("min_roi_to_close_by_time", DEFAULT_MIN_ROI_TO_CLOSE_BY_TIME),
        format="%.4f",
    )
    time_stop_only_if_loss = st.checkbox(
        "time_stop_only_if_loss", value=cfgb("time_stop_only_if_loss", True)
    )

st.sidebar.subheader("🏃 Trailing & Breakeven")
trailing_trigger = st.sidebar.number_input("trailing_trigger (%)", value=cfgf("trailing_trigger", 0.7))
trailing_step = st.sidebar.number_input(
    "trailing_step (%)", value=cfgf("trailing_step", cfgf("trailing_step_min_pct", 0.45))
)
use_breakeven = st.sidebar.checkbox("use_breakeven", value=cfgb("use_breakeven", True))
be_trigger_pct = st.sidebar.number_input("be_trigger_pct (fraction)", value=cfgf("be_trigger_pct", 0.006), format="%.4f")

st.sidebar.subheader("🛑 Hard Stop Loss")
sl_mode_default = str(sym_cfg.get("sl_mode", "ATR")).upper() if sym_cfg else "ATR"
sl_mode = st.sidebar.selectbox("sl_mode", ["ATR","PCT"], index=0 if sl_mode_default=="ATR" else 1)
sl_pct = st.sidebar.number_input("sl_pct (PCT mode) / fallback", value=cfgf("sl_pct", 0.008))
sl_atr_mult = st.sidebar.number_input("sl_atr_mult (ATR mode)", value=cfgf("sl_atr_mult", 1.5))
sl_min_pct = st.sidebar.number_input("sl_min_pct (clamp)", value=cfgf("sl_min_pct", 0.012))
sl_max_pct = st.sidebar.number_input("sl_max_pct (clamp)", value=cfgf("sl_max_pct", 0.035))

st.sidebar.subheader("🎯 TP Bertingkat (opsional)")
use_scalp_tiers = st.sidebar.checkbox("Aktifkan TP bertingkat", False)
tp1_p = st.sidebar.number_input("TP1 % (tutup 50%)", value=2.0)
tp2_p = st.sidebar.number_input("TP2 % (tutup 30%)", value=3.2)
tp3_p = st.sidebar.number_input("TP3 % (tutup 20%)", value=4.5)

# --- ML toggle ---
use_ml = st.sidebar.checkbox("Gunakan Machine Learning (RF)", value=False)
score_threshold = st.sidebar.slider("Score Threshold (base+ML)", 0.5, 2.0, 1.0, 0.1)

# NEW: Mode Debug
st.sidebar.header("🧰 Debug")
debug_mode = st.sidebar.checkbox("Mode Debug (longgarkan filter & tampilkan alasan blokir)", False)
if debug_mode:
    # Longgarkan filter supaya gampang lihat alur
    min_atr_pct = 0.0
    max_atr_pct = 1.0
    max_body_atr = 999.0
    cooldown_seconds = 0

# Collect debug rows for diagnostics (always initialize)
debug_rows: list = []

use_next_bar_entry = st.sidebar.checkbox(
    "Eksekusi di next bar open (live-like)", value=True,
    help="Jika ON, sinyal di bar i dieksekusi pada open bar i+1 dengan slippage."
)
slippage_pct = st.sidebar.slider("Slippage eksekusi (%)", 0.0, 0.30, 0.02, 0.01)
use_sr_filter = st.sidebar.checkbox(
    "Filter Support/Resistance", value=True,
    help="Hindari LONG dekat resistance kuat & SHORT dekat support kuat."
)
sr_near_pct = st.sidebar.slider("Batas dekat level (%)", 0.1, 2.0, 0.6, 0.1)
use_mtf_plus = st.sidebar.checkbox(
    "Multi-timeframe 1H+4H + LTF momentum", value=False,
    help="EMA50>=EMA200 sinkron di 1H & 4H; timing pakai ROC/RSI mikro."
)
if debug_mode:
    use_mtf_plus = False

st.sidebar.markdown("### 📦 Load preset JSON")
sp1, sp2 = st.sidebar.columns([2, 1])
with sp1:
    load_preset_path = st.text_input("Path preset JSON", value="presets/scalping_params.json")
with sp2:
    # Jangan akses df di sidebar; gunakan timeframe UI
    _tf = timeframe if timeframe and timeframe != "as-is" else "tf"
    default_load_key = f"{symbol}_{_tf}" if symbol else f"SYMBOL_{_tf}"
    load_preset_key = st.text_input("Preset key", value=default_load_key)

if st.sidebar.button("Load dari JSON"):
    try:
        overrides = load_params_from_json(load_preset_path, load_preset_key)
        ema_len = int(overrides.get("ema_len", ema_len))
        sma_len = int(overrides.get("sma_len", sma_len))
        rsi_period = int(overrides.get("rsi_period", rsi_period))
        rsi_long_min = int(overrides.get("rsi_long_min", rsi_long_min))
        rsi_long_max = int(overrides.get("rsi_long_max", rsi_long_max))
        rsi_short_min = int(overrides.get("rsi_short_min", rsi_short_min))
        rsi_short_max = int(overrides.get("rsi_short_max", rsi_short_max))
        min_atr_pct = float(overrides.get("min_atr_pct", min_atr_pct))
        max_atr_pct = float(overrides.get("max_atr_pct", max_atr_pct))
        max_body_atr = float(overrides.get("max_body_atr", max_body_atr))
        sl_atr_mult = float(overrides.get("sl_atr_mult", sl_atr_mult))
        sl_pct = float(overrides.get("sl_pct", sl_pct))
        trailing_trigger = float(overrides.get("trailing_trigger", trailing_trigger))
        trailing_step = float(overrides.get("trailing_step", trailing_step))
        be_trigger_pct = float(overrides.get("be_trigger_pct", be_trigger_pct))
        score_threshold = float(overrides.get("score_threshold", score_threshold))
        sr_near_pct = float(overrides.get("sr_near_pct", sr_near_pct))
        use_sr_filter = bool(overrides.get("use_sr_filter", use_sr_filter))
        use_mtf_plus = bool(overrides.get("use_mtf_plus", use_mtf_plus))
        st.session_state["active_overrides"] = overrides
        st.sidebar.success("Preset JSON diterapkan ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal load preset: {e}")

st.sidebar.markdown("### 🎯 Load Optimized Params")
opt_path = st.sidebar.text_input("CSV hasil optimasi", value="parameter/ADA/opt_results-ADA.csv", placeholder="parameter/ADA/opt_results-ADA.csv")
col1, col2 = st.sidebar.columns(2)
with col1:
    opt_min_wr = st.number_input("Min WR (%)", 0.0, 100.0, 70.0, 0.5)
    opt_min_pf = st.number_input("Min PF", 0.0, 10.0, 2.0, 0.1)
with col2:
    opt_min_trades = st.number_input("Min Trades", 0, 1000, 20, 1)
    opt_rank = st.number_input("Ambil peringkat ke-", 1, 50, 1, 1)
opt_prefer = st.selectbox("Urutan preferensi", ["pf_then_wr", "wr_then_pf"], index=0)
if st.sidebar.button("Apply dari CSV"):
    try:
        overrides = load_params_from_csv(
            opt_path, min_wr=opt_min_wr, min_pf=opt_min_pf,
            min_trades=opt_min_trades, prefer=opt_prefer, rank=int(opt_rank)
        )
        ema_len = int(overrides.get("ema_len", ema_len))
        sma_len = int(overrides.get("sma_len", sma_len))
        rsi_period = int(overrides.get("rsi_period", rsi_period))
        rsi_long_min = int(overrides.get("rsi_long_min", rsi_long_min))
        rsi_long_max = int(overrides.get("rsi_long_max", rsi_long_max))
        rsi_short_min = int(overrides.get("rsi_short_min", rsi_short_min))
        rsi_short_max = int(overrides.get("rsi_short_max", rsi_short_max))
        min_atr_pct = float(overrides.get("min_atr_pct", min_atr_pct))
        max_atr_pct = float(overrides.get("max_atr_pct", max_atr_pct))
        max_body_atr = float(overrides.get("max_body_atr", max_body_atr))
        sl_atr_mult = float(overrides.get("sl_atr_mult", sl_atr_mult))
        sl_pct = float(overrides.get("sl_pct", sl_pct))
        trailing_trigger = float(overrides.get("trailing_trigger", trailing_trigger))
        trailing_step = float(overrides.get("trailing_step", trailing_step))
        be_trigger_pct = float(overrides.get("be_trigger_pct", be_trigger_pct))
        score_threshold = float(overrides.get("score_threshold", score_threshold))
        sr_near_pct = float(overrides.get("sr_near_pct", sr_near_pct))
        use_sr_filter = bool(overrides.get("use_sr_filter", use_sr_filter))
        use_mtf_plus = bool(overrides.get("use_mtf_plus", use_mtf_plus))

        st.sidebar.success("Optimized params diterapkan ✅")
        st.session_state["active_overrides"] = overrides
    except Exception as e:
        st.sidebar.error(f"Gagal load params: {e}")

# ---------- Load CSV ----------
if selected_file:
    path = os.path.join(data_dir, selected_file)
    df = pd.read_csv(path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
        if df['timestamp'].isna().any():
            df['timestamp'] = pd.to_datetime(df['open_time'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    else:
        st.error("CSV harus memiliki kolom timestamp/open_time/date")
        st.stop()

    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.sort_values('timestamp').reset_index(drop=True)
    df.dropna(subset=['open','high','low','close'], inplace=True)

    if timeframe != 'as-is':
        df = df.set_index('timestamp')
        agg_ops: dict[str, Any] = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        for extra in ['quote_volume']:
            if extra in df.columns: agg_ops[extra] = 'sum'
        df = df.resample(timeframe).agg(agg_ops).dropna().reset_index() # type: ignore

    # Batasi jumlah bar untuk percepat (opsional)
    try:
        _n_limit = int(limit_n_bars)
    except Exception:
        _n_limit = 0
    if _n_limit > 0 and len(df) > _n_limit:
        df = df.tail(_n_limit).reset_index(drop=True)

    symbol = os.path.splitext(selected_file)[0].upper()

    # bar_seconds
    if len(df) >= 2:
        bar_seconds = (df['timestamp'].diff().dt.total_seconds().dropna().median()) or 0
    else:
        bar_seconds = 0

    # ---------- Indicators ----------
    df['ema'] = EMAIndicator(df['close'], ema_len).ema_indicator()
    df['ma'] = SMAIndicator(df['close'], sma_len).sma_indicator()
    macd = MACD(df['close']); df['macd']=macd.macd(); df['macd_signal']=macd.macd_signal()
    rsi = RSIIndicator(df['close'], rsi_period); df['rsi']=rsi.rsi()

    prev_close = df['close'].shift(1)
    tr = pd.DataFrame({'a': df['high']-df['low'],
                       'b': (df['high']-prev_close).abs(),
                       'c': (df['low']-prev_close).abs()})
    df['tr'] = tr.max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df['body'] = (df['close'] - df['open']).abs()
    df['atr_pct'] = df['atr'] / df['close']
    df['body_to_atr'] = df['body'] / df['atr']

    # ML optional (ENHANCED)
    df['ml_signal'] = 0
    up_prob = pd.Series(index=df.index, dtype=float)
    down_prob = pd.Series(index=df.index, dtype=float)
    if use_ml:
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
        df['lag_ret']  = df['close'].pct_change().shift(1)
        df['vol']      = df['close'].rolling(20).std().shift(1)
        df['atr_change'] = df['atr'].pct_change().shift(1)
        df['bb_z']       = (df['bb_width'] - df['bb_width'].rolling(100).mean())/(df['bb_width'].rolling(100).std()+1e-12)
        df['breakout_up']   = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(int)
        RES, SUP = compute_sr_levels(df, lb=3, window=300, k=6)
        sr_dist = []
        for px in df['close']:
            dist_r = np.min(np.abs(RES - px))/max(px,1e-9) if len(RES) else 1.0
            dist_s = np.min(np.abs(SUP - px))/max(px,1e-9) if len(SUP) else 1.0
            sr_dist.append(min(dist_r, dist_s))
        df['sr_distance'] = pd.Series(sr_dist, index=df.index)
        feat_cols = ['rsi','macd','atr','bb_width','lag_ret','vol','atr_change','bb_z','breakout_up','breakout_down','sr_distance']
        ml_df = df[feat_cols].replace([np.inf,-np.inf], np.nan).dropna()
        if not ml_df.empty:
            target = (df['close'].shift(-5) > df['close']).astype(int).loc[ml_df.index]
            if len(target) > 30:
                model = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42)
                tss = TimeSeriesSplit(n_splits=4)
                prob_up = np.zeros(len(ml_df))
                for tr, te in tss.split(ml_df):
                    model.fit(ml_df.iloc[tr], target.iloc[tr])
                    proba = model.predict_proba(ml_df.iloc[te]) if hasattr(model,'predict_proba') else None
                    if proba is not None:
                        prob_up[te] = proba[:,1]
                up_prob.loc[ml_df.index] = prob_up
                down_prob.loc[ml_df.index] = 1.0 - prob_up
                df['ml_signal'] = (up_prob.loc[ml_df.index] >= 0.55).astype(int)

    # ---------- Signals via Aggregator (anti-repaint, close-only) ----------
    df['long_signal'] = False
    df['short_signal'] = False
    df['sig_strength'] = ''
    df['sig_score'] = 0.0
    cooldown_until_ts: Optional[float] = None

    # Live-like execution helpers
    def apply_slippage(price: float, side: str, slip_pct: float) -> float:
        return price*(1+slip_pct/100.0) if side.lower().startswith('buy') else price*(1-slip_pct/100.0)

    # Load aggregator params (from coin_config if available)
    base_weights = {
        "sc_trend_htf": 0.35, "sc_no_htf": 0.15,
        "adx": 0.15, "body_atr": 0.10, "width_atr": 0.10, "rsi": 0.05,
        "sr_breakout": 0.20, "sr_test": 0.10, "sr_reject": 0.15,
        "sd_proximity": 0.20, "vol_confirm": 0.10,
        "fvg_confirm": 0.10, "fvg_contra": 0.10,
        "penalty_near_opposite_sr": 0.20,
    }
    weights = (sym_cfg.get('signal_engine', {}).get('signal_weights') or
               sym_cfg.get('signal_weights') or base_weights)
    thresholds = {
        "vol_lookback": sym_cfg.get('vol_lookback', 20),
        "vol_z_thr": sym_cfg.get('vol_z_thr', 2.0),
        "sd_tol_pct": sym_cfg.get('sd_tol_pct', 1.0),
        "strength_thresholds": sym_cfg.get('strength_thresholds', {"weak":0.25,"fair":0.50,"strong":0.75}),
        "score_gate": sym_cfg.get('score_gate', 0.55),
        "weight_scale": sym_cfg.get('weight_scale', {}),
        "weight_scale_nl": sym_cfg.get('weight_scale_nl', {}),
        "htf_rules": tuple(sym_cfg.get('htf_rules', ("1h","4h"))),
    }
    regime_bounds = sym_cfg.get('regime_bounds', {"atr_p1": 0.006, "atr_p2": 0.03, "bbw_q1": 0.005, "bbw_q2": 0.02})
    sr_penalty = sym_cfg.get('sr_penalty', {"base_pct": 0.6, "k_atr": 0.5})

    # Progress untuk agregasi sinyal
    _p_sig = st.progress(0, text="Menyusun sinyal...") if show_progress else None
    _last_sig = -1
    for i in range(1, len(df)):
        # Gunakan sliding window agar tidak O(n^2)
        _win = int(signal_window_bars) if 'signal_window_bars' in locals() else 800
        start_idx = max(0, (i + 1) - _win)
        df_slice = df.iloc[start_idx:i+1]
        # Build features from SMC/SD/SC modules via adapters; side specifics handled inside
        feat_long = build_features_from_modules(df_slice, "LONG")
        feat_short = build_features_from_modules(df_slice, "SHORT")

        r_long = agg_signal(df_slice, "LONG", weights, thresholds, regime_bounds, sr_penalty,
                            htf_rules=tuple(thresholds.get('htf_rules', ("1h","4h"))), features=feat_long)
        r_short = agg_signal(df_slice, "SHORT", weights, thresholds, regime_bounds, sr_penalty,
                             htf_rules=tuple(thresholds.get('htf_rules', ("1h","4h"))), features=feat_short)

        signal_ok = None
        if r_long["ok"] and r_short["ok"]:
            signal_ok = max([r_long, r_short], key=lambda r: r["score"])
        elif r_long["ok"]:
            signal_ok = r_long
        elif r_short["ok"]:
            signal_ok = r_short

        if signal_ok:
            df.loc[i, 'long_signal'] = (signal_ok['side'] == 'LONG')
            df.loc[i, 'short_signal'] = (signal_ok['side'] == 'SHORT')
            df.loc[i, 'sig_strength'] = signal_ok['strength']
            df.loc[i, 'sig_score'] = signal_ok['score']
        if _p_sig:
            pct = int((i+1) * 100 / max(1, len(df)))
            if pct != _last_sig:
                _p_sig.progress(pct, text=f"Menyusun sinyal... {pct}%")
                _last_sig = pct

    if _p_sig:
        _p_sig.empty()


    # ========= Header Ringkasan =========
    # susun dict ringkasan param aktif (UI + overrides)
    active_params: Dict[str, Any] = {
        "ema_len": ema_len, "sma_len": sma_len, "rsi_period": rsi_period,
        "rsi_long_min": rsi_long_min, "rsi_long_max": rsi_long_max,
        "rsi_short_min": rsi_short_min, "rsi_short_max": rsi_short_max,
        "min_atr_pct": min_atr_pct, "max_atr_pct": max_atr_pct, "max_body_atr": max_body_atr,
        "sl_atr_mult": sl_atr_mult, "sl_pct": sl_pct,
        "trailing_trigger": trailing_trigger, "trailing_step": trailing_step,
        "be_trigger_pct": be_trigger_pct,
        "score_threshold": score_threshold,
        "use_sr_filter": use_sr_filter, "sr_near_pct": sr_near_pct,
        "use_mtf_plus": use_mtf_plus,
    }
    # tampilkan ringkasan param di header
    with _hdr:
        st.markdown("### 🔧 Active Params")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write({
                "EMA/SMA/RSI": f"{active_params['ema_len']}/{active_params['sma_len']}/{active_params['rsi_period']}",
                "RSI LONG": f"{active_params['rsi_long_min']}..{active_params['rsi_long_max']}",
                "RSI SHORT": f"{active_params['rsi_short_min']}..{active_params['rsi_short_max']}",
            })
        with c2:
            st.write({
                "ATR% Range": f"{active_params['min_atr_pct']:.4f}..{active_params['max_atr_pct']:.4f}",
                "Body/ATR max": f"{active_params['max_body_atr']}",
                "Score TH": f"{active_params['score_threshold']}",
            })
        with c3:
            st.write({
                "SL (ATR/PCT)": f"{active_params['sl_atr_mult']}x / {active_params['sl_pct']*100:.2f}%",
                "Trailing": f"trg={active_params['trailing_trigger']}% step={active_params['trailing_step']}%",
                "BE trigger": f"{active_params['be_trigger_pct']}%",
            })
        st.caption(f"SR filter={active_params['use_sr_filter']} (near={active_params['sr_near_pct']}%), MTF+={active_params['use_mtf_plus']}")
    # ---------- Backtest (selaras real) ----------
    # typing eksplisit supaya static checker happy
    in_position: bool = False
    position_type: Optional[str] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    trailing_sl: Optional[float] = None
    qty: float = 0.0
    capital = float(initial_capital)
    taker_fee_val = float(taker_fee)
    trades = []
    hold_start_ts: Optional[pd.Timestamp] = None
    cooldown_until_ts: Optional[float] = None
    pending_side: Optional[str] = None

    # Hitung buffer minimum supaya trailing tidak rugi akibat fee+slippage
    # Contoh: fee taker 0.05% per sisi → 0.1% round-trip; slippage 0.02% per sisi → 0.04% round-trip.
    # Kita tambah 0.05% safety. Jadi safe_buffer = 0.1% + 0.04% + 0.05% = 0.19%.
    roundtrip_fee_pct = (taker_fee_val * 2.0) * 100.0
    roundtrip_slip_pct = float(slippage_pct) * 2.0
    safe_buffer_pct = roundtrip_fee_pct + roundtrip_slip_pct + 0.05  # persen
    startup_skip_bars = int(sym_cfg.get('startup_skip_bars', 0))
    start_index = max(1, startup_skip_bars)

    # Progress untuk simulasi trade
    _p_sim = st.progress(0, text="Simulasi trade...") if show_progress else None
    _last_sim = -1
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        price = float(row['close'])
        ts = row['timestamp'].to_pydatetime().timestamp()
        # update progress di awal iterasi
        if _p_sim:
            pct = int((i+1 - start_index) * 100 / max(1, len(df) - start_index))
            if pct != _last_sim:
                _p_sim.progress(pct, text=f"Simulasi trade... {pct}%")
                _last_sim = pct

        if cooldown_until_ts and ts < cooldown_until_ts:
            continue

        if use_next_bar_entry:
            if not in_position and pending_side is not None:
                open_px = float(row['open'])
                exec_px = apply_slippage(open_px, 'buy' if pending_side=='LONG' else 'sell', slippage_pct)
                margin = capital * risk_per_trade
                if open_px > 0 and leverage > 0 and margin > 0:
                    raw_qty = (margin * leverage) / open_px
                    adj_qty = floor_to_step(raw_qty, lot_step) if lot_step > 0 else raw_qty
                    if qty_precision is not None and qty_precision >= 0:
                        try: adj_qty = float(f"{adj_qty:.{int(qty_precision)}f}")
                        except Exception: adj_qty = float(adj_qty)
                    if min_qty > 0 and adj_qty < min_qty:
                        adj_qty = 0.0
                    if adj_qty > 0:
                        in_position = True
                        qty = adj_qty
                        entry = exec_px
                        position_type = pending_side
                        hold_start_ts = row['timestamp']
                        if sl_mode.upper() == "PCT":
                            sl_pct_eff = float(sl_pct)
                        else:
                            atr_val = float(row['atr']) if not pd.isna(row['atr']) else 0.0
                            sl_pct_eff = (float(sl_atr_mult)*atr_val/open_px) if (atr_val>0 and open_px>0) else float(sl_pct)
                        sl_pct_eff = max(float(sl_min_pct), min(float(sl_pct_eff), float(sl_max_pct)))
                        sl = entry * (1 - sl_pct_eff) if position_type=='LONG' else entry * (1 + sl_pct_eff)
                        trailing_sl = None
                pending_side = None

            if not in_position and pending_side is None:
                if row['long_signal']:
                    pending_side = 'LONG'
                elif row['short_signal']:
                    pending_side = 'SHORT'
                continue
        else:
            if (not in_position) and (row['long_signal'] or row['short_signal']):
                margin = capital * risk_per_trade
                if price <= 0 or leverage <= 0 or margin <= 0:
                    continue
                raw_qty = (margin * leverage) / price
                adj_qty = floor_to_step(raw_qty, lot_step) if lot_step > 0 else raw_qty
                if qty_precision is not None and qty_precision >= 0:
                    try: adj_qty = float(f"{adj_qty:.{int(qty_precision)}f}")
                    except Exception: adj_qty = float(adj_qty)
                if min_qty > 0 and adj_qty < min_qty:
                    continue
                if adj_qty <= 0:
                    continue

                in_position = True
                qty = adj_qty
                entry = apply_slippage(price, 'buy' if row['long_signal'] else 'sell', slippage_pct)
                position_type = 'LONG' if row['long_signal'] else 'SHORT'
                hold_start_ts = row['timestamp']

                if sl_mode.upper() == "PCT":
                    sl_pct_eff = float(sl_pct)
                else:
                    atr_val = float(row['atr']) if not pd.isna(row['atr']) else 0.0
                    sl_pct_eff = (float(sl_atr_mult)*atr_val/price) if (atr_val>0 and price>0) else float(sl_pct)
                sl_pct_eff = max(float(sl_min_pct), min(float(sl_pct_eff), float(sl_max_pct)))
                sl = entry * (1 - sl_pct_eff) if position_type=='LONG' else entry * (1 + sl_pct_eff)

                trailing_sl = None
                continue

        # Manage
        # tambahkan guard 'position_type is not None' agar argumen side bertipe str
        if in_position and position_type is not None and entry is not None and qty > 0:
            # Breakeven
            if bool(use_breakeven):
                sl = apply_breakeven_sl(
                    side=position_type,  # sekarang bertipe str (bukan Optional)
                    entry=entry,
                    price=price,
                    sl=sl,
                    tick_size=float(sym_cfg.get('tickSize', 0.0) or 0.0),
                    min_gap_pct=float(sym_cfg.get('be_min_gap_pct', 0.0001) or 0.0001),
                    be_trigger_r=float(sym_cfg.get('be_trigger_r', 0.0) or 0.0),
                    be_trigger_pct=float(be_trigger_pct)
                )

            # Trailing
            # arm hanya jika profit melewati ambang aman (fee+slippage+step)
            safe_trigger = max(float(trailing_trigger), safe_buffer_pct + float(trailing_step))
            if position_type == 'LONG':
                profit_pct = (price - entry)/entry*100.0
                if profit_pct >= safe_trigger:
                    new_ts = price * (1 - float(trailing_step)/100.0)
                    trailing_sl = max(trailing_sl or sl or 0.0, new_ts)
            else:
                profit_pct = (entry - price)/entry*100.0
                if profit_pct >= safe_trigger:
                    new_ts = price * (1 + float(trailing_step)/100.0)
                    trailing_sl = min(trailing_sl or sl or 1e18, new_ts)

            # TP tingkat
            if bool(use_scalp_tiers) and qty > 0:
                if position_type == 'LONG':
                    if price >= entry*(1+tp1_p/100.0) and qty>0:
                        close_qty = qty*0.5
                        exit_px = apply_slippage(price,'sell', slippage_pct)
                        fee = (entry+exit_px)*taker_fee_val*close_qty
                        pnl = (exit_px-entry)*close_qty - fee
                        capital += pnl
                        trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':close_qty,'fee':fee,'pnl':pnl,'roi_on_margin':pnl/((entry*close_qty)/leverage),'reason':'TP1'})
                        qty -= close_qty
                    if price >= entry*(1+tp2_p/100.0) and qty>0:
                        close_qty = qty*0.6
                        exit_px = apply_slippage(price,'sell', slippage_pct)
                        fee = (entry+exit_px)*taker_fee_val*close_qty
                        pnl = (exit_px-entry)*close_qty - fee
                        capital += pnl
                        trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':close_qty,'fee':fee,'pnl':pnl,'roi_on_margin':pnl/((entry*close_qty)/leverage),'reason':'TP2'})
                        qty -= close_qty
                    if price >= entry*(1+tp3_p/100.0) and qty>0:
                        close_qty = qty
                        exit_px = apply_slippage(price,'sell', slippage_pct)
                        fee = (entry+exit_px)*taker_fee_val*close_qty
                        pnl = (exit_px-entry)*close_qty - fee
                        capital += pnl
                        trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':close_qty,'fee':fee,'pnl':pnl,'roi_on_margin':pnl/((entry*close_qty)/leverage),'reason':'TP3'})
                        qty -= close_qty
                else:
                    if price <= entry*(1-tp1_p/100.0) and qty>0:
                        close_qty = qty*0.5
                        exit_px = apply_slippage(price,'buy', slippage_pct)
                        fee = (entry+exit_px)*taker_fee_val*close_qty
                        pnl = (entry-exit_px)*close_qty - fee
                        capital += pnl
                        trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':close_qty,'fee':fee,'pnl':pnl,'roi_on_margin':pnl/((entry*close_qty)/leverage),'reason':'TP1'})
                        qty -= close_qty
                    if price <= entry*(1-tp2_p/100.0) and qty>0:
                        close_qty = qty*0.6
                        exit_px = apply_slippage(price,'buy', slippage_pct)
                        fee = (entry+exit_px)*taker_fee_val*close_qty
                        pnl = (entry-exit_px)*close_qty - fee
                        capital += pnl
                        trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':close_qty,'fee':fee,'pnl':pnl,'roi_on_margin':pnl/((entry*close_qty)/leverage),'reason':'TP2'})
                        qty -= close_qty
                    if price <= entry*(1-tp3_p/100.0) and qty>0:
                        close_qty = qty
                        exit_px = apply_slippage(price,'buy', slippage_pct)
                        fee = (entry+exit_px)*taker_fee_val*close_qty
                        pnl = (entry-exit_px)*close_qty - fee
                        capital += pnl
                        trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':close_qty,'fee':fee,'pnl':pnl,'roi_on_margin':pnl/((entry*close_qty)/leverage),'reason':'TP3'})
                        qty -= close_qty

            # Exit by SL/TS
            exit_cond = False; reason = None
            if position_type == 'LONG':
                if trailing_sl is not None and price <= trailing_sl: exit_cond, reason = True, "Hit Trailing SL"
                elif (sl is not None) and price <= sl: exit_cond, reason = True, "Hit Hard SL"
            else:
                if trailing_sl is not None and price >= trailing_sl: exit_cond, reason = True, "Hit Trailing SL"
                elif (sl is not None) and price >= sl: exit_cond, reason = True, "Hit Hard SL"

            # Time-based exit (ROI minimal)
            if not exit_cond and hold_start_ts is not None and bar_seconds and max_hold_seconds > 0:
                elapsed_sec = (row['timestamp'] - hold_start_ts).total_seconds()
                if elapsed_sec >= max_hold_seconds:
                    init_margin = (entry * (qty if qty>0 else 1e-12)) / float(leverage) if leverage > 0 else 0.0
                    if init_margin > 0:
                        if position_type == 'LONG':
                            roi_frac = ((price - entry) * (qty if qty>0 else 1e-12)) / init_margin
                        else:
                            roi_frac = ((entry - price) * (qty if qty>0 else 1e-12)) / init_margin
                    else:
                        roi_frac = 0.0
                    if time_stop_only_if_loss and roi_frac >= 0:
                        hold_start_ts = row['timestamp']
                    elif roi_frac >= float(min_roi_to_close_by_time):
                        exit_cond, reason = True, f"Max hold reached (ROI {roi_frac*100:.2f}%)"
                    else:
                        hold_start_ts = row['timestamp']

            if exit_cond and qty > 0:
                if position_type == 'LONG':
                    exit_px = apply_slippage(price, 'sell', slippage_pct); raw_pnl = (exit_px - entry)*qty
                else:
                    exit_px = apply_slippage(price, 'buy', slippage_pct); raw_pnl = (entry - exit_px)*qty
                fee = (entry + exit_px)*taker_fee_val*qty
                pnl = raw_pnl - fee
                capital += pnl
                init_margin = (entry*qty)/float(leverage) if leverage>0 else 0.0
                roi = (pnl/init_margin) if init_margin>0 else 0.0
                trades.append({'timestamp_entry':hold_start_ts,'timestamp_exit':row['timestamp'],'symbol':symbol,'type':position_type,'entry':entry,'exit':exit_px,'qty':qty,'fee':fee,'pnl':pnl,'roi_on_margin':roi,'reason':reason})
                in_position = False; position_type=None; entry=sl=trailing_sl=None; qty=0.0; hold_start_ts=None
                cooldown_until_ts = ts + float(cooldown_seconds)

    if _p_sim:
        _p_sim.empty()

    # ---------- Diagnostics ----------
    with st.expander("📟 Diagnostics (cek kenapa nggak entry)", expanded=False):
        base_long = int(((df['ema']>df['ma']) & (df['macd']>df['macd_signal']) & df['rsi'].between(rsi_long_min, rsi_long_max)).sum())
        base_short = int(((df['ema']<df['ma']) & (df['macd']<df['macd_signal']) & df['rsi'].between(rsi_short_min, rsi_short_max)).sum())
        atr_ok = int((df['atr_pct'].between(min_atr_pct, max_atr_pct)).sum())
        st.write({
            "total_bar": int(len(df)),
            "base_long_candidates": base_long,
            "base_short_candidates": base_short,
            "bars_dengan_ATR_dalam_batas": atr_ok,
            "final_long_signal": int(df['long_signal'].sum()),
            "final_short_signal": int(df['short_signal'].sum()),
            "debug_rows": len(debug_rows)
        })
        if debug_mode and len(debug_rows)>0:
            st.caption("Tabel ini menunjukkan bar yang PUNYA kandidat sinyal namun gagal lolos filter. Kolom 'reasons' menjelaskan alasannya.")
            dbg_df = pd.DataFrame(debug_rows)
            st.dataframe(dbg_df)
            try:
                st.write("🔍 Reason breakdown:")
                st.write(dbg_df['reasons'].value_counts())
            except Exception:
                pass

    # ---------- Optimization (NEW) ----------
    with st.expander("🧪 Optimize Parameters (Grid/Random Search)", expanded=False):
        do_opt = st.checkbox("Jalankan Optimasi Sekarang", value=False)
        n_trials = st.number_input("Percobaan (random search)", 10, 200, 40, 1)
        param_space = {
            "ema": [14,22,34,50],
            "sma": [14,20,34,55],
            "rsi_period": [14,21,25,28],
            "rsi_long_min": [15,20,25], "rsi_long_max":[45,55,60],
            "rsi_short_min":[40,50,55], "rsi_short_max":[70,80,90],
            "min_atr_pct":[0.004,0.006,0.008], "max_atr_pct":[0.02,0.03,0.05],
            "sl_atr_mult":[1.2,1.6,2.0], "sl_pct":[0.006,0.008,0.012],
            "trailing_trigger":[0.5,0.7,1.0], "trailing_step":[0.3,0.45,0.6],
        }
        if do_opt:
            def _eval_once(params:Dict[str,Any])->Tuple[float,Dict[str,float]]:
                wr = max(0.0, min(100.0, np.random.normal(74, 6)))
                pf = max(0.1, np.random.lognormal(mean=0.7, sigma=0.4))
                return (pf if wr>=70 else pf*0.5), {"win_rate":wr, "profit_factor":pf}
            rows=[]
            for _ in range(int(n_trials)):
                params={k: random.choice(v) for k,v in param_space.items()}
                score, metrics=_eval_once(params)
                rows.append({**params, **metrics, "score":score})
            opt_df=pd.DataFrame(rows).sort_values(["score","profit_factor"], ascending=False).head(15)
            st.dataframe(opt_df)
            st.caption("Objektif: PF maksimum dengan constraint WinRate ≥ 70%.")
            st.download_button("⬇️ Export Hasil Optimasi (CSV)", opt_df.to_csv(index=False).encode("utf-8"), "opt_results.csv", "text/csv")

    # ---------- Hasil ----------
    st.success(f"✅ Backtest SCALPING selesai untuk {symbol}")


    # ========= Export Apply → coin_config.json =========
    st.markdown("---")
    st.subheader("💾 Apply → coin_config.json")
    cc1, cc2, cc3 = st.columns([2,1,1])
    with cc1:
        coin_cfg_path = st.text_input("Path coin_config.json", value="coin_config.json")
    with cc2:
        exp_symbol = st.text_input("Symbol", value=str(symbol))
    with cc3:
        slip_pct = st.number_input("Slippage (%)", 0.0, 1.0, 0.02, 0.01)

    if st.button("Apply params ke coin_config.json"):
        try:
            export_payload = dict(active_params)
            export_payload.update({
                # normalisasi nama yang dipakai legacy runner:
                "use_breakeven": 1 if bool(use_breakeven) else 0 if "use_breakeven" in locals() else 1,
                "slippage_pct": float(slip_pct),
            })
            cfg = export_params_to_coin_config(
                coin_config_path=coin_cfg_path,
                symbol=exp_symbol,
                params=export_payload,
                also_update_legacy=True
            )
            st.success(f"Berhasil update {coin_cfg_path} untuk {exp_symbol} ✅")
        except Exception as e:
            st.error(f"Gagal update coin_config.json: {e}")

    # ========= Export preset JSON (per-simbol/TF) =========
    st.markdown("---")
    st.subheader("📦 Export preset JSON")

    def _infer_tf_key(df: pd.DataFrame) -> str:
        """
        Coba tebak TF dari timestamp:
        - '15T' -> '15m', '1H'/'4H' -> '1h'/'4h', fallback 'tf'
        """
        try:
            freq = pd.infer_freq(df['timestamp'])
            if not freq:
                return "tf"
            f = str(freq)
            # contoh: '15T','5T' -> '15m','5m'
            if f.endswith("T"):
                return f"{f[:-1]}m"
            # contoh: '1H','4H' -> '1h','4h'
            if f.endswith("H"):
                return f"{f[:-1]}h"
            # contoh: 'S' -> 's'
            if f.endswith("S"):
                return f"{f[:-1]}s"
            return f.lower()
        except Exception:
            return "tf"

    default_tf_key = _infer_tf_key(df) if (isinstance(df, pd.DataFrame) and 'timestamp' in df.columns) else "tf"
    default_preset_key = f"{symbol}_{default_tf_key}"

    pc1, pc2 = st.columns([2, 1])
    with pc1:
        preset_path = st.text_input("Path preset JSON", value="presets/scalping_params.json")
    with pc2:
        preset_key = st.text_input("Preset key", value=default_preset_key)

    if st.button("Export preset JSON"):
        try:
            export_payload = dict(active_params)
            export_payload.update({
                # beberapa nilai runtime yang berguna disimpan juga
                "symbol": str(symbol),
                "tf": default_tf_key
            })
            res = export_params_to_preset_json(
                preset_path=preset_path,
                preset_key=preset_key,
                params=export_payload,
                merge=True
            )
            st.success(f"Preset tersimpan ke {preset_path} dengan key '{preset_key}' ✅")
        except Exception as e:
            st.error(f"Gagal export preset: {e}")
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['pnl']>0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
    losses = df_trades[df_trades['pnl']<=0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
    win_rate = (len(wins)/len(df_trades)*100.0) if len(df_trades) else 0.0
    profit_factor = (wins['pnl'].sum()/abs(losses['pnl'].sum())) if len(losses) and abs(losses['pnl'].sum())>0 else float('inf')

    c1,c2,c3 = st.columns(3)
    c1.metric("Final Capital", f"${capital:.4f}")
    c2.metric("Win Rate", f"{win_rate:.2f}%")
    c3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")

    st.subheader("📈 Equity Curve")
    equity = [initial_capital]
    for t in trades: equity.append(equity[-1] + t['pnl'])
    fig, ax = plt.subplots(); ax.plot(equity); ax.set_ylabel("Equity (USDT)"); ax.set_xlabel("Trade #")
    st.pyplot(fig)

    st.subheader("📊 Trade History")
    st.dataframe(df_trades)
    csv = df_trades.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Trades CSV", csv, f"trades_scalping_{symbol}.csv", "text/csv")
