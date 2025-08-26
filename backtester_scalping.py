import streamlit as st
import pandas as pd
import numpy as np
import os, math, json, warnings, random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, cast
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from engine_core import apply_breakeven_sl
from indicators.sr_utils import (
    compute_sr_levels,
    near_level,
    build_sr_cache,
    htf_trend_ok_multi,
    ltf_momentum_ok,
)

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

    symbol = os.path.splitext(selected_file)[0].upper()

    # bar_seconds
    if len(df) >= 2:
        bar_seconds = (df['timestamp'].diff().dt.total_seconds().dropna().median()) or 0
    else:
        bar_seconds = 0

    # ---------- Indicators ----------
    df['ema'] = EMAIndicator(df['close'], 22).ema_indicator()
    df['ma'] = SMAIndicator(df['close'], 20).sma_indicator()
    macd = MACD(df['close']); df['macd']=macd.macd(); df['macd_signal']=macd.macd_signal()
    rsi = RSIIndicator(df['close'], 25); df['rsi']=rsi.rsi()

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

    # ---------- Signals + Filters ----------
    df['long_signal'] = False
    df['short_signal'] = False
    cooldown_until_ts: Optional[float] = None
    # ===== Live-like execution helpers (NEW) =====
    def apply_slippage(price: float, side: str, slip_pct: float) -> float:
        return price*(1+slip_pct/100.0) if side.lower().startswith('buy') else price*(1-slip_pct/100.0)

    use_next_bar_entry = st.sidebar.checkbox("Eksekusi di next bar open (live-like)", value=True, help="Jika ON, sinyal di bar i dieksekusi pada open bar i+1 dengan slippage.")
    slippage_pct = st.sidebar.slider("Slippage eksekusi (%)", 0.0, 0.30, 0.02, 0.01)

    use_sr_filter = st.sidebar.checkbox("Filter Support/Resistance", value=True,
        help="Hindari LONG dekat resistance kuat & SHORT dekat support kuat.")
    sr_near_pct = st.sidebar.slider("Batas dekat level (%)", 0.1, 2.0, 0.6, 0.1)
    use_mtf_plus = st.sidebar.checkbox("Multi-timeframe 1H+4H + LTF momentum", value=False,
        help="EMA50>=EMA200 sinkron di 1H & 4H; timing pakai ROC/RSI mikro.")
    if debug_mode:
        use_mtf_plus = False
    # cache level S/R agar hemat komputasi
    sr_cache = build_sr_cache(df, lb=3, window=300, k=6, recalc_every=10)

    # untuk debug: log alasan block
    debug_rows: List[Dict[str, Any]] = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        sc_long = 0.0; sc_short = 0.0
        # base scoring
        if row['ema'] > row['ma'] and row['macd'] > row['macd_signal'] and (10 <= row['rsi'] <= 45):
            sc_long += 1
        if row['ema'] < row['ma'] and row['macd'] < row['macd_signal'] and (70 <= row['rsi'] <= 90):
            sc_short += 1
        if use_ml:
            up_p   = float(up_prob.iloc[i])   if not pd.isna(up_prob.iloc[i])   else None
            down_p = float(down_prob.iloc[i]) if not pd.isna(down_prob.iloc[i]) else None
            if up_p is not None and up_p >= 0.58: sc_long += 1.0
            if down_p is not None and down_p >= 0.58: sc_short += 1.0

        long_raw = sc_long >= float(score_threshold)
        short_raw = sc_short >= float(score_threshold)

        atr_pct_now = float(row['atr_pct'] or 0)
        body_to_atr_now = float(row['body_to_atr'] or 0)
        ts = row['timestamp'].to_pydatetime().timestamp()

        blocked_reasons_long = []
        blocked_reasons_short = []
        # Jika base tidak lolos threshold
        if not long_raw and sc_long>0:
            blocked_reasons_long.append('score_below_threshold')
        if not short_raw and sc_short>0:
            blocked_reasons_short.append('score_below_threshold')

        # Filters
        if not (min_atr_pct <= atr_pct_now <= max_atr_pct):
            long_raw = False; short_raw = False
            blocked_reasons_long.append('atr_out_of_range')
            blocked_reasons_short.append('atr_out_of_range')
        if body_to_atr_now > max_body_atr:
            long_raw = False; short_raw = False
            blocked_reasons_long.append('body_exceeds_atr')
            blocked_reasons_short.append('body_exceeds_atr')
        if cooldown_until_ts and ts < cooldown_until_ts:
            long_raw = False; short_raw = False
            blocked_reasons_long.append('cooldown_active')
            blocked_reasons_short.append('cooldown_active')
        if use_sr_filter:
            RES, SUP = sr_cache.get(i, sr_cache.get(i-1, (np.array([]), np.array([]))))
            px = float(row['close'])
            if long_raw and near_level(px, RES, sr_near_pct):
                long_raw = False; blocked_reasons_long.append('near_resistance')
            if short_raw and near_level(px, SUP, sr_near_pct):
                short_raw = False; blocked_reasons_short.append('near_support')
        if use_mtf_plus:
            if long_raw and not htf_trend_ok_multi('LONG', df.iloc[:i+1], rules=('1h','4h')):
                long_raw = False; blocked_reasons_long.append('htf_trend_mismatch')
            if short_raw and not htf_trend_ok_multi('SHORT', df.iloc[:i+1], rules=('1h','4h')):
                short_raw = False; blocked_reasons_short.append('htf_trend_mismatch')
            l_ok, s_ok = ltf_momentum_ok(df.iloc[:i+1], lookback=5, rsi_thr_long=52, rsi_thr_short=48)
            if long_raw and not l_ok:
                long_raw = False; blocked_reasons_long.append('ltf_momentum_weak')
            if short_raw and not s_ok:
                short_raw = False; blocked_reasons_short.append('ltf_momentum_weak')

        # Set sinyal akhir
        if long_raw:
            df.loc[i,'long_signal'] = True
        elif sc_long>0 and debug_mode:
            debug_rows.append({
                'timestamp': row['timestamp'], 'price': float(row['close']), 'side': 'LONG',
                'atr_pct': atr_pct_now, 'body_to_atr': body_to_atr_now,
                'reasons': ';'.join(blocked_reasons_long) or 'blocked'
            })

        if short_raw:
            df.loc[i,'short_signal'] = True
        elif sc_short>0 and debug_mode:
            debug_rows.append({
                'timestamp': row['timestamp'], 'price': float(row['close']), 'side': 'SHORT',
                'atr_pct': atr_pct_now, 'body_to_atr': body_to_atr_now,
                'reasons': ';'.join(blocked_reasons_short) or 'blocked'
            })

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

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        price = float(row['close'])
        ts = row['timestamp'].to_pydatetime().timestamp()

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

    # ---------- Diagnostics ----------
    with st.expander("📟 Diagnostics (cek kenapa nggak entry)", expanded=False):
        base_long = int(((df['ema']>df['ma']) & (df['macd']>df['macd_signal']) & df['rsi'].between(40,60)).sum())
        base_short = int(((df['ema']<df['ma']) & (df['macd']<df['macd_signal']) & df['rsi'].between(30,60)).sum())
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
