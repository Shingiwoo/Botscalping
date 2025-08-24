import streamlit as st
import pandas as pd
import os
import math
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

"""
============================================================
APP: Scalping & Swing Backtester (Selaras Real Trading)
FILE: app_fixscalping.py
UPDATE: 2025-08-10 (Mode Swing/AnalisaBot dari Analisa bot.txt)
============================================================

🔄 PERUBAHAN UTAMA (RINGKAS):
1) Tambah "Mode Strategi":
   - Scalping (logika lama, 1 simbol)
   - Swing / AnalisaBot (sesuai Analisa bot.txt → hold 2-4 hari)

2) Data historis tetap dari CSV (satu file per simbol). Pilihan resample timeframe (15m/1H/4H/1D) agar cocok untuk swing.

3) Aturan Swing yang diimplementasikan:
   - MA50 vs MA200 (trend), optional filter EMA255.
   - RSI(14) rebound/cross dari oversold/overbought.
   - Volume spike > X% dari rata-rata N bar.
   - Breakout level (close > rolling high / < rolling low) + konfirmasi beberapa bar.
   - Konfirmasi tambahan: MACD histogram cross, pola candlestick engulfing (sederhana).
   - Risk/Position size: margin = balance * risk_per_trade, qty = (margin * leverage) / price (LOT_SIZE aware).
   - SL dinamis (3% dari swing high/low) + Trailing aktif setelah profit >= trigger %.
   - Take-Profit bertingkat: 5%/8%/12% (50%/30%/20% posisi) dengan fee taker diperhitungkan.
   - Time-based exit: auto close di akhir hari ke-4 (≈ 4*24 jam).
   - Force exit: RSI ekstrim berbalik atau volume turun > 40% dari volume entry rata-rata.

4) Hasil backtest disajikan: metrics, equity curve, dan riwayat trade. Dapat diunduh CSV.

CATATAN:
- Screening 100 coin & top 6 posisi dari Analisa bot.txt tidak diaktifkan di UI ini (karena fokus 1 file CSV/simbol). Namun, seluruh logika entry/exit swing sudah sesuai.
- Jika Anda ingin mode screener multi-CSV (100 coin) nanti, tinggal menambahkan loop folder & ranking skor pada level simbol.

============================================================
"""

# =============================
#  Helper & Defaults (match realtrading.py)
# =============================
def floor_to_step(x: float, step: float) -> float:
    if step is None or step <= 0:
        return float(x)
    return math.floor(float(x) / float(step)) * float(step)

DEFAULT_TAKER_FEE = float(os.getenv('TAKER_FEE', '0.0005'))  # 0.05%
DEFAULT_MIN_ROI_TO_CLOSE_BY_TIME = float(os.getenv('MIN_ROI_TO_CLOSE_BY_TIME', '0.05'))  # 5% ROI (fraction)
DEFAULT_MAX_HOLD_SECONDS = int(os.getenv('MAX_HOLD_SECONDS', '3600'))  # 1 jam (untuk scalping)

st.set_page_config(page_title="Scalping/Swing Backtester (RealLogic)", layout="wide")
st.title("📈 Backtester — Scalping & Swing (AnalisaBot)")

# === Sidebar Settings ===
st.sidebar.header("⚙️ Mode & Data")
strategi_mode = st.sidebar.radio("Pilih Mode Strategi", ["Scalping", "Swing / AnalisaBot"], index=0)

data_dir = st.sidebar.text_input("Folder Data CSV", value="./data")
try:
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
except Exception:
    csv_files = []
selected_file = st.selectbox("Pilih data file (1 simbol per backtest)", options=csv_files)

st.sidebar.header("🕒 Timeframe")
timeframe = st.sidebar.selectbox("Resample Timeframe", ["as-is", "15T", "1H", "4H", "1D"], index=0,
                                 help="Gunakan 4H/1D untuk swing. 'as-is' mengikuti timeframe CSV.")

st.sidebar.header("💰 Money Management")
initial_capital = st.sidebar.number_input("Available Balance (USDT)", value=20.0, min_value=0.0, step=1.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1.0, 10.0, 8.0) / 100.0
leverage = st.sidebar.slider("Leverage", 1, 125, 15)
taker_fee = st.sidebar.number_input("Taker Fee", value=DEFAULT_TAKER_FEE, format="%.6f", help="Contoh: 0.0005 = 0.05%")
slippage_pct = st.sidebar.slider("Slippage (%)", 0.0, 0.3, 0.02, step=0.01, help="Disimulasikan pada harga eksekusi.")

with st.sidebar.expander("⚙️ LOT_SIZE & Precision"):
    lot_step = st.number_input("stepSize (LOT_SIZE)", value=0.0, min_value=0.0, format="%.10f")
    min_qty = st.number_input("minQty (LOT_SIZE)", value=0.0, min_value=0.0, format="%.10f")
    qty_precision = st.number_input("quantityPrecision", value=0, min_value=0, max_value=8, step=1)

# === SCALPING parameters (mode lama) ===
if strategi_mode == "Scalping":
    st.sidebar.header("📏 Param Scalping (lama)")
    sma_period = st.sidebar.slider("SMA Period", 7, 50, 20)
    ema_period = st.sidebar.slider("EMA Period", 5, 50, 22)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    bb_std = st.sidebar.slider("Bollinger Band StdDev", 1.0, 3.0, 1.4, step=0.1)
    score_threshold = st.sidebar.slider("Minimal Skor Sinyal", 1.0, 5.0, 2.0, step=0.1)

    with st.sidebar.expander("🛡️ Exit Guards (Time-based)"):
        max_hold_seconds = st.number_input("MAX_HOLD_SECONDS", value=DEFAULT_MAX_HOLD_SECONDS, step=60)
        min_roi_to_close_by_time = st.number_input("MIN_ROI_TO_CLOSE_BY_TIME (fraction)", value=DEFAULT_MIN_ROI_TO_CLOSE_BY_TIME, format="%.4f", help="0.05 = 5% ROI terhadap initial margin")

    use_trailing = st.sidebar.checkbox("Gunakan Trailing SL", True)
    trailing_trigger = st.sidebar.slider("Trigger Trailing SL (%)", 0.1, 3.0, 0.5)
    trailing_step = st.sidebar.slider("Step Trailing SL (%)", 0.05, 2.0, 0.3)

# === SWING parameters (AnalisaBot) ===
if strategi_mode == "Swing / AnalisaBot":
    st.sidebar.header("📏 Param Swing / AnalisaBot")
    ma_fast = st.sidebar.number_input("MA Fast (MA50)", value=50, min_value=5)
    ma_slow = st.sidebar.number_input("MA Slow (MA200)", value=200, min_value=20)
    ema_trend = st.sidebar.number_input("EMA Trend Filter (EMA255)", value=255, min_value=50)
    rsi_period_s = st.sidebar.number_input("RSI Period", value=14, min_value=5)
    rsi_os = st.sidebar.slider("RSI Oversold", 10, 45, 40)
    rsi_ob = st.sidebar.slider("RSI Overbought", 55, 90, 70)
    vol_window = st.sidebar.number_input("Window Rata-rata Volume (bar)", value=42, min_value=5)
    vol_spike_mult = st.sidebar.slider("Volume Spike Multiplier", 1.0, 3.0, 1.5, step=0.1)
    breakout_lookback = st.sidebar.number_input("Breakout Lookback (bar)", value=50, min_value=5)
    confirm_bars = st.sidebar.number_input("Konfirmasi Close Beruntun (bar)", value=4, min_value=1)
    score_th_swing = st.sidebar.slider("Minimal Skor Sinyal", 1.0, 5.0, 2.5, step=0.1)

    use_trailing_s = st.sidebar.checkbox("Gunakan Trailing SL (aktif setelah profit)", True)
    trigger_p = st.sidebar.slider("Trigger Profit Trailing (%)", 1.0, 10.0, 5.0)
    step_p = st.sidebar.slider("Step Trailing (%)", 0.2, 5.0, 1.0)

    st.sidebar.subheader("🎯 Take Profit Bertingkat")
    tp1_p = st.sidebar.number_input("TP1 % (tutup 50%)", value=5.0)
    tp2_p = st.sidebar.number_input("TP2 % (tutup 30%)", value=8.0)
    tp3_p = st.sidebar.number_input("TP3 % (tutup 20%)", value=12.0)

    st.sidebar.subheader("🛡️ Exit Rules Lainnya")
    swing_hold_days = st.sidebar.slider("Maks Hold (hari)", 1, 10, 4)
    force_rsi_reversal = st.sidebar.checkbox("Force Exit pada RSI reversal ekstrem", True)
    vol_drop_pct = st.sidebar.slider("Force Exit jika volume turun > (%)", 10, 80, 40)

# === Load CSV ===
if selected_file:
    path = os.path.join(data_dir, selected_file)
    df = pd.read_csv(path)

    # kolom standar minimal: timestamp, open, high, low, close, volume
    # dukung beberapa nama timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='ignore')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    else:
        st.error("CSV harus memiliki kolom timestamp/open_time/date")
        st.stop()

    # pastikan tipe numerik utama
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.sort_values('timestamp').reset_index(drop=True)
    df.dropna(subset=['open','high','low','close'], inplace=True)

    # Resample jika dipilih
    if timeframe != 'as-is':
        df = df.set_index('timestamp')
        agg = {
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last',
            'volume':'sum'
        }
        for extra in ['quote_volume']:
            if extra in df.columns:
                agg[extra] = 'sum'
        df = df.resample(timeframe).agg(agg).dropna().reset_index()

    symbol = os.path.splitext(selected_file)[0].upper()

    # Hitung bar_seconds (untuk fitur time-based)
    if len(df) >= 2:
        bar_seconds = (df['timestamp'].diff().dt.total_seconds().dropna().median()) or 0
    else:
        bar_seconds = 0

    # ===================
    # MODE: SCALPING (lama)
    # ===================
    if strategi_mode == "Scalping":
        # === Apply Indicators (identik secara logika dgn real) ===
        df['ema'] = EMAIndicator(df['close'], ema_period).ema_indicator()
        df['ma'] = SMAIndicator(df['close'], sma_period).sma_indicator()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = RSIIndicator(df['close'], rsi_period).rsi()

        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # === ML Signal ala realtrading.py ===
        df['lag_ret'] = df['close'].pct_change().shift(1)
        df['vol'] = df['close'].rolling(20).std().shift(1)
        ml_df = df[['rsi', 'macd', 'atr', 'bb_width', 'lag_ret', 'vol']].dropna()
        df['ml_signal'] = 0
        if not ml_df.empty:
            target = (df['close'].shift(-5) > df['close']).astype(int).loc[ml_df.index]
            if len(target) > 0:
                model = RandomForestClassifier(n_estimators=100)
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                preds = np.zeros(len(ml_df))
                for tr, _ in skf.split(ml_df, target):
                    model.fit(ml_df.iloc[tr], target.iloc[tr])
                    preds[tr] = model.predict(ml_df.iloc[tr])
                df.loc[ml_df.index, 'ml_signal'] = preds

        # === Generate step-wise signals (scoring) ===
        df['long_signal'] = False
        df['short_signal'] = False
        for i in range(1, len(df)):
            sc_long = 0.0
            if df['ema'].iloc[i] > df['ma'].iloc[i] and df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                sc_long += 1
            if 40 < df['rsi'].iloc[i] < 70:
                sc_long += 0.5
            if df['ml_signal'].iloc[i] == 1:
                sc_long += 1
            if sc_long >= score_threshold:
                df.loc[i, 'long_signal'] = True

            sc_short = 0.0
            if df['ema'].iloc[i] < df['ma'].iloc[i] and df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                sc_short += 1
            if 30 < df['rsi'].iloc[i] < 60:
                sc_short += 0.5
            if df['ml_signal'].iloc[i] == 0:
                sc_short += 1
            if sc_short >= score_threshold:
                df.loc[i, 'short_signal'] = True

        # === Backtest — logic disamakan dengan realtrading.py ===
        in_position = False
        position_type = None  # 'LONG' / 'SHORT'
        entry = sl = tp = trailing_sl = None
        qty = 0.0
        capital = float(initial_capital)
        taker_fee_val = float(taker_fee)

        trades = []
        long_count = short_count = 0
        hold_start_ts = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            price = float(row['close'])
            # aplikasikan slippage pada eksekusi
            exec_price_buy = price * (1 + slippage_pct/100)
            exec_price_sell = price * (1 - slippage_pct/100)
            ts = row['timestamp']

            if not in_position:
                if row['long_signal'] or row['short_signal']:
                    margin = capital * risk_per_trade
                    if price <= 0 or leverage <= 0 or margin <= 0:
                        continue
                    raw_qty = (margin * leverage) / price
                    adj_qty = floor_to_step(raw_qty, lot_step) if lot_step > 0 else raw_qty
                    if qty_precision is not None and qty_precision >= 0:
                        try:
                            adj_qty = float(f"{adj_qty:.{int(qty_precision)}f}")
                        except Exception:
                            adj_qty = float(adj_qty)
                    if min_qty > 0 and adj_qty < min_qty:
                        continue
                    if adj_qty <= 0:
                        continue

                    in_position = True
                    qty = adj_qty
                    entry = price
                    position_type = 'LONG' if row['long_signal'] else 'SHORT'
                    hold_start_ts = ts

                    atr_val = float(row['atr']) if not pd.isna(row['atr']) else None
                    if atr_val is not None and atr_val > 0:
                        if position_type == 'LONG':
                            sl = entry - atr_val * 1.5
                            tp = entry + atr_val * 3.0
                        else:
                            sl = entry + atr_val * 1.5
                            tp = entry - atr_val * 3.0
                        trailing_sl = sl
                    else:
                        if position_type == 'LONG':
                            sl = entry * 0.99
                            tp = entry * 1.02
                        else:
                            sl = entry * 1.01
                            tp = entry * 0.98
                        trailing_sl = sl

                    if position_type == 'LONG':
                        long_count += 1
                    else:
                        short_count += 1
                    continue

            # ====== Dalam posisi ======
            if in_position and entry is not None and qty > 0:
                if use_trailing:
                    if position_type == 'LONG':
                        profit_pct = (price - entry) / entry * 100.0
                        if profit_pct >= trailing_trigger:
                            new_ts = price - (trailing_step / 100.0) * price
                            trailing_sl = max(trailing_sl, new_ts) if trailing_sl is not None else new_ts
                    else:
                        profit_pct = (entry - price) / entry * 100.0
                        if profit_pct >= trailing_trigger:
                            new_ts = price + (trailing_step / 100.0) * price
                            trailing_sl = min(trailing_sl, new_ts) if trailing_sl is not None else new_ts

                exit_cond = False
                reason = None
                if position_type == 'LONG':
                    if trailing_sl is not None and price <= trailing_sl:
                        exit_cond, reason = True, "Hit Trailing SL"
                    elif tp is not None and price >= tp:
                        exit_cond, reason = True, "Hit Take Profit"
                else:
                    if trailing_sl is not None and price >= trailing_sl:
                        exit_cond, reason = True, "Hit Trailing SL"
                    elif tp is not None and price <= tp:
                        exit_cond, reason = True, "Hit Take Profit"

                if not exit_cond and hold_start_ts is not None and bar_seconds and max_hold_seconds > 0:
                    elapsed_sec = (ts - hold_start_ts).total_seconds()
                    if elapsed_sec >= max_hold_seconds:
                        if position_type == 'LONG':
                            unreal = (price - entry) * qty
                        else:
                            unreal = (entry - price) * qty
                        init_margin = (entry * qty) / float(leverage) if leverage > 0 else 0.0
                        roi = (unreal / init_margin) if init_margin > 0 else 0.0
                        if roi >= float(min_roi_to_close_by_time):
                            exit_cond, reason = True, f"Max hold reached (ROI {roi*100:.2f}%)"
                        else:
                            hold_start_ts = ts

                if exit_cond:
                    if position_type == 'LONG':
                        raw_pnl = (exec_price_sell - entry) * qty
                    else:
                        raw_pnl = (entry - exec_price_buy) * qty
                    fee = (entry + price) * taker_fee_val * qty
                    pnl = raw_pnl - fee
                    capital += pnl

                    init_margin = (entry * qty) / float(leverage) if leverage > 0 else 0.0
                    roi = (pnl / init_margin) if init_margin > 0 else 0.0

                    trades.append({
                        'timestamp_entry': hold_start_ts,
                        'timestamp_exit': ts,
                        'symbol': symbol,
                        'type': position_type,
                        'entry': entry,
                        'exit': price,
                        'qty': qty,
                        'fee': fee,
                        'pnl': pnl,
                        'roi_on_margin': roi,
                        'reason': reason
                    })

                    in_position = False
                    position_type = None
                    entry = sl = tp = trailing_sl = None
                    qty = 0.0
                    hold_start_ts = None

        # === Result Display ===
        st.success(f"✅ Backtest selesai untuk {symbol}: {len(trades)} trade(s)")
        df_trades = pd.DataFrame(trades)
        wins = df_trades[df_trades['pnl'] > 0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
        losses = df_trades[df_trades['pnl'] <= 0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
        win_rate = (len(wins) / len(df_trades) * 100.0) if len(df_trades) else 0.0
        profit_factor = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) and abs(losses['pnl'].sum()) > 0 else float('inf')

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Capital", f"${capital:.4f}")
        col2.metric("Win Rate", f"{win_rate:.2f}%")
        col3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")
        col4, col5 = st.columns(2)
        col4.metric("Jumlah Long Trades", int((df_trades['type'] == 'LONG').sum()))
        col5.metric("Jumlah Short Trades", int((df_trades['type'] == 'SHORT').sum()))

        # Equity Curve
        st.subheader("📈 Equity Curve")
        equity = [initial_capital]
        for t in trades:
            equity.append(equity[-1] + t['pnl'])
        fig, ax = plt.subplots()
        ax.plot(equity)
        ax.set_ylabel("Equity (USDT)")
        ax.set_xlabel("Trade #")
        st.pyplot(fig)

        # Trade History
        st.subheader("📊 Trade History")
        st.dataframe(df_trades)
        csv = df_trades.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Trades CSV", csv, f"trades_{symbol}.csv", "text/csv")

    # ===================
    # MODE: SWING / ANALISABOT
    # ===================
    if strategi_mode == "Swing / AnalisaBot":
        # ---------------- Indicators ----------------
        df['ma_fast'] = SMAIndicator(df['close'], ma_fast).sma_indicator()
        df['ma_slow'] = SMAIndicator(df['close'], ma_slow).sma_indicator()
        df['ema_trend'] = EMAIndicator(df['close'], ema_trend).ema_indicator()
        rsi_obj = RSIIndicator(df['close'], rsi_period_s)
        df['rsi'] = rsi_obj.rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # Volume metrics
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        df['vol_ma'] = df['volume'].rolling(int(vol_window)).mean()
        df['vol_spike'] = df['volume'] > (vol_spike_mult * df['vol_ma'])

        # Breakout levels
        df['roll_high'] = df['high'].rolling(int(breakout_lookback)).max()
        df['roll_low'] = df['low'].rolling(int(breakout_lookback)).min()

        # Candlestick engulfing (sederhana)
        def bullish_engulf(i):
            if i < 1:
                return False
            prev_o, prev_c = df['open'].iloc[i-1], df['close'].iloc[i-1]
            o, c = df['open'].iloc[i], df['close'].iloc[i]
            return (prev_c < prev_o) and (c > o) and (c >= prev_o) and (o <= prev_c)
        def bearish_engulf(i):
            if i < 1:
                return False
            prev_o, prev_c = df['open'].iloc[i-1], df['close'].iloc[i-1]
            o, c = df['open'].iloc[i], df['close'].iloc[i]
            return (prev_c > prev_o) and (c < o) and (c <= prev_o) and (o >= prev_c)

        # ---------------- Sinyal / Skoring ----------------
        df['long_signal'] = False
        df['short_signal'] = False
        long_confirm = 0
        short_confirm = 0

        for i in range(len(df)):
            sc_long = 0.0
            sc_short = 0.0
            # Trend MA
            if df['ma_fast'].iloc[i] > df['ma_slow'].iloc[i]:
                sc_long += 1
            if df['ma_fast'].iloc[i] < df['ma_slow'].iloc[i]:
                sc_short += 1
            # Trend filter EMA 255 (opsional, dihitung sama; bobot kecil)
            if df['close'].iloc[i] > df['ema_trend'].iloc[i]:
                sc_long += 0.25
            else:
                sc_short += 0.25
            # RSI rebound / rejection
            if i >= 1:
                prev_rsi = df['rsi'].iloc[i-1]
                rsi_now = df['rsi'].iloc[i]
                # Rebound dari oversold → cross up level rsi_os
                if prev_rsi < rsi_os and rsi_now >= rsi_os:
                    sc_long += 1
                # Rejection dari overbought → cross down level rsi_ob
                if prev_rsi > rsi_ob and rsi_now <= rsi_ob:
                    sc_short += 1
            # Volume spike
            if bool(df['vol_spike'].iloc[i]):
                sc_long += 0.5
                sc_short += 0.5  # spike mendukung breakout dua arah, nilai kecil
            # Breakout
            if df['close'].iloc[i] > df['roll_high'].iloc[i]:
                long_confirm += 1
                short_confirm = 0
            elif df['close'].iloc[i] < df['roll_low'].iloc[i]:
                short_confirm += 1
                long_confirm = 0
            else:
                long_confirm = max(0, long_confirm-1)
                short_confirm = max(0, short_confirm-1)

            if long_confirm >= int(confirm_bars):
                sc_long += 1
            if short_confirm >= int(confirm_bars):
                sc_short += 1

            # MACD histogram cross
            if i >= 1:
                if df['macd_hist'].iloc[i-1] <= 0 and df['macd_hist'].iloc[i] > 0:
                    sc_long += 0.5
                if df['macd_hist'].iloc[i-1] >= 0 and df['macd_hist'].iloc[i] < 0:
                    sc_short += 0.5

            # Candlestick pattern
            if bullish_engulf(i):
                sc_long += 0.5
            if bearish_engulf(i):
                sc_short += 0.5

            # Threshold
            if sc_long >= float(score_th_swing):
                df.loc[i, 'long_signal'] = True
            if sc_short >= float(score_th_swing):
                df.loc[i, 'short_signal'] = True

        # ---------------- Backtest Swing ----------------
        in_position = False
        position_type = None
        entry_price = None
        entry_ts = None
        entry_vol_ma = None
        qty = 0.0
        capital = float(initial_capital)
        taker_fee_val = float(taker_fee)

        trailing_sl = None
        sl = None

        trades = []

        def apply_slippage(px: float, side: str) -> float:
            # side: 'buy' or 'sell'
            if side == 'buy':
                return px * (1 + slippage_pct/100)
            else:
                return px * (1 - slippage_pct/100)

        def add_trade(timestamp_entry, timestamp_exit, side, entry, exit_px, q, reason, portion=1.0):
            capital_box = [initial_capital]
            # fee dua sisi (entry+exit) pada taker fee
            fee = (entry + exit_px) * taker_fee_val * q
            if side == 'LONG':
                pnl = (exit_px - entry) * q - fee
            else:
                pnl = (entry - exit_px) * q - fee
            capital_box[0] += pnl
            init_margin = (entry * q) / float(leverage) if leverage > 0 else 0.0
            roi = (pnl / init_margin) if init_margin > 0 else 0.0
            trades.append({
	        'timestamp_entry': timestamp_entry,
	        'timestamp_exit': timestamp_exit,
	        'symbol': symbol,
	        'type': side,
	        'entry': entry,
	        'exit': exit_px,
	        'qty': q,
	        'portion': portion,
	        'fee': fee,
	        'pnl': pnl,
	        'roi_on_margin': roi,
	        'reason': reason
	        })            
            return pnl
        
        max_hold_seconds_swing = int(swing_hold_days * 86400)
        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = row['timestamp']
            price = float(row['close'])

            # buka posisi?
            if not in_position and (row['long_signal'] or row['short_signal']):
                margin = capital * risk_per_trade
                if price <= 0 or leverage <= 0 or margin <= 0:
                    continue
                raw_qty = (margin * leverage) / price
                adj_qty = floor_to_step(raw_qty, lot_step) if lot_step > 0 else raw_qty
                if qty_precision is not None and qty_precision >= 0:
                    try:
                        adj_qty = float(f"{adj_qty:.{int(qty_precision)}f}")
                    except Exception:
                        adj_qty = float(adj_qty)
                if min_qty > 0 and adj_qty < min_qty:
                    continue
                if adj_qty <= 0:
                    continue

                in_position = True
                qty = adj_qty
                entry_price = price
                entry_ts = ts
                position_type = 'LONG' if row['long_signal'] else 'SHORT'
                entry_vol_ma = float(row['vol_ma']) if not pd.isna(row['vol_ma']) else None

                # SL awal 3% dari swing high/low terakhir (gunakan roll_high/roll_low)
                swing_high = float(df['roll_high'].iloc[i-1]) if not pd.isna(df['roll_high'].iloc[i-1]) else price
                swing_low = float(df['roll_low'].iloc[i-1]) if not pd.isna(df['roll_low'].iloc[i-1]) else price
                if position_type == 'LONG':
                    sl = min(swing_low * 0.97, entry_price * 0.97)
                else:
                    sl = max(swing_high * 1.03, entry_price * 1.03)
                trailing_sl = sl
                continue

            # kelola posisi aktif
            if in_position and entry_price is not None and qty > 0:
                # 1) Trailing aktif setelah profit ≥ trigger_p
                if use_trailing_s:
                    if position_type == 'LONG':
                        profit_pct_now = (price - entry_price) / entry_price * 100
                        if profit_pct_now >= trigger_p:
                            new_trail = price * (1 - step_p/100.0)
                            trailing_sl = max(trailing_sl, new_trail) if trailing_sl is not None else new_trail
                    else:
                        profit_pct_now = (entry_price - price) / entry_price * 100
                        if profit_pct_now >= trigger_p:
                            new_trail = price * (1 + step_p/100.0)
                            trailing_sl = min(trailing_sl, new_trail) if trailing_sl is not None else new_trail

                # 2) Partial TP levels
                #   kita cek berurutan: TP1 (50%), TP2 (30%), TP3 (20%)
                #   setiap tercapai, close porsi terkait & kurangi qty tersisa.
                #   agar tidak double-close, gunakan flag pada bar.
                if qty > 0:
                    # LONG
                    if position_type == 'LONG':
                        if price >= entry_price * (1 + tp1_p/100.0) and qty > 0:
                            close_qty = qty * 0.5
                            exit_px = apply_slippage(price, 'sell')
                            add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP1', portion=0.5)
                            qty -= close_qty
                        if price >= entry_price * (1 + tp2_p/100.0) and qty > 0:
                            close_qty = qty * 0.6  # 30% dari awal ≈ 60% dari sisa setelah TP1
                            exit_px = apply_slippage(price, 'sell')
                            add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP2', portion=0.3)
                            qty -= close_qty
                        if price >= entry_price * (1 + tp3_p/100.0) and qty > 0:
                            close_qty = qty  # sisanya 20%
                            exit_px = apply_slippage(price, 'sell')
                            add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP3', portion=0.2)
                            qty -= close_qty
                    else:  # SHORT
                        if price <= entry_price * (1 - tp1_p/100.0) and qty > 0:
                            close_qty = qty * 0.5
                            exit_px = apply_slippage(price, 'buy')
                            add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP1', portion=0.5)
                            qty -= close_qty
                        if price <= entry_price * (1 - tp2_p/100.0) and qty > 0:
                            close_qty = qty * 0.6
                            exit_px = apply_slippage(price, 'buy')
                            add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP2', portion=0.3)
                            qty -= close_qty
                        if price <= entry_price * (1 - tp3_p/100.0) and qty > 0:
                            close_qty = qty
                            exit_px = apply_slippage(price, 'buy')
                            add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP3', portion=0.2)
                            qty -= close_qty

                # 3) Trailing SL / Hard SL
                exit_now = False
                reason = None
                if position_type == 'LONG':
                    if trailing_sl is not None and price <= trailing_sl:
                        exit_now, reason = True, 'Hit Trailing SL'
                    elif price <= sl:
                        exit_now, reason = True, 'Hit Initial SL'
                else:
                    if trailing_sl is not None and price >= trailing_sl:
                        exit_now, reason = True, 'Hit Trailing SL'
                    elif price >= sl:
                        exit_now, reason = True, 'Hit Initial SL'

                # 4) Force exit RSI ekstrem reversal
                if not exit_now and force_rsi_reversal and i >= 1:
                    prev_rsi = df['rsi'].iloc[i-1]
                    rsi_now = df['rsi'].iloc[i]
                    if position_type == 'LONG' and prev_rsi > rsi_ob and rsi_now <= rsi_ob:
                        exit_now, reason = True, 'Force Exit RSI OB Reversal'
                    if position_type == 'SHORT' and prev_rsi < rsi_os and rsi_now >= rsi_os:
                        exit_now, reason = True, 'Force Exit RSI OS Reversal'

                # 5) Force exit volume drop
                if not exit_now and entry_vol_ma is not None and not pd.isna(row['vol_ma']):
                    if row['vol_ma'] < (1 - vol_drop_pct/100.0) * entry_vol_ma:
                        exit_now, reason = True, 'Force Exit Volume Drop'

                # 6) Time-based exit (akhir hari ke-4)
                if not exit_now and bar_seconds and max_hold_seconds_swing > 0 and entry_ts is not None:
                    elapsed = (ts - entry_ts).total_seconds()
                    if elapsed >= max_hold_seconds_swing:
                        exit_now, reason = True, f'Max Hold {swing_hold_days}d'

                if exit_now and qty > 0:
                    if position_type == 'LONG':
                        exit_px = apply_slippage(price, 'sell')
                    else:
                        exit_px = apply_slippage(price, 'buy')
                    add_trade(entry_ts, ts, position_type, entry_price, exit_px, qty, reason, portion=round(qty,8))
                    # reset posisi
                    in_position = False
                    position_type = None
                    entry_price = None
                    entry_ts = None
                    qty = 0.0
                    trailing_sl = None
                    sl = None
                    entry_vol_ma = None

        # --------- Hasil Swing ---------
        st.success(f"✅ Backtest Swing selesai untuk {symbol}: {len(trades)} eksekusi (termasuk partial TP)")
        df_trades = pd.DataFrame(trades)

        # gabung per entry (optional) tetap tampil granular
        wins = df_trades[df_trades['pnl'] > 0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
        losses = df_trades[df_trades['pnl'] <= 0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
        win_rate = (len(wins) / len(df_trades) * 100.0) if len(df_trades) else 0.0
        profit_factor = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) and abs(losses['pnl'].sum()) > 0 else float('inf')

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Capital", f"${capital:.4f}")
        col2.metric("Win Rate (per eksekusi)", f"{win_rate:.2f}%")
        col3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")

        st.subheader("📈 Equity Curve")
        equity = [initial_capital]
        for t in trades:
            equity.append(equity[-1] + t['pnl'])
        fig, ax = plt.subplots()
        ax.plot(equity)
        ax.set_ylabel("Equity (USDT)")
        ax.set_xlabel("Eksekusi ke-")
        st.pyplot(fig)

        st.subheader("📊 Riwayat Eksekusi (termasuk partial TP)")
        st.dataframe(df_trades)
        csv = df_trades.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Trades CSV", csv, f"trades_swing_{symbol}.csv", "text/csv")

