
import streamlit as st
import pandas as pd
import numpy as np
import os, math
from typing import Any
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

"""
============================================================
APP: Backtester — SWING / AnalisaBot
FILE: backtester_swing.py
UPDATE: 2025-08-11
============================================================
Fitur:
- Sinyal gabungan: MA50/MA200, EMA255, RSI rebound/rejection, volume spike, breakout, MACD hist, engulfing.
- Take Profit bertingkat (50/30/20), Trailing setelah profit.
- Money management identik Binance Futures + normalisasi LOT_SIZE.
============================================================
"""

def floor_to_step(x: float, step: float) -> float:
    if step is None or step <= 0: return float(x)
    return math.floor(float(x)/float(step))*float(step)

st.set_page_config(page_title="Backtester Swing", layout="wide")
st.title("📈 Backtester — Swing / AnalisaBot")

st.sidebar.header("📂 Data")
data_dir = st.sidebar.text_input("Folder Data CSV", value="/mnt/data")
try:
    csv_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
except Exception:
    csv_files = []
selected_file = st.selectbox("Pilih data file (1 simbol per backtest)", options=csv_files)

st.sidebar.header("🕒 Timeframe")
timeframe = st.sidebar.selectbox("Resample", ["as-is","1H","4H","1D"], index=0)

st.sidebar.header("💰 Money Management")
initial_capital = st.sidebar.number_input("Available Balance (USDT)", value=20.0, min_value=0.0, step=1.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1.0, 10.0, 8.0)/100.0
leverage = st.sidebar.slider("Leverage", 1, 125, 15)
taker_fee = st.sidebar.number_input("Taker Fee", value=0.0005, format="%.6f")
slippage_pct = st.sidebar.slider("Slippage (%)", 0.0, 0.3, 0.02, step=0.01)

with st.sidebar.expander("⚙️ LOT_SIZE & Precision"):
    lot_step = st.number_input("stepSize (LOT_SIZE)", value=0.0, min_value=0.0, format="%.10f")
    min_qty = st.number_input("minQty (LOT_SIZE)", value=0.0, min_value=0.0, format="%.10f")
    qty_precision = st.number_input("quantityPrecision", value=0, min_value=0, max_value=8, step=1)

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
tp1_p_s = st.sidebar.number_input("TP1 % (tutup 50%)", value=5.0)
tp2_p_s = st.sidebar.number_input("TP2 % (tutup 30%)", value=8.0)
tp3_p_s = st.sidebar.number_input("TP3 % (tutup 20%)", value=12.0)

st.sidebar.subheader("🛡️ Exit Rules Lainnya")
swing_hold_days = st.sidebar.slider("Maks Hold (hari)", 1, 10, 4)

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
    df['ma_fast'] = SMAIndicator(df['close'], ma_fast).sma_indicator()
    df['ma_slow'] = SMAIndicator(df['close'], ma_slow).sma_indicator()
    df['ema_trend'] = EMAIndicator(df['close'], ema_trend).ema_indicator()
    rsi_obj = RSIIndicator(df['close'], rsi_period_s); df['rsi'] = rsi_obj.rsi()
    macd = MACD(df['close']); df['macd']=macd.macd(); df['macd_signal']=macd.macd_signal()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    if 'volume' not in df.columns: df['volume'] = 0.0
    df['vol_ma'] = df['volume'].rolling(int(vol_window)).mean()
    df['vol_spike'] = df['volume'] > (vol_spike_mult * df['vol_ma'])

    df['roll_high'] = df['high'].rolling(int(breakout_lookback)).max()
    df['roll_low'] = df['low'].rolling(int(breakout_lookback)).min()

    def bullish_engulf(i):
        if i<1: return False
        po, pc = df['open'].iloc[i-1], df['close'].iloc[i-1]
        o, c = df['open'].iloc[i], df['close'].iloc[i]
        return (pc < po) and (c > o) and (c >= po) and (o <= pc)

    def bearish_engulf(i):
        if i<1: return False
        po, pc = df['open'].iloc[i-1], df['close'].iloc[i-1]
        o, c = df['open'].iloc[i], df['close'].iloc[i]
        return (pc > po) and (c < o) and (c <= po) and (o >= pc)

    # ---------- Signals ----------
    df['long_signal'] = False
    df['short_signal'] = False
    long_confirm = 0; short_confirm = 0

    for i in range(len(df)):
        sc_long = 0.0; sc_short = 0.0
        if df['ma_fast'].iloc[i] > df['ma_slow'].iloc[i]: sc_long += 1
        if df['ma_fast'].iloc[i] < df['ma_slow'].iloc[i]: sc_short += 1
        if df['close'].iloc[i] > df['ema_trend'].iloc[i]: sc_long += 0.25
        else: sc_short += 0.25
        if i>=1:
            pr, rn = df['rsi'].iloc[i-1], df['rsi'].iloc[i]
            if pr < rsi_os and rn >= rsi_os: sc_long += 1
            if pr > rsi_ob and rn <= rsi_ob: sc_short += 1
        if bool(df['vol_spike'].iloc[i]): sc_long += 0.5; sc_short += 0.5
        if df['close'].iloc[i] > df['roll_high'].iloc[i]:
            long_confirm += 1; short_confirm = 0
        elif df['close'].iloc[i] < df['roll_low'].iloc[i]:
            short_confirm += 1; long_confirm = 0
        else:
            long_confirm = max(0,long_confirm-1); short_confirm = max(0,short_confirm-1)
        if long_confirm >= int(confirm_bars): sc_long += 1
        if short_confirm >= int(confirm_bars): sc_short += 1
        if i>=1:
            if df['macd_hist'].iloc[i-1] <= 0 and df['macd_hist'].iloc[i] > 0: sc_long += 0.5
            if df['macd_hist'].iloc[i-1] >= 0 and df['macd_hist'].iloc[i] < 0: sc_short += 0.5
        if bullish_engulf(i): sc_long += 0.5
        if bearish_engulf(i): sc_short += 0.5
        if sc_long >= float(score_th_swing): df.loc[i,'long_signal'] = True
        if sc_short >= float(score_th_swing): df.loc[i,'short_signal'] = True

    # ---------- Backtest ----------
    in_position = False
    position_type = None
    entry_price = None
    entry_ts = None
    qty = 0.0
    capital = float(initial_capital)
    taker_fee_val = float(taker_fee)
    trailing_sl = None
    sl = None
    trades = []

    def apply_slippage(px: float, side: str) -> float:
        return px * (1 + slippage_pct/100) if side=='buy' else px * (1 - slippage_pct/100)

    def add_trade(timestamp_entry, timestamp_exit, side, entry, exit_px, q, reason, portion=1.0):
        fee = (entry + exit_px) * taker_fee_val * q
        pnl = (exit_px - entry) * q - fee if side=='LONG' else (entry - exit_px)*q - fee
        init_margin = (entry*q)/float(leverage) if leverage>0 else 0.0
        roi = (pnl/init_margin) if init_margin>0 else 0.0
        trades.append({'timestamp_entry':timestamp_entry,'timestamp_exit':timestamp_exit,'symbol':symbol,'type':side,'entry':entry,'exit':exit_px,'qty':q,'portion':portion,'fee':fee,'pnl':pnl,'roi_on_margin':roi,'reason':reason})
        return pnl

    max_hold_seconds_swing = int(swing_hold_days * 86400)

    for i in range(1, len(df)):
        row = df.iloc[i]; ts = row['timestamp']; price = float(row['close'])

        # open
        if not in_position and (row['long_signal'] or row['short_signal']):
            margin = capital * risk_per_trade
            if price <= 0 or leverage <= 0 or margin <= 0: continue
            raw_qty = (margin * leverage) / price
            adj_qty = floor_to_step(raw_qty, lot_step) if lot_step > 0 else raw_qty
            if qty_precision is not None and qty_precision >= 0:
                try: adj_qty = float(f"{adj_qty:.{int(qty_precision)}f}")
                except Exception: adj_qty = float(adj_qty)
            if min_qty > 0 and adj_qty < min_qty: continue
            if adj_qty <= 0: continue

            in_position = True
            qty = adj_qty
            entry_price = price
            entry_ts = ts
            position_type = 'LONG' if row['long_signal'] else 'SHORT'

            swing_high = float(df['roll_high'].iloc[i-1]) if not pd.isna(df['roll_high'].iloc[i-1]) else price
            swing_low = float(df['roll_low'].iloc[i-1]) if not pd.isna(df['roll_low'].iloc[i-1]) else price
            sl = min(swing_low * 0.97, entry_price*0.97) if position_type=='LONG' else max(swing_high*1.03, entry_price*1.03)
            trailing_sl = sl
            continue

        # manage
        if in_position and entry_price is not None and qty > 0:
            if use_trailing_s:
                if position_type=='LONG':
                    profit_pct_now = (price - entry_price)/entry_price*100
                    if profit_pct_now >= trigger_p:
                        new_trail = price * (1 - step_p/100.0)
                        trailing_sl = max(trailing_sl, new_trail) if trailing_sl is not None else new_trail
                else:
                    profit_pct_now = (entry_price - price)/entry_price*100
                    if profit_pct_now >= trigger_p:
                        new_trail = price * (1 + step_p/100.0)
                        trailing_sl = min(trailing_sl, new_trail) if trailing_sl is not None else new_trail

            # partial TP
            if qty > 0:
                if position_type=='LONG':
                    if price >= entry_price*(1+tp1_p_s/100.0) and qty>0:
                        close_qty = qty*0.5; exit_px = apply_slippage(price,'sell'); add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP1', portion=0.5); qty -= close_qty
                    if price >= entry_price*(1+tp2_p_s/100.0) and qty>0:
                        close_qty = qty*0.6; exit_px = apply_slippage(price,'sell'); add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP2', portion=0.3); qty -= close_qty
                    if price >= entry_price*(1+tp3_p_s/100.0) and qty>0:
                        close_qty = qty; exit_px = apply_slippage(price,'sell'); add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP3', portion=0.2); qty -= close_qty
                else:
                    if price <= entry_price*(1-tp1_p_s/100.0) and qty>0:
                        close_qty = qty*0.5; exit_px = apply_slippage(price,'buy'); add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP1', portion=0.5); qty -= close_qty
                    if price <= entry_price*(1-tp2_p_s/100.0) and qty>0:
                        close_qty = qty*0.6; exit_px = apply_slippage(price,'buy'); add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP2', portion=0.3); qty -= close_qty
                    if price <= entry_price*(1-tp3_p_s/100.0) and qty>0:
                        close_qty = qty; exit_px = apply_slippage(price,'buy'); add_trade(entry_ts, ts, position_type, entry_price, exit_px, close_qty, 'TP3', portion=0.2); qty -= close_qty

            # exits
            exit_now = False; reason = None
            if position_type=='LONG':
                if trailing_sl is not None and price <= trailing_sl: exit_now, reason = True, 'Hit Trailing SL'
                elif (sl is not None) and price <= sl: exit_now, reason = True, 'Hit Initial SL'
            else:
                if trailing_sl is not None and price >= trailing_sl: exit_now, reason = True, 'Hit Trailing SL'
                elif (sl is not None) and price >= sl: exit_now, reason = True, 'Hit Initial SL'

            if not exit_now and bar_seconds and int(swing_hold_days*86400) > 0 and entry_ts is not None:
                elapsed = (ts - entry_ts).total_seconds()
                if elapsed >= int(swing_hold_days*86400):
                    exit_now, reason = True, f'Max Hold {swing_hold_days}d'

            if exit_now and qty > 0:
                exit_px = apply_slippage(price, 'sell' if position_type=='LONG' else 'buy')
                add_trade(entry_ts, ts, position_type, entry_price, exit_px, qty, reason, portion=round(qty,8))
                in_position=False; position_type=None; entry_price=None; entry_ts=None; qty=0.0; trailing_sl=None; sl=None

    # ---------- Hasil ----------
    st.success(f"✅ Backtest Swing selesai untuk {symbol}")
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['pnl']>0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
    losses = df_trades[df_trades['pnl']<=0] if not df_trades.empty else pd.DataFrame(columns=['pnl'])
    win_rate = (len(wins)/len(df_trades)*100.0) if len(df_trades) else 0.0
    profit_factor = (wins['pnl'].sum()/abs(losses['pnl'].sum())) if len(losses) and abs(losses['pnl'].sum())>0 else float('inf')

    c1,c2,c3 = st.columns(3)
    c1.metric("Final Capital", f"${capital:.4f}")
    c2.metric("Win Rate (per eksekusi)", f"{win_rate:.2f}%")
    c3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")

    st.subheader("📈 Equity Curve")
    equity = [initial_capital]
    for t in trades: equity.append(equity[-1] + t['pnl'])
    fig, ax = plt.subplots(); ax.plot(equity); ax.set_ylabel("Equity (USDT)"); ax.set_xlabel("Eksekusi #")
    st.pyplot(fig)

    st.subheader("📊 Riwayat Eksekusi (termasuk partial TP)")
    st.dataframe(df_trades)
    csv = df_trades.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Trades CSV", csv, f"trades_swing_{symbol}.csv", "text/csv")
