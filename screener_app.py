#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, json
from typing import List, Optional
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screener import run_screener  # reuse core logic

# === Cache wrapper agar UI tidak "stuck" untuk input yang sama ===
@st.cache_data(ttl=45, show_spinner=False)
def cached_run_screener(symbols_tuple, mode, interval, sel_market, limit, params_json, preset_key, coin_config_path, use_coin_cfg):
    return run_screener(
        symbols=list(symbols_tuple),
        mode=mode,
        interval=interval,
        market=sel_market,
        limit=int(limit),
        params_json=params_json,
        preset_key=preset_key,
        coin_config_path=coin_config_path,
        use_coin_cfg=use_coin_cfg,
    )

DEFAULT_SYMBOLS = "ADAUSDT,DOGEUSDT,XRPUSDT,SOLUSDT,BNBUSDT,ETHUSDT,BTCUSDT,APTUSDT,OPUSDT,SEIUSDT,ARBUSDT,SUIUSDT,TONUSDT,LTCUSDT,LINKUSDT,ATOMUSDT"

st.set_page_config(page_title="Crypto Screener (Scalping & Spot/Long)", layout="wide")
st.title("📊 Crypto Screener — Scalping & Spot/Long-only")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    symbols = st.text_area("Symbols (comma-separated)", value=DEFAULT_SYMBOLS, height=120)
    colA, colB = st.columns(2)
    mode = colA.selectbox("Mode", ["scalping","spot_long"], index=0)
    market = colB.selectbox("Market", ["auto","spot","futures"], index=0)
    interval = st.selectbox("Interval", ["5m","15m","30m","1h"], index=1)
    limit = st.slider("Bar (history)", min_value=260, max_value=1500, value=720, step=20)
    params_json = st.text_input("Preset JSON", value="presets/scalping_params.json")
    preset_key = st.text_input("Preset Key", value="GLOBAL_15m")
    coin_config_path = st.text_input("Coin Config", value="coin_config.json")
    use_coin_cfg = st.checkbox("Gunakan override dari coin_config.json", value=True)
    auto_refresh = st.checkbox("Auto-refresh tiap 180 detik (aktif setelah hasil tampil)", value=False)
    run_btn = st.button("▶️ Jalankan Screener", type="primary")

# NOTE:
# Auto-refresh dipindah ke bagian paling bawah SETELAH hasil tampil.
# Menghindari rerun loop sebelum analisa berjalan.

if run_btn:
    syms: List[str] = [x.strip().upper() for x in symbols.split(",") if x.strip()]
    sel_market = None if market=="auto" else market
    with st.spinner("Mengambil data & menghitung skor..."):
        # Progress bar (indikatif), karena run_screener sudah paralel
        prog = st.progress(10, text="Memproses simbol...")
        df = cached_run_screener(tuple(syms), mode, interval, sel_market, int(limit), params_json, preset_key, coin_config_path, use_coin_cfg)
        prog.progress(100, text="Selesai memproses.")
    st.success(f"Selesai. {len(df)} simbol.")

    if df.empty:
        st.info("Tidak ada data.")
        st.stop()

    # Ringkasan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ENTRY", int((df["status"]=="ENTRY").sum()))
    c2.metric("WATCHLIST", int((df["status"]=="WATCHLIST").sum()))
    c3.metric("AVOID", int((df["status"]=="AVOID").sum()))
    c4.metric("Avg Score", f'{df["score"].fillna(0).mean():.3f}')

    # Tabs
    t_all, t_long, t_short = st.tabs(["🧾 Semua", "🟦 LONG picks", "🟪 SHORT picks (scalping)"])

    def _style(dfv: pd.DataFrame) -> Styler:
        dfv = dfv.copy()
        return dfv.style \
            .apply(lambda s: ["background-color:#132a13;color:#9cffad" if v=="ENTRY" else
                              "background-color:#2b2b00;color:#ffe36e" if v=="WATCHLIST" else
                              "color:#888" for v in s], subset=["status"]) \
            .format({"score":"{:.3f}"})

    with t_all:
        st.dataframe(_style(df), use_container_width=True, height=480)
        st.download_button("💾 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"screener_{mode}.csv", mime="text/csv")

    with t_long:
        df_long = df[(df["side"]=="LONG") & (df["status"]!="NO DATA")]
        st.dataframe(_style(df_long), use_container_width=True, height=480)

    with t_short:
        if mode!="scalping":
            st.info("SHORT hanya tersedia di mode SCALPING.")
        else:
            df_short = df[(df["side"]=="SHORT") & (df["status"]!="NO DATA")]
            st.dataframe(_style(df_short), use_container_width=True, height=480)

    # === Auto-refresh aman (30 dtk) – hanya setelah hasil tampil ===
    if auto_refresh:
        st.caption("⏳ Auto-refresh aktif. App akan refresh tiap 30 detik…")
        time.sleep(180)
        st.rerun()
