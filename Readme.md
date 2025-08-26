# Bot Trading — Scalping & Swing (Binance Futures)

> **Target**: strategi custom untuk **scalping & swing** dengan **WR ≥ 75%** dan **Profit Factor > 2** (berbasis penyaringan ATR, rasio body/ATR, gate ML, SL/BE/Trailing, serta time‑stop).

---

## Daftar Isi

* [Arsitektur & Berkas](#arsitektur--berkas)
* [Instalasi & Persiapan](#instalasi--persiapan)
* [Konfigurasi `coin_config.json`](#konfigurasi-coin_configjson)
* [Logika Strategi (ringkas)](#logika-strategi-ringkas)
* [Backtest Scalping (Streamlit)](#backtest-scalping-streamlit)
* [Dry‑Run (Replay CSV seperti Live)](#dryrun-replay-csv-seperti-live)
* [Papertrade (Live data, tanpa order)](#papertrade-live-data-tanpa-order)
* [Real Trading (USDT‑M Futures)](#real-trading-usdtm-futures)
* [API / Dokumentasi Fungsi & Kelas](#api--dokumentasi-fungsi--kelas)

  * [engine\_core.py](#engine_corepy)
  * [ml\_signal\_plugin.py](#ml_signal_pluginpy)
  * [newrealtrading.py](#newrealtradingpy)
  * [papertrade.py (helper & CLI)](#papertradepy-helper--cli)
  * [tools\_dryrun\_summary.py](#tools_dryrun_summarypy)
  * [backtester\_scalping.py](#backtester_scalpingpy)
* [Troubleshooting & Tips Operasional](#troubleshooting--tips-operasional)
* [Changelog Ringkas](#changelog-ringkas)
* [Lisensi & Kredit](#lisensi--kredit)

---

## Arsitektur & Berkas

```
.
├── backtester_scalping.py        # Streamlit backtester (selaras logic live)
├── backtester_swing.py           # (opsional) backtester swing
├── tools_dryrun_summary.py       # Replay CSV -> summary WR/PF & CSV trades
├── papertrade.py                 # Live data tanpa order (journaling CSV)
├── newrealtrading.py             # Real trading USDT‑M Futures (order nyata)
├── engine_core.py                # Indikator, filter, sizing & util trading
├── ml_signal_plugin.py           # Plugin ML (RandomForest) + gate skor
├── coin_config.json              # Tuning per‑symbol + lot/precision exchange
├── requirements.txt              # Dependensi Python
├── Dockerfile                    # Image runtime (Python 3.11 slim)
└── Readme.md                     # (dok ini)
```

**Alur kerja** (disarankan):

1. Validasi ide di **backtester\_scalping.py** →
2. **tools\_dryrun\_summary.py** untuk cek WR/PF cepat (replay bar) →
3. Jalankan **papertrade.py** (live data, tanpa order) untuk verifikasi sinyal & sizing →
4. **newrealtrading.py** di **testnet** → lanjut **real**.

---

## Instalasi & Persiapan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Buat **.env** (opsional):

```
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
INSTANCE_ID=botA
# ML & scoring (opsional)
USE_ML=1
SCORE_THRESHOLD=1.2
ML_MIN_TRAIN_BARS=400
ML_RETRAIN_EVERY=5000
```

> `INSTANCE_ID` dipakai untuk `newClientOrderId` dan pemisahan log multi‑bot.

**Docker (opsional)**

```bash
docker build -t analisa-bot:latest .
```

---

## Konfigurasi `coin_config.json`

Contoh key penting per simbol (disederhanakan):

* **Leverage & Risk**: `leverage`, `risk_per_trade`
* **Biaya/Slippage**: `taker_fee` (fraksi per sisi), `SLIPPAGE_PCT` (% per sisi)
* **Filter entri**: `min_atr_pct`, `max_atr_pct`, `max_body_atr`, `use_htf_filter`
* **Exit**: `sl_mode` (ATR/PCT), `sl_pct`/`sl_atr_mult`, `sl_min_pct`, `sl_max_pct`,
  `use_breakeven`, `be_trigger_pct`, `trailing_trigger`, `trailing_step`,
  `max_hold_seconds`, `min_roi_to_close_by_time`, `time_stop_only_if_loss`
* **TP**: `tp_mode`, blok `strength_rules`, `weak_tp_roi_pct`, `use_trailing_on_strong`
* **Presisi/Lot**: `stepSize`, `minQty`, `quantityPrecision`, `minNotional`
* **ML**: blok `ml` + `score_threshold` (atau ENV `SCORE_THRESHOLD`)

> `papertrade.py` dapat **meng‑inject otomatis** `stepSize/minQty/quantityPrecision/minNotional`
> dari **futures\_exchange\_info** bila belum ada.

---

## Logika Strategi (ringkas)

* **Base signal**: EMA(22) vs SMA(20), MACD vs signal, RSI (long: 10–45, short: 70–90).
* **Filter**: ATR regime (`atr_pct` dalam rentang) + rasio **body/ATR** ≤ batas.
* **HTF filter** (opsional): EMA50 vs EMA200 pada 1h (sinkron tren besar).
* **ML gate** (opsional): RandomForest memproduksi **up\_prob** → dihitung **odds** dan
  dibandingkan `score_threshold` untuk **konfirmasi LONG/SHORT**.
* **Exit**:

  * **Hard SL** (clamp min/max %; mode PCT/ATR).
  * **Breakeven** (naikkan SL ke harga masuk bila profit ≥ trigger).
  * **Trailing** (di‑arm hanya jika profit > buffer aman **fee+slip**).
  * **Time‑stop** (tutup bila durasi > `max_hold_seconds` & ROI ≥ minimum; atau aturan *loss only*).
  * **TP dua mode**: sinyal lemah → TP tetap, sinyal kuat → trailing + BE.
* **Money Management**: risk % pada balance dan leverage; normalisasi qty ke **LOT\_SIZE**.

---

## Backtest Scalping (Streamlit)

```bash
streamlit run backtester_scalping.py --server.port 8501
```

* Pilih CSV OHLCV (kolom: `timestamp`/`open_time`/`date`, `open/high/low/close/volume`).
* **use\_htf\_filter** default **OFF** untuk eksplorasi awal.
* **Mode Debug**: melonggarkan filter dan menampilkan alasan blokir sinyal.
* Output: **Equity Curve**, tabel **Trade History** + **CSV unduh**.

## 🎯 Preset Parameter (Load/Export) — Backtester Scalping

Fitur ini memudahkan siklus **Optimasi → Terapkan → Uji Dryrun/Paper/Real** tanpa mengetik manual parameter.

### Alur Singkat
1. **Optimasi** (UI Backtester):
   - Buka expander **🧪 Optimize Parameters** dan jalankan optimasi (Grid/Random).
   - **Target**: Win Rate ≥ 70–75%, Profit Factor ≥ 2–3, minimal trades ≥ 20 (hindari overfit).
   - Unduh `opt_results-<SYMBOL>.csv`.

2. **Apply dari CSV** (opsional):
   - Di sidebar, isi `CSV hasil optimasi` → klik **Apply dari CSV**.
   - Panel **🔧 Active Params** menampilkan ringkasan parameter yang aktif.

3. **Export preset JSON**:
   - Scroll ke bawah → **📦 Export preset JSON**.
   - `Path preset JSON`: `presets/scalping_params.json` (default).
   - `Preset key`: otomatis `<SYMBOL>_<TF>` (contoh `ADAUSDT_15m`).
   - Klik **Export preset JSON** → file preset berisi snapshot parameter aktif.

4. **Load dari preset JSON**:
   - Di **sidebar**, bagian **📦 Load preset JSON**.
   - Isi `Path preset JSON` dan `Preset key`.
   - Klik **Load dari JSON** → parameter UI akan di-overwrite dari preset.
   - Cocok untuk *repeatable research* dan *team handoff*.

5. **Apply → coin_config.json** (untuk paper/real):
   - Scroll ke **💾 Apply → coin_config.json**.
   - Isi path `coin_config.json` dan `Symbol`.
   - Klik **Apply params ke coin_config.json**.
   - File akan menyimpan snapshot lengkap di `<symbol>.strategy_scalping` dan mengisi beberapa kunci legacy (mis. `be_trigger_pct`, `trailing_step`, `trailing_trigger`, `use_breakeven`, `SLIPPAGE_PCT`) agar kompatibel dengan runner lama.

---

### CLI Dryrun — pakai preset JSON (disarankan)
```bash
python tools_dryrun_summary.py \
  --symbol ADAUSDT \
  --csv data/ADAUSDT_15m_2025-06-01_to_2025-08-09.csv \
  --steps 2500 --balance 20 \
  --params-json presets/scalping_params.json \
  --preset-key ADAUSDT_15m \
  --debug-reasons
```

Prioritas Loader: --params-json (preset) lebih tinggi dari --params-csv (opt results).
Jika ingin pakai CSV: sediakan --params-csv beserta --min-wr, --min-pf, --min-trades, --prefer, --rank.

Best Practices (menuju WR 70–75% & PF ≥ 2–3)

Pastikan Eksekusi di next bar open (live-like) = ON untuk konsistensi dengan real.

Jika terlalu ketat & trade terlalu sedikit:

- Turunkan score_threshold sedikit (mis. 1.2 → 1.1),
- Longgarkan sr_near_pct (0.6 → 0.8–1.2),
- Matikan use_mtf_plus saat eksplor awal; aktifkan lagi setelah stabil.

ML gate:

- Coba naikkan score_threshold jika sinyal terlalu banyak dan PF turun.
- Sesuaikan be_trigger_pct (0.45%) agar lock-in profit lebih cepat atau lebih longgar.

Validasi multi-simbol (ADA, DOGE, XRP) → konfirmasi kestabilan preset.

### Unit Test

Jalankan:

```bash
pytest -q
```

tests/test_param_overrides.py: validasi ekspor ke coin_config.json.

tests/test_export_preset.py: validasi export preset JSON.

tests/test_param_loader_json.py: validasi load preset JSON.

---

## Dry‑Run (Replay CSV seperti Live)

```bash
python tools_dryrun_summary.py \
  --symbol ADAUSDT \
  --csv data/ADAUSDT_15m_2025-06-01_to_2025-08-09.csv \
  --coin_config coin_config.json \
  --steps 500 --balance 20 \
  --out ADA_dryrun_trades_500.csv
```

* Menggunakan **logic real** (ML + filter + trailing/BE) bar‑by‑bar.
* Output **summary** (WR, PF, avg PnL) dan **CSV trades**.

---

## Papertrade (Live data, tanpa order)

```bash
python papertrade.py \
  --live-paper \
  --symbols ADAUSDT,DOGEUSDT,XRPUSDT \
  --interval 15m \
  --balance 20 \
  --coin_config coin_config.json \
  --instance-id PAPER01 \
  --verbose
```

* Menarik **klines futures** via REST, memproses sinyal, **mencatat entry/exit & PnL** ke CSV di `logs/<INSTANCE_ID>`.
* Menjamin presisi **LOT\_SIZE** dan **MIN\_NOTIONAL** sesuai exchange.
* **Balance** dipisah per simbol (simulasi saldo lokal).

**Argumen utama**: `--risk_pct`, `--ml-thr`, `--htf`, `--heikin`, `--fee_bps`, `--slip_bps`, `--max-concurrent`, `--limit_bars`, `--timeout`, `--retries`, `--verbose`.

---

## Real Trading (USDT‑M Futures)

> **Wajib uji** di **testnet** terlebih dahulu.

```bash
python newrealtrading.py \
  --live \
  --symbols ADAUSDT,DOGEUSDT,XRPUSDT \
  --real-exec \
  --testnet \
  --instance-id botA \
  --account-guard \
  --verbose
```

* Sumber **balance**: `availableBalance` dari akun futures (bukan `--balance`).
* **Account Guard**: jika ada posisi aktif pada simbol itu, **skip entry baru**.
* **Trailing SL** otomatis **update order STOP\_MARKET** saat berubah.
* **Cooldown** setelah exit; multiplier khusus untuk HardSL/Trailing/TimeStop.

---

## API / Dokumentasi Fungsi & Kelas

### engine\_core.py

* **Konversi & boolean**

  * `_to_float(v, d)` / `_to_int(v, d)` / `_to_bool(v, d)` — casting aman dari config/ENV.
* **Presisi & lot**

  * `round_to_step(x, step)` / `floor_to_step(x, step)` / `ceil_to_step(x, step)` — normalisasi angka sesuai **tick/step**.
  * `enforce_precision(sym_cfg, price, qty)` — kembalikan `(price, qty)` sudah dibulatkan ke **tickSize/stepSize**.
  * `meets_min_notional(sym_cfg, price, qty)` — cek notional ≥ minimal.
* **Config**

  * `load_coin_config(path)` — baca JSON config.
  * `merge_config(symbol, base_cfg)` — ambil dict config untuk satu simbol.
* **Indikator & sinyal**

  * `compute_indicators(df, heikin=False)` — tambah kolom `ema/ma/macd/macd_signal/rsi/atr/atr_pct/body_to_atr/body_atr/long_base/short_base`.
  * `htf_trend_ok(side, df, htf='1h')` — cek EMA50 vs EMA200 di TF lebih tinggi.
  * `apply_filters(ind, coin_cfg)` — kembalikan `(atr_ok, body_ok, meta)` dari bar terakhir.
  * `decide_base(ind, coin_cfg)` — dict `{L:bool, S:bool}` dari sinyal dasar.
  * `confirm_htf(htf_ind, coin_cfg)` — placeholder konfirmasi HTF.
  * `apply_ml_gate(up_prob, ml_thr)` — gate odds dari probabilitas.
* **Money management & PnL**

  * `risk_size(balance, risk_pct, entry, stop, fee_bps, slip_bps, sym_cfg)` — hitung qty berdasar risiko/distance, lalu floor ke step.
  * `pnl_net(side, entry, exit, qty, fee_bps, slip_bps)` — PnL bersih & ROI%.
  * `r_multiple(entry, sl, price)` — nilai **R** berjalan.
* **Jurnal & eksekusi simulasi**

  * `journal_row(...)` — baris CSV standar untuk trade.
  * `init_stops(side, entry, ind, coin_cfg)` — inisialisasi SL/TSL.
  * `step_trailing(side, bar, prev_state, ind, coin_cfg)` — kalkulasi trailing stepwise.
  * `maybe_move_to_BE(side, entry, tsl, rule)` — BE otomatis ketika profit ≥ trigger.
  * `check_time_stop(entry_time, now, roi, rule)` — apakah **time‑stop** aktif.
  * `cooldown_until(now, rule)` — timestamp selesai cooldown.
  * `simulate_fill_on_candle(side, state, bar, sym_cfg, fee_bps, slip_bps)` — simulasi fill SL/TSL pada OHLC bar.

### ml\_signal\_plugin.py

* **`MLParams` (dataclass)** — parameter: `use_ml`, `score_threshold`, `min_train_bars`, `lookahead`, `retrain_every_bars`, `up_prob_thres`, `down_prob_thres`.
* **`MLSignal(coin_cfg=None, thr=1.2, htf=None, heikin=False, model_path=None, device='cpu')`**

  * `fit_if_needed(df)` — build dataset (fit **RandomForest** bila perlu).
  * `predict_up_prob(df)` — probabilitas naik (kelas=1) di bar terakhir.
  * `score_and_decide(base_long, base_short, up_prob=None)` — kombinasikan base signal + ML menjadi `(go_long, go_short)`.
  * Internal builder: `_build_dataset(df)`, `_build_latest_features(df)`.

### newrealtrading.py

* **`ExecutionClient(api_key, api_secret, testnet=False, verbose=False)`** — wrapper python‑binance (USDT‑M):

  * `get_available_balance()` — `availableBalance` untuk sizing.
  * `set_margin_type(symbol, isolated='ISOLATED')`, `set_leverage(symbol, lev)`.
  * `round_price(symbol, price)` / `round_qty(symbol, qty)`.
  * `market_entry(symbol, side, qty, client_id=None)`; `stop_market(symbol, side, stop_price, qty, reduce_only=True, client_id=None)`; `market_close(symbol, side, qty, client_id=None)`; `cancel_all(symbol)`.
* **`CoinTrader(symbol, config, instance_id='bot', account_guard=False, verbose=False, exec_client=None)`**

  * State: `pos` (side/entry/qty/sl/trailing\_sl/entry\_time), `cooldown_until_ts`.
  * `_enter_position(side, price, atr, available_balance)` — cek guard, kalkulasi qty (leverage & margin), clamp `minNotional`, set SL, kirim order (bila `exec_client`).
  * `_exit_position(price, reason)` — market close + cancel SL/TSL, set **cooldown** (multiplier sesuai reason).
  * `_apply_breakeven(price)` — BE berbasis `be_trigger_pct` atau `be_trigger_r` (R multiple).
  * `_update_trailing(price)` — trailing SL + update STOP\_MARKET di exchange.
  * `_should_exit(price)` — apakah kena SL/TSL.
  * `check_trading_signals(df_raw, available_balance)` — inti keputusan: indikator, filter ATR/body, HTF, ML gating, entry/exit, time‑stop. *Return* margin terpakai (untuk balancing antar simbol).
* **`TradingManager(coin_config_path, symbols, instance_id='bot', account_guard=False, exec_client=None, verbose=False)`**

  * Menyusun `CoinTrader` per simbol; *hot‑reload* perubahan aman dari `coin_config.json`.
  * `run_once(data_map)` — jalankan tiap simbol; ketika ada entry, **share** sumber balance langkah itu (global availableBalance) antar simbol.

### papertrade.py (helper & CLI)

* **Rate limiter & BinanceFutures**

  * `BinanceFutures.exchange_info()` / `klines(symbol, interval, limit)` — REST dengan retry/backoff.
* **Config injection**

  * `ensure_filters_to_coin_config(bnc, coin_cfg_path, symbols)` — isi `stepSize/minQty/quantityPrecision/minNotional` bila kosong.
* **Jurnal**

  * `Journal(dir, instance_id, cfg_by_sym, start_balance)` — CSV per simbol; `on_entry(...)`, `on_exit(...)` (hitung PnL bersih: biaya + slip).
  * `JournaledCoinTrader` — turunan `CoinTrader` yang meng‑hook `_enter_position/_exit_position` untuk journaling otomatis.
* **Util**

  * `build_df_from_klines(kl)` — DataFrame dari klines (pakai `close_time` sbg timestamp bar close).
  * `interval_seconds`, `next_close_sleep`, `last_closed_kline`, `ml_gate`.
* **CLI utama** — argumen penting:

  * `--live-paper`, `--symbols`, `--interval`, `--balance`, `--coin_config`, `--instance-id`, `--logs_dir`,
  * `--risk_pct`, `--ml-thr`, `--htf`, `--heikin`, `--fee_bps`, `--slip_bps`, `--max-concurrent`, `--limit_bars`, `--timeout`, `--retries`, `--verbose`.

### tools\_dryrun\_summary.py

* **`run_dry(symbol, csv_path, coin_config_path, steps_limit, balance)`** — replay CSV memakai **TradingManager**/**CoinTrader** real; hook PnL untuk **summary** (WR, PF, avg PnL) + **DataFrame trades**.
* **CLI** — lihat contoh pada bagian *Dry‑Run* di atas; dapat menyimpan trades ke CSV via `--out`.

### backtester\_scalping.py

* Aplikasi **Streamlit** untuk backtest *scalping* **selaras real logic**.
* Opsi: resample TF, money management, SL/BE/TSL, TP bertingkat, **ML**, **Debug mode** (menampilkan alasan blokir sinyal).
* Output: **Equity curve**, **Trade history** & **CSV**.

---
# Deploy newrealtrading — Satu Perintah Build & Jalan (Docker)

Dokumen ini menyiapkan **sekali klik**: build image + run container **real trading** (atau testnet), serta perintah ringkas untuk **cek log**, **stop**, **restart**, dan **update**.

---

## Struktur folder yang disarankan (host)

```
/var/www/rajadollarb/
├── coin_config.json            # konfigurasi per‑koin
├── .env                        # kunci API & variabel runtime (isi sendiri)
├── logs/                       # output jurnal & log bot
├── docker-compose.yml          # file compose (di bawah)
├── Makefile                    # perintah singkat (opsional, di bawah)
└── scripts/
    ├── deploy.sh               # build + up -d sekali jalan
    ├── logs.sh                 # tail log container
    ├── stop.sh                 # stop container
    ├── restart.sh              # restart container
    └── down.sh                 # stop & remove container
```

Buat folder jika belum ada:

```bash
sudo mkdir -p /var/www/rajadollarb/logs /var/www/rajadollarb/scripts
sudo chown -R $USER:$USER /var/www/rajadollarb
```

---

## 1) Isi file `.env` (mainnet / real)

Buat `/var/www/rajadollarb/.env`:

```
# Binance API (MAINNET)
BINANCE_API_KEY=ISI_KUNCI_KAMU
BINANCE_API_SECRET=ISI_RAHASIA_KAMU

# Identitas & zona waktu
INSTANCE_ID=botA
TZ=Asia/Jakarta
```

> Catatan: Untuk **testnet**, tetap gunakan .env yang sama — nanti service `newrealtrading_testnet` menambahkan flag `--testnet`.

---

## 2) `docker-compose.yml`

Simpan di `/var/www/rajadollarb/docker-compose.yml`:

```yaml
version: "3.8"

services:
  newrealtrading:
    container_name: newrealtrading
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      - TZ=${TZ:-Asia/Jakarta}
    command: >-
      python -u /app/newrealtrading.py
      --live --real-exec
      --symbols "DOGEUSDT,XRPUSDT"
      --interval 15m
      --coin_config /data/coin_config.json
      --instance-id ${INSTANCE_ID:-botA}
      --account-guard
      --verbose
    read_only: true
    tmpfs:
      - /tmp
      - /run
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 256
    mem_limit: 512m
    restart: unless-stopped
    volumes:
      - ./coin_config.json:/data/coin_config.json:ro
      - ./logs:/app/logs
      # mount project ke image hasil build; tidak perlu expose port

  # Opsi uji di TESTNET (aman sebelum real)
  newrealtrading_testnet:
    container_name: newrealtrading_testnet
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      - TZ=${TZ:-Asia/Jakarta}
    command: >-
      python -u /app/newrealtrading.py
      --live --real-exec --testnet
      --symbols "DOGEUSDT,XRPUSDT"
      --interval 15m
      --coin_config /data/coin_config.json
      --instance-id ${INSTANCE_ID:-botTEST}
      --account-guard
      --verbose
    read_only: true
    tmpfs:
      - /tmp
      - /run
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 256
    mem_limit: 512m
    restart: unless-stopped
    volumes:
      - ./coin_config.json:/data/coin_config.json:ro
      - ./logs:/app/logs
```

> Ganti daftar simbol & interval di bagian `command` sesuai kebutuhan.

---

## 3) Script sekali jalan — `scripts/deploy.sh`

Simpan di `/var/www/rajadollarb/scripts/deploy.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Pastikan folder ada
mkdir -p logs

# Build image dan jalankan (REAL)
docker compose build newrealtrading
docker compose up -d newrealtrading

echo "\n✅ newrealtrading real mode berjalan. Lihat log: ./scripts/logs.sh"
```

Lalu beri izin eksekusi:

```bash
chmod +x /var/www/rajadollarb/scripts/deploy.sh
```

### Variasi testnet cepat

Jalankan:

```bash
# testnet
cd /var/www/rajadollarb
docker compose build newrealtrading_testnet
docker compose up -d newrealtrading_testnet
```

---

## 4) Script util

**`scripts/logs.sh`** — tail log container (REAL):

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
docker compose logs -f --tail=300 newrealtrading
```

**`scripts/stop.sh`** — stop container (REAL):

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
docker compose stop newrealtrading || true
```

**`scripts/restart.sh`** — restart cepat (REAL):

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
docker compose restart newrealtrading
```

**`scripts/down.sh`** — stop & remove (REAL):

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
docker compose down newrealtrading || true
```

Beri izin eksekusi:

```bash
chmod +x /var/www/rajadollarb/scripts/{logs.sh,stop.sh,restart.sh,down.sh}
```

---

## 5) Makefile (opsional, perintah singkat)

Simpan di `/var/www/rajadollarb/Makefile`:

```make
.PHONY: build up logs stop restart down testnet

build:
	docker compose build newrealtrading

up:
	docker compose up -d newrealtrading

logs:
	docker compose logs -f --tail=300 newrealtrading

stop:
	docker compose stop newrealtrading || true

restart:
	docker compose restart newrealtrading

down:
	docker compose down newrealtrading || true

testnet:
	docker compose up -d --build newrealtrading_testnet
```

---

## 6) Cara menjalankan (REAL)

```bash
# sekali jalan (build + up)
/var/www/rajadollarb/scripts/deploy.sh

# atau gunakan Makefile
cd /var/www/rajadollarb && make build && make up
```

**Cek log (live):**

```bash
/var/www/rajadollarb/scripts/logs.sh
# atau
cd /var/www/rajadollarb && docker compose logs -f --tail=300 newrealtrading
```

**Stop / Restart / Down:**

```bash
/var/www/rajadollarb/scripts/stop.sh
/var/www/rajadollarb/scripts/restart.sh
/var/www/rajadollarb/scripts/down.sh
```

**Update ke patch terbaru:**

```bash
cd /var/www/rajadollarb
# (opsional) git pull
docker compose build --no-cache newrealtrading && docker compose up -d newrealtrading
```

---

## 7) Mode Testnet (disarankan sebelum real)

```bash
cd /var/www/rajadollarb
# build + run service testnet
docker compose up -d --build newrealtrading_testnet
# lihat log
docker compose logs -f --tail=300 newrealtrading_testnet
```

**Berhenti & hapus (testnet):**

```bash
docker compose down newrealtrading_testnet || true
```

---

## 8) Catatan keamanan & operasional

* **INSTANCE\_ID** unik jika kamu jalan beberapa bot sekaligus.
* Pastikan `coin_config.json` telah memuat **LOT\_SIZE/MIN\_NOTIONAL**; papertrade biasa meng‑inject otomatis.
* Clock server sinkron (NTP) → tanda tangan Binance valid.
* Gunakan `newrealtrading_testnet` untuk UAT; pindah ke **REAL** hanya ketika hasil papertrade/testnet sudah stabil.

---

Selesai. Dengan file di atas, kamu bisa:

* **Sekali perintah**: `scripts/deploy.sh` → build & run **real**
* **Cek log**: `scripts/logs.sh`
* **Stop/Restart/Down**: script terkait atau target `make`

---

## Lampiran — Compose YAML (versi aman tanpa fold)

> Perbaikan untuk error `yaml: line XX: could not find expected ':'`. Gunakan **array form** untuk `command` (lebih tahan salah indent & komentar).

Simpan sebagai `/var/www/rajadollarb/docker-compose.yml` (atau folder proyekmu):

```yaml
version: "3.8"

services:
  newrealtrading:
    container_name: newrealtrading
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      - TZ=${TZ:-Asia/Jakarta}
    command:
      - python
      - -u
      - /app/newrealtrading.py
      - --live
      - --real-exec
      - --symbols
      - DOGEUSDT,XRPUSDT
      - --interval
      - 15m
      - --coin_config
      - /data/coin_config.json
      - --instance-id
      - ${INSTANCE_ID:-botA}
      - --account-guard
      - --verbose
    read_only: true
    tmpfs:
      - /tmp
      - /run
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 256
    mem_limit: 512m
    restart: unless-stopped
    volumes:
      - ./coin_config.json:/data/coin_config.json:ro
      - ./logs:/app/logs

  newrealtrading_testnet:
    container_name: newrealtrading_testnet
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      - TZ=${TZ:-Asia/Jakarta}
    command:
      - python
      - -u
      - /app/newrealtrading.py
      - --live
      - --real-exec
      - --testnet
      - --symbols
      - DOGEUSDT,XRPUSDT
      - --interval
      - 15m
      - --coin_config
      - /data/coin_config.json
      - --instance-id
      - ${INSTANCE_ID:-botTEST}
      - --account-guard
      - --verbose
    read_only: true
    tmpfs:
      - /tmp
      - /run
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 256
    mem_limit: 512m
    restart: unless-stopped
    volumes:
      - ./coin_config.json:/data/coin_config.json:ro
      - ./logs:/app/logs
```

### Validasi YAML

Sebelum deploy:

```bash
cd /var/www/rajadollarb
docker compose config > /dev/null && echo "YAML OK"
```

### Deploy cepat

```bash
cd /var/www/rajadollarb
# REAL
docker compose up -d --build newrealtrading
# atau TESTNET
# docker compose up -d --build newrealtrading_testnet
```

> Catatan: Jangan gunakan `//` untuk komentar di YAML (pakai `#`). Semua item di bawah `command:` harus
> berada dalam satu list atau satu blok fold yang indentasinya benar.

```
Kalau yang diubah:

Dockerfile, requirements.txt, atau file Python yang dibake ke image (engine_core.py, newrealtrading.py, papertrade.py, backtester_scalping.py, ml_signal_plugin.py) → build ulang image + redeploy.

Hanya .env (mis. REQUEST_TIMEOUT, dsb.) atau startup flags di compose → restart container agar ENV baru terbaca.

coin_config.json (risk/leverage/trailing/threshold, dll.) → sebagian besar auto-reload saat jalan. Namun parameter yang dievaluasi saat entry baru (contoh: margin_type) baru efektif pada entry berikutnya; kalau mau langsung konsisten, restart saja.
```

Perintah praktis (pakai skrip)
Build + jalankan (recreate):
```bash
sudo scripts/deploy.sh
```
(Ini melakukan docker compose up -d --build, cocok setelah ubah Dockerfile/requirements/kode Python yang dibake.)

Restart cepat (ambil ENV baru, terapkan config terbaru):
```bash
sudo scripts/restart.sh
```
Lihat log:
```bash
sudo scripts/logs.sh
```
Stop / down seluruh stack:
```bash
sudo scripts/stop.sh     # stop container
sudo scripts/down.sh     # stop + remove container & network (data volume tetap)
```

```
## Catatan tambahan

Mengubah margin_type ke CROSSED di coin_config.json akan dipakai saat entry berikutnya. Kalau Anda ingin semua trader langsung konsisten sekarang, lakukan restart.

Mengubah REQUEST_TIMEOUT (atau ENV lain) di .env wajib restart, karena ENV hanya dibaca saat proses start.

Posisi yang sedang terbuka aman saat restart: bot akan membaca ulang posisi dari exchange saat start, lalu melanjutkan trailing/SL sesuai aturan—jadi tidak auto-close hanya karena restart.

👉 Jadi, kalau Anda baru saja mengedit kode atau dependencies → build ulang (scripts/deploy.sh).
Kalau hanya ubah .env/flag atau ingin memastikan config langsung dipakai → restart (scripts/restart.sh).
```
---

## Troubleshooting & Tips Operasional

* **-2019 Margin insufficient**: qty otomatis dikecilkan; pastikan memenuhi `minNotional`/tambahkan leverage.
* **Duplikat orderId**: gunakan `INSTANCE_ID` berbeda per proses.
* **Koneksi/testnet**: jalankan dengan `--testnet` saat uji; periksa endpoint futures testnet.
* **Lot/Presisi**: pastikan `stepSize/minQty/quantityPrecision` sesuai exchange (papertrade dapat meng‑inject otomatis).
* **Multi‑bot**: pisahkan direktori kerja & `INSTANCE_ID`; gunakan volume berbeda untuk `logs/`.

---

## Changelog Ringkas

* **ML plugin**: builder fitur kini mengembalikan **DataFrame** untuk menghindari warning *feature names*.
* **Papertrade**: inject **exchange filters** ke `coin_config.json` bila kosong; journaling CSV detail; rounding qty/price sesuai LOT\_SIZE.
* **Real trading**: **Account Guard**, **safe trailing buffer** (memperhitungkan fee+slip), time‑stop & cooldown multiplier, share **availableBalance** antar simbol per langkah.
* **Backtester scalping**: default **HTF OFF**, **Debug mode** untuk menelusuri alasan sinyal terblokir.

---

## Lisensi & Kredit

**Kredit**

* Pembuat: **shingiwoo**
  Email: **[shingiwoo.ind@gmail.com](mailto:shingiwoo.ind@gmail.com)**

### MIT License

```
Copyright (c) 2025 shingiwoo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
> **Catatan**: Dependensi pihak ketiga tunduk pada lisensi masing‑masing (mis. `python-binance` MIT, `ta` BSD, `streamlit` Apache‑2.0, `scikit-learn` BSD, dll.).

---

### Penafian Risiko

Perdagangan aset kripto berisiko tinggi. Tidak ada jaminan keuntungan masa depan. Gunakan **testnet** terlebih dahulu dan pahami seluruh risiko sebelum mengaktifkan mode real.
