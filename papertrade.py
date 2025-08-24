# papertrade.py — Live paper feed + CSV journal (Binance Futures, no orders)
# ==============================================================
# Kebutuhan:
#   pip install python-binance pandas numpy ta python-dotenv
# Catatan:
#   - Membaca sinyal dari newrealtrading.CoinTrader (patch A–D sudah di sana)
#   - Tidak mengirim order; hanya mencatat entry/exit & PnL netto (fees+slippage) ke CSV
# ==============================================================

import os, sys, time, json, math, csv, argparse, random, string
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv(override=True)
from filelock import FileLock, Timeout

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "1.3"))

# Binance
from binance.client import Client
from binance.enums import HistoricalKlinesType

# Import core engine dan trader
try:
    from newrealtrading import CoinTrader
    from engine_core import (
        _to_float, _to_bool, _to_int,
        load_coin_config, merge_config,
        apply_breakeven_sl, roi_frac_now, base_supports_side,
        floor_to_step, to_scalar
    )
except ImportError as e:
    print(f"[ERROR] Failed to import core modules: {e}")
    sys.exit(1)

# ML plugin opsional
try:
    from ml_signal_plugin import MLSignal  # noqa: F401
    ML_AVAILABLE = True
except Exception as e:  # pragma: no cover
    print(f"[WARN] ML plugin disabled: {e}")
    ML_AVAILABLE = False

# -----------------------------
# Util waktu & interval
# -----------------------------
INTERVAL_MAP = {
    "1m":  Client.KLINE_INTERVAL_1MINUTE,
    "3m":  Client.KLINE_INTERVAL_3MINUTE,
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "2h":  Client.KLINE_INTERVAL_2HOUR,
    "4h":  Client.KLINE_INTERVAL_4HOUR,
    "6h":  Client.KLINE_INTERVAL_6HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d":  Client.KLINE_INTERVAL_1DAY,
}

def interval_seconds(s: str) -> int:
    num = int(s[:-1]); unit = s[-1]
    if unit == 'm': return num * 60
    if unit == 'h': return num * 3600
    if unit == 'd': return num * 86400
    raise ValueError(f"Interval tidak didukung: {s}")

def next_close_sleep(interval_s: int, skew: float = 1.0) -> float:
    # tidur sampai boundary bar berikutnya + skew detik
    now = time.time()
    return (math.floor(now / interval_s) + 1) * interval_s + skew - now


def last_closed_kline(kl: List[list]) -> list:
    """Kembalikan bar terakhir yang sudah close dari list klines."""
    if not kl or len(kl) < 2:
        raise ValueError("Klines kurang dari 2 bar")
    return kl[-2]


def ml_gate(up_prob: Optional[float], thr: float, eps: float = 1e-9) -> int:
    """Gate odds berbasis probabilitas."""
    if up_prob is None:
        return 0
    up_odds = up_prob / max(1 - up_prob, eps)
    dn_odds = (1 - up_prob) / max(up_prob, eps)
    if up_odds >= thr:
        return +1
    if dn_odds >= thr:
        return -1
    return 0


def round_qty(flt: dict, qty: float) -> float:
    step = _to_float(flt.get("stepSize", 0.0), 0.0)
    min_qty = _to_float(flt.get("minQty", 0.0), 0.0)
    q = floor_to_step(to_scalar(qty), step) if step > 0 else to_scalar(qty)
    qp = flt.get("quantityPrecision")
    if qp is not None:
        q = float(f"{q:.{int(qp)}f}")
    if q < min_qty:
        return 0.0
    return q

def round_price(flt: dict, price: float) -> float:
    tick = _to_float(flt.get("tickSize", 0.0), 0.0)
    minp = _to_float(flt.get("minPrice", 0.0), 0.0)
    p = to_scalar(price)
    if p < minp:
        p = minp
    p = floor_to_step(p, tick) if tick > 0 else p
    pp = flt.get("pricePrecision")
    if pp is not None:
        p = float(f"{p:.{int(pp)}f}")
    return p

# -----------------------------
# Binance client wrapper (retry)
# -----------------------------
class RateLimiter:
    def __init__(self, rate: float, per: float = 1.0):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()

    def wait(self):
        now = time.monotonic()
        time_passed = now - self.last_check
        self.last_check = now
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate
        if self.allowance < 1.0:
            sleep_for = (1.0 - self.allowance) * (self.per / self.rate)
            time.sleep(sleep_for)
            self.allowance = 0.0
        else:
            self.allowance -= 1.0

class BinanceFutures:
    def __init__(self, requests_timeout: float = REQUEST_TIMEOUT, max_retries: int = HTTP_RETRIES):
        self.client = Client(requests_params={"timeout": requests_timeout})
        self.max_retries = max_retries

    def _retry(self, fn, *args, **kwargs):
        last_ex = None
        for attempt in range(max(1, self.max_retries)):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_ex = e
                print(f"[WARN] Binance call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(HTTP_BACKOFF * (attempt + 1))
        raise last_ex if last_ex else RuntimeError("Unknown error occurred during Binance call retries.")

    def exchange_info(self) -> dict:
        # Futures USDⓈ-M
        return self._retry(self.client.futures_exchange_info)

    def klines(self, symbol: str, interval: str, limit: int = 500) -> List[list]:
        # Ambil klines futures (closed bars)
        result = self._retry(
            self.client.futures_klines,
            symbol=symbol.upper(),
            interval=interval,
            limit=limit
        )
        if not isinstance(result, list) or not all(isinstance(item, list) for item in result):
            raise TypeError("Expected a List of lists, but got a different type.")
        return result

# -----------------------------
# Inject exchange filters -> coin_config.json
# -----------------------------
def ensure_filters_to_coin_config(bnc: BinanceFutures, coin_cfg_path: str, symbols: List[str]) -> dict:
    cfg_all = load_coin_config(coin_cfg_path) if os.path.exists(coin_cfg_path) else {}
    info = bnc.exchange_info()
    by_sym = {s['symbol']: s for s in info.get('symbols', [])}

    changed = False
    for sym in symbols:
        ex = by_sym.get(sym.upper())
        if not ex:
            print(f"[WARN] {sym}: tidak ditemukan di exchangeInfo (futures). Lewati.")
            continue

        lot = next((f for f in ex.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        min_notional = next((f for f in ex.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)

        step_size = float(lot.get('stepSize')) if lot else 0.0
        min_qty = float(lot.get('minQty')) if lot else 0.0
        qty_prec = int(ex.get('quantityPrecision', 0))
        min_not = float(min_notional.get('notional')) if (min_notional and 'notional' in min_notional) else 0.0

        if sym not in cfg_all: cfg_all[sym] = {}
        tgt = cfg_all[sym]

        # Tulis hanya jika belum ada (atau kosong)
        if float(tgt.get('stepSize', 0) or 0) == 0.0:
            tgt['stepSize'] = step_size; changed = True
        if float(tgt.get('minQty', 0) or 0) == 0.0:
            tgt['minQty'] = min_qty; changed = True
        if int(tgt.get('quantityPrecision', 0) or 0) == 0:
            tgt['quantityPrecision'] = qty_prec; changed = True
        if float(tgt.get('minNotional', 0) or 0) == 0.0 and min_not > 0:
            tgt['minNotional'] = min_not; changed = True

        # default biaya/slippage bila kosong (boleh kamu set ENV)
        if 'taker_fee' not in tgt: tgt['taker_fee'] = _to_float(os.getenv('TAKER_FEE', '0.0005'), 0.0005); changed = True
        if 'SLIPPAGE_PCT' not in tgt: tgt['SLIPPAGE_PCT'] = _to_float(os.getenv('SLIPPAGE_PCT', '0.02'), 0.02); changed = True

    if changed:
        for _ in range(5):
            try:
                with FileLock(coin_cfg_path + '.lock', timeout=1):
                    with open(coin_cfg_path, 'w') as f:
                        json.dump(cfg_all, f, indent=2)
                break
            except Timeout:
                time.sleep(random.uniform(0.1, 0.3))
        print(f"[INFO] coin_config.json diupdate dengan filter exchange untuk: {', '.join(symbols)}")
    return cfg_all

# -----------------------------
# CSV Journal
# -----------------------------
class Journal:
    def __init__(self, base_dir: str, instance_id: str, cfg_by_sym: Dict[str, dict], start_balance: float):
        self.dir = os.path.join(base_dir, instance_id)
        os.makedirs(self.dir, exist_ok=True)
        self.cfg = cfg_by_sym
        self.balance = {s: float(start_balance) for s in cfg_by_sym.keys()}
        self.open = {}  # symbol -> dict

    def _csv_path(self, symbol: str) -> str:
        return os.path.join(self.dir, f"{symbol}_paper_trades.csv")

    def _ensure_header(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["timestamp","exit_timestamp","symbol","side","entry","qty","sl_init","tsl_init",
                            "exit_price","exit_reason","pnl_usdt","roi_on_margin","balance_after"])

    def on_entry(self, symbol: str, side: str, entry: float, qty: float, sl_init: Optional[float], tsl_init: Optional[float]):
        cfg = self.cfg[symbol]
        entry = round_price(cfg, entry)
        qty = round_qty(cfg, qty)
        slv = round_price(cfg, sl_init) if sl_init is not None else ""
        tslv = round_price(cfg, tsl_init) if tsl_init is not None else ""
        self.open[symbol] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "side": side, "entry": float(entry), "qty": float(qty),
            "sl_init": slv,
            "tsl_init": tslv
        }

    def on_exit(self, symbol: str, exit_price: float, reason: str):
        if symbol not in self.open:
            return
        o = self.open.pop(symbol)
        bal_before = self.balance.get(symbol, 0.0)
        cfg = self.cfg[symbol]
        taker_fee = _to_float(cfg.get("taker_fee", 0.0005), 0.0005)
        slip_pct = _to_float(cfg.get("SLIPPAGE_PCT", 0.02), 0.02) / 100.0

        side = o["side"]; qty = o["qty"]; entry = o["entry"]
        exit_price = round_price(cfg, exit_price)
        # PnL koin USDT-M futures (qty dalam coin):
        if side == "LONG":
            gross = (exit_price - entry) * qty
        else:
            gross = (entry - exit_price) * qty

        # Biaya & slippage model sederhana (kedua sisi):
        fees = taker_fee * (entry + exit_price) * qty
        slipp = slip_pct * (entry + exit_price) * qty
        pnl = gross - fees - slipp

        bal_after = bal_before + pnl
        lev = _to_int(cfg.get("leverage", 1), 1)
        init_margin = (entry * qty) / max(lev, 1)
        roi_on_margin = (pnl / init_margin) if init_margin > 0 else 0.0
        self.balance[symbol] = bal_after

        path = self._csv_path(symbol)
        self._ensure_header(path)
        exit_ts_iso = pd.Timestamp.utcnow().isoformat()
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                o["ts"], exit_ts_iso, symbol, side, f"{entry:.6f}", f"{qty:.6f}",
                o["sl_init"], o["tsl_init"], f"{exit_price:.6f}",
                reason, f"{pnl:.6f}", f"{roi_on_margin:.6f}", f"{bal_after:.6f}"
            ])
        print(f"[{symbol}] EXIT {reason} pnl={pnl:.4f} roi_margin={roi_on_margin:.4f} balance={bal_before:.4f}->{bal_after:.4f}")

# -----------------------------
# Trader yg di-journal
# -----------------------------
class JournaledCoinTrader(CoinTrader):
    def __init__(self, symbol: str, config: dict, journal: Journal):
        super().__init__(symbol, config)
        self.journal = journal
        self._last_seen_len = None
        self.startup_skip_bars = int(self.config.get('startup_skip_bars', 2))
        self.post_restart_skip_entries_bars = int(self.config.get('post_restart_skip_entries_bars', 1))
        self.pending_skip_entries = self.startup_skip_bars
        self.rehydrated = False
        self.rehydrate_protect_profit = bool(self.config.get('rehydrate_protect_profit', True))
        self.rehydrate_profit_min_pct = float(self.config.get('rehydrate_profit_min_pct', 0.0005))
        self.signal_confirm_bars_after_restart = int(self.config.get('signal_confirm_bars_after_restart', 2))
        self.signal_flip_confirm_left = 0

    def _apply_breakeven(self, price: float) -> None:
        return super()._apply_breakeven(price)

    def _enter_position(self, side: str, price: float, atr: float, available_balance: float) -> float:
        before_qty = getattr(self.pos, "qty", 0.0) or 0.0
        used = super()._enter_position(side, price, atr, available_balance)
        # Jika benar-benar masuk (qty > 0 & ada side)
        if self.pos.side and self.pos.qty > 0 and self.pos.qty != before_qty:
            self.journal.on_entry(
                self.symbol, self.pos.side, self.pos.entry if self.pos.entry is not None else 0.0, self.pos.qty,
                self.pos.sl, self.pos.trailing_sl
            )
        return used

    def _exit_position(self, price: float, reason: str) -> None:
        # simpan dulu utk CSV
        if self.pos.side:
            self.journal.on_exit(self.symbol, price, reason)
        super()._exit_position(price, reason)

# -----------------------------
# Live runner
# -----------------------------
def build_df_from_klines(kl: List[list]) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","tbbav","tbqav","ignore"]
    d = pd.DataFrame(kl, columns=cols)
    d["open"] = d["open"].astype(float)
    d["high"] = d["high"].astype(float)
    d["low"] = d["low"].astype(float)
    d["close"] = d["close"].astype(float)
    d["volume"] = d["volume"].astype(float)
    # pakai close_time sebagai timestamp bar
    d["timestamp"] = pd.to_datetime(d["close_time"], unit="ms", utc=True)
    return d[["timestamp","open","high","low","close","volume"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-paper", action="store_true", help="Mode live paper (tanpa order)")
    ap.add_argument("--symbols", required=True, help="Comma separated, e.g. ADAUSDT,DOGEUSDT")
    ap.add_argument("--interval", default="15m", choices=list(INTERVAL_MAP.keys()))
    ap.add_argument("--balance", type=float, default=20.0, help="Start balance per symbol (USDT)")
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--instance-id", default=None)
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--risk_pct", type=float, default=None, help="Override risk_per_trade (contoh: 0.01)")
    ap.add_argument("--ml-thr", type=float, default=1.20, help="Threshold odds ML")
    ap.add_argument("--htf", default=None, help="Higher timeframe (contoh: 1h)")
    ap.add_argument("--heikin", action="store_true", help="Gunakan Heikin-Ashi")
    ap.add_argument("--fee_bps", type=float, default=10.0, help="Biaya taker per sisi (bps)")
    ap.add_argument("--slip_bps", type=float, default=0.0, help="Slippage per sisi (bps)")
    ap.add_argument("--max-concurrent", type=int, default=0, help="Batasi jumlah posisi aktif lintas-simbol (0=tanpa batas)")
    ap.add_argument("--limit_bars", type=int, default=600, help="Kline limit pull (<= 1500)")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=6, help="HTTP retries")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-atr-filter", action="store_true", help="Disable ATR filter")
    ap.add_argument("--no-body-filter", action="store_true", help="Disable body candle filter")
    ap.add_argument("--ml-override", action="store_true", help="Allow ML to bypass filters when triggered")
    args = ap.parse_args()

    if not args.instance_id:
        args.instance_id = "bot" + ''.join(random.choices(string.ascii_uppercase, k=2))
    if not args.logs_dir:
        args.logs_dir = os.path.join("logs", args.instance_id)
    os.environ.setdefault("VERBOSE", "1" if args.verbose else "0")

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval_key = INTERVAL_MAP[args.interval]
    int_s = interval_seconds(args.interval)

    rules = {
        "risk_pct": args.risk_pct,
        "ml_thr": args.ml_thr,
        "htf": args.htf,
        "heikin": bool(args.heikin),
        "fee_bps": args.fee_bps,
        "slip_bps": args.slip_bps,
    }

    bnc = BinanceFutures(requests_timeout=args.timeout, max_retries=args.retries)
    cfg_all = ensure_filters_to_coin_config(bnc, args.coin_config, syms)
    valid_syms = []
    for s in syms:
        if s in cfg_all:
            valid_syms.append(s)
        else:
            print(f"[WARN] {s} tidak ada di USDM Futures → di-skip")
    syms = valid_syms

    # Build trader per symbol
    cfg_by_sym = {}
    traders: Dict[str, JournaledCoinTrader] = {}
    for s in syms:
        merged = merge_config(s, cfg_all)
        if rules["risk_pct"] is not None:
            merged["risk_per_trade"] = float(rules["risk_pct"])
        merged.setdefault("ml", {})["score_threshold"] = rules["ml_thr"]
        if rules["ml_thr"] is not None:
            merged["USE_ML"] = 1
        if rules["htf"]:
            merged["htf"] = rules["htf"]
        merged["heikin"] = rules["heikin"]
        merged["taker_fee"] = rules["fee_bps"] / 10000.0
        merged["SLIPPAGE_PCT"] = rules["slip_bps"] * 0.01
        if args.no_atr_filter:
            merged.setdefault('filters', {})['atr_filter_enabled'] = False
        if args.no_body_filter:
            merged.setdefault('filters', {})['body_filter_enabled'] = False
        cfg_by_sym[s] = merged

    journal = Journal(os.path.dirname(args.logs_dir), args.instance_id, cfg_by_sym, args.balance)
    for s in syms:
        traders[s] = JournaledCoinTrader(s, cfg_by_sym[s], journal)

    print(f"[SYSTEM] Live paper feed start symbols={','.join(syms)} interval={args.interval}")

    # Loop live
    limiter = RateLimiter(rate=5)
    last_bar_ts: Dict[str, Optional[pd.Timestamp]] = {s: None for s in syms}
    while True:
        for s in syms:
            try:
                limiter.wait()
                time.sleep(0.3 + random.random() * 0.5)
                raw = bnc.klines(s, interval_key, limit=min(max(args.limit_bars, 200), 1500))
                if len(raw) < 2:
                    continue
                lc = last_closed_kline(raw)
                last_ts = pd.to_datetime(lc[6], unit="ms", utc=True)
                if last_bar_ts[s] is not None and last_ts <= last_bar_ts[s]:
                    continue
                df = build_df_from_klines(raw[:-1])
                last_bar_ts[s] = last_ts
                if args.max_concurrent and len(journal.open) >= args.max_concurrent:
                    continue
                used = traders[s].check_trading_signals(df, journal.balance[s])
                if used and used > 0:
                    before = journal.balance[s]
                    after = max(0.0, before - used)
                    print(f"[{s}] used_margin={used:.4f} balance_before={before:.4f} -> after={after:.4f}")
                    journal.balance[s] = after
            except Exception as e:
                print(f"[ERROR] live loop {s}: {e}")

        sleep_s = next_close_sleep(int_s, skew=1.0)
        if sleep_s > 0:
            time.sleep(sleep_s)

if __name__ == "__main__":
    if not _to_bool(os.getenv("BINANCE_FORCE_PUBLIC_ONLY", "1"), True):
        # (opsional) Masukkan API KEY kalau mau private call lain — untuk feed klines tidak wajib.
        Client(os.getenv("BINANCE_API_KEY",""), os.getenv("BINANCE_API_SECRET",""))
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user")
