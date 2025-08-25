# newrealtrading.py — FULL PATCH (revised)
# =============================================================
# Update: 2025-08-16
# - Melengkapi ekspor simbol supaya `from newrealtrading import ...` dikenali Pylance/IDE
# - Tetap kompatibel dengan papertrade.py (live paper feed / tanpa order)
# - Struktur modular: CoinTrader, TradingManager, utils, indikator, loader config, ML plugin hook
# =============================================================
from __future__ import annotations
import os, json, time, math, threading, random, string
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from uuid import uuid4

import pandas as pd
import numpy as np
from decimal import getcontext

# --- tipe live posisi & helper presisi ---
@dataclass
class LivePos:
    side: str               # "LONG" / "SHORT"
    qty: float              # abs(quantity)
    entry_price: float      # entry price dari exchange (fallback mark)
    leverage: int

def qty_to_str(step_size: float, quantity_precision: int, qty: float) -> str:
    from decimal import Decimal, ROUND_DOWN
    q = Decimal(str(qty))
    if step_size and step_size > 0:
        step = Decimal(str(step_size))
        q = (q / step).to_integral_value(rounding=ROUND_DOWN) * step
    qprec = max(int(quantity_precision or 0), 0)
    return f"{q:.{qprec}f}"

def price_to_str(p: Optional[float], dp: int = 8) -> str:
    # Fail-fast agar kita tak pernah mengirim 0/None ke exchange
    if p is None:
        raise ValueError("price_to_str: 'p' tidak boleh None")
    return f"{float(p):.{dp}f}"

# --- TA
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

# --- ML plugin (pastikan file ml_signal_plugin.py ada 1 folder dengan file ini)
from ml_signal_plugin import MLSignal

# (opsional) dotenv agar ENV dari .env terbaca jika ada
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from engine_core import (
    _to_float, _to_int, _to_bool, floor_to_step, ceil_to_step, to_scalar, to_bool,
    load_coin_config, merge_config, compute_indicators as calculate_indicators,
    htf_trend_ok, r_multiple, apply_breakeven_sl, roi_frac_now, base_supports_side,
    safe_div, as_float, compute_base_signals_backtest, make_decision, round_to_tick,
    as_scalar
)

from binance.client import Client
from binance.exceptions import BinanceAPIException

INSTANCE_ID = os.getenv("INSTANCE_ID", "bot")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "1.3"))


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


def safe_fetch_klines(client, symbol: str, interval: str, limit: int):
    for attempt in range(max(1, HTTP_RETRIES)):
        try:
            return client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        except Exception:
            if attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(HTTP_BACKOFF * (attempt + 1))
class ExecutionClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool=False, verbose: bool=False):
        self.client = Client(api_key, api_secret, testnet=testnet,
                             requests_params={"timeout": REQUEST_TIMEOUT})
        if testnet:
            self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
        self.verbose = verbose
        self.symbol_filters: Dict[str, Dict[str, Any]] = {}
        getcontext().prec = 28
        self._load_filters(log=True)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[EXEC] {msg}")

    def _load_filters(self, log: bool = False) -> None:
        info = self.client.futures_exchange_info()
        self.futures_info = info
        self.symbol_filters = {}
        for s in info.get("symbols", []):
            sym = s.get("symbol", "").upper()
            f = {ft.get("filterType"): ft for ft in s.get("filters", [])}
            price_filter = f.get("PRICE_FILTER", {})
            lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE") or {}
            self.symbol_filters[sym] = {
                "tickSize": price_filter.get("tickSize", "0"),
                "minPrice": price_filter.get("minPrice", "0"),
                "stepSize": lot.get("stepSize", "0"),
                "minQty": lot.get("minQty", "0"),
                "quantityPrecision": s.get("quantityPrecision"),
                "pricePrecision": s.get("pricePrecision"),
            }
        if log:
            for sym, flt in self.symbol_filters.items():
                print(f"[FILTER:{sym}] step={flt['stepSize']} tick={flt['tickSize']} minQty={flt['minQty']} qPrec={flt['quantityPrecision']} pPrec={flt['pricePrecision']}")

    def has_symbol(self, symbol: str) -> bool:
        if not isinstance(symbol, str) or not symbol:
            return False
        return symbol.upper() in self.symbol_filters

    def get_step_size(self, symbol: str) -> float:
        symbol = (symbol or "").upper()
        flt = self.symbol_filters.get(symbol)
        if not flt:
            raise ValueError(f"symbol {symbol} tidak dikenal")
        return float(flt.get("stepSize") or 0.0)

    def get_tick_size(self, symbol: str) -> float:
        symbol = (symbol or "").upper()
        flt = self.symbol_filters.get(symbol)
        if not flt:
            raise ValueError(f"symbol {symbol} tidak dikenal")
        return float(flt.get("tickSize") or 0.0)

    def round_qty(self, symbol: str, qty: float) -> float:
        symbol = (symbol or "").upper()
        if not symbol:
            raise ValueError("symbol is required")
        qty = float(_to_float(qty, 0.0))
        step = self.get_step_size(symbol)
        return floor_to_step(qty, step)

    def round_price(self, symbol: str, price: float) -> float:
        symbol = (symbol or "").upper()
        if not symbol:
            raise ValueError("symbol is required")
        price = float(_to_float(price, 0.0))
        tick = self.get_tick_size(symbol)
        return round_to_tick(price, tick)

    def _qty_to_str(self, symbol: str, qty: float) -> str:
        flt = self.symbol_filters.get(symbol.upper(), {})
        qprec = int(flt.get("quantityPrecision") or 0)
        step = float(flt.get("stepSize") or 0.0)
        base = float(_to_float(qty, 0.0))
        q = floor_to_step(base + 1e-12, step) if step > 0 else base
        return f"{q:.{qprec}f}"

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            self._log(f"set_leverage fail: {e}")

    def set_margin_type(self, symbol: str, isolated: bool | str = "ISOLATED"):
        try:
            if isinstance(isolated, bool):
                mtype = "ISOLATED" if isolated else "CROSSED"
            else:
                mtype = str(isolated)
            self.client.futures_change_margin_type(symbol=symbol, marginType=mtype)
        except BinanceAPIException as e:
            if e.code == -4046:
                self._log(f"set_margin_type fail: {e}")
                return None
        except Exception as e:
            self._log(f"set_margin_type fail: {e}")

    def has_position(self, symbol: str) -> bool:
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            qty = float(pos[0].get("positionAmt", 0.0))
            return abs(qty) > 0.0
        except Exception:
            return False

    def get_available_balance(self) -> float:
        """
        Ambil saldo USDT yang available untuk buka posisi baru (Futures USDT-M).
        """
        try:
            bal = self.client.futures_account_balance()
            usdt = next((b for b in bal if b.get("asset") == "USDT"), None)
            return float(usdt.get("availableBalance", 0.0)) if usdt else 0.0
        except Exception:
            try:
                acc = self.client.futures_account()
                return float(acc.get("availableBalance", 0.0))
            except Exception:
                return 0.0

    def _order_with_retry(self, params: Dict[str, Any], raw_qty: float, raw_price: Optional[float] = None) -> Dict[str, Any]:
        symbol = params.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            self._log("order retry skipped: invalid symbol")
            return {}
        try:
            return self.client.futures_create_order(**params)
        except BinanceAPIException as e:
            if e.code == -1111:
                flt = self.symbol_filters.get(symbol.upper(), {})
                self._log(f"[ROUND-RETRY] {symbol}: raw_qty={raw_qty} step={flt.get('stepSize')} qPrec={flt.get('quantityPrecision')} -> retry")
                self._load_filters(log=False)
                params["quantity"] = self._qty_to_str(symbol, raw_qty)
                if raw_price is not None:
                    params["stopPrice"] = f"{self.round_price(symbol, raw_price):.8f}"
                try:
                    return self.client.futures_create_order(**params)
                except Exception as e2:
                    self._log(f"order retry fail: {e2}")
            else:
                self._log(f"order fail: {e}")
        except Exception as e:
            self._log(f"order fail: {e}")
        return {}

    # --- Helper: clamp qty & side untuk reduce-only ----
    def _clamp_reduceonly_qty(self, symbol: str, side: str, req_qty: float) -> Tuple[str, float]:
        """
        Samakan side penutupan dengan arah posisi aktual dan batasi qty <= abs(positionAmt).
        Return: (side_close, allowed_qty_rounded)
        """
        try:
            info = self.position_info(symbol) or {}
        except Exception:
            info = {}
        pos_amt = float(info.get("positionAmt") or 0.0)
        if abs(pos_amt) < 1e-12:
            if self.verbose:
                print(f"[EXEC] SKIP close: no open position for {symbol}")
            return side, 0.0
        expected_side = "SELL" if pos_amt > 0 else "BUY"
        if side != expected_side:
            side = expected_side
        allowed = min(abs(pos_amt), float(req_qty or 0.0))
        allowed = self.round_qty(symbol, allowed)
        return side, allowed

    def market_entry(self, symbol: str, side: str, qty: float, client_id: Optional[str]=None) -> Dict[str, Any]:
        r_qty = self.round_qty(symbol, qty)
        if r_qty <= 0:
            self._log(f"SKIP entry: qty<minQty (raw={qty})")
            return {}
        params = dict(symbol=symbol, side=side, type="MARKET",
                      quantity=self._qty_to_str(symbol, r_qty), reduceOnly=False)
        if client_id:
            params["newClientOrderId"] = client_id
        o = self._order_with_retry(params, qty)
        if self.verbose and o:
            print(f"[EXEC] MARKET ENTRY {symbol} {side} {r_qty} -> orderId={o.get('orderId')}")
        return o

    def stop_market(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        qty: Optional[float] = None,
        *,
        reduce_only: bool = True,
        client_id: Optional[str] = None,
        order_type: str = "STOP_MARKET",
        close_all: bool = False,
    ) -> Dict[str, Any]:
        raw_price = stop_price
        sp = self.round_price(symbol, stop_price)
        params = dict(
            symbol=symbol,
            side=side,
            type=order_type,
            stopPrice=f"{sp:.8f}",
            closePosition=True if close_all else False,
            reduceOnly=True if close_all else reduce_only,
            workingType="MARK_PRICE",
            timeInForce="GTC",
        )
        r_qty = 0.0
        raw_qty = 0.0
        if not close_all:
            raw_qty = float(qty or 0.0)
            r_qty = self.round_qty(symbol, raw_qty)
            if r_qty <= 0:
                self._log(f"SKIP stop: qty<minQty (raw={raw_qty})")
                return {}
            params["quantity"] = self._qty_to_str(symbol, r_qty)
        else:
            params.pop("quantity", None)
        if client_id:
            params["newClientOrderId"] = client_id
        try:
            raw_for_retry = raw_qty if not close_all else 0.0
            o = self._order_with_retry(params, raw_for_retry, raw_price)
            if self.verbose and o:
                print(
                    f"[EXEC] {order_type} {symbol} {side} stop={sp} closePos={params['closePosition']} -> orderId={o.get('orderId')}"
                )
            return o
        except BinanceAPIException as e:
            if getattr(e, "code", None) == -2022 and "ReduceOnly" in str(e):
                if self.verbose:
                    print(f"[EXEC] {order_type} ignore -2022 (no position to reduce) {symbol}")
                return {}
            raise

    def cancel_all(self, symbol: str):
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
            if self.verbose:
                print(f"[EXEC] Cancel all {symbol}")
        except Exception as e:
            if self.verbose:
                print(f"[EXEC] cancel_all fail: {e}")

    def position_info(self, symbol: str):
        infos = self.client.futures_position_information(symbol=symbol)
        return infos[0] if infos else None

    def open_orders(self, symbol: str):
        return self.client.futures_get_open_orders(symbol=symbol)

    # --- Alias kompatibilitas lama (untuk code yang masih memanggil get_position/get_last_price) ---
    def get_position(self, symbol: str):
        """Alias lama -> gunakan position_info."""
        return self.position_info(symbol)

    def get_last_price(self, symbol: str) -> float:
        """Alias lama -> ambil mark price sebagai 'last' untuk keperluan kalkulasi cepat."""
        try:
            mp = self.client.futures_mark_price(symbol=symbol)
            return float(mp.get("markPrice") or mp.get("price") or 0.0)
        except Exception:
            return 0.0

    def get_live_position(self, symbol: str) -> Optional[LivePos]:
        try:
            info = self.position_info(symbol) or {}
        except Exception:
            info = {}
        amt = float(info.get("positionAmt", 0) or 0)
        if abs(amt) < 1e-12:
            return None
        entry = float(info.get("entryPrice") or 0) or float(info.get("markPrice") or 0)
        lev = int(float(info.get("leverage") or 1))
        side = "LONG" if amt > 0 else "SHORT"
        return LivePos(side=side, qty=abs(amt), entry_price=entry, leverage=lev)

    def has_active_sl_close_all(self, symbol: str) -> bool:
        try:
            orders = self.open_orders(symbol)
        except Exception:
            orders = []
        for od in orders:
            t = (od.get("type") or "").upper()
            if t in ("STOP_MARKET", "STOP"):
                if str(od.get("closePosition", "")).lower() == "true":
                    return True
        return False

    def place_protective_sl_close_all(self, symbol: str, side: str, stop_price: float) -> bool:
        params = dict(
            symbol=symbol,
            side=("SELL" if side == "LONG" else "BUY"),
            type="STOP_MARKET",
            stopPrice=price_to_str(stop_price, dp=8),
            workingType="MARK_PRICE",
            closePosition=True,
            reduceOnly=True,
            timeInForce="GTC",
        )
        try:
            o = self._order_with_retry(params, 0.0, stop_price)
            return bool(o)
        except Exception as e:
            self._log(f"protective SL fail: {e}")
            return False

    def market_close(self, symbol: str, side: str, qty: float, client_id: Optional[str]=None) -> Dict[str, Any]:
        try:
            info = self.position_info(symbol) or {}
        except Exception:
            info = {}
        pos_amt = float(info.get("positionAmt") or 0.0)
        if abs(pos_amt) <= 0.0:
            self._log(f"SKIP close: no open position for {symbol}")
            return {}
        close_side = "SELL" if pos_amt > 0 else "BUY"
        qty_live = abs(pos_amt)
        r_qty = self.round_qty(symbol, qty_live)
        params = dict(symbol=symbol, side=close_side, type="MARKET",
                      quantity=self._qty_to_str(symbol, r_qty), reduceOnly=True)
        if client_id:
            params["newClientOrderId"] = client_id
        try:
            o = self._order_with_retry(params, qty_live)
            if self.verbose and o:
                print(f"[EXEC] MARKET CLOSE {symbol} {close_side} {r_qty} -> orderId={o.get('orderId')}")
        except BinanceAPIException as e:
            if getattr(e, "code", None) == -2022 and "ReduceOnly" in str(e):
                if self.verbose:
                    print(f"[EXEC] MARKET CLOSE ignore -2022 (no position to reduce) {symbol}")
                return {}
            raise
        step = self.get_step_size(symbol)
        try:
            info2 = self.position_info(symbol) or {}
        except Exception:
            info2 = {}
        rem = abs(float(info2.get("positionAmt") or 0.0))
        if rem >= step / 2:
            r2 = self.round_qty(symbol, rem)
            if r2 > 0:
                params["quantity"] = self._qty_to_str(symbol, r2)
                if client_id:
                    params["newClientOrderId"] = f"{client_id}-r"
                try:
                    o = self._order_with_retry(params, rem)
                    if self.verbose and o:
                        print(f"[EXEC] MARKET CLOSE RETRY {symbol} {close_side} {r2} -> orderId={o.get('orderId')}")
                except Exception as e:
                    self._log(f"close retry fail: {e}")
                try:
                    info2 = self.position_info(symbol) or {}
                except Exception:
                    info2 = {}
                rem = abs(float(info2.get("positionAmt") or 0.0))
        min_qty = float(self.symbol_filters.get(symbol.upper(), {}).get("minQty") or 0.0)
        if 0 < rem < min_qty:
            try:
                mark = float(self.client.futures_mark_price(symbol=symbol).get("markPrice"))
            except Exception:
                mark = 0.0
            tick = self.get_tick_size(symbol)
            sp = mark + tick if close_side == "SELL" else mark - tick
            params2 = dict(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=f"{sp:.8f}",
                closePosition=True,
                reduceOnly=True,
                workingType="MARK_PRICE",
                timeInForce="GTC",
            )
            try:
                self._order_with_retry(params2, 0.0, sp)
                if self.verbose:
                    print(f"[EXEC] DUST SWEEP {symbol} {close_side} stop={sp}")
            except Exception as e:
                self._log(f"dust sweep fail: {e}")
        return o

__all__ = [
    "ExecutionClient",
    "CoinTrader",
    "TradingManager",
    "floor_to_step",
    "_to_float",
    "_to_bool",
    "load_coin_config",
    "merge_config",
    "calculate_indicators",
]

# ============================
# Config
# ============================
DEFAULTS = {
    "leverage": 15,
    "risk_per_trade": 0.08,
    "taker_fee": 0.0005,  # fraksi per sisi
    "min_atr_pct": 0.006,
    "max_atr_pct": 0.03,
    "max_body_atr": 0.95,
    "use_htf_filter": 1,
    "cooldown_seconds": 1500,
    # SL / BE / Trailing
    "sl_mode": "ATR",
    "sl_pct": 0.008,
    "sl_atr_mult": 1.6,
    "sl_min_pct": 0.010,
    "sl_max_pct": 0.030,
    "use_breakeven": 1,
    "be_trigger_pct": 0.0045,  # fraksi (0.45%)
    "trailing_trigger": 0.7,   # %
    "trailing_step": 0.45,     # %
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.005,
    # lot size & precision (opsional, jika tersedia dari exchange info)
    "stepSize": 0.0,
    "minQty": 0.0,
    "quantityPrecision": 0,
}

ENV_DEFAULTS = {
    "SLIPPAGE_PCT": 0.02,  # % per sisi
    "SCORE_THRESHOLD": 1.2,
}

# ============================
# Loader coin_config.json
# ============================

# ============================
# Trader per-coin
# ============================
@dataclass
class Position:
    side: Optional[str] = None   # 'LONG' | 'SHORT' | None
    entry: Optional[float] = None
    qty: float = 0.0
    sl: Optional[float] = None
    trailing_sl: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    allow_trailing: bool = True


class CoinTrader:
    def __init__(self, symbol: str, config: Dict[str, Any], *, instance_id: str="bot", account_guard: bool=False, verbose: bool=False, exec_client: Optional[ExecutionClient]=None):
        self.symbol = symbol.upper()
        self.config = config
        self.ml = MLSignal(self.config)
        self.pos = Position()
        self.cooldown_until_ts: Optional[float] = None
        self.instance_id = instance_id
        self.account_guard = account_guard
        self.verbose = verbose or _to_bool(self.config.get('VERBOSE', os.getenv('VERBOSE','0')), False)
        self.exec = exec_client
        self._last_seen_len = None            # legacy (tidak dipakai lagi untuk skip)
        self.last_bar_close_ts = None         # <- tambah: jejak bar terakhir (UTC close timestamp)
        self.startup_skip_bars = int(self.config.get('startup_skip_bars', 2))
        self.post_restart_skip_entries_bars = int(self.config.get('post_restart_skip_entries_bars', 1))
        self.pending_skip_entries = self.startup_skip_bars
        self.rehydrated = False
        self.rehydrate_protect_profit = bool(self.config.get('rehydrate_protect_profit', True))
        self.rehydrate_profit_min_pct = float(self.config.get('rehydrate_profit_min_pct', 0.0005))
        self.signal_confirm_bars_after_restart = int(self.config.get('signal_confirm_bars_after_restart', 2))
        self.signal_flip_confirm_left = 0

    def _log(self, msg: str) -> None:
        if getattr(self, 'verbose', False):
            print(f"[{self.instance_id}:{self.symbol}] {pd.Timestamp.utcnow().isoformat()} | {msg}")

    def _clamp_pos(self, x: float, min_x: float = 1e-9) -> float:
        return x if (x is not None and x > min_x) else min_x

    def _clear_position_state(self, cancel_orders: bool = False) -> None:
        self.pos = Position()
        if cancel_orders and self.exec:
            try:
                self.exec.cancel_all(self.symbol)
            except Exception:
                pass

    # Hook: ganti dengan sumber data live kamu
    def fetch_recent_klines(self) -> pd.DataFrame:
        """Return DataFrame dengan kolom: timestamp, open, high, low, close, volume"""
        raise NotImplementedError("Implement fetch_recent_klines() sesuai exchange kamu.")

    def _on_new_bar(self, last_close_ts):
        """
        Dipanggil tiap loop dengan timestamp CLOSE bar terbaru (sudah close).
        last_close_ts bisa int epoch ms atau pd.Timestamp.
        """
        # Normalisasi ke int epoch ms agar perbandingan konsisten
        try:
            if hasattr(last_close_ts, "value"):  # pd.Timestamp
                last_ts = int(last_close_ts.value // 1_000_000)  # ns -> ms
            else:
                last_ts = int(last_close_ts)
        except Exception:
            # fallback: biarkan apa adanya
            last_ts = last_close_ts

        if self.last_bar_close_ts is None:
            self.last_bar_close_ts = last_ts
            return

        if last_ts > self.last_bar_close_ts:
            # terdeteksi bar baru yang SUDAH tutup
            self.last_bar_close_ts = last_ts
            if self.pending_skip_entries > 0:
                self.pending_skip_entries = max(0, self.pending_skip_entries - 1)
                self._log(f"STARTUP SKIP: tunda entry {self.pending_skip_entries} bar lagi")
            if self.signal_flip_confirm_left > 0:
                self.signal_flip_confirm_left = max(0, self.signal_flip_confirm_left - 1)

    def _safe_trailing_params(self) -> Tuple[float, float]:
        taker_fee = _to_float(self.config.get('taker_fee', DEFAULTS['taker_fee']), DEFAULTS['taker_fee'])
        slippage_pct = _to_float(self.config.get('SLIPPAGE_PCT', ENV_DEFAULTS['SLIPPAGE_PCT']), ENV_DEFAULTS['SLIPPAGE_PCT'])
        roundtrip_fee_pct = taker_fee * 2.0 * 100.0
        roundtrip_slip_pct = slippage_pct * 2.0
        safe_buffer_pct = roundtrip_fee_pct + roundtrip_slip_pct + 0.05
        trailing_trigger = _to_float(
            self.config.get('trailing_trigger', DEFAULTS['trailing_trigger']),
            DEFAULTS['trailing_trigger']
        )
        if 'trailing_step' in self.config:
            trailing_step_val = self.config.get('trailing_step')
        else:
            fallback_ts = self.config.get('trailing_step_min_pct', self.config.get('trail', {}).get('min_step_pct'))
            trailing_step_val = fallback_ts if fallback_ts is not None else DEFAULTS['trailing_step']
        trailing_step = self._clamp_pos(float(trailing_step_val), float(self.config.get('trailing_step_min', 1e-9)))
        safe_trigger = max(trailing_trigger, safe_buffer_pct + trailing_step)
        return safe_trigger, trailing_step

    def _size_position(self, price: float, sl: float, balance: float) -> float:
        risk_usdt = float(balance) * _to_float(self.config.get('risk_per_trade', DEFAULTS['risk_per_trade']), DEFAULTS['risk_per_trade'])
        dist = abs(price - sl)
        # gunakan nilai minimal agar tidak nol
        dist = max(dist, max(_to_float(self.config.get('tickSize', 0.0), 0.0), 1e-12))
        qty = safe_div(risk_usdt, dist)
        if self.exec:
            qty = self.exec.round_qty(self.symbol, as_scalar(qty))
        else:
            step = _to_float(self.config.get('stepSize', 0.0), 0.0)
            qty = floor_to_step(qty, step) if step>0 else qty
            prec = _to_int(self.config.get('quantityPrecision', 0), 0)
            try:
                qty = float(f"{qty:.{prec}f}")
            except Exception:
                qty = float(qty)
        minq = _to_float(self.config.get('minQty', 0.0), 0.0)
        return qty if qty >= minq else 0.0

    def _hard_sl_price(self, entry: float, atr: float, side: str) -> float:
        mode = str(self.config.get('sl_mode', DEFAULTS['sl_mode'])).upper()
        sl_min = _to_float(self.config.get('sl_min_pct', DEFAULTS['sl_min_pct']), DEFAULTS['sl_min_pct'])
        sl_max = _to_float(self.config.get('sl_max_pct', DEFAULTS['sl_max_pct']), DEFAULTS['sl_max_pct'])
        if mode == 'PCT':
            sl_pct = _to_float(self.config.get('sl_pct', DEFAULTS['sl_pct']), DEFAULTS['sl_pct'])
        else:
            sl_atr_mult = _to_float(self.config.get('sl_atr_mult', DEFAULTS['sl_atr_mult']), DEFAULTS['sl_atr_mult'])
            sl_pct = (sl_atr_mult * safe_div(atr, entry)) if atr>0 else _to_float(self.config.get('sl_pct', DEFAULTS['sl_pct']), DEFAULTS['sl_pct'])
        sl_pct = max(sl_min, min(sl_pct, sl_max))
        return entry * (1 - sl_pct) if side=='LONG' else entry * (1 + sl_pct)

    def _apply_breakeven(self, price: float) -> None:
        if not _to_bool(self.config.get('use_breakeven', DEFAULTS['use_breakeven']), DEFAULTS['use_breakeven']):
            return
        if not (self.pos.side and self.pos.entry is not None):
            return
        tick = _to_float(self.config.get('tickSize', 0.0), 0.0)
        new_sl = apply_breakeven_sl(
            side=self.pos.side,
            entry=self.pos.entry,
            price=price,
            sl=self.pos.sl,
            tick_size=tick,
            min_gap_pct=_to_float(self.config.get('be_min_gap_pct', 0.0001), 0.0001),
            be_trigger_r=_to_float(self.config.get('be_trigger_r', 0.0), 0.0),
            be_trigger_pct=_to_float(self.config.get('be_trigger_pct', DEFAULTS['be_trigger_pct']),
                                     DEFAULTS['be_trigger_pct'])
        )
        self.pos.sl = new_sl
        tick = _to_float(self.config.get('tickSize', 0.0), 0.0)
        if tick > 0 and self.pos.entry is not None and self.pos.sl is not None:
            if self.pos.side == 'LONG':
                self.pos.sl = max(self.pos.sl, self.pos.entry + 2.0 * tick)
            else:
                self.pos.sl = min(self.pos.sl, self.pos.entry - 2.0 * tick)

    def _update_trailing(self, price: float) -> None:
        safe_trigger, step = self._safe_trailing_params()
        entry = self.pos.entry
        if not (entry is not None and self.pos.side and self.pos.allow_trailing):
            return
        if self.pos.side=='LONG':
            profit_pct = safe_div((price - float(entry)), float(entry)) * 100.0
            if profit_pct >= safe_trigger:
                new_ts = price * (1 - step/100.0)
                prev = self.pos.trailing_sl
                self.pos.trailing_sl = max(self.pos.trailing_sl or self.pos.sl or 0.0, new_ts)
                if self.pos.trailing_sl != prev:
                    self._log(f"TRAIL LONG -> {self.pos.trailing_sl:.6f} (prev={prev})")
                    if self.exec and self.pos.side and self.pos.qty and self.pos.trailing_sl:
                        try:
                            self.exec.cancel_all(self.symbol)
                        except Exception:
                            pass
                        stop_side = "SELL"
                        cid = f"x-{self.instance_id}-{self.symbol}-{uuid4().hex[:8]}"
                        self.exec.stop_market(
                            self.symbol,
                            stop_side,
                            self.pos.trailing_sl,
                            reduce_only=True,
                            close_all=True,
                            client_id=cid,
                        )
        else:
            profit_pct = safe_div((float(entry) - price), float(entry)) * 100.0
            if profit_pct >= safe_trigger:
                new_ts = price * (1 + step/100.0)
                prev = self.pos.trailing_sl
                self.pos.trailing_sl = min(self.pos.trailing_sl or self.pos.sl or 1e18, new_ts)
                if self.pos.trailing_sl != prev:
                    self._log(f"TRAIL SHORT -> {self.pos.trailing_sl:.6f} (prev={prev})")
                    if self.exec and self.pos.side and self.pos.qty and self.pos.trailing_sl:
                        try:
                            self.exec.cancel_all(self.symbol)
                        except Exception:
                            pass
                        stop_side = "BUY"
                        cid = f"x-{self.instance_id}-{self.symbol}-{uuid4().hex[:8]}"
                        self.exec.stop_market(
                            self.symbol,
                            stop_side,
                            self.pos.trailing_sl,
                            reduce_only=True,
                            close_all=True,
                            client_id=cid,
                        )

    def _calc_sl_on_attach(self, live: LivePos, indicators: dict) -> float:
        mode = (self.config.get("manual_guard", {}).get("sl_on_attach_mode", "ATR") or "ATR").upper()
        entry = live.entry_price
        atr = float(indicators.get("atr") or 0)
        sl_min = float(self.config.get("manual_guard", {}).get("sl_min_pct", 0.010))
        sl_max = float(self.config.get("manual_guard", {}).get("sl_max_pct", 0.030))
        if mode == "PCT" or atr <= 0:
            sl_pct = float(self.config.get("manual_guard", {}).get("sl_on_attach_pct", 0.012))
        else:
            mult = float(self.config.get("manual_guard", {}).get("sl_on_attach_atr_mult", 1.6))
            sl_pct = (mult * (atr / entry)) if entry else 0.0
        sl_pct = max(sl_min, min(sl_pct, sl_max))
        if live.side == "LONG":
            return entry * (1 - sl_pct)
        else:
            return entry * (1 + sl_pct)

    def reconcile_manual_position(self, kline: dict, indicators: dict) -> None:
        if not self.config.get("adopt_manual_positions", 0):
            return
        if not self.exec:
            return
        live = self.exec.get_live_position(self.symbol)
        have_state = bool(self.pos and self.pos.qty > 0)
        if (live is not None) and (not have_state):
            self._log(f"[{self.symbol}] ADOPT manual position: side={live.side} qty={live.qty} entry={live.entry_price}")
            self.pos = Position(
                side=live.side,
                qty=live.qty,
                entry=live.entry_price,
                entry_time=pd.Timestamp.utcnow(),
                allow_trailing=True,
            )
            if self.config.get("manual_guard", {}).get("place_sl_on_attach", 1):
                if not self.exec.has_active_sl_close_all(self.symbol):
                    stop = self._calc_sl_on_attach(live, indicators)
                    if stop is not None:
                        ok = self.exec.place_protective_sl_close_all(self.symbol, live.side, stop)
                        if ok:
                            self._log(f"[{self.symbol}] Protective SL placed on attach @ {stop}")
                        else:
                            self._log(f"[{self.symbol}] FAILED place protective SL on attach")
            return
        if have_state and (live is None):
            self._log(f"[{self.symbol}] Live flat but state had pos -> clearing state")
            self._clear_position_state(cancel_orders=True)
            return
        if have_state and (live is not None):
            if abs(live.qty - self.pos.qty) > 1e-12:
                self._log(f"[{self.symbol}] Live qty changed: {self.pos.qty} -> {live.qty}")
                self.pos.qty = live.qty

    def _cooldown_active(self) -> bool:
        return bool(self.cooldown_until_ts and time.time() < self.cooldown_until_ts)

    def _enter_position(self, side: str, price: float, atr: float, atr_pct: float, available_balance: float) -> float:
        if self._cooldown_active():
            return 0.0
        if self.exec and self.account_guard and self.exec.has_position(self.symbol):
            self._log("SKIP entry: posisi masih terbuka di akun")
            return 0.0

        side_binance = "BUY" if side == "LONG" else "SELL"
        lev = _to_int(self.config.get('leverage', DEFAULTS['leverage']), DEFAULTS['leverage'])
        risk = float(self.config.get('risk_per_trade', DEFAULTS['risk_per_trade']))
        fee_buf = float(self.config.get('fee_buffer_pct', 0.001))
        atr_pct_val = as_float(atr_pct, 0.0)
        estimated_roi_future = as_float(self.config.get('weak_if_roi_future_probe', atr_pct_val * lev), atr_pct_val * lev)
        rules = self.config.get('strength_rules', {}) or {}
        weak_if_atr = _to_float(rules.get('weak_if_atr_pct_lt', 0.01), 0.01)
        weak_if_roi = _to_float(rules.get('weak_if_roi_future_lt', 0.15), 0.15)
        is_weak = False
        if str(self.config.get('tp_mode', '')).lower() == 'dual_strength':
            if atr_pct_val < weak_if_atr or estimated_roi_future < weak_if_roi:
                is_weak = True
        weak_tp_roi_pct = _to_float(self.config.get('weak_tp_roi_pct', 0.10), 0.10)
        use_trail_strong = _to_bool(self.config.get('use_trailing_on_strong', 1), True)

        mt = str(self.config.get('margin_type', 'ISOLATED')).upper()
        try:
            if self.exec:
                self.exec.set_margin_type(self.symbol, isolated=mt)
                self.exec.set_leverage(self.symbol, lev)
        except Exception as e:
            self._log(f"[WARN] set_margin_type({mt}) gagal: {e}")

        margin = max(0.0, available_balance) * max(0.0, risk)
        if margin <= 0.0:
            self._log(f"SKIP entry: available={available_balance:.4f}, margin=0")
            return 0.0

        raw_qty = safe_div((margin * lev), price)
        if self.exec:
            qty = self.exec.round_qty(self.symbol, as_scalar(raw_qty))
        else:
            step = _to_float(self.config.get('stepSize', 0.0), 0.0)
            qty = floor_to_step(raw_qty, step) if step > 0 else raw_qty
        if qty <= 0.0:
            self._log(f"SKIP entry: qty setelah pembulatan LOT_SIZE = 0 (raw={raw_qty:.8f})")
            return 0.0

        need = safe_div((price * qty), lev) * (1.0 + fee_buf)
        try:
            live_avail = self.exec.get_available_balance() if self.exec else available_balance
        except Exception:
            live_avail = available_balance
        if need > max(live_avail, 0.0):
            max_qty_by_margin = safe_div(safe_div(live_avail, lev), price) * 0.95
            if self.exec:
                shrunk = self.exec.round_qty(self.symbol, as_scalar(max(0.0, max_qty_by_margin)))
            else:
                step = _to_float(self.config.get('stepSize', 0.0), 0.0)
                shrunk = floor_to_step(max(0.0, max_qty_by_margin), step) if step > 0 else max(0.0, max_qty_by_margin)
            if shrunk <= 0:
                self._log(f"SKIP entry: need={need:.4f} > available={live_avail:.4f}")
                return 0.0
            self._log(f"SHRINK qty {qty:.6f} -> {shrunk:.6f} (avail={live_avail:.4f}, need={need:.4f})")
            qty = shrunk
            need = safe_div((price * qty), lev) * (1.0 + fee_buf)

        min_not = float(self.config.get("minNotional", 0.0) or 0.0)
        if min_not > 0 and price * qty < min_not:
            step = self.exec.get_step_size(self.symbol) if self.exec else _to_float(self.config.get("stepSize", 0.0), 0.0)
            need_qty = safe_div(min_not, price)
            if step > 0:
                need_qty = ceil_to_step(need_qty, step)
            if self.exec:
                need_qty = self.exec.round_qty(self.symbol, as_scalar(need_qty))
            if to_scalar(safe_div((need_qty * price), lev) * (1.0 + fee_buf)) > live_avail:
                self._log(f"SKIP entry: below MIN_NOTIONAL {min_not}, avail={live_avail:.2f}")
                return 0.0
            qty = need_qty

        sl = self._hard_sl_price(price, atr, side)
        dist = abs(price - sl)
        tick = _to_float(self.config.get('tickSize', 0.0), 0.0)
        eps = tick if tick > 0 else max(price, 1e-9) * 1e-6
        if not math.isfinite(dist) or dist <= eps:
            self._log(f"SKIP entry: SL==entry (dist={dist:.12f})")
            return 0.0

        try:
            cid = f"x-{os.getenv('INSTANCE_ID', 'bot')}-{self.symbol}-{uuid4().hex[:8]}"
            od = self.exec.market_entry(self.symbol, side_binance, qty, client_id=cid) if self.exec else {}
            if not od or not od.get('orderId'):
                return 0.0
            fill_price = float(od.get('avgPrice') or od.get('price') or price)
            self.pos = Position(
                side=side,
                entry=fill_price,
                qty=qty,
                sl=sl,
                trailing_sl=None,
                entry_time=pd.Timestamp.utcnow(),
                allow_trailing=False,
            )
            self._log(f"ENTRY {side} price={price} qty={qty:.6f}")
            if sl and self.exec:
                stop_side = 'SELL' if side == 'LONG' else 'BUY'
                self.exec.stop_market(
                    self.symbol,
                    stop_side,
                    sl,
                    reduce_only=True,
                    close_all=True,
                    client_id=cid + "-sl",
                )
                if is_weak:
                    if self.pos.side == 'LONG':
                        tp_price = self.pos.entry * (1.0 + weak_tp_roi_pct / lev)
                        stop_side = "SELL"
                    else:
                        tp_price = self.pos.entry * (1.0 - weak_tp_roi_pct / lev)
                        stop_side = "BUY"
                    cid_tp = f"x-{self.instance_id}-{self.symbol}-{uuid4().hex[:8]}"
                    self.exec.stop_market(
                        self.symbol,
                        stop_side,
                        tp_price,
                        reduce_only=True,
                        close_all=True,
                        order_type="TAKE_PROFIT_MARKET",
                        client_id=cid_tp + "-tp",
                    )
                else:
                    self.pos.allow_trailing = bool(use_trail_strong)
            return safe_div((price * qty), lev)
        except BinanceAPIException as e:
            if getattr(e, "code", None) == -2019:
                self._log("WARN: Margin insufficient. Cooldown & skip.")
                self.cooldown_until_ts = time.time() + int(self.config.get("cooldown_seconds", DEFAULTS.get("cooldown_seconds", 60)))
                return 0.0
            self._log(f"error MARKET ENTRY: {e}")
            return 0.0
        except Exception as e:
            self._log(f"error MARKET ENTRY: {e}")
            return 0.0

    def _should_exit(self, price: float) -> Tuple[bool, Optional[str]]:
        if not self.pos.side:
            return False, None
        # trailing/hard SL
        if self.pos.side=='LONG':
            if self.pos.trailing_sl is not None and price <= self.pos.trailing_sl:
                return True, 'Hit Trailing SL'
            if (self.pos.sl is not None) and price <= self.pos.sl:
                return True, 'Hit Hard SL'
        else:
            if self.pos.trailing_sl is not None and price >= self.pos.trailing_sl:
                return True, 'Hit Trailing SL'
            if (self.pos.sl is not None) and price >= self.pos.sl:
                return True, 'Hit Hard SL'
        return False, None

    def _force_close_residual(self) -> None:
        if not self.exec:
            return
        live = self.exec.get_position(self.symbol)
        if not live or not live.qty:
            return
        flt = self.exec.symbol_filters.get(self.symbol, {}) if self.exec else {}
        step = _to_float(flt.get('stepSize', self.config.get('stepSize', 0.0)), 0.0)
        minq = _to_float(flt.get('minQty', self.config.get('minQty', 0.0)), 0.0)
        raw = abs(live.qty)
        if step > 0:
            from engine_core import ceil_to_step
            qty_close = max(minq, ceil_to_step(raw, step))
        else:
            qty_close = max(minq, raw)
        close_side = 'SELL' if live.side == 'LONG' else 'BUY'
        cid = f"x-{self.instance_id}-{self.symbol}-{uuid4().hex[:8]}"
        try:
            self.exec.market_close(self.symbol, close_side, qty_close, client_id=cid + "-rescue")
        except Exception:
            try:
                last = self.exec.get_last_price(self.symbol)
                stop_price = last * (0.999 if close_side == 'SELL' else 1.001)
                self.exec.stop_market(
                    self.symbol,
                    close_side,
                    stop_price,
                    reduce_only=True,
                    close_all=True,
                    client_id=cid + "-rescue-stop",
                )
            except Exception:
                pass

    def _exit_position(self, price: float, reason: str) -> None:
        if self.exec and self.pos.side and self.pos.qty:
            try:
                self.exec.cancel_all(self.symbol)
            except Exception:
                pass
            close_side = 'SELL' if self.pos.side == 'LONG' else 'BUY'
            cid = f"x-{self.instance_id}-{self.symbol}-{uuid4().hex[:8]}"
            self.exec.market_close(self.symbol, close_side, self.pos.qty, client_id=cid)
            self._force_close_residual()
        self._log(f"EXIT {reason} price={price:.6f}")
        self.pos = Position()  # reset
        base_cd = _to_int(self.config.get('cooldown_seconds', DEFAULTS['cooldown_seconds']), DEFAULTS['cooldown_seconds'])
        mul = 1.0
        if 'Hard SL' in (reason or ''):
            mul = _to_float(self.config.get('cooldown_mul_hard_sl', 1.0), 1.0)
        elif 'Trailing' in (reason or ''):
            mul = _to_float(self.config.get('cooldown_mul_trailing', 1.0), 1.0)
        elif 'Max hold' in (reason or '') or 'early' in (reason or '').lower():
            mul = _to_float(self.config.get('cooldown_mul_early_stop', 1.0), 1.0)
        cd = max(0, int(base_cd * mul))
        self.cooldown_until_ts = time.time() + cd
        try:
            cd_ts = float(self.cooldown_until_ts) if self.cooldown_until_ts is not None else None
            if cd_ts is not None:
                self._log(f"COOLDOWN until {pd.Timestamp.utcfromtimestamp(cd_ts).isoformat()}")
        except Exception:
            pass

    def check_trading_signals(self, df_raw: pd.DataFrame, available_balance: float) -> float:
        if df_raw is None or df_raw.empty:
            return 0.0
        heikin = _to_bool(self.config.get('heikin', False), False)
        df = calculate_indicators(df_raw, heikin=heikin)
        # Ambil CLOSE timestamp terakhir dari kolom 'timestamp' bila tersedia
        if 'timestamp' in df.columns:
            last_ts = df['timestamp'].iloc[-1]
        else:
            last_ts = df.index[-1]
        self._on_new_bar(last_ts)
        last = df.iloc[-1]
        price = as_float(last.get('close'), 0.0)
        htf = str(self.config.get('htf', '1h'))
        atr = as_float(last.get('atr'), 0.0)
        if not math.isfinite(price):
            raise ValueError("Non-finite price")

        self.reconcile_manual_position(kline=last.to_dict(), indicators=last.to_dict())

        up_prob = None
        if self.ml.use_ml:
            self.ml.fit_if_needed(df)
            up_prob = self.ml.predict_up_prob(df)
            if up_prob is not None:
                up_prob = to_scalar(up_prob)

        try:
            if self.pending_skip_entries > 0:
                self._apply_breakeven(price)
                self._update_trailing(price)
                return 0.0

            if self.pos.side:
                self._apply_breakeven(price)
                self._update_trailing(price)
                ex, rs = self._should_exit(price)
                if ex:
                    self._exit_position(price, rs or 'Exit')
                    return 0.0

            filters_cfg = (self.config.get('filters') if isinstance(self.config.get('filters'), dict) else {}) or {}
            min_atr_threshold = _to_float(filters_cfg.get('min_atr_threshold', self.config.get('min_atr_pct', DEFAULTS['min_atr_pct'])), DEFAULTS['min_atr_pct'])
            max_body_over_atr = _to_float(filters_cfg.get('max_body_over_atr', self.config.get('max_body_atr', DEFAULTS['max_body_atr'])), DEFAULTS['max_body_atr'])

            atr_ok = (last['atr_pct'] >= min_atr_threshold) and (
                last['atr_pct'] <= _to_float(self.config.get('max_atr_pct', DEFAULTS['max_atr_pct']), DEFAULTS['max_atr_pct'])
            )
            body_val = last.get('body_to_atr', last.get('body_atr'))
            body_ok = (as_float(body_val) <= max_body_over_atr)
            if self.verbose and (not atr_ok or not body_ok):
                print(f"[{self.symbol}] FILTER INFO atr_ok={atr_ok} body_ok={body_ok} price={price} pos={self.pos.side or 'None'}")

            # HTF filter (opsional)
            if _to_bool(self.config.get('use_htf_filter', DEFAULTS['use_htf_filter']), DEFAULTS['use_htf_filter']):
                if last['ema_22'] > last['ma_22'] and not htf_trend_ok('LONG', df, htf=htf):
                    long_htf_ok = False
                else:
                    long_htf_ok = True
                if last['ema_22'] < last['ma_22'] and not htf_trend_ok('SHORT', df, htf=htf):
                    short_htf_ok = False
                else:
                    short_htf_ok = True
            else:
                long_htf_ok = short_htf_ok = True

            long_base, short_base = compute_base_signals_backtest(df)
            long_base = long_base and long_htf_ok
            short_base = short_base and short_htf_ok

            if self.rehydrated and self.pos.side:
                lev = _to_int(self.config.get('leverage', DEFAULTS['leverage']), DEFAULTS['leverage'])
                entry_safe = as_float(self.pos.entry, 0.0)
                price_safe = as_float(price, 0.0)
                qty_safe = as_float(self.pos.qty, 0.0)
                if entry_safe == 0.0 or qty_safe == 0.0:
                    roi = 0.0
                else:
                    roi = roi_frac_now(self.pos.side, entry_safe, price_safe, qty_safe, lev)
                if roi >= self.rehydrate_profit_min_pct and self.rehydrate_protect_profit:
                    if base_supports_side(long_base, short_base, self.pos.side):
                        self._log("REHYDRATE PROTECT: profit & signal searah → tahan posisi (skip close by-signal)")
                        self.signal_flip_confirm_left = max(self.signal_flip_confirm_left, self.signal_confirm_bars_after_restart)
                    else:
                        pass

            if self.signal_flip_confirm_left > 0:
                self._log(f"SIGNAL CONFIRM: menunggu {self.signal_flip_confirm_left} bar utk validasi flip")
                if self.pos.side:
                    self._apply_breakeven(price)
                    self._update_trailing(price)
                return 0.0

            decision = make_decision(df, self.symbol, self.config, up_prob)
            long_sig = decision == 'LONG'
            short_sig = decision == 'SHORT'
            # Update SL/TS saat pegang posisi
            if self.pos.side:
                self._apply_breakeven(price)
                self._update_trailing(price)
                ex, reason = self._should_exit(price)
                if ex:
                    self._exit_position(price, reason or 'Exit')
                    return 0.0

            # ---- Time-stop (selaras backtest) ----
            max_hold = _to_int(self.config.get("max_hold_seconds", DEFAULTS.get("max_hold_seconds", 3600)), 3600)
            min_roi  = _to_float(self.config.get("min_roi_to_close_by_time", DEFAULTS.get("min_roi_to_close_by_time", 0.005)), 0.005)

            if self.pos.side and self.pos.entry_time and max_hold > 0:
                elapsed = (pd.Timestamp.utcnow() - self.pos.entry_time).total_seconds()
                lev = _to_int(self.config.get("leverage", DEFAULTS["leverage"]), DEFAULTS["leverage"])
                entry = self.pos.entry
                qty = self.pos.qty
                roi_frac = 0.0
                if entry is not None and qty is not None:
                    init_margin = safe_div(float(entry) * float(qty), lev)
                    if self.pos.side == "LONG":
                        roi_frac = safe_div((price - float(entry)) * float(qty), init_margin)
                    else:
                        roi_frac = safe_div((float(entry) - price) * float(qty), init_margin)
                else:
                    init_margin = 0.0

                only_if_loss = _to_bool(self.config.get('time_stop_only_if_loss', 0), False)
                if elapsed >= max_hold:
                    if only_if_loss:
                        if roi_frac <= 0:
                            self._exit_position(price, f"Max hold reached (loss, ROI {roi_frac*100:.2f}%)")
                            return 0.0
                        else:
                            self.pos.entry_time = pd.Timestamp.utcnow()
                    else:
                        if roi_frac >= min_roi:
                            self._exit_position(price, f"Max hold reached (ROI {roi_frac*100:.2f}%)")
                            return 0.0
                        else:
                            self.pos.entry_time = pd.Timestamp.utcnow()
            # --------------------------------------
            used = 0.0
            # Entry baru
            if not self.pos.side and not self._cooldown_active():
                if long_sig:
                    used = self._enter_position('LONG', price, atr, as_float(last.get('atr_pct'), 0.0), available_balance)
                elif short_sig:
                    used = self._enter_position('SHORT', price, atr, as_float(last.get('atr_pct'), 0.0), available_balance)
            return used or 0.0
        except ZeroDivisionError as e:
            self._log(f"[{self.symbol}] ZDIV DIAG: price={price} atr={atr} entry={getattr(self.pos,'entry',None)} step={self.config.get('trailing_step')}")
            raise


# ============================
# Manager (contoh sederhana)
# ============================
class TradingManager:
    def __init__(self, coin_config_path: str, symbols: List[str], *, instance_id: str="bot", account_guard: bool=False, exec_client: Optional[ExecutionClient]=None, verbose: bool=False, no_atr_filter: bool=False, no_body_filter: bool=False, ml_override: bool=False):
        self.coin_config_path = coin_config_path
        raw_syms = [s.upper() for s in symbols]
        self.exec_client = exec_client
        self.verbose = verbose
        self.symbols: List[str] = []
        for s in raw_syms:
            if self.exec_client and not self.exec_client.has_symbol(s):
                print(f"[WARN] {s} tidak ada di USDM Futures → di-skip")
                continue
            self.symbols.append(s)
        self._cfg = load_coin_config(coin_config_path)
        self.traders: Dict[str, CoinTrader] = {}
        for s in self.symbols:
            merged_cfg = merge_config(s, self._cfg)
            if no_atr_filter:
                merged_cfg.setdefault('filters', {})['atr_filter_enabled'] = False
            if no_body_filter:
                merged_cfg.setdefault('filters', {})['body_filter_enabled'] = False
            self.traders[s] = CoinTrader(s, merged_cfg, instance_id=instance_id, account_guard=account_guard, verbose=self.verbose, exec_client=self.exec_client)
        self.boot_time = pd.Timestamp.utcnow()
        self._stop = False
        t = threading.Thread(target=self._watch_config, daemon=True)
        t.start()

    def _rehydrate_from_exchange(self, trader: CoinTrader):
        """Jika ada posisi di exchange tapi state lokal kosong, rehydrate agar trailing/stop lanjut."""
        if not trader.exec:
            return
        if trader.pos and trader.pos.qty:
            return
        info = trader.exec.position_info(trader.symbol)
        if not info:
            return
        qty = float(info.get('positionAmt', 0) or 0)
        if abs(qty) < 1e-12:
            return
        entry_price = float(info.get('entryPrice', 0) or 0)
        side = 'LONG' if qty > 0 else 'SHORT'
        qty = abs(qty)
        orders = trader.exec.open_orders(trader.symbol) or []
        sl_price = None
        for o in orders:
            if o.get('type') == 'STOP_MARKET' and o.get('reduceOnly'):
                sp = float(o.get('stopPrice', 0) or 0)
                if sp > 0:
                    sl_price = sp
                    break
        trader.pos.side = side
        trader.pos.entry = entry_price
        trader.pos.qty = qty
        trader.pos.entry_time = pd.Timestamp.utcnow()
        trader.pos.sl = sl_price
        trader.pos.trailing_sl = sl_price
        if not trader.rehydrated:
            trader.rehydrated = True
            trader.pending_skip_entries = max(trader.pending_skip_entries, trader.post_restart_skip_entries_bars)
        trader._log(f"[SYNC] Rehydrate posisi dari exchange side={side} entry={entry_price} qty={qty} sl={sl_price}")

    def _watch_config(self):
        last_ts = 0.0
        safe_keys = {
            "risk_per_trade","leverage","trailing_trigger","trailing_step","taker_fee",
            "min_atr_pct","max_atr_pct","max_body_atr","use_htf_filter","cooldown_seconds",
            "allow_sar","reverse_confirm_bars","min_hold_seconds","max_hold_seconds","min_roi_to_close_by_time",
            # ML & scoring
            "USE_ML","SCORE_THRESHOLD","ML_MIN_TRAIN_BARS","ML_LOOKAHEAD",
            "ML_RETRAIN_EVERY","ML_UP_PROB","ML_DOWN_PROB",
            # precision
            "stepSize","minQty","quantityPrecision"
        }
        while not self._stop:
            try:
                ts = os.path.getmtime(self.coin_config_path)
                if ts != last_ts:
                    last_ts = ts
                    cfg_all = load_coin_config(self.coin_config_path)
                    for sym, trader in self.traders.items():
                        new_cfg = merge_config(sym, cfg_all)
                        float_keys = {
                            "risk_per_trade","trailing_trigger","trailing_step","taker_fee",
                            "min_atr_pct","max_atr_pct","max_body_atr","min_roi_to_close_by_time",
                            "stepSize","minQty"
                        }
                        int_keys = {
                            "leverage","cooldown_seconds","reverse_confirm_bars","min_hold_seconds","max_hold_seconds",
                            "ML_MIN_TRAIN_BARS","ML_LOOKAHEAD","ML_RETRAIN_EVERY","quantityPrecision"
                        }
                        bool_keys = {"use_htf_filter","allow_sar","USE_ML"}
                        # hanya update key aman dengan tipe yang tepat
                        for k, v in list(new_cfg.items()):
                            if k in safe_keys:
                                if k in float_keys:
                                    trader.config[k] = _to_float(v, _to_float(trader.config.get(k, 0.0), 0.0))
                                elif k in int_keys:
                                    trader.config[k] = _to_int(v, _to_int(trader.config.get(k, 0), 0))
                                elif k in bool_keys:
                                    trader.config[k] = _to_bool(v)
                                else:
                                    trader.config[k] = v
                        # sinkron threshold ke plugin
                        ml_cfg = trader.config.get("ml", {}) if isinstance(trader.config.get("ml"), dict) else {}
                        if "score_threshold" in ml_cfg:
                            trader.ml.params.score_threshold = _to_float(ml_cfg.get("score_threshold"), ENV_DEFAULTS["SCORE_THRESHOLD"])
                        else:
                            trader.ml.params.score_threshold = _to_float(trader.config.get("SCORE_THRESHOLD", ENV_DEFAULTS["SCORE_THRESHOLD"]), ENV_DEFAULTS["SCORE_THRESHOLD"])
                time.sleep(1.0)
            except Exception:
                time.sleep(2.0)

    # Hook: implement loop fetch + dispatch ke trader
    def run_once(self, data_map: Dict[str, pd.DataFrame], _balances_unused=None):
        step_available = 0.0
        any_trader = next(iter(self.traders.values()))
        exec_client = getattr(any_trader, "exec", None)
        if exec_client is not None:
            try:
                step_available = exec_client.get_available_balance()
            except Exception:
                step_available = 0.0
        for sym in self.symbols:
            trader = self.traders.get(sym)
            df = data_map.get(sym)
            if trader is None or df is None or len(df) == 0:
                continue
            try:
                self._rehydrate_from_exchange(trader)
                used_margin = trader.check_trading_signals(df, step_available)
                if used_margin and used_margin > 0:
                    before = step_available
                    step_available = max(0.0, step_available - used_margin)
                    print(f"[{sym}] used_margin={used_margin:.4f} balance_before={before:.4f} -> after={step_available:.4f}")
            except Exception as e:
                print(f"[{sym}] error: {e}")
                continue

    def stop(self):
        self._stop = True


# ============================
# Contoh penggunaan (dummy)
# ============================
if __name__ == "__main__":
    # contoh minimal: jalankan 1x pakai CSV lokal untuk verifikasi logika
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--csv", required=False, help="Path CSV OHLCV (timestamp, open, high, low, close, volume)")
    ap.add_argument("--symbol", default="ADAUSDT")
    ap.add_argument("--balance", type=float, default=20.0)
    ap.add_argument("--verbose", action="store_true", help="Print log keputusan & aksi")
    ap.add_argument("--live", action="store_true", help="Jalankan real-trading live (REST polling)")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols; default = --symbol")
    ap.add_argument("--interval", default=os.getenv("INTERVAL","15m"))
    ap.add_argument("--dry-run-loop", action="store_true", help="Replay CSV bar-by-bar (simulasi real-time)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Delay per step (detik) saat dry-run")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah langkah dry-run (0 = semua)")
    ap.add_argument("--real-exec", action="store_true", help="Jika true, submit order ke Binance Futures (USDT-M). Default off untuk safety.")
    ap.add_argument("--testnet", action="store_true", help="Gunakan Binance Futures Testnet (disarankan uji dulu).")
    ap.add_argument("--instance-id", default="bot", help="ID instance unik (untuk penamaan order/log)")
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--account-guard", action="store_true")
    ap.add_argument("--no-atr-filter", action="store_true", help="Disable ATR/body candle filter (ML always allowed)")
    ap.add_argument("--ml-override", action="store_true", help="Allow ML to bypass ATR/body filter when triggered")
    args = ap.parse_args()
    if args.verbose:
        os.environ["VERBOSE"] = "1"
    os.environ["INSTANCE_ID"] = args.instance_id
    INSTANCE_ID = args.instance_id

    cfg_all = load_coin_config(args.coin_config) if os.path.exists(args.coin_config) else {}
    cfg = merge_config(args.symbol.upper(), cfg_all)
    if args.no_atr_filter:
        cfg.setdefault('filters', {})['atr_filter_enabled'] = False
    if getattr(args, 'no_body_filter', False):
        cfg.setdefault('filters', {})['body_filter_enabled'] = False

    exec_client = None
    if args.real_exec:
        api_key = os.getenv("BINANCE_API_KEY","")
        api_sec = os.getenv("BINANCE_API_SECRET","")
        exec_client = ExecutionClient(api_key, api_sec, testnet=args.testnet, verbose=args.verbose)

    if not args.logs_dir:
        args.logs_dir = os.path.join("logs", args.instance_id)

    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        if 'timestamp' not in df.columns:
            if 'open_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        sym = args.symbol.upper()

        if args.dry_run_loop:
            # warmup agar indikator & ML cukup data
            try:
                min_train = int(float(os.getenv('ML_MIN_TRAIN_BARS', '400')))
            except Exception:
                min_train = 400
            warmup = max(300, min_train + 10)
            start_i = min(warmup, len(df)-1)
            steps = 0
            mgr = TradingManager(args.coin_config, [sym], instance_id=args.instance_id, account_guard=args.account_guard, exec_client=exec_client, verbose=args.verbose, no_atr_filter=args.no_atr_filter, ml_override=args.ml_override)
            for i in range(start_i, len(df)):
                data_map = {sym: df.iloc[:i+1].copy()}
                mgr.run_once(data_map, {sym: args.balance})
                steps += 1
                if args.limit and steps >= args.limit:
                    break
                if args.sleep and args.sleep > 0:
                    time.sleep(args.sleep)
            last_side = mgr.traders[sym].pos.side if sym in mgr.traders else None
            print(f"Dry-run completed: {steps} steps (start={start_i}, total_bars={len(df)}). Last position: {last_side}")
        else:
            data_map = {sym: df}
            TradingManager(args.coin_config, [sym], instance_id=args.instance_id, account_guard=args.account_guard, exec_client=exec_client, verbose=args.verbose, no_atr_filter=args.no_atr_filter, ml_override=args.ml_override).run_once(data_map, {sym: args.balance})
            print("Run once completed (dummy). Cek log/print sesuai hook eksekusi.")
    else:
        if args.live:
            # ==== LIVE MODE (python-binance) ====
            from binance.client import Client
            import time, pandas as pd

            def _sec(tf: str) -> int:
                tf = tf.strip().lower()
                if tf.endswith("min"): return int(tf[:-3]) * 60
                if tf.endswith("m"):   return int(tf[:-1]) * 60
                if tf.endswith("h"):   return int(tf[:-1]) * 3600
                if tf.endswith("d"):   return int(tf[:-1]) * 86400
                return 900

            api_key = os.getenv("BINANCE_API_KEY","")
            api_sec = os.getenv("BINANCE_API_SECRET","")
            client = exec_client.client if exec_client else Client(api_key, api_sec, testnet=args.testnet)
            if args.testnet and not exec_client:
                client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

            syms = [s.strip().upper() for s in (args.symbols.split(",") if args.symbols else [args.symbol])]
            interval = args.interval
            step = _sec(interval)

            mgr = TradingManager(args.coin_config, syms, instance_id=args.instance_id, account_guard=args.account_guard, exec_client=exec_client, verbose=args.verbose, no_atr_filter=args.no_atr_filter, ml_override=args.ml_override)

            print(f"[LIVE] balance sumber: account availableBalance (bukan arg --balance)")
            print(f"[LIVE] start symbols={','.join(syms)} interval={interval}")
            limiter = RateLimiter(rate=5)
            while True:
                data_map = {}
                for s in syms:
                    limiter.wait()
                    time.sleep(0.3 + random.random() * 0.5)
                    kl = safe_fetch_klines(client, s, interval, 600)
                    # Format ke DataFrame
                    df = pd.DataFrame(kl, columns=[
                        "open_time","open","high","low","close","volume","close_time",
                        "qav","num_trades","taker_base","taker_quote","ignore"
                    ])
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                    for c in ["open","high","low","close","volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df[["timestamp","open","high","low","close","volume"]].dropna()
                    if len(df) > 0:
                        df = df.iloc[:-1].copy()
                    data_map[s] = df

                mgr.run_once(data_map)

                # tidur sampai candle berikutnya + buffer 1s
                now = time.time()
                sleep_for = step - (int(now) % step) + 1
                time.sleep(max(5, sleep_for))
        else:
            print("Tidak ada CSV. Mode dummy selesai.")
