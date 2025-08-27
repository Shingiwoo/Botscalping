from __future__ import annotations

"""
Modul: supply_demand_ws_lux

Supply & Demand (Visible Range) versi Python yang mengikuti metodologi
"Supply and Demand Visible Range [LuxAlgo]":
- Menggunakan *visible range* dari data OHLCV.
- Membangun histogram **volume-per-harga** berbasis *intra-bar* (volume
  tiap candle disebar proporsional ke price-bins yang dilalui [low, high]).
- **Ekspansi zona sekuensial** dari tepi (atas = supply, bawah = demand)
  sampai kumulasi volume mencapai **Threshold %** (mengontrol lebar zona).
- Menghasilkan **average level** (solid) & **weighted average level** (dashed)
  untuk tiap zona seperti deskripsi LuxAlgo.

Integrasi real-time: Binance WebSocket (python-binance >= 1.0.20).

Lisensi: MIT
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any, TYPE_CHECKING, cast
import numpy as np
import pandas as pd
import numpy.typing as npt

try:
    from binance import AsyncClient, BinanceSocketManager
except Exception:  # pragma: no cover
    AsyncClient = None  # type: ignore
    BinanceSocketManager = None  # type: ignore

if TYPE_CHECKING:
    from binance import AsyncClient as _AsyncClient
    from binance import BinanceSocketManager as _BinanceSocketManager
else:  # fallback agar type checker tidak error jika lib tak terpasang
    _AsyncClient = Any  # type: ignore
    _BinanceSocketManager = Any  # type: ignore


# =============================
# Data Structures
# =============================

@dataclass
class Zone:
    label: str  # 'supply' atau 'demand'
    top: float
    bottom: float
    average: float
    weighted_average: float
    volume_ratio: float  # ~ threshold_percent
    start_ts: pd.Timestamp


@dataclass
class ZonesResult:
    supply_zones: List[Zone]
    demand_zones: List[Zone]
    equilibrium: Optional[float]
    weighted_equilibrium: Optional[float]
    visible_start: pd.Timestamp
    visible_end: pd.Timestamp


# =============================
# Inti Indikator
# =============================

class SupplyDemandVisibleRange:
    """Supply & Demand Visible Range (LuxAlgo-style).

    Parameter
    ---------
    threshold_percent : float
        Persentase volume dari visible range untuk lebar zona (0..100).
        Contoh LuxAlgo default umum 10%. Semakin besar → zona lebih lebar.
    resolution : int
        Jumlah price bins (2..500). 50–200 umumnya cukup.
    max_bars_back : int
        Jumlah bar maksimum dipakai untuk visible range.
    """

    def __init__(self, threshold_percent: float = 10.0, resolution: int = 100, max_bars_back: int = 500) -> None:
        if resolution < 2:
            raise ValueError("resolution minimal 2")
        if not (0 <= threshold_percent <= 100):
            raise ValueError("threshold_percent harus 0..100")
        self.threshold_percent = float(threshold_percent)
        self.resolution = int(resolution)
        self.max_bars_back = int(max_bars_back)

    # ---- Utilitas ----
    @staticmethod
    def _weighted_avg(prices: npt.NDArray[np.float_], weights: npt.NDArray[np.float_]) -> float:
        w = float(np.sum(weights))
        if w <= 0 or prices.size == 0:
            return float("nan")
        return float(np.sum(prices * weights) / w)

    # ---- API Utama ----
    def hitung_zona(self, df: pd.DataFrame) -> ZonesResult:
        if df is None or df.empty:
            raise ValueError("DataFrame kosong")
        req = {'open','high','low','close','volume'}
        if not req.issubset(df.columns):
            missing = req - set(df.columns)
            raise ValueError(f"Kolom hilang: {missing}")

        # Visible range (window terakhir)
        dfv = df.iloc[-self.max_bars_back :].copy()
        visible_start, visible_end = dfv.index[0], dfv.index[-1]

        hi = float(dfv['high'].max())
        lo = float(dfv['low'].min())
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            return ZonesResult([], [], None, None, visible_start, visible_end)

        # Build price grid & vol-per-price histogram (uniform intra-bar spread)
        res = self.resolution
        edges = np.linspace(lo, hi, res + 1)
        step = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2.0
        vol_bins = np.zeros(res, dtype=float)

        # Sebar volume tiap candle ke bin yang overlap dengan [low, high]
        # (approx intrabar)
        for row in dfv.itertuples(index=False):
            c_lo = float(getattr(row, 'low'))
            c_hi = float(getattr(row, 'high'))
            c_vol = float(getattr(row, 'volume'))
            if not (np.isfinite(c_lo) and np.isfinite(c_hi) and np.isfinite(c_vol)):
                continue
            if c_hi <= c_lo or c_vol <= 0:
                continue
            start_idx = int(np.floor((c_lo - lo) / step))
            end_idx = int(np.ceil((c_hi - lo) / step)) - 1
            start_idx = max(0, min(res - 1, start_idx))
            end_idx = max(0, min(res - 1, end_idx))
            span = c_hi - c_lo
            for i in range(start_idx, end_idx + 1):
                b_lo = edges[i]
                b_hi = edges[i + 1]
                overlap = max(0.0, min(c_hi, b_hi) - max(c_lo, b_lo))
                if overlap > 0:
                    vol_bins[i] += c_vol * (overlap / span)

        total_vol = float(vol_bins.sum())
        if total_vol <= 0:
            return ZonesResult([], [], None, None, visible_start, visible_end)

        target = total_vol * (self.threshold_percent / 100.0)

        def expand_from_top() -> Optional[Zone]:
            cum = 0.0
            idx = res - 1
            while idx >= 0 and cum + vol_bins[idx] < target:
                cum += vol_bins[idx]
                idx -= 1
            if idx < 0:
                return None
            need = max(0.0, target - cum)
            frac = 0.0 if vol_bins[idx] <= 0 else need / vol_bins[idx]
            top_price = edges[-1]
            bottom_price = edges[idx + 1] - frac * step
            # Weighted avg: bins penuh + parsial
            w_prices: List[float] = []
            w_weights: List[float] = []
            for j in range(res - 1, idx, -1):
                w_prices.append(float(centers[j]))
                w_weights.append(float(vol_bins[j]))
            if need > 0:
                part_center = edges[idx + 1] - 0.5 * (frac * step)
                w_prices.append(float(part_center))
                w_weights.append(float(need))
            avg = (top_price + bottom_price) / 2.0
            wavg = self._weighted_avg(np.array(w_prices), np.array(w_weights))
            return Zone('supply', top=top_price, bottom=bottom_price, average=avg,
                        weighted_average=wavg, volume_ratio=self.threshold_percent,
                        start_ts=visible_start)

        def expand_from_bottom() -> Optional[Zone]:
            cum = 0.0
            idx = 0
            while idx < res and cum + vol_bins[idx] < target:
                cum += vol_bins[idx]
                idx += 1
            if idx >= res:
                return None
            need = max(0.0, target - cum)
            frac = 0.0 if vol_bins[idx] <= 0 else need / vol_bins[idx]
            bottom_price = edges[0]
            top_price = edges[idx] + frac * step
            w_prices: List[float] = []
            w_weights: List[float] = []
            for j in range(0, idx):
                w_prices.append(float(centers[j]))
                w_weights.append(float(vol_bins[j]))
            if need > 0:
                part_center = edges[idx] + 0.5 * (frac * step)
                w_prices.append(float(part_center))
                w_weights.append(float(need))
            avg = (top_price + bottom_price) / 2.0
            wavg = self._weighted_avg(np.array(w_prices), np.array(w_weights))
            return Zone('demand', top=top_price, bottom=bottom_price, average=avg,
                        weighted_average=wavg, volume_ratio=self.threshold_percent,
                        start_ts=visible_start)

        s = expand_from_top()
        d = expand_from_bottom()

        s_list = [s] if s else []
        d_list = [d] if d else []

        equi = None
        w_equi = None
        if s and d:
            equi = (s.average + d.average) / 2.0
            w_equi = (s.weighted_average + d.weighted_average) / 2.0

        return ZonesResult(s_list, d_list, equi, w_equi, visible_start, visible_end)

    def generate_sinyal(self, zones: ZonesResult, current_price: float, proximity_pct: float = 0.5) -> List[Dict[str, object]]:
        sinyal: List[Dict[str, object]] = []
        tol_buy = 1.0 + proximity_pct / 100.0
        tol_sell = 1.0 - proximity_pct / 100.0
        for z in zones.demand_zones:
            if current_price <= z.weighted_average * tol_buy:
                sinyal.append({'type': 'BUY', 'price': current_price, 'zone': z, 'ts': pd.Timestamp.utcnow()})
        for z in zones.supply_zones:
            if current_price >= z.weighted_average * tol_sell:
                sinyal.append({'type': 'SELL', 'price': current_price, 'zone': z, 'ts': pd.Timestamp.utcnow()})
        return sinyal


# =============================
# Streaming Binance (WebSocket + bootstrap)
# =============================

class BinanceKlineStreamer:
    def __init__(self, symbol: str = 'BTCUSDT', interval: str = '1m', limit: int = 500,
                 api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = int(limit)
        self.api_key = api_key
        self.api_secret = api_secret
        self._client: Optional[_AsyncClient] = None
        self._bm: Optional[_BinanceSocketManager] = None

    async def __aenter__(self):
        if AsyncClient is None:
            raise RuntimeError("python-binance belum terpasang. pip install python-binance")
        self._client = await AsyncClient.create(self.api_key, self.api_secret)
        client = self._client
        assert client is not None
        self._bm = BinanceSocketManager(client)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.close_connection()

    async def fetch_hist_df(self) -> pd.DataFrame:
        client = self._client
        assert client is not None, "Gunakan dalam 'async with'"
        kl = await client.get_klines(symbol=self.symbol, interval=self.interval, limit=self.limit)
        df = pd.DataFrame(kl, columns=['open_time','open','high','low','close','volume','close_time','quote','trades','taker_base','taker_quote','ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.set_index('open_time', inplace=True)
        return df[['open','high','low','close','volume']].astype(float)

    async def stream_klines(self):
        bm = self._bm
        assert bm is not None, "Gunakan dalam 'async with'"
        async with bm.kline_socket(self.symbol, interval=self.interval) as stream:
            while True:
                msg = await stream.recv()
                k = msg.get('k') or {}
                if not k:
                    continue
                if k.get('x') is True:  # hanya saat candle close
                    yield {
                        'open_time': pd.to_datetime(k['t'], unit='ms', utc=True),
                        'open': float(k['o']),
                        'high': float(k['h']),
                        'low': float(k['l']),
                        'close': float(k['c']),
                        'volume': float(k['v']),
                    }


# =============================
# Orkestrator real-time
# =============================

class SDWSRunner:
    def __init__(self, indicator: SupplyDemandVisibleRange, symbol: str = 'BTCUSDT', interval: str = '1m',
                 limit_bootstrap: int = 500, proximity_pct: float = 0.5,
                 api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        self.indicator = indicator
        self.streamer = BinanceKlineStreamer(symbol=symbol, interval=interval, limit=limit_bootstrap,
                                             api_key=api_key, api_secret=api_secret)
        self.df = pd.DataFrame(columns=['open','high','low','close','volume'])
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.proximity_pct = float(proximity_pct)

    def on_update(self, fn: Callable[[Dict[str, object]], None]) -> None:
        self.callbacks.append(fn)

    def _emit(self, payload: Dict[str, object]) -> None:
        for fn in self.callbacks:
            try:
                fn(payload)
            except Exception as e:  # pragma: no cover
                print(f"[SDWSRunner] callback error: {e}")

    async def run(self) -> None:
        async with self.streamer as s:
            self.df = await s.fetch_hist_df()
            zones = self.indicator.hitung_zona(self.df)
            last_price = float(self.df['close'].iloc[-1])
            signals = self.indicator.generate_sinyal(zones, last_price, self.proximity_pct)
            self._emit({'zones': zones, 'last_bar': self.df.iloc[-1].to_dict(), 'signals': signals})

            async for bar in s.stream_klines():
                new_row = pd.DataFrame([bar]).set_index('open_time')
                self.df = pd.concat([self.df, new_row]).iloc[-self.indicator.max_bars_back :]
                zones = self.indicator.hitung_zona(self.df)
                last_price = float(self.df['close'].iloc[-1])
                signals = self.indicator.generate_sinyal(zones, last_price, self.proximity_pct)
                self._emit({'zones': zones, 'last_bar': self.df.iloc[-1].to_dict(), 'signals': signals})


# =============================
# Demo CLI (opsional)
# =============================

async def _demo_cli():  # pragma: no cover
    import os
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    ind = SupplyDemandVisibleRange(threshold_percent=10, resolution=100, max_bars_back=500)
    runner = SDWSRunner(indicator=ind, symbol='BTCUSDT', interval='1m', proximity_pct=0.5,
                        api_key=api_key, api_secret=api_secret)

    def print_update(ev: Dict[str, Any]):
        zones = cast(ZonesResult, ev['zones'])
        last = cast(Dict[str, Any], ev['last_bar'])
        sigs = cast(List[Dict[str, Any]], ev.get('signals', []))
        ts = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] close={last['close']:.2f} signals={len(sigs)}")
        for s in sigs:
            z: Zone = s['zone']  # type: ignore
            print(f"  -> {s['type']} near {z.label.upper()} wavg={z.weighted_average:.2f}")

    runner.on_update(print_update)
    await runner.run()


if __name__ == '__main__':  # pragma: no cover
    try:
        asyncio.run(_demo_cli())
    except KeyboardInterrupt:
        pass
