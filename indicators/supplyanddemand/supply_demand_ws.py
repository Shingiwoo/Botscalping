from __future__ import annotations

"""
Modul: supply_demand_ws

Indicator Supply & Demand versi Python dengan integrasi Binance WebSocket
(kompatibel python-binance >= 1.0.20). Cocok dipakai sebagai modul di aplikasi
trading (Streamlit/CLI/Service) dengan API sederhana.

Fitur utama:
- Perhitungan zona Supply & Demand berbasis volume pada visible range
- Equilibrium & Weighted Equilibrium
- Streaming data real-time via Binance WebSocket (klines)
- API modular: hitung_zona(df), generate_sinyal(current_price), dan kelas
  runner async untuk streaming & callback

Catatan kompatibilitas:
- Gunakan AsyncClient.create(API_KEY, API_SECRET), lalu pass client ke
  BinanceSocketManager(client). Jangan memberi api_key ke BSM secara langsung.

Lisensi: MIT
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # Dependensi runtime opsional (hanya diperlukan untuk streaming & contoh)
    from binance import AsyncClient, BinanceSocketManager
except Exception:  # pragma: no cover
    AsyncClient = None  # type: ignore
    BinanceSocketManager = None  # type: ignore


# =============================
# Data Structures & Typing
# =============================

@dataclass
class Zone:
    label: str  # 'supply' atau 'demand'
    top: float
    bottom: float
    average: float
    weighted_average: float
    volume_ratio: float
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
# Inti Indicator Supply & Demand
# =============================

class SupplyDemandVisibleRange:
    """Indicator Supply & Demand ala Pine (Visible Range) versi Python.

    Parameter
    ---------
    threshold_percent : float
        Ambang persentase (0-100) terhadap total volume visible range untuk
        melabel zona sebagai Supply/Demand pertama yang valid.
    resolution : int
        Jumlah interval harga (2-500). Makin besar = makin detail.
    max_bars_back : int
        Batas jumlah bar lookback untuk visible range.
    """

    def __init__(
        self,
        threshold_percent: float = 10.0,
        resolution: int = 50,
        max_bars_back: int = 500,
    ) -> None:
        if resolution < 2:
            raise ValueError("resolution minimal 2")
        if not (0 <= threshold_percent <= 100):
            raise ValueError("threshold_percent harus 0..100")
        self.threshold_percent = float(threshold_percent)
        self.resolution = int(resolution)
        self.max_bars_back = int(max_bars_back)

    # ---- Utilitas internal ----
    @staticmethod
    def _weighted_avg(prices: np.ndarray, weights: np.ndarray) -> float:
        w = np.sum(weights)
        if w <= 0:
            return float(np.mean(prices)) if prices.size else np.nan
        return float(np.sum(prices * weights) / w)

    # ---- API utama ----
    def hitung_zona(self, df: pd.DataFrame) -> ZonesResult:
        """Hitung zona Supply/Demand dari DataFrame OHLCV.

        `df` wajib memiliki kolom: ['open','high','low','close','volume']
        dan index bertipe waktu (DatetimeIndex). Visible range = window terakhir
        hingga `max_bars_back` bar.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame kosong")
        missing = {c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in df.columns}
        if missing:
            raise ValueError(f"Kolom hilang: {missing}")

        # Batasi visible range
        dfv = df.iloc[-self.max_bars_back :].copy()
        visible_start, visible_end = dfv.index[0], dfv.index[-1]

        high = float(dfv['high'].max())
        low = float(dfv['low'].min())
        total_vol = float(dfv['volume'].sum())

        # Hindari pembagian nol
        if not np.isfinite(high) or not np.isfinite(low) or high <= low or total_vol <= 0:
            return ZonesResult([], [], None, None, visible_start, visible_end)

        price_range = high - low
        step = price_range / float(self.resolution)

        supply_found = None  # type: Optional[Zone]
        demand_found = None  # type: Optional[Zone]

        # Loop interval
        for i in range(self.resolution):
            # ---- Supply bin: dari atas turun ----
            sup_upper = high - i * step
            sup_lower = sup_upper - step
            # Bar yang high-nya jatuh di interval (sup_lower, sup_upper]
            sup_mask = (dfv['high'] > sup_lower) & (dfv['high'] <= sup_upper)
            sup_vol = float(dfv.loc[sup_mask, 'volume'].sum())
            sup_ratio = (sup_vol / total_vol * 100.0) if total_vol > 0 else 0.0

            if supply_found is None and sup_ratio > self.threshold_percent:
                sup_avg = (high + sup_upper) / 2.0
                # wavg harga berbasis HIGH terobservasi di interval
                sup_wavg = self._weighted_avg(
                    dfv.loc[sup_mask, 'high'].to_numpy(dtype=float),
                    dfv.loc[sup_mask, 'volume'].to_numpy(dtype=float),
                )
                supply_found = Zone(
                    label='supply',
                    top=high,
                    bottom=sup_upper,
                    average=sup_avg,
                    weighted_average=sup_wavg,
                    volume_ratio=sup_ratio,
                    start_ts=visible_start,
                )

            # ---- Demand bin: dari bawah naik ----
            dem_lower = low + i * step
            dem_upper = dem_lower + step
            # Bar yang low-nya jatuh di interval [dem_lower, dem_upper)
            dem_mask = (dfv['low'] < dem_upper) & (dfv['low'] >= dem_lower)
            dem_vol = float(dfv.loc[dem_mask, 'volume'].sum())
            dem_ratio = (dem_vol / total_vol * 100.0) if total_vol > 0 else 0.0

            if demand_found is None and dem_ratio > self.threshold_percent:
                dem_avg = (low + dem_lower) / 2.0
                dem_wavg = self._weighted_avg(
                    dfv.loc[dem_mask, 'low'].to_numpy(dtype=float),
                    dfv.loc[dem_mask, 'volume'].to_numpy(dtype=float),
                )
                demand_found = Zone(
                    label='demand',
                    top=dem_upper,
                    bottom=low,
                    average=dem_avg,
                    weighted_average=dem_wavg,
                    volume_ratio=dem_ratio,
                    start_ts=visible_start,
                )

            if supply_found and demand_found:
                break

        supply_list = [supply_found] if supply_found else []
        demand_list = [demand_found] if demand_found else []

        # Equilibrium
        equi = (high + low) / 2.0 if (supply_found or demand_found) else None
        w_equi = None
        if supply_found and demand_found:
            w_equi = (supply_found.weighted_average + demand_found.weighted_average) / 2.0

        return ZonesResult(
            supply_zones=supply_list,
            demand_zones=demand_list,
            equilibrium=equi,
            weighted_equilibrium=w_equi,
            visible_start=visible_start,
            visible_end=visible_end,
        )

    def generate_sinyal(
        self,
        zones: ZonesResult,
        current_price: float,
        proximity_pct: float = 0.5,
    ) -> List[Dict[str, object]]:
        """Buat sinyal sederhana berdasarkan jarak ke zona.

        - BUY jika harga <= demand.wavg * (1 + proximity_pct/100)
        - SELL jika harga >= supply.wavg * (1 - proximity_pct/100)
        """
        sinyal: List[Dict[str, object]] = []
        tol_buy = 1.0 + proximity_pct / 100.0
        tol_sell = 1.0 - proximity_pct / 100.0

        for z in zones.demand_zones:
            if current_price <= z.weighted_average * tol_buy:
                sinyal.append({
                    'type': 'BUY',
                    'price': current_price,
                    'zone': z,
                    'ts': pd.Timestamp.utcnow(),
                })

        for z in zones.supply_zones:
            if current_price >= z.weighted_average * tol_sell:
                sinyal.append({
                    'type': 'SELL',
                    'price': current_price,
                    'zone': z,
                    'ts': pd.Timestamp.utcnow(),
                })

        return sinyal


# =======================================
# Streaming Binance (WebSocket + bootstrap)
# =======================================

class BinanceKlineStreamer:
    """Helper untuk ambil data historis & stream kline via WebSocket.

    Contoh interval: '1m','3m','5m','15m','1h', dst.
    """

    def __init__(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '1m',
        limit: int = 500,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = int(limit)
        self.api_key = api_key
        self.api_secret = api_secret
        self._client: Optional[AsyncClient] = None
        self._bm: Optional[BinanceSocketManager] = None

    async def __aenter__(self):
        if AsyncClient is None:
            raise RuntimeError("python-binance belum terpasang. pip install python-binance")
        self._client = await AsyncClient.create(self.api_key, self.api_secret)
        self._bm = BinanceSocketManager(self._client)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.close_connection()

    async def fetch_hist_df(self) -> pd.DataFrame:
        assert self._client is not None, "Panggil dalam context async with"
        kl = await self._client.get_klines(
            symbol=self.symbol, interval=self.interval, limit=self.limit
        )
        df = pd.DataFrame(
            kl,
            columns=[
                'open_time','open','high','low','close','volume',
                'close_time','quote','trades','taker_base','taker_quote','ignore'
            ],
        )
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.set_index('open_time', inplace=True)
        df = df[['open','high','low','close','volume']].astype(float)
        return df

    async def stream_klines(self):
        """Async generator: mengirimkan kline *closed* (msg['k']['x'] == True)."""
        assert self._bm is not None, "Panggil dalam context async with"
        async with self._bm.kline_socket(self.symbol, interval=self.interval) as stream:
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


# =======================================
# Orkestrator real-time: indikator + stream
# =======================================

class SDWSRunner:
    """Orkestrator: memelihara window OHLCV, hitung zona & kirim callback.

    Callback akan dipanggil setiap ada candle close baru dengan payload:
    {
        'zones': ZonesResult,
        'last_bar': Dict[str, float],
        'signals': List[...]
    }
    """

    def __init__(
        self,
        indicator: SupplyDemandVisibleRange,
        symbol: str = 'BTCUSDT',
        interval: str = '1m',
        limit_bootstrap: int = 500,
        proximity_pct: float = 0.5,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self.indicator = indicator
        self.streamer = BinanceKlineStreamer(
            symbol=symbol,
            interval=interval,
            limit=limit_bootstrap,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.df = pd.DataFrame(columns=['open','high','low','close','volume'])
        self.callbacks: List[Callable[[Dict[str, object]], None]] = []
        self.proximity_pct = float(proximity_pct)

    def on_update(self, fn: Callable[[Dict[str, object]], None]) -> None:
        """Daftarkan callback (dipanggil setiap update)."""
        self.callbacks.append(fn)

    def _emit(self, payload: Dict[str, object]) -> None:
        for fn in self.callbacks:
            try:
                fn(payload)
            except Exception as e:  # pragma: no cover
                print(f"[SDWSRunner] callback error: {e}")

    async def run(self) -> None:
        """Jalankan: bootstrap data historis, lalu stream websocket."""
        async with self.streamer as s:
            # Bootstrap
            self.df = await s.fetch_hist_df()
            zones = self.indicator.hitung_zona(self.df)
            last_price = float(self.df['close'].iloc[-1])
            signals = self.indicator.generate_sinyal(zones, last_price, self.proximity_pct)
            self._emit({'zones': zones, 'last_bar': self.df.iloc[-1].to_dict(), 'signals': signals})

            # Stream
            async for bar in s.stream_klines():
                # append & keep window length
                new_row = pd.DataFrame([bar]).set_index('open_time')
                self.df = pd.concat([self.df, new_row]).iloc[-self.indicator.max_bars_back :]

                zones = self.indicator.hitung_zona(self.df)
                last_price = float(self.df['close'].iloc[-1])
                signals = self.indicator.generate_sinyal(zones, last_price, self.proximity_pct)
                self._emit({'zones': zones, 'last_bar': self.df.iloc[-1].to_dict(), 'signals': signals})


# =============================
# Contoh pemakaian mandiri (CLI)
# =============================

async def _demo_cli():  # pragma: no cover
    import os

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    ind = SupplyDemandVisibleRange(threshold_percent=10, resolution=50, max_bars_back=500)
    runner = SDWSRunner(
        indicator=ind,
        symbol='BTCUSDT',
        interval='1m',
        proximity_pct=0.5,
        api_key=api_key,
        api_secret=api_secret,
    )

    def print_update(ev: Dict[str, object]):
        zones: ZonesResult = ev['zones']  # type: ignore
        last = ev['last_bar']  # type: ignore
        sigs = ev['signals']  # type: ignore
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

