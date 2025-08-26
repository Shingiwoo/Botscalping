from __future__ import annotations

"""
Adapter SDWS → RajaDollar (Queue/Event)

Tujuan:
- Menjembatani modul indikator Supply & Demand WebSocket (SDWSRunner)
  ke arsitektur RajaDollar berbasis event/queue.
- Mengirim event snapshot zona & event sinyal ke publisher (mis. asyncio.Queue
  pusat, Telegram, logger, dsb).

Pemakaian umum:

    from supply_demand_ws import SupplyDemandVisibleRange, SDWSRunner
    from sdws_raja_adapter import SDWSRajaDollarAdapter, AsyncQueuePublisher
    import asyncio

    event_queue = asyncio.Queue()

    ind = SupplyDemandVisibleRange(threshold_percent=10, resolution=50, max_bars_back=500)
    runner = SDWSRunner(indicator=ind, symbol="BTCUSDT", interval="1m", proximity_pct=0.5,
                        api_key=API_KEY, api_secret=API_SECRET)

    adapter = SDWSRajaDollarAdapter(runner, publishers=[AsyncQueuePublisher(event_queue)],
                                    source="SDWS", topic_signal="indicator.sdws.signal",
                                    topic_snapshot="indicator.sdws.snapshot")

    # MODE A (disarankan untuk Streamlit): jalan di thread terpisah
    adapter.start_in_background()

    # Di tempat lain, konsumer:
    async def consumer():
        while True:
            event = await event_queue.get()
            # TODO: mapping ke eksekusi order, notifikasi, dll.
            # print("EVENT:", event)

    asyncio.get_event_loop().create_task(consumer())

Catatan:
- Tidak melakukan sizing/eksekusi order di sini. Hanya publikasi event.
- Dedup sinyal dasar disediakan (menghindari spam dalam 1 candle per tipe aksi).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

try:
    from supply_demand_ws import SDWSRunner, ZonesResult, Zone
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pastikan 'supply_demand_ws.py' ada di PYTHONPATH.")


# =============================
# Publisher Interfaces
# =============================

class BasePublisher:
    async def publish(self, event: Dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError


class AsyncQueuePublisher(BasePublisher):
    """Publisher ke asyncio.Queue pusat (event bus internal RajaDollar)."""

    def __init__(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        self.queue = queue

    async def publish(self, event: Dict[str, Any]) -> None:
        await self.queue.put(event)


class LoggingPublisher(BasePublisher):
    """Publisher ke logging (untuk debug)."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("SDWSAdapter")

    async def publish(self, event: Dict[str, Any]) -> None:
        self.logger.info("%s", event)


class TelegramPublisher(BasePublisher):
    """Publisher ke Telegram (stub).

    Beri fungsi async `send(text: str)` sesuai implementasi Telegram kamu
    (mis. python-telegram-bot async).
    """

    def __init__(self, send_func) -> None:
        self.send_func = send_func

    async def publish(self, event: Dict[str, Any]) -> None:
        text = event.get("message") or event.get("summary") or str(event)
        await self.send_func(text)


# =============================
# Adapter
# =============================

@dataclass
class AdapterConfig:
    source: str = "SDWS"
    topic_signal: str = "indicator.sdws.signal"
    topic_snapshot: str = "indicator.sdws.snapshot"
    dedup_per_candle: bool = True


class SDWSRajaDollarAdapter:
    """Adapter antara SDWSRunner dan event bus RajaDollar.

    - Mendaftarkan callback pada SDWSRunner
    - Memformat payload menjadi event siap konsumsi
    - Mempublikasikan ke satu/lebih publisher
    - Opsi jalan sebagai service async atau di thread terpisah
    """

    def __init__(
        self,
        runner: SDWSRunner,
        publishers: Sequence[BasePublisher],
        source: str = "SDWS",
        topic_signal: str = "indicator.sdws.signal",
        topic_snapshot: str = "indicator.sdws.snapshot",
    ) -> None:
        self.runner = runner
        self.publishers = list(publishers)
        self.cfg = AdapterConfig(source=source, topic_signal=topic_signal, topic_snapshot=topic_snapshot)
        self._last_candle_key: Optional[str] = None
        self._started = False
        self._stop_event = asyncio.Event()
        self._bg_thread = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --------- Mode inline (satu event loop) ---------
    async def serve(self) -> None:
        """Jalankan adapter & runner pada event loop sekarang."""
        if self._started:
            return
        self._started = True

        def _on_update(ev: Dict[str, Any]) -> None:
            # Callback sync dari runner; delegasikan ke task async
            asyncio.create_task(self._handle_update(ev))

        self.runner.on_update(_on_update)
        await self.runner.run()

    # --------- Mode background thread (untuk Streamlit/UI) ---------
    def start_in_background(self) -> None:
        """Jalankan adapter+runner pada event loop baru di thread lain."""
        if self._bg_thread is not None:
            return

        import threading

        def _bg():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            def _on_update(ev: Dict[str, Any]) -> None:
                # Jadwalkan ke loop thread background
                asyncio.run_coroutine_threadsafe(self._handle_update(ev), self._loop)

            self.runner.on_update(_on_update)
            try:
                self._loop.run_until_complete(self.runner.run())
            finally:
                try:
                    pending = asyncio.all_tasks(loop=self._loop)
                    for t in pending:
                        t.cancel()
                except Exception:
                    pass
                self._loop.stop()
                self._loop.close()

        self._bg_thread = threading.Thread(target=_bg, daemon=True)
        self._bg_thread.start()

    async def stop(self) -> None:
        # Runner memiliki kontrol loop internal; untuk penyederhanaan,
        # penghentian dilakukan dengan KeyboardInterrupt dari pemanggil.
        self._stop_event.set()

    # --------- Format & Publish ---------
    async def _handle_update(self, ev: Dict[str, Any]) -> None:
        zones: ZonesResult = ev["zones"]  # type: ignore
        last_bar: Dict[str, float] = ev["last_bar"]  # type: ignore
        signals: List[Dict[str, Any]] = ev.get("signals", [])  # type: ignore

        # Kunci candle untuk dedup per-candle (gunakan visible_end)
        candle_key = str(zones.visible_end.value)  # nanosecond timestamp
        is_new_candle = candle_key != self._last_candle_key
        if is_new_candle:
            self._last_candle_key = candle_key

        # 1) Publish snapshot zona sekali per candle
        if is_new_candle:
            snapshot = self._format_snapshot_event(zones, last_bar)
            await self._publish(snapshot)

        # 2) Publish sinyal (BUY/SELL) – dedup per candle per aksi
        seen_actions: set[str] = set()
        for s in signals:
            action = s.get("type", "").upper()
            if action and action not in seen_actions:
                seen_actions.add(action)
                evt = self._format_signal_event(s, zones, last_bar)
                await self._publish(evt)

    def _format_snapshot_event(self, zones: ZonesResult, last_bar: Dict[str, float]) -> Dict[str, Any]:
        supply = [self._zone_to_dict(z) for z in zones.supply_zones]
        demand = [self._zone_to_dict(z) for z in zones.demand_zones]
        return {
            "topic": self.cfg.topic_snapshot,
            "source": self.cfg.source,
            "ts": pd.Timestamp.utcnow().isoformat(),
            "symbol": getattr(self.runner.streamer, "symbol", None),
            "interval": getattr(self.runner.streamer, "interval", None),
            "last_close": float(last_bar.get("close", float("nan"))),
            "equilibrium": zones.equilibrium,
            "weighted_equilibrium": zones.weighted_equilibrium,
            "supply_zones": supply,
            "demand_zones": demand,
            "visible_start": zones.visible_start.isoformat(),
            "visible_end": zones.visible_end.isoformat(),
            "summary": self._summary_text(last_bar, zones, n_sig=None),
        }

    def _format_signal_event(self, s: Dict[str, Any], zones: ZonesResult, last_bar: Dict[str, float]) -> Dict[str, Any]:
        z: Zone = s.get("zone")  # type: ignore
        data = {
            "topic": self.cfg.topic_signal,
            "source": self.cfg.source,
            "ts": pd.Timestamp.utcnow().isoformat(),
            "symbol": getattr(self.runner.streamer, "symbol", None),
            "interval": getattr(self.runner.streamer, "interval", None),
            "action": s.get("type"),  # BUY/SELL
            "price": float(s.get("price", float("nan"))),
            "zone": self._zone_to_dict(z) if isinstance(z, Zone) else None,
            "equilibrium": zones.equilibrium,
            "weighted_equilibrium": zones.weighted_equilibrium,
            "last_close": float(last_bar.get("close", float("nan"))),
            # Ruang untuk confidence score/filters tambahan jika diperlukan
            "confidence": self._default_confidence(s, zones),
            "message": self._summary_text(last_bar, zones, n_sig=1, action=s.get("type")),
        }
        return data

    @staticmethod
    def _zone_to_dict(z: Zone) -> Dict[str, Any]:
        return {
            "label": z.label,
            "top": z.top,
            "bottom": z.bottom,
            "average": z.average,
            "weighted_average": z.weighted_average,
            "volume_ratio": z.volume_ratio,
            "start_ts": z.start_ts.isoformat(),
        }

    @staticmethod
    def _default_confidence(s: Dict[str, Any], zones: ZonesResult) -> float:
        """Skor sederhana berbasis kedekatan harga ke wavg zona (0..1)."""
        try:
            price = float(s.get("price"))
            z: Zone = s.get("zone")  # type: ignore
            w = float(z.weighted_average) if isinstance(z, Zone) else float("nan")
            if not (pd.notna(price) and pd.notna(w)):
                return 0.0
            # Jarak relatif (lebih dekat → skor lebih tinggi)
            rel = abs(price - w) / max(1e-8, w)
            score = max(0.0, 1.0 - min(rel / 0.01, 1.0))  # 1% jarak → score 0
            return round(float(score), 3)
        except Exception:
            return 0.0

    @staticmethod
    def _summary_text(last_bar: Dict[str, float], zones: ZonesResult, n_sig: Optional[int], action: Optional[str] = None) -> str:
        p = last_bar.get("close")
        act = f" {action}" if action else ""
        parts = [
            f"SDWS{act}: close={p:.4f}" if isinstance(p, (int, float)) else f"SDWS{act}",
        ]
        if zones.weighted_equilibrium is not None:
            parts.append(f"WEQ={zones.weighted_equilibrium:.4f}")
        if zones.equilibrium is not None:
            parts.append(f"EQ={zones.equilibrium:.4f}")
        if n_sig is not None:
            parts.append(f"signals={n_sig}")
        return " | ".join(parts)

    async def _publish(self, event: Dict[str, Any]) -> None:
        for pub in self.publishers:
            try:
                await pub.publish(event)
            except Exception as e:  # pragma: no cover
                logging.getLogger("SDWSAdapter").error("publisher error: %s", e)

