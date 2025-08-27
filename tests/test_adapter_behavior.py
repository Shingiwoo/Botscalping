import asyncio
from typing import Any, Dict, List
import pandas as pd
from datetime import timedelta
import pytest

from indicators.supplyanddemand.adapter_sdws import SDWSRajaDollarAdapter, BasePublisher
try:
    from indicators.supplyanddemand.supply_demand_ws_lux import Zone, ZonesResult
except Exception:
    from indicators.supplyanddemand.supply_demand_ws import Zone, ZonesResult


class _CollectPublisher(BasePublisher):
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def publish(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


class _DummyRunner:
    def __init__(self) -> None:
        self.callbacks: List[Any] = []
        self.streamer = type("S", (), {"symbol": "BTCUSDT", "interval": "1m"})()

    def on_update(self, cb) -> None:
        self.callbacks.append(cb)

    async def run(self) -> None:  # pragma: no cover
        pass


def _zones(ts: str = "2024-01-01T00:00:00Z") -> ZonesResult:
    t = pd.Timestamp(ts)
    z_sup = Zone("supply", top=111, bottom=109, average=110, weighted_average=110.2, volume_ratio=10, start_ts=t)
    z_dem = Zone("demand", top=101, bottom=99, average=100, weighted_average=100.1, volume_ratio=10, start_ts=t)
    return ZonesResult([z_sup], [z_dem], equilibrium=105.0, weighted_equilibrium=105.15, visible_start=t, visible_end=t)


def _last_bar(price: float = 105.5) -> Dict[str, Any]:
    return {"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 10}


@pytest.mark.asyncio
async def test_snapshot_once_and_signal_dedup_per_call():
    pub = _CollectPublisher()
    adapter = SDWSRajaDollarAdapter(_DummyRunner(), publishers=[pub], cooldown_candles=0, min_confidence=0.0)
    z = _zones()
    lb = _last_bar()
    sigs = [
        {"type": "BUY", "price": z.demand_zones[0].weighted_average, "zone": z.demand_zones[0]},
        {"type": "BUY", "price": z.demand_zones[0].weighted_average, "zone": z.demand_zones[0]},
    ]
    await adapter._handle_update({"zones": z, "last_bar": lb, "signals": sigs})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 1
    pub.events.clear()


@pytest.mark.asyncio
async def test_cooldown_and_min_confidence():
    pub = _CollectPublisher()
    adapter = SDWSRajaDollarAdapter(_DummyRunner(), publishers=[pub], cooldown_candles=2, min_confidence=0.5)
    z1 = _zones("2024-01-01T00:00:00Z")
    lb = _last_bar(100.10)
    buy_sig = {"type": "BUY", "price": z1.demand_zones[0].weighted_average, "zone": z1.demand_zones[0]}
    await adapter._handle_update({"zones": z1, "last_bar": lb, "signals": [buy_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 1
    pub.events.clear()

    z2 = _zones("2024-01-01T00:01:00Z")
    await adapter._handle_update({"zones": z2, "last_bar": lb, "signals": [buy_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.snapshot") == 1 and topics.count("indicator.sdws.signal") == 0
    pub.events.clear()

    z3 = _zones("2024-01-01T00:02:00Z")
    await adapter._handle_update({"zones": z3, "last_bar": lb, "signals": [buy_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 0
    pub.events.clear()

    z4 = _zones("2024-01-01T00:03:00Z")
    await adapter._handle_update({"zones": z4, "last_bar": lb, "signals": [buy_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 1
    pub.events.clear()

    z5 = _zones("2024-01-01T00:04:00Z")
    far_price = z5.demand_zones[0].weighted_average * 1.05
    weak_sig = {"type": "BUY", "price": far_price, "zone": z5.demand_zones[0]}
    await adapter._handle_update({"zones": z5, "last_bar": lb, "signals": [weak_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 0
    pub.events.clear()

    z6 = _zones("2024-01-01T00:05:00Z")
    strong_sig = {"type": "BUY", "price": z6.demand_zones[0].weighted_average, "zone": z6.demand_zones[0]}
    await adapter._handle_update({"zones": z6, "last_bar": lb, "signals": [strong_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 0
    pub.events.clear()

    z7 = _zones("2024-01-01T00:06:00Z")
    await adapter._handle_update({"zones": z7, "last_bar": lb, "signals": [strong_sig]})
    topics = [e["topic"] for e in pub.events]
    assert topics.count("indicator.sdws.signal") == 1
