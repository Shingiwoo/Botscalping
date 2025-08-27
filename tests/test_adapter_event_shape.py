import os, sys
import pandas as pd
import numpy as np
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from indicators.supplyanddemand.supply_demand_ws_lux import Zone, ZonesResult
except Exception:
    from indicators.supplyanddemand.supply_demand_ws import Zone, ZonesResult

def _dummy_zones() -> ZonesResult:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    z_sup = Zone("supply", top=111, bottom=109, average=110, weighted_average=110.2, volume_ratio=10, start_ts=ts)
    z_dem = Zone("demand", top=101, bottom= 99, average=100, weighted_average=100.1, volume_ratio=10, start_ts=ts)
    return ZonesResult([z_sup], [z_dem], equilibrium=105.0, weighted_equilibrium=105.15,
                       visible_start=ts, visible_end=ts)

def test_event_payload_iterable_and_sized():
    zones = _dummy_zones()
    last_bar: Dict[str, Any] = {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.5, "volume": 10}
    signals: List[Dict[str, Any]] = []
    ev: Dict[str, Any] = {"zones": zones, "last_bar": last_bar, "signals": signals}
    # memastikan downstream dapat memakai len() dan iterasi
    assert len(ev["signals"]) == 0
    for _ in ev["signals"]:
        pass
