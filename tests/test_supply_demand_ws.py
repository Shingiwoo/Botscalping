import os, sys
import numpy as np
import pandas as pd
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Prefer lux version; fallback ke versi lama
try:
    from indicators.supplyanddemand.supply_demand_ws_lux import SupplyDemandVisibleRange, Zone, ZonesResult
except Exception:
    from indicators.supplyanddemand.supply_demand_ws import SupplyDemandVisibleRange, Zone, ZonesResult

def _df(n=300):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    base = np.linspace(100, 110, n)
    df = pd.DataFrame({
        "open": base,
        "high": base + 0.8,
        "low": base - 0.8,
        "close": base + 0.1,
        "volume": np.linspace(10, 20, n),
    }, index=ts)
    return df

def test_zones_shape_and_types():
    ind = SupplyDemandVisibleRange(threshold_percent=10, resolution=50, max_bars_back=200)
    res: ZonesResult = ind.hitung_zona(_df())
    assert hasattr(res, "supply_zones") and hasattr(res, "demand_zones")
    # zona (bila ada) bertipe Zone dan memiliki angka valid
    for z in (res.supply_zones + res.demand_zones):
        assert isinstance(z, Zone)
        assert np.isfinite(z.average)
        assert np.isfinite(z.weighted_average)

def test_generate_signals_near_wavg():
    ind = SupplyDemandVisibleRange(threshold_percent=10, resolution=50, max_bars_back=200)
    df = _df()
    res = ind.hitung_zona(df)
    price = float(df["close"].iloc[-1])
    sigs: List[Dict[str, Any]] = ind.generate_sinyal(res, price, proximity_pct=1.0)
    # Harus berupa list of dict (iterable & sized)
    assert isinstance(sigs, list)
    if sigs:
        s0 = sigs[0]
        assert "type" in s0 and "price" in s0 and "zone" in s0
