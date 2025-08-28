from __future__ import annotations
import importlib
from typing import Any, Optional


def _try(paths: list[tuple[str, str]]) -> Optional[Any]:
    for mod_path, attr in paths:
        try:
            mod = importlib.import_module(mod_path)
            if not attr:
                return mod
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception:
            continue
    return None

SC_API = _try([
    ("indicators.scsignal", "compute_all"),
    ("indicators.scsignal.core", "compute_all"),
])

SMC_STRUCTURE = _try([
    ("indicators.smartmoney.smc.structure", "detect_structure"),
    ("indicators.smartmoney.smc.structure", "StructureDetector"),
])

SMC_ORDERBLOCK = _try([
    ("indicators.smartmoney.smc.orderblock", "detect_orderblocks"),
    ("indicators.smartmoney.smc.orderblock", "OrderBlockDetector"),
])

SMC_FVG = _try([
    ("indicators.smartmoney.smc.fvg", "detect_fvg"),
    ("indicators.smartmoney.smc.fvg", "FVGDetector"),
])

SMC_LEVELS = _try([
    ("indicators.smartmoney.smc.levels", "prev_period_levels"),
    ("indicators.smartmoney.smc.levels", "LevelsDetector"),
])

SMC_ZONES = _try([
    ("indicators.smartmoney.smc.zones", "compute_premium_discount"),
    ("indicators.smartmoney.smc.zones", "ZonesDetector"),
])

# ---------- Supply & Demand (visible range) ----------
# Tangani beberapa kemungkinan path serta fallback wrapper sederhana
SD_API = _try([
    ("indicators.supplyanddemand", "compute_visible_range"),
    ("indicators.supplyanddemand.core", "compute_visible_range"),
])

if SD_API is None:
    def _sd_wrapper(df):  # type: ignore
        try:
            from indicators.supplyanddemand.supply_demand_ws import SupplyDemandVisibleRange
            sd = SupplyDemandVisibleRange()
            res = sd.hitung_zona(df)
            out = {}
            if getattr(res, "demand_zones", None):
                out["demand_wavg"] = float(res.demand_zones[0].weighted_average)
            if getattr(res, "supply_zones", None):
                out["supply_wavg"] = float(res.supply_zones[0].weighted_average)
            return out
        except Exception:
            return {}

    SD_API = _sd_wrapper

SR_NEAR = _try([
    ("indicators.sr_utils", "near_level"),
])
SR_BUILD_CACHE = _try([
    ("indicators.sr_utils", "build_sr_cache"),
])
