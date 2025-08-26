from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any, Tuple
import pandas as pd

EventType = Literal[
    "internal_bos","internal_choch","swing_bos","swing_choch",
    "internal_ob_new","internal_ob_mitigated","swing_ob_new","swing_ob_mitigated",
    "eqh","eql","fvg_bull","fvg_bear","mtf_levels","zones_update"
]

StructureBias = Literal["bull","bear","neutral"]

@dataclass
class OrderBlock:
    kind: Literal["internal","swing"]
    bias: StructureBias  # "bull" -> demand OB, "bear" -> supply OB
    top: float
    bottom: float
    start_idx: int
    start_time: pd.Timestamp
    mitigation: Literal["close","highlow"] = "highlow"

@dataclass
class FairValueGap:
    bias: StructureBias  # bull: low > high[2]; bear: high < low[2]
    top: float
    bottom: float
    left_time: pd.Timestamp
    right_time: pd.Timestamp

@dataclass
class LevelLines:
    daily_high: Optional[float] = None
    daily_low: Optional[float] = None
    weekly_high: Optional[float] = None
    weekly_low: Optional[float] = None
    monthly_high: Optional[float] = None
    monthly_low: Optional[float] = None

@dataclass
class ZonesState:
    premium_top: Optional[float] = None
    equilibrium: Optional[float] = None
    discount_bottom: Optional[float] = None
    ref_top_time: Optional[pd.Timestamp] = None
    ref_bottom_time: Optional[pd.Timestamp] = None

@dataclass
class SMCEvent:
    type: EventType
    symbol: str
    timeframe: str
    index: int
    time: pd.Timestamp
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SMCConfig:
    # Panjang swing dan internal (mirip 'swingsLengthInput' & 5 pada PineScript)
    swing_len: int = 50
    internal_len: int = 5

    # Threshold EQH/EQL (dalam ATR multipler)
    eq_threshold_atr_mult: float = 0.10  # ~0.1*ATR mirip 0..0.5 di Pine, tapi dalam ATR

    # FVG
    enable_fvg: bool = True
    fvg_timeframe: Optional[str] = None  # None => gunakan TF chart saat ini
    fvg_auto_threshold: bool = True

    # Order Blocks
    show_internal_ob: bool = True
    show_swing_ob: bool = True
    ob_mitigation: Literal["close","highlow"] = "highlow"
    ob_filter_method: Literal["atr","cmr"] = "atr"  # 'atr' atau 'cumulative mean range'
    atr_period: int = 200
    ob_max_keep: int = 100

    # MTF Levels
    show_daily: bool = False
    show_weekly: bool = False
    show_monthly: bool = False

    # Premium/Discount Zones
    show_zones: bool = True

    # Mode Present/Historical (hapus objek lama bila present=True)
    present_mode: bool = False
