from __future__ import annotations
from typing import TypedDict, Literal, Dict, Any, List, Optional

Side = Literal["LONG", "SHORT"]

class ScoreBreakdown(TypedDict, total=False):
    sc_trend_htf: float
    sc_no_htf: float
    adx: float
    body_atr: float
    width_atr: float
    rsi: float
    sr_breakout: float
    sr_test: float
    sr_reject: float
    sd_proximity: float
    vol_confirm: float
    fvg_confirm: float
    penalty_near_opposite_sr: float
    htf_fallback_discount: float

class AggResult(TypedDict):
    ok: bool
    side: Optional[Side]
    score: float
    strength: Literal["netral", "lemah", "cukup", "kuat"]
    reasons: List[str]
    breakdown: ScoreBreakdown
    context: Dict[str, Any]
