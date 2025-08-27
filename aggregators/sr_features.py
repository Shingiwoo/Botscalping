from __future__ import annotations
from typing import Dict, List, Tuple
import math
import pandas as pd
from indicators.srmtf.support_resistance_mtf import Zone, Signal

def _atr_last(chart_df: pd.DataFrame, length: int = 17) -> float:
    high, low, close = chart_df["high"], chart_df["low"], chart_df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(length, min_periods=1).mean().iloc[-1])

def _nearest_zone_distance(price: float, zones: List[Zone]) -> Tuple[str, float]:
    """
    Kembalikan (kind, distance_abs) terhadap tepi zona terdekat (bukan center).
    kind: 'R'|'S' atau '' bila tak ada zona.
    """
    best = ("", math.inf)
    for z in zones:
        # distance ke tepi terdekat
        if price > z.top:
            d = price - z.top
        elif price < z.bottom:
            d = z.bottom - price
        else:
            d = 0.0
        if d < best[1]:
            best = (z.kind, d)
    return best

def sr_features_from_signals(chart_df: pd.DataFrame,
                             signals_by_group: Dict[str, List[Signal]],
                             zones_by_group: Dict[str, List[Zone]],
                             cfg: dict) -> Dict[str, float]:
    """
    Ubah sinyal S/R ke fitur numerik (0..1) dan boolean (0/1) untuk aggregator.
    Aturan mengikuti alur: konfirmasi di close, skor di-clamp [0,1], penalti jarak/ATR.
    """
    features: Dict[str, float] = {}
    if not signals_by_group:
        return features

    last_close = float(chart_df["close"].iloc[-1])
    atr = _atr_last(chart_df, length=int(cfg.get("atr_len", 17)))
    near_mult = float(cfg.get("near_zone_atr_mult", 0.8))  # threshold “near” (× ATR)

    # Features “near zone” berdasar semua zona gabungan
    all_zones: List[Zone] = []
    for zs in zones_by_group.values():
        all_zones.extend(zs)
    kind_near, dist = _nearest_zone_distance(last_close, all_zones) if all_zones else ("", math.inf)
    features["sr_near_support"] = 1.0 if (kind_near == "S" and dist <= near_mult * atr) else 0.0
    features["sr_near_resistance"] = 1.0 if (kind_near == "R" and dist <= near_mult * atr) else 0.0

    # Sinyal khusus (breakout/test/retest/rejection) → 1.0 pada bar ini
    kinds = {"BULL_BREAKOUT": "sr_breakout_bull",
             "BEAR_BREAKOUT": "sr_breakout_bear",
             "TEST_R": "sr_test_r", "TEST_S": "sr_test_s",
             "RETEST_R": "sr_retest_r", "RETEST_S": "sr_retest_s",
             "REJECT_UP": "sr_reject_up", "REJECT_DOWN": "sr_reject_down"}

    for g, sigs in signals_by_group.items():
        for s in sigs:
            name = kinds.get(s.kind)
            if not name:
                continue
            # optional distance penalty (lebih jauh dari zona → lebih lemah)
            # d=0 pada breakout sebenarnya, TEST/RETEST/REJECT diberi sedikit penalti
            if s.ref_zone.kind == "R":
                dist_edge = max(0.0, last_close - s.ref_zone.bottom) if s.kind in ("TEST_R","RETEST_R","REJECT_UP") else 0.0
            else:
                dist_edge = max(0.0, s.ref_zone.top - last_close) if s.kind in ("TEST_S","RETEST_S","REJECT_DOWN") else 0.0
            penalty = 1.0 - min(1.0, dist_edge / (atr if atr > 0 else 1.0))
            val = max(0.0, min(1.0, penalty))
            features[name] = max(features.get(name, 0.0), val)

    return features

def sr_reason_weights_default() -> Dict[str, float]:
    """Default bobot agar aggregator bisa langsung dipakai kalau belum ada di config."""
    return {
        "sr_breakout_bull": 0.35,
        "sr_breakout_bear": 0.35,
        "sr_test_r": 0.15,
        "sr_test_s": 0.15,
        "sr_retest_r": 0.20,
        "sr_retest_s": 0.20,
        "sr_reject_up": 0.25,
        "sr_reject_down": 0.25,
        "sr_near_support": 0.10,
        "sr_near_resistance": 0.10,
    }
