from __future__ import annotations
import pandas as pd
from typing import List, Optional
from .types import ZonesState, SMCEvent, StructureBias

class ZonesEngine:
    """ Premium/Discount/Equilibrium berdasarkan trailing HH/LL terbaru. """
    def __init__(self):
        self.state = ZonesState()
        self._top: Optional[float] = None
        self._bot: Optional[float] = None

    def step(self, df: pd.DataFrame, i: int, symbol: str, timeframe: str, swing_bias: StructureBias) -> List[SMCEvent]:
        out: List[SMCEvent] = []
        h = float(df['high'].iloc[i]); l = float(df['low'].iloc[i]); t = df.index[i]

        # Update trailing extremes
        self._top = max(h, self._top) if self._top is not None else h
        self._bot = min(l, self._bot) if self._bot is not None else l

        # Tentukan label strong/weak (informasi ditaruh di payload)
        eq = (self._top + self._bot) / 2.0
        self.state.premium_top = self._top
        self.state.equilibrium = eq
        self.state.discount_bottom = self._bot
        self.state.ref_top_time = t if h == self._top else self.state.ref_top_time
        self.state.ref_bottom_time = t if l == self._bot else self.state.ref_bottom_time

        out.append(SMCEvent("zones_update", symbol, timeframe, i, t, {
            'premium_top': self._top,
            'equilibrium': eq,
            'discount_bottom': self._bot,
            'strong_high': (swing_bias == "bear"),
            'strong_low': (swing_bias == "bull")
        }))
        return out
