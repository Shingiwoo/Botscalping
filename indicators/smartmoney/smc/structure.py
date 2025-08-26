from __future__ import annotations
import pandas as pd
from typing import List, Tuple
from .types import SMCEvent, EventType, StructureBias, SMCConfig
from .utils import atr, rolling_pivot_high, rolling_pivot_low

class StructureState:
    def __init__(self, swing_len: int, internal_len: int):
        self.swing_len = swing_len
        self.internal_len = internal_len
        # pivot levels (price) & cross flags
        self.swing_high_level = None
        self.swing_low_level = None
        self.internal_high_level = None
        self.internal_low_level = None
        # trend bias
        self.swing_bias: StructureBias = "neutral"
        self.internal_bias: StructureBias = "neutral"

    def _update_pivots(self, df: pd.DataFrame, idx: int):
        # Update pivot high/low pada bar yang sudah "confirmed"
        # pivot dianggap valid jika bar tengah dari window; untuk streaming bisa tertunda  (approx Pine)
        hl = self.internal_len
        sl = self.swing_len
        if idx >= hl and idx < len(df) - hl:
            # internal
            if rolling_pivot_high(df['high'], hl).iloc[idx]:
                self.internal_high_level = float(df['high'].iloc[idx])
            if rolling_pivot_low(df['low'], hl).iloc[idx]:
                self.internal_low_level = float(df['low'].iloc[idx])
        if idx >= sl and idx < len(df) - sl:
            # swing
            if rolling_pivot_high(df['high'], sl).iloc[idx]:
                self.swing_high_level = float(df['high'].iloc[idx])
            if rolling_pivot_low(df['low'], sl).iloc[idx]:
                self.swing_low_level = float(df['low'].iloc[idx])

    def step(self, df: pd.DataFrame, i: int, symbol: str, timeframe: str) -> List[SMCEvent]:
        events: List[SMCEvent] = []
        self._update_pivots(df, i)
        price = df['close'].iloc[i]
        t = df.index[i]

        # Internal structure cross -> BOS / CHoCH
        if self.internal_high_level is not None:
            if price > self.internal_high_level:
                tag = "internal_choch" if self.internal_bias == "bear" else "internal_bos"
                self.internal_bias = "bull"
                events.append(SMCEvent(tag, symbol, timeframe, i, t, {'level': self.internal_high_level}))
        if self.internal_low_level is not None:
            if price < self.internal_low_level:
                tag = "internal_choch" if self.internal_bias == "bull" else "internal_bos"
                self.internal_bias = "bear"
                events.append(SMCEvent(tag, symbol, timeframe, i, t, {'level': self.internal_low_level}))

        # Swing structure
        if self.swing_high_level is not None:
            if price > self.swing_high_level:
                tag = "swing_choch" if self.swing_bias == "bear" else "swing_bos"
                self.swing_bias = "bull"
                events.append(SMCEvent(tag, symbol, timeframe, i, t, {'level': self.swing_high_level}))
        if self.swing_low_level is not None:
            if price < self.swing_low_level:
                tag = "swing_choch" if self.swing_bias == "bull" else "swing_bos"
                self.swing_bias = "bear"
                events.append(SMCEvent(tag, symbol, timeframe, i, t, {'level': self.swing_low_level}))

        return events
