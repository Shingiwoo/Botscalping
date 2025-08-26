from __future__ import annotations
import pandas as pd
from typing import List
from .types import SMCEvent, SMCConfig
from .structure import StructureState
from .orderblock import OrderBlockEngine
from .fvg import FVGDetector
from .levels import MTFLevels
from .zones import ZonesEngine
from .utils import atr

class SMCIndicator:
    def __init__(self, symbol: str, timeframe: str, config: SMCConfig = SMCConfig()):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cfg = config
        self._struct = StructureState(config.swing_len, config.internal_len)
        self._ob = OrderBlockEngine(config)
        self._fvg = FVGDetector(config)
        self._lvl = MTFLevels(config)
        self._zones = ZonesEngine()
        self._last_atr = None

    def process_dataframe(self, df: pd.DataFrame) -> List[SMCEvent]:
        """Proses seluruh DataFrame OHLCV (index datetime). Kembalikan daftar event."""
        events: List[SMCEvent] = []
        if len(df) == 0:
            return events

        # precompute ATR untuk threshold EQH/EQL jika dibutuhkan
        self._last_atr = atr(df, self.cfg.atr_period)

        for i in range(len(df)):
            events.extend(self.step(df, i))
        return events

    def step(self, df: pd.DataFrame, i: int) -> List[SMCEvent]:
        evs: List[SMCEvent] = []
        evs += self._struct.step(df, i, self.symbol, self.timeframe)

        # Order blocks mengikuti event structure (BOS/CHoCH) + mitigasi
        ob_events: List[SMCEvent] = []
        for e in list(evs):
            if e.type in ("internal_bos","internal_choch","swing_bos","swing_choch"):
                ob_events += self._ob.on_structure_event(e, df)
        evs += ob_events

        # FVG
        if self.cfg.enable_fvg:
            evs += self._fvg.step(df, i, self.symbol, self.timeframe)

        # MTF Levels (opsional)
        evs += self._lvl.step(df, i, self.symbol, self.timeframe)

        # Zones
        evs += self._zones.step(df, i, self.symbol, self.timeframe, self._struct.swing_bias)

        return evs
