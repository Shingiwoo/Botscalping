from __future__ import annotations
import pandas as pd
from typing import List
from .types import FairValueGap, SMCEvent, SMCConfig

class FVGDetector:
    def __init__(self, cfg: SMCConfig):
        self.cfg = cfg
        self.active: List[FairValueGap] = []

    def step(self, df: pd.DataFrame, i: int, symbol: str, timeframe: str) -> List[SMCEvent]:
        out: List[SMCEvent] = []
        if i < 2: return out
        hi = float(df['high'].iloc[i])
        lo = float(df['low'].iloc[i])
        hi2 = float(df['high'].iloc[i-2])
        lo2 = float(df['low'].iloc[i-2])
        t1 = df.index[i-1]; t0 = df.index[i]

        # Bullish FVG: low[0] > high[2]
        if lo > hi2:
            top = lo
            bottom = hi2
            fvg = FairValueGap('bull', top=top, bottom=bottom, left_time=t1, right_time=t0)
            self.active.insert(0, fvg)
            out.append(SMCEvent("fvg_bull", symbol, timeframe, i, t0, {'top': top, 'bottom': bottom}))

        # Bearish FVG: high[0] < low[2]
        if hi < lo2:
            top = lo2
            bottom = hi
            fvg = FairValueGap('bear', top=top, bottom=bottom, left_time=t1, right_time=t0)
            self.active.insert(0, fvg)
            out.append(SMCEvent("fvg_bear", symbol, timeframe, i, t0, {'top': top, 'bottom': bottom}))

        # Auto-delete jika gap terisi
        for f in list(self.active):
            if (f.bias == 'bull' and float(df['low'].iloc[i]) <= f.bottom) or                (f.bias == 'bear' and float(df['high'].iloc[i]) >= f.top):
                try: self.active.remove(f)
                except: pass
        return out
