from __future__ import annotations
import pandas as pd
from typing import List
from .utils import resample_ohlcv
from .types import LevelLines, SMCEvent, SMCConfig

def prev_period_hilo(df: pd.DataFrame, rule: str):
    rs = resample_ohlcv(df, rule)
    if len(rs) < 2: 
        return None, None
    prev = rs.iloc[-2]
    return float(prev['high']), float(prev['low'])

class MTFLevels:
    def __init__(self, cfg: SMCConfig):
        self.cfg = cfg
        self.lines = LevelLines()

    def step(self, df: pd.DataFrame, i: int, symbol: str, timeframe: str) -> List[SMCEvent]:
        out: List[SMCEvent] = []
        t = df.index[i]
        changed = False

        if self.cfg.show_daily:
            h,l = prev_period_hilo(df, "1D")
            if h is not None:
                self.lines.daily_high, self.lines.daily_low = h,l; changed = True
        if self.cfg.show_weekly:
            h,l = prev_period_hilo(df, "1W")
            if h is not None:
                self.lines.weekly_high, self.lines.weekly_low = h,l; changed = True
        if self.cfg.show_monthly:
            h,l = prev_period_hilo(df, "1M")
            if h is not None:
                self.lines.monthly_high, self.lines.monthly_low = h,l; changed = True

        if changed:
            out.append(SMCEvent("mtf_levels", symbol, timeframe, i, t, {'levels': self.lines.__dict__}))
        return out
