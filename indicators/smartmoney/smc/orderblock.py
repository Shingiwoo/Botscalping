from __future__ import annotations
import pandas as pd
from typing import List, Deque, Optional
from collections import deque
from .types import OrderBlock, SMCEvent, StructureBias, SMCConfig

class OrderBlockEngine:
    def __init__(self, cfg: SMCConfig):
        self.cfg = cfg
        self.internal_obs: Deque[OrderBlock] = deque([], maxlen=cfg.ob_max_keep)
        self.swing_obs: Deque[OrderBlock] = deque([], maxlen=cfg.ob_max_keep)
        # last cross info untuk pembuatan OB (cari candle berlawanan sebelum cross)
        self._last_bull_cross_i: Optional[int] = None
        self._last_bear_cross_i: Optional[int] = None

    def _find_last_opposite_candle_range(self, df: pd.DataFrame, i_cross: int, lookback: int = 20):
        # Cari candle lawan sebelum cross untuk menentukan range OB
        start = max(0, i_cross - lookback)
        sub = df.iloc[start:i_cross]
        # bearish candle terakhir sebelum bullish cross
        bear = sub[sub['close'] < sub['open']]
        bull = sub[sub['close'] > sub['open']]
        last_bear = bear.iloc[-1] if len(bear) else None
        last_bull = bull.iloc[-1] if len(bull) else None
        return last_bull, last_bear

    def on_structure_event(self, ev: SMCEvent, df: pd.DataFrame) -> List[SMCEvent]:
        out: List[SMCEvent] = []
        i = ev.index
        t = ev.time
        if ev.type in ("internal_bos","internal_choch","swing_bos","swing_choch"):
            is_bull = ev.type in ("internal_bos","swing_bos") and ev.payload.get('level') is not None and df['close'].iloc[i] >= ev.payload['level']
            is_bear = ev.type in ("internal_bos","swing_bos") and ev.payload.get('level') is not None and df['close'].iloc[i] <= ev.payload['level']
            # The CHoCH will still create an OB in opposite side; we can treat all crosses similarly

            last_bull, last_bear = self._find_last_opposite_candle_range(df, i)

            if 'internal' in ev.type and self.cfg.show_internal_ob:
                if "bull" in ev.type or is_bull:
                    # Bullish break -> cari last bearish candle jadi Demand OB
                    if last_bear is not None:
                        ob = OrderBlock("internal","bull",
                                        top=float(last_bear['high']), bottom=float(last_bear['low']),
                                        start_idx=int(last_bear.name.value if hasattr(last_bear.name,'value') else df.index.get_loc(last_bear.name)),
                                        start_time=last_bear.name,
                                        mitigation=self.cfg.ob_mitigation)
                        self.internal_obs.appendleft(ob)
                        out.append(SMCEvent("internal_ob_new", ev.symbol, ev.timeframe, i, t, {'ob': ob.__dict__}))
                if "bear" in ev.type or is_bear:
                    # Bearish break -> cari last bullish candle jadi Supply OB
                    if last_bull is not None:
                        ob = OrderBlock("internal","bear",
                                        top=float(last_bull['high']), bottom=float(last_bull['low']),
                                        start_idx=int(last_bull.name.value if hasattr(last_bull.name,'value') else df.index.get_loc(last_bull.name)),
                                        start_time=last_bull.name,
                                        mitigation=self.cfg.ob_mitigation)
                        self.internal_obs.appendleft(ob)
                        out.append(SMCEvent("internal_ob_new", ev.symbol, ev.timeframe, i, t, {'ob': ob.__dict__}))

            if 'swing' in ev.type and self.cfg.show_swing_ob:
                if "bull" in ev.type or is_bull:
                    if last_bear is not None:
                        ob = OrderBlock("swing","bull",
                                        top=float(last_bear['high']), bottom=float(last_bear['low']),
                                        start_idx=int(last_bear.name.value if hasattr(last_bear.name,'value') else df.index.get_loc(last_bear.name)),
                                        start_time=last_bear.name,
                                        mitigation=self.cfg.ob_mitigation)
                        self.swing_obs.appendleft(ob)
                        out.append(SMCEvent("swing_ob_new", ev.symbol, ev.timeframe, i, t, {'ob': ob.__dict__}))
                if "bear" in ev.type or is_bear:
                    if last_bull is not None:
                        ob = OrderBlock("swing","bear",
                                        top=float(last_bull['high']), bottom=float(last_bull['low']),
                                        start_idx=int(last_bull.name.value if hasattr(last_bull.name,'value') else df.index.get_loc(last_bull.name)),
                                        start_time=last_bull.name,
                                        mitigation=self.cfg.ob_mitigation)
                        self.swing_obs.appendleft(ob)
                        out.append(SMCEvent("swing_ob_new", ev.symbol, ev.timeframe, i, t, {'ob': ob.__dict__}))
        # Cek mitigasi di bar i (price menyentuh / menyeberangi area OB)
        out += self._check_mitigation(df, i, ev.symbol, ev.timeframe, t)
        return out

    def _check_mitigation(self, df: pd.DataFrame, i: int, symbol: str, timeframe: str, t) -> List[SMCEvent]:
        out: List[SMCEvent] = []
        if i < 0: return out
        h = float(df['high'].iloc[i]); l = float(df['low'].iloc[i]); c = float(df['close'].iloc[i])

        def mitigated(ob: OrderBlock) -> bool:
            if ob.mitigation == "highlow":
                if ob.bias == "bear":  # supply
                    return h > ob.top
                else:
                    return l < ob.bottom
            else:  # "close"
                if ob.bias == "bear":
                    return c > ob.top
                else:
                    return c < ob.bottom

        # internal
        for ob in list(self.internal_obs):
            if mitigated(ob):
                out.append(SMCEvent("internal_ob_mitigated", symbol, timeframe, i, t, {'ob': ob.__dict__}))
                try: self.internal_obs.remove(ob)
                except: pass

        # swing
        for ob in list(self.swing_obs):
            if mitigated(ob):
                out.append(SMCEvent("swing_ob_mitigated", symbol, timeframe, i, t, {'ob': ob.__dict__}))
                try: self.swing_obs.remove(ob)
                except: pass

        return out
