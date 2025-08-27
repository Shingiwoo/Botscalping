from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple
import pandas as pd
from .support_resistance_mtf import SupportResistanceMTF, Zone, Signal, resample_ohlcv

@dataclass
class SRServiceConfig:
    enabled: bool = True
    chart_tf: str = "15m"
    use_presets: bool = True                 # INTRADAY(1m/5m/15m) & HTF(1h/4h)
    detection_tfs: Optional[Iterable[str]] = None  # contoh: ["4h"]
    detection_length: int = 15
    sr_margin: float = 2.0
    filter_false_breakouts: bool = True
    use_prev_historical_levels: bool = True
    atr_len: int = 17
    vol_sma_len: int = 17
    rejection_shadow_mult: float = 1.618
    rejection_vol_mult: float = 1.618
    false_breakout_atr_mult: float = 0.5

class SRMTFService:
    """
    Layanan S/R MTF berulir:
    - bootstrap() sekali di awal (dari base_1m atau data_by_tf)
    - on_bar_close(chart_df) setiap bar close → emisi sinyal untuk bar terakhir
    - simpan levels_by_group untuk dipakai fitur "near zone"
    """
    def __init__(self, cfg: SRServiceConfig):
        self.cfg = cfg
        self.srm = SupportResistanceMTF(
            detection_length=cfg.detection_length,
            sr_margin=cfg.sr_margin,
            filter_false_breakouts=cfg.filter_false_breakouts,
            use_prev_historical_levels=cfg.use_prev_historical_levels,
            atr_len=cfg.atr_len,
            vol_sma_len=cfg.vol_sma_len,
            rejection_shadow_mult=cfg.rejection_shadow_mult,
            rejection_vol_mult=cfg.rejection_vol_mult,
            false_breakout_atr_mult=cfg.false_breakout_atr_mult,
        )
        self.levels_by_group: Dict[str, List[Zone]] = {}
        self._last_ts: Optional[pd.Timestamp] = None

    @staticmethod
    def from_coin_cfg(coin_cfg: dict) -> "SRMTFService":
        sr = coin_cfg.get("sr_mtf", {})
        cfg = SRServiceConfig(
            enabled=sr.get("enabled", True),
            chart_tf=sr.get("chart_tf", "15m"),
            use_presets=sr.get("use_presets", True),
            detection_tfs=sr.get("detection_tfs"),
            detection_length=int(sr.get("detection_length", 15)),
            sr_margin=float(sr.get("sr_margin", 2.0)),
            filter_false_breakouts=bool(sr.get("filter_false_breakouts", True)),
            use_prev_historical_levels=bool(sr.get("use_prev_historical_levels", True)),
            atr_len=int(sr.get("atr_len", 17)),
            vol_sma_len=int(sr.get("vol_sma_len", 17)),
            rejection_shadow_mult=float(sr.get("rejection_shadow_mult", 1.618)),
            rejection_vol_mult=float(sr.get("rejection_vol_mult", 1.618)),
            false_breakout_atr_mult=float(sr.get("false_breakout_atr_mult", 0.5)),
        )
        return SRMTFService(cfg)

    def bootstrap(self, *, chart_df: pd.DataFrame, base_1m: Optional[pd.DataFrame]=None,
                  data_by_tf: Optional[Dict[str, pd.DataFrame]]=None) -> None:
        """Hitung level awal sesuai konfigurasi (sekali di start atau saat reload data)."""
        if not self.cfg.enabled:
            self.levels_by_group = {}
            return
        levels, _ = self.srm.compute(
            chart_df=chart_df,
            chart_tf=self.cfg.chart_tf,
            base_1m=base_1m,
            data_by_tf=data_by_tf,
            detection_tfs=self.cfg.detection_tfs,
            use_presets=self.cfg.use_presets,
            include_sentiment_profile=True,
        )
        self.levels_by_group = levels
        self._last_ts = None

    def on_bar_close(self, chart_df: pd.DataFrame) -> Dict[str, List[Signal]]:
        """
        Panggil pada setiap bar close dari TF chart.
        Return: sinyal hanya untuk bar terakhir (per group).
        """
        if not self.cfg.enabled or not self.levels_by_group:
            return {}
        # Deteksi sinyal dari group level yang ada, hanya ambil bar terakhir
        ts_last = chart_df.index[-1]
        signals_by_group: Dict[str, List[Signal]] = {}
        for g, zones in self.levels_by_group.items():
            sigs = self.srm.detect_signals(chart_df, self.cfg.chart_tf, zones)
            sigs_last = [s for s in sigs if s.ts == ts_last]
            if sigs_last:
                signals_by_group[g] = sigs_last
        self._last_ts = ts_last
        return signals_by_group

    def zones(self) -> Dict[str, List[Zone]]:
        return self.levels_by_group
