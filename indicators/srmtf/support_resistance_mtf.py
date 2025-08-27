from __future__ import annotations

"""
Support & Resistance Signals MTF (LuxAlgo‑style)
================================================

Implementasi Python modular yang diselaraskan dengan deskripsi indikator
TradingView "Support and Resistance Signals MTF [LuxAlgo]"
(Breakouts/Test/Retest/Rejections + MTF) dan rilis fiturnya
("filter false breakouts", non‑repaint pada breakout, Detection Timeframe,
Detection Length, opsi cek level historis, Sentiment Profile & Bull/Bear Nodes).

Rujukan ide/fitur: lihat halaman resmi LuxAlgo di TradingView.

Catatan:
- Deteksi sinyal **non‑repaint**: semua keputusan berbasis **candle close**
  yang sudah final.
- **Breakouts** tidak difilter volume (sesuai deskripsi), namun volume
  tetap dilaporkan pada event. Opsi filter false‑breakout disediakan.
- **Rejections** = pin bar volume tinggi (upper/lower shadow besar +
  volume di atas SMA). 
- **MTF**: pengguna bisa memilih satu atau banyak Detection Timeframe
  (mis. hanya "4h" untuk proyeksi HTF ke chart 15m), atau memakai
  preset Intraday(1m/5m/15m) & HTF(1h/4h) seperti permintaan user.
- **Sentiment Profile & Bullish/Bearish Nodes**: opsional; dihitung dari
  interaksi sinyal terakhir pada suatu zona.

© 2025 — Indikator Trading / Modular RajaDollar Compatible
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Util Timeframe & Resample
# ---------------------------------------------------------------------------
_TF_ALIASES = {"1m": "1T", "5m": "5T", "15m": "15T", "1h": "1H", "4h": "4H"}
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


def tf_to_offset(tf: str) -> str:
    if tf not in _TF_ALIASES:
        raise ValueError(f"TF tidak didukung: {tf}")
    return _TF_ALIASES[tf]


def tf_to_minutes(tf: str) -> int:
    if tf not in _TF_MINUTES:
        raise ValueError(f"TF tidak didukung: {tf}")
    return _TF_MINUTES[tf]


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample OHLCV ke timeframe target.

    df: index datetime, kolom ['open','high','low','close','volume']
    """
    rule = tf_to_offset(tf)
    # Resample dengan method chaining yang lebih eksplisit
    resampled = df.resample(rule, label="right", closed="right")
    out = resampled.agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    kind: str  # 'R' (resistance) atau 'S' (support)
    top: float
    bottom: float
    level: float  # level pivot inti
    tf: str  # timeframe deteksi zona
    created_at: pd.Timestamp
    margin_factor: float  # volatilitas relatif di saat pembentukan
    broken: bool = False  # ditandai ketika pernah ditembus tegas
    sentiment: Optional[str] = None  # 'bullish'|'bearish'|'neutral'
    score: float = 0.0  # skor sentimen sederhana

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def mid(self) -> float:
        return 0.5 * (self.top + self.bottom)


@dataclass
class Signal:
    ts: pd.Timestamp
    tf_chart: str
    kind: str  # 'BULL_BREAKOUT','BEAR_BREAKOUT','TEST_R','TEST_S','RETEST_R','RETEST_S','REJECT_UP','REJECT_DOWN'
    price: float
    ref_zone: Zone
    volume: Optional[float] = None
    volume_sma: Optional[float] = None
    atr: Optional[float] = None


# ---------------------------------------------------------------------------
# Inti Perhitungan
# ---------------------------------------------------------------------------
class SupportResistanceMTF:
    def __init__(
        self,
        detection_length: int = 15,
        sr_margin: float = 2.0,  # 0.1..10 serupa pengali ketebalan zona
        filter_false_breakouts: bool = True,  # opsi baru (rilis 2023)
        use_prev_historical_levels: bool = True,
        atr_len: int = 17,
        vol_sma_len: int = 17,
        rejection_shadow_mult: float = 1.618,
        rejection_vol_mult: float = 1.618,
        false_breakout_atr_mult: float = 0.5,  # syarat ekstra utk breakout valid bila filter aktif
    ):
        """
        detection_length : panjang pivot deteksi pada TF deteksi.
        sr_margin        : pengali ketebalan zona (lebih tebal pada volatilitas tinggi).
        filter_false_breakouts : aktifkan filter breakout palsu (opsional).
        use_prev_historical_levels : sertakan level historis lama (tidak hanya terbaru).
        atr_len, vol_sma_len : parameter ATR & SMA Volume (rekomendasi 17 dari skrip asli).
        """
        self.detection_length = int(detection_length)
        self.sr_margin = float(sr_margin)
        self.filter_false_breakouts = bool(filter_false_breakouts)
        self.use_prev_historical_levels = bool(use_prev_historical_levels)
        self.atr_len = int(atr_len)
        self.vol_sma_len = int(vol_sma_len)
        self.rejection_shadow_mult = float(rejection_shadow_mult)
        self.rejection_vol_mult = float(rejection_vol_mult)
        self.false_breakout_atr_mult = float(false_breakout_atr_mult)

    # --------------------------- Pivot Helpers ---------------------------- #
    @staticmethod
    def _pivot_high(high: pd.Series, L: int) -> pd.Series:
        roll = high.rolling(window=2 * L + 1, center=True)
        ph = (high == roll.max()) & (~high.isna())
        return ph.fillna(False)

    @staticmethod
    def _pivot_low(low: pd.Series, L: int) -> pd.Series:
        roll = low.rolling(window=2 * L + 1, center=True)
        pl = (low == roll.min()) & (~low.isna())
        return pl.fillna(False)

    @staticmethod
    def _atr(df: pd.DataFrame, length: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(length, min_periods=1).mean()

    # --------------------------- Zona dari Pivot -------------------------- #
    def _build_zones_from_pivots(self, df: pd.DataFrame, tf: str, L_scaled: int) -> List[Zone]:
        # Volatilitas relatif untuk ketebalan zona
        pHST = df["high"].rolling(L_scaled, min_periods=1).max()
        pLST = df["low"].rolling(L_scaled, min_periods=1).min()
        range_pct = (pHST - pLST) / pHST.replace(0, np.nan)
        range_pct = range_pct.fillna(0.0)

        ph = self._pivot_high(df["high"], L_scaled)
        pl = self._pivot_low(df["low"], L_scaled)

        zones: List[Zone] = []
        for ts, is_ph in ph.items():
            if not is_ph:
                continue
            level = float(df.at[ts, "high"])
            r = float(range_pct.at[ts])
            thickness = level * (r * 0.17 * self.sr_margin)
            top = level
            bottom = max(level - thickness, 0.0)
            # Konversi eksplisit ke string terlebih dahulu untuk menghindari error Hashable
            ts_pd = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(str(ts))
            zones.append(Zone("R", top, bottom, level, tf, ts_pd, r))

        for ts, is_pl in pl.items():
            if not is_pl:
                continue
            level = float(df.at[ts, "low"])
            r = float(range_pct.at[ts])
            thickness = level * (r * 0.17 * self.sr_margin)
            bottom = level
            top = level + thickness
            # Konversi eksplisit ke string terlebih dahulu untuk menghindari error Hashable
            ts_pd = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(str(ts))
            zones.append(Zone("S", top, bottom, level, tf, ts_pd, r))

        zones = self._merge_close_zones(zones)
        return zones

    @staticmethod
    def _overlap(z1: Zone, z2: Zone) -> bool:
        if z1.kind != z2.kind:
            return False
        return not (z1.top < z2.bottom or z2.top < z1.bottom)

    def _merge_close_zones(self, zones: List[Zone]) -> List[Zone]:
        if not zones:
            return zones
        zones_sorted = sorted(zones, key=lambda z: (z.kind, z.level, z.created_at))
        merged: List[Zone] = []
        cur = zones_sorted[0]
        for z in zones_sorted[1:]:
            if self._overlap(cur, z):
                cur_top = max(cur.top, z.top)
                cur_bottom = min(cur.bottom, z.bottom)
                cur_level = (cur.level + z.level) / 2.0
                cur = Zone(cur.kind, cur_top, cur_bottom, cur_level, cur.tf, min(cur.created_at, z.created_at), (cur.margin_factor + z.margin_factor) / 2.0, broken=(cur.broken or z.broken))
            else:
                merged.append(cur)
                cur = z
        merged.append(cur)
        return merged

    # --------------------------- Pipeline MTF ----------------------------- #
    def compute_levels_for_tf(self, df_tf: pd.DataFrame, tf_detect: str, chart_tf: Optional[str] = None) -> List[Zone]:
        L = self.detection_length
        if chart_tf is not None:
            # scale detection length agar konsisten antar TF (mirip tf_m/ch_m)
            L = int(max(2, round(L * tf_to_minutes(tf_detect) / tf_to_minutes(chart_tf))))
        return self._build_zones_from_pivots(df_tf, tf_detect, L)

    def compute_mtf_levels(
        self,
        data_by_tf: Dict[str, pd.DataFrame],
        chart_tf: str,
        detection_tfs: Optional[Iterable[str]] = None,
        use_presets: bool = False,
    ) -> Dict[str, List[Zone]]:
        """Hitung zona per TF.
        - Jika detection_tfs diberikan → kembalikan di key 'DETECT'.
        - Jika use_presets=True → kelompok 'INTRADAY' (1m/5m/15m) & 'HTF' (1h/4h).
        """
        if detection_tfs:
            levels: List[Zone] = []
            for tf in detection_tfs:
                if tf not in data_by_tf:
                    raise ValueError(f"Data TF {tf} tidak tersedia untuk deteksi")
                levels.extend(self.compute_levels_for_tf(data_by_tf[tf], tf, chart_tf))
            return {"DETECT": self._merge_close_zones(levels)}

        if not use_presets:
            # default: gunakan semua TF sebagai satu kumpulan
            all_tfs = [tf for tf in ("1m","5m","15m","1h","4h") if tf in data_by_tf]
            levels_all: List[Zone] = []
            for tf in all_tfs:
                levels_all.extend(self.compute_levels_for_tf(data_by_tf[tf], tf, chart_tf))
            return {"ALL": self._merge_close_zones(levels_all)}

        # presets: INTRADAY & HTF
        intraday, htf = ("1m","5m","15m"), ("1h","4h")
        lev_i: List[Zone] = []
        lev_h: List[Zone] = []
        for tf in intraday:
            if tf in data_by_tf:
                lev_i.extend(self.compute_levels_for_tf(data_by_tf[tf], tf, chart_tf))
        for tf in htf:
            if tf in data_by_tf:
                lev_h.extend(self.compute_levels_for_tf(data_by_tf[tf], tf, chart_tf))
        return {"INTRADAY": self._merge_close_zones(lev_i), "HTF": self._merge_close_zones(lev_h)}

    # --------------------------- Sinyal ----------------------------------- #
    def detect_signals(self, chart_df: pd.DataFrame, chart_tf: str, zones: List[Zone]) -> List[Signal]:
        signals: List[Signal] = []
        if chart_df.empty or not zones:
            return signals

        atr = self._atr(chart_df, self.atr_len)
        v_sma = chart_df["volume"].rolling(self.vol_sma_len, min_periods=1).mean()

        close = chart_df["close"]
        high = chart_df["high"]
        low = chart_df["low"]
        open_ = chart_df["open"]

        for i in range(1, len(chart_df)):
            ts = chart_df.index[i]
            c0, c1 = float(close.iloc[i - 1]), float(close.iloc[i])
            h1, l1 = float(high.iloc[i]), float(low.iloc[i])
            o1 = float(open_.iloc[i])
            atr1 = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
            v1 = float(chart_df["volume"].iloc[i])
            vs1 = float(v_sma.iloc[i]) if not np.isnan(v_sma.iloc[i]) else 0.0

            for z in zones:
                # Ambang breakout: gunakan close, opsi filter palsu dengan ATR
                if z.kind == "R":
                    passed = c0 <= z.top and c1 > z.top
                    if passed:
                        if self.filter_false_breakouts and atr1 > 0:
                            passed = (c1 - z.top) >= self.false_breakout_atr_mult * atr1
                        if passed:
                            signals.append(Signal(ts, chart_tf, "BULL_BREAKOUT", c1, z, v1, vs1, atr1))
                            z.broken = True
                            continue

                    # Test/Retest Resistance: sentuh lalu gagal tembus (non‑repaint)
                    touched = (h1 >= z.bottom) and (c1 < z.bottom) and (high.iloc[i - 1] < z.bottom)
                    if touched:
                        kind = "RETEST_R" if any(s.kind == "BULL_BREAKOUT" and s.ref_zone is z for s in signals[-50:]) else "TEST_R"
                        signals.append(Signal(ts, chart_tf, kind, c1, z, v1, vs1, atr1))
                        continue

                    # Rejection up: upper shadow besar + volume tinggi
                    upper_shadow = h1 - max(o1, c1)
                    if upper_shadow >= self.rejection_shadow_mult * atr1 and v1 >= self.rejection_vol_mult * vs1 and h1 >= z.bottom:
                        signals.append(Signal(ts, chart_tf, "REJECT_UP", c1, z, v1, vs1, atr1))

                else:  # Support
                    passed = (c0 >= z.bottom) and (c1 < z.bottom)
                    if passed:
                        if self.filter_false_breakouts and atr1 > 0:
                            passed = (z.bottom - c1) >= self.false_breakout_atr_mult * atr1
                        if passed:
                            signals.append(Signal(ts, chart_tf, "BEAR_BREAKOUT", c1, z, v1, vs1, atr1))
                            z.broken = True
                            continue

                    touched = (l1 <= z.top) and (c1 > z.top) and (low.iloc[i - 1] > z.top)
                    if touched:
                        kind = "RETEST_S" if any(s.kind == "BEAR_BREAKOUT" and s.ref_zone is z for s in signals[-50:]) else "TEST_S"
                        signals.append(Signal(ts, chart_tf, kind, c1, z, v1, vs1, atr1))
                        continue

                    lower_shadow = min(o1, c1) - l1
                    if lower_shadow >= self.rejection_shadow_mult * atr1 and v1 >= self.rejection_vol_mult * vs1 and l1 <= z.top:
                        signals.append(Signal(ts, chart_tf, "REJECT_DOWN", c1, z, v1, vs1, atr1))

        return signals

    # -------------------- Sentiment Profile & Nodes (opsional) ------------- #
    def build_sentiment(self, zones: List[Zone], signals: List[Signal]) -> None:
        if not zones or not signals:
            return
        # Skor sederhana: breakout searah +2, rejection searah +1, test/retest +0.5, lawan arah negatif
        weights = {
            "BULL_BREAKOUT": {"R": +2, "S": +2},
            "BEAR_BREAKOUT": {"R": -2, "S": -2},
            "REJECT_UP": {"R": -1, "S": +1},
            "REJECT_DOWN": {"R": +1, "S": -1},
            "TEST_R": {"R": -0.5, "S": 0},
            "TEST_S": {"R": 0, "S": +0.5},
            "RETEST_R": {"R": -0.5, "S": 0},
            "RETEST_S": {"R": 0, "S": +0.5},
        }
        # akumulasikan per zona
        for z in zones:
            z.score = 0.0
        for s in signals:
            z = s.ref_zone
            z.score += weights.get(s.kind, {}).get(z.kind, 0.0)
        for z in zones:
            if z.score > 1.0:
                z.sentiment = "bullish"
            elif z.score < -1.0:
                z.sentiment = "bearish"
            else:
                z.sentiment = "neutral"

    # --------------------------- API Tingkat Tinggi ----------------------- #
    def _prepare_data_by_tf(self, base_1m: Optional[pd.DataFrame], data_by_tf: Optional[Dict[str, pd.DataFrame]], tfs: Iterable[str]) -> Dict[str, pd.DataFrame]:
        if data_by_tf is None:
            if base_1m is None:
                raise ValueError("Berikan either data_by_tf atau base_1m untuk resample.")
            out = {}
            for tf in tfs:
                out[tf] = resample_ohlcv(base_1m, tf) if tf != "1m" else base_1m.copy()
            return out
        # gunakan hanya TF yang diminta
        return {tf: data_by_tf[tf] for tf in tfs if tf in data_by_tf}

    def compute(
        self,
        chart_df: pd.DataFrame,
        chart_tf: str,
        base_1m: Optional[pd.DataFrame] = None,
        data_by_tf: Optional[Dict[str, pd.DataFrame]] = None,
        detection_tfs: Optional[Iterable[str]] = None,
        use_presets: bool = True,  # default sesuai permintaan user (INTRADAY & HTF)
        include_sentiment_profile: bool = True,
    ) -> Tuple[Dict[str, List[Zone]], Dict[str, List[Signal]]]:
        """
        Hitung zona & sinyal.
        - Jika detection_tfs ditentukan (mis. ["4h"]) → output key 'DETECT'.
        - Jika use_presets=True → key 'INTRADAY' & 'HTF'.
        - Selainnya → key 'ALL'.
        Return: (levels_by_group, signals_by_group)
        """
        tfs_all = ("1m","5m","15m","1h","4h")
        data_by_tf = self._prepare_data_by_tf(base_1m, data_by_tf, tfs_all)

        groups = self.compute_mtf_levels(
            data_by_tf=data_by_tf,
            chart_tf=chart_tf,
            detection_tfs=detection_tfs,
            use_presets=use_presets,
        )

        signals_by_group: Dict[str, List[Signal]] = {}
        for k, zlist in groups.items():
            signals = self.detect_signals(chart_df, chart_tf, zlist)
            if include_sentiment_profile:
                self.build_sentiment(zlist, signals)
            # Jika use_prev_historical_levels=False, buang zona yang sudah 'broken' jauh dari harga
            if not self.use_prev_historical_levels:
                last_close = float(chart_df["close"].iloc[-1])
                keep: List[Zone] = []
                for z in zlist:
                    if not z.broken:
                        keep.append(z)
                    else:
                        # masih simpan bila dekat dengan harga saat ini (potensi retest)
                        if abs(z.mid() - last_close) <= 2.5 * chart_df["close"].rolling(self.atr_len, min_periods=1).std().iloc[-1]:
                            keep.append(z)
                groups[k] = keep
            signals_by_group[k] = signals

        return groups, signals_by_group


# ---------------------------------------------------------------------------
# Contoh Pemakaian (sanity)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock data 1m
    rng = pd.date_range("2025-08-01", periods=2000, freq="1T")
    price = np.cumsum(np.random.randn(len(rng)) * 0.1) + 1.0
    high = price + np.random.rand(len(rng)) * 0.05
    low = price - np.random.rand(len(rng)) * 0.05
    df1m = pd.DataFrame({
        "open": price,
        "high": high,
        "low": low,
        "close": price,
        "volume": np.random.randint(100, 2000, size=len(rng)),
    }, index=rng)

    chart_tf = "15m"
    chart_df = resample_ohlcv(df1m, chart_tf)

    srm = SupportResistanceMTF(
        detection_length=15,
        sr_margin=2.0,
        filter_false_breakouts=True,
        use_prev_historical_levels=True,
    )

    # Mode preset (INTRADAY & HTF)
    levels, signals = srm.compute(chart_df, chart_tf, base_1m=df1m, use_presets=True)
    print("Preset →", {k: (len(v), len(signals[k])) for k, v in levels.items()})

    # Mode Detection TF tunggal (contoh: 4h diproyeksikan ke chart 15m)
    levels2, signals2 = srm.compute(chart_df, chart_tf, base_1m=df1m, detection_tfs=["4h"], use_presets=False)
    print("Detect[4h] →", {k: (len(v), len(signals2[k])) for k, v in levels2.items()})
