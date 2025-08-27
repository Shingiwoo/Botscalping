# indicators/scsignals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Mapping
import numpy as np
import pandas as pd
from collections import deque

# ---------- Konfigurasi ----------
@dataclass
class SCConfig:
    # channel & line
    length: int = 20
    sma_period: int = 15
    atr_len: int = 14
    atr_mult: float = 0.6

    # HTF trend filter (EMA fast/slow)
    use_htf: bool = True
    htf: str = "1m"          # "1m","5m","15m","1h","4h"
    ema_fast_len: int = 20
    ema_slow_len: int = 60

    # ADX filter
    use_adx: bool = True
    adx_len: int = 16
    min_adx: float = 18.0

    # Momentum filters
    use_body_atr: bool = True
    min_body_atr: float = 0.38
    use_width_atr: bool = True
    min_width_atr: float = 1.20

    # RSI (opsional)
    use_rsi: bool = False
    rsi_len: int = 14
    rsi_buy: float = 52.0
    rsi_sell: float = 48.0

    # Lainnya
    cooldown_bars: int = 5   # cegah spam sinyal (jumlah bar di TF dasar)
    base_tf: str = "1m"      # TF data yang masuk ke indikator (untuk resample HTF)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]] = None) -> "SCConfig":
        if not d:
            return cls()
        fields = {k: d[k] for k in cls.__annotations__.keys() if k in d}
        return cls(**fields)

# ---------- Utilitas Teknis ----------
_PANDAS_OFFSET = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D"
}


def _num(s: pd.Series) -> pd.Series:
    """Pastikan seri bertipe float64 agar operasi numerik aman."""
    return pd.to_numeric(s, errors="coerce").astype("float64").fillna(0.0)

def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA. Kompatibel dengan Pine ta.rma."""
    if length <= 0:
        return series.copy()
    alpha = 1.0 / length
    out = np.empty(len(series), dtype=float)
    out[:] = np.nan
    if len(series) == 0:
        return pd.Series(out, index=series.index)
    out[0] = series.iloc[:length].mean() if len(series) >= length else series.iloc[0]
    for i in range(1, len(series)):
        prev = out[i-1]
        out[i] = alpha * series.iloc[i] + (1.0 - alpha) * prev
    return pd.Series(out, index=series.index)

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int) -> pd.Series:
    return _rma(_true_range(h, l, c), length)

def _rsi(c: pd.Series, length: int) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = _rma(up, length) / _rma(down, length)
    return 100 - (100 / (1 + rs))

def _adx(h: pd.Series, l: pd.Series, c: pd.Series, length: int) -> pd.Series:
    # Pastikan numerik untuk komparasi dan ketenangan type checker
    h = _num(h)
    l = _num(l)
    c = _num(c)
    up_move = h.diff()
    dn_move = -l.diff()
    # Gunakan mask vektor & bandingkan terhadap 0.0 (float)
    cond_plus  = (up_move > dn_move) & (up_move > 0.0)
    cond_minus = (dn_move > up_move) & (dn_move > 0.0)
    plus_dm  = pd.Series(np.where(cond_plus,  up_move,  0.0), index=h.index, dtype='float64')
    minus_dm = pd.Series(np.where(cond_minus, dn_move, 0.0), index=h.index, dtype='float64')
    atr_len = length
    atrx = _atr(h, l, c, atr_len).replace(0, np.nan)
    pdi = 100 * _rma(plus_dm, length) / atrx
    mdi = 100 * _rma(minus_dm, length) / atrx
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _rma(dx, length)

def _donchian_high(h: pd.Series, length: int) -> pd.Series:
    return h.rolling(length, min_periods=1).max()

def _donchian_low(l: pd.Series, length: int) -> pd.Series:
    return l.rolling(length, min_periods=1).min()

def _resample_close(df: pd.DataFrame, tf: str) -> pd.Series:
    """Ambil close HTF (label right, closed right) untuk anti-repaint."""
    off = _PANDAS_OFFSET.get(tf, tf)
    rs = df["close"].resample(off, label="right", closed="right").last()
    return rs

# ---------- Inti Indikator ----------
class SCSignals:
    """
    Implementasi Python dari "Scalper's Channel with Signals v2.2".
    - Sinyal hanya diproduksi pada bar close TF dasar (selaras Pine `barstate.isconfirmed`)
    - Semua filter (HTF EMA, ADX, Body/ATR, Width/ATR, RSI, cooldown) tersedia.
    """
    def __init__(self, config: Optional[SCConfig] = None):
        self.cfg = config or SCConfig()
        self.df: pd.DataFrame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.last_buy_bar_i: Optional[int] = None
        self.last_sell_bar_i: Optional[int] = None

    # ---------- Batch / vectorized ----------
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: DataFrame dengan index datetime & kolom: open, high, low, close, volume
        return df_out dengan kolom tambahan: scalper_line, upper, lower, width_atr, adx, body_to_atr,
                                            ema_fast_htf, ema_slow_htf, trend_up, trend_dn,
                                            buy_signal, sell_signal
        """
        assert {"open", "high", "low", "close"}.issubset(df.columns)
        d = df.copy().sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in d:
                d[col] = _num(d[col])

        # ATR & Scalper line
        atr = _atr(d["high"], d["low"], d["close"], self.cfg.atr_len)
        line = d["close"].rolling(self.cfg.sma_period).mean() - (atr * self.cfg.atr_mult)

        upper = _donchian_high(d["high"], self.cfg.length)
        lower = _donchian_low(d["low"], self.cfg.length)
        width_atr = (upper - lower) / atr.replace(0, np.nan)

        # ADX / BodyATR / RSI
        adx = _adx(d["high"], d["low"], d["close"], self.cfg.adx_len)
        body_to_atr = (d["close"] - d["open"]).abs() / atr.replace(0, np.nan)
        rsi = _rsi(d["close"], self.cfg.rsi_len)

        # HTF EMA
        if self.cfg.use_htf:
            close_htf = _resample_close(d, self.cfg.htf).reindex(d.index, method="ffill")
            ema_fast_htf = close_htf.ewm(span=self.cfg.ema_fast_len, adjust=False).mean()
            ema_slow_htf = close_htf.ewm(span=self.cfg.ema_slow_len, adjust=False).mean()
            trend_up = ema_fast_htf > ema_slow_htf
            trend_dn = ema_fast_htf < ema_slow_htf
        else:
            ema_fast_htf = ema_slow_htf = trend_up = trend_dn = pd.Series(False, index=d.index)

        # Base cross (bar close)
        cross_up = (d["close"].shift(1) <= line.shift(1)) & (d["close"] > line)
        cross_dn = (d["close"].shift(1) >= line.shift(1)) & (d["close"] < line)

        # Filters
        buy_filters = (
            (~self.cfg.use_htf | trend_up) &
            (~self.cfg.use_adx | (adx >= self.cfg.min_adx)) &
            (~self.cfg.use_body_atr | (body_to_atr >= self.cfg.min_body_atr)) &
            (~self.cfg.use_width_atr | (width_atr >= self.cfg.min_width_atr)) &
            (~self.cfg.use_rsi | (rsi > self.cfg.rsi_buy)) &
            (d["close"] < upper)
        )

        sell_filters = (
            (~self.cfg.use_htf | trend_dn) &
            (~self.cfg.use_adx | (adx >= self.cfg.min_adx)) &
            (~self.cfg.use_body_atr | (body_to_atr >= self.cfg.min_body_atr)) &
            (~self.cfg.use_width_atr | (width_atr >= self.cfg.min_width_atr)) &
            (~self.cfg.use_rsi | (rsi < self.cfg.rsi_sell)) &
            (d["close"] > lower)
        )

        # Cooldown (vector: kita hitung belakangan di streamer; di sini flag mentah)
        buy_raw = (cross_up & buy_filters).astype("float64")
        sell_raw = (cross_dn & sell_filters).astype("float64")

        out = d.copy()
        out["scalper_line"] = line
        out["upper"] = upper
        out["lower"] = lower
        out["width_atr"] = width_atr
        out["atr"] = atr
        out["adx"] = adx
        out["body_to_atr"] = body_to_atr
        out["rsi"] = rsi
        out["ema_fast_htf"] = ema_fast_htf if self.cfg.use_htf else np.nan
        out["ema_slow_htf"] = ema_slow_htf if self.cfg.use_htf else np.nan
        out["trend_up"] = trend_up if self.cfg.use_htf else False
        out["trend_dn"] = trend_dn if self.cfg.use_htf else False
        out["buy_raw"] = buy_raw
        out["sell_raw"] = sell_raw
        return out

    # ---------- Streaming ----------
    def on_candle(self, ts_ms: int, o: float, h: float, l: float, c: float, v: float,
                  is_closed: bool, symbol: str = "") -> Optional[Dict[str, Any]]:
        """
        Panggil fungsi ini setiap terima event kline.
        - Hanya saat `is_closed=True` sinyal akan dihitung & bisa keluar.
        Return dict event sinyal atau None.
        """
        # simpan bar (pakai index datetime UTC)
        ts = pd.to_datetime(ts_ms, unit="ms", utc=True)
        self.df.loc[ts, ["open", "high", "low", "close", "volume"]] = [o, h, l, c, v]

        if not is_closed:
            return None

        # hitung hanya pada window yang diperlukan (hemat)
        win = max(self.cfg.length, self.cfg.sma_period, self.cfg.atr_len, self.cfg.adx_len, self.cfg.ema_slow_len) + 50
        dfw = self.df.iloc[-win:].copy()
        out = self.compute_all(dfw)

        last = out.iloc[-1]
        idx = len(out) - 1

        # base raw signals
        buy_raw = bool(last["buy_raw"])
        sell_raw = bool(last["sell_raw"])

        # cooldown logic
        def _ok_cooldown(last_bar_i: Optional[int]) -> bool:
            return (last_bar_i is None) or ((idx - last_bar_i) >= self.cfg.cooldown_bars)

        event: Optional[Dict[str, Any]] = None
        if buy_raw and _ok_cooldown(self.last_buy_bar_i):
            self.last_buy_bar_i = idx
            event = self._make_event(symbol, "BUY", last, ts)
        elif sell_raw and _ok_cooldown(self.last_sell_bar_i):
            self.last_sell_bar_i = idx
            event = self._make_event(symbol, "SELL", last, ts)

        return event

    def _make_event(self, symbol: str, side: str, row: pd.Series, ts: pd.Timestamp) -> Dict[str, Any]:
        return {
            "type": "SCSignals",
            "symbol": symbol,
            "side": side,                 # "BUY" / "SELL"
            "time": ts.isoformat(),
            "price": float(row["close"]),
            "scalper_line": float(row["scalper_line"]),
            "upper": float(row["upper"]),
            "lower": float(row["lower"]),
            "atr": float(row["atr"]),
            "adx": float(row["adx"]),
            "body_to_atr": float(row["body_to_atr"]),
            "width_atr": float(row["width_atr"]),
            "trend_up": bool(row.get("trend_up", False)),
            "trend_dn": bool(row.get("trend_dn", False)),
            "meta": {
                "cfg": self.cfg.__dict__
            }
        }
 

def compute_all(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper util: mengembalikan seri penting tanpa field palsu."""
    cfg = cfg or {}
    ind = SCSignals(SCConfig.from_dict(cfg))
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    out = ind.compute_all(df)
    return {
        "scalper_line": out["scalper_line"],
        "upper": out["upper"],
        "lower": out["lower"],
        "atr": out["atr"],
        "adx": out["adx"],
        "body_to_atr": out["body_to_atr"],
        "width_atr": out["width_atr"],
        "rsi": out["rsi"],
        "buy_raw": out["buy_raw"],
        "sell_raw": out["sell_raw"],
    }

