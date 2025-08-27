import sys
from pathlib import Path

# Pastikan paket "Botscalping" bisa di-import saat tes dijalankan dari repo root
sys.path.append(str(Path(__file__).resolve().parents[1].parent))

import math
import queue
import numpy as np
import pandas as pd
import pytest

from Botscalping.indicators.rsi.ultimate_rsi import (
    ursi_vectorized,
    compute_ursi_df,
    UltimateRSI,
    URSIConfig,
    URSIAdapter,
)

# ---------- Helpers ----------
def make_df(n=600, seed=1337):
    rng = np.random.default_rng(seed)
    close = pd.Series(
        np.cumsum(rng.normal(0, 1, n)) + 100.0,
        index=pd.date_range("2024-01-01", periods=n, freq="1min"),
    )
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
    })
    return df


def triangle_df(n=300, low=99.0, high=101.0):
    # Sinyal deterministik (naik turun) untuk memancing cross
    half = n // 2
    up = np.linspace(low, high, half, dtype=float)
    dn = np.linspace(high, low, n - half, dtype=float)
    close = pd.Series(
        np.concatenate([up, dn]),
        index=pd.date_range("2024-02-01", periods=n, freq="1min"),
    )
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
    })
    return df


# ---------- Tests ----------
@pytest.mark.parametrize("source", ["close", "hl2", "hlc3", "ohlc4"])
def test_sources_vectorized_ok(source):
    df = make_df()
    out = ursi_vectorized(
        df, length=14, method1="RMA", smooth=14, method2="EMA", source=source
    )
    assert set(["arsi", "signal", "ob", "os", "mid"]).issubset(out.columns)
    assert len(out) == len(df)
    # arsi/signal di [0..100] ketika tidak NaN
    na = out["arsi"].notna()
    assert (out.loc[na, "arsi"].between(0, 100)).all()
    ns = out["signal"].notna()
    assert (out.loc[ns, "signal"].between(0, 100)).all()


def test_input_variants_series_iterable_ok():
    df = make_df()
    # DataFrame
    out_df = ursi_vectorized(df)
    # Series
    out_s = ursi_vectorized(df["close"])
    # Iterable (list)
    out_l = ursi_vectorized(df["close"].tolist())
    assert len(out_df) == len(out_s) == len(out_l) == len(df)
    for out in (out_df, out_s, out_l):
        assert (out["arsi"].dropna().between(0, 100)).all()
        assert (out["signal"].dropna().between(0, 100)).all()


@pytest.mark.parametrize("m1", ["EMA", "SMA", "RMA", "TMA"])
@pytest.mark.parametrize("m2", ["EMA", "SMA", "RMA", "TMA"])
def test_all_ma_combinations(m1, m2):
    df = make_df()
    out = ursi_vectorized(df, method1=m1, method2=m2)
    assert "arsi" in out and "signal" in out
    # tidak ada inf
    assert np.isfinite(out["arsi"].dropna()).all()
    assert np.isfinite(out["signal"].dropna()).all()


def test_compute_wrapper_columns_and_values():
    df = make_df()
    out = compute_ursi_df(df, source="hlc3")
    cols = {"URSI", "URSI_signal", "URSI_ob", "URSI_os", "URSI_mid"}
    assert cols.issubset(out.columns)
    assert out["URSI_ob"].iloc[-1] == 80.0
    assert out["URSI_os"].iloc[-1] == 20.0
    assert out["URSI_mid"].iloc[-1] == 50.0
    # original kolom tidak hilang
    assert {"open", "high", "low", "close"}.issubset(out.columns)


def test_streaming_ready_flag_and_snapshot():
    df = make_df(40)
    L, S = 14, 14
    ursi = UltimateRSI(URSIConfig(length=L, smooth=S, source="ohlc4"))
    ready_seen = False
    snaps = []
    for _, r in df.iterrows():
        snap = ursi.update((r.open, r.high, r.low, r.close))
        snaps.append(snap)
        if snap.ready:
            ready_seen = True
    assert ready_seen, "ready harus True setelah warmup"
    # Snapshot fields terisi (upper/lower/r boleh NaN sebelum cukup data)
    last = snaps[-1]
    assert last.arsi is None or (0.0 <= last.arsi <= 100.0)
    assert last.signal is None or (0.0 <= last.signal <= 100.0)


def test_streaming_vs_vectorized_cross_positions_align():
    df = make_df(400)
    L, S = 14, 14
    # Vectorized di sumber yang sama (ohlc4)
    vec = ursi_vectorized(df, length=L, smooth=S, source="ohlc4")
    vec_cross_up = (vec["arsi"].shift(1) <= vec["signal"].shift(1)) & (
        vec["arsi"] > vec["signal"]
    )
    vec_cross_dn = (vec["arsi"].shift(1) >= vec["signal"].shift(1)) & (
        vec["arsi"] < vec["signal"]
    )

    # Streaming (kumpulkan seri arsi/signal)
    ursi = UltimateRSI(URSIConfig(length=L, smooth=S, source="ohlc4"))
    stream_arsi, stream_sig = [], []
    for _, r in df.iterrows():
        snap = ursi.update((r.open, r.high, r.low, r.close))
        stream_arsi.append(ursi.arsi)
        stream_sig.append(ursi.signal)
    sA = pd.Series(stream_arsi, index=df.index, dtype="float64")
    sS = pd.Series(stream_sig, index=df.index, dtype="float64")
    st_cross_up = (sA.shift(1) <= sS.shift(1)) & (sA > sS)
    st_cross_dn = (sA.shift(1) >= sS.shift(1)) & (sA < sS)

    # Abaikan warmup awal
    warmup = max(L, S) + 5
    cu_diff = abs(int(st_cross_up.iloc[warmup:].sum()) - int(vec_cross_up.iloc[warmup:].sum()))
    cd_diff = abs(int(st_cross_dn.iloc[warmup:].sum()) - int(vec_cross_dn.iloc[warmup:].sum()))
    # Seharusnya sangat dekat; toleransi kecil
    assert cu_diff <= 2 and cd_diff <= 2


def test_adapter_event_structure_and_timestamp_queue_variants():
    df = triangle_df(120)  # seri deterministik
    # 1) queue.SimpleQueue (punya .put)
    q1 = queue.SimpleQueue()
    adp1 = URSIAdapter("BTCUSDT", URSIConfig(source="ohlc4"), q1)
    evt1 = None
    for t, r in df.iterrows():
        evt1 = adp1.on_price((r.open, r.high, r.low, r.close), timestamp=pd.Timestamp(t)) or evt1
    assert evt1 is not None
    for k in ["type", "name", "symbol", "time", "arsi", "signal", "config"]:
        assert k in evt1
    assert isinstance(evt1["time"], pd.Timestamp)
    assert 0.0 <= evt1["arsi"] <= 100.0
    assert 0.0 <= evt1["signal"] <= 100.0

    # 2) Dummy queue dengan put_nowait (menguji cabang alternatif)
    class DummyQ:
        def __init__(self):
            self.buf = []

        def put_nowait(self, x):
            self.buf.append(x)

    q2 = DummyQ()
    adp2 = URSIAdapter("ETHUSDT", URSIConfig(source="ohlc4"), q2)
    evt2 = None
    for t, r in df.iterrows():
        evt2 = (
            adp2.on_price(
                (r.open, r.high, r.low, r.close), timestamp=pd.Timestamp(t)
            )
            or evt2
        )
    assert evt2 is not None
    assert len(q2.buf) >= 1


def test_invalid_source_raises():
    df = make_df(80)
    with pytest.raises(ValueError):
        _ = ursi_vectorized(df, source="INVALID")  # type: ignore[arg-type]


@pytest.mark.parametrize("length,smooth", [(1, 1), (2, 1), (1, 3), (5, 1), (14, 14)])
def test_edge_lengths_no_crash(length, smooth):
    df = make_df(120)
    out = ursi_vectorized(df, length=length, smooth=smooth)
    assert "arsi" in out and "signal" in out


def test_constant_series_behavior():
    n = 120
    close = pd.Series(
        [100.0] * n, index=pd.date_range("2024-03-01", periods=n, freq="1min")
    )
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    out = ursi_vectorized(df)
    # Tidak crash; sebagian besar bisa NaN (karena den=0), tapi tidak boleh keluar dari [0..100]
    mask = out["arsi"].notna()
    assert (out.loc[mask, "arsi"].between(0, 100)).all()
    mask2 = out["signal"].notna()
    assert (out.loc[mask2, "signal"].between(0, 100)).all()

