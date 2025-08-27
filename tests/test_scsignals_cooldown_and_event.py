# tests/test_scsignals_cooldown_and_event.py
import pandas as pd
import numpy as np
from indicators.scsignal.scsignals import SCSignals, SCConfig


def make_df(n=50):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    half = n // 2
    down = np.linspace(10.0, 9.0, half, endpoint=False)
    up = np.linspace(9.0, 11.0, n - half)
    close = np.concatenate([down, up])
    df = pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": np.ones(n),
    }, index=ts)
    return df


def test_on_candle_event_shape_and_cooldown():
    cfg = SCConfig(
        length=5, sma_period=3, atr_len=3, atr_mult=0.1,
        use_htf=False, use_adx=False, use_body_atr=False, use_width_atr=False,
        cooldown_bars=3, base_tf="1m",
    )
    ind = SCSignals(cfg)

    df = make_df(15)
    events = []
    for ts, row in df.iterrows():
        e = ind.on_candle(
            ts_ms=int(ts.timestamp() * 1000),
            o=float(row.open), h=float(row.high), l=float(row.low),
            c=float(row.close), v=float(row.volume),
            is_closed=True, symbol="TESTUSDT",
        )
        if e:
            events.append(e)

    assert len(events) >= 1
    ev = events[-1]
    for key in ("type", "symbol", "side", "time", "price", "atr", "adx", "body_to_atr", "width_atr"):
        assert key in ev
    assert isinstance(ev["price"], float)

    # cooldown membatasi jumlah event
    assert len(events) <= (len(df) // max(cfg.cooldown_bars, 1)) + 2

