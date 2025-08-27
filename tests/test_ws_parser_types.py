# tests/test_ws_parser_types.py
import pytest

ws = pytest.importorskip("indicators.scsignal.binance_ws_scsignals", reason="ws adapter belum tersedia")


def test_safe_int_and_float_handle_none():
    assert ws.as_int(None, 7) == 7
    assert ws.as_float(None, 3.14) == 3.14
    assert ws.as_int("123") == 123
    assert ws.as_float("1.23") == 1.23


def test_parse_kline_typing_stable():
    msg = {"k": {"t": "1711111111111", "o": "1.0", "h": "2.0", "l": "0.5", "c": "1.5", "v": "10"}}
    k = ws.parse_kline(msg)
    assert isinstance(k["open_time"], int)
    for fld in ["open", "high", "low", "close", "volume"]:
        assert isinstance(k[fld], float)

