# adapters/binance_ws_scsignals.py
"""
Adapter WebSocket Binance -> SCSignals -> antrean event.
- Memakai python-binance >= 1.0.20 (async).
- Mengirim dict event ke asyncio.Queue agar mudah dirangkai dengan flow RajaDollar.
"""
import asyncio
from typing import Sequence, Optional, Any, Dict
from binance import AsyncClient, BinanceSocketManager
from indicators.scsignal import SCSignals, SCConfig


def as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_kline(msg: Dict[str, Any]) -> Dict[str, float]:
    """Konversi payload WS kline Binance ke tipe numerik stabil."""
    k = msg.get("k") or msg.get("data", {}).get("k") or {}
    return {
        "open_time": as_int(k.get("t")),
        "open": as_float(k.get("o")),
        "high": as_float(k.get("h")),
        "low": as_float(k.get("l")),
        "close": as_float(k.get("c")),
        "volume": as_float(k.get("v")),
    }

def _normalize_interval(tf: str) -> str:
    # Binance interval sama dengan string Pine: "1m","5m","15m","1h","4h"
    return tf

def _merge_cfg(default_cfg: Optional[dict], override: Optional[dict]) -> SCConfig:
    base = SCConfig.from_dict(default_cfg or {})
    if override:
        merged = {**base.__dict__, **override}
        return SCConfig.from_dict(merged)
    return base

async def run_scsignals_ws(
    api_key: str,
    api_secret: str,
    symbols: Sequence[str],
    interval: str,
    queue: asyncio.Queue,
    cfg: Optional[SCConfig] = None,
    cfg_by_symbol: Optional[Dict[str, Any]] = None,
    default_cfg: Optional[Dict[str, Any]] = None,
):
    client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
    bsm = BinanceSocketManager(client)
    tf = _normalize_interval(interval)

    indicators = {}
    for sym in symbols:
        if cfg is not None:
            final_cfg = cfg
        else:
            ov = (cfg_by_symbol or {}).get(sym)
            merged = _merge_cfg(default_cfg or {"base_tf": tf, "htf": "15m"}, ov)
            final_cfg = merged
        indicators[sym] = SCSignals(final_cfg)

    # buka beberapa socket kline sekaligus
    sockets = [bsm.kline_socket(symbol=sym, interval=tf) for sym in symbols]

    async def _consume(idx: int, sym: str):
        ind = indicators[sym]
        async with sockets[idx] as stream:
            while True:
                msg = await stream.recv()
                parsed = parse_kline(msg)
                k = msg.get("k") or {}
                is_closed = bool(k.get("x"))
                event = ind.on_candle(
                    ts_ms=parsed["open_time"],
                    o=parsed["open"],
                    h=parsed["high"],
                    l=parsed["low"],
                    c=parsed["close"],
                    v=parsed["volume"],
                    is_closed=is_closed,
                    symbol=sym,
                )
                if event:
                    await queue.put(event)

    tasks = [asyncio.create_task(_consume(i, s)) for i, s in enumerate(symbols)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await client.close_connection()

