# adapters/binance_ws_scsignals.py
"""
Adapter WebSocket Binance -> SCSignals -> event queue.
- Memakai python-binance >= 1.0.20 (async).
- Emit event dict ke asyncio.Queue agar plug-and-play ke flow RajaDollar.
"""
import asyncio
from typing import Sequence, Optional
from binance import AsyncClient, BinanceSocketManager
from indicators.scsignals import SCSignals, SCConfig

def _normalize_interval(tf: str) -> str:
    # Binance interval sama dengan string Pine: "1m","5m","15m","1h","4h"
    return tf

async def run_scsignals_ws(
    api_key: str,
    api_secret: str,
    symbols: Sequence[str],
    interval: str,
    queue: asyncio.Queue,
    cfg: Optional[SCConfig] = None,
):
    client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
    bsm = BinanceSocketManager(client)
    tf = _normalize_interval(interval)

    indicators = {sym: SCSignals((cfg or SCConfig(base_tf=tf, htf="15m")) ) for sym in symbols}

    # buka beberapa socket kline sekaligus
    sockets = [bsm.kline_socket(symbol=sym, interval=tf) for sym in symbols]

    async def _consume(idx: int, sym: str):
        ind = indicators[sym]
        async with sockets[idx] as stream:
            while True:
                msg = await stream.recv()
                k = msg.get("k") or {}
                is_closed = bool(k.get("x"))
                event = ind.on_candle(
                    ts_ms=int(k.get("t")),
                    o=float(k.get("o")),
                    h=float(k.get("h")),
                    l=float(k.get("l")),
                    c=float(k.get("c")),
                    v=float(k.get("v")),
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

