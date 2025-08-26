"""
Contoh minimal pemakaian SMCIndicator secara incremental.
Untuk streaming Binance, sesuaikan dengan infrastruktur websocket Anda (python-binance/ccxtpro/taapi dll).
Di sini diberikan contoh loop pseudo (offline) agar mudah diadaptasi.

pip install pandas numpy

"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from smc import SMCIndicator, SMCConfig
from adapters.rajadollar_adapter import RajaDollarSMCAdapter
import asyncio

async def demo_offline():
    # Buat data OHLCV dummy 1m
    ts = pd.date_range(datetime.utcnow()-timedelta(minutes=400), periods=400, freq='1min')
    price = np.cumsum(np.random.randn(len(ts))*0.1) + 100
    df = pd.DataFrame({
        'open': price,
        'high': price + np.random.rand(len(ts))*0.3,
        'low':  price - np.random.rand(len(ts))*0.3,
        'close': price + np.random.randn(len(ts))*0.02,
        'volume': np.random.randint(50, 500, size=len(ts))
    }, index=ts)

    smc = SMCIndicator(symbol="BTCUSDT", timeframe="1m", config=SMCConfig(
        swing_len=50, internal_len=5, show_daily=True, show_weekly=True, show_monthly=False
    ))

    evs = smc.process_dataframe(df)
    q = asyncio.Queue()
    adapter = RajaDollarSMCAdapter(queue=q)

    await adapter.emit_many(evs, bot_id="RajaDollar", run_id="test-001")

    # Konsumsi event dari queue
    items = []
    while not q.empty():
        items.append(await q.get())
    print(f"Jumlah event: {len(items)} | Contoh 3 event pertama:")
    for m in items[:3]:
        print(m)

if __name__ == "__main__":
    asyncio.run(demo_offline())
