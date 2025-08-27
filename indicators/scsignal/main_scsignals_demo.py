# Contoh pemakaian cepat (RajaDollar event/queue)
# main_scsignals_demo.py
import asyncio
from indicators.scsignal.binance_ws_scsignals import run_scsignals_ws
from indicators.scsignal import SCConfig

API_KEY = "BINANCE_API_KEY"
API_SECRET = "BINANCE_API_SECRET"

async def consumer(q: asyncio.Queue):
    while True:
        evt = await q.get()
        # Integrasi ke engine RajaDollar:
        #   - mapping ke Event(type="signal", strategy="SCSignals", side=evt["side"], ...)
        print("[SIGNAL]", evt["symbol"], evt["side"], "price=", evt["price"])
        q.task_done()

async def main():
    q = asyncio.Queue(maxsize=1000)
    # konfigurasi default untuk semua simbol
    default_cfg = dict(
        length=20, sma_period=15, atr_len=14, atr_mult=0.6,
        use_htf=True, htf="1m", ema_fast_len=20, ema_slow_len=60,
        use_adx=True, adx_len=16, min_adx=18.0,
        use_body_atr=True, min_body_atr=0.38,
        use_width_atr=True, min_width_atr=1.20,
        use_rsi=False, rsi_len=14, rsi_buy=52.0, rsi_sell=48.0,
        cooldown_bars=5, base_tf="1m",
    )
    # override per-simbol (hanya kunci yang berbeda dari default)
    cfg_by_symbol = {
        "DOGEUSDT": {"length": 22, "atr_mult": 0.7, "htf": "5m"},
        "XRPUSDT": {"length": 18, "min_body_atr": 0.30},
    }
    symbols = ["DOGEUSDT", "XRPUSDT"]
    producer = run_scsignals_ws(
        API_KEY, API_SECRET, symbols, interval="1m", queue=q,
        cfg=None, cfg_by_symbol=cfg_by_symbol, default_cfg=default_cfg,
    )
    await asyncio.gather(consumer(q), producer)

if __name__ == "__main__":
    asyncio.run(main())

