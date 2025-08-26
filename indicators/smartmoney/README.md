# SMC Indicator (Python) — Modular

Paket ini mengimplementasikan indikator **Smart Money Concepts** (SMC) versi Python,
terinspirasi dari *Smart Money Concepts [LuxAlgo]* (CC BY-NC-SA 4.0).

## Fitur
- Struktur Internal & Swing (BOS / CHoCH)
- Order Blocks (internal & swing) + mitigasi
- Equal Highs/Lows (via pivot + ATR threshold; disederhanakan)
- Fair Value Gaps (FVG) 3-candle
- MTF Levels (prev D/W/M high-low)
- Premium/Discount/Equilibrium Zones (berdasarkan trailing extremes)
- Adapter ke arsitektur RajaDollar (queue/event)

## Instalasi
```bash
pip install pandas numpy
```

## Cara Pakai Singkat
```python
from smc import SMCIndicator, SMCConfig
ind = SMCIndicator("BTCUSDT","1m", SMCConfig(show_daily=True, show_weekly=True))
events = ind.process_dataframe(df)  # df: DataFrame OHLCV indexed by datetime
```

## Adapter RajaDollar
Lihat `adapters/rajadollar_adapter.py` untuk mengalirkan event ke asyncio.Queue.

## Catatan
- Implementasi ini bukan 1:1 replika PineScript; beberapa heuristik disederhanakan
  agar mudah dipakai di bot Python & real-time streaming.
- Hormati lisensi LuxAlgo (CC BY-NC-SA 4.0) untuk penggunaan non-komersial.
