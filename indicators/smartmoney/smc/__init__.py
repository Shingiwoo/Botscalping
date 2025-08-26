"""
SMC (Smart Money Concepts) Indicator Package

Fitur utama yang diadopsi dari Pine Script "Smart Money Concepts [LuxAlgo]" (CC BY-NC-SA 4.0):
- Internal & Swing Structure (BOS / CHoCH)
- Order Blocks (internal & swing) + mitigasi
- Equal Highs / Equal Lows
- Fair Value Gaps (FVG)
- MTF Levels (Daily/Weekly/Monthly prev high/low)
- Premium / Discount Zones

Catatan lisensi:
Script referensi berlisensi CC BY-NC-SA 4.0 © LuxAlgo.
Implementasi Python ini adalah interpretasi ulang untuk keperluan edukasi & riset.
Hindari penggunaan komersial yang melanggar lisensi aslinya.
"""

from .indicator import SMCIndicator, SMCConfig, SMCEvent
from .types import EventType, StructureBias, OrderBlock, FairValueGap, LevelLines, ZonesState
