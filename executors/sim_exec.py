from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SimPosition:
    side: str
    entry: float
    qty: float
    sl: Optional[float] = None
    tp: Optional[float] = None


class SimExecutor:
    """
    Eksekutor sederhana untuk DRYRUN:
    - Menyimpan balance per symbol.
    - Mencatat open/close posisi (tanpa order real).
    - Tidak punya jaringan / API eksternal.
    """

    def __init__(self, balances: Dict[str, float]):
        self.balances = dict(balances)  # {symbol: balance}
        self.positions: Dict[str, Optional[SimPosition]] = {}

    def get_balance(self, symbol: str) -> float:
        return float(self.balances.get(symbol, 0.0))

    def set_balance(self, symbol: str, value: float) -> None:
        self.balances[symbol] = float(value)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions and self.positions[symbol] is not None

    def open_position(
        self,
        symbol: str,
        side: str,
        entry: float,
        qty: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> None:
        self.positions[symbol] = SimPosition(side=side, entry=entry, qty=qty, sl=sl, tp=tp)

    def close_position(self, symbol: str, exit_price: float) -> float:
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        pnl = (exit_price - pos.entry) * pos.qty * (1 if pos.side == "LONG" else -1)
        self.positions[symbol] = None
        return float(pnl)

