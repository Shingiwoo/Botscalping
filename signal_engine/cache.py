from __future__ import annotations
from typing import Any, Dict, Tuple

class FeatureCache:
    """Cache sederhana per indeks bar."""
    def __init__(self) -> None:
        self._store: Dict[Tuple[str, int], Any] = {}

    def get(self, key: str, i: int):
        return self._store.get((key, i))

    def set(self, key: str, i: int, val: Any):
        self._store[(key, i)] = val
