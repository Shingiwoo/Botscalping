from __future__ import annotations
import csv
import os
from typing import Dict, Any, List, Mapping
from datetime import datetime, timezone


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _should_log(step: int, every_n: int) -> bool:
    if every_n <= 1:
        return True
    return (step % every_n) == 0


def log_reject(symbol: str,
               step: int,
               score: float,
               strength: str | None,
               reasons: Mapping[str, Any] | None,
               extras: List[Any] | None = None) -> None:
    """
    Append a compact CSV row describing a reject decision to logs/rejects_{symbol}.csv
    Respects env DEBUG_REASONS=1 and REASON_EVERY_N for sampling.
    """
    if os.getenv("DEBUG_REASONS", "0").strip() != "1":
        return
    every_n = int(os.getenv("REASON_EVERY_N", "20"))
    if not _should_log(int(step), every_n):
        return

    _ensure_dir("logs")
    path = os.path.join("logs", f"rejects_{symbol.upper()}.csv")
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "step": int(step),
        "score": float(score),
        "strength": (strength or "")
    }
    # Flatten boolean flags in reasons (only simple keys)
    if isinstance(reasons, Mapping):
        for k, v in reasons.items():
            if isinstance(v, bool):
                row[str(k)] = int(v)
            else:
                row[str(k)] = v
    # Extras as a compact string
    if extras:
        try:
            row["extras"] = ";".join(map(str, extras))
        except Exception:
            row["extras"] = str(extras)

    header = sorted(row.keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)
