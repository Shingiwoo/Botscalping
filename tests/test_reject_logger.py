import os
import csv
from logger.reject_logger import log_reject


def test_reject_logger_writes_csv(tmp_path, monkeypatch):
    # ensure logs dir in tmp
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEBUG_REASONS", "1")
    monkeypatch.setenv("REASON_EVERY_N", "1")
    sym = "ADAUSDT"
    log_reject(sym, step=10, score=0.75, strength="kuat", reasons={"htf_fail": True, "atr_fail": False}, extras=["used_gate=0.70"])
    path = tmp_path / "logs" / f"rejects_{sym}.csv"
    assert path.exists(), "reject log file should be created"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    r = rows[-1]
    assert r["symbol"] == sym
    assert r.get("htf_fail") in ("1", 1, 1.0)

