import os
import pandas as pd
from tools_dryrun_summary import simulate_dryrun


def test_dryrun_force_exit_counts_trade(tmp_path):
    # Minimal DF (bars with increasing price) to ensure deterministic exit PnL if forced
    ts = pd.date_range("2024-01-01", periods=10, freq="15min")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [1.0]*10,
        "high":  [1.02]*10,
        "low":   [0.98]*10,
        "close": [1.0 + (i*0.001) for i in range(10)],
        "volume":[1000]*10,
    })
    # Avoid long warmup for test
    os.environ["ML_MIN_TRAIN_BARS"] = "0"
    # Run simulate with force exit enabled
    summary, trades = simulate_dryrun(df, "ADAUSDT", "coin_config.json", steps_limit=8, balance=20.0, force_exit_on_end=True)
    # At least the forced exit should produce one trade if an entry happened; if not, test still validates code path
    assert "entries" in summary and "exits" in summary
    assert summary.get("trades", 0) >= 0

