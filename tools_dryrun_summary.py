#!/usr/bin/env python3
"""
Dry-run helper untuk newrealtrading.py
- Replay bar-by-bar dari CSV, 100% pakai logic real (ML + filter + trailing/BE)
- Hitung ringkasan: entries, exits, trades (complete), WinRate, ProfitFactor, avg PnL
- Export detail trades ke CSV (--out)

Contoh pakai:
python tools_dryrun_summary.py \
  --symbol ADAUSDT \
  --csv data/ADAUSDT_15m_2025-06-01_to_2025-08-09.csv \
  --coin_config coin_config.json \
  --steps 500 --balance 20 \
  --out ADA_dryrun_trades_500.csv

Tips percepat:
export USE_ML=1; export SCORE_THRESHOLD=1.2; export ML_RETRAIN_EVERY=5000
"""
from __future__ import annotations
import os, sys, time, argparse, json
import pandas as pd
import numpy as np
from optimizer.param_loader import load_params_from_csv, load_params_from_json

# pastikan bisa import modul project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import newrealtrading as nrt
except Exception:
    # fallback: jika tools/ ada di subfolder, coba parent
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import newrealtrading as nrt


def run_dry(symbol: str, csv_path: str, coin_config_path: str, steps_limit: int, balance: float) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # warmup supaya indikator & ML siap
    min_train = int(float(os.getenv("ML_MIN_TRAIN_BARS", "400")))
    warmup = max(300, min_train + 10)
    start_i = min(warmup, len(df) - 1)

    mgr = nrt.TradingManager(coin_config_path, [symbol])
    trader = mgr.traders[symbol]
    trader._log = lambda *args, **kwargs: None  # matikan log biar cepat

    # Pre-fit ML sekali di warmup (opsional, untuk speed)
    try:
        warm_df = df.iloc[: start_i + 1].copy()
        ind_warm = nrt.calculate_indicators(warm_df)
        trader.ml.fit_if_needed(ind_warm)
    except Exception:
        pass

    # Hook kumpulkan trades & PnL dummy
    trades: list[dict] = []
    trader._entry_count = 0
    trader._exit_count = 0
    _orig_enter = trader._enter_position
    _orig_exit = trader._exit_position

    # === FIX: terima **kwargs (mis. now_ts) dan teruskan ke fungsi asli ===
    def _enter_wrap(side, price, atr, balance, **kwargs):
        trader._entry_count += 1
        _orig_enter(side, price, atr, balance, **kwargs)

    def _exit_wrap(price, reason, **kwargs):
        pos = trader.pos
        # gunakan waktu yang konsisten dgn loop (kalau ada now_ts)
        exit_time = pd.to_datetime(kwargs.get("now_ts"), unit="s", utc=True) if ("now_ts" in kwargs and kwargs.get("now_ts") is not None) else pd.Timestamp.utcnow()
        if pos.side and pos.entry and pos.qty:
            pnl = (price - pos.entry) * pos.qty if pos.side == "LONG" else (pos.entry - price) * pos.qty
            trades.append({
                "symbol": symbol,
                "side": pos.side,
                "entry_price": float(pos.entry),
                "exit_price": float(price),
                "qty": float(pos.qty),
                "pnl": float(pnl),
                "reason": reason,
                "entry_time": pos.entry_time,
                "exit_time": exit_time,
            })
        trader._exit_count += 1
        _orig_exit(price, reason, **kwargs)

    trader._enter_position = _enter_wrap
    trader._exit_position = _exit_wrap

    # Replay loop
    t0 = time.time()
    steps = 0
    for i in range(start_i, min(len(df), start_i + steps_limit)):
        data_map = {symbol: df.iloc[: i + 1].copy()}
        mgr.run_once(data_map, {symbol: balance})
        steps += 1
    elapsed = time.time() - t0

    # Ringkasan
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        losses = (trades_df["pnl"] <= 0).sum()
        pf = (
            trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
            / abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())
            if losses > 0
            else np.inf
        )
        wr = (trades_df["pnl"] > 0).mean() * 100
        avg_pnl = trades_df["pnl"].mean()
    else:
        wins = losses = 0
        pf = np.nan
        wr = 0.0
        avg_pnl = 0.0

    summary = {
        "symbol": symbol,
        "rows_total": len(df),
        "warmup_index": start_i,
        "steps_executed": steps,
        "entries": trader._entry_count,
        "exits": trader._exit_count,
        "last_position": trader.pos.side,
        "trades": len(trades),
        "win_rate_pct": round(float(wr), 2),
        "profit_factor": float(pf) if pf == np.inf else (round(float(pf), 2) if not np.isnan(pf) else None),
        "avg_pnl": round(float(avg_pnl), 6),
        "elapsed_sec": round(elapsed, 2),
    }
    return summary, trades_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--balance", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="Path CSV untuk menyimpan trades")
    ap.add_argument("--use-ml", type=int, choices=[0,1], default=None)
    ap.add_argument("--ml-thr", type=float, default=None)
    ap.add_argument("--trailing-step", type=float, default=None)
    ap.add_argument("--trailing-trigger", type=float, default=None)
    ap.add_argument("--params-csv", type=str, default=None)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--min-wr", type=float, default=70.0)
    ap.add_argument("--min-pf", type=float, default=2.0)
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--prefer", type=str, default="pf_then_wr", choices=["pf_then_wr","wr_then_pf"])
    ap.add_argument("--params-json", type=str, default=None, help="Path preset JSON.")
    ap.add_argument("--preset-key", type=str, default=None, help="Key preset, mis. ADAUSDT_15m")
    args = ap.parse_args()

    # saran env untuk speed
    os.environ.setdefault("USE_ML", "1")
    os.environ.setdefault("SCORE_THRESHOLD", "1.0")
    os.environ.setdefault("ML_MIN_TRAIN_BARS", "400")
    os.environ.setdefault("ML_RETRAIN_EVERY", "5000")

    if args.use_ml is not None:
        os.environ["USE_ML"] = str(int(args.use_ml))
    if args.ml_thr is not None:
        os.environ["SCORE_THRESHOLD"] = str(float(args.ml_thr))

    overrides: dict[str, float | int | bool] = {}
    if args.params_json and args.preset_key:
        overrides = load_params_from_json(args.params_json, args.preset_key)
    elif args.params_csv:
        overrides = load_params_from_csv(
            args.params_csv,
            min_wr=args.min_wr,
            min_pf=args.min_pf,
            min_trades=args.min_trades,
            prefer=args.prefer,
            rank=args.rank,
        )

    cfg_path = args.coin_config
    if args.trailing_step is not None or args.trailing_trigger is not None or overrides:
        try:
            with open(args.coin_config, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        sym_cfg = cfg.get(args.symbol.upper(), {})
        if args.trailing_step is not None:
            sym_cfg["trailing_step"] = float(args.trailing_step)
        if args.trailing_trigger is not None:
            sym_cfg["trailing_trigger"] = float(args.trailing_trigger)
        for k, v in overrides.items():
            if k == "score_threshold":
                sym_cfg.setdefault("ml", {})["score_threshold"] = float(v)
            else:
                sym_cfg[k] = v
        cfg[args.symbol.upper()] = sym_cfg
        tmp_cfg_path = f"_tmp_{args.symbol.upper()}_cfg.json"
        with open(tmp_cfg_path, "w") as f:
            json.dump(cfg, f)
        cfg_path = tmp_cfg_path
        if "score_threshold" in overrides:
            os.environ["SCORE_THRESHOLD"] = str(float(overrides["score_threshold"]))

    summary, trades_df = run_dry(args.symbol.upper(), args.csv, cfg_path, args.steps, args.balance)

    if overrides:
        print("\n=== PARAMETER DIPAKAI ===")
        for k, v in overrides.items():
            print(f"{k}: {v}")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out and not trades_df.empty:
        trades_df.to_csv(args.out, index=False)
        print(f"\nTrades saved to: {args.out}")

if __name__ == "__main__":
    main()
