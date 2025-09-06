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
from typing import Any, Tuple, Dict, List
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


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def simulate_dryrun(df: pd.DataFrame, symbol: str, coin_config_path: str, steps_limit: int, balance: float) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = _normalize_df(df.copy())

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

    # Hook kumpulkan trades & PnL dummy (hindari atribut baru pada trader untuk Pylance)
    trades: list[dict[str, Any]] = []
    entry_count = 0
    exit_count = 0
    _orig_enter = trader._enter_position
    _orig_exit = trader._exit_position

    # Gunakan *args/**kwargs supaya kompatibel dg signature asli apa pun, dan return value diteruskan
    def _enter_wrap(*args: Any, **kwargs: Any):
        nonlocal entry_count
        entry_count += 1
        return _orig_enter(*args, **kwargs)

    def _safe_to_datetime_from_seconds(ts: Any):
        if ts is None:
            return pd.Timestamp.utcnow()
        try:
            # int/float-like seconds
            if isinstance(ts, (int, float, np.integer, np.floating)):
                return pd.to_datetime(int(ts), unit="s", utc=True)
            # string timestamp; let pandas parse
            if isinstance(ts, str):
                dt = pd.to_datetime(ts, utc=True, errors="coerce")
                return dt if pd.notna(dt) else pd.Timestamp.utcnow()
        except Exception:
            pass
        return pd.Timestamp.utcnow()

    def _exit_wrap(*args: Any, **kwargs: Any):
        nonlocal exit_count
        pos = trader.pos
        # ambil price dan reason dari args/kwargs jika ada
        price = kwargs.get("price") if "price" in kwargs else (args[0] if len(args) >= 1 else None)
        reason = kwargs.get("reason") if "reason" in kwargs else (args[1] if len(args) >= 2 else "UNKNOWN")
        # gunakan waktu yang konsisten dgn loop (kalau ada now_ts)
        exit_time = _safe_to_datetime_from_seconds(kwargs.get("now_ts"))
        try:
            if price is not None and pos.side and pos.entry and pos.qty:
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
        finally:
            exit_count += 1
        return _orig_exit(*args, **kwargs)

    trader._enter_position = _enter_wrap  # type: ignore
    trader._exit_position = _exit_wrap    # type: ignore

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
        "entries": entry_count,
        "exits": exit_count,
        "last_position": trader.pos.side,
        "trades": len(trades),
        "win_rate_pct": round(float(wr), 2),
        "profit_factor": float(pf) if pf == np.inf else (round(float(pf), 2) if not np.isnan(pf) else None),
        "avg_pnl": round(float(avg_pnl), 6),
        "elapsed_sec": round(elapsed, 2),
    }
    # Diagnostik: jika tidak ada trade sama sekali, tampilkan ringkas indikator bar terakhir
    try:
        if summary.get("trades", 0) == 0:
            # Ambil index terakhir yang diproses
            last_idx = start_i + steps - 1
            last_idx = min(max(last_idx, 0), len(df) - 1)
            from engine_core import ensure_base_indicators  # lazy import to avoid circulars
            df_proc = df.iloc[: last_idx + 1].copy()
            ind_df = nrt.calculate_indicators(df_proc)
            ind_df = ensure_base_indicators(ind_df, trader.config)
            last = ind_df.iloc[-1]
            ema_len = int(trader.config.get("ema_len", 22))
            sma_len = int(trader.config.get("sma_len", 20))
            print(
                f"[{symbol}] LAST IND summary: close={last.get('close')}, rsi={last.get('rsi')}, "
                f"ema={last.get(f'ema_{ema_len}')}, sma={last.get(f'sma_{sma_len}')}, "
                f"macd={last.get('macd')}/{last.get('macd_signal')}"
            )
    except Exception as e:
        print(f"[WARN] print last IND failed: {e}")
    return summary, trades_df

def run_dry(symbol: str, csv_path: str, coin_config_path: str, steps_limit: int, balance: float) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    return simulate_dryrun(df, symbol, coin_config_path, steps_limit, balance)


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
    ap.add_argument("--profile", default=None, help="Pilih profil coin_config.[SYMBOL].profiles.[name] untuk dioverlay (mis. aggressive)")
    # Tambahan baru: OOS split, Monte Carlo, dan debug config
    ap.add_argument("--oos-split", type=str, default=None, help="Tanggal pemisah OOS (YYYY-MM-DD)")
    ap.add_argument("--mc-runs", type=int, default=0, help="Jumlah Monte-Carlo block bootstrap runs (0=off)")
    ap.add_argument("--mc-block", type=int, default=40, help="Ukuran blok bootstrap")
    ap.add_argument("--debug-cfg", action="store_true", help="Tampilkan konfigurasi efektif yang dipakai engine")
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
        # Inject aggregator preset (if present in params_json under preset_key)
        try:
            if args.params_json and args.preset_key:
                with open(args.params_json, "r") as f:
                    preset_root = json.load(f)
                preset_root = preset_root.get(args.preset_key, preset_root)
                agg_keys = {
                    "signal_weights","strength_thresholds","regime_bounds","weight_scale",
                    "sr_penalty","sd_tol_pct","vol_lookback","vol_z_thr","score_gate",
                    "htf_rules","htf_fallback_discount","weight_scale_nl","min_confirms",
                    # key baru untuk gate adaptif & bonus konfirmasi
                    "score_gate_no_confirms","min_strength","min_strength_no_confirms",
                    "no_confirms_require","confirm_bonus_per","confirm_bonus_max"
                }
                agg_block = {k: preset_root[k] for k in agg_keys if k in preset_root}
                if agg_block:
                    sym_cfg["_agg"] = agg_block
                    print(f"[{args.symbol.upper()}] Aggregator preset injected: keys={list(agg_block.keys())}")
        except Exception as e:
            print(f"[{args.symbol.upper()}] WARN cannot inject aggregator preset: {e}")
        # Overlay profil kalau diminta, timpa subset kunci di level atas
        try:
            prof_name = (args.profile or "").strip().lower()
            if prof_name and isinstance(sym_cfg.get("profiles"), dict):
                prof = sym_cfg["profiles"].get(prof_name)
                if isinstance(prof, dict):
                    for k, v in prof.items():
                        if k in (
                            "filters","min_atr_pct","max_atr_pct","max_body_atr",
                            "sl_atr_mult","sl_pct","be_trigger_pct",
                            "trailing_trigger","trailing_step",
                            "use_sr_filter","sr_near_pct","score_gate","use_htf_filter",
                        ):
                            sym_cfg[k] = v
                    print(f"[{args.symbol.upper()}] Profile overlay applied: {prof_name}")
        except Exception as e:
            print(f"[{args.symbol.upper()}] WARN cannot apply profile overlay: {e}")
        cfg[args.symbol.upper()] = sym_cfg
        tmp_cfg_path = f"_tmp_{args.symbol.upper()}_cfg.json"
        with open(tmp_cfg_path, "w") as f:
            json.dump(cfg, f)
        cfg_path = tmp_cfg_path
        if "score_threshold" in overrides:
            os.environ["SCORE_THRESHOLD"] = str(float(overrides["score_threshold"]))

    # Terapkan overlay profil walau tanpa overrides lainnya
    if args.profile:
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        sym = args.symbol.upper()
        sym_cfg = cfg.get(sym, {})
        prof_name = (args.profile or "").strip().lower()
        if prof_name and isinstance(sym_cfg.get("profiles"), dict):
            prof = sym_cfg["profiles"].get(prof_name)
            if isinstance(prof, dict):
                for k, v in prof.items():
                    if k in (
                        "filters","min_atr_pct","max_atr_pct","max_body_atr",
                        "sl_atr_mult","sl_pct","be_trigger_pct",
                        "trailing_trigger","trailing_step",
                        "use_sr_filter","sr_near_pct","score_gate","use_htf_filter",
                    ):
                        sym_cfg[k] = v
                cfg[sym] = sym_cfg
                tmp_cfg_path = f"_tmp_{sym}_cfg.json"
                with open(tmp_cfg_path, "w") as f:
                    json.dump(cfg, f)
                cfg_path = tmp_cfg_path
                print(f"[{sym}] Profile overlay applied: {prof_name}")

    # Debug config efektif
    if args.debug_cfg:
        try:
            with open(cfg_path, "r") as f:
                eff = json.load(f)
        except Exception:
            eff = {}
        print("\n=== DEBUG CONFIG (effective) ===")
        try:
            print(json.dumps(eff, indent=2))
        except Exception:
            print(eff)

    # Helper untuk cetak summary ringkas
    def print_summary(title: str, summary: Dict[str, Any]):
        print(f"\n=== {title} ===")
        keys = [
            "symbol", "rows_total", "warmup_index", "steps_executed",
            "trades", "win_rate_pct", "profit_factor", "avg_pnl", "elapsed_sec",
        ]
        for k in keys:
            if k in summary:
                print(f"{k}: {summary[k]}")

    # MC utils
    def block_bootstrap(seq: List[float], block: int = 40, runs: int = 100) -> List[List[float]]:
        import random
        n = len(seq)
        out: List[List[float]] = []
        if n == 0 or runs <= 0 or block <= 0:
            return out
        for _ in range(runs):
            i, bag = 0, []
            while i < n:
                j = random.randint(0, max(0, n - block))
                bag.extend(seq[j:j + block])
                i += block
            out.append(bag[:n])
        return out

    def summarize_mc(samples: List[List[float]]) -> Dict[str, Any]:
        if not samples:
            return {}
        wrs, pnls = [], []
        for s in samples:
            if not s:
                continue
            wrs.append(sum(1 for x in s if x > 0) / len(s))
            pnls.append(sum(s))
        if not wrs:
            return {}
        return {
            "wr_p5_p50_p95": list(np.percentile(wrs, [5, 50, 95])),
            "pnl_p5_p50_p95": list(np.percentile(pnls, [5, 50, 95])),
        }

    # Load data sekali untuk reuse pada OOS/MC
    df_full = pd.read_csv(args.csv)

    # 1) Baseline (FULL)
    full_summary, full_trades = simulate_dryrun(df_full, args.symbol.upper(), cfg_path, args.steps, args.balance)
    if overrides:
        print("\n=== PARAMETER DIPAKAI ===")
        for k, v in overrides.items():
            print(f"{k}: {v}")
    print_summary("SUMMARY (FULL)", full_summary)

    # 2) OOS split opsional
    did_oos = False
    if args.oos_split:
        try:
            split_dt = pd.Timestamp(args.oos_split)
            df_full = _normalize_df(df_full)
            df_in = df_full[df_full["timestamp"] < split_dt]
            df_oos = df_full[df_full["timestamp"] >= split_dt]
            if len(df_in) > 0:
                ins_summary, _ = simulate_dryrun(df_in, args.symbol.upper(), cfg_path, args.steps, args.balance)
                print_summary("IN-SAMPLE", ins_summary)
            if len(df_oos) > 0:
                did_oos = True
                oos_summary, oos_trades = simulate_dryrun(df_oos, args.symbol.upper(), cfg_path, args.steps, args.balance)
                print_summary("OUT-OF-SAMPLE", oos_summary)
                if args.mc_runs > 0:
                    seq = [float(x) for x in list(oos_trades.get("pnl", pd.Series([], dtype=float)))]
                    bags = block_bootstrap(seq, block=args.mc_block, runs=args.mc_runs)
                    mc = summarize_mc(bags)
                    if mc:
                        print(f"[MC-OOS] WR% p5/50/95 = {mc['wr_p5_p50_p95']}, PnL p5/50/95 = {mc['pnl_p5_p50_p95']}")
        except Exception as e:
            print(f"[WARN] OOS split gagal diproses: {e}")

    # 3) MC atas FULL bila tidak OOS
    if not did_oos and args.mc_runs > 0:
        seq = [float(x) for x in list(full_trades.get("pnl", pd.Series([], dtype=float)))]
        bags = block_bootstrap(seq, block=args.mc_block, runs=args.mc_runs)
        mc = summarize_mc(bags)
        if mc:
            print(f"[MC] WR% p5/50/95 = {mc['wr_p5_p50_p95']}, PnL p5/50/95 = {mc['pnl_p5_p50_p95']}")

    # Export trades baseline jika diminta
    if args.out and isinstance(full_trades, pd.DataFrame) and not full_trades.empty:
        full_trades.to_csv(args.out, index=False)
        print(f"\nTrades saved to: {args.out}")

if __name__ == "__main__":
    main()
