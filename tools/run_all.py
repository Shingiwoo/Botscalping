import os, argparse, subprocess, sys, shlex, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

def run(cmd, env=None):
    print(f"\n$ {cmd}")
    res = subprocess.run(cmd if isinstance(cmd, list) else shlex.split(cmd), env=env)
    if res.returncode != 0:
        sys.exit(res.returncode)

def ensure_dirs():
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=os.getenv("SYMBOL","ADAUSDT"))
    ap.add_argument("--tf", choices=["5m","15m"], default=os.getenv("TF","15m"))
    ap.add_argument("--balance", type=float, default=float(os.getenv("BALANCE","50")))
    ap.add_argument("--steps", type=int, default=int(os.getenv("STEPS_15M","1500")))
    ap.add_argument("--seed", default=os.getenv("SEED","123"))
    ap.add_argument("--profile", default=os.getenv("PROFILE","aggressive"))
    ap.add_argument("--coin-config-in", default=os.getenv("COINCFG_IN","coin_config.json"))
    ap.add_argument("--presets", default=os.getenv("PRESETS","presets/scalping_params.json"))
    ap.add_argument("--preset-5m", default=os.getenv("PRESET_5M","AGGRESSIVE_5m"))
    ap.add_argument("--preset-15m", default=os.getenv("PRESET_15M","AGGRESSIVE_15m"))
    ap.add_argument("--coin-config-out-5m", default=os.getenv("COINCFG_OUT_5M","coin_config_agg_5m.json"))
    ap.add_argument("--coin-config-out-15m", default=os.getenv("COINCFG_OUT_15M","coin_config_agg_15m.json"))
    ap.add_argument("--csv", required=False, default=os.getenv("CSV_15M",""))
    ap.add_argument("--use-ml", type=int, default=int(os.getenv("USE_ML","0")))
    ap.add_argument("--debug-reasons", type=int, default=int(os.getenv("DEBUG_REASONS","1")))
    ap.add_argument("--reason-every-n", type=int, default=int(os.getenv("REASON_EVERY_N","25")))
    args = ap.parse_args()

    ensure_dirs()
    env = os.environ.copy()
    env["BOT_SEED"] = str(args.seed)
    env["DEBUG_REASONS"] = str(args.debug_reasons)
    env["REASON_EVERY_N"] = str(args.reason_every_n)

    # 1) apply presets
    run(f'{PY} tools_apply_preset_profiles.py --coin-config {args.coin_config_in} --params {args.presets} --preset {args.preset_5m} --out {args.coin_config_out_5m} --force', env)
    run(f'{PY} tools_apply_preset_profiles.py --coin-config {args.coin_config_in} --params {args.presets} --preset {args.preset_15m} --out {args.coin_config_out_15m} --force', env)

    # 2) audit (sesuaikan tf)
    tf = args.tf
    cfg_out = args.coin_config_out_15m if tf=="15m" else args.coin_config_out_5m
    glob = f'data/*_{tf}_*.csv'
    run(f'{PY} tools/audit_data.py --symbols all --tf {tf} --glob "{glob}" --coin-config {cfg_out}', env)

    # 3) baseline dryrun
    csv = args.csv or (os.getenv("CSV_15M") if tf=="15m" else os.getenv("CSV_5M"))
    preset = args.preset_15m if tf=="15m" else args.preset_5m
    run(f'{PY} tools_dryrun_summary.py --symbol {args.symbol} --csv {csv} --coin_config {cfg_out} --params-json {args.presets} --preset-key {preset} --profile {args.profile} --steps {args.steps} --balance {args.balance} --use-ml {args.use_ml} --debug-cfg', env)

    # 4) telemetry
    run(f'{PY} analysis/agg_hist.py --symbol {args.symbol}', env)
    run(f'{PY} analysis/reject_matrix.py --tf {tf}', env)

    # 5) AB gates
    run(f'{PY} experiments/ab_gates.py --symbol {args.symbol} --csv {csv} --coin-config {cfg_out} --steps {args.steps} --balance {args.balance}', env)

    # 6) AB exit (bandingkan dua preset)
    other = args.preset_5m if tf=="15m" else args.preset_15m
    run(f'{PY} experiments/ab_exit.py --symbol {args.symbol} --csv {csv} --coin-config {cfg_out} --params-json {args.presets} --preset-a {preset} --preset-b {other}', env)

    # 7) Matrix multi-coin
    symbols = os.getenv("SYMBOLS","BTCUSDT,ETHUSDT,ADAUSDT,XRPUSDT")
    run(f'{PY} experiments/run_matrix.py --symbols {symbols} --tf {tf} --coin-config {cfg_out} --steps {args.steps} --balance {args.balance} --use-ml {args.use_ml}', env)

    # 8) Walk-forward
    window = 500 if tf=="15m" else 1500
    run(f'{PY} experiments/walkforward.py --symbol {args.symbol} --csv {csv} --coin-config {cfg_out} --window {window} --steps {args.steps} --balance {args.balance} --tf {tf}', env)

    print("\n✅ Selesai. Cek folder reports/ dan logs/ untuk hasil lengkap.")

if __name__ == "__main__":
    main()

