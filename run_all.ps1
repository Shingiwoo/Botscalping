param(
  [string]$Symbol = $env:SYMBOL,
  [string]$TF = "15m",
  [string]$CSV = "",
  [int]$Steps = 1500,
  [int]$Balance = 50,
  [int]$UseML = 0
)

$env:BOT_SEED = $env:SEED
$env:DEBUG_REASONS = "1"
$env:REASON_EVERY_N = "25"

python tools_apply_preset_profiles.py --coin-config coin_config.json --params presets/scalping_params.json --preset AGGRESSIVE_5m --out coin_config_agg_5m.json --force
python tools_apply_preset_profiles.py --coin-config coin_config.json --params presets/scalping_params.json --preset AGGRESSIVE_15m --out coin_config_agg_15m.json --force

$cfg = "coin_config_agg_15m.json"
if ($TF -eq "5m") { $cfg = "coin_config_agg_5m.json" }
if ($CSV -eq "") { $CSV = if ($TF -eq "5m") { $env:CSV_5M } else { $env:CSV_15M } }

python tools/audit_data.py --symbols all --tf $TF --glob "data/*_${TF}_*.csv" --coin-config $cfg
python tools_dryrun_summary.py --symbol $Symbol --csv $CSV --coin_config $cfg --params-json presets/scalping_params.json --preset-key ("AGGRESSIVE_"+$TF) --profile aggressive --steps $Steps --balance $Balance --use-ml $UseML --debug-cfg
python analysis/agg_hist.py --symbol $Symbol
python analysis/reject_matrix.py --tf $TF
python experiments/ab_gates.py --symbol $Symbol --csv $CSV --coin-config $cfg --steps $Steps --balance $Balance
python experiments/ab_exit.py --symbol $Symbol --csv $CSV --coin-config $cfg --params-json presets/scalping_params.json --preset-a ("AGGRESSIVE_"+$TF) --preset-b ("AGGRESSIVE_"+($(if($TF -eq "5m"){"15m"}else{"5m"})))
python experiments/run_matrix.py --symbols $env:SYMBOLS --tf $TF --coin-config $cfg --steps $Steps --balance $Balance --use-ml $UseML
python experiments/walkforward.py --symbol $Symbol --csv $CSV --coin-config $cfg --window ($(if($TF -eq "5m"){1500}else{500})) --steps $Steps --balance $Balance --tf $TF
Write-Host "Done. Cek reports/ dan logs/"

