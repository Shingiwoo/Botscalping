-include .env

PY ?= python
SYMBOL ?= $(SYMBOL)
SYMBOLS ?= $(SYMBOLS)
TF ?= $(TF)
BAL ?= $(BALANCE)
SEED ?= $(SEED)
PROFILE ?= $(PROFILE)

COINCFG_IN ?= $(COINCFG_IN)
PRESETS ?= $(PRESETS)
PRESET_5M ?= $(PRESET_5M)
PRESET_15M ?= $(PRESET_15M)
COINCFG_OUT_5M ?= $(COINCFG_OUT_5M)
COINCFG_OUT_15M ?= $(COINCFG_OUT_15M)

CSV_15M ?= $(CSV_15M)
CSV_5M ?= $(CSV_5M)
STEPS_5M ?= $(STEPS_5M)
STEPS_15M ?= $(STEPS_15M)

export BOT_SEED=$(SEED)
export DEBUG_REASONS?=$(DEBUG_REASONS)
export REASON_EVERY_N?=$(REASON_EVERY_N)

.PHONY: help setup apply-presets audit-5m audit-15m baseline-5m baseline-15m \
        telemetry-5m telemetry-15m ab-gates-5m ab-gates-15m ab-exit-5m ab-exit-15m \
        ab-ml-15m run-matrix-5m run-matrix-15m walkforward-15m walkforward-5m \
        mc trade-metrics all-15m all-5m clean-reports

help:
	@echo "== Botscalping Makefile =="
	@echo "make apply-presets        # terapkan preset ke semua coin (5m & 15m)"
	@echo "make audit-15m            # audit data & warning config (15m)"
	@echo "make baseline-15m         # dryrun baseline (15m, tanpa ML)"
	@echo "make telemetry-15m        # histogram skor + reject matrix (15m)"
	@echo "make ab-gates-15m         # A/B gate kecil (15m)"
	@echo "make ab-exit-15m          # A/B exit (15m)"
	@echo "make run-matrix-15m       # multi-coin summary (15m)"
	@echo "make walkforward-15m      # walk-forward (15m)"
	@echo "make all-15m              # jalankan semua tahap 15m berurutan"
	@echo "Variabel env di .env atau override: SYMBOL, SYMBOLS, CSV_15M, CSV_5M, BALANCE, STEPS_15M, STEPS_5M"

setup:
	@mkdir -p reports logs

apply-presets: setup
	$(PY) tools_apply_preset_profiles.py \
		--coin-config $(COINCFG_IN) \
		--params $(PRESETS) \
		--preset $(PRESET_5M) \
		--out $(COINCFG_OUT_5M) --force
	$(PY) tools_apply_preset_profiles.py \
		--coin-config $(COINCFG_IN) \
		--params $(PRESETS) \
		--preset $(PRESET_15M) \
		--out $(COINCFG_OUT_15M) --force

audit-15m: setup
	$(PY) tools/audit_data.py --symbols all --tf 15m --glob "data/*_15m_*.csv" --coin-config $(COINCFG_OUT_15M)

audit-5m: setup
	$(PY) tools/audit_data.py --symbols all --tf 5m --glob "data/*_5m_*.csv" --coin-config $(COINCFG_OUT_5M)

baseline-15m: setup
	$(PY) tools_dryrun_summary.py \
		--symbol $(SYMBOL) \
		--csv $(CSV_15M) \
		--coin_config $(COINCFG_OUT_15M) \
		--params-json $(PRESETS) --preset-key $(PRESET_15M) \
		--profile $(PROFILE) \
		--steps $(STEPS_15M) --balance $(BAL) --use-ml 0 --debug-cfg

baseline-5m: setup
	$(PY) tools_dryrun_summary.py \
		--symbol $(SYMBOL) \
		--csv $(CSV_5M) \
		--coin_config $(COINCFG_OUT_5M) \
		--params-json $(PRESETS) --preset-key $(PRESET_5M) \
		--profile $(PROFILE) \
		--steps $(STEPS_5M) --balance $(BAL) --use-ml 0 --debug-cfg

telemetry-15m: setup
	$(PY) analysis/agg_hist.py --symbol $(SYMBOL)
	$(PY) analysis/reject_matrix.py --tf 15m

telemetry-5m: setup
	$(PY) analysis/agg_hist.py --symbol $(SYMBOL)
	$(PY) analysis/reject_matrix.py --tf 5m

ab-gates-15m: setup
	$(PY) experiments/ab_gates.py --symbol $(SYMBOL) --csv $(CSV_15M) --coin-config $(COINCFG_OUT_15M) --steps $(STEPS_15M) --balance $(BAL)

ab-gates-5m: setup
	$(PY) experiments/ab_gates.py --symbol $(SYMBOL) --csv $(CSV_5M) --coin-config $(COINCFG_OUT_5M) --steps $(STEPS_5M) --balance $(BAL)

ab-exit-15m: setup
	$(PY) experiments/ab_exit.py --symbol $(SYMBOL) --csv $(CSV_15M) --coin-config $(COINCFG_OUT_15M) \
		--params-json $(PRESETS) --preset-a $(PRESET_15M) --preset-b $(PRESET_5M)

ab-exit-5m: setup
	$(PY) experiments/ab_exit.py --symbol $(SYMBOL) --csv $(CSV_5M) --coin-config $(COINCFG_OUT_5M) \
		--params-json $(PRESETS) --preset-a $(PRESET_5M) --preset-b $(PRESET_15M)

ab-ml-15m: setup
	$(PY) experiments/ab_ml.py --symbol $(SYMBOL) --csv $(CSV_15M) --coin-config $(COINCFG_OUT_15M) --steps $(STEPS_15M) --balance $(BAL) --thr 0.70 0.75 0.80 0.85 0.90

run-matrix-15m: setup
	$(PY) experiments/run_matrix.py --symbols $(SYMBOLS) --tf 15m --coin-config $(COINCFG_OUT_15M) --steps $(STEPS_15M) --balance $(BAL) --use-ml 0

run-matrix-5m: setup
	$(PY) experiments/run_matrix.py --symbols $(SYMBOLS) --tf 5m --coin-config $(COINCFG_OUT_5M) --steps $(STEPS_5M) --balance $(BAL) --use-ml 0

walkforward-15m: setup
	$(PY) experiments/walkforward.py --symbol $(SYMBOL) --csv $(CSV_15M) --coin-config $(COINCFG_OUT_15M) --window 500 --steps $(STEPS_15M) --balance $(BAL) --tf 15m

walkforward-5m: setup
	$(PY) experiments/walkforward.py --symbol $(SYMBOL) --csv $(CSV_5M) --coin-config $(COINCFG_OUT_5M) --window 1500 --steps $(STEPS_5M) --balance $(BAL) --tf 5m

trade-metrics: setup
	# ganti path trades CSV sesuai output tools_dryrun_summary kamu
	$(PY) metrics/trade_analyzer.py --symbol $(SYMBOL) --trades $(SYMBOL)_dryrun_trades_real.csv --csv $(CSV_15M) --coin-config $(COINCFG_OUT_15M)

mc: setup
	$(PY) analysis/mc_sim.py --trades $(SYMBOL)_dryrun_trades_real.csv --runs 10000 --horizon 200

all-15m: apply-presets audit-15m baseline-15m telemetry-15m ab-gates-15m ab-exit-15m run-matrix-15m walkforward-15m

all-5m: apply-presets audit-5m baseline-5m telemetry-5m ab-gates-5m ab-exit-5m run-matrix-5m walkforward-5m

clean-reports:
	rm -rf reports/* logs/*

