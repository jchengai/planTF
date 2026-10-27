#!/usr/bin/env bash
# sweep_lambda.sh
# ---------------
# Sweep the mode re-ranker weight (lambda) on mini_eval_v2 and compare every
# setting against a baseline (argmax) run, scenario by scenario.
#
# Choose lambda on the "tune" half of the tokens and report it on the
# "holdout" half (see script/compare_runs.py). Each run is ~20 min on an
# RTX 3060 laptop, so the default 5-value sweep plus baseline takes ~2 h.
#
# Usage:
#   bash script/sweep_lambda.sh [lambda ...]
#   bash script/sweep_lambda.sh 0.1 0.2 0.3 0.5
#
# Environment:
#   NUPLAN_DATA_ROOT  nuPlan dataset root (required by mini_eval_v2.sh)
#   NUPLAN_EXP_ROOT   nuPlan experiment root (default: datasets/nuplan/exp)
#   CHALLENGE         default closed_loop_nonreactive_agents
#   SKIP_BASELINE=1   reuse an existing baseline CSV given in BASELINE_CSV
#
# Outputs:
#   outputs/eval_v2/eval_v2_<tag>_<timestamp>.csv / .md   per run
#   outputs/eval_v2/sweep_<timestamp>/compare_<tag>_{tune,holdout}.md

set -euo pipefail

CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
EXP_ROOT="${NUPLAN_EXP_ROOT:-datasets/nuplan/exp}"
LAMBDAS=("$@")
if [ ${#LAMBDAS[@]} -eq 0 ]; then
    LAMBDAS=(0.1 0.2 0.3 0.5 0.7)
fi

SWEEP_ID="sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="outputs/eval_v2/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"

# Run mini_eval_v2 with the given tag and overrides, then save its per-scenario
# CSV and print the CSV path.
run_and_summarize() {
    local tag="$1"; shift
    RUN_TAG="${SWEEP_ID}_${tag}" bash script/mini_eval_v2.sh "$CHALLENGE" "$@" >&2

    local agg
    agg=$(find "$EXP_ROOT" -type d -path "*${SWEEP_ID}_${tag}/aggregator_metric" | head -1)
    if [ -z "$agg" ]; then
        echo "ERROR: no aggregator_metric dir found for run ${SWEEP_ID}_${tag} under $EXP_ROOT" >&2
        exit 1
    fi
    python script/summarize_eval.py --challenge "$CHALLENGE" --run-dir "$agg" \
        --save --tag "$tag" >&2
    ls -t outputs/eval_v2/eval_v2_"${tag}"_*.csv | head -1
}

if [ "${SKIP_BASELINE:-0}" = "1" ]; then
    BASE_CSV="${BASELINE_CSV:?set BASELINE_CSV when SKIP_BASELINE=1}"
else
    BASE_CSV=$(run_and_summarize baseline planner.imitation_planner.use_mode_reranker=false)
fi
echo "Baseline CSV: $BASE_CSV"

for lam in "${LAMBDAS[@]}"; do
    tag="rerank_l${lam}"
    csv_path=$(run_and_summarize "$tag" \
        planner.imitation_planner.use_mode_reranker=true \
        planner.imitation_planner.reranker_lambda="$lam")
    for split in tune holdout; do
        python script/compare_runs.py "$BASE_CSV" "$csv_path" --split "$split" \
            --md "$SWEEP_DIR/compare_${tag}_${split}.md" > /dev/null
    done
    echo "lambda=$lam -> $csv_path"
done

echo ""
echo "Tune-split results (pick lambda here, then read only its holdout report):"
grep -H "Paired mean delta\|Bootstrap" "$SWEEP_DIR"/compare_*_tune.md
echo ""
echo "Reports: $SWEEP_DIR"
