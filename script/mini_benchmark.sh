SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python run_simulation.py \
  +simulation=closed_loop_nonreactive_agents \
  planner=planTF \
  scenario_builder=nuplan \
  scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1_mini/data/cache/mini" \
  scenario_filter=mini_benchmark \
  worker=sequential \
  verbose=true \
  planner.imitation_planner.planner_ckpt="$REPO_ROOT/checkpoints/planTF.ckpt"

# --- Export aggregated metrics to CSV ---
LATEST=$(ls -dt "$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents"/*/ 2>/dev/null | head -1)
PARQUET=$(ls "$LATEST/aggregator_metric/"*.parquet 2>/dev/null | head -1)
CSV_OUT="$LATEST/aggregator_metric/metrics.csv"

if [ -n "$PARQUET" ]; then
  python - <<EOF
import pandas as pd
df = pd.read_parquet("$PARQUET")
df.to_csv("$CSV_OUT", index=False)
print(df.to_string(index=False))
print("\nCSV saved to: $CSV_OUT")
EOF
else
  echo "WARNING: No aggregator parquet found in $LATEST/aggregator_metric/"
fi