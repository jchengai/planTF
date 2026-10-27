SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python run_simulation.py \
  +simulation=closed_loop_nonreactive_agents \
  planner=planTF \
  scenario_builder=nuplan \
  scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1_mini/data/cache/mini" \
  scenario_filter=mini \
  worker=sequential \
  verbose=true \
  planner.imitation_planner.planner_ckpt="$REPO_ROOT/checkpoints/planTF.ckpt"

# export HYDRA_FULL_ERROR=1

# python run_simulation.py \
#   +simulation=closed_loop_nonreactive_agents \
#   planner=planTF \
#   scenario_builder=nuplan \
#   scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1_mini/data/cache/mini" \
#   scenario_filter=all_scenarios \
#   worker=sequential \
#   verbose=true \
#   planner.imitation_planner.planner_ckpt="checkpoints/planTF.ckpt"


  # This won't run, its building around 0.4Million  scenarios and using just one CPU thread and therefore gets stuck.
  # We can use this as a sanity check to see if the code runs without errors, but we won't be able to run 
  #it end to end. We can use the single_right_turn scenario filter to run a single scenario and make sure everything is working end to end.    