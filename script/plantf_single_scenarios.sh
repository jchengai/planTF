cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"

python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    scenario_builder=nuplan_challenge \
    scenario_filter=single_right_turn \
    worker=sequential \
    verbose=true \
    planner.imitation_planner.planner_ckpt="$CKPT_ROOT/$PLANNER.ckpt"