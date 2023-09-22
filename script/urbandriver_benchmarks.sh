cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER="urban_driver_open_loop"
SPLIT=$1
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        model=urban_driver_open_loop_model \
        planner=ml_planner \
        'planner.ml_planner.model_config=${model}' \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=20 \
        experiment_uid=$SPLIT/$planner \
        verbose=true \
        planner.ml_planner.checkpoint_path="$CKPT_ROOT/$PLANNER.ckpt"
done