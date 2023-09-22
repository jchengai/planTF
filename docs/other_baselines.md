# Gallery

- [RasterModel](#rastermodel)
- [UrbanDriver (open-loop)](#urbandriver-open-loop)

## RasterModel

### Feature cache

```
python ./run_training.py \
    +training=training_raster_model \
    py_func=cache \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_rater_model_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

### Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./run_training.py \
    +training=training_raster_model \
    py_func=train \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_raster_model_1M \
    data_loader.params.batch_size=32 \
    data_loader.params.num_workers=32 \
    cache.use_cache_without_dataset=true \
    worker=single_machine_thread_pool \
    worker.max_workers=32 \
    optimizer=adam \
    optimizer.lr=1e-4 \
    lightning.trainer.params.max_epochs=60 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones='[20, 40]' \
    lr_scheduler.gamma=0.1 \
    wandb.mode=online wandb.project=nuplan_baseline wandb.name=RasterModel_1M
```

### Evaluation

- run **Test14-random**: `sh ./script/raster_model_benchmarks.sh test14-random`
- run **Test14-hard**: `sh ./script/raster_model_benchmarks.sh test14-hard`
- run **Val14** (this may take a long time): `sh ./script/raster_model_benchmarks.sh val14`

## UrbanDriver (open-loop)

### Feature cache

```
python ./run_training.py \
    +training=training_urban_driver_open_loop_model \
    py_func=cache \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_urban_driver_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

### Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./run_training.py \
    +training=training_urban_driver_open_loop_model \
    py_func=train \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_urban_driver_1M \
    data_loader.params.batch_size=32 \
    data_loader.params.num_workers=32 \
    cache.use_cache_without_dataset=true \
    worker=single_machine_thread_pool \
    worker.max_workers=32 \
    optimizer=adam \
    optimizer.lr=1e-4 \
    lightning.trainer.params.max_epochs=30 \
    lr_scheduler=multistep_lr \
    lr_scheduler.milestones='[20]' \
    lr_scheduler.gamma=0.1 \
    wandb.mode=online wandb.project=nuplan_baseline wandb.name=urban_driver_open_loop_1M
```

### Evaluation

- run **Test14-random**: `sh ./script/urbandriver_benchmarks.sh test14-random`
- run **Test14-hard**: `sh ./script/urbandriver_benchmarks.sh test14-hard`
- run **Val14** (this may take a long time): `sh ./script/urbandriver_benchmarks.sh val14`