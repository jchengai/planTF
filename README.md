This is the official repository of

**Rethink Imitation-based Planner for Autonomous Driving**,
*Jie Cheng,Yingbing chen,Xiaodong Mei,Bowen Yang,Bo Li and Ming Liu*, arXiv 2023

<p align="left">
<a href="https://jchengai.github.io/planTF">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/pdf/2309.10443.pdf' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

<image src="https://github.com/jchengai/planTF/assets/86758357/42170613-8759-4359-80ce-0b17c51676b8" height=400 width=400>
<image src="https://github.com/jchengai/planTF/assets/86758357/2c06a97e-d543-4b82-8a75-bbf859abe148" height=400 width=400>

## Highlight
- A good starting point for research on learning-based planner on the [nuPlan](https://www.nuscenes.org/nuplan) dataset. This repo provides detailed instructions on data preprocess, training and benchmark.
- A simple pure learning-based baseline model **planTF**, that achieves decent performance **without** any rule-based strategies or post-optimization.

## Get Started

- [Get Started](#get-started)
- [Setup Environment](#setup-environment)
- [Feature cache](#feature-cache)
- [Training](#training)
- [Trained models](#trained-models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


## Setup Environment

- setup the nuPlan dataset following the [offiical-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)
- setup conda environment
```
conda create -n plantf python=3.9
conda activate plantf

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt

# setup planTF
cd ..
git clone https://github.com/jchengai/planTF.git && cd planTF
sh ./script/setup_env.sh
```

## Feature cache

Preprocess the dataset to accelerate training. The following command generates 1M frames of training data from the whole nuPlan training set. You may need:
- change `cache.cache_path` to suit your condition
- decrease/increase `worker.threads_per_node` depends on your RAM and CPU.

```sh
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_planTF \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_plantf_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

This process may take some time, be patient (20+hours in my setting).

## Training

We modified the training scipt provided by [nuplan-devkit](https://github.com/autonomousvision/tuplan_garage) a little bit for more flexible training.
By default, the training script will use all visible GPUs for training. PlanTF is quite lightweight, which takes about 4~6G GPU memory under the batch size of 32 (each GPU).

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/nuplan/exp/cache_plantf_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=online wandb.project=nuplan wandb.name=plantf
```

you can remove wandb related configurations if your prefer tensorboard.

## Trained models

Place the trained models at `planTF/checkpoints/`

| Model                  | Document                                              | Download                                                                                                                                          |
| ---------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| PlanTF (state6+SDE)    | -                                                     | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EW7HbklkAhVNpcDUEga2aLABxioVA1S98vyqk2VbziYfTw?e=fe3CxI) |
| RasterModel            | [Doc](./docs/other_baselines.md#rastermodel)           | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EcfVyHFUoV1KhAv7D_JPqtwBlwR-2zT2suGHD1rLXsBtKA?e=PIwD7U) |
| UrbanDriver (openloop) | [Doc](./docs/other_baselines.md#urbandriver-open-loop) | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EbM_BSpFS9NBqIWuhlVHMrYBMrSOtusHjH6hwfamZCuI_Q?e=Q2bN75) |



## Evaluation


- run a single scenario simulation (for sanity check): `sh ./script/plantf_single_scenarios.sh`
- run **Test14-random**: `sh ./script/plantf_benchmarks.sh test14-random`
- run **Test14-hard**: `sh ./script/plantf_benchmarks.sh test14-hard`
- run **Val14** (this may take a long time): `sh ./script/plantf_benchmarks.sh val14`

## Results

### Test14-random and Test14-hard benchmarks

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2">Planners</th>
    <th class="tg-c3ow" colspan="3">Test14-random</th>
    <th class="tg-c3ow" colspan="3">Test14-hard</th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Type</td>
    <td class="tg-0pky">Method</td>
    <td class="tg-c3ow">OLS↑</td>
    <td class="tg-c3ow">NR-CLS↑</td>
    <td class="tg-c3ow">R-CLS↑</td>
    <td class="tg-c3ow">OLS↑</td>
    <td class="tg-c3ow">NR-CLS↑</td>
    <td class="tg-c3ow">R-CLS↑</td>
    <td class="tg-0pky">Time</td>
  </tr>
  <tr>
    <td class="tg-0pky">Expert</td>
    <td class="tg-0pky">LogReplay</td>
    <td class="tg-c3ow">100.0</td>
    <td class="tg-c3ow">94.03</td>
    <td class="tg-c3ow">75.86</td>
    <td class="tg-c3ow">100.0</td>
    <td class="tg-c3ow">85.96</td>
    <td class="tg-c3ow">68.80</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">Rule-based</td>
    <td class="tg-0pky">IDM</td>
    <td class="tg-c3ow">34.15</td>
    <td class="tg-c3ow">70.39</td>
    <td class="tg-c3ow">72.42</td>
    <td class="tg-c3ow">20.07</td>
    <td class="tg-c3ow">56.16</td>
    <td class="tg-c3ow">62.26</td>
    <td class="tg-c3ow">32</td>
  </tr>
  <tr>
    <td class="tg-0pky">PDM-Closed</td>
    <td class="tg-c3ow">46.32</td>
    <td class="tg-c3ow">90.05</td>
    <td class="tg-7btt">91.64</td>
    <td class="tg-c3ow">26.43</td>
    <td class="tg-c3ow">65.07</td>
    <td class="tg-c3ow">75.18</td>
    <td class="tg-c3ow">140</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">Hybrid</td>
    <td class="tg-0pky">GameFormer</td>
    <td class="tg-c3ow">79.35</td>
    <td class="tg-c3ow">80.80</td>
    <td class="tg-c3ow">79.31</td>
    <td class="tg-c3ow">75.27</td>
    <td class="tg-c3ow">66.59</td>
    <td class="tg-c3ow">68.83</td>
    <td class="tg-c3ow">443</td>
  </tr>
  <tr>
    <td class="tg-0pky">PDM-Hybrid</td>
    <td class="tg-c3ow">82.21</td>
    <td class="tg-7btt">90.20</td>
    <td class="tg-c3ow">91.56</td>
    <td class="tg-c3ow">73.81</td>
    <td class="tg-c3ow">65.95</td>
    <td class="tg-7btt">75.79</td>
    <td class="tg-c3ow">152</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="5">Learning-based<br><br></td>
    <td class="tg-0pky">PlanCNN</td>
    <td class="tg-c3ow">62.93</td>
    <td class="tg-c3ow">69.66</td>
    <td class="tg-c3ow">67.54</td>
    <td class="tg-c3ow">52.4</td>
    <td class="tg-c3ow">49.47</td>
    <td class="tg-c3ow">52.16</td>
    <td class="tg-c3ow">82</td>
  </tr>
  <tr>
    <td class="tg-0pky">UrbanDriver <sup><span>&#8224</span></sup></td>
    <td class="tg-c3ow">82.44</td>
    <td class="tg-c3ow">63.27</td>
    <td class="tg-c3ow">61.02</td>
    <td class="tg-c3ow">76.9</td>
    <td class="tg-c3ow">51.54</td>
    <td class="tg-c3ow">49.07</td>
    <td class="tg-c3ow">124</td>
  </tr>
  <tr>
    <td class="tg-0pky">GC-PGP</td>
    <td class="tg-c3ow">77.33</td>
    <td class="tg-c3ow">55.99</td>
    <td class="tg-c3ow">51.39</td>
    <td class="tg-c3ow">73.78</td>
    <td class="tg-c3ow">43.22</td>
    <td class="tg-c3ow">39.63</td>
    <td class="tg-c3ow">160</td>
  </tr>
  <tr>
    <td class="tg-0pky">PDM-Open</td>
    <td class="tg-c3ow">84.14</td>
    <td class="tg-c3ow">52.80</td>
    <td class="tg-c3ow">57.23</td>
    <td class="tg-c3ow">79.06</td>
    <td class="tg-c3ow">33.51</td>
    <td class="tg-c3ow">35.83</td>
    <td class="tg-c3ow">101</td>
  </tr>
  <tr>
    <td class="tg-0pky">PlanTF (Ours)</td>
    <td class="tg-7btt">87.07</td>
    <td class="tg-c3ow">86.48</td>
    <td class="tg-c3ow">80.59</td>
    <td class="tg-7btt">83.32</td>
    <td class="tg-7btt">72.68</td>
    <td class="tg-c3ow">61.7</td>
    <td class="tg-c3ow">155</td>
  </tr>
</tbody>
</table>

<p>
<sup><span>&#8224</span></sup> open-loop re-implementation
</p>

### Val14 benchmark

| Method        | OLS   | NR-CLS | R-CLS |
| ------------- | ----- | ------ | ----- |
| Log-replay    | 100   | 94     | 80    |
| IDM           | 38    | 77     | 76    |
| GC-PGP        | 82    | 57     | 54    |
| PlanCNN       | 64    | 73     | 72    |
| PDM-Hybrid    | 84    | 93     | 92    |
| PlanTF (Ours) | 89.18 | 84.83  | 76.78 |

## Acknowledgements

Many thanks to the open-source community, also checkout these works:
- [tuplan_garage](https://github.com/autonomousvision/tuplan_garage)
- [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner)

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@misc{cheng2023plantf,
      title={Rethinking Imitation-based Planner for Autonomous Driving},
      author={Jie Cheng and Yingbing Chen and Xiaodong Mei and Bowen Yang and Bo Li and Ming Liu},
      year={2023},
      eprint={2309.10443},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
