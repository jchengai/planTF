import time
from typing import List, Optional, Type
import os
import csv

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.feature_builders.common.utils import rotate_round_z_axis

from .mode_reranker import DEFAULT_LAMBDA, rerank_modes
from .planner_utils import global_trajectory_to_states, load_checkpoint
from .tl_safety import apply_tl_brake

_MODEL_PATCHES = ("none", "natten_exact", "natten_onnx", "natten_faithful", "onnx_export")


class ImitationPlanner(AbstractPlanner):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: str = None,
        replan_interval: int = 1,
        use_gpu: bool = True,
        record_latency: bool = False,
        use_mode_reranker: bool = False,
        reranker_lambda: float = DEFAULT_LAMBDA,
        use_tl_brake: bool = False,
        model_patch: str = "none",
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        :param record_latency: If True, append per-step latency to
            benchmarks/mini_v1/latency.csv.  Default False (off in eval runs).
        :param use_mode_reranker: If True, select the output mode with
            mode_reranker.rerank_modes instead of probability argmax.
        :param reranker_lambda: Coverage weight passed to rerank_modes.
        :param use_tl_brake: If True, post-process the trajectory with
            tl_safety.apply_tl_brake (experimental, off by default).
        :param model_patch: Patch the model before simulation, so deployable
            variants can be evaluated closed-loop. One of
            "none", "natten_exact" (exact NATTEN re-implementation),
            "natten_onnx" (legacy global-MHA approximation),
            "natten_faithful" (local-window MHA) or
            "onnx_export" (every patch applied before ONNX export, i.e. the
            graph that is exported to ONNX/TensorRT).
        """
        if model_patch not in _MODEL_PATCHES:
            raise ValueError(
                f"model_patch must be one of {_MODEL_PATCHES}, got {model_patch!r}"
            )

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt
        self._initialization: Optional[PlannerInitialization] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1

        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        # Latency CSV (opt-in only — leave off for normal eval runs)
        self._record_latency = record_latency
        if self._record_latency:
            self._benchmark_dir = os.path.join(os.getcwd(), "benchmarks", "mini_v1")
            self._latency_csv = os.path.join(self._benchmark_dir, "latency.csv")
            os.makedirs(self._benchmark_dir, exist_ok=True)

            if not os.path.exists(self._latency_csv):
                with open(self._latency_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step",
                        "feature_build_ms",
                        "forward_ms",
                        "total_planner_ms",
                        "gpu_mem_mb",
                    ])

        self._use_mode_reranker = use_mode_reranker
        self._reranker_lambda = reranker_lambda
        self._use_tl_brake = use_tl_brake
        self._model_patch = model_patch

        self._step_idx = 0

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        # Patch after loading weights: the patches copy the trained NATTEN
        # projections into the replacement attention modules.
        if self._model_patch != "none":
            from inference.common import patches

            {
                "natten_exact": patches.patch_natten_exact,
                "natten_onnx": patches.patch_natten_for_onnx,
                "natten_faithful": patches.patch_natten_faithful,
                "onnx_export": patches.apply_onnx_export_patches,
            }[self._model_patch](self._planner)

        self._planner.eval()
        self._planner = self._planner.to(self.device)
        self._initialization = initialization

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _planning(self, current_input: PlannerInput):
        planner_start = time.perf_counter()

        feature_start = time.perf_counter()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor().to_device(self.device)]
        )
        feature_end = time.perf_counter()
        feature_build_ms = (feature_end - feature_start) * 1000.0
        self._feature_building_runtimes.append(feature_end - feature_start)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_start = time.perf_counter()

        out = self._planner.forward(planner_feature_torch.data)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_ms = (forward_end - forward_start) * 1000.0

        if self._use_mode_reranker:
            local_trajectory = rerank_modes(
                out, planner_feature_torch.data, lambda_=self._reranker_lambda
            )
        else:
            local_trajectory = out["output_trajectory"][0].cpu().numpy()

        if self._use_tl_brake:
            ego_state = current_input.history.ego_states[-1]
            local_trajectory = apply_tl_brake(
                local_trajectory.astype(np.float64),
                planner_feature_torch.data,
                current_speed=ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                dt=self._step_interval,
            )

        planner_end = time.perf_counter()
        total_planner_ms = (planner_end - planner_start) * 1000.0

        gpu_mem_mb = 0.0
        if self.device.type == "cuda":
            gpu_mem_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

        if self._record_latency:
            with open(self._latency_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self._step_idx,
                    feature_build_ms,
                    forward_ms,
                    total_planner_ms,
                    gpu_mem_mb,
                ])

        self._step_idx += 1
        self._start_time = planner_start

        return local_trajectory.astype(np.float64)

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        ego_state = current_input.history.ego_states[-1]

        if self._last_plan_elapsed_step >= self._replan_interval:
            local_trajectory = self._planning(current_input)
            self._global_trajectory = self._get_global_trajectory(
                local_trajectory, ego_state
            )
            self._last_plan_elapsed_step = 0
        else:
            self._global_trajectory = self._global_trajectory[1:]

        trajectory = InterpolatedTrajectory(
            trajectory=global_trajectory_to_states(
                global_trajectory=self._global_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(self._global_trajectory) * self._step_interval,
                step_interval=self._step_interval,
            )
        )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)

        self._last_plan_elapsed_step += 1

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory
