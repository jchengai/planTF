import logging
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from src.features.nuplan_feature import NuplanFeature
from src.utils.collision_checker import CollisionChecker


logger = logging.getLogger(__name__)


class StatePerturbation(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)


    Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std.
    """

    def __init__(
        self,
        dt: float = 0.1,
        hist_len: int = 21,
        low: List[float] = [0.0, -1.5, -0.55, -1, -0.5, -0.2, -0.2],
        high: List[float] = [2.0, 1.5, 0.55, 1, 0.5, 0.2, 0.2],
        augment_prob: float = 0.5,
        normalize=True,
    ) -> None:
        """
        Initialize the augmentor,
        state: [x, y, yaw, vel, acc, steer, steer_rate, angular_vel, angular_acc],
        :param dt: Time interval between trajectory points.
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._dt = dt
        self._hist_len = hist_len
        self._random_offset_generator = UniformNoise(low, high)
        self._augment_prob = augment_prob
        self._normalize = normalize
        self._collision_checker = CollisionChecker()
        self._rear_to_cog = get_pacifica_parameters().rear_axle_to_center

    def safety_check(
        self,
        ego_position: npt.NDArray[np.float32],
        ego_heading: npt.NDArray[np.float32],
        agents_position: npt.NDArray[np.float32],
        agents_heading: npt.NDArray[np.float32],
        agents_shape: npt.NDArray[np.float32],
    ) -> bool:
        if len(agents_position) == 0:
            return True

        ego_center = (
            ego_position
            + np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=-1)
            * self._rear_to_cog
        )
        ego_state = torch.from_numpy(
            np.concatenate([ego_center, [ego_heading]], axis=-1)
        ).unsqueeze(0)
        objects_state = torch.from_numpy(
            np.concatenate([agents_position, agents_heading[..., None]], axis=-1)
        ).unsqueeze(0)

        collisions = self._collision_checker.collision_check(
            ego_state=ego_state,
            objects=objects_state,
            objects_width=torch.from_numpy(agents_shape[:, 0]).unsqueeze(0),
            objects_length=torch.from_numpy(agents_shape[:, 1]).unsqueeze(0),
        )

        return not collisions.any()

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType = None,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        data = features["feature"].data

        current_state = data["current_state"]
        new_state = current_state + self._random_offset_generator.sample()
        new_state[3] = max(0.0, new_state[3])

        # consider nearest 10 agents
        agents_position = data["agent"]["position"][1:11, self._hist_len - 1]
        agents_shape = data["agent"]["shape"][1:11, self._hist_len - 1]
        agents_heading = data["agent"]["heading"][1:11, self._hist_len - 1]
        agents_shape = data["agent"]["shape"][1:11, self._hist_len - 1]

        if not self.safety_check(
            ego_position=new_state[:2],
            ego_heading=new_state[2],
            agents_position=agents_position,
            agents_heading=agents_heading,
            agents_shape=agents_shape,
        ):
            return features, targets

        data["current_state"] = new_state
        data["agent"]["position"][0, self._hist_len - 1] = new_state[:2]
        data["agent"]["heading"][0, self._hist_len - 1] = new_state[2]

        if self._normalize:
            features["feature"] = NuplanFeature.normalize(data)

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f"{self._augment_prob=}".partition("=")[0].split(".")[1],
            scaling_direction=ScalingDirection.MAX,
        )

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(
            List[ParameterToScale],
            self._random_offset_generator.get_schedulable_attributes(),
        )
