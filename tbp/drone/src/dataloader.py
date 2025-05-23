from typing import List

import numpy as np

from tbp.drone.src.actions import Action
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoader,
    EnvironmentDataLoaderPerObject,
    EnvironmentDataset,
    MotorSystem,
    MotorSystemState,
)


class DroneDataLoader(EnvironmentDataLoader):
    def __init__(
        self, dataset, motor_system, object_name: str, actions: List[Action], rng=None
    ):
        self.dataset = dataset
        self.motor_system = motor_system
        self.primary_target = {
            "object": object_name,
            "rotation": np.quaternion(0, 0, 0, 1),
        }


class DroneStreamDataLoader(EnvironmentDataLoader):
    def __init__(
        self, dataset, motor_system, object_name: str, actions: List[Action], rng=None
    ):
        self.dataset = dataset
        self.motor_system = motor_system
        self.primary_target = {
            "object": object_name,
            "rotation": np.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }
        self._actions = list(actions)
        self._action_counter = 0

    def pre_episode(self):
        pass

    def post_episode(self):
        pass

    def __iter__(self):
        # Reset the environment before iterating
        self._observation, _ = self.dataset.reset()  # starts the drone up
        self._action_counter = 0
        return self

    def __next__(self):
        if self._action_counter == len(self._actions):
            raise StopIteration
        if self._action_counter == 0:
            # Return first observation after 'reset' before any action is applied
            self._action_counter += 1
            return self._observation
        else:
            action = self._actions[self._action_counter]
            self._observation, _ = self.dataset[action]
            self._action_counter += 1
            return self._observation


class DroneImageDataLoader(EnvironmentDataLoaderPerObject):
    """Dataloader for moving over a 2D image with depth channel."""

    def __init__(
        self,
        dataset: EnvironmentDataset,
        motor_system: MotorSystem,
        *args,
        **kwargs,
    ):
        """Initialize dataloader.

        Args:
            scenes: List of scenes
            versions: List of versions
            dataset (EnvironmentDataset): The environment dataset.
            motor_system (MotorSystem): The motor system.
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        """
        self.dataset = dataset
        self.motor_system = motor_system
        self._observation, proprioceptive_state = self.dataset.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0

        self.object_names = self.dataset.env.scene_names
        self.current_scene_version = 0

        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def post_episode(self):
        self.motor_system.post_episode()
        self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        """Switch to the next scene image."""
        next_scene = (self.current_scene_version + 1) % self.n_versions
        logging.info(
            f"\n\nGoing from {self.current_scene_version} to {next_scene} of "
            f"{self.n_versions}"
        )
        self.change_object_by_idx(next_scene)

    def change_object_by_idx(self, idx):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params
        """
        assert idx <= self.n_versions, "idx must be <= self.n_versions"
        logging.info(
            f"changing to obj {idx} -> scene {self.scenes[idx]}, version "
            f"{self.versions[idx]}"
        )
        self.dataset.env.switch_to_object(self.scenes[idx], self.versions[idx])
        self.current_scene_version = idx
        # TODO: Currently not differentiating between different poses/views
        target_object = self.object_names[self.scenes[idx]]
        # remove scene index from name
        target_object_formatted = "_".join(target_object.split("_")[1:])
        self.primary_target = {
            "object": target_object_formatted,
            "rotation": np.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }

    def __iter__(self):
        # Overwrite original because we don't want to reset agent at this stage
        # (already done in pre-episode)

        return self
