from typing import List

import numpy as np

from tbp.drone.src.actions import Action
from tbp.monty.frameworks.environments.embodied_data import EnvironmentDataLoader


class DroneDataLoader(EnvironmentDataLoader):
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
