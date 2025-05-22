import datetime
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import quaternion
from djitellopy import Tello

from tbp.drone.src.actions import (
    Action,
    DroneActionSpace,
    Land,
    MoveBackward,
    MoveDown,
    MoveForward,
    MoveLeft,
    MoveRight,
    MoveUp,
    SetHeight,
    SetYaw,
    TakeOff,
    TurnLeft,
    TurnRight,
)
from tbp.drone.src.drone_pilot import DronePilot
from tbp.drone.src.spatial import compute_relative_angle
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment

DATA_DIR = Path("~/tbp/results/drone").expanduser()
MINIMUM_DISTANCE = 0.2  # Minimal traversible distance by drone in meters.


class DroneEnvironment(EmbodiedEnvironment):
    """Main interface to Drone simulator.

    Gets created by DroneEnvironmentDataset.
    """

    def __init__(self):
        super().__init__()

        self._agent_id = "agent_id_0"

        self._pilot = DronePilot()
        self._position = np.zeros(3)
        self._rotation = quaternion.quaternion(1, 0, 0, 0)
        self._step_counter = 0

        # dead-reckoning pose
        self.agent_pose_dr = {
            "position": np.zeros(3),
            "rotation": quaternion.quaternion(1, 0, 0, 0),
        }

    @property
    def action_space(self) -> DroneActionSpace:
        return DroneActionSpace(
            "TakeOff",
            "Land",
            "MoveForward",
            "MoveBackward",
            "MoveLeft",
            "MoveRight",
            "MoveUp",
            "MoveDown",
            "TurnLeft",
            "TurnRight",
            "SetYaw",
            "SetHeight",
        )

    """
    ------------------------------------------------------------------------------------------------
    Drone Actuating Methods
    """

    def actuate_takeoff(self, action: TakeOff):
        self._pilot.takeoff()

    def actuate_land(self, action: Land):
        self._pilot.land()

    def actuate_move_forward(self, action: MoveForward) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return
        if distance < 0:
            self.actuate_move_backward(MoveBackward(-distance))
        elif distance < MINIMUM_DISTANCE:
            self.actuate_move_backward(MoveBackward(MINIMUM_DISTANCE))
            self.actuate_move_forward(MoveForward(distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_forward(distance)

    def actuate_move_backward(self, action: MoveBackward) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return

        if distance < 0:
            self.actuate_move_forward(MoveForward(-distance))
        elif distance < MINIMUM_DISTANCE:
            self.actuate_move_forward(MoveForward(MINIMUM_DISTANCE))
            self.actuate_move_backward(MoveBackward(distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_backward(distance)

    def actuate_move_left(self, action: MoveLeft) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return
        if distance < 0:
            self.actuate_move_right(MoveRight(-distance))
        elif distance < MINIMUM_DISTANCE:
            self.actuate_move_right(MoveRight(MINIMUM_DISTANCE))
            self.actuate_move_left(MoveLeft(distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_left(distance)

    def actuate_move_right(self, action: MoveRight) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return
        if distance < MINIMUM_DISTANCE:
            self.actuate_move_left(MoveLeft(MINIMUM_DISTANCE))
            self.actuate_move_right(MoveRight(distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_right(distance)

    def actuate_move_up(self, action: MoveUp) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return
        if distance < MINIMUM_DISTANCE:
            self.actuate_move_down(MoveDown(MINIMUM_DISTANCE))
            self.actuate_move_up(MoveUp(distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_up(distance)

    def actuate_move_down(self, action: MoveDown) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return
        if distance < MINIMUM_DISTANCE:
            self.actuate_move_up(MoveUp(MINIMUM_DISTANCE))
            self.actuate_move_down(MoveDown(distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_down(distance)

    def actuate_turn_left(self, action: TurnLeft) -> None:
        angle = action.angle
        if np.isclose(angle, 0):
            return
        if angle < 0:
            self.actuate_turn_right(TurnRight(-angle))
        else:
            self._pilot.rotate_counter_clockwise(angle)

    def actuate_turn_right(self, action: TurnRight) -> None:
        angle = action.angle
        if np.isclose(angle, 0):
            return
        if angle < 0:
            self.actuate_turn_left(TurnLeft(-angle))
        else:
            self._pilot.rotate_clockwise(angle)

    def actuate_set_height(self, action: SetHeight) -> None:
        current_height = self._pilot.get_height()
        desired_height = action.height
        delta = int(np.round(desired_height - current_height))
        if np.isclose(delta, 0):
            return
        if delta < 0:
            self.actuate_move_down(MoveDown(-delta))
        else:
            self.actuate_move_up(MoveUp(delta))

    def actuate_set_yaw(self, action: SetYaw) -> None:
        current_yaw = self._pilot.get_yaw()
        desired_yaw = action.angle
        # delta is in [-180, 180]
        delta = compute_relative_angle(current_yaw, desired_yaw)
        if np.isclose(delta, 0):
            return
        if delta < 0:
            self.actuate_turn_left(TurnLeft(-delta))
        else:
            self.actuate_turn_right(TurnRight(delta))

    """
    ------------------------------------------------------------------------------------------------
    Interface used by Dataset (required)
    """

    def apply_action(self, action: Action) -> Dict[str, Dict]:
        """Execute given action in the environment.

        Args:
            action: Dictionary containing action name and parameters

        Returns:
            Dictionary with observations grouped by agent_id
        """
        action_name = action.get("name")
        if action_name not in self._action_space:
            raise ValueError(f"Invalid action name: {action_name}")

        # Initialize Tello connection if needed
        if self._pilot is None:
            self.start()

        action.act(self)

        # Get updated observations
        return self.get_observations()

    def get_state(self) -> Dict[str, Dict]:
        """Get agent and sensor states.

        Returns:
            Dictionary with agent poses and states
        """
        patch_state = {
            "patch.depth": {
                "rotation": quaternion.quaternion(1, 0, 0, 0),
                "position": np.zeros(3),
            },
            "patch.rgba": {
                "rotation": quaternion.quaternion(1, 0, 0, 0),
                "position": np.zeros(3),
            },
        }
        view_finder_state = {
            "view_finder.depth": {
                "rotation": quaternion.quaternion(1, 0, 0, 0),
                "position": np.zeros(3),
            },
            "view_finder.rgba": {
                "rotation": quaternion.quaternion(1, 0, 0, 0),
                "position": np.zeros(3),
            },
        }
        agent_state = {
            "sensors": {
                "patch": patch_state,
                "view_finder": view_finder_state,
            },
            "rotation": quaternion.quaternion(1, 0, 0, 0),
            "position": np.zeros(3),
        }
        return {self._agent_id: agent_state}

    def step(self, action) -> Dict[str, Dict]:
        obs = self.apply_action(action)
        self._step_counter += 1
        return obs

    def reset(self):
        self._pilot.start()
        self._pilot.takeoff()
        return self.get_observations()

    def close(self):
        """Close simulator and release resources."""
        if self._pilot is not None:
            self._pilot.land()
            self._pilot.shutdown()

    """
    ------------------------------------------------------------------------------------------------
    Internally Used Methods
    """

    def get_observations(self) -> Dict[str, Dict]:
        """Get sensor observations.

        Called by `DroneEnvironment.apply_action`.

        Returns:
            Dictionary with all sensor observations grouped by agent_id
        """
        if self._pilot is None:
            return {}
        if isinstance(self._pilot, Pilot):
            frame = self._pilot.get_frame_read().frame

        # Process observations for each agent
        observations = {
            "agent_id_0": {
                "patch": {
                    "depth": None,
                    "rgba": None,
                    "semantic_3d": None,
                    "sensor_frame_data": None,
                    "world_camera": None,
                },
                "view_finder": {
                    "depth": None,
                    "rgba": frame,
                },
            }
        }
        return observations

    """
    ------------------------------------------------------------------------------------------------
    Unused but required methods
    """

    def add_object(self, *args, **kwargs):
        # TODO The NotImplementedError highlights an issue with the EmbodiedEnvironment
        #      interface and how the class hierarchy is defined and used.
        raise NotImplementedError("DroneEnvironment does not support adding objects")

    def remove_all_objects(self):
        # TODO The NotImplementedError highlights an issue with the EmbodiedEnvironment
        #      interface and how the class hierarchy is defined and used.
        raise NotImplementedError(
            "DroneEnvironment does not support removing all objects"
        )

