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
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)

DATA_DIR = Path("~/tbp/results/drone").expanduser()
MINIMUM_DISTANCE = 0.2  # Minimal traversible distance by drone in meters.


class Pilot:
    def __init__(self):
        self._tello = Tello()
        self._tello.connect(wait_for_state=True)
        self._tello.streamon()
        self._tello.get_frame_read()  # first capture is always blank

    def takeoff(self):
        self._tello.takeoff()

    def land(self):
        self._tello.land()

    def move_forward(self, distance: int):
        self._tello.move_forward(round(100 * distance))

    def move_backward(self, distance: int):
        self._tello.move_backward(round(100 * distance))

    def move_left(self, distance: int):
        self._tello.move_left(round(100 * distance))

    def move_right(self, distance: int):
        self._tello.move_right(round(100 * distance))

    def move_up(self, distance: int):
        self._tello.move_up(round(100 * distance))

    def move_down(self, distance: int):
        self._tello.move_down(round(100 * distance))

    def rotate_counter_clockwise(self, angle: int):
        self._tello.rotate_counter_clockwise(round(angle))

    def rotate_clockwise(self, angle: int):
        self._tello.rotate_clockwise(round(angle))

    def get_frame_read(self):
        return self._tello.get_frame_read()

    def get_battery(self):
        return self._tello.get_battery()

    def get_height(self):
        return self._tello.get_height() / 100

    def take_picture(self):
        return self._tello.get_frame_read().frame


class DroneEnvironment(EmbodiedEnvironment):
    """Main interface to Drone simulator.

    Gets created by DroneEnvironmentDataset.
    """

    def __init__(self, agent_id: str):
        super().__init__()

        self._agent_id = agent_id

        self._pilot = Pilot()
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
        distance = int(np.round(action.distance))
        if np.isclose(distance, 0):
            return
        if distance < 0:
            self.actuate_move_backward(MoveBackward(self._agent_id, -distance))
        elif distance < MINIMUM_DISTANCE:
            self.actuate_move_backward(MoveBackward(self._agent_id, MINIMUM_DISTANCE))
            self.actuate_move_forward(
                MoveForward(self._agent_id, distance + MINIMUM_DISTANCE)
            )
        else:
            self._pilot.move_forward(distance)

    def actuate_move_backward(self, action: MoveBackward) -> None:
        distance = int(np.round(action.distance))
        if np.isclose(distance, 0):
            return

        if distance < 0:
            self.actuate_move_forward(MoveForward(self._agent_id, -distance))
        elif distance < MINIMUM_DISTANCE:
            self.actuate_move_forward(MoveForward(self._agent_id, MINIMUM_DISTANCE))
            self.actuate_move_backward(
                MoveBackward(self._agent_id, distance + MINIMUM_DISTANCE)
            )
        else:
            self._pilot.move_backward(distance)

    def actuate_move_left(self, action: MoveLeft) -> None:
        distance = int(np.round(action.distance))
        if np.isclose(distance, 0):
            return
        if distance < 0:
            self.actuate_move_right(MoveRight(self._agent_id, -distance))
        elif distance < MINIMUM_DISTANCE:
            self.actuate_move_right(MoveRight(self._agent_id, MINIMUM_DISTANCE))
            self.actuate_move_left(
                MoveLeft(self._agent_id, distance + MINIMUM_DISTANCE)
            )
        else:
            self._pilot.move_left(distance)

    def actuate_move_right(self, action: MoveRight) -> None:
        distance = int(np.round(action.distance))
        if np.isclose(distance, 0):
            return
        if distance < MINIMUM_DISTANCE:
            self.actuate_move_left(MoveLeft(self._agent_id, MINIMUM_DISTANCE))
            self.actuate_move_right(
                MoveRight(self._agent_id, distance + MINIMUM_DISTANCE)
            )
        else:
            self._pilot.move_right(distance)

    def actuate_move_up(self, action: MoveUp) -> None:
        distance = int(np.round(action.distance))
        if np.isclose(distance, 0):
            return
        if distance < MINIMUM_DISTANCE:
            self.actuate_move_down(MoveDown(self._agent_id, MINIMUM_DISTANCE))
            self.actuate_move_up(MoveUp(self._agent_id, distance + MINIMUM_DISTANCE))
        else:
            self._pilot.move_up(distance)

    def actuate_move_down(self, action: MoveDown) -> None:
        distance = action.distance
        if np.isclose(distance, 0):
            return
        if distance < MINIMUM_DISTANCE:
            self.actuate_move_up(MoveUp(self._agent_id, MINIMUM_DISTANCE))
            self.actuate_move_down(
                MoveDown(self._agent_id, distance + MINIMUM_DISTANCE)
            )
        else:
            self._pilot.move_down(distance)

    def actuate_turn_left(self, action: TurnLeft) -> None:
        angle = action.angle
        if np.isclose(angle, 0):
            return
        if angle < 0:
            self.actuate_turn_right(TurnRight(self._agent_id, -angle))
        else:
            self._pilot.rotate_counter_clockwise(angle)

    def actuate_turn_right(self, action: TurnRight) -> None:
        angle = action.angle
        if np.isclose(angle, 0):
            return
        if angle < 0:
            self.actuate_turn_left(TurnLeft(self._agent_id, -angle))
        else:
            self._pilot.rotate_clockwise(angle)

    def actuate_set_yaw(self, action: SetYaw) -> None:
        cur_yaw = self._tello.get_yaw()
        desired_yaw = action.angle
        delta = desired_yaw - cur_yaw
        if np.isclose(delta, 0):
            return
        if delta < 0:
            self.actuate_turn_left(TurnLeft(self._agent_id, -delta))
        else:
            self.actuate_turn_right(TurnRight(self._agent_id, delta))

    def actuate_set_height(self, action: SetHeight) -> None:
        cur_height = self._pilot.get_height()
        desired_height = action.height
        delta = int(np.round(desired_height - cur_height))
        if delta == 0:
            return
        if delta < 0:
            self.actuate_move_down(MoveDown(self._agent_id, -delta))
        else:
            self.actuate_move_up(MoveUp(self._agent_id, delta))

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

    def close(self):
        """Close simulator and release resources."""
        if self._pilot is not None:
            try:
                self._pilot.land()
                self._pilot.streamoff()
                self._pilot.end()
            except:
                pass
            self._pilot = None

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

    def start(self) -> None:
        """Initialize the drone.

        Initializes the output directory and the camera. Then takes off and
        initializes position and rotation based on the drone's post-takeoff state.
        """
        self._pilot = Tello()
        self._pilot.connect(wait_for_state=True)

        # Initialize camera
        self._pilot.streamon()
        self._pilot.get_frame_read()  # first capture is always blank
        self._image_counter = 0

        # Take off, and initialize position, rotation, etc.
        self._pilot.takeoff()
        self._position = np.zeros(3)
        self._rotation = quaternion.quaternion(1, 0, 0, 0)

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
            "SaccadeOnImageEnvironment does not support removing all objects"
        )


class DroneLogger:
    def __init__(self, out_dir: Optional[os.PathLike] = None):
        if out_dir is None:
            now = datetime.datetime.now()
            out_dir = now.strftime("%Y-%m-%d_%H:%M")
        self.out_dir = DATA_DIR / out_dir
        if self.out_dir.exists():
            self.out_dir.rename(self.out_dir.with_suffix(".old"))
        self.out_dir.mkdir(parents=True)
        self._step_counter = -1

    def get_cwd(self, require: bool = True) -> Path:
        path = self.out_dir / self.step
        if require:
            path.mkdir(parents=True)
        return path

    def step(self):
        self._step_counter += 1

    def log(self, message: str):
        self.log_file.write(message + "\n")
        self.log_file.flush()
