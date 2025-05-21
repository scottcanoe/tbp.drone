import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

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
    SetYaw,
    TakeOff,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class DroneEnvironment(EmbodiedEnvironment):
    """Main interface to Drone simulator.

    Gets created by DroneEnvironmentDataset.
    """

    def __init__(self, agent_id: str):
        super().__init__()

        self._agent_id = agent_id
        self._tello = None
        self._position = np.zeros(3)
        self._rotation = quaternion.quaternion(1, 0, 0, 0)
        self._step_counter = 0

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
        )

    """
    ------------------------------------------------------------------------------------------------
    Drone Actuating Methods
    """

    def actuate_takeoff(self, action: TakeOff):
        self.tello.takeoff()

    def actuate_land(self, action: Land):
        self.tello.land()

    def actuate_move_forward(self, action: MoveForward) -> None:
        self.tello.move_forward(int(action.distance))

    def actuate_move_backward(self, action: MoveBackward) -> None:
        self.tello.move_back(int(action.distance))

    def actuate_move_left(self, action: MoveLeft) -> None:
        self.tello.move_left(int(action.distance))

    def actuate_move_right(self, action: MoveRight) -> None:
        self.tello.move_right(int(action.distance))

    def actuate_move_up(self, action: MoveUp) -> None:
        self.tello.move_up(int(action.distance))

    def actuate_move_down(self, action: MoveDown) -> None:
        self.tello.move_down(int(action.distance))

    def actuate_turn_left(self, action: TurnLeft) -> None:
        self.tello.rotate_counter_clockwise(int(action.rotation_degrees))

    def actuate_turn_right(self, action: TurnRight) -> None:
        self.tello.rotate_clockwise(int(action.rotation_degrees))

    def actuate_set_yaw(self, action: SetYaw) -> None:
        raise NotImplementedError("SetYaw is not implemented for the Tello simulator")

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
        if self._tello is None:
            self.start()

        action.act(self)

        # Get updated observations
        return self.get_observations()

    def get_state(self) -> Dict[str, Dict]:
        """Get agent and sensor states.

        Returns:
            Dictionary with agent poses and states
        """
        result = {}
        for agent in self.agents:
            result[agent.agent_id] = {
                "position": agent.position.tolist(),
                "rotation": agent.rotation.tolist(),
                "velocity": agent.velocity.tolist(),
            }
        return result

    def step(self, action) -> Dict[str, Dict]:
        return self.apply_action(action)

    def reset(self):
        self.start()

    def close(self):
        """Close simulator and release resources."""
        if self._tello is not None:
            try:
                self._tello.land()
                self._tello.streamoff()
                self._tello.end()
            except:
                pass
            self._tello = None

    """
    ------------------------------------------------------------------------------------------------
    Internally Used Methods
    """

    def start(self) -> None:
        """Initialize the drone.

        Initializes the output directory and the camera. Then takes off and
        initializes position and rotation based on the drone's post-takeoff state.
        """
        self._tello = Tello()
        self._tello.connect(wait_for_state=True)

        # Initialize camera
        self.tello.streamon()
        self.tello.get_frame_read()  # first capture is always blank
        self._image_counter = 0

        # Take off, and initialize position, rotation, etc.
        self._tello.takeoff()
        self._position = np.zeros(3)
        self._rotation = quaternion.quaternion(1, 0, 0, 0)

    def get_observations(self) -> Dict[str, Dict]:
        """Get sensor observations.

        Returns:
            Dictionary with all sensor observations grouped by agent_id
        """
        if self._tello is None:
            return {}

        # Get Tello state
        state = {
            "battery": self._tello.get_battery(),
        }

        # Get camera frame if available
        try:
            frame = self._tello.get_frame_read().frame
            state["camera"] = frame
        except:
            pass

        # Process observations for each agent
        observations = defaultdict(dict)
        for agent in self.agents:
            observations[agent.agent_id] = agent.process_observations(state)

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
            "SaccadeOnImageEnvironment does not support removing all objects"
        )
