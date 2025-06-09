import datetime
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import imageio
import numpy as np
import quaternion
from djitellopy import Tello

from tbp.drone.src.actions import (
    Action,
    DroneActionSpace,
    Land,
    LookDown,
    LookLeft,
    LookRight,
    LookUp,
    MoveBackward,
    MoveDown,
    MoveForward,
    MoveLeft,
    MoveRight,
    MoveUp,
    NextImage,
    SetHeight,
    SetYaw,
    TakeOff,
    TurnLeft,
    TurnRight,
)
from tbp.drone.src.drone_pilot import DronePilot
from tbp.drone.src.spatial import compute_relative_angle, pitch_roll_yaw_to_quaternion
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment

DATA_DIR = Path("~/tbp/results/drone").expanduser()
MINIMUM_DISTANCE = 0.2  # Minimal traversible distance by drone in meters.


@dataclass
class Sensor:
    name: str
    position: np.ndarray
    rotation: quaternion.quaternion
    rgb: Optional[np.ndarray] = None
    rgba: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None


@dataclass
class Agent:
    name: str
    position: np.ndarray
    rotation: quaternion.quaternion
    sensors: Dict[str, Sensor]

    def observation_dict(self):
        observation = {self.name: {}}
        for sensor_name, sensor in self.sensors.items():
            observation[self.name][sensor_name] = {
                "depth": sensor.depth,
                "rgba": sensor.rgba,
            }
        return observation

    def state_dict(self):
        state = {
            self.name: {
                "position": self.position,
                "rotation": self.rotation,
                "sensors": {},
            }
        }
        for sensor_name, sensor in self.sensors.items():
            state[self.name]["sensors"][f"{sensor_name}.depth"] = {
                "position": sensor.position,
                "rotation": sensor.rotation,
            }
            state[self.name]["sensors"][f"{sensor_name}.rgba"] = {
                "position": sensor.position,
                "rotation": sensor.rotation,
            }

        return state


class DroneEnvironment(EmbodiedEnvironment):
    """Summary of the DroneEnvironment class.

    Abstract methods
    - add_object
    - remove_all_objects
    - get_state
    - reset
    - step
    - close

    Subclasses must implement:
    - get_state
    - step
    """

    _action_space = DroneActionSpace([])
    _initial_agent_position = np.array([0.0, 0.0, 0.0])
    _initial_agent_rotation = quaternion.quaternion(1, 0, 0, 0)
    _initial_sensor_position = np.array([0.0, 0.0, 0.0])
    _initial_sensor_rotation = quaternion.quaternion(1, 0, 0, 0)

    def __init__(self, patch_size: int = 64):
        super().__init__()

        self._agent = Agent(
            name="agent_id_0",
            position=self._initial_agent_position.copy(),
            rotation=self._initial_agent_rotation.copy(),
            sensors={
                "patch": Sensor(
                    name="patch",
                    position=self._initial_sensor_position.copy(),
                    rotation=self._initial_sensor_rotation.copy(),
                ),
                "view_finder": Sensor(
                    name="view_finder",
                    position=self._initial_sensor_position.copy(),
                    rotation=self._initial_sensor_rotation.copy(),
                ),
            },
        )
        self._patch_size = patch_size
        self._step_counter = 0

    @property
    def action_space(self) -> DroneActionSpace:
        return self._action_space

    """
    ------------------------------------------------------------------------------------
    Helper Methods
    """

    def _get_observation(self) -> Dict[str, Dict]:
        """Get sensor observations."""
        return self._agent.observation_dict()

    def _reset_agent(self):
        self._agent.position = self._initial_agent_position.copy()
        self._agent.rotation = self._initial_agent_rotation.copy()
        for sensor in self._agent.sensors.values():
            sensor.position = self._initial_sensor_position.copy()
            sensor.rotation = self._initial_sensor_rotation.copy()
            sensor.rgba = None
            sensor.depth = None

    """
    ------------------------------------------------------------------------------------
    Reimplemented Methods
    """

    def get_state(self) -> Dict[str, Dict]:
        """Get agent and sensor states.

        Returns:
            Dictionary with agent poses and states
        """
        return self._agent.state_dict()

    def reset(self):
        pass

    def close(self):
        """Close simulator and release resources."""
        pass

    def add_object(self, *args, **kwargs):
        raise NotImplementedError("DroneEnvironment does not support adding objects")

    def remove_all_objects(self):
        raise NotImplementedError(
            "DroneEnvironment does not support removing all objects"
        )


class DroneStreamEnvironment(DroneEnvironment):
    """Main interface to Drone simulator.

    Gets created by DroneEnvironmentDataset.
    """

    _action_space = DroneActionSpace(
        [
            "take_off",
            "land",
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "turn_left",
            "turn_right",
            "set_height",
            "set_yaw",
        ]
    )

    def __init__(self, patch_size: int = 64):
        super().__init__(patch_size)

        self._agent_id = "agent_id_0"
        self._agent_position = np.zeros(3, dtype=float)
        self._agent_rotation = quaternion.quaternion(1, 0, 0, 0)
        self._sensor_position = np.array([0, 0, 0.05])
        self._sensor_rotation = quaternion.quaternion(1, 0, 0, 0)

        self._pilot = DronePilot()
        self._position = np.zeros(3)
        self._rotation = quaternion.quaternion(1, 0, 0, 0)
        self._step_counter = 0

        # dead-reckoning pose
        self.agent_pose_dr = {
            "position": np.zeros(3),
            "rotation": quaternion.quaternion(1, 0, 0, 0),
        }

    """
    ------------------------------------------------------------------------------------
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

    def apply_action(self, action: Action) -> Dict[str, Dict]:
        """Execute given action in the environment.

        Args:
            action: Dictionary containing action name and parameters

        Returns:
            Dictionary with observations grouped by agent_id
        """
        action_name = action.get("name")
        if action_name not in self.action_space:
            raise ValueError(f"Invalid action name: {action_name}")

        action.act(self)

        # Get updated observations
        return self.get_observations()

    def get_observations(self) -> Dict[str, Dict]:
        """Get sensor observations.

        Called by `DroneEnvironment.apply_action`.

        Returns:
            Dictionary with all sensor observations grouped by agent_id
        """
        if self._pilot is None:
            return {}
        if isinstance(self._pilot, DronePilot):
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


class DroneImageEnvironment(DroneEnvironment):
    """Environment for moving over a 2D image with depth channel.

    Images should be stored in .png format for rgb and .data format for depth.
    """

    _action_space = DroneActionSpace(
        [
            "look_up",
            "look_down",
            "look_left",
            "look_right",
            "next_image",
        ]
    )

    def __init__(
        self,
        patch_size: int = 64,
        data_path: Optional[os.PathLike] = None,
        depth_scale_factor: float = 1.0,
        depth_range: Tuple[float, float] = (0, 10000)
    ):
        """Initialize environment.

        Args:
            patch_size: height and width of patch in pixels, defaults to 64
            data_path: path to the image dataset. If None its set to
                ~/tbp/data/worldimages/labeled_scenes/
        """
        super().__init__(patch_size)
        self.depth_scale_factor = depth_scale_factor
        self.depth_range = depth_range

        # Initialize data path
        if data_path is None:
            raise ValueError("data_path is required")
        data_path = Path(data_path).expanduser()
        if not data_path.exists():
            monty_data_dir = Path(
                os.environ.get("MONTY_DATA", "~/tbp/data")
            ).expanduser()
            drone_data_dir = monty_data_dir / "worldimages/drone"
            data_path = drone_data_dir / data_path
            if not data_path.exists():
                raise FileNotFoundError(f"Data path {data_path} does not exist")
        self.data_path = data_path

        # Find the poses/images in the data path.
        stepisode_dirs = [
            p for p in self.data_path.glob("*") if p.is_dir() and p.name.isdigit()
        ]
        stepisode_nums = sorted([int(p.name) for p in stepisode_dirs])
        self._stepisode_dirs = [self.data_path / str(num) for num in stepisode_nums]

        self._current_stepisode = 0
        self._next_action = None

    """
    ------------------------------------------------------------------------------------
    Drone Actuating Methods
    """

    def actuate_look_up(self, action: LookUp):
        pass

    def actuate_look_down(self, action: LookDown):
        pass

    def actuate_look_left(self, action: LookLeft):
        pass

    def actuate_look_right(self, action: LookRight):
        pass

    def actuate_next_image(self, action: NextImage):
        pass

    """
    ------------------------------------------------------------------------------------
    Drone Actuating Methods
    """

    def step(self, action: Action) -> Dict[str, Dict]:
        obs = self.apply_action(action)
        self._step_counter += 1
        return obs

    def _init_stepisode(self):
        # Load image and state
        image, state = self._load_stepisode_data(self._current_stepisode)

        # Update agent and sensor states
        pitch, roll, yaw = state["pitch"], state["roll"], state["yaw"]
        self._agent.rotation = pitch_roll_yaw_to_quaternion(pitch, roll, yaw)
        self._agent.sensors["view_finder"].rgba = image

        # Do processing.
        # - estimate depth
        # - find arcuro, update drone position from it, etc.

    def _load_stepisode_data(self, stepisode: int):
        """Load depth and rgb data for next scene environment.

        Returns:
            current_depth_image: The depth image.
            current_rgb_image: The rgb image.
            start_location: The start location.
        """
        stepisode_dir = self._stepisode_dirs[stepisode]
        data = {}

        # RGB image
        image = imageio.imread(stepisode_dir / "image.png")
        data["image"] = image

        # Depth image
        depth = np.load(stepisode_dir / "depth.npy")
        data["depth"] = self.depth_scale_factor * depth

        # Drone state
        # with open(stepisode_dir / "drone_state.json", "r") as f:
        #     drone_state = json.load(f)
        # data["drone_state"] = drone_state

        # Agent state
        with open(stepisode_dir / "agent_state.json", "r") as f:
            agent_state = json.load(f)
        data["agent_state"] = agent_state

        # Bounding box
        with open(stepisode_dir / "bbox.json", "r") as f:
            bbox = json.load(f)
        data["bbox"] = bbox

        return data

        # state = self._agent.state_dict()
        # # Set data paths
        # current_depth_path = (
        #     self.data_path + f"{self.current_scene}/depth_{self.scene_version}.data"
        # )
        # current_rgb_path = (
        #     self.data_path + f"{self.current_scene}/rgb_{self.scene_version}.png"
        # )
        # # Load & process data
        # current_rgb_image = self.load_rgb_data(current_rgb_path)
        # height, width, _ = current_rgb_image.shape
        # current_depth_image = self.load_depth_data(current_depth_path, height, width)
        # current_depth_image = self.process_depth_data(current_depth_image)
        # # set start location to center of image
        # # TODO: find object if not in center
        # obs_shape = current_depth_image.shape
        # start_location = [obs_shape[0] // 2, obs_shape[1] // 2]
        # return current_depth_image, current_rgb_image, start_location

    def get_3d_scene_point_cloud(self):
        """Turn 2D depth image into 3D pointcloud using DepthTo3DLocations.

        This point cloud is used to estimate the sensor displacement in 3D space
        between two subsequent steps. Without this we get displacements in pixel
        space which does not work with our 3D models.

        Returns:
            current_scene_point_cloud: The 3D scene point cloud.
            current_sf_scene_point_cloud: The 3D scene point cloud in sensor frame.
        """
        agent_id = "agent_01"
        sensor_id = "patch_01"
        obs = {agent_id: {sensor_id: {"depth": self.current_depth_image}}}
        rotation = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        state = {
            agent_id: {
                "sensors": {
                    sensor_id + ".depth": {
                        "rotation": rotation,
                        "position": np.array([0, 0, 0]),
                    }
                },
                "rotation": rotation,
                "position": np.array([0, 0, 0]),
            }
        }

        # Apply gaussian smoothing transform to depth image
        # Uncomment line below and add import, if needed
        # transform = GaussianSmoothing(agent_id=agent_id, sigma=2, kernel_width=3)
        # obs = transform(obs, state=state)

        transform = DepthTo3DLocations(
            agent_id=agent_id,
            sensor_ids=[sensor_id],
            resolutions=[self.current_depth_image.shape],
            world_coord=True,
            zooms=1,
            # hfov of iPad front camera from
            # https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/Cameras/Cameras.html
            # TODO: determine dynamically from which device is sending data
            hfov=54.201,
            get_all_points=True,
            use_semantic_sensor=False,
            depth_clip_sensors=(0,),
            clip_value=1.1,
        )
        obs_3d = transform(obs, state=state)
        current_scene_point_cloud = obs_3d[agent_id][sensor_id]["semantic_3d"]
        image_shape = self.current_depth_image.shape
        current_scene_point_cloud = current_scene_point_cloud.reshape(
            (image_shape[0], image_shape[1], 4)
        )
        current_sf_scene_point_cloud = obs_3d[agent_id][sensor_id]["sensor_frame_data"]
        current_sf_scene_point_cloud = current_sf_scene_point_cloud.reshape(
            (image_shape[0], image_shape[1], 4)
        )
        self.world_camera = obs_3d[agent_id][sensor_id]["world_camera"]
        return current_scene_point_cloud, current_sf_scene_point_cloud

    def get_3d_coordinates_from_pixel_indices(self, pixel_idx):
        """Retrieve 3D coordinates of a pixel.

        Returns:
            The 3D coordinates of the pixel.
        """
        [i, j] = pixel_idx
        loc_3d = np.array(self.current_scene_point_cloud[i, j, :3])
        return loc_3d

    def get_move_area(self):
        """Calculate area in which patch can move on the image.

        Returns:
            The move area.
        """
        obs_shape = self.current_depth_image.shape
        half_patch_size = self.patch_size // 2 + 1
        move_area = np.array(
            [
                [half_patch_size, obs_shape[0] - half_patch_size],
                [half_patch_size, obs_shape[1] - half_patch_size],
            ]
        )
        return move_area

    def get_next_loc(self, action_name, amount):
        """Calculate next location in pixel space given the current action.

        Returns:
            The next location in pixel space.
        """
        new_loc = np.array(self.current_loc)
        if action_name == "look_up":
            new_loc[0] -= amount
        elif action_name == "look_down":
            new_loc[0] += amount
        elif action_name == "turn_left":
            new_loc[1] -= amount
        elif action_name == "turn_right":
            new_loc[1] += amount
        else:
            logging.error(f"{action_name} is not a valid action, not moving.")
        # Make sure location stays within move area
        if new_loc[0] < self.move_area[0][0]:
            new_loc[0] = self.move_area[0][0]
        elif new_loc[0] > self.move_area[0][1]:
            new_loc[0] = self.move_area[0][1]
        if new_loc[1] < self.move_area[1][0]:
            new_loc[1] = self.move_area[1][0]
        elif new_loc[1] > self.move_area[1][1]:
            new_loc[1] = self.move_area[1][1]
        return new_loc

    def get_image_patch(self, loc):
        """Extract 2D image patch from a location in pixel space.

        Returns:
            depth_patch: The depth patch.
            rgb_patch: The rgb patch.
            depth3d_patch: The depth3d patch.
            sensor_frame_patch: The sensor frame patch.
        """
        loc = np.array(loc, dtype=int)
        x_start = loc[0] - self.patch_size // 2
        x_stop = loc[0] + self.patch_size // 2
        y_start = loc[1] - self.patch_size // 2
        y_stop = loc[1] + self.patch_size // 2
        depth_patch = self.current_depth_image[x_start:x_stop, y_start:y_stop]
        rgb_patch = self.current_rgb_image[x_start:x_stop, y_start:y_stop]
        depth3d_patch = self.current_scene_point_cloud[x_start:x_stop, y_start:y_stop]
        depth_shape = depth3d_patch.shape
        depth3d_patch = depth3d_patch.reshape(
            (depth_shape[0] * depth_shape[1], depth_shape[2])
        )
        sensor_frame_patch = self.current_sf_scene_point_cloud[
            x_start:x_stop, y_start:y_stop
        ]
        sensor_frame_patch = sensor_frame_patch.reshape(
            (depth_shape[0] * depth_shape[1], depth_shape[2])
        )

        assert (
            depth_patch.shape[0] * depth_patch.shape[1]
            == self.patch_size * self.patch_size
        ), f"Didn't extract a patch of size {self.patch_size}"
        return depth_patch, rgb_patch, depth3d_patch, sensor_frame_patch

