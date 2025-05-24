import datetime
import json
import logging
import os
import pprint as pp
import shutil
import subprocess as sp
import threading
import time
import warnings
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation

from tbp.drone.src.environment import DroneImageEnvironment
from tbp.drone.src.spatial import (
    as_signed_angle,
    as_unsigned_angle,
    compute_relative_angle,
    pitch_roll_yaw_to_quaternion,
    quaternion_to_rotation,
    reorder_quat_array,
)
from tbp.drone.src.utils import as_rgba

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/potted_meat_can_v4"


def axes3d_set_aspect_equal(ax) -> None:
    """Set equal aspect ratio for 3D axes."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Get the max range
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    half_max_range = max(x_range, y_range, z_range) / 2

    # Find midpoints
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set new limits
    ax.set_xlim([x_middle - half_max_range, x_middle + half_max_range])
    ax.set_ylim([y_middle - half_max_range, y_middle + half_max_range])
    ax.set_zlim([z_middle - half_max_range, z_middle + half_max_range])

    # Set aspect ratio.
    ax.set_box_aspect([1, 1, 1])


def draw_origin(ax, length=0.1, **kwargs):
    """
    Draw origin
    """
    kw = kwargs.copy()
    kw["color"] = kw.get("color", "k")
    kw["s"] = kw.get("s", 100)

    ax.scatter(0, 0, 0, **kw)
    ax.quiver(0, 0, 0, 1, 0, 0, color="r", length=length)
    ax.quiver(0, 0, 0, 0, 1, 0, color="g", length=length)
    ax.quiver(0, 0, 0, 0, 0, 1, color="b", length=length)


def draw_agent(ax, pos, rot, length=0.1, **kwargs):
    """
    Draw agent positions
    """

    rot = quaternion_to_rotation(rot)
    mat = rot.as_matrix()
    colors = ["red", "green", "blue"]
    for i in range(3):
        ax.quiver(
            *pos,
            *mat[:, i],
            color=colors[i],
            length=length,
        )
    kw = kwargs.copy()
    kw["color"] = kw.get("color", "k")
    kw["s"] = kw.get("s", 10)
    ax.scatter(pos[0], pos[1], pos[2], **kw)


class DroneDepthTo3DLocations:
    """Transform RGB image into 3D point cloud with semantic labels.

    This class takes an RGB image and transforms it into a 3D point cloud where each point
    has both spatial coordinates (x, y, z) and a semantic label. It uses DepthAnything V2
    for depth estimation and SAM for object segmentation.

    The output is a numpy array with shape (N, 4) where N is the number of points and
    each point has format [x, y, z, semantic_id].

    Attributes:
        resolution: Camera resolution (H, W)
        focal_length: Focal length in pixels (calculated from physical parameters)
        cx: Optical center x-coordinate in pixels (cx)
        cy: Optical center y-coordinate in pixels (cy)
        zoom: Camera zoom factor. Default 1.0 (no zoom)
        get_all_points: Whether to return all 3D coordinates or only object points
        max_depth: Maximum depth value to use for background points
    """

    def __init__(
        self,
        agent_id: str,
        sensor_ids: Iterable[str],
        resolutions: Iterable[Tuple[int, int]],
        zooms: float = 1.0,
        focal_length: float = 920.0,
        optical_center: Tuple[float, float] = (
            459.904354,
            351.238301,
        ),  # cx, cy from calibration
        world_coord: bool = True,
        get_all_points: bool = True,
        max_depth: float = 0.4,
    ):
        """Initialize the 3D point cloud generator.

        Args:
            resolution: Image resolution as (height, width) for Tello camera (720p)
            focal_length_pixels: Focal length in pixels (average of calibrated fx and fy)
            optical_center: Optical center in pixels (cx, cy) from calibration
            zoom: Camera zoom factor
            get_all_points: If True, return all points including background
            max_depth: Maximum depth value for background points
            depth_model_path: Path to depth model weights
            sam_model_path: Path to SAM model weights
        """
        self.agent_id = agent_id
        self.sensor_ids = sensor_ids
        self.resolutions = resolutions
        self.focal_length = focal_length
        self.cx, self.cy = optical_center

        self.get_all_points = get_all_points
        self.max_depth = max_depth
        self.world_coord = world_coord
        self.needs_rng = False

        self.inv_k = []
        self.h, self.w = [], []

        if isinstance(zooms, (int, float)):
            zooms = [zooms] * len(sensor_ids)
        self.zooms = zooms

        self.method = self.get_intrinsic_matrix_2
        for i in range(len(sensor_ids)):
            h, w = self.resolutions[i]
            k, inv_k = self.method(i)
            self.h.append(h)
            self.w.append(w)
            self.inv_k.append(inv_k)

    def get_intrinsic_matrix_1(self, sensor_id: 0, hfov=120):
        h, w = self.resolutions[sensor_id]
        zoom = self.zooms[sensor_id]
        hfov = float(hfov * np.pi / 180.0)
        fx = np.tan(hfov / 2.0) / zoom
        fy = fx * h / w
        fx, fy = 1 / fx, 1 / fy
        cx = cy = 0
        k = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return k, np.linalg.inv(k)

    def get_intrinsic_matrix_2(self, sensor_id: 0, hfov=90):
        h, w = self.resolutions[sensor_id]
        zoom = self.zooms[sensor_id]
        cx = self.cx
        cy = self.cy
        fx = fy = self.focal_length
        k = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return k, np.linalg.inv(k)

    def __call__(self, observations: dict, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert depth image to 3D point cloud.

        Args:
            image: Input RGB image path or numpy array
            input_points: List of (x,y) coordinates to segment
            input_labels: List of labels for input_points (1=foreground, 0=background)

        Returns:
            Nx4 array of 3D points and semantic labels
        """
        for i, sensor_id in enumerate(self.sensor_ids):
            agent_obs = observations[self.agent_id][sensor_id]
            depth_patch = agent_obs["depth"]
            if self.method == self.get_intrinsic_matrix_1:
                # Approximate true world coordinates
                x, y = np.meshgrid(
                    np.linspace(-1, 1, self.w[i]), np.linspace(1, -1, self.h[i])
                )
                x = x.reshape(1, self.h[i], self.w[i])
                y = y.reshape(1, self.h[i], self.w[i])

                # Unproject 2D camera coordinates into 3D coordinates relative to the agent
                depth = depth_patch.reshape(1, self.h[i], self.w[i])
                xyz = np.vstack((x * depth, y * depth, -depth, np.ones(depth.shape)))
                xyz = xyz.reshape(4, -1)
                xyz = np.matmul(self.inv_k[i], xyz)
                sensor_frame_data = xyz.T.copy()
                print(f"------ xyz shape: {xyz.shape}")
                print(f"------ xyz 0-5: {xyz[:, :5]}")
            elif self.method == self.get_intrinsic_matrix_2:
                xyz = depth_to_point_cloud(-1 * depth_patch)
                xyz = xyz.reshape(-1, 3)
                xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
                xyz = xyz.T
                print(f"------ xyz shape: {xyz.shape}")
                sensor_frame_data = xyz.T.copy()
            else:
                raise ValueError(f"Invalid method: {self.method}")

            if self.world_coord and state is not None:
                # Get agent and sensor states from state dictionary
                agent_state = state[self.agent_id]
                depth_state = agent_state["sensors"][sensor_id + ".depth"]
                agent_rotation = agent_state["rotation"]
                agent_rotation_matrix = qt.as_rotation_matrix(agent_rotation)
                agent_position = agent_state["position"]
                sensor_rotation = depth_state["rotation"]
                sensor_position = depth_state["position"]
                # --- Apply camera transformations to get world coordinates ---
                # Combine body and sensor rotation (since sensor rotation is relative to
                # the agent this will give us the sensor rotation in world coordinates)
                sensor_rotation_rel_world = agent_rotation * sensor_rotation
                # Calculate sensor position in world coordinates -> sensor_position is
                # in the agent's coordinate frame, so we need to rotate it first by
                # agent_rotation_matrix and then add it to the agent's position
                rotated_sensor_position = agent_rotation_matrix @ sensor_position
                sensor_translation_rel_world = agent_position + rotated_sensor_position
                # Apply the rotation and translation to get the world coordinates
                rotation_matrix = qt.as_rotation_matrix(sensor_rotation_rel_world)
                world_camera = np.eye(4)
                world_camera[0:3, 0:3] = rotation_matrix
                world_camera[0:3, 3] = sensor_translation_rel_world
                xyz = np.matmul(world_camera, xyz)

                # Add sensor-to-world coordinate frame transform, used for point-normal
                # extraction. View direction is the third column of the matrix.
                observations[self.agent_id][sensor_id]["world_camera"] = world_camera

            # Extract 3D coordinates of detected objects (semantic_id != 0)
            # semantic = surface_patch.reshape(1, -1)
            semantic = np.ones_like(depth_patch, dtype=int).reshape(1, -1)
            if self.get_all_points:
                semantic_3d = xyz.transpose(1, 0)
                semantic_3d[:, 3] = semantic[0]
                sensor_frame_data[:, 3] = semantic[0]

                # Add point-cloud data expressed in sensor coordinate frame. Used for
                # point-normal extraction
                observations[self.agent_id][sensor_id]["sensor_frame_data"] = (
                    sensor_frame_data
                )
            else:
                detected = semantic.any(axis=0)
                xyz = xyz.transpose(1, 0)
                semantic_3d = xyz[detected]
                semantic_3d[:, 3] = semantic[0, detected]

            # Add transformed observation to existing dict. We don't need to create
            # a deepcopy because we are appending a new observation
            observations[self.agent_id][sensor_id]["semantic_3d"] = semantic_3d

        return observations


def depth_to_point_cloud(depth_image):
    height, width = depth_image.shape
    fx = 920
    fy = 920
    cx = 460
    cy = 351

    # Create a meshgrid of pixel coordinates (u, v)
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # Compute 3D coordinates
    z = depth_image
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # Stack into point cloud shape (H, W, 3)
    points = np.stack((x, y, z), axis=-1)

    return points  # Shape: (H, W, 3)


def sample_every_other(step, *arrays):
    return [array[::step] for array in arrays]


env = DroneImageEnvironment(data_path="potted_meat_can_v4")
agent = env.agent
view_finder = agent.sensors["view_finder"]

# Data options
stepisodes = list(range(12))
update_sensor_data = True
update_agent_state = True
recenter_depth = True

# Plotting options
lim = 0.3

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection="3d")
draw_origin(ax)

tform = DroneDepthTo3DLocations(
    agent_id="agent_id_0",
    sensor_ids=["view_finder"],
    resolutions=[(720, 960)],
    focal_length=920.0,
    zooms=1.0,
)


all_xyz = []
all_colors = []
for i, stepisode in enumerate(stepisodes):
    stepisode_dir = env.data_path / f"{stepisode}"
    data = env.load_stepisode_data(stepisode)
    agent_state = data["agent_state"]

    # Update sensor data
    if i == 0 or update_sensor_data:
        view_finder.rgb = data["image"]
        view_finder.rgba = as_rgba(view_finder.rgb) / 255
        view_finder.depth = data["depth"]
        object_mask = data["object_mask"]
        # view_finder.depth *= -1
        if recenter_depth:
            depth_vals = view_finder.depth[object_mask]
            delta = 0.2 - np.mean(depth_vals)
            view_finder.depth += delta

    # Update agent state
    if i == 0 or update_agent_state:
        agent.position = data["agent_state"]["position"]
        agent.rotation = data["agent_state"]["rotation"]

    draw_agent(ax, agent.position, agent.rotation)
    # agent.rotation = agent.rotation.inverse()

    obs_in = agent.observation_dict()
    state_in = agent.state_dict()
    obs_out = tform(obs_in, state_in)

    xyz = obs_out["agent_id_0"]["view_finder"]["semantic_3d"][:, 0:3]
    rgba = obs_out["agent_id_0"]["view_finder"]["rgba"]
    colors = rgba.reshape(4, -1).T

    # Semantic Masking
    semantic_1d = object_mask.reshape(-1)
    inds = np.where(semantic_1d)[0]
    xyz = xyz[inds]
    colors = colors[inds]

    all_xyz.append(xyz)
    all_colors.append(colors)


color_method = "fixed"
subsample_method = "every_other"


for i, xyz in enumerate(all_xyz):
    if color_method == "fixed":
        cmap = plt.cm.inferno
        color = cmap(np.linspace(0, 1, len(all_xyz)))[i]
        colors = np.broadcast_to(color, (len(xyz), 4))

    elif color_method == "true":
        colors = all_colors[i]
    else:
        raise ValueError(f"Invalid color method: {color_method}")

    if subsample_method == "every_other":
        xyz, colors = sample_every_other(25, xyz, colors)
    else:
        raise ValueError(f"Invalid subsample method: {subsample_method}")

    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    ax.scatter(X, Y, Z, c=colors, s=10, alpha=0.2)
    ax.view_init(elev=-42, azim=90)  # orient Z into screen


ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])

# Adjust camera view: elev = elevation (Y rotation), azim = azimuth (Z rotation)
ax.view_init(elev=-42, azim=90)  # orient Z into screen
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
