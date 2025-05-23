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
import numpy.typing as npt
import quaternion as qt
from djitellopy import Tello
from scipy.spatial.transform import Rotation

from tbp.drone.src.actions import (
    Action,
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
from tbp.drone.src.dataloader import DroneDataLoader
from tbp.drone.src.drone_pilot import DronePilot
from tbp.drone.src.environment import DroneEnvironment, DroneImageEnvironment
from tbp.drone.src.spatial import (
    as_signed_angle,
    as_unsigned_angle,
    compute_relative_angle,
    pitch_roll_yaw_to_quaternion,
    quaternion_to_rotation,
    reorder_quat_array,
)
from tbp.drone.src.vision.depth_processing.depth_estimator import DepthEstimator
from tbp.drone.src.vision.depth_processing.object_segmenter import ObjectSegmenter

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/potted_meat_can_v1"


OCLOCK_TO_STEPISODE = {
    6: 0,
    5: 1,
    4: 2,
    3: 3,
    2: 4,
    1: 5,
    12: 6,
    11: 7,
    10: 8,
    9: 9,
    8: 10,
    7: 11,
}


def add_images(
    src_dir: os.PathLike,
    dst_dir: os.PathLike,
) -> None:
    monty_data_dir = Path(os.environ.get("MONTY_DATA", "~/tbp/data/")).expanduser()
    drone_data_dir = monty_data_dir / "worldimages/drone"
    src_dir = drone_data_dir / src_dir
    dst_dir = drone_data_dir / dst_dir

    for oclock, stepisode in OCLOCK_TO_STEPISODE.items():
        src_image_path = src_dir / f"spam_{oclock}oclock.png"
        src_state_path = src_dir / f"spam_{oclock}oclock_state.txt"

        dst_subdir = dst_dir / f"{stepisode}"
        dst_subdir.mkdir(parents=True, exist_ok=True)
        dst_image_path = dst_subdir / "image.png"
        dst_state_path = dst_subdir / "drone_state.json"

        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_state_path, dst_state_path)


def compute_agent_states():
    n_stepisodes = 12
    radius = 0.22
    height = 0.05

    stepisodes = np.arange(n_stepisodes)
    # For positions
    location_radians_delta = 2 * np.pi / n_stepisodes
    rotation_radians = np.mod(
        np.pi / 2 + location_radians_delta * stepisodes, 2 * np.pi
    )

    # For rotations
    rotation_yaw_delta = -360 / n_stepisodes
    rotation_yaws = rotation_yaw_delta * np.arange(12)

    positions = []
    rotations = []
    positions_x = -radius * np.cos(rotation_radians)
    positions_z = radius * np.sin(rotation_radians)
    for i in range(n_stepisodes):
        pos = np.array([positions_x[i], height, positions_z[i]])
        positions.append(pos)
        quat = pitch_roll_yaw_to_quaternion(0, 0, rotation_yaws[i])
        rotations.append(quat)

    return positions, rotations


def plot_agent_states(positions, rotations):
    for i, pos in enumerate(positions):
        x, y, z = 100 * pos
        print(f"Position {i}: ({x:.1f}, {y:.1f}, {z:.1f}) cm")

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.invert_xaxis()
    positions_x = np.array([p[0] for p in positions])
    positions_z = np.array([p[2] for p in positions])
    ax.plot(positions_x, positions_z, "rx")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    lim = 0.3
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    arrow_length = 0.05

    for i, pos in enumerate(positions):
        x, y, z = pos
        ax.annotate(f"{i}", (x, z), xytext=(5, 5), textcoords="offset points")
        quat = rotations[i]
        rot = quaternion_to_rotation(quat)
        mat = rot.as_matrix()

        origin = np.array([pos[0], pos[2]])
        x_component = np.array([mat[0, 0], mat[0, 2]])
        y_component = np.array([mat[2, 0], mat[2, 2]])

        x_end = origin + x_component * arrow_length
        y_end = origin + y_component * arrow_length
        ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], "r-")
        ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], "b-")

    plt.show()


def add_agent_positions_and_rotations():
    positions, rotations = compute_agent_states()
    for i in range(12):
        stepisode_dir = DATA_PATH / f"{i}"
        path = stepisode_dir / "agent_state.json"
        if path.exists():
            with open(path, "r") as f:
                info = json.load(f)
        else:
            info = {}
        info["agent_position"] = positions[i].tolist()
        info["agent_rotation"] = qt.as_float_array(rotations[i]).tolist()
        with open(path, "w") as f:
            json.dump(info, f)


def add_depth():
    depth_estimator = DepthEstimator()
    data_path = Path("/Users/sknudstrup/tbp/data/worldimages/drone/potted_meat_can_v1")
    for i in range(12):
        stepisode_dir = data_path / f"{i}"
        image = imageio.imread(stepisode_dir / "image.png")
        depth, _ = depth_estimator.estimate_depth(image)
        npy_path = stepisode_dir / "depth.npy"
        np.save(npy_path, depth)


def add_bbox_annotations():
    src_dir = Path.home() / "Downloads/bbox_annotations"
    dst_dir = DATA_PATH
    for oclock, stepisode in OCLOCK_TO_STEPISODE.items():
        src_path = src_dir / f"spam_{oclock}oclock_annotations.json"
        with open(src_path, "r") as f:
            bboxes_in = json.load(f)
        keys = list(bboxes_in.keys())
        keys.remove("aruco")
        object_key = keys[0]
        bboxes_out = {
            "aruco": bboxes_in["aruco"],
            "object": bboxes_in[object_key],
        }
        dst_path = dst_dir / f"{stepisode}/bbox.json"
        with open(dst_path, "w") as f:
            json.dump(bboxes_out, f)


def add_summaries():
    env = DroneImageEnvironment(data_path="potted_meat_can_v1")
    for i in range(12):
        data = env._load_stepisode_data(i)
        stepisode_dir = env.data_path / f"{i}"
        image = data["image"]
        depth = data["depth"]
        bboxes = data["bbox"]

        fig, axes = plt.subplots(3, 2, figsize=(10, 10))

        ax = axes[0, 0]
        ax.imshow(image)
        ax.set_title("Image")
        ax.axis("off")

        ax = axes[0, 1]
        depth[depth < env.depth_range[0]] = np.nan
        depth[depth > env.depth_range[1]] = np.nan
        im = ax.imshow(depth, cmap="inferno")
        plt.colorbar(im, label="Depth")
        ax.axis("off")
        ax.set_title("Depth")

        ax = axes[1, 0]
        bbox = bboxes["object"]
        x1, y1, x2, y2 = bbox
        img = np.zeros_like(image)
        img[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        ax.imshow(img)
        ax.set_title("Object bbox")
        ax.axis("off")

        ax = axes[1, 1]
        bbox = bboxes["aruco"]
        x1, y1, x2, y2 = bbox
        img = np.zeros_like(image)
        img[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        ax.imshow(img)
        ax.set_title("Aruco bbox")
        ax.axis("off")

        ax = axes[2, 0]
        bbox = bboxes["object"]
        x1, y1, x2, y2 = bbox
        is_in_depth_range = ~np.isnan(depth)
        is_in_object_bbox = np.zeros_like(depth, dtype=bool)
        is_in_object_bbox[y1:y2, x1:x2] = True
        is_valid = is_in_depth_range & is_in_object_bbox
        img = image.copy()
        img[~is_valid] = (0, 0, 0)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Object (depth-clipped + bbox)")

        ax = axes[2, 1]
        bbox = bboxes["aruco"]
        x1, y1, x2, y2 = bbox
        is_in_depth_range = ~np.isnan(depth)
        is_in_object_bbox = np.zeros_like(depth, dtype=bool)
        is_in_object_bbox[y1:y2, x1:x2] = True
        is_valid = is_in_depth_range & is_in_object_bbox
        img = image.copy()
        img[~is_valid] = (0, 0, 0)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Aruco (depth-clipped + bbox)")

        summary_path = stepisode_dir / "summary.png"
        fig.savefig(summary_path)
        plt.show()
