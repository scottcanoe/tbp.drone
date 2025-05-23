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

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/"


def draw_rect(ax, image):
    h, w = image.shape[:2]
    center_x = w // 2
    center_y = h // 2
    rect_size = 50
    rect_x = center_x - rect_size // 2
    rect_y = center_y - rect_size // 2

    # Draw rectangle
    rect = plt.Rectangle(
        (rect_x, rect_y), rect_size, rect_size, fill=False, color="red", linewidth=2
    )
    ax.add_patch(rect)


def depthanything_to_meters(depth):
    """Map DepthAnything depth to meters."""
    slope = -0.06080131811283916
    intercept = 0.6446867000355985
    return slope * depth + intercept


# def fit_depthanything_to_ground_truth():
"""Fit a linear function to the depth map."""
depth_estimator = DepthEstimator()
path = DATA_PATH / "depth_estimation/image.png"
image = imageio.imread(path)
da_depth_map = depth_estimator(image)

x_pixels = np.array([330, 412, 482, 548, 602])
gt_depths = np.array([(25 + (i - 2) * 2.54 * np.sqrt(2)) / 100 for i in range(5)])
da_depths = np.array([da_depth_map[230, x] for x in x_pixels])

# slope, intercept = np.polyfit(est_depths, gt_depths, 1)
# new_depths = slope * est_depths + intercept
corrected_depths = depthanything_to_meters(da_depths)

# Plot results
# plt.figure(figsize=(8, 6))
# plt.scatter(da_depths, gt_depths, color="blue", label="Data points")
# plt.plot(da_depths, corrected_depths, color="red", label="Linear fit")
# plt.xlabel("Estimated depth")
# plt.ylabel("Ground truth depth")
# plt.legend()
# plt.grid(True)
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
ax = axes[0]
ax.imshow(image)
ax = axes[1]
ax.imshow(da_depth_map, cmap="inferno")
ax = axes[2]
corrected_depth_map = depthanything_to_meters(da_depth_map)
ax.imshow(corrected_depth_map, cmap="inferno")
plt.show()
