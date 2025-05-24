import datetime
import json
import os
import pprint as pp
import shutil
import subprocess as sp
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import quaternion as qt
from matplotlib.animation import FuncAnimation, PillowWriter
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
from tbp.drone.src.vision.depth_processing.depth_estimator import DepthEstimator

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/potted_meat_can_v4"


class StepisodeData:
    def __init__(self, data_dir: os.PathLike):
        self.data_dir = Path(data_dir).expanduser()

    @property
    def n_stepisodes(self) -> int:
        return len(self.get_dirs())

    def get_dirs(self) -> List[Path]:
        paths = list(self.data_dir.glob("*"))
        paths = [p for p in paths if p.is_dir() and p.name.isdigit()]
        steps = sorted([int(p.name) for p in paths])
        if not np.array_equal(steps, np.arange(len(steps))):
            raise ValueError(f"Stepisodes are not consecutive: {steps}")
        return [self.data_dir / f"{i}" for i in steps]

    def iterdirs(self) -> Iterable[Path]:
        for p in self.get_dirs():
            yield p

    def add_depth_maps(self):
        depth_estimator = DepthEstimator()
        for i, stepisode_dir in enumerate(self.iterdirs()):
            image = imageio.imread(stepisode_dir / "image.png")
            depth = depth_estimator(image)
            np.save(stepisode_dir / "depth.npy", depth)

    def add_rgba(self):
        for i, stepisode_dir in enumerate(self.iterdirs()):
            rgb = imageio.imread(stepisode_dir / "image.png")
            rgba = as_rgba(rgb)
            np.save(stepisode_dir / "rgba.npy", rgba)

    def add_agent_states(self):
        positions, rotations = self.compute_agent_states()
        for i, stepisode_dir in enumerate(self.iterdirs()):
            agent_state = {
                "position": positions[i].tolist(),
                "rotation": qt.as_float_array(rotations[i]).tolist(),
            }
            with open(stepisode_dir / "agent_state.json", "w") as f:
                json.dump(agent_state, f)

    def compute_agent_states(self, distance: float = 0.25, height: float = 0.05):
        n_stepisodes = self.n_stepisodes
        stepisodes = np.arange(n_stepisodes)
        # For positions
        location_radians_delta = 2 * np.pi / n_stepisodes
        rotation_radians = np.mod(
            np.pi / 2 + location_radians_delta * stepisodes, 2 * np.pi
        )

        # For rotations
        rotation_yaw_delta = 360 / n_stepisodes
        rotation_yaws = rotation_yaw_delta * np.arange(12)

        positions = []
        rotations = []
        positions_x = -distance * np.cos(rotation_radians)
        positions_z = distance * np.sin(rotation_radians)
        for i in range(n_stepisodes):
            pos = np.array([positions_x[i], height, positions_z[i]])
            positions.append(pos)
            quat = pitch_roll_yaw_to_quaternion(0, 0, rotation_yaws[i])
            rotations.append(quat)

        return positions, rotations

    def visualize_agent_states(
        self, save_path: Optional[os.PathLike] = None, from_file: bool = True
    ):
        def update(frame):
            ax.cla()  # Clear previous frame (optional if you're redrawing completely)

            """
            Draw origin
            """
            ax.scatter(0, 0, 0, color="k", s=50)
            ax.quiver(0, 0, 0, 1, 0, 0, color="r", label="X", length=0.1)
            ax.quiver(0, 0, 0, 0, 1, 0, color="g", label="Y", length=0.1)
            ax.quiver(0, 0, 0, 0, 0, 1, color="b", label="Z", length=0.1)

            """
            Draw agent positions
            """
            pos = positions[frame]
            rot = quaternion_to_rotation(rotations[frame])

            mat = rot.as_matrix()
            colors = ["red", "green", "blue"]
            for i in range(3):
                ax.quiver(
                    *pos,
                    *mat[:, i],
                    color=colors[i],
                    length=0.1,
                )
            ax.scatter(pos[0], pos[1], pos[2], color="k", s=10)

            lim = 0.3
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_zlim([-lim, lim])

            # Adjust camera view: elev = elevation (Y rotation), azim = azimuth (Z rotation)
            ax.view_init(elev=-42, azim=90)  # orient Z into screen
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            plt.legend()

            # Object image
            image_ax.cla()
            image_ax.imshow(images[frame])
            image_ax.axis("off")

            fig.suptitle(f"Frame {frame}")

            plt.show()

        if from_file:
            positions, rotations = [], []
            for stepisode_dir in self.iterdirs():
                with open(stepisode_dir / "agent_state.json", "r") as f:
                    agent_state = json.load(f)
                positions.append(np.array(agent_state["position"]))
                rotations.append(qt.from_float_array(np.array(agent_state["rotation"])))
        else:
            positions, rotations = StepisodeData(DATA_PATH).compute_agent_states()
        images = []
        for i in range(len(positions)):
            image = imageio.imread(DATA_PATH / f"{i}" / "image.png")
            images.append(image)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        image_ax = fig.add_subplot(1, 2, 2)

        anim = FuncAnimation(fig, update, frames=12, interval=500, repeat=True)
        if save_path is not None:
            anim.save(save_path, writer=PillowWriter(fps=1))
        else:
            plt.show()

    def add_object_masks(self, stepisode: int, depth_range: Tuple[float, float]):
        for i, subdir in enumerate(self.iterdirs()):
            object_mask, fig = self.make_object_mask(
                i, show=True, depth_range=depth_range
            )
            fig.savefig(subdir / "masks.png")
            np.save(subdir / "object_mask.npy", object_mask)

    def make_object_mask(
        self,
        stepisode: int,
        show: bool = False,
        depth_range: Optional[Tuple[float, float]] = None,
    ):
        if depth_range is None:
            depth_range = (-np.inf, np.inf)

        stepisode_dir = self.data_dir / f"{stepisode}"
        image = imageio.imread(stepisode_dir / "image.png")
        with open(stepisode_dir / "bbox.json", "r") as f:
            bbox = json.load(f)["object"]
        depth = np.load(stepisode_dir / "depth.npy")

        in_depth_range = np.ones_like(depth, dtype=bool)
        in_depth_range[depth < depth_range[0]] = False
        in_depth_range[depth > depth_range[1]] = False
        in_bbox = np.zeros_like(depth, dtype=bool)
        in_bbox[bbox[1] : bbox[3], bbox[0] : bbox[2]] = True
        object_mask = in_depth_range & in_bbox
        masked_image = image.copy()
        masked_image[~object_mask] = (0, 0, 0)
        if show:
            fig, axes = plt.subplots(3, 2, figsize=(10, 10))
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Image")

            axes[0, 1].imshow(depth)
            axes[0, 1].set_title("Depth")

            img = image.copy()
            img[~in_bbox] = (0, 0, 0)
            axes[1, 0].imshow(img)
            axes[1, 0].set_title("bbox masked")

            img = image.copy()
            img[~in_depth_range] = (0, 0, 0)
            axes[1, 1].imshow(img)
            axes[1, 1].set_title("depth masked")

            axes[2, 0].imshow(masked_image)
            axes[2, 0].set_title("Object Mask")

            ax = axes[2, 1]
            object_depth = depth[object_mask]
            ax.hist(object_depth, bins=100)
            ax.set_title("On-object Depth")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Count")

            for ax in axes.flatten()[:-1]:
                ax.axis("off")
            fig.suptitle(f"Stepisode {stepisode}: depth_range={depth_range}")
            plt.show()
        if show:
            return object_mask, fig
        else:
            return object_mask


dset = StepisodeData(DATA_PATH)

