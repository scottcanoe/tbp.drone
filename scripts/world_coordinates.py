"""Development/Exploratory code for mapping observations to world coordinates."""

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quaternion as qt

from tbp.drone.src.environment import DroneImageEnvironment

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/potted_meat_can_v1"


def compute_agent_states():
    """Compute the agent states for the synthetic data."""
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
    rotation_yaw_delta = 360 / n_stepisodes
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

        hfov = 90
        new = False
        for i, zoom in enumerate(zooms):
            # Pinhole camera, focal length fx = fy
            h, w = resolutions[i]

            if new:
                hvov_x = 70
                hfov_y = 50
                # fx =
                cx = self.cx
                cy = self.cy
                fx = fy = focal_length
                k = np.array(
                    [
                        [1.0 / fx, 0.0, cx, 0.0],
                        [0.0, 1 / fy, cy, 0.0],
                        [0.0, 0.0, 1.0, 0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            else:
                fx = fy = np.tan(hfov / 2.0) / zoom
                fx, fy = 1 / fx, 1 / fy
                k = np.array(
                    [
                        [fx, 0.0, 0.0, 0.0],
                        [0.0, fy, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            print(f"fx: {fx}, fy: {fy}, k: \n{k}")
            # Adjust fy for aspect ratio
            self.h.append(h)
            self.w.append(w)

            # Intrinsic matrix, K
            # Assuming skew is 0 for pinhole camera and center at (0,0)

            # Inverse K
            self.inv_k.append(np.linalg.inv(k))

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
            observations[self.agent_id][sensor_id]["xyz"] = xyz

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


if __name__ == "__main__":
    """Testing/Development code for putting data into world coordinates."""

    agent_positions, agent_rotations = compute_agent_states()

    env = DroneImageEnvironment(data_path="potted_meat_can_v1")
    all_xyz = []
    all_colors = []
    counter = 0
    tform = DroneDepthTo3DLocations(
        agent_id="agent_id_0",
        sensor_ids=["view_finder"],
        resolutions=[(720, 960)],
        focal_length=920.0,
        zooms=1.0,
    )

    for stepisode in [0, 1, 2, 3]:
        data = env._load_stepisode_data(stepisode)
        stepisode_dir = env.data_path / f"{stepisode}"
        image = data["image"]
        agent_state = data["agent_state"]
        depth = data["depth"]
        bboxes = data["bbox"]

        # Update agent state
        agent = env._agent
        agent.position = agent_positions[stepisode]
        agent.rotation = agent_rotations[stepisode]

        # Update sensor data
        view_finder = agent.sensors["view_finder"]
        view_finder.rgb = image
        view_finder.rgba = image / 255.0
        alpha = np.ones((*view_finder.rgb.shape[:-1], 1))
        view_finder.rgba = np.concatenate([view_finder.rgba, alpha], axis=-1)
        view_finder.depth = depth

        obs_in = agent.observation_dict()
        state_in = agent.state_dict()
        print(f"stepisode {stepisode}")
        print("=======================")
        print(state_in)
        print("======================")
        obs_out = tform(obs_in, state_in)

        xyz = obs_out["agent_id_0"]["view_finder"]["semantic_3d"][:, 0:3]
        rgba = obs_out["agent_id_0"]["view_finder"]["rgba"]
        colors = rgba.reshape(4, -1).T

        # Semantic Masking
        in_bbox = np.zeros_like(depth, dtype=bool)
        bbox = bboxes["object"]
        x1, y1, x2, y2 = bbox
        in_bbox[y1:y2, x1:x2] = True
        in_range = np.ones_like(depth, dtype=bool)
        in_range[depth < env.depth_range[0]] = False
        in_range[depth > env.depth_range[1]] = False
        semantic = in_range & in_bbox
        semantic_1d = semantic.reshape(-1)
        inds = np.where(semantic_1d)[0]
        xyz = xyz[inds]
        colors = colors[inds]

        all_xyz.append(xyz)
        all_colors.append(colors)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    scatter_colors = ["black", "red", "blue", "yellow"]

    for i, xyz in enumerate(all_xyz):
        colors = all_colors[i]
        xyz = xyz[::50]
        colors = colors[::50]
        X = xyz[:, 0]
        Y = xyz[:, 1]
        Z = xyz[:, 2]
        ax.scatter(X, Z, Y, c=colors, s=10, alpha=0.1)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_box_aspect([1, 1, 1])
    plt.show()
