import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from tbp.drone.src.dataloader import DroneImageDataLoader, DroneStreamDataLoader
from tbp.drone.src.dataset import DroneEnvironmentDataset
from tbp.drone.src.environment import (
    DroneEnvironment,
    DroneImageEnvironment,
    DroneStreamEnvironment,
)


@dataclass
class DroneStreamEnvInitArgs:
    """Args for DroneStreamEnvironment"""

    patch_size: int = field(default=64)


@dataclass
class DroneImageEnvInitArgs:
    """Args for DroneImageEnvironment"""

    patch_size: int = field(default=64)
    data_path: Optional[os.PathLike] = field(default=None)


@dataclass
class DroneStreamDatasetArgs:
    env_init_func: Callable = field(default=DroneStreamEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: DroneStreamEnvInitArgs().__dict__
    )
    transform = None
    rng = None

    def __post_init__(self):
        self.transform = []
        #     DepthTo3DLocations(
        #         agent
        #         resolution=(720, 960),  # DJI Tello camera resolution
        #         focal_length_pixels=1825.1,  # Calculated from physical parameters
        #         optical_center=(480.0, 360.0),  # Half of resolution
        #         get_all_points=False,  # Only get object points
        #     )
        # ]


@dataclass
class DroneImageDatasetArgs:
    env_init_func: Callable = field(default=DroneImageEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: DroneImageEnvInitArgs().__dict__
    )
    transform = None
    rng = None

    def __post_init__(self):
        self.transform = []
        #     DepthTo3DLocations(
        #         agent
        #         resolution=(720, 960),  # DJI Tello camera resolution
        #         focal_length_pixels=1825.1,  # Calculated from physical parameters
        #         optical_center=(480.0, 360.0),  # Half of resolution
        #         get_all_points=False,  # Only get object points
        #     )
        # ]


@dataclass
class DroneImageDataloaderArgs:
    pass
