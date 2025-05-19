# The below class models after PatchViewFinderMountHabitatDatasetArgs
# Located: https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/simulators/habitat/configs.py#L165

from dataclasses import dataclass, field
from typing import Callable, Dict
from tbp.drone.src.environment import DroneEnvironment
from tbp.drone.src.vision import DroneDepthTo3DLocations

@dataclass
class DroneEnvInitArgs:
    """Args for DroneEnvironment"""
    pass

@dataclass
class DroneDatasetArgs:
    env_init_function: Callable = field(default=DroneEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: DroneEnvInitArgs().__dict__
    )
    transform = None
    rng = None

    def __post_init__(self):
        self.transform = [
            DroneDepthTo3DLocations(
                resolution=(720, 960),  # DJI Tello camera resolution
                focal_length_pixels=1825.1,  # Calculated from physical parameters
                optical_center=(480.0, 360.0),  # Half of resolution
                get_all_points=False  # Only get object points
            )
        ]

class EstimateDepthWithDepthToAnything:
    pass

class DroneDepthTo3DLocations:
    pass
