# The below class models after PatchViewFinderMountHabitatDatasetArgs
# Located: https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/simulators/habitat/configs.py#L165

from dataclasses import dataclass, field
from typing import Callable, Dict
from tbp.drone.src.environment import DroneEnvironment

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
            EstimateDepthWithDepthToAnything(),
            DroneDepthTo3DLocations()
        ]

class EstimateDepthWithDepthToAnything:
    pass

class DroneDepthTo3DLocations:
    pass
