# The below class models after PatchViewFinderMountHabitatDatasetArgs
# Located: https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/simulators/habitat/configs.py#L165

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List
from tbp.drone.src.environment import DroneEnvironment  
from tbp.drone.src.vision import DroneDepthTo3DLocations
from tbp.monty.frameworks.environments.embodied_data import EnvironmentDataset
from tbp.drone.src.dji_tello.simulator import DroneAgentConfig, DroneAgent

@dataclass
class DroneEnvInitArgs:
    """Args for DroneEnvironment"""
    agents: List[DroneAgentConfig] = field(default_factory=lambda: [DroneAgentConfig(agent_type=DroneAgent, agent_args={"agent_id": "drone1", "positions": (0.0, 0.0, 0.0)})])
    scene_id: int = field(default=1)
    seed: int = field(default=42)
    data_path: str = field(default=None)

@dataclass
class DroneDatasetArgs:
    env_init_func: Callable = field(default=DroneEnvironment)
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

class DroneEnvironmentDataset(EnvironmentDataset):
    def __init__(self, env_init_func, env_init_args, rng, transform=None):
        super().__init__(env_init_func, env_init_args, rng, transform)

    def reset(self):
        # below is just a copy and paste of the parent class
        observation = self.env.reset() # returns {}
        state = self.env.get_state() # returns {'drone1': {'position': [...], 'rotation': [...], 'velocity': [...]}}
        # if self.transform is not None:
        #     observation = self.apply_transform(self.transform, observation, state)
        return observation, state
