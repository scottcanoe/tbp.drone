# The below class models after PatchViewFinderMountHabitatDatasetArgs
# Located: https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/simulators/habitat/configs.py#L165

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List

from tbp.drone.src.environment import DroneEnvironment

# from tbp.drone.src.vision import DroneDepthTo3DLocations
from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
from tbp.monty.frameworks.environments.embodied_data import EnvironmentDataset


class DroneEnvironmentDataset(EnvironmentDataset):
    def __init__(self, env_init_func, env_init_args, rng, transform=None):
        super().__init__(env_init_func, env_init_args, rng, transform)

    def reset(self):
        # below is just a copy and paste of the parent class
        observation = self.env.reset()  # starts the drone
        state = self.env.get_state()  # returns {'agent_id_0': {'position': [...], 'rotation': [...], 'velocity': [...]}}
        # if self.transform is not None:
        #     observation = self.apply_transform(self.transform, observation, state)
        return observation, state
