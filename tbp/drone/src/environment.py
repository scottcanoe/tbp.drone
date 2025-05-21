from typing import List, Dict, Optional, Tuple
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment
from tbp.drone.src.dji_tello.simulator import DroneSim
from tbp.drone.src.dji_tello.simulator import DroneAgentConfig

class DroneEnvironment(EmbodiedEnvironment):
    def __init__(
        self,
        agents: List[DroneAgentConfig],
        scene_id: Optional[str] = None,
        seed: int = 42,
        data_path: Optional[str] = None,
    ):
        super().__init__()
        self.agents = []
        for config in agents:
            agent_type = config["agent_type"]
            agent_args = config["agent_args"]
            agent = agent_type(**agent_args)
            self.agents.append(agent)
            
        self.env = DroneSim(
            agents=self.agents,
            scene_id=scene_id,
            seed=seed,
            data_path=data_path,
        )

    @property
    def action_space(self):
        return self.env.action_space

    def add_object(
        self,
        name: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        semantic_id: Optional[str] = None,
    ):
        return self.env.add_object(name, position, rotation, semantic_id)

    def step(self, action) -> Dict[str, Dict]:
        return self.env.apply_action(action)

    def remove_all_objects(self):
        return self.env.remove_all_objects()

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()

    def get_state(self):
        return self.env.get_state()