import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union
from djitellopy import Tello
from dataclasses import dataclass

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]



class DroneAgent:
    """Base class for drone agents in simulation."""
    def __init__(self, agent_id: str, positions: Vector3 = (0.0, 0.0, 0.0), rotation: Quaternion = (1.0, 0.0, 0.0, 0.0), velocity: Vector3 = (0.0, 0.0, 0.0)):
        self.agent_id = agent_id
        self._position = np.array(positions)  # x, y, z in meters
        self._rotation = np.array(rotation)  # quaternion
        self._velocity = np.array(velocity)  # velocity in m/s

    def initialize(self, sim):
        """Initialize agent with simulator instance."""
        self.sim = sim

    def process_observations(self, obs):
        """Process raw observations from sensors."""
        return obs

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = np.array(value)

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.array(value)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = np.array(value)

@dataclass
class DroneAgentConfig:
    """Agent configuration used by :class:`HabitatEnvironment`."""

    agent_type: List[DroneAgent]
    agent_args: Dict
class DroneSim:
    """DJI Tello drone simulator interface.
    
    This class provides a simulation interface for the DJI Tello drone that mirrors
    the structure of HabitatSim while adapting it for drone-specific functionality.
    
    Attributes:
        agents: List of DroneAgents to simulate
        scene_id: Identifier for the scene/environment
        seed: Random seed for reproducibility
        data_path: Path to simulation data/assets
        enable_physics: Whether to enable physics simulation
    """
    def __init__(
        self,
        agents: List[DroneAgent],
        scene_id: Optional[str] = None,
        seed: int = 42,
        data_path: Optional[str] = None,
        enable_physics: bool = True,
    ):
        self.agents = agents
        self.scene_id = scene_id
        self.seed = seed
        self.data_path = data_path
        self.enable_physics = enable_physics
        
        # Initialize random number generator
        self.np_rng = np.random.default_rng(seed)
        
        # Setup action space based on Tello capabilities
        self._action_space = {
            "takeoff",
            "land",
            "move_forward",
            "move_back",
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "rotate_clockwise",
            "rotate_counter_clockwise",
            "flip_forward",
            "flip_back",
            "flip_left",
            "flip_right",
            "stop",
            "send_xyz_speed",
            "set_yaw",
        }
        
        # Initialize agent tracking
        self._agent_id_to_index = {
            agent.agent_id: idx for idx, agent in enumerate(self.agents)
        }
        
        # Initialize simulated environment
        self._objects = {}  # Dictionary to track objects in scene
        self._object_counter = 0
        
        # Initialize each agent
        for agent in self.agents:
            agent.initialize(self)
        
        # Initialize Tello interface (but don't connect yet)
        self._tello = None

    def initialize_agent(self, agent_id: str, agent_state: Dict):
        """Update agent runtime state.
        
        Args:
            agent_id: Agent id of the agent to be updated
            agent_state: Agent state to update to
        """
        agent_index = self._agent_id_to_index[agent_id]
        agent = self.agents[agent_index]
        
        if "position" in agent_state:
            agent.position = agent_state["position"]
        if "rotation" in agent_state:
            agent.rotation = agent_state["rotation"]
        if "velocity" in agent_state:
            agent.velocity = agent_state["velocity"]

    def add_object(
        self,
        name: str,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        semantic_id: Optional[str] = None,
    ):
        """Add an object to the simulation environment.
        
        Args:
            name: Object identifier
            position: Object position (x, y, z) in meters
            rotation: Object rotation quaternion
            semantic_id: Optional semantic identifier
        """
        obj = {
            "name": name,
            "position": np.array(position),
            "rotation": np.array(rotation),
            "semantic_id": semantic_id,
            "id": self._object_counter,
        }
        self._objects[self._object_counter] = obj
        self._object_counter += 1
        return obj

    def remove_all_objects(self):
        """Remove all objects from simulated environment."""
        self._objects.clear()
        self._object_counter = 0

    def get_action_space(self):
        """Returns a set with all available actions."""
        return set(self._action_space)

    def apply_action(self, action: Dict) -> Dict[str, Dict]:
        """Execute given action in the environment.
        
        Args:
            action: Dictionary containing action name and parameters
            
        Returns:
            Dictionary with observations grouped by agent_id
        """
        action_name = action.get("name")
        if action_name not in self._action_space:
            raise ValueError(f"Invalid action name: {action_name}")
            
        # Initialize Tello connection if needed
        if self._tello is None:
            self._tello = Tello()
            self._tello.connect(wait_for_state=True)
            
        # Execute action on Tello
        params = action.get("params", {})
        if action_name == "takeoff":
            self._tello.takeoff()
        elif action_name == "land":
            self._tello.land()
        elif action_name == "move_forward":
            distance = params.get("distance", 20)  # cm
            self._tello.move_forward(distance)
        elif action_name == "move_back":
            distance = params.get("distance", 20)  # cm
            self._tello.move_back(distance)
        elif action_name == "move_left":
            distance = params.get("distance", 20)  # cm
            self._tello.move_left(distance)
        elif action_name == "move_right":
            distance = params.get("distance", 20)  # cm
            self._tello.move_right(distance)
        elif action_name == "rotate_clockwise":
            degrees = params.get("degrees", 30)
            self._tello.rotate_clockwise(degrees)
        elif action_name == "rotate_counter_clockwise":
            degrees = params.get("degrees", 30)
            self._tello.rotate_counter_clockwise(degrees)
        elif action_name == "stop":
            self._tello.stop()
            
        # Get updated observations
        return self.get_observations()

    def get_observations(self) -> Dict[str, Dict]:
        """Get sensor observations.
        
        Returns:
            Dictionary with all sensor observations grouped by agent_id
        """
        if self._tello is None:
            return {}
            
        # Get Tello state
        state = {
            "battery": self._tello.get_battery(),
        }
        
        # Get camera frame if available
        try:
            frame = self._tello.get_frame_read().frame
            state["camera"] = frame
        except:
            pass
            
        # Process observations for each agent
        observations = defaultdict(dict)
        for agent in self.agents:
            observations[agent.agent_id] = agent.process_observations(state)
            
        return observations

    def get_state(self) -> Dict[str, Dict]:
        """Get agent and sensor states.
        
        Returns:
            Dictionary with agent poses and states
        """
        result = {}
        for agent in self.agents:
            result[agent.agent_id] = {
                "position": agent.position.tolist(),
                "rotation": agent.rotation.tolist(),
                "velocity": agent.velocity.tolist(),
            }
        return result

    def reset(self):
        """Reset the simulation to initial state."""
        # Land the drone if it's flying
        if self._tello is not None:
            try:
                self._tello.land()
            except:
                pass
                
        # Reset internal state
        self.remove_all_objects()
        for agent in self.agents:
            agent.position = np.array([0.0, 0.0, 0.0])
            agent.rotation = np.array([1.0, 0.0, 0.0, 0.0])
            agent.velocity = np.array([0.0, 0.0, 0.0])
            
        # Get initial observations
        return self.get_observations()

    def close(self):
        """Close simulator and release resources."""
        if self._tello is not None:
            try:
                self._tello.land()
                self._tello.end()
            except:
                pass
            self._tello = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    