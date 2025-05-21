from numbers import Number

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.actions.actuator import Actuator
from tbp.monty.frameworks.environments.embodied_environment import ActionSpace

__all__ = [
    "Action",
    "DroneActionSpace",
    "TakeOff",
    "Land",
    "TurnLeft",
    "TurnRight",
    "SetHeight",
    "SetYaw",
    "MoveForward",
    "MoveBackward",
    "MoveLeft",
    "MoveRight",
    "MoveUp",
    "MoveDown",
]


class DroneActionSpace(tuple, ActionSpace):
    """Action space for 2D data environments."""

    def sample(self):
        return self.rng.choice(self)


class TakeOff(Action):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)

    def act(self, actuator: Actuator):
        actuator.actuate_takeoff(self)


class Land(Action):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)

    def act(self, actuator: Actuator):
        actuator.actuate_land(self)


class TurnLeft(Action):
    def __init__(self, agent_id: str, angle: Number):
        super().__init__(agent_id=agent_id)
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_turn_left(self)


class TurnRight(Action):
    def __init__(self, agent_id: str, angle: Number):
        super().__init__(agent_id=agent_id)
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_turn_right(self)


class SetYaw(Action):
    def __init__(self, agent_id: str, angle: Number):
        super().__init__(agent_id=agent_id)
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_set_yaw(self)


class MoveForward(Action):
    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_forward(self)


class MoveBackward(Action):
    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_backward(self)


class MoveLeft(Action):
    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_left(self)


class MoveRight(Action):
    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_right(self)


class MoveUp(Action):
    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_up(self)


class MoveDown(Action):
    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_down(self)

class SetHeight(Action):
    def __init__(self, agent_id: str, height: Number):
        super().__init__(agent_id=agent_id)
        self.height = height

    def act(self, actuator: Actuator):
        actuator.actuate_set_height(self)
