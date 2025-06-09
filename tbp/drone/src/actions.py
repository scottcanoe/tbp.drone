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
    "LookLeft",
    "LookRight",
    "LookUp",
    "LookDown",
    "NextImage",
]


class DroneActionSpace(tuple, ActionSpace):
    """Action space for 2D data environments."""

    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> None:
        return None

    def sample(self):
        raise NotImplementedError


class DroneAction(Action):
    """Base class for all drone actions. Hard-codes agent_id to "agent_id_0"."""

    agent_id: str = "agent_id_0"

    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> None:
        raise NotImplementedError


class TakeOff(DroneAction):
    def act(self, actuator: Actuator):
        actuator.actuate_takeoff(self)


class Land(DroneAction):
    def act(self, actuator: Actuator):
        actuator.actuate_land(self)


class TurnLeft(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_turn_left(self)


class TurnRight(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_turn_right(self)


class SetYaw(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_set_yaw(self)


class MoveForward(DroneAction):
    def __init__(self, distance: Number):
        super().__init__()
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_forward(self)


class MoveBackward(DroneAction):
    def __init__(self, distance: Number):
        super().__init__()
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_backward(self)


class MoveLeft(DroneAction):
    def __init__(self, distance: Number):
        super().__init__()
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_left(self)


class MoveRight(DroneAction):
    def __init__(self, distance: Number):
        super().__init__()
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_right(self)


class MoveUp(DroneAction):
    def __init__(self, distance: Number):
        super().__init__()
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_up(self)


class MoveDown(DroneAction):
    def __init__(self, distance: Number):
        super().__init__()
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_down(self)


class SetHeight(DroneAction):
    def __init__(self, height: Number):
        super().__init__()
        self.height = height

    def act(self, actuator: Actuator):
        actuator.actuate_set_height(self)


class LookLeft(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_look_left(self)


class LookRight(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_look_right(self)


class LookUp(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_look_up(self)


class LookDown(DroneAction):
    def __init__(self, angle: Number):
        super().__init__()
        self.angle = angle

    def act(self, actuator: Actuator):
        actuator.actuate_look_down(self)


class NextImage(DroneAction):
    def __init__(self):
        super().__init__()

    def act(self, actuator: Actuator):
        actuator.actuate_next_image(self)
