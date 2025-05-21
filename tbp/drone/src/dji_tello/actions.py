from numbers import Number

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.actions.actuator import Actuator


class TakeOff(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "TakeOff":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)

    def act(self, actuator: Actuator):
        actuator.actuate_takeoff(self)


class Land(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "Land":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)

    def act(self, actuator: Actuator):
        actuator.actuate_land(self)


class MoveBackward(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "MoveBackward":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_backward(self)


class MoveLeft(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "MoveLeft":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_left(self)


class MoveRight(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "MoveRight":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_right(self)


class MoveUp(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "MoveUp":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_up(self)


class MoveDown(Action):
    @classmethod
    def sample(cls, agent_id: str, sampler: "ActionSampler") -> "MoveDown":
        return sampler.sample_turn_right(agent_id)

    def __init__(self, agent_id: str, distance: Number):
        super().__init__(agent_id=agent_id)
        self.distance = distance

    def act(self, actuator: Actuator):
        actuator.actuate_move_down(self)
