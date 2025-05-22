from __future__ import annotations

import logging
import queue
import time
from datetime import datetime, timedelta
from multiprocessing import Pipe, Process, Queue
from typing import Protocol

from djitellopy import Tello, TelloException
from imageio import imwrite

log = logging.getLogger('drone_pilot')


class DronePilot(Process):
    """A Process that manages the drone.

    The main purpose for this process is to ensure we continually send commands
    to the drone to prevent it from autolanding.

    Usage example:
    >>> dp = DronePilot()
    >>> dp.start()
    >>> dp.takeoff()
    >>> dp.save_photo(f"photo_0.png")
    >>> dp.land()
    >>> dp.shutdown()

    """
    def __init__(self, tello_log_level=logging.ERROR):
        super().__init__()
        self._action_queue = Queue()
        self._response_out, self._response_in = Pipe(duplex=False)
        self._shutdown = False
        self._tello = None
        self._frame_read = None
        self._photo_counter = 0
        self._last_keepalive = datetime.now()
        self._tello_log_level = tello_log_level

    # ----- Client API -----

    def takeoff(self):
        return self.call(TakeOff())

    def land(self):
        return self.call(Land())

    def get_battery(self):
        return self.call(GetBattery())

    def get_height(self):
        return self.call(GetHeight()) / 100

    def get_yaw(self):
        return self.call(GetYaw())

    def move_left(self, distance_m=0.20):
        distance = round(distance_m * 100)
        return self.call(MoveLeft(distance))

    def move_right(self, distance_m=0.20):
        distance = round(distance_m * 100)
        return self.call(MoveRight(distance))

    def move_forward(self, distance_m = 0.20):
        distance = round(distance_m * 100)
        return self.call(MoveForward(distance))

    def move_backward(self, distance_m = 0.20):
        distance = round(distance_m * 100)
        return self.call(MoveBackward(distance))

    def move_up(self, distance_m = 0.20):
        distance = round(distance_m * 100)
        return self.call(MoveUp(distance))

    def move_down(self, distance_m = 0.20):
        distance = round(distance_m * 100)
        return self.call(MoveDown(distance))

    def rotate_left(self, degrees):
        return self.call(RotateLeft(round(degrees)))

    def rotate_right(self, degrees):
        return self.call(RotateRight(round(degrees)))

    def take_photo(self):
        return self.call(TakePhoto())

    def save_photo(self, filename):
        frame = self.call(TakePhoto())
        imwrite(filename, frame)
        self._photo_counter += 1

    def shutdown(self):
        return self.call(Shutdown())

    # ----- Internal API -----

    def call(self, msg: DroneCommand):
        """Sends a message and waits for a response.

        Note: the `call` nomenclature comes from Elixir/Erlang GenServers"""
        self._action_queue.put(msg)
        return self._response_out.recv()

    @property
    def frame(self):
        return self._frame_read.frame

    def run(self):
        self._tello = Tello()
        self._tello.LOGGER.setLevel(self._tello_log_level)
        self._tello.connect(wait_for_state=True)
        self._tello.streamon()
        self._frame_read = self._tello.get_frame_read(with_queue=True, max_queue_len=1)

        while not self._shutdown:
            try:
                action: DroneCommand = self._action_queue.get_nowait()
                log.info(f"Got action: {action.__class__.__name__}")
                response = action.act(self, self._tello)
                self._response_in.send(response)

                self._last_keepalive = datetime.now()
                time.sleep(self._tello.TIME_BTW_COMMANDS)
            except queue.Empty:
                pass
            except TelloException as e:
                log.error(f"Exception: {e}")

            # Send a query in case we haven't processed a message during this loop
            now = datetime.now()
            if now - self._last_keepalive >= timedelta(seconds=10):
                self._tello.send_read_command("battery?")
                self._last_keepalive = now


class DroneCommand(Protocol):
    def act(self, pilot: DronePilot, tello: Tello) -> dict | None: ...


class TakeOff(DroneCommand):
    def act(self, _pilot, tello):
        tello.takeoff()


class Land(DroneCommand):
    def act(self, _pilot, tello):
        tello.land()

class GetBattery(DroneCommand):
    def act(self, _pilot, tello):
        return tello.get_battery()


class GetHeight(DroneCommand):
    def act(self, _pilot, tello):
        return tello.get_height()

class GetYaw(DroneCommand):
    def act(self, _pilot, tello):
        return tello.get_yaw()


class MoveBase(DroneCommand):
    """Base for move commands.

    Handles error checking during construction."""
    def __init__(self, distance):
        if not 20 < distance < 500:
            raise ValueError("Tello drone cannot move less than 20 cm or "
                             "more than 500 cm in a single move command.")

class MoveLeft(MoveBase):
    def __init__(self, distance):
        super().__init__(distance)
        self.distance = distance

    def act(self, _pilot, tello):
        tello.move_left(self.distance)


class MoveRight(MoveBase):
    def __init__(self, distance):
        super().__init__(distance)
        self.distance = distance

    def act(self, _pilot, tello):
        tello.move_right(self.distance)


class MoveForward(MoveBase):
    def __init__(self, distance):
        super().__init__(distance)
        self.distance = distance

    def act(self, pilot, tello):
        tello.move_forward(self.distance)


class MoveBackward(MoveBase):
    def __init__(self, distance):
        super().__init__(distance)
        self.distance = distance

    def act(self, pilot, tello):
        tello.move_back(self.distance)


class MoveUp(MoveBase):
    def __init__(self, distance):
        super().__init__(distance)
        self.distance = distance

    def act(self, pilot, tello):
        tello.move_up(self.distance)


class MoveDown(MoveBase):
    def __init__(self, distance):
        super().__init__(distance)
        self.distance = distance

    def act(self, pilot, tello):
        tello.move_down(self.distance)


class RotateLeft(DroneCommand):
    def __init__(self, degrees):
        self.degrees = degrees

    def act(self, pilot, tello):
        tello.rotate_counter_clockwise(self.degrees)


class RotateRight(DroneCommand):
    def __init__(self, degrees):
        self.degrees = degrees

    def act(self, pilot, tello):
        tello.rotate_clockwise(self.degrees)


class TakePhoto(DroneCommand):
    def act(self, pilot, _tello):
        return pilot.frame


class QueryState(DroneCommand):
    def act(self, _pilot, tello):
        return tello.get_current_state()

class Shutdown(DroneCommand):
    def act(self, pilot, tello):
        pilot._shutdown = True
