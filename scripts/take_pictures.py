import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import quaternion as qt

# from tbp.drone.src.drone_pilot import DronePilot
from djitellopy import Tello, TelloException
from scipy.spatial.transform import Rotation

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# tello = None


def init():
    global tello
    tello = Tello()
    tello.connect(wait_for_state=True)
    tello.streamon()
    tello.get_frame_read(with_queue=True, max_queue_len=1)


class Photographer:
    def __init__(self, data_dir: os.PathLike):
        self.tello = Tello()
        self.tello.connect(wait_for_state=True)
        self.tello.streamon()
        self.tello.get_frame_read(with_queue=True, max_queue_len=1)
        self.counter = 0
        self.data_path = Path.home() / "tbp/data/worldimages/drone/potted_meat_can"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def show(self):
        # Save image
        frame = self.tello.get_frame_read(with_queue=True, max_queue_len=1).frame
        imageio.imwrite(self.data_path / f"{self.counter}.png", frame)
        fig, ax = plt.subplots()
        ax.imshow(frame)
        plt.show()

    def __call__(self):
        # Save image
        frame = self.tello.get_frame_read(with_queue=True, max_queue_len=1).frame
        imageio.imwrite(self.data_path / f"{self.counter}.png", frame)
        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.set_title(f"Frame {self.counter}")
        plt.show()
        self.counter += 1
