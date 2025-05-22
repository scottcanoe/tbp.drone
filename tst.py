import datetime
import json
import os
import pprint as pp
import subprocess as sp
import threading
import time
import warnings
from numbers import Number
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from djitellopy import Tello
from scipy.spatial.transform import Rotation

from tbp.drone.src.drone_pilot import DronePilot
from tbp.drone.src.environment import DroneEnvironment

camera = {"height": 720, "width": 960}


def snap():
    image = env._pilot.take_photo()
    plt.imshow(image)
    return image


# env = DroneEnvironment("agen")
pilot = DronePilot()

# pilot.start()

# env.reset()
