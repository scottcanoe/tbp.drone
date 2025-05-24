import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Container, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def as_rgba(img: np.ndarray) -> np.ndarray:
    """Adds an opaque alpha channel to an image if it doesn't have one.

    Args:
        img: Input image. Values must be in the range [0, 255].

    Returns:
        np.ndarray: Output image with an opaque alpha channel. Values are in the
          range [0, 255].
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("Input must be a 3 or 4 channel image.")
    if img.shape[2] == 3:
        out = 255 * np.ones((img.shape[0], img.shape[1], 4), dtype=img.dtype)
        out[:, :, :3] = img
        return out
    else:
        return img
