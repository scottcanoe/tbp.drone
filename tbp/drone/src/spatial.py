from numbers import Number

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation


def as_signed_angle(angle: Number) -> Number:
    """Convert an angle in [0, 360] to an angle in [-180, 180]."""
    return (angle + 180) % 360 - 180


def as_unsigned_angle(angle: Number) -> Number:
    """Convert an angle in [-180, 180] to an angle in [0, 360]."""
    return np.mod(angle, 360)


def compute_relative_angle(a: Number, b: Number) -> float:
    """Compute the angular distance (in degrees) between two angles.

    Args:
        a: The starting angle in degrees. Maybe be a signed or unsigned angle.
        b: The ending angle in degrees. Maybe be a signed or unsigned angle.

    Returns:
        The (signed) angular distance between the two angles in degrees.
    """
    a = as_unsigned_angle(a)
    b = as_unsigned_angle(b)
    return as_signed_angle(b - a)


def reorder_quat_array(quat_array: np.ndarray, order: str) -> np.ndarray:
    """Swap the order of a quaternion array.

    For converting between the 'xyzw' numpy/scipy convention and the
    'wxyz' convention used by quaternion.

    Args:
        quat_array: A quaternion array.
        order: The order to convert to. Must be one of "xyzw" or "wxyz".

    Returns:
        A quaternion array in the new order.
    """
    if order == "xyzw":
        return np.array(
            [quat_array[1], quat_array[2], quat_array[3], quat_array[0]],
            dtype=quat_array.dtype,
        )
    elif order == "wxyz":
        return np.array(
            [quat_array[3], quat_array[0], quat_array[1], quat_array[2]],
            dtype=quat_array.dtype,
        )
    else:
        msg = f"Invalid order: '{order}'. Must be one of 'xyzw' or 'wxyz'."
        raise ValueError(msg)


def pitch_roll_yaw_to_quaternion(
    pitch: Number,
    roll: Number,
    yaw: Number,
) -> qt.quaternion:
    """Convert pitch, roll, and yaw to a quaternion.

    Args:
        pitch: The pitch angle in degrees.
        roll: The roll angle in degrees.
        yaw: The yaw angle in degrees.

    Returns:
        A quaternion representing the rotation.
    """
    rot = Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True)
    quat_array_xyzw = rot.as_quat()
    quat_array_wxyz = reorder_quat_array(quat_array_xyzw, "wxyz")
    return qt.quaternion(*quat_array_wxyz)
