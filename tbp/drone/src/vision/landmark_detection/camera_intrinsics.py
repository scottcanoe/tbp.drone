import numpy as np

# Camera intrinsic matrix (from calibration)
camera_matrix = np.array(
    [
        [921.170702, 0.000000, 459.904354],
        [0.000000, 919.018377, 351.238301],
        [0.000000, 0.000000, 1.000000],
    ]
)

# Distortion coefficients (from calibration)
dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
