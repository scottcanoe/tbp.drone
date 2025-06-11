"""Development/Exploratory code for calibrating estimated depth values"""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from tbp.drone.src.vision.depth_processing.depth_estimator import DepthEstimator

DATA_PATH = Path.home() / "tbp/data/worldimages/drone/"


# def fit_depthanything_to_ground_truth():
"""Fit a linear function to the depth map."""
depth_estimator = DepthEstimator()
path = DATA_PATH / "depth_estimation/image.png"
image = imageio.imread(path)
da_depth_map = depth_estimator(image)

x_pixels = np.array([330, 412, 482, 548, 602])
gt_depths = np.array([(25 + (i - 2) * 2.54 * np.sqrt(2)) / 100 for i in range(5)])
da_depths = np.array([da_depth_map[230, x] for x in x_pixels])

slope, intercept = np.polyfit(da_depths, gt_depths, 1)
corrected_depths = slope * da_depths + intercept
# or `corrected_depths = depth_estimator.to_meters(da_depth_map)``

plt.figure(figsize=(8, 6))
plt.scatter(da_depths, gt_depths, color="blue", label="Data points")
plt.plot(da_depths, corrected_depths, color="red", label="Linear fit")
plt.xlabel("Estimated depth")
plt.ylabel("Ground truth depth")
plt.legend()
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
ax = axes[0]
ax.imshow(image)
ax = axes[1]
ax.imshow(da_depth_map, cmap="inferno")
ax = axes[2]
corrected_depth_map = depth_estimator.to_meters(da_depth_map)
ax.imshow(corrected_depth_map, cmap="inferno")
plt.show()
