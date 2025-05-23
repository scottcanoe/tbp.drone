import glob
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import natsort
import numpy as np
from PIL import Image

image_dir = Path.home() / "tbp/data/worldimages/drone/potted_meat_can_v4"
output_path = image_dir / "animation.gif"
fps = 3
loop = False

h, w = 720, 960

image_dir = Path(image_dir)
image_files = []
for i in range(12):
    image_files.append(image_dir / f"{i}/image.png")

if not image_files:
    raise ValueError(f"No images found in {image_dir} matching pattern {pattern}")

# Load images
images = []
for file in image_files:
    img = Image.open(file)
    images.append(np.array(img))

h, w = images[0].shape[:2]
ratio = h / w
fig_width = 2
fig_height = fig_width * ratio

# Create figure and animation
fig = plt.figure(figsize=(fig_width, fig_height))
plt.axis("off")
ax = plt.gca()
ax.set_position([0, 0, 1, 1])

# Calculate frame duration
# Create animation
duration = 1 / fps
anim = animation.ArtistAnimation(
    fig,
    [[plt.imshow(img)] for img in images],
    interval=duration * 1000,  # Convert to milliseconds
    blit=True,
)

# Save animation
anim.save(
    output_path,
    writer="pillow",
    fps=1 / duration if fps is None else fps,
)
plt.close()
