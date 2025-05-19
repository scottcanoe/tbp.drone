import cv2
import torch
from PIL import Image
import sys
import os
from pathlib import Path

# Assuming Depth-Anything-V2 is at the same level as tbp.drone
depth_anything_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Depth-Anything-V2'))
sys.path.insert(0, depth_anything_path)

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE= "cuda" if torch.cuda.is_available() else "cpu" # can't use mps because need newer version of pytorch that is not available in python 3.8

model_configs = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
}

encoder = "vitb"

depth_to_anything_model_path = Path("~/tbp/tbp.drone/models/depth_anything_v2_vitb.pth").expanduser()
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(depth_to_anything_model_path, map_location="cpu"))
model = model.to(DEVICE).eval()

import numpy as np
data = np.fromfile(Path("~/tbp/tbp.drone/picture.png").expanduser(), dtype=np.uint8) # 720 x 960 x 3
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
depth = model.infer_image(img)

import matplotlib.pyplot as plt
plt.imshow(depth)
plt.colorbar()
plt.savefig("depth.png")