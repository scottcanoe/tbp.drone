# Codebase for Drone Team

Template from tbp.monty paper. Uses tbp.monty version 0.5.0. 

## Getting Started

- Install Tello app from iOS or Android (Optional, but good to get a sense of what the Drone can do). 
- Python Interfaces to DJI Tello
	- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy)
	- Python 2 Project: [Tello-Python](https://github.com/dji-sdk/Tello-Python)
		- Will likely not use due to Python 2 but may be helpful to look at scripts like `Tello_Video(With_Pose_Recognition)`
	- Drone Programming with Python: [Youtube 3.5 hours](https://www.youtube.com/watch?v=LmEcyQnfpDA)
- Very handy resources
    - Tello User Manual (In drone channel on Slack)
- [Tello SDK Manual](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf)

## Working with Monty
- Setup something similar to the Paper Repo Template (to use a certain version of Monty for reproducibility).
- **Notes**: 
	- When working with drone, computer must be connected to Wifi: `TELLO-xxxx` (there will be no internet access)
	- Bluetooth must be off (at least true for my Mac)
- Update dependencies:
	- `pip install djitellopy`

### DJITelloPy Notes
```
tello = Tello()

# Output
[INFO] tello.py - 129 - Tello instance was initialized. Host: '192.168.10.1'. Port: '8889'.
```

## Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

To create the environment, run:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate drone
conda config --env --set subdir osx-64
```

### Important Note for Apple Silicon Users
Before running any depth estimation tasks, set the following environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
This is required because some operations in the Depth Anything model are not yet supported in MPS (Metal Performance Shaders) and need to fall back to CPU. Without this setting, you may encounter runtime errors.

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate drone
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate drone 
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate drone 
```

## Project Structure

The vision-related code is organized as follows:
```
src/
  vision/
    __init__.py              # Main package interface
    point_cloud.py           # Main interface for 3D point cloud generation
    depth_processing/        # Implementation details
      __init__.py
      depth_estimator.py     # Depth estimation using Depth Anything V2
      object_segmenter.py    # Object segmentation using SAM
```

The main interface you'll interact with is `DepthTo3DLocations` from the `vision` package. The depth estimation and segmentation implementations are internal details handled by the `depth_processing` subpackage.

## Estimating Depth using Depth-Anything-v2

[Depth-Anythin-v2 Github Repo](https://github.com/DepthAnything/Depth-Anything-V2)

1. Get the code via `git clone`
```bash
cd ~/tbp
git clone https://github.com/DepthAnything/Depth-Anything-V2
# Note: tbp.drone is at ~/tbp/tbp.drone
```
2. Download the Depth-Anything-V2-Base model from [here](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) 
**Note**: There are bigger models, but we have to run on CPU, so `Base` is a good starting point. 
**Note**: You will get error for the line in `dpt.py`:
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# Replace with the following line
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```
This is because `torch.backends.mps.is_available()` is not available in our `torch` version (v1.11.0) because of our Python 3.8 dependency. So we cannot take advantage of `mps` in our Macbook Pros. 

I have already copied and pasted their requirements into our `environment.yml` so you don't need to install anything. Do update your `drone` conda environment if you have already installed it. 

## Object Segmentation using Segment Anything Model (SAM)

[Segment Anything Model Github Repo](https://github.com/facebookresearch/segment-anything)

1. Get the code via `git clone`
```bash
cd ~/tbp
git clone https://github.com/facebookresearch/segment-anything
# Note: tbp.drone is at ~/tbp/tbp.drone
```

2. Download the SAM ViT-B model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
**Note**: We're using the ViT-B model as it provides a good balance between performance and resource usage when running on CPU.

The model should be placed in the `~/tbp/tbp.drone/models/` directory as `sam_vit_b_01ec64.pth`.

The requirements for SAM have already been added to our `environment.yml`, so you don't need to install anything additional. Just make sure your `drone` conda environment is up to date.

## Combined Depth Estimation and Object Segmentation

The project includes modules for depth estimation and object segmentation in the `src/vision` package. The main interface is the `DepthTo3DLocations` class which combines depth estimation and segmentation to create 3D point clouds.

### Usage

```bash
# 1. Make sure you have a `picture.png` in tbp.drone directory before running

# The below script will:
# - Generate a 3D point cloud with semantic labels
# - Save visualization as pointcloud.png

# Option 1: Run from parent directory
cd ~/tbp  # Go to parent directory containing tbp.drone
PYTHONPATH=$PWD python -m tbp.drone.src.vision.point_cloud

# Option 2: Run from tbp.drone directory
cd ~/tbp/tbp.drone
PYTHONPATH=~/tbp python -m src.vision.point_cloud
```

Or you can write your own python file and feed in any image:

```python
from tbp.drone.src.vision import DepthTo3DLocations

# Initialize the processor (default max_depth is 100.0)
point_cloud_generator = DepthTo3DLocations()

# Process an image with a point prompt
# You can specify where to segment the object using input points
center_point = [(image_width/2, image_height/2)]  # Point in center of image
center_label = [1]  # 1 indicates foreground

points_3d = point_cloud_generator(
    "path/to/image.png",
    input_points=center_point,
    input_labels=center_label
)
```

The module combines both Depth-Anything-V2 and SAM to:
1. Estimate depth for the entire image
2. Segment objects of interest
3. Convert the depth and segmentation information into a 3D point cloud
4. Set the depth of background regions (non-object areas) to a maximum value (default: 100.0)

This is particularly useful for drone navigation where you want to focus on the depth of specific objects while treating the background as far away.

## Using ArUco Detection and Point Cloud Generation

### ArUco Marker Detection

The `ArucoDetector` class can be used to detect ArUco markers in images and estimate camera pose. Here's how to use it:

```python
from tbp.drone.src.vision.landmark_detection.aruco_detection import ArucoDetector

# Initialize detector (marker_size is the physical size of your ArUco marker in meters)
detector = ArucoDetector(marker_size=0.05)  # 5cm marker

# Load and process your image
image = cv2.imread("path/to/your/image.png")
corners, ids, rejected = detector.detect(image, output_dir=".")

# Draw markers and pose information on the image
if ids is not None and len(ids) > 0:
    # Draw detected markers and pose axes
    image = detector.draw_markers_with_pose(image, corners, ids)
    
    # Optional: Draw rejected markers
    image = detector.draw_rejected_markers(image, rejected)
    
    # Save the result
    cv2.imwrite("result.png", image)
```

### 3D Point Cloud Generation

The `DroneDepthTo3DLocations` class combines depth estimation and object segmentation to create 3D point clouds from images. Here's how to use it:

```python
from tbp.drone.src.vision.point_cloud import DroneDepthTo3DLocations
import cv2
import numpy as np

# Initialize the processor
processor = DroneDepthTo3DLocations(
    resolution=(720, 960),  # Height, Width format
    max_depth=1.0,  # Maximum depth in meters
    get_all_points=False  # True to include background points
)

# Load your image to get dimensions
image_path = "path/to/your/image.png"
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Create point prompts for object segmentation
# These points help identify the object of interest
input_points = np.array([
    [w/2, h/2],      # Center
    [w/2, h/2-50],   # Top
    [w/2, h/2+50],   # Bottom
    [w/2-50, h/2],   # Left
    [w/2+50, h/2]    # Right
])
input_labels = np.array([1, 1, 1, 1, 1])  # All points are foreground

# Generate 3D point cloud
points_3d = processor(
    image_path,
    input_points=input_points.tolist(),
    input_labels=input_labels.tolist()
)

# points_3d shape is (N, 6) where each point is [x, y, z, r, g, b]
# Save the point cloud
np.save("points_3d.npy", points_3d)

# Optional: Save depth map and segmentation mask
depth_map, rgb_image = processor.depth_estimator.estimate_depth(image_path)
mask, _ = processor.object_segmenter.segment_image(
    rgb_image,
    input_points=input_points.tolist(),
    input_labels=input_labels.tolist()
)

# Save depth map
np.save("depthmap.npy", depth_map)
depth_norm = (255 * (depth_map - depth_map.min()) / (depth_map.ptp() + 1e-8)).astype(np.uint8)
cv2.imwrite("depthmap.png", depth_norm)

# Save segmentation mask
np.save("mask.npy", mask)
mask_img = (mask * 255).astype(np.uint8)
cv2.imwrite("mask.png", mask_img)
```

### Visualizing Results

The point cloud can be visualized using either Plotly (interactive) or Open3D:

```python
# Using Open3D
from tbp.drone.src.vision.point_cloud import visualize_point_cloud_with_camera

# Visualize point cloud (optionally with camera pose if ArUco markers were detected)
visualize_point_cloud_with_camera(points_3d, camera_position=None, camera_rotation=None)

# If you have Plotly installed, you can also create an interactive HTML visualization:
import plotly.graph_objs as go
import plotly.io as pio

x, y, z = points_3d[:, 0], points_3d[:, 1], -points_3d[:, 2]
point_colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                for r,g,b in points_3d[:, 3:6]]

fig = go.Figure(data=[
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=point_colors,
            opacity=0.6
        )
    )
])
pio.write_html(fig, file="pointcloud_3d.html")
```

### Important Notes

1. For ArUco detection:
   - The marker size must match the physical size of your printed ArUco markers
   - The camera must be calibrated (camera matrix and distortion coefficients)
   - Good lighting and marker visibility are essential for accurate detection

2. For point cloud generation:
   - The input image should be well-lit and in focus
   - The object of interest should be clearly visible
   - Point prompts help identify the object for segmentation
   - The depth estimation works best for indoor scenes within a few meters

3. System requirements:
   - Make sure you have all required models downloaded:
     - SAM ViT-B model (`sam_vit_b_01ec64.pth`)
     - Depth-Anything-V2-Base model (`depth_anything_v2_vitb.pth`)
   - For Apple Silicon users, remember to set:
     ```bash
     export PYTORCH_ENABLE_MPS_FALLBACK=1
     ```

