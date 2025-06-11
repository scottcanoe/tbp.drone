# Data Processing Guide

This guide explains how to process drone imagery and work with our sample dataset for development and testing. Before integrating with Monty (which currently obtains RGBD dataset from habitat-sim), we experimented with various computer vision approaches including depth extraction and object segmentation.

## Prerequisites

Before you begin, ensure you have:
1. Python 3.8 installed (required for compatibility)
2. Conda environment manager installed
3. Git installed
4. Access to download large model files (approximately 1GB total)

## Initial Setup

1. Clone the required repositories:
```bash
cd ~/tbp
git clone https://github.com/DepthAnything/Depth-Anything-V2
git clone https://github.com/facebookresearch/segment-anything
```

2. Set up the conda environment:
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate drone
```

3. Download required model files:
   - [Depth-Anything-V2-Base model](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true)
   - [SAM ViT-B model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

   Place both model files in the `~/tbp/tbp.drone/models/` directory.

4. For macOS users (especially with Apple Silicon), set this environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Sample Dataset Structure

The sample dataset is located in `dataset/potted_meat_can_v4/`. It contains 12 different views of a potted meat can, taken at 30-degree increments (folders 0 through 11). This dataset is designed for testing vision processing pipelines before deploying on the actual drone.

Each view folder (0-11) contains:
- `image.png`: Original image of the potted meat can
- `bbox.png`: Image with bounding box visualization overlaid
- `bbox.json`: JSON file containing bounding box coordinates in pixels:
  ```json
  {
      "spam_can": [x1, y1, x2, y2],  // Top-left and bottom-right coordinates
      "aruco": [x1, y1, x2, y2]      // ArUco marker coordinates (optional)
  }
  ```

## Project Structure

```
src/
  vision/
    __init__.py              # Main package interface
    point_cloud.py           # 3D point cloud generation interface
    depth_processing/        # Depth estimation implementation
      __init__.py
      depth_estimator.py     # Uses Depth Anything V2
      object_segmenter.py    # Uses Segment Anything Model (SAM)
    landmark_detection/      # ArUco marker processing
      aruco_detection.py     # Marker detection and pose estimation

dataset/
  potted_meat_can_v4/       # Test dataset
    0/                      # Views at 30° increments
    1/
    ...
    11/
```

## Vision Processing Pipeline

### Quick Start

To test the pipeline with a sample image:

1. Place a test image named `picture.png` in the tbp.drone directory
2. Run the processing script using either method:

```bash
# Option 1: From parent directory
cd ~/tbp
PYTHONPATH=$PWD python -m tbp.drone.src.vision.point_cloud

# Option 2: From project directory
cd ~/tbp/tbp.drone
PYTHONPATH=~/tbp python -m src.vision.point_cloud
```

The script will:
- Generate a 3D point cloud with semantic labels
- Save visualization as `pointcloud.png`

### Using the Pipeline in Your Code

#### Basic Point Cloud Generation

```python
from tbp.drone.src.vision import DepthTo3DLocations

# Initialize processor (default max_depth=100.0 meters)
processor = DepthTo3DLocations()

# Process image with a center point prompt
image_width, image_height = 960, 720  # Adjust to your image size
center_point = [(image_width/2, image_height/2)]
center_label = [1]  # 1 indicates foreground

points_3d = processor(
    "path/to/image.png",
    input_points=center_point,
    input_labels=center_label
)
```

#### ArUco Marker Detection

```python
from tbp.drone.src.vision.landmark_detection.aruco_detection import ArucoDetector
import cv2

# Initialize detector (specify physical marker size)
detector = ArucoDetector(marker_size=0.05)  # 5cm marker

# Process image
image = cv2.imread("path/to/image.png")
corners, ids, rejected = detector.detect(image, output_dir=".")

# Visualize results
if ids is not None and len(ids) > 0:
    image = detector.draw_markers_with_pose(image, corners, ids)
    cv2.imwrite("aruco_result.png", image)
```

#### Advanced Point Cloud Processing

```python
from tbp.drone.src.vision.point_cloud import DroneDepthTo3DLocations
import cv2
import numpy as np

# Initialize with specific parameters
processor = DroneDepthTo3DLocations(
    resolution=(720, 960),  # Height, Width
    max_depth=1.0,         # Maximum depth in meters
    get_all_points=False   # True to include background
)

# Load image
image_path = "path/to/image.png"
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Create multiple point prompts for better object segmentation
input_points = np.array([
    [w/2, h/2],      # Center
    [w/2, h/2-50],   # Top
    [w/2, h/2+50],   # Bottom
    [w/2-50, h/2],   # Left
    [w/2+50, h/2]    # Right
])
input_labels = np.array([1, 1, 1, 1, 1])  # All foreground

# Generate point cloud
points_3d = processor(
    image_path,
    input_points=input_points.tolist(),
    input_labels=input_labels.tolist()
)

# Save results
np.save("points_3d.npy", points_3d)  # Point cloud
```

### Visualizing Results

#### Using Open3D (Static)
```python
from tbp.drone.src.vision.point_cloud import visualize_point_cloud_with_camera

visualize_point_cloud_with_camera(
    points_3d,
    camera_position=None,  # Optional camera pose
    camera_rotation=None   # Optional camera orientation
)
```

#### Using Plotly (Interactive)
```python
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

## Troubleshooting

1. **Model Loading Issues**
   - Verify model files are in `~/tbp/tbp.drone/models/`
   - Check file permissions
   - Ensure correct model versions:
     - SAM: `sam_vit_b_01ec64.pth`
     - Depth-Anything: `depth_anything_v2_vitb.pth`

2. **macOS/Apple Silicon Issues**
   - Set `export PYTORCH_ENABLE_MPS_FALLBACK=1`
   - Use Python 3.8 for compatibility
   - Update PyTorch if needed: `pip install --upgrade torch`

3. **ArUco Detection Problems**
   - Ensure good lighting conditions
   - Check marker size matches physical size
   - Verify marker is clearly visible in image
   - Confirm camera calibration parameters

4. **Point Cloud Quality**
   - Use well-lit, focused images
   - Adjust `max_depth` parameter if needed
   - Try multiple point prompts for better segmentation
   - Check input image resolution matches processor settings

## Performance Notes

- Processing time varies with image size and hardware
- CPU-only processing is significantly slower than GPU
- Memory usage increases with point cloud density
- Consider downscaling large images for faster processing

