import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Union, Tuple, Optional, List
import numpy.typing as npt

# Add parent directory to Python path for imports to work with direct execution
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tbp.drone.src.vision.depth_processing.depth_estimator import DepthEstimator
from tbp.drone.src.vision.depth_processing.object_segmenter import ObjectSegmenter
from tbp.drone.src.vision.landmark_detection.camera_intrinsics import camera_matrix
from tbp.drone.src.vision.landmark_detection.aruco_detection import ArucoDetector

class DroneDepthTo3DLocations:
    """Transform RGB image into 3D point cloud with semantic labels.
    
    This class takes an RGB image and transforms it into a 3D point cloud where each point
    has both spatial coordinates (x, y, z) and a semantic label. It uses DepthAnything V2
    for depth estimation and SAM for object segmentation.
    
    The output is a numpy array with shape (N, 4) where N is the number of points and
    each point has format [x, y, z, semantic_id].
    
    Attributes:
        resolution: Camera resolution (H, W)
        focal_length: Focal length in pixels (calculated from physical parameters)
        cx: Optical center x-coordinate in pixels (cx)
        cy: Optical center y-coordinate in pixels (cy)
        zoom: Camera zoom factor. Default 1.0 (no zoom)
        get_all_points: Whether to return all 3D coordinates or only object points
        max_depth: Maximum depth value to use for background points
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (720, 960),  # Height, Width format for Tello
        zoom: float = 1.0,
        get_all_points: bool = False,
        max_depth: float = 1.0, # 1 meter
        depth_model_path: Optional[str] = None,
        sam_model_path: Optional[str] = None,
        aruco_marker_size: float = 0.05,  # meters, default 5cm
    ):
        """Initialize the 3D point cloud generator.
        
        Args:
            resolution: Image resolution as (height, width) for Tello camera (720p)
            zoom: Camera zoom factor
            get_all_points: If True, return all points including background
            max_depth: Maximum depth value for background points
            depth_model_path: Path to depth model weights
            sam_model_path: Path to SAM model weights
            aruco_marker_size: Size of ArUco marker in meters
        """
        self.resolution = resolution
        
        # Use camera intrinsics directly
        self.camera_matrix = camera_matrix
        
        self.zoom = zoom
        self.get_all_points = get_all_points
        self.max_depth = max_depth
        
        # Initialize depth and segmentation models
        self.depth_estimator = DepthEstimator(model_path=depth_model_path)
        self.object_segmenter = ObjectSegmenter(model_path=sam_model_path)
        # Initialize ArUco detector
        self.aruco_detector = ArucoDetector(marker_size=aruco_marker_size)

    def __call__(
        self,
        image: Union[str, npt.NDArray[np.uint8]],
        input_points: Optional[List[Tuple[float, float]]] = None,
        input_labels: Optional[List[int]] = None,
        marker_world_position: Optional[List[float]] = None,
        marker_world_orientation: Optional[List[float]] = None,
    ) -> npt.NDArray[np.float32]:
        """Convert depth image to 3D point cloud and project to world coordinates if ArUco marker is detected.
        
        Args:
            image: Input RGB image path or numpy array
            input_points: List of (x,y) coordinates to segment
            input_labels: List of labels for input_points (1=foreground, 0=background)
            marker_world_position: [x, y, z] world position of the marker's center (default [0,0,0])
            marker_world_orientation: Optional [rx, ry, rz] world orientation of marker
            
        Returns:
            Nx4 array of 3D points and semantic labels, in world coordinates if marker detected, else camera coordinates
        """
        # Get depth map and RGB image
        depth_map, rgb_image_from_estimator = self.depth_estimator.estimate_depth(image)
        
        # Get segmentation mask
        if input_points is None or input_labels is None:
            # Default to center point if no points provided
            h_rgb, w_rgb = rgb_image_from_estimator.shape[:2]
            input_points = [(w_rgb/2, h_rgb/2)]
            input_labels = [1]  # 1 indicates foreground
            
        mask, _ = self.object_segmenter.segment_image(
            rgb_image_from_estimator,
            input_points=input_points,
            input_labels=input_labels
        )
        
        # Create modified depth map where background has max depth
        modified_depth = depth_map.copy()
        modified_depth[~mask] = self.max_depth
        
        # Prepare RGB image for color sampling (will be resized if necessary)
        rgb_image_for_colors = rgb_image_from_estimator

        # Resize depth, mask, and rgb_image to match target resolution if needed
        current_depth_h, current_depth_w = modified_depth.shape
        if (current_depth_h, current_depth_w) != self.resolution:
            modified_depth = cv2.resize(modified_depth, (self.resolution[1], self.resolution[0]))
            # Resize mask from its original dimensions (tied to rgb_image_from_estimator)
            mask = cv2.resize(mask.astype(np.float32), (self.resolution[1], self.resolution[0])) > 0.5

        # Resize rgb_image_for_colors if its original dimensions differ from target resolution
        orig_rgb_h, orig_rgb_w = rgb_image_from_estimator.shape[:2]
        if (orig_rgb_h, orig_rgb_w) != self.resolution:
            rgb_image_for_colors = cv2.resize(rgb_image_from_estimator, (self.resolution[1], self.resolution[0]))

        # Create pixel coordinate grid
        v_coords, u_coords = np.meshgrid(
            np.arange(self.resolution[1]),  # x-coordinates
            np.arange(self.resolution[0])   # y-coordinates
        )
        # Save modified depth map with colorbar 
        depth_norm = (255 * (modified_depth - modified_depth.min()) / (modified_depth.ptp() + 1e-8)).astype(np.uint8)
        import matplotlib.pyplot as plt
        plt.imshow(depth_norm)
        plt.colorbar()
        plt.savefig(f"depthmap_spam_modified.png")
        plt.close()
        
        # Reshape depth and coordinates
        depth = modified_depth.reshape(-1)
        u = u_coords.reshape(-1)
        v = v_coords.reshape(-1)
        
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx_cam, cy_cam = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        x = ((u - cx_cam) * depth) / fx
        y = ((v - cy_cam) * depth) / fy
        z = depth

        xyz = np.stack((x, y, z), axis=1)  # shape: (N, 3)
        
        # Prepare colors (normalized and in RGB order)
        # rgb_image_for_colors is already at self.resolution
        colors_normalized_rgb = cv2.cvtColor(rgb_image_for_colors, cv2.COLOR_BGR2RGB) / 255.0
        colors_flat = colors_normalized_rgb.reshape(-1, 3) # shape: (N, 3)

        # Combine xyz with colors
        xyz_colors = np.concatenate([xyz, colors_flat], axis=1) # shape: (N, 6)
        
        # Filter points based on get_all_points flag
        object_mask_flat = mask.reshape(-1) # mask is already at self.resolution
        if self.get_all_points:
            points_3d = xyz_colors
        else:
            points_3d = xyz_colors[object_mask_flat]
        
        # --- ArUco marker detection and world projection ---
        # Default marker world position if not provided
        if marker_world_position is None:
            marker_world_position = [0.0, 0.0, 0.0]
        # Detect ArUco markers in the RGB image
        corners, ids, rejected = self.aruco_detector.detect(rgb_image_from_estimator, output_dir=".")
        if ids is not None and len(ids) > 0:
            # Use the first detected marker for pose estimation
            # Consider passing self.camera_matrix if aruco_detector requires it
            camera_position, camera_rotation = self.aruco_detector.get_camera_pose(
                corners[0], marker_world_position, marker_world_orientation
            )
        else:
            camera_position, camera_rotation = None, None
        # Transform points to world coordinates if pose is available
        if camera_position is not None and camera_rotation is not None:
            xyz_to_transform = points_3d[:, :3]
            colors_to_keep = points_3d[:, 3:6]
            # Apply rotation and translation: X_world = R @ X_cam + t
            xyz_world = xyz_to_transform @ camera_rotation.T + camera_position
            points_3d_world = np.concatenate([xyz_world, colors_to_keep], axis=1)
            return points_3d_world
        else:
            return points_3d

def main():
    """Example usage showing visualization of the 3D point cloud."""
    # Initialize processor
    processor = DroneDepthTo3DLocations()
    
    # Process example image with center point prompt
    object_name = "spam"
    image_path = str(Path(f"~/tbp/tbp.drone/imgs/{object_name}.png").expanduser())
    
    # Load image to get dimensions
    image = cv2.imread(image_path)
    h, w = image.shape[:2] # (720, 960)
    
    # Create point prompts for better segmentation
    input_points = np.array([
        [w/2, h/2],      # Center
        [w/2, h/2-50],   # Top
        [w/2, h/2+50],   # Bottom
        [w/2-50, h/2],   # Left
        [w/2+50, h/2]    # Right
    ])
    input_labels = np.array([1, 1, 1, 1, 1])  # All points are foreground

    # --- Save depth map and segmentation mask ---
    # Get depth map and RGB image
    depth_map, rgb_image = processor.depth_estimator.estimate_depth(image_path)
    # Get segmentation mask
    mask, _ = processor.object_segmenter.segment_image(
        rgb_image,
        input_points=input_points.tolist(),
        input_labels=input_labels.tolist()
    )
    # Save depth map
    np.save(f"depthmap_{object_name}.npy", depth_map)
    # Normalize for PNG
    depth_norm = (255 * (depth_map - depth_map.min()) / (depth_map.ptp() + 1e-8)).astype(np.uint8)
    cv2.imwrite(f"depthmap_{object_name}.png", depth_norm)
    # Save mask
    np.save(f"mask_{object_name}.npy", mask)
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(f"mask_{object_name}.png", mask_img)

    # Get 3D point cloud
    points_3d = processor(
        image_path,
        input_points=input_points.tolist(),
        input_labels=input_labels.tolist()
    ) # shape: (N, 6)
    
    # Filter points to keep only meaningful ones
    # Remove points that are too far to the sides (x-axis)
    points_3d = points_3d[np.abs(points_3d[:, 0]) < 0.2]
    # Remove points that are too far in depth (y-axis)
    points_3d = points_3d[points_3d[:, 1] < 0.5]
    # Remove points that are too high or low (z-axis)
    points_3d = points_3d[np.abs(points_3d[:, 2]) < 0.2]
    
    # Get all coordinates for 3D plotting
    x = points_3d[:, 0]  # right
    y = points_3d[:, 1]  # forward
    z = -points_3d[:, 2] # up (negative because camera coordinates are flipped)
    
    # Prepare colors for Plotly (list of 'rgb(r,g,b)' strings)
    point_colors_plotly = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in points_3d[:, 3:6]]

    # --- Plotly interactive plot ---
    try:
        import plotly.graph_objs as go
        import plotly.io as pio
        plotly_available = True
    except ImportError:
        plotly_available = False
    
    if plotly_available:
        # Create 3D scatter plot
        fig_plotly = go.Figure(data=[
            # Point cloud
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=point_colors_plotly,
                    opacity=0.6
                ),
                name='Point Cloud'
            ),
            # Origin marker
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle'
                ),
                name='Origin'
            )
        ])
        fig_plotly.update_layout(
            scene=dict(
                xaxis_title='X (right +)',
                yaxis_title='Y (forward +)',
                zaxis_title='Z (up +)',
                aspectmode='data'
            ),
            title='3D Point Cloud with RGB Colors (Interactive)',
            width=800,
            height=800,
            showlegend=True
        )
        pio.write_html(fig_plotly, file=f"pointcloud_{object_name}_3d.html", auto_open=False)
        print(f"Plotly interactive 3D plot saved as pointcloud_{object_name}_3d.html")
    else:
        print("Plotly is not installed. Skipping interactive plot.")

    # Save scatter as npy
    np.save(f"./points_3d_{object_name}.npy", points_3d)

    # --- Matplotlib 3D static plot ---
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    scatter = ax.scatter(
        x, y, z,
        c=points_3d[:, 3:6],  # RGB colors
        alpha=0.6,
        s=1,
        label='Point Cloud'
    )
    
    # Plot origin
    ax.scatter(
        [0], [0], [0],
        color='red',
        s=100,  # Larger size for visibility
        label='Origin'
    )
    
    ax.set_xlabel('X (right +)')
    ax.set_ylabel('Y (forward +)')
    ax.set_zlabel('Z (up +)')
    ax.set_title('3D Point Cloud with RGB Colors')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(f"pointcloud_{object_name}_3d.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 