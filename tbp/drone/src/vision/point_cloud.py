import cv2
import numpy as np
from pathlib import Path
import sys
import json
from typing import Union, Tuple, Optional, List
import numpy.typing as npt
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to Python path for imports to work with direct execution
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tbp.drone.src.vision.depth_processing.depth_estimator import DepthEstimator
from tbp.drone.src.vision.depth_processing.object_segmenter import ObjectSegmenter
from tbp.drone.src.vision.landmark_detection.camera_intrinsics import camera_matrix
from tbp.drone.src.vision.landmark_detection.aruco_detection import ArucoDetector

class DroneDepthTo3DLocations:
    """Transform RGB image into 3D point cloud with semantic labels.
    
    This class takes an RGB image and transforms it into a 3D point cloud where each point
    has both spatial coordinates (x, y, z) and color information. It uses DepthAnything V2
    for depth estimation and SAM with bbox guidance for object segmentation.
    
    The output is a numpy array with shape (N, 6) where N is the number of points and
    each point has format [x, y, z, r, g, b].
    
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
        
        # Initialize models
        self.depth_estimator = DepthEstimator(model_path=depth_model_path)
        self.object_segmenter = ObjectSegmenter(model_path=sam_model_path)
        self.aruco_detector = ArucoDetector(marker_size=aruco_marker_size)

    def __call__(
        self,
        image: Union[str, npt.NDArray[np.uint8]],
        bbox_annotation_path: Optional[str] = None,
        marker_world_position: Optional[List[float]] = None,
        marker_world_orientation: Optional[List[float]] = None,
    ) -> npt.NDArray[np.float32]:
        """Convert depth image to 3D point cloud using bbox-guided SAM segmentation."""
        # Get depth map and RGB image
        depth_map, rgb_image_from_estimator = self.depth_estimator.estimate_depth(image)
        
        # Get segmentation mask using bbox as input to SAM
        if bbox_annotation_path and Path(bbox_annotation_path).exists():
            with open(bbox_annotation_path, 'r') as f:
                annotations = json.load(f)
            
            if 'spam_can' in annotations:
                # Convert bbox to SAM input format
                x1, y1, x2, y2 = annotations['spam_can']
                bbox_input = [[x1, y1, x2, y2]]  # SAM expects a list of boxes
                
                # Get SAM segmentation using bbox
                mask, _ = self.object_segmenter.segment_image(
                    rgb_image_from_estimator,
                    input_boxes=bbox_input  # Pass bbox to SAM
                )
            else:
                mask = np.ones_like(depth_map, dtype=bool)
        else:
            # If no annotation, use entire image
            mask = np.ones_like(depth_map, dtype=bool)
        
        # Create modified depth map where background has max depth
        modified_depth = depth_map.copy()
        modified_depth[~mask] = self.max_depth
        
        # Prepare RGB image for color sampling
        rgb_image_for_colors = rgb_image_from_estimator

        # Resize depth, mask, and rgb_image to match target resolution if needed
        current_depth_h, current_depth_w = modified_depth.shape
        if (current_depth_h, current_depth_w) != self.resolution:
            modified_depth = cv2.resize(modified_depth, (self.resolution[1], self.resolution[0]))
            mask = cv2.resize(mask.astype(np.float32), (self.resolution[1], self.resolution[0])) > 0.5

        # Resize rgb_image_for_colors if needed
        orig_rgb_h, orig_rgb_w = rgb_image_from_estimator.shape[:2]
        if (orig_rgb_h, orig_rgb_w) != self.resolution:
            rgb_image_for_colors = cv2.resize(rgb_image_from_estimator, (self.resolution[1], self.resolution[0]))

        # Create pixel coordinate grid
        v_coords, u_coords = np.meshgrid(
            np.arange(self.resolution[1]),  # x-coordinates
            np.arange(self.resolution[0])   # y-coordinates
        )
        
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
        colors_normalized_rgb = cv2.cvtColor(rgb_image_for_colors, cv2.COLOR_BGR2RGB) / 255.0
        colors_flat = colors_normalized_rgb.reshape(-1, 3) # shape: (N, 3)

        # Combine xyz with colors
        xyz_colors = np.concatenate([xyz, colors_flat], axis=1) # shape: (N, 6)
        
        # Filter points based on get_all_points flag
        object_mask_flat = mask.reshape(-1)
        if self.get_all_points:
            points_3d = xyz_colors
        else:
            points_3d = xyz_colors[object_mask_flat]
        
        # --- ArUco marker detection and world projection ---
        if marker_world_position is None:
            marker_world_position = [0.0, 0.0, 0.0]
            
        # Detect ArUco markers in the RGB image
        corners, ids, rejected = self.aruco_detector.detect(rgb_image_from_estimator, output_dir=".")
        if ids is not None and len(ids) > 0:
            # Use the first detected marker for pose estimation
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

def process_spam_dataset_v2():
    """Process all PNG images in the spam_dataset_v2 directory and generate point clouds.
    
    This function:
    1. Finds all PNG files in the spam_dataset_v2 directory
    2. For each image, looks for corresponding bbox annotation
    3. Generates and saves point cloud data and visualization
    4. Creates individual scatter plots and a combined visualization
    """
    # Initialize processor
    processor = DroneDepthTo3DLocations(get_all_points=False)  # Only get spam can points
    
    # Setup paths
    base_dir = Path("~/tbp/tbp.drone").expanduser()
    dataset_dir = base_dir / "imgs" / "spam_dataset_v2"
    bbox_dir = dataset_dir / "bbox_annotations"
    point_clouds_dir = dataset_dir / "point_clouds"
    
    # Create point clouds directory if it doesn't exist
    point_clouds_dir.mkdir(exist_ok=True)
    
    # Get all PNG files in the dataset directory
    png_files = list(dataset_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files in {dataset_dir}")
    
    # Store all points for combined visualization
    all_points = []
    
    # Process each image
    for img_path in png_files:
        print(f"Processing {img_path.name}")
        
        # Get corresponding bbox annotation path
        bbox_path = bbox_dir / f"{img_path.stem}_annotations.json"
        
        try:
            # Get 3D point cloud
            points_3d = processor(
                str(img_path),
                bbox_annotation_path=str(bbox_path) if bbox_path.exists() else None
            )
            
            # Store points for combined visualization
            all_points.append(points_3d)
            
            # Save point cloud data
            output_npy = point_clouds_dir / f"points_3d_{img_path.stem}.npy"
            np.save(str(output_npy), points_3d)
            print(f"Saved point cloud data to {output_npy}")
            
            # Get coordinates for plotting
            x = points_3d[:, 0]  # right
            y = points_3d[:, 1]  # forward
            z = points_3d[:, 2]  # up
            colors = points_3d[:, 3:6]  # RGB colors
            
            # Create matplotlib 3D scatter plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x, y, z, c=colors, s=1)
            ax.set_xlabel('X (right +)')
            ax.set_ylabel('Y (forward +)')
            ax.set_zlabel('Z (up +)')
            ax.set_title(f'3D Point Cloud: {img_path.stem}')
            
            # Save matplotlib plot
            output_scatter = point_clouds_dir / f"scatter3d_{img_path.stem}.png"
            plt.savefig(output_scatter, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved scatter plot to {output_scatter}")
            
            # Create and save Plotly visualization
            try:
                import plotly.graph_objs as go
                import plotly.io as pio
                
                # Prepare colors for Plotly
                point_colors_plotly = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                                     for r,g,b in colors]
                
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
                    title=f'3D Point Cloud: {img_path.stem}',
                    width=800,
                    height=800,
                    showlegend=True
                )
                
                # Save interactive plot
                output_html = point_clouds_dir / f"pointcloud_{img_path.stem}_3d.html"
                pio.write_html(fig_plotly, file=str(output_html), auto_open=False)
                print(f"Saved interactive plot to {output_html}")
                
            except ImportError:
                print("Plotly is not installed. Skipping interactive plot.")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    # Create combined visualization of all points
    if all_points:
        combined_points = np.concatenate(all_points, axis=0)
        
        # Save combined point cloud data
        np.save(str(point_clouds_dir / "points_3d_combined.npy"), combined_points)
        
        # Create matplotlib combined scatter plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            combined_points[:, 0],
            combined_points[:, 1],
            combined_points[:, 2],
            c=combined_points[:, 3:6],
            s=1
        )
        ax.set_xlabel('X (right +)')
        ax.set_ylabel('Y (forward +)')
        ax.set_zlabel('Z (up +)')
        ax.set_title('Combined 3D Point Cloud')
        
        # Save combined scatter plot
        plt.savefig(str(point_clouds_dir / "scatter3d_combined.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create combined Plotly visualization
        try:
            import plotly.graph_objs as go
            import plotly.io as pio
            
            # Prepare colors for Plotly
            combined_colors_plotly = [
                f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                for r,g,b in combined_points[:, 3:6]
            ]
            
            # Create combined 3D scatter plot
            fig_plotly = go.Figure(data=[
                # Point cloud
                go.Scatter3d(
                    x=combined_points[:, 0],
                    y=combined_points[:, 1],
                    z=combined_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=combined_colors_plotly,
                        opacity=0.6
                    ),
                    name='Combined Point Cloud'
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
                title='Combined 3D Point Cloud',
                width=1000,
                height=1000,
                showlegend=True
            )
            
            # Save combined interactive plot
            pio.write_html(
                fig_plotly,
                file=str(point_clouds_dir / "pointcloud_combined_3d.html"),
                auto_open=False
            )
            
        except ImportError:
            print("Plotly is not installed. Skipping combined interactive plot.")

if __name__ == "__main__":
    process_spam_dataset_v2() 