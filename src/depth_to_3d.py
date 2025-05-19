import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy.typing as npt

from segmented_depth import SegmentedDepth

class DepthTo3DLocations:
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
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1080, 1920),  # Height, Width format
        focal_length_pixels: float = 1825.1,  # Calculated from physical parameters
        optical_center: Tuple[float, float] = (960.0, 540.0),  # cx, cy
        zoom: float = 1.0,
        get_all_points: bool = False,
    ):
        """Initialize the 3D point cloud generator.
        
        Args:
            resolution: Image resolution as (height, width)
            focal_length_pixels: Focal length in pixels (calculated from physical parameters)
            optical_center: Optical center in pixels (cx, cy)
            zoom: Camera zoom factor
            get_all_points: If True, return all points including background
        """
        self.resolution = resolution
        self.focal_length = focal_length_pixels
        self.cx, self.cy = optical_center
        self.zoom = zoom
        self.get_all_points = get_all_points
        
        # Initialize segmented depth processor
        self.segmented_depth = SegmentedDepth()

    def __call__(
        self,
        image: Union[str, npt.NDArray[np.uint8]],
        input_points: Optional[List[Tuple[float, float]]] = None,
        input_labels: Optional[List[int]] = None,
    ) -> npt.NDArray[np.float32]:
        """Convert depth image to 3D point cloud.
        
        Args:
            image: Input RGB image path or numpy array
            input_points: List of (x,y) coordinates to segment
            input_labels: List of labels for input_points (1=foreground, 0=background)
            
        Returns:
            Nx4 array of 3D points and semantic labels
        """
        # Process image through depth and segmentation pipeline
        modified_depth, mask, rgb_image = self.segmented_depth.process_image(
            image, input_points, input_labels
        )
        
        # Resize depth and mask to match target resolution
        h, w = modified_depth.shape
        if (h, w) != self.resolution:
            modified_depth = cv2.resize(modified_depth, (self.resolution[1], self.resolution[0]))
            mask = cv2.resize(mask.astype(np.float32), (self.resolution[1], self.resolution[0])) > 0.5

        # Create pixel coordinate grid
        v, u = np.meshgrid(
            np.arange(self.resolution[1]),  # x-coordinates
            np.arange(self.resolution[0])   # y-coordinates
        )
        
        # Reshape depth and coordinates
        depth = modified_depth.reshape(-1)
        u = u.reshape(-1)
        v = v.reshape(-1)
        
        # Calculate 3D coordinates using pinhole camera model
        x = (v - self.cx) * depth / self.focal_length
        y = (u - self.cy) * depth / self.focal_length
        z = depth
        
        # Stack coordinates
        xyz = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        
        # Extract semantic labels
        semantic = mask.reshape(-1)
        
        # Filter points based on get_all_points flag
        if self.get_all_points:
            points_3d = xyz
            points_3d[:, 3] = semantic
        else:
            detected = semantic > 0
            points_3d = xyz[detected]
            points_3d[:, 3] = semantic[detected]
        
        return points_3d

def main():
    """Example usage showing visualization of the 3D point cloud."""
    # Initialize processor
    processor = DepthTo3DLocations()
    
    # Process example image with center point prompt
    image_path = str(Path("~/tbp/tbp.drone/picture.png").expanduser())
    
    # Load image to get dimensions
    image = cv2.imread(image_path)
    h, w = image.shape[:2] # (720, 960)
    
    # Create point prompts for better segmentation
    # Add multiple points to better capture the cup
    input_points = np.array([
        [w/2, h/2],      # Center
        [w/2, h/2-50],   # Top
        [w/2, h/2+50],   # Bottom
        [w/2-50, h/2],   # Left
        [w/2+50, h/2]    # Right
    ])
    input_labels = np.array([1, 1, 1, 1, 1])  # All points are foreground
    
    # Get 3D point cloud
    points_3d = processor(
        image_path,
        input_points=input_points.tolist(),
        input_labels=input_labels.tolist()
    )
    
    # Visualize point cloud using matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by semantic ID
    scatter = ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c=points_3d[:, 3],
        cmap='viridis',
        alpha=0.6,
        s=1  # Smaller point size for better detail
    )
    
    plt.colorbar(scatter, label='Semantic ID')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud with Semantic Labels')

    # Set the viewing angle to see the cup surface better
    ax.view_init(elev=20, azim=45)  # Adjust elevation and azimuth angles
    
    # Set equal scaling for proper proportions
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust the axis limits to zoom in slightly
    ax.set_xlim(points_3d[:, 0].min() * 0.8, points_3d[:, 0].max() * 0.8)
    ax.set_ylim(points_3d[:, 1].min() * 0.8, points_3d[:, 1].max() * 0.8)
    ax.set_zlim(points_3d[:, 2].min() * 0.8, points_3d[:, 2].max() * 0.8)
    
    plt.savefig("pointcloud.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 