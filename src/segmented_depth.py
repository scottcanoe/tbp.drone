import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy.typing as npt

from src.depth_estimation import DepthEstimator
from src.object_segmentation import ObjectSegmenter

class SegmentedDepth:
    """A class that combines depth estimation and object segmentation.
    
    This class provides functionality to create a depth map where the background
    (non-object regions) is set to a maximum depth value. It uses DepthAnything V2
    for depth estimation and SAM for object segmentation.
    """
    
    def __init__(
        self,
        max_depth: float = 100.0,
        depth_model_path: Optional[str] = None,
        sam_model_path: Optional[str] = None,
    ):
        """Initialize the segmented depth estimator.
        
        Args:
            max_depth (float): Value to use for background depth. Defaults to 10.0.
            depth_model_path (Optional[str]): Path to depth model weights.
            sam_model_path (Optional[str]): Path to SAM model weights.
        """
        self.max_depth = max_depth
        self.depth_estimator = DepthEstimator(model_path=depth_model_path)
        self.object_segmenter = ObjectSegmenter(model_path=sam_model_path)
    
    def process_image(
        self,
        image: Union[str, npt.NDArray[np.uint8]],
        input_points: Optional[List[Tuple[float, float]]] = None,
        input_boxes: Optional[List[List[float]]] = None,
        input_labels: Optional[List[int]] = None,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.uint8]]:
        """Process an image to create a segmented depth map.
        
        Args:
            image (Union[str, np.ndarray]): Input RGB image, either as file path or numpy array.
            input_points (Optional[List[Tuple[float, float]]]): List of (x, y) point prompts.
            input_boxes (Optional[List[List[float]]]): List of [x1, y1, x2, y2] box prompts.
            input_labels (Optional[List[int]]): Labels for the prompts (1 for foreground, 0 for background).
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - Modified depth map with background set to max_depth
                - Binary segmentation mask
                - Original RGB image
        """
        # Get depth map
        depth_map, rgb_image = self.depth_estimator.estimate_depth(image)
        
        # Get segmentation mask
        mask, _ = self.object_segmenter.segment_image(
            rgb_image,
            input_points=input_points,
            input_boxes=input_boxes,
            input_labels=input_labels
        )
        
        # Create modified depth map where background (non-object) has max depth
        modified_depth = depth_map.copy()
        modified_depth[~mask] = self.max_depth
        
        return modified_depth, mask, rgb_image

def main():
    """Example usage of the SegmentedDepth class."""
    # Initialize segmented depth processor
    processor = SegmentedDepth()
    
    # Process an example image with a point prompt
    image_path = str(Path("~/tbp/tbp.drone/picture.png").expanduser())
    
    # Example: Click in the center of the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    center_point = [(w/2, h/2)]
    center_label = [1]  # 1 indicates foreground
    
    modified_depth, mask, rgb_image = processor.process_image(
        image_path,
        input_points=center_point,
        input_labels=center_label
    )
    
    # Visualize and save the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.plot(center_point[0][0], center_point[0][1], 'rx')  # Show clicked point
    plt.title("RGB Image with Point Prompt")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Segmentation Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(modified_depth)
    plt.colorbar(label='Depth')
    plt.title("Segmented Depth Map")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("segmented_depth.png")
    plt.close()

if __name__ == "__main__":
    main() 