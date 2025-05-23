import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from configs.paths import DEPTH_ANYTHING_PATH

sys.path.insert(0, DEPTH_ANYTHING_PATH)

from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    """A class to handle depth estimation using the Depth Anything V2 model.
    
    This class provides functionality to estimate depth from RGB images using the 
    Depth Anything V2 model. It supports both CUDA and CPU inference.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the depth estimator.
        
        Args:
            model_path (Optional[str]): Path to the model weights. If None, uses default path.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_configs = {
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
        }
        self.encoder = "vitb"
        self.dataset = "hypersim" # 'hypersim' for indoor model, 'vkitti' for outdoor model
        # self.max_depth = 20 # 20 for indoor model, 80 for outdoor model
        
        if model_path is None:
            # model_path = str(Path(f"~/tbp/tbp.drone/models/depth_anything_v2_metric_{self.dataset}_{self.encoder}.pth").expanduser())
            model_path = str(Path("~/tbp/tbp.drone/models/depth_anything_v2_vitb.pth").expanduser())
        self.model = self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str) -> DepthAnythingV2:
        """Initialize and load the Depth Anything V2 model.
        
        Args:
            model_path (str): Path to the model weights.
            
        Returns:
            DepthAnythingV2: Initialized model.
        """
        model = DepthAnythingV2(**{**self.model_configs[self.encoder]})
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()
    
    def estimate_depth(self, image: Union[str, npt.NDArray[np.uint8]]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Estimate depth from an RGB image.
        
        Args:
            image (Union[str, np.ndarray]): Input RGB image, either as a file path or numpy array.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Depth map as a 2D float32 array
                - Original RGB image as uint8 array
        """
        rgb_image = self._as_ndarray(image)
        with torch.no_grad():
            depth_map = self.model.infer_image(rgb_image) # outputs "affine invariant" INVERSE depth map
            depth_map = 1.0 / depth_map
            # use interquartile range to keep center 95% of the data
            q1 = np.percentile(depth_map, 25)
            q3 = np.percentile(depth_map, 75)
            depth_map = np.where(depth_map < q1, q1, depth_map)
            depth_map = np.where(depth_map > q3, q3, depth_map)
            print(depth_map.min(), depth_map.max())
            # fill nans with q3
            depth_map = np.where(np.isnan(depth_map), q3, depth_map)
            
        return depth_map, rgb_image

    def _as_ndarray(
        self, image: Union[str, npt.NDArray[np.uint8]]
    ) -> npt.NDArray[np.uint8]:
        if isinstance(image, os.PathLike):
            data = np.fromfile(Path(image).expanduser(), dtype=np.uint8)
            rgb_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            rgb_image = image
        return rgb_image

    def __call__(
        self,
        image: Union[os.PathLike, npt.NDArray[np.uint8]],
        meters: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Estimate depth from an RGB image.

        Args:
            image (Union[str, np.ndarray]): Input RGB image, either as a file path
            or numpy array.
            meters (bool): Whether to convert the depth map to meters.

        Returns:
            npt.NDArray[np.float32]: Depth map as a 2D float32 array

        """
        rgb_image = self._as_ndarray(image)
        with torch.no_grad():
            depth_map = self.model.infer_image(rgb_image)
        if meters:
            depth_map = self.to_meters(depth_map)
        return depth_map

    @staticmethod
    def to_meters(
        depth_map: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Correct the depth map."""
        slope = -0.06080131811283916
        intercept = 0.6446867000355985
        return slope * depth_map + intercept


def main():
    """Example usage of the DepthEstimator class."""
    # Initialize depth estimator
    depth_estimator = DepthEstimator()
    
    # Process an example image
    image_path = str(Path("~/tbp/tbp.drone/picture.png").expanduser())
    depth_map, rgb_image = depth_estimator.estimate_depth(image_path)
    print(depth_map.shape) # (720, 960)

    # Visualize and save the depth map
    import matplotlib.pyplot as plt
    

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title("RGB Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map)
    plt.colorbar(label='Depth')
    plt.title("Depth Map")
    
    plt.tight_layout()
    plt.savefig("depth.png")
    plt.close()

if __name__ == "__main__":
    main()