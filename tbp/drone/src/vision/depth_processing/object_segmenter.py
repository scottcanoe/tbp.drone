import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy.typing as npt

from configs.paths import SEGMENT_ANYTHING_PATH

sys.path.insert(0, SEGMENT_ANYTHING_PATH)

from segment_anything import SamPredictor, sam_model_registry


class ObjectSegmenter:
    """A class to handle object segmentation using the Segment Anything Model (SAM).

    This class provides functionality to segment objects in images using SAM.
    It supports both CUDA and CPU inference, and can handle various types of prompts
    (points, boxes, or text).
    """

    def __init__(self, model_type: str = "vit_b", model_path: Optional[str] = None):
        """Initialize the object segmenter.

        Args:
            model_type (str): Type of SAM model to use ('vit_b').
            model_path (Optional[str]): Path to the model weights. If None, uses default path.
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model_type = model_type

        if model_path is None:
            model_path = str(
                Path("~/tbp/tbp.drone/models/sam_vit_b_01ec64.pth").expanduser()
            )

        self.model = self._initialize_model(model_path)
        self.predictor = SamPredictor(self.model)

    def _initialize_model(self, model_path: str):
        """Initialize and load the SAM model.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            sam_model_registry: Initialized model.
        """
        model = sam_model_registry[self.model_type](checkpoint=model_path)
        return model.to(self.device)

    def segment_image(
        self,
        image: Union[str, npt.NDArray[np.uint8]],
        input_points: Optional[List[Tuple[float, float]]] = None,
        input_boxes: Optional[List[List[float]]] = None,
        input_labels: Optional[List[int]] = None,
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.uint8]]:
        """Segment objects in an image using various types of prompts.

        Args:
            image (Union[str, np.ndarray]): Input RGB image, either as a file path or numpy array.
            input_points (Optional[List[Tuple[float, float]]]): List of (x, y) point prompts.
            input_boxes (Optional[List[List[float]]]): List of [x1, y1, x2, y2] box prompts.
            input_labels (Optional[List[int]]): Labels for the prompts (1 for foreground, 0 for background).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Binary mask of segmented objects
                - Original RGB image
        """
        if isinstance(image, str):
            data = np.fromfile(Path(image).expanduser(), dtype=np.uint8)
            rgb_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Set the image in the predictor
        self.predictor.set_image(rgb_image)

        # Prepare input prompts
        input_point = np.array(input_points) if input_points is not None else None
        input_box = np.array(input_boxes) if input_boxes is not None else None
        input_label = np.array(input_labels) if input_labels is not None else None

        # Generate masks
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )

        # Return the mask with highest score
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx], rgb_image


def main():
    """Example usage of the ObjectSegmenter class."""
    # Initialize object segmenter
    segmenter = ObjectSegmenter()

    # Process an example image with a point prompt
    image_path = str(Path("~/tbp/tbp.drone/picture.png").expanduser())

    # Example: Click in the center of the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    center_point = [(w / 2, h / 2)]
    center_label = [1]  # 1 indicates foreground

    mask, rgb_image = segmenter.segment_image(
        image_path, input_points=center_point, input_labels=center_label
    )

    # Visualize and save the segmentation
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.plot(center_point[0][0], center_point[0][1], "rx")  # Show clicked point
    plt.title("RGB Image with Point Prompt")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(rgb_image)
    plt.imshow(mask, alpha=0.5)  # Overlay mask on RGB image
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation.png")
    plt.close()


if __name__ == "__main__":
    main()
