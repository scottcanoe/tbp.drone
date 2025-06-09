import cv2
import os
from pathlib import Path


def convert_to_rgb(input_dir, output_dir):
    """
    Convert all images in input_dir to RGB format and save them to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")

    # Get all image files in the input directory
    image_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)
    ]

    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"rgb_{image_file}")

        # Read image in BGR format
        bgr_image = cv2.imread(input_path)
        if bgr_image is None:
            print(f"Could not read image: {input_path}")
            continue

        # Convert to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Save in RGB format
        cv2.imwrite(output_path, rgb_image)
        print(f"Converted and saved: {output_path}")


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/hlee/tbp/tbp.drone/imgs/spam_v3"
    output_dir = "/Users/hlee/tbp/tbp.drone/imgs/spam_v3/rgb_converted"

    convert_to_rgb(input_dir, output_dir)
    print("Conversion complete!")
