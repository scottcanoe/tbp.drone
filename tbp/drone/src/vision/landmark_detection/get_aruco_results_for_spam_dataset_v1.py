import cv2
import os
from pathlib import Path
from aruco_detection import (
    ArucoDetector,
    MARKER_WORLD_POSITIONS,
    MARKER_WORLD_ORIENTATIONS,
)


def process_dataset():
    """Process all PNG images in the spam_dataset_v2 directory for ArUco marker detection."""
    # Initialize ArUco detector
    detector = ArucoDetector()

    # Setup paths
    imgs_dir = os.path.expanduser("~/tbp/tbp.drone/imgs")
    dataset_dir = os.path.join(imgs_dir, "spam_dataset_v2")
    output_dir = os.path.join(dataset_dir, "aruco_results")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all PNG files in the dataset directory
    png_files = list(Path(dataset_dir).glob("*.png"))

    print(f"Found {len(png_files)} PNG files in {dataset_dir}")

    # Process each image
    for img_path in png_files:
        print(f"\nProcessing {img_path.name}...")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue

        print(f"Image shape: {image.shape}")
        print(f"Image type: {image.dtype}")

        # Detect markers
        corners, ids, rejected = detector.detect(image, output_dir)
        print(f"Detected markers: {ids}")

        # Draw markers and pose axes if any were detected
        if ids is not None and len(ids) > 0:
            image = detector.draw_markers_with_pose(image, corners, ids)

            # Get camera poses for each marker
            camera_positions = []
            camera_rotations = []
            marker3_found = False
            marker3_index = None

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                marker_world_pos = MARKER_WORLD_POSITIONS.get(marker_id, None)
                marker_world_orient = MARKER_WORLD_ORIENTATIONS.get(marker_id, None)
                if marker_world_pos is None:
                    continue

                if marker_id == 3:
                    marker3_found = True
                    marker3_index = i
                    print(
                        f"World coordinate of marker 3 center: {MARKER_WORLD_POSITIONS[3]}"
                    )
                    print(
                        f"World orientation of marker 3: {MARKER_WORLD_ORIENTATIONS[3]}"
                    )

                camera_pos, camera_rot = detector.get_camera_pose(
                    corner, marker_world_pos, marker_world_orient
                )
                if camera_pos is not None and camera_rot is not None:
                    camera_positions.append(camera_pos)
                    camera_rotations.append(camera_rot)

            # Calculate and print aggregated pose
            if camera_positions and camera_rotations:
                import numpy as np

                camera_positions = np.array(camera_positions)
                camera_rotations = np.array(camera_rotations)

                if marker3_found:
                    idx = marker3_index
                    agg_pos = camera_positions[idx]
                    agg_rot = camera_rotations[idx]
                else:
                    agg_pos, agg_rot = detector.aggregate_camera_poses(
                        camera_positions, camera_rotations
                    )

                print(
                    "Aggregated Camera Position (using marker 3 if present):", agg_pos
                )
                print("Aggregated Camera Rotation:\n", agg_rot)
                agg_euler = np.degrees(cv2.RQDecomp3x3(agg_rot)[0])
                agg_euler = agg_euler % 360  # Wrap to [0, 360)
                print(
                    f"Aggregated Euler Angles (deg, wrapped): Roll: {agg_euler[0]:.1f}, Pitch: {agg_euler[1]:.1f}, Yaw: {agg_euler[2]:.1f}"
                )
        else:
            print("No markers detected")

        # Save the result image
        output_path = os.path.join(output_dir, f"result_{img_path.name}")
        cv2.imwrite(output_path, image)
        print(f"Saved result to {output_path}")


if __name__ == "__main__":
    process_dataset()
