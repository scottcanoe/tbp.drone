import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tbp.drone.src.vision.landmark_detection.camera_intrinsics import (
    camera_matrix,
    dist_coeffs,
)
import scipy.spatial.transform

# --- Marker world positions relative to cube center (meters) ---
CUBE_LENGTH = 2.25 * 0.0254  # meters
HALF = CUBE_LENGTH / 2
MARKER_WORLD_POSITIONS = {
    3: np.array([0.0, 0.0, -HALF]),  # Marker 3 at -Z
    4: np.array([-HALF, 0.0, 0.0]),  # Marker 4 at -X
    1: np.array([0.0, 0.0, +HALF]),  # Marker 1 at +Z
    2: np.array([+HALF, 0.0, 0.0]),  # Marker 2 at +X
}
MARKER_WORLD_ORIENTATIONS = {
    3: np.array([0.0, np.pi, 0.0]),  # Marker 3 at -Z (rotated 180° around Y)
    4: np.array([0.0, -np.pi / 2, 0.0]),  # Marker 4 at -X (rotated -90° around Y)
    1: np.array([0.0, np.pi, 0.0]),  # Marker 1 at +Z (no rotation needed)
    2: np.array([0.0, np.pi / 2, 0.0]),  # Marker 2 at +X (rotated 90° around Y)
}


class ArucoDetector:
    def __init__(self, marker_size=0.05):
        """Initialize ArUco detector.

        Args:
            dictionary_name (str): ArUco dictionary to use
            marker_size (float): Physical size of the marker in meters
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        params = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

        # Camera parameters (imported from camera_intrinsics.py)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size

    def get_marker_center(self, corner):
        """Calculate the center point of a marker.

        Args:
            corner: Corner points of the marker

        Returns:
            tuple: (x, y) coordinates of the center point
        """
        corner_points = corner.reshape((4, 2))
        center_x = np.mean(corner_points[:, 0])
        center_y = np.mean(corner_points[:, 1])
        return (center_x, center_y)

    def detect(self, image, output_dir):
        """Detect ArUco markers in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        return corners, ids, rejected

    def get_camera_pose(
        self, corner, marker_world_position, marker_world_orientation=None
    ):
        """Calculate absolute camera pose given marker's world position and orientation.

        Args:
            corner: Corner points of the detected marker
            marker_world_position: [x, y, z] world position of the marker's center
            marker_world_orientation: Optional [rx, ry, rz] world orientation of marker

        Returns:
            tuple: (camera_position, camera_orientation)
                - camera_position: [x, y, z] absolute position of camera
                - camera_orientation: 3x3 rotation matrix of camera orientation
        """
        # Get marker pose relative to camera
        objPoints = np.array(
            [
                [-self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0],
            ]
        )

        imgPoints = corner.reshape(4, 2)

        # Get marker's pose relative to camera
        success, rvec, tvec = cv2.solvePnP(
            objPoints,
            imgPoints,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            return None, None

        # Convert rotation vector to matrix
        R_marker_to_cam, _ = cv2.Rodrigues(rvec)

        # Get marker world orientation from dictionary if not provided
        if marker_world_orientation is None:
            marker_id = None
            # Find the marker ID by comparing world positions
            for id, pos in MARKER_WORLD_POSITIONS.items():
                if np.allclose(pos, marker_world_position):
                    marker_id = id
                    break
            if marker_id is not None and marker_id in MARKER_WORLD_ORIENTATIONS:
                marker_world_orientation = MARKER_WORLD_ORIENTATIONS[marker_id]
            else:
                # Default to no rotation if marker ID not found
                marker_world_orientation = np.array([0.0, 0.0, 0.0])

        # Convert marker world orientation to rotation matrix
        R_world_to_marker, _ = cv2.Rodrigues(np.array(marker_world_orientation))

        # Calculate camera rotation in world coordinates
        # R_cam_to_world = R_world_to_marker.T @ R_marker_to_cam.T
        R_cam_to_world = R_marker_to_cam.T @ R_world_to_marker

        # Calculate camera position in world coordinates
        camera_position = np.array(
            marker_world_position
        ) - R_cam_to_world @ tvec.reshape(3)

        return camera_position, R_cam_to_world

    def aggregate_camera_poses(self, camera_positions, camera_rotations):
        """
        Aggregate multiple camera poses (positions and rotations) into a single pose.
        Uses mean for positions and quaternion averaging for rotations.
        """
        avg_position = np.mean(camera_positions, axis=0)
        quats = scipy.spatial.transform.Rotation.from_matrix(camera_rotations).as_quat()
        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)
        avg_rot = scipy.spatial.transform.Rotation.from_quat(avg_quat).as_matrix()
        return avg_position, avg_rot

    def draw_markers_with_pose(self, image, corners, ids):
        """Draw detected markers and their pose axes on the image."""
        if ids is None:
            return image

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Text position for camera info
        text_x = 30
        text_y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Restored to original size
        font_thickness = 2  # Restored to original thickness

        camera_positions = []
        camera_rotations = []
        marker3_found = False
        marker3_index = None

        # Calculate total height needed per marker
        line_height = 30
        marker_info_height = line_height * 11  # 11 lines of text per marker

        # Estimate total height needed
        total_height_needed = len(corners) * marker_info_height

        # If total height would exceed image height, adjust starting position
        if total_height_needed > image.shape[0]:
            text_y = max(30, image.shape[0] - total_height_needed)

        # Process each marker
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            marker_world_pos = MARKER_WORLD_POSITIONS.get(
                marker_id, np.array([0.0, 0.0, 0.0])
            )
            marker_world_orient = MARKER_WORLD_ORIENTATIONS.get(
                marker_id, np.array([0.0, 0.0, 0.0])
            )

            # Get rotation and translation vectors
            objPoints = np.array(
                [
                    [-self.marker_size / 2, self.marker_size / 2, 0],
                    [self.marker_size / 2, self.marker_size / 2, 0],
                    [self.marker_size / 2, -self.marker_size / 2, 0],
                    [-self.marker_size / 2, -self.marker_size / 2, 0],
                ]
            )

            # Reshape corner points for solvePnP
            imgPoints = corner.reshape(4, 2)

            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(
                objPoints,
                imgPoints,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            if success:
                if marker_id == 3:
                    marker3_found = True
                    marker3_index = i
                    print(
                        f"World coordinate of marker 3 center: {MARKER_WORLD_POSITIONS[3]}"
                    )
                    print(
                        f"World orientation of marker 3: {MARKER_WORLD_ORIENTATIONS[3]}"
                    )

                camera_pos, camera_rot = self.get_camera_pose(
                    corner, marker_world_pos, marker_world_orient
                )
                if camera_pos is not None and camera_rot is not None:
                    camera_positions.append(camera_pos)
                    camera_rotations.append(camera_rot)

                # Calculate starting y position for this marker's text block
                current_text_y = int(text_y + i * marker_info_height)

                # Draw camera position and rotation on image
                cv2.putText(
                    image,
                    f"Marker ID: {marker_id}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                cv2.putText(
                    image,
                    f"Camera Position (m):",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                cv2.putText(
                    image,
                    f"X: {camera_pos[0]:.3f}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                cv2.putText(
                    image,
                    f"Y: {camera_pos[1]:.3f}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                cv2.putText(
                    image,
                    f"Z: {camera_pos[2]:.3f}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += int(
                    line_height * 1.5
                )  # Extra spacing before rotation

                cv2.putText(
                    image,
                    f"Camera Rotation (deg):",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                euler_angles = scipy.spatial.transform.Rotation.from_matrix(
                    camera_rot
                ).as_euler("xyz", degrees=True)
                cv2.putText(
                    image,
                    f"Roll: {euler_angles[0]:.1f}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                cv2.putText(
                    image,
                    f"Pitch: {euler_angles[1]:.1f}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                current_text_y += line_height

                cv2.putText(
                    image,
                    f"Yaw: {euler_angles[2]:.1f}",
                    (text_x, current_text_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )

        # Calculate and draw aggregated pose
        if camera_positions and camera_rotations:
            camera_positions = np.array(camera_positions)
            camera_rotations = np.array(camera_rotations)

            if marker3_found:
                idx = marker3_index
                agg_pos = camera_positions[idx]
                agg_rot = camera_rotations[idx]
            else:
                agg_pos, agg_rot = self.aggregate_camera_poses(
                    camera_positions, camera_rotations
                )

            # Convert aggregated rotation matrix to rotation vector
            agg_rvec, _ = cv2.Rodrigues(agg_rot)

            # Convert aggregated position to translation vector (negative because we want camera-to-world)
            agg_tvec = -agg_rot.T @ agg_pos

            # Draw aggregated pose axis
            cv2.drawFrameAxes(
                image,
                self.camera_matrix,
                self.dist_coeffs,
                agg_rvec,
                agg_tvec.reshape(3, 1),
                self.marker_size
                * 2.0,  # Make the aggregated axis larger for visibility
            )

            # Draw aggregated pose on the right side
            right_text_x = int(image.shape[1] - 300)  # 300 pixels from right edge
            right_text_y = 30
            line_height = 30

            cv2.putText(
                image,
                "Aggregated Camera Pose:",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += int(line_height * 1.5)

            cv2.putText(
                image,
                "Position (m):",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += line_height

            cv2.putText(
                image,
                f"X: {agg_pos[0]:.3f}",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += line_height

            cv2.putText(
                image,
                f"Y: {agg_pos[1]:.3f}",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += line_height

            cv2.putText(
                image,
                f"Z: {agg_pos[2]:.3f}",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += int(line_height * 1.5)

            cv2.putText(
                image,
                "Rotation (deg):",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += line_height

            agg_euler = scipy.spatial.transform.Rotation.from_matrix(agg_rot).as_euler(
                "xyz", degrees=True
            )

            cv2.putText(
                image,
                f"Roll: {agg_euler[0]:.1f}",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += line_height

            cv2.putText(
                image,
                f"Pitch: {agg_euler[1]:.1f}",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )
            right_text_y += line_height

            cv2.putText(
                image,
                f"Yaw: {agg_euler[2]:.1f}",
                (right_text_x, right_text_y),
                font,
                font_scale,
                (255, 0, 0),
                font_thickness,
            )

            print("Aggregated Camera Position:\n", agg_pos)
            print("Aggregated Camera Rotation:\n", agg_rot)
            print(
                f"Aggregated Euler Angles (deg, wrapped): Roll: {agg_euler[0]:.1f}, Pitch: {agg_euler[1]:.1f}, Yaw: {agg_euler[2]:.1f}"
            )

        return image

    def draw_rejected_markers(self, image, rejected):
        """Draw rejected marker candidates on the image in red."""
        if rejected is None or len(rejected) == 0:
            return image

        # Draw each rejected marker in red
        for corners in rejected:
            # Convert to integer coordinates
            corners = corners.reshape((4, 2)).astype(np.int32)

            # Draw the contour
            cv2.polylines(
                image,
                [corners],
                isClosed=True,
                color=(0, 0, 255),  # Red color
                thickness=2,
            )

            # Calculate center as mean of all corner points
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            size = 10

            # Draw X at the center
            cv2.line(
                image,
                (center_x - size, center_y - size),
                (center_x + size, center_y + size),
                (0, 0, 255),
                2,
            )
            cv2.line(
                image,
                (center_x + size, center_y - size),
                (center_x - size, center_y + size),
                (0, 0, 255),
                2,
            )

        return image


def main():
    # Create detector instance
    detector = ArucoDetector()

    # Setup paths
    imgs_dir = os.path.expanduser("~/tbp/tbp.drone/imgs")
    image_path = os.path.join(imgs_dir, "tello_aruco_img.png")
    # image_path = os.path.join(imgs_dir, "singlemarkersoriginal.jpg")
    # image_path = os.path.join(imgs_dir, "aruco_spam_img3.png")
    image_path = os.path.join(imgs_dir, "aruco_phone.jpg")
    image_path = os.path.join(imgs_dir, "aruco_2_img2_5in.png")
    image_path = os.path.join(imgs_dir, "spam.png")
    image_path = "/Users/hlee/tbp/tbp.drone/imgs/spam_dataset_v2/aruco_2_spam_img2_roll0_pitch2_yaw0_location6.png"

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    print(f"Image shape: {image.shape}")
    print(f"Image type: {image.dtype}")

    # Detect markers
    corners, ids, _ = detector.detect(image, imgs_dir)
    print(f"Detected markers: {ids}")

    # Draw markers and pose axes if any were detected
    if ids is not None and len(ids) > 0:
        image = detector.draw_markers_with_pose(image, corners, ids)
    else:
        print("No markers detected")

    # Draw rejected markers
    # image = detector.draw_rejected_markers(image, rejected)

    # Save the result image
    cv2.imwrite(os.path.join(imgs_dir, "result.png"), image)


if __name__ == "__main__":
    main()
