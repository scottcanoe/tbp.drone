import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tbp.drone.src.vision.landmark_detection.camera_intrinsics import camera_matrix, dist_coeffs
import scipy.spatial.transform

# --- Marker world positions relative to cube center (meters) ---
CUBE_LENGTH = 2.25 * 0.0254  # meters
HALF = CUBE_LENGTH / 2
MARKER_WORLD_POSITIONS = {
    3: np.array([0.0, 0.0, -HALF]),  # Marker 3 at +Z
    4: np.array([-HALF, 0.0, 0.0]),  # Marker 4 at +X
    1: np.array([0.0, 0.0, +HALF]),  # Marker 1 at -Z
    2: np.array([+HALF, 0.0, 0.0]),  # Marker 2 at -X
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

        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            params
        )
        
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

    def get_camera_pose(self, corner, marker_world_position, marker_world_orientation=None):
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
        objPoints = np.array([
            [-self.marker_size/2, self.marker_size/2, 0],
            [self.marker_size/2, self.marker_size/2, 0],
            [self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]
        ])
        
        imgPoints = corner.reshape(4, 2)
        
        # Get marker's pose relative to camera
        success, rvec, tvec = cv2.solvePnP(
            objPoints,
            imgPoints,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if not success:
            return None, None
            
        # Convert rotation vector to matrix
        R_marker_to_cam, _ = cv2.Rodrigues(rvec)
        
        # If marker world orientation is provided, use it
        if marker_world_orientation is not None:
            R_world_to_marker, _ = cv2.Rodrigues(np.array(marker_world_orientation))
        else:
            # Assume marker is aligned with world coordinates
            R_world_to_marker = np.eye(3)
        
        # Calculate camera rotation in world coordinates
        R_cam_to_world = R_world_to_marker.T @ R_marker_to_cam.T
        
        # Calculate camera position in world coordinates
        # p_world = p_marker_world - R_world_to_cam.T @ t_cam_to_marker
        camera_position = np.array(marker_world_position) - R_cam_to_world @ tvec.reshape(3)
        
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
        font_scale = 0.7
        font_thickness = 2
        
        camera_positions = []
        camera_rotations = []
        marker3_found = False
        marker3_index = None
        
        # Estimate pose for each marker
        for i, corner in enumerate(corners):
            # Get rotation and translation vectors
            objPoints = np.array([
                [-self.marker_size/2, self.marker_size/2, 0],
                [self.marker_size/2, self.marker_size/2, 0],
                [self.marker_size/2, -self.marker_size/2, 0],
                [-self.marker_size/2, -self.marker_size/2, 0]
            ])
            
            # Reshape corner points for solvePnP
            imgPoints = corner.reshape(4, 2)
            
            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(
                objPoints,
                imgPoints,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                # Draw axis for each marker
                cv2.drawFrameAxes(
                    image,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_size * 0.5  # Length of the axes
                )
                
                # Use correct world position for each marker
                marker_id = ids[i][0]
                marker_world_pos = MARKER_WORLD_POSITIONS.get(marker_id, np.array([0.0, 0.0, 0.0]))
                if marker_id == 3:
                    marker3_found = True
                    marker3_index = i
                    print(f"World coordinate of marker 3 center: {MARKER_WORLD_POSITIONS[3]}")
                camera_pos, camera_rot = self.get_camera_pose(corner, marker_world_pos)
                if camera_pos is not None and camera_rot is not None:
                    camera_positions.append(camera_pos)
                    camera_rotations.append(camera_rot)
                
                # Draw camera position and rotation on image
                cv2.putText(image, f"Marker ID: {marker_id}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                cv2.putText(image, f"Camera Position (m):", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                cv2.putText(image, f"X: {camera_pos[0]:.3f}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                cv2.putText(image, f"Y: {camera_pos[1]:.3f}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                cv2.putText(image, f"Z: {camera_pos[2]:.3f}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 45
                 
                cv2.putText(image, f"Camera Rotation (deg):", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                euler_angles = np.degrees(cv2.RQDecomp3x3(camera_rot)[0])
                euler_angles = euler_angles % 360  # Wrap to [0, 360)
                cv2.putText(image, f"Roll: {euler_angles[0]:.1f}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                cv2.putText(image, f"Pitch: {euler_angles[1]:.1f}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                text_y += 30
                
                cv2.putText(image, f"Yaw: {euler_angles[2]:.1f}", 
                          (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
        
        # --- Aggregate pose using marker 3 as origin if present ---
        if camera_positions and camera_rotations:
            camera_positions = np.array(camera_positions)
            camera_rotations = np.array(camera_rotations)
            if marker3_found:
                # Use only marker 3 for pose
                idx = marker3_index
                agg_pos = camera_positions[idx]
                agg_rot = camera_rotations[idx]
            else:
                # Aggregate all
                agg_pos, agg_rot = self.aggregate_camera_poses(camera_positions, camera_rotations)
            # Annotate aggregated pose
            text_y += 30
            cv2.putText(image, f"Aggregated Camera Pose (using marker 3 if present):", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            text_y += 30
            cv2.putText(image, f"X: {agg_pos[0]:.3f}", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            text_y += 30
            cv2.putText(image, f"Y: {agg_pos[1]:.3f}", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            text_y += 30
            cv2.putText(image, f"Z: {agg_pos[2]:.3f}", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            text_y += 30
            agg_euler = np.degrees(cv2.RQDecomp3x3(agg_rot)[0])
            agg_euler = agg_euler % 360  # Wrap to [0, 360)
            cv2.putText(image, f"Roll: {agg_euler[0]:.1f}", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            text_y += 30
            cv2.putText(image, f"Pitch: {agg_euler[1]:.1f}", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            text_y += 30
            cv2.putText(image, f"Yaw: {agg_euler[2]:.1f}", (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
        
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
                thickness=2
            )
            
            # Calculate center as mean of all corner points
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            size = 10
            
            # Draw X at the center
            cv2.line(image,
                    (center_x - size, center_y - size),
                    (center_x + size, center_y + size),
                    (0, 0, 255),
                    2)
            cv2.line(image,
                    (center_x + size, center_y - size),
                    (center_x - size, center_y + size),
                    (0, 0, 255),
                    2)
        
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
    corners, ids, rejected = detector.detect(image, imgs_dir)
    print(f"Detected markers: {ids}")
    print(f"Number of rejected markers: {len(rejected)}")
    if len(rejected) > 0:
        print("Rejected marker corners:")
        for i, rej in enumerate(rejected):
            print(f"  Rejected marker {i}: {rej.shape}")
    
    # Draw markers and pose axes if any were detected
    if ids is not None and len(ids) > 0:
        image = detector.draw_markers_with_pose(image, corners, ids)
        # Also print the aggregated pose to console
        camera_positions = []
        camera_rotations = []
        marker3_found = False
        marker3_index = None
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            marker_world_pos = MARKER_WORLD_POSITIONS.get(marker_id, np.array([0.0, 0.0, 0.0]))
            if marker_id == 3:
                marker3_found = True
                marker3_index = i
                print(f"World coordinate of marker 3 center: {MARKER_WORLD_POSITIONS[3]}")
            camera_pos, camera_rot = detector.get_camera_pose(corner, marker_world_pos)
            if camera_pos is not None and camera_rot is not None:
                camera_positions.append(camera_pos)
                camera_rotations.append(camera_rot)
        if camera_positions and camera_rotations:
            camera_positions = np.array(camera_positions)
            camera_rotations = np.array(camera_rotations)
            if marker3_found:
                idx = marker3_index
                agg_pos = camera_positions[idx]
                agg_rot = camera_rotations[idx]
            else:
                agg_pos, agg_rot = detector.aggregate_camera_poses(camera_positions, camera_rotations)
            print("Aggregated Camera Position (using marker 3 if present):", agg_pos)
            print("Aggregated Camera Rotation:\n", agg_rot)
            agg_euler = np.degrees(cv2.RQDecomp3x3(agg_rot)[0])
            agg_euler = agg_euler % 360  # Wrap to [0, 360)
            print(f"Aggregated Euler Angles (deg, wrapped): Roll: {agg_euler[0]:.1f}, Pitch: {agg_euler[1]:.1f}, Yaw: {agg_euler[2]:.1f}")
    else:
        print("No markers detected")
    
    # Draw rejected markers
    # image = detector.draw_rejected_markers(image, rejected)
        
    # Save the result image
    cv2.imwrite(os.path.join(imgs_dir, "result.png"), image)

if __name__ == "__main__":
    main()