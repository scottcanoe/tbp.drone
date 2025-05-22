import cv2
import numpy as np
import os
import sys
import torch
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry

class ArucoDetector:
    def __init__(self, dictionary_name="DICT_6X6_50", marker_size=0.05):
        """Initialize ArUco detector and SAM model.
        
        Args:
            dictionary_name (str): ArUco dictionary to use
            marker_size (float): Physical size of the marker in meters
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            cv2.aruco.DetectorParameters()
        )
        
        # Initialize SAM
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model_path = str(Path("~/tbp/tbp.drone/models/sam_vit_b_01ec64.pth").expanduser())
        self.sam = sam_model_registry["vit_b"](checkpoint=model_path)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Camera parameters (these are example values - adjust for your camera)
        self.camera_matrix = np.array([
            [921.170702, 0.000000, 459.904354],
            [0.000000, 919.018377, 351.238301],
            [0.000000, 0.000000, 1.000000]
        ])
        self.dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
        self.marker_size = marker_size

    def generate_marker(self, marker_id: int, size_pixels: int = 200) -> np.ndarray:
        """Generate an ArUco marker image.
        
        Args:
            marker_id: ID of the marker to generate
            size_pixels: Size of the output marker image in pixels
            
        Returns:
            np.ndarray: Generated marker image
        """
        marker_image = np.zeros((size_pixels, size_pixels), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(
            self.aruco_dict,
            marker_id,
            size_pixels,
            marker_image,
            1
        )
        return marker_image

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
        """Detect ArUco markers in the image using SAM pre-segmentation."""
        # Convert to RGB for SAM
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in SAM predictor
        self.predictor.set_image(rgb_image)
        
        # Use center point of image as prompt
        h, w = image.shape[:2]
        center_point = np.array([[w/2, h/2]])
        
        # Create input label (foreground)
        input_label = np.array([1])
        
        # Get segmentation masks
        masks, scores, _ = self.predictor.predict(
            point_coords=center_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Take the highest scoring mask
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        # Save visualization of point and segmentation
        vis_image = image.copy()
        # Draw center point
        cv2.circle(vis_image, (int(w/2), int(h/2)), 5, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(output_dir, "center_point.png"), vis_image)
        
        # Create segmentation overlay
        overlay = vis_image.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imwrite(os.path.join(output_dir, "segmentation_overlay.png"), overlay)
        
        # Apply mask to create segmented image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Create binary image only in segmented regions
        segmented_gray = np.zeros_like(gray)
        segmented_gray[mask] = gray[mask]
        
        # Save intermediate results for debugging
        cv2.imwrite(os.path.join(output_dir, "segmented_gray.png"), segmented_gray)
        
        # Try to detect markers in the processed image
        corners, ids, rejected = self.detector.detectMarkers(segmented_gray)
        
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
        R_world_to_cam = R_marker_to_cam @ R_world_to_marker
        
        # Calculate camera position in world coordinates
        # p_world = p_marker_world - R_world_to_cam.T @ t_cam_to_marker
        camera_position = np.array(marker_world_position) - R_world_to_cam.T @ tvec.reshape(3)
        
        return camera_position, R_world_to_cam

    def draw_markers_with_pose(self, image, corners, ids):
        """Draw detected markers and their pose axes on the image."""
        if ids is None:
            return image
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
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
                
                # Print pose information
                print(f"\nMarker {ids[i][0]} pose:")
                print(f"Translation (x,y,z): {tvec.flatten()}")
                print(f"Rotation vector: {rvec.flatten()}")
                
                # Calculate rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                print(f"Rotation matrix:\n{rmat}")
        
        return image

    def segment_marker(self, image, corners, ids, target_id=0):
        """Segment a specific ArUco marker using SAM.
        
        Args:
            image: Input RGB image
            corners: Detected marker corners
            ids: Marker IDs
            target_id: ID of the marker to segment (default: 0)
            
        Returns:
            tuple: (mask, rgb_image) - Binary mask and original image
        """
        if ids is None:
            return None, image

        # Find the marker with target_id
        target_idx = np.where(ids == target_id)[0]
        if len(target_idx) == 0:
            return None, image

        # Get the center point of the target marker
        target_corner = corners[target_idx[0]]
        center_point = self.get_marker_center(target_corner)
        
        # Use SAM to segment the marker
        mask, rgb_image = self.segmenter.segment_image(
            image,
            input_points=[(center_point[0], center_point[1])],
            input_labels=[1]  # 1 for foreground
        )
        
        return mask, rgb_image

def main():
    # Create detector instance
    detector = ArucoDetector()
    
    # Setup paths
    imgs_dir = os.path.expanduser("~/tbp/tbp.drone/imgs")
    image_path = os.path.join(imgs_dir, "tello_aruco_img.png")
    
    # Generate and save ArUco marker with ID 0
    marker_image = detector.generate_marker(0)
    cv2.imwrite(os.path.join(imgs_dir, "aruco_marker_id0.png"), marker_image)
    
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
    
    # Draw markers and pose axes if any were detected
    if ids is not None and len(ids) > 0:
        image = detector.draw_markers_with_pose(image, corners, ids)
    else:
        print("No markers detected")
        
    # Save the result image
    cv2.imwrite(os.path.join(imgs_dir, "result.png"), image)

if __name__ == "__main__":
    main()