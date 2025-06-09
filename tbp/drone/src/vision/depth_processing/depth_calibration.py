import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from depth_estimator import DepthEstimator

class DepthCalibrator:
    def __init__(self, image_path: str):
        self.depth_estimator = DepthEstimator()
        self.image_path = str(Path(image_path).expanduser())
        self.depth_map, self.rgb_image = self.depth_estimator.estimate_depth(self.image_path)
        self.calibration_points = []  # List of (x, y, depth) tuples
        self.current_display = None
        
    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the raw depth value at the clicked point
            raw_depth = self.depth_map[y, x]
            
            # Prompt for true depth
            depth_cm = float(input(f"Enter true depth in cm for point ({x}, {y}), raw depth is {raw_depth:.2f}: "))
            self.calibration_points.append((x, y, raw_depth, depth_cm))
            
            # Draw the point and update display
            cv2.circle(self.current_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(self.current_display, f"{depth_cm}cm", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Depth Calibration', self.current_display)

            # Close after 3 points
            if len(self.calibration_points) >= 3:
                cv2.destroyAllWindows()

    def calibrate_depth_map(self):
        if len(self.calibration_points) < 2:
            print("Need at least 2 calibration points!")
            return self.depth_map
        
        # Extract raw depths and true depths
        raw_depths = np.array([p[2] for p in self.calibration_points])
        true_depths = np.array([p[3] for p in self.calibration_points])
        
        # Fit linear regression
        coeffs = np.polyfit(raw_depths, true_depths, 1)
        
        # Apply calibration to entire depth map
        calibrated_depth = coeffs[0] * self.depth_map + coeffs[1]
        return calibrated_depth

    def run_calibration(self):
        # Normalize depth map for display
        depth_display = cv2.normalize(self.depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
        
        # Create side-by-side display
        self.current_display = np.hstack((cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB), depth_colored))
        
        cv2.namedWindow('Depth Calibration')
        cv2.setMouseCallback('Depth Calibration', self.click_callback)
        
        print("Click 3 points on the image and enter their true depths in cm.")
        print("Window will close automatically after 3 points.")
        
        while len(self.calibration_points) < 3:
            cv2.imshow('Depth Calibration', self.current_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key to exit early
                break
        
        cv2.destroyAllWindows()
        
        # Calibrate and save results
        if self.calibration_points:
            calibrated_depth = self.calibrate_depth_map()
            
            # Save visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
            plt.title("RGB Image")
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(self.depth_map)
            plt.colorbar(label='Raw Depth')
            plt.title("Raw Depth Map")
            
            plt.subplot(133)
            plt.imshow(calibrated_depth)
            plt.colorbar(label='Depth (cm)')
            plt.title("Calibrated Depth Map")
            
            # Plot calibration points
            for x, y, _, depth_cm in self.calibration_points:
                plt.plot(x, y, 'r+', markersize=10)
                plt.annotate(f"{depth_cm}cm", (x+5, y+5))
            
            plt.tight_layout()
            plt.savefig("calibrated_depth.png")
            plt.close()
            
            print(f"Calibration points: {self.calibration_points}")
            print("Results saved as calibrated_depth.png")
            
            return calibrated_depth
        return None

def main():
    image_path = str(Path("~/tbp/tbp.drone/imgs/depth_estimation_v2.png").expanduser())
    calibrator = DepthCalibrator(image_path)
    calibrated_depth = calibrator.run_calibration()

if __name__ == "__main__":
    main() 