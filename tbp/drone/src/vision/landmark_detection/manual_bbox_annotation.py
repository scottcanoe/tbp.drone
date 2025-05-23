import cv2
import os
import json
from pathlib import Path
import numpy as np
import argparse

class BBoxAnnotator:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.bbox = []
        self.current_image = None
        self.temp_image = None
        
    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_image = self.current_image.copy()
                cv2.rectangle(self.temp_image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Draw final rectangle
            cv2.rectangle(self.current_image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            # Store coordinates as [x1, y1, x2, y2]
            x1, x2 = min(self.ix, x), max(self.ix, x)
            y1, y2 = min(self.iy, y), max(self.iy, y)
            self.bbox = [x1, y1, x2, y2]

def process_dataset():
    """Process all PNG images in the dataset directory for manual bounding box annotation."""
    # Setup paths
    imgs_dir = os.path.expanduser("~/tbp/tbp.drone/imgs")
    dataset_dir = os.path.join(imgs_dir, "spam_dataset_v1")
    annotations_dir = os.path.join(dataset_dir, "bbox_annotations")
    
    # Create output directory if it doesn't exist
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Get all PNG files in the dataset directory
    png_files = list(Path(dataset_dir).glob("*.png"))
    print(f"Found {len(png_files)} PNG files in {dataset_dir}")
    
    # Initialize annotator
    annotator = BBoxAnnotator()
    cv2.namedWindow('Annotation')
    cv2.setMouseCallback('Annotation', annotator.draw_bbox)
    
    # Process each image
    for img_path in png_files:
        print(f"\nProcessing {img_path.name}...")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue
            
        annotations = {}
        
        # Annotate Spam Can
        print("Please draw a bounding box around the SPAM CAN")
        print("Left click and drag to draw. Press 'r' to retry, 's' to save and continue, 'q' to quit")
        
        while True:
            annotator.current_image = image.copy()
            annotator.temp_image = image.copy()
            
            while True:
                if annotator.temp_image is not None:
                    cv2.imshow('Annotation', annotator.temp_image)
                else:
                    cv2.imshow('Annotation', annotator.current_image)
                    
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):  # Reset
                    annotator.current_image = image.copy()
                    annotator.bbox = []
                    break
                elif key == ord('s'):  # Save and continue
                    if annotator.bbox:
                        annotations['spam_can'] = annotator.bbox
                        break
                    else:
                        print("Please draw a bounding box first")
                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    return
                    
            if 'spam_can' in annotations:
                break
                
        # Annotate ArUco Marker
        print("\nPlease draw a bounding box around the ARUCO MARKER")
        print("Left click and drag to draw. Press 'r' to retry, 's' to save and continue, 'q' to quit")
        
        while True:
            annotator.current_image = image.copy()
            annotator.temp_image = image.copy()
            
            # Draw previous spam can bbox
            if 'spam_can' in annotations:
                x1, y1, x2, y2 = annotations['spam_can']
                cv2.rectangle(annotator.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotator.current_image, "Spam Can", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            while True:
                if annotator.temp_image is not None:
                    cv2.imshow('Annotation', annotator.temp_image)
                else:
                    cv2.imshow('Annotation', annotator.current_image)
                    
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):  # Reset
                    annotator.current_image = image.copy()
                    annotator.bbox = []
                    break
                elif key == ord('s'):  # Save and continue
                    if annotator.bbox:
                        annotations['aruco'] = annotator.bbox
                        break
                    else:
                        print("Please draw a bounding box first")
                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    return
                    
            if 'aruco' in annotations:
                break
        
        # Save annotations
        annotation_file = os.path.join(annotations_dir, f"{img_path.stem}_annotations.json")
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=4)
        print(f"Saved annotations to {annotation_file}")
        
        # Draw final visualization
        final_image = image.copy()
        for obj_type, bbox in annotations.items():
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if obj_type == 'spam_can' else (255, 0, 0)
            cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(final_image, obj_type, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        # Save visualization
        vis_file = os.path.join(annotations_dir, f"{img_path.stem}_visualization.png")
        cv2.imwrite(vis_file, final_image)
        print(f"Saved visualization to {vis_file}")
        
    cv2.destroyAllWindows()

def process_single_file(img_path):
    """Process a single image file for manual bounding box annotation."""
    # Setup paths
    imgs_dir = os.path.expanduser("~/tbp/tbp.drone/imgs")
    dataset_dir = os.path.join(imgs_dir, "spam_dataset_v1")
    annotations_dir = os.path.join(dataset_dir, "bbox_annotations")
    
    # Create output directory if it doesn't exist
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Initialize annotator
    annotator = BBoxAnnotator()
    cv2.namedWindow('Annotation')
    cv2.setMouseCallback('Annotation', annotator.draw_bbox)
    
    print(f"\nProcessing {img_path}...")
    
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not load image {img_path}")
        return
        
    annotations = {}
    
    # Annotate Spam Can
    print("Please draw a bounding box around the SPAM CAN")
    print("Left click and drag to draw. Press 'r' to retry, 's' to save and continue, 'q' to quit")
    
    while True:
        annotator.current_image = image.copy()
        annotator.temp_image = image.copy()
        
        while True:
            if annotator.temp_image is not None:
                cv2.imshow('Annotation', annotator.temp_image)
            else:
                cv2.imshow('Annotation', annotator.current_image)
                
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                annotator.current_image = image.copy()
                annotator.bbox = []
                break
            elif key == ord('s'):  # Save and continue
                if annotator.bbox:
                    annotations['spam_can'] = annotator.bbox
                    break
                else:
                    print("Please draw a bounding box first")
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return
                
        if 'spam_can' in annotations:
            break
            
    # Annotate ArUco Marker
    print("\nPlease draw a bounding box around the ARUCO MARKER")
    print("Left click and drag to draw. Press 'r' to retry, 's' to save and continue, 'q' to quit")
    
    while True:
        annotator.current_image = image.copy()
        annotator.temp_image = image.copy()
        
        # Draw previous spam can bbox
        if 'spam_can' in annotations:
            x1, y1, x2, y2 = annotations['spam_can']
            cv2.rectangle(annotator.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotator.current_image, "Spam Can", (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        while True:
            if annotator.temp_image is not None:
                cv2.imshow('Annotation', annotator.temp_image)
            else:
                cv2.imshow('Annotation', annotator.current_image)
                
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                annotator.current_image = image.copy()
                annotator.bbox = []
                break
            elif key == ord('s'):  # Save and continue
                if annotator.bbox:
                    annotations['aruco'] = annotator.bbox
                    break
                else:
                    print("Please draw a bounding box first")
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return
                
        if 'aruco' in annotations:
            break
    
    # Save annotations
    img_path = Path(img_path)
    annotation_file = os.path.join(annotations_dir, f"{img_path.stem}_annotations.json")
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved annotations to {annotation_file}")
    
    # Draw final visualization
    final_image = image.copy()
    for obj_type, bbox in annotations.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if obj_type == 'spam_can' else (255, 0, 0)
        cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(final_image, obj_type, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    # Save visualization
    vis_file = os.path.join(annotations_dir, f"{img_path.stem}_visualization.png")
    cv2.imwrite(vis_file, final_image)
    print(f"Saved visualization to {vis_file}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manual bounding box annotation tool')
    parser.add_argument('--file', type=str, help='Path to specific image file to annotate')
    args = parser.parse_args()
    
    if args.file:
        if os.path.exists(args.file):
            process_single_file(args.file)
        else:
            print(f"Error: File {args.file} does not exist")
    else:
        process_dataset() 