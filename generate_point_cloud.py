#!/usr/bin/env python3

"""Script to generate 3D point cloud from an image."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from tbp.drone.src.vision.point_cloud import main

if __name__ == "__main__":
    main() 