import os
from pathlib import Path

# Get the root directory of tbp.drone project
ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))

# Define paths to external modules relative to tbp.drone root
DEPTH_ANYTHING_PATH = str(ROOT_DIR.parent / "Depth-Anything-V2")
SEGMENT_ANYTHING_PATH = str(ROOT_DIR.parent / "segment-anything")

# Add any other external module paths here as needed 