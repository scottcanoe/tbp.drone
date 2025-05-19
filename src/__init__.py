"""
tbp.drone package.

This package contains modules for drone-related functionality including depth estimation,
object segmentation, and 3D reconstruction.
"""

from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
