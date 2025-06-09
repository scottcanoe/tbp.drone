"""
Vision module for 3D perception.

This module provides high-level functionality for:
1. Converting RGB images to 3D point clouds with semantic labels
2. Processing depth information from images
3. Segmenting objects in images
"""

from .point_cloud import DroneDepthTo3DLocations

__all__ = ["DroneDepthTo3DLocations"]
