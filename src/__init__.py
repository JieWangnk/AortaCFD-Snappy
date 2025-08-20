"""
AortaCFD Snappy - Automated Mesh Optimization for Arterial Geometries
=====================================================================

A robust, flexible tool for iterative mesh refinement of cardiovascular geometries
using OpenFOAM's snappyHexMesh with intelligent quality assessment.

Key Features:
- Adaptive refinement strategies
- Two-pass boundary layer generation
- Automated quality assessment
- Multi-geometry support
- Publication-ready mesh generation

Authors: Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .geometry_utils import GeometryProcessor
from .mesh_functions import MeshGenerator
from .quality_assessment import QualityEvaluator
from .config_manager import ConfigManager

__all__ = [
    'GeometryProcessor',
    'MeshGenerator', 
    'QualityEvaluator',
    'ConfigManager'
]