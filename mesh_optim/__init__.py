"""
AortaCFD Mesh Optimization Package

Two-stage mesh optimization approach:
- Stage 1: Geometry-driven mesh generation (inner loop)
- Stage 2: QoI-driven mesh adaptation (outer loop)
"""

from .stage1_mesh import Stage1MeshOptimizer
from .stage2_gci import Stage2GCIVerifier
from .physics_mesh import PhysicsAwareMeshGenerator
from .utils import run_command, check_mesh_quality, parse_layer_coverage

__version__ = "1.0.0"
__all__ = ["Stage1MeshOptimizer", "Stage2GCIVerifier", "PhysicsAwareMeshGenerator", "run_command", "check_mesh_quality", "parse_layer_coverage"]