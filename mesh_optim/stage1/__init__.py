"""
Stage1 mesh optimization package - modular, maintainable architecture.

This package provides a clean, modular approach to automated mesh generation
with proper separation of concerns:

- ConfigManager: Configuration loading and validation
- PhysicsCalculator: CFD physics calculations (y+, Womersley, etc.)
- QualityAnalyzer: Mesh quality assessment and convergence detection
- GeometryProcessor: STL handling and geometric analysis
- MeshGenerator: OpenFOAM dictionary generation
- IterationManager: Optimization loop control

Usage:
    from mesh_optim.stage1 import Stage1MeshOptimizer
    
    optimizer = Stage1MeshOptimizer(geometry_dir, config_file, output_dir)
    best_mesh = optimizer.optimize()
"""

# Import main class for backward compatibility
from .optimizer import Stage1Optimizer as Stage1MeshOptimizer

# Import key dataclasses for external use
from .constants import MeshQualityLimits, LayerParams, PhysicsConstants
from .config_manager import Stage1Targets

__version__ = "2.0.0"
__all__ = [
    "Stage1MeshOptimizer", 
    "MeshQualityLimits", 
    "LayerParams", 
    "PhysicsConstants",
    "Stage1Targets"
]