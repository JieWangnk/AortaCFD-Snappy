"""
Constants for Stage1 mesh optimization.
All magic numbers and thresholds centralized here for easy configuration.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MeshQualityLimits:
    """OpenFOAM mesh quality thresholds"""
    MAX_NON_ORTHO_DEFAULT = 65.0
    MAX_SKEWNESS_DEFAULT = 4.0
    MIN_VOL_DEFAULT = 1e-13
    MIN_TET_QUALITY_SNAP = -1e15
    MIN_TET_QUALITY_LAYER = 1e-6
    MIN_FACE_WEIGHT = 0.02
    MIN_VOL_RATIO = 0.01
    
    # Solver-specific presets
    LES_MAX_NON_ORTHO = 60.0
    LES_MAX_SKEWNESS = 3.0
    LES_MIN_COVERAGE = 0.80
    
    RANS_MAX_NON_ORTHO = 65.0
    RANS_MAX_SKEWNESS = 4.0
    RANS_MIN_COVERAGE = 0.70

@dataclass
class LayerParams:
    """Boundary layer generation parameters"""
    MIN_THICKNESS_RATIO = 0.15    # minThickness = 0.15 × firstLayerThickness
    MAX_THICKNESS_RATIO = 0.20    # Trigger validation if exceeded
    
    # Geometric controls
    FEATURE_ANGLE_DEFAULT = 70
    MIN_MEDIAN_AXIS_ANGLE = 70
    MAX_THICKNESS_TO_MEDIAL_RATIO = 0.45
    N_GROW_DEFAULT = 1
    N_RELAX_ITER = 8
    
    # Physics constraints
    MIN_FIRST_LAYER_MICRONS = 5.0    # 5 μm minimum for numerical stability
    MAX_FIRST_LAYER_MICRONS = 200.0  # 200 μm maximum for reasonable BL
    MIN_TOTAL_THICKNESS_MICRONS = 10.0
    MAX_TOTAL_THICKNESS_MICRONS = 1000.0

@dataclass
class PhysicsConstants:
    """Fluid physics and CFD parameters"""
    # Blood properties
    BLOOD_DENSITY_DEFAULT = 1060.0      # kg/m³
    BLOOD_VISCOSITY_DEFAULT = 0.0035    # Pa·s
    
    # Flow parameters
    HEART_RATE_DEFAULT_HZ = 1.2         # 72 bpm
    Y_PLUS_LES = 1.0
    Y_PLUS_RANS = 30.0
    
    # Womersley boundary layer
    WOMERSLEY_SCALING_FACTOR = 2.0      # δ_ω = √(2ν/ω)
    
    # Reynolds number thresholds
    RE_LAMINAR_THRESHOLD = 2300
    RE_TRANSITIONAL_MIN = 2300
    RE_TRANSITIONAL_MAX = 4000

@dataclass
class MemoryLimits:
    """OpenFOAM memory usage modeling"""
    # Memory per cell (GB)
    BASE_MEMORY_PER_CELL = 1.5e-3
    GRADIENT_OVERHEAD = 0.8e-3
    SOLVER_WORKSPACE = 0.5e-3
    PARALLEL_OVERHEAD_PER_PROC = 0.2e-3
    
    # snappyHexMesh factor
    SNAPPY_MEMORY_FACTOR = 1.3
    
    # System limits
    MEMORY_SAFETY_FACTOR = 0.7    # Use 70% of available RAM
    MAX_MEMORY_GB = 12.0
    
    # Cell count limits
    SERIAL_MAX_CELLS = 2_000_000
    PARALLEL_MAX_CELLS = 20_000_000

@dataclass
class GeometryLimits:
    """Geometry processing parameters"""
    # STL scaling detection
    MM_TO_M_THRESHOLD = 0.01        # If max dimension < 1cm, assume mm->m needed
    AUTO_SCALE_FACTOR = 0.001       # mm to m conversion
    
    # Feature detection
    FEATURE_ANGLE_MIN = 20
    FEATURE_ANGLE_MAX = 90
    
    # Refinement levels
    MAX_SURFACE_LEVEL = 4
    MAX_VOLUME_LEVEL = 3
    
    # Distance band scaling
    NEAR_BAND_FACTOR = 4    # 4 × base_size
    FAR_BAND_FACTOR = 10    # 10 × base_size

@dataclass
class ConvergenceParams:
    """Iteration and convergence control"""
    MAX_ITERATIONS_DEFAULT = 4
    MIN_ITERATIONS_FOR_CONVERGENCE = 3
    
    # Convergence detection (coefficient of variation thresholds)
    CV_THRESHOLD_NON_ORTHO = 0.05      # 5% variation
    CV_THRESHOLD_SKEWNESS = 0.10       # 10% variation  
    CV_THRESHOLD_CELLS = 0.02          # 2% variation
    
    # Quality improvement thresholds
    QUALITY_IMPROVEMENT_THRESHOLD = 2.0    # Significant if Δ > 2
    LAYER_DEGRADATION_THRESHOLD_NO = 5.0   # Non-orthogonality
    LAYER_DEGRADATION_THRESHOLD_SK = 0.5   # Skewness
    
    # Timeout values (seconds)
    SURFACE_CHECK_TIMEOUT = 600
    SNAPPY_TIMEOUT_MAX = 3600    # 1 hour max
    CHECKMESH_TIMEOUT = 300

# Export all constants as a single config dict for backward compatibility
DEFAULT_CONSTANTS: Dict[str, Any] = {
    'mesh_quality': MeshQualityLimits(),
    'layers': LayerParams(),
    'physics': PhysicsConstants(),
    'memory': MemoryLimits(),
    'geometry': GeometryLimits(),
    'convergence': ConvergenceParams(),
}

# Magic number replacements - use these instead of hardcoded values
MAGIC_NUMBERS = {
    # Replace scattered hardcoded values
    'base_cells_per_diameter': 22,
    'min_cells_per_throat': 28,
    'cells_per_cm_density': 12,
    'surface_refinement_ladder': [[1,1], [1,2], [2,2], [2,3]],
    'expansion_ratio_min': 1.1,
    'expansion_ratio_max': 2.0,
    'layer_count_min': 3,
    'layer_count_max': 20,
    'feature_snap_iterations': 10,
    'smooth_patch_iterations': 3,
    'solve_iterations': 30,
    'relax_iterations': 5,
}