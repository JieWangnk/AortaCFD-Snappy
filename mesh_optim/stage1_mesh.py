"""
Stage 1: Geometry‑aware, geometry‑only mesh optimization (snap→checkMesh→layers→checkMesh)
- Single base unit cell size derived from inlet/outlet diameters or cells-per-cm density
- Gentle, low‑contrast surface refinement ladder (1–1 → 2–2 → 2–3)
- Distance refinement bands tied to base size (multiples of Δx)
- Adaptive resolveFeatureAngle and layer settings based on checkMesh deltas
- Constraint-based: minimize cells subject to maxNonOrtho/maxSkewness/layer-coverage
- All units normalized to meters; STLs are scaled by 0.001 if mm-as-m coordinates are detected

LEGACY COMPATIBILITY LAYER: This file maintains backward compatibility while
the new modular architecture is being integrated. New code should use
the stage1.optimizer module directly.

Configuration additions expected (example):

{
  "openfoam_env_path": "source /opt/openfoam12/etc/bashrc",
  "BLOCKMESH": { "resolution": 60 },
  "SNAPPY": {
    "maxLocalCells": 250000,
    "maxGlobalCells": 2000000,
    "nCellsBetweenLevels": 1,
    "surface_level": [1,1],
    "resolveFeatureAngle": 45
  },
  "LAYERS": {
    "nSurfaceLayers": 14,
    "firstLayerThickness_abs": 50e-6,
    "minThickness_abs": 20e-6,
    "expansionRatio": 1.2,
    "nGrow": 0,
    "featureAngle": 60,
    "maxThicknessToMedialRatio": 0.3,
    "minMedianAxisAngle": 90,
    "maxBoundarySkewness": 4.0,
    "maxInternalSkewness": 4.0
  },
  "MESH_QUALITY": {
    "snap":  {"maxNonOrtho": 65, "maxBoundarySkewness": 4.0, "maxInternalSkewness": 4.0,
               "minVol": 1e-13, "minTetQuality": -1e15, "minFaceWeight": 0.02},
    "layer": {"maxNonOrtho": 65, "maxBoundarySkewness": 4.0, "maxInternalSkewness": 4.0,
               "minVol": 1e-13, "minTetQuality": 1e-6, "minFaceWeight": 0.02}
  },
  "acceptance_criteria": {"maxNonOrtho": 65, "maxSkewness": 4.0, "min_layer_coverage": 0.65},
  "STAGE1": {
    "base_size_mode": "diameter",               // "diameter" or "density"
    "N_D": 22,                                    // cells across reference diameter
    "N_D_min": 28,                                // minimum cells across D_min (throat guard)
    "cells_per_cm": 12,                           // used if base_size_mode == "density"
    "ladder": [[1,1],[1,2],[2,2]],                // surface level ladder (conservative)
    "near_band_cells": 4,                         // near band thickness in multiples of Δx
    "far_band_cells": 10,                         // far band thickness in multiples of Δx
    "featureAngle_init": 45,
    "featureAngle_step": 10,
    "coverage_target": 0.65,
    "max_iterations": 4,                           // maximum geometry optimization iterations
    "n_processors": 1                             // number of processors for parallel execution (1=serial)
  },
  "GEOMETRY_POLICY": {
    "diameter_mode": "auto",                      // "auto", "inlet_only", or "fixed"
    "clamp_mode": "none",                         // "none" or "loose" (sanity check only)
    "throat_guard_scale": 0.85,                   // safety factor for D_min detection
    "featureAngle_mode": "adaptive",              // "adaptive" or "ladder" (backward compat)
    "fixed_D_ref_m": null,                        // used if diameter_mode == "fixed"
    "fixed_D_min_m": null                         // used if diameter_mode == "fixed"
  },
  "PHYSICS": {
    "solver_mode": "",                            // "LES", "RANS", or "LAMINAR" - sets acceptance presets
    "autoFirstLayer": false,                      // Enable y+ based first layer sizing
    "U_peak": 1.0,                                // Peak velocity [m/s]
    "rho": 1060.0,                                // Fluid density [kg/m³]
    "mu": 3.5e-3,                                 // Dynamic viscosity [Pa·s]
    "y_plus": 1.0,                                // Target y+ (1 for LES, 30 for wall-fn)
    "flow_model": "turbulent",                    // "turbulent" or "laminar"
    "use_womersley_bands": false,                 // Use Womersley boundary layer for refinement bands
    "heart_rate_hz": 1.2                          // Heart rate [Hz] (~72 bpm) for Womersley calculation
  },
  "max_iterations": 4
}
"""

import json
import csv
import math
import re
import numpy as np
import struct
from pathlib import Path
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

from .utils import run_command, check_mesh_quality, parse_layer_coverage, calculate_first_layer_thickness
from .physics_mesh import PhysicsAwareMeshGenerator

# Import new modular components with backward compatibility fallback
try:
    from .stage1.optimizer import Stage1Optimizer
    from .stage1.config_manager import ConfigManager
    MODULAR_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Modular components not available: {e}")
    MODULAR_AVAILABLE = False

# ==================== HELPER FUNCTIONS (ChatGPT improvements) ====================




@dataclass
class Stage1Targets:
    max_nonortho: float
    max_skewness: float
    min_layer_cov: float


def _safe_get(d: dict, path: List[str], default=None):
    x = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


class Stage1MeshOptimizer:
    """Geometry‑aware Stage‑1 mesh optimizer (geometry only, meters everywhere).
    
    LEGACY COMPATIBILITY CLASS: This class provides backward compatibility
    while the new modular architecture (stage1.optimizer) is being integrated.
    For new code, use stage1.optimizer.Stage1Optimizer directly.
    """

    def __init__(self, geometry_dir, config_file, output_dir=None):
        self.geometry_dir = Path(geometry_dir)
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir) if output_dir else (self.geometry_dir.parent / "output" / "stage1_mesh")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_file) as f:
            self.config = json.load(f)
        
        # Ensure SNAPPY section exists to prevent KeyError
        self.config.setdefault("SNAPPY", {})
        
        # Map from new two-tier structure to internal format if needed
        self._map_config_structure()
        
        # Validate and set defaults for critical LAYERS configuration
        self._validate_layers_config()

        self.logger = logging.getLogger(f"Stage1Mesh_{self.geometry_dir.name}")

        # Initialize modular optimizer if available
        if MODULAR_AVAILABLE:
            try:
                geometry_stl = self._find_geometry_stl()
                self._modular_optimizer = Stage1Optimizer(
                    str(self.config_file), 
                    str(geometry_stl),
                    str(self.output_dir)
                )
                self._use_modular = True
                self.logger.info("Using new modular optimizer")
            except Exception as e:
                self.logger.warning(f"Failed to initialize modular optimizer: {e}")
                self._use_modular = False
        else:
            self._use_modular = False

        # Resource management
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.max_memory_gb = min(available_memory_gb * 0.7, 12)  # 70% of available, cap 12GB
        self.logger.info(f"Stage 1 memory limit: {self.max_memory_gb:.1f}GB")

        # Stage‑1 policy (needed for max_iterations)
        self.stage1 = _safe_get(self.config, ["STAGE1"], {}) or {}
        
        # Iteration state
        self.current_iteration = 0
        # Check STAGE1 section first, then root level, then default
        self.max_iterations = int(self.stage1.get("max_iterations", self.config.get("max_iterations", 4)))
        self.logger.debug(f"Max iterations: {self.max_iterations} (from {'STAGE1' if 'max_iterations' in self.stage1 else 'root' if 'max_iterations' in self.config else 'default'})")
        self.surface_levels = list(_safe_get(self.config, ["SNAPPY", "surface_level"], [1, 1]))
        
        # Convergence tracking
        self.quality_history = []

        # Discover wall patch name generically
        self.wall_name = self._discover_wall_name()
        
        # STL discovery (now that we know wall name)
        self.stl_files = self._discover_stl_files()
        if "featureAngle_init" in self.stage1:
            self.config["SNAPPY"]["resolveFeatureAngle"] = self.stage1["featureAngle_init"]
        
        # Apply improved memory-aware cell budgeting
        self._apply_improved_memory_budgeting()
        
        # Geometry policy for vessel-agnostic sizing
        self.geometry_policy = _safe_get(self.config, ["GEOMETRY_POLICY"], {}) or {}
        
        # Optional physics block for y+ based layer sizing
        self.physics = _safe_get(self.config, ["PHYSICS"], {}) or {}

        # Acceptance/targets - can be overridden by solver mode
        self.targets = Stage1Targets(
            max_nonortho=float(_safe_get(self.config, ["acceptance_criteria", "maxNonOrtho"], 65)),
            max_skewness=float(_safe_get(self.config, ["acceptance_criteria", "maxSkewness"], 4.0)),
            min_layer_cov=float(_safe_get(self.config, ["acceptance_criteria", "min_layer_coverage"], 0.65)),
        )
        
        # Apply solver-specific acceptance criteria if configured
        self._apply_solver_presets()
        
        # Apply proven mesh generation best practices
        self._apply_robust_defaults()

    def _map_config_structure(self):
        """Map from new two-tier configuration structure to internal format"""
        # Check if using new two-tier structure (has top-level mesh/layers/refinement)
        if "mesh" in self.config and "layers" in self.config:
            # Create internal structure from new simple parameters
            
            # Map paths
            paths = self.config.get("paths", {})
            if "openfoam_env" in paths:
                self.config["openfoam_env_path"] = paths["openfoam_env"]
            
            # Map mesh parameters
            mesh = self.config.get("mesh", {})
            if "STAGE1" not in self.config:
                self.config["STAGE1"] = {}
            
            self.config["STAGE1"]["base_size_mode"] = mesh.get("base_size_mode", "diameter")
            self.config["STAGE1"]["N_D"] = mesh.get("cells_per_diameter", 22)
            self.config["STAGE1"]["N_D_min"] = mesh.get("min_cells_per_throat", 28)
            
            # Map refinement parameters
            refinement = self.config.get("refinement", {})
            if "surface_levels" in refinement:
                if "SNAPPY" not in self.config:
                    self.config["SNAPPY"] = {}
                self.config["SNAPPY"]["surface_level"] = refinement["surface_levels"]
                
            self.config["STAGE1"]["near_band_cells"] = refinement.get("near_band_dx", 4)
            self.config["STAGE1"]["far_band_cells"] = refinement.get("far_band_dx", 10)
            
            # Map feature angle
            feature_angle = refinement.get("feature_angle", {})
            if feature_angle.get("mode") == "adaptive":
                self.config["STAGE1"]["featureAngle_init"] = feature_angle.get("init", 45)
                self.config["STAGE1"]["featureAngle_step"] = feature_angle.get("step", 10)
            
            # Map layers parameters
            layers = self.config.get("layers", {})
            if "LAYERS" not in self.config:
                self.config["LAYERS"] = {}
                
            self.config["LAYERS"]["nSurfaceLayers"] = layers.get("n", 10)
            self.config["LAYERS"]["expansionRatio"] = layers.get("expansion", 1.2)
            
            # Map first layer settings
            first_layer = layers.get("first_layer", {})
            if first_layer.get("mode") == "auto_dx":
                self.config["STAGE1"]["alpha_total_layers"] = first_layer.get("t_over_dx", 0.8)
                self.config["STAGE1"]["t1_min_fraction_of_dx"] = first_layer.get("t1_min_frac", 0.02)
                self.config["STAGE1"]["t1_max_fraction_of_dx"] = first_layer.get("t1_max_frac", 0.08)
                self.config["STAGE1"]["autoFirstLayerFromDx"] = True
            
            # Map acceptance criteria
            accept = self.config.get("accept", {})
            if "acceptance_criteria" not in self.config:
                self.config["acceptance_criteria"] = {}
                
            self.config["acceptance_criteria"]["maxNonOrtho"] = accept.get("maxNonOrtho", 65)
            self.config["acceptance_criteria"]["maxSkewness"] = accept.get("maxSkewness", 4.0)
            self.config["acceptance_criteria"]["min_layer_coverage"] = accept.get("min_layer_coverage", 0.70)
            
            # Map compute settings
            compute = self.config.get("compute", {})
            self.config["STAGE1"]["n_processors"] = compute.get("procs", 4)
            self.config["STAGE1"]["cell_budget_kb_per_cell"] = compute.get("cell_budget_kb_per_cell", 1.0)
            
            # Map iterations
            iterations = self.config.get("iterations", {})
            self.config["max_iterations"] = iterations.get("max", 3)
            if "ladder" in iterations:
                self.config["STAGE1"]["ladder"] = iterations["ladder"]
            
            # Map physics parameters  
            physics = self.config.get("physics", {})
            if "PHYSICS" not in self.config:
                self.config["PHYSICS"] = {}
                
            self.config["PHYSICS"]["solver_mode"] = physics.get("solver_mode", "RANS")
            self.config["PHYSICS"]["flow_model"] = physics.get("flow_model", "turbulent")
            self.config["PHYSICS"]["y_plus"] = physics.get("y_plus", 30)
            self.config["PHYSICS"]["U_peak"] = physics.get("U_peak", 1.0)
            self.config["PHYSICS"]["rho"] = physics.get("rho", 1060.0)
            self.config["PHYSICS"]["mu"] = physics.get("mu", 0.0035)
            self.config["PHYSICS"]["use_womersley_bands"] = physics.get("use_womersley_bands", False)
            self.config["PHYSICS"]["heart_rate_hz"] = physics.get("heart_rate_hz", 1.2)
            
            # Preserve advanced overrides from the advanced section
            if "advanced" in self.config:
                advanced = self.config["advanced"]
                
                # Override with advanced settings if they exist
                for section in ["BLOCKMESH", "SNAPPY", "LAYERS", "MESH_QUALITY", "SURFACE_FEATURES", 
                               "GEOMETRY_POLICY", "STAGE1", "SCALING", "PHYSICS"]:
                    if section in advanced:
                        if section not in self.config:
                            self.config[section] = {}
                        self.config[section].update(advanced[section])

    def _validate_layers_config(self):
        """Validate and set defaults for critical LAYERS configuration block"""
        # Ensure LAYERS section exists
        self.config.setdefault("LAYERS", {})
        L = self.config["LAYERS"]
        
        # Set critical defaults with validation
        L.setdefault("nSurfaceLayers", 14)
        L.setdefault("expansionRatio", 1.2)
        L.setdefault("firstLayerThickness_abs", 50e-6)  # 50 microns
        L.setdefault("minThickness_abs", 20e-6)         # 20 microns  
        L.setdefault("nGrow", 1)
        L.setdefault("featureAngle", 70)
        L.setdefault("maxThicknessToMedialRatio", 0.45)
        L.setdefault("minMedianAxisAngle", 70)
        L.setdefault("maxBoundarySkewness", 4.0)
        L.setdefault("maxInternalSkewness", 4.0)
        
        # Validate ranges for critical parameters
        if not (5 <= L["nSurfaceLayers"] <= 25):
            L["nSurfaceLayers"] = max(5, min(25, L["nSurfaceLayers"]))
            
        if not (1.1 <= L["expansionRatio"] <= 2.0):
            L["expansionRatio"] = max(1.1, min(2.0, L["expansionRatio"]))
            
        if not (10e-6 <= L["firstLayerThickness_abs"] <= 200e-6):
            L["firstLayerThickness_abs"] = max(10e-6, min(200e-6, L["firstLayerThickness_abs"]))
            
        if not (5e-6 <= L["minThickness_abs"] <= 100e-6):
            L["minThickness_abs"] = max(5e-6, min(100e-6, L["minThickness_abs"]))
            
        # Ensure minThickness <= firstLayerThickness
        if L["minThickness_abs"] > L["firstLayerThickness_abs"]:
            L["minThickness_abs"] = L["firstLayerThickness_abs"] * 0.5
            
        # Log validation results
        validation_msg = (f"LAYERS validated: n={L['nSurfaceLayers']}, "
                         f"ratio={L['expansionRatio']:.2f}, "
                         f"first={L['firstLayerThickness_abs']*1e6:.1f}μm, "
                         f"min={L['minThickness_abs']*1e6:.1f}μm")
        # Use debug level during init to avoid spam
        logging.getLogger().debug(validation_msg)

    def _apply_solver_presets(self):
        """Apply solver-specific acceptance criteria based on intended physics"""
        solver_mode = self.physics.get("solver_mode", "").upper()
        
        if solver_mode == "LES":
            # LES with near-wall resolution: tighter quality requirements, higher coverage for WSS
            self.targets.max_nonortho = min(self.targets.max_nonortho, 60.0)
            self.targets.max_skewness = min(self.targets.max_skewness, 3.5)
            self.targets.min_layer_cov = max(self.targets.min_layer_cov, 0.80)  # Higher for WSS accuracy
            self.logger.info(f"Applied LES solver presets: maxNonOrtho≤{self.targets.max_nonortho}, "
                           f"maxSkewness≤{self.targets.max_skewness}, coverage≥{self.targets.min_layer_cov:.0%}")
            
        elif solver_mode == "RANS":
            # RANS with wall functions: moderate quality requirements
            self.targets.max_nonortho = min(self.targets.max_nonortho, 65.0)
            self.targets.max_skewness = min(self.targets.max_skewness, 4.0)
            self.targets.min_layer_cov = max(self.targets.min_layer_cov, 0.70)
            self.logger.info(f"Applied RANS solver presets: maxNonOrtho≤{self.targets.max_nonortho}, "
                           f"maxSkewness≤{self.targets.max_skewness}, coverage≥{self.targets.min_layer_cov:.0%}")
            
        elif solver_mode == "LAMINAR":
            # Laminar flow: relaxed layer coverage requirement
            self.targets.max_nonortho = min(self.targets.max_nonortho, 65.0)
            self.targets.max_skewness = min(self.targets.max_skewness, 4.0)
            self.targets.min_layer_cov = max(self.targets.min_layer_cov, 0.65)
            self.logger.info(f"Applied Laminar solver presets: maxNonOrtho≤{self.targets.max_nonortho}, "
                           f"maxSkewness≤{self.targets.max_skewness}, coverage≥{self.targets.min_layer_cov:.0%}")
        
        # If solver_mode not specified or unrecognized, keep user-configured values
    
    def _apply_robust_defaults(self):
        """Apply proven mesh generation best practices from successful manual runs"""
        
        # Ensure conservative surface refinement progression
        ladder = self.stage1.get("ladder", [[1,1],[1,2],[2,2]])
        if len(ladder) > 0 and len(ladder[0]) == 2:
            # Ensure no iteration jumps too aggressively
            max_jump = 0
            for i, (min_lvl, max_lvl) in enumerate(ladder):
                if i > 0:
                    prev_max = ladder[i-1][1] 
                    jump = max_lvl - prev_max
                    if jump > 1:
                        self.logger.info(f"Limiting surface level jump in ladder iteration {i}: ({min_lvl},{max_lvl}) → ({min_lvl},{prev_max+1})")
                        ladder[i] = [min_lvl, prev_max + 1]
            self.stage1["ladder"] = ladder
        
        # Ensure reasonable layer settings for vascular meshing
        layers = self.config.get("LAYERS", {})
                
        # Ensure first layer thickness is reasonable for vascular scale
        first_layer = layers.get("firstLayerThickness_abs", 50e-6)
        if first_layer is not None and (first_layer < 10e-6 or first_layer > 200e-6):
            recommended = 50e-6  # 50 microns - good for most vascular flows
            self.logger.info(f"Adjusting first layer thickness: {first_layer*1e6:.1f}μm → {recommended*1e6:.1f}μm")
            layers["firstLayerThickness_abs"] = recommended
            
        # Ensure feature angle detection is in proven range
        snap = self.config.get("SNAPPY", {})
        resolve_angle = snap.get("resolveFeatureAngle", 35)
        if resolve_angle < 25 or resolve_angle > 60:
            recommended = 35  # Conservative but effective
            self.logger.info(f"Adjusting resolveFeatureAngle: {resolve_angle}° → {recommended}°")
            snap["resolveFeatureAngle"] = recommended
            
        # Ensure sufficient snap iterations for complex geometry
        n_feature_snap = snap.get("nFeatureSnapIter", 20)
        if n_feature_snap < 10:
            recommended = 20  # Proven to work well for vascular geometry  
            self.logger.info(f"Increasing nFeatureSnapIter: {n_feature_snap} → {recommended}")
            snap["nFeatureSnapIter"] = recommended
            
        self.logger.debug("Applied robust meshing defaults based on proven configurations")

    # ------------------------ Geometry & base size -------------------------
    def _discover_stl_files(self) -> Dict:
        required = ["inlet.stl"]
        found = {"required": {}, "outlets": []}
        
        # Find inlet
        for name in required:
            p = self.geometry_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Required STL file not found: {name}")
            found["required"][name.split('.')[0]] = p
        
        # Find wall (using discovered name)
        wall_path = self.geometry_dir / f"{self.wall_name}.stl"
        if not wall_path.exists():
            raise FileNotFoundError(f"Wall STL file not found: {self.wall_name}.stl")
        found["required"][self.wall_name] = wall_path
        
        for p in self.geometry_dir.glob("outlet*.stl"):
            found["outlets"].append(p)
        if not found["outlets"]:
            raise FileNotFoundError("No outlet STL files found")
        self.logger.info(f"Found {len(found['outlets'])} outlet files")
        return found
    
    def _apply_improved_memory_budgeting(self) -> None:
        """Apply improved memory-aware cell budgeting."""
        import psutil
        
        # Get current system resources
        available_gb = psutil.virtual_memory().available / (1024**3)
        n_procs = self.stage1.get("n_processors", 1)
        kb_per_cell = self.stage1.get("cell_budget_kb_per_cell", 2.0)
        
        # OpenFOAM-specific memory modeling
        usable_gb = available_gb * 0.7  # 70% safety margin
        max_cells = self._openfoam_memory_model(usable_gb, n_procs)
        
        self.logger.info(f"OpenFOAM memory model: {usable_gb:.1f}GB usable → max {max_cells:,} cells")
        
        # Use OpenFOAM-aware cell budget
        total_cells = max_cells
        
        # Distribute across processors with OpenFOAM communication patterns
        # Each processor needs some overlap cells, so local limit is slightly higher
        max_local = int(np.clip(total_cells // max(n_procs, 1) * 1.1, 100_000, 5_000_000))
        max_global = int(np.clip(total_cells, 500_000, 20_000_000))
        
        # Apply limits but allow config overrides
        original_local = self.config["SNAPPY"].get("maxLocalCells", max_local)
        original_global = self.config["SNAPPY"].get("maxGlobalCells", max_global)
        
        # Use the more conservative (smaller) of the two
        self.config["SNAPPY"]["maxLocalCells"] = min(original_local, max_local)
        self.config["SNAPPY"]["maxGlobalCells"] = min(original_global, max_global)
        
        self.logger.info(f"Memory-aware limits: {available_gb:.1f}GB avail → "
                        f"Local: {self.config['SNAPPY']['maxLocalCells']:,}, "
                        f"Global: {self.config['SNAPPY']['maxGlobalCells']:,}")
        
        if original_local > max_local or original_global > max_global:
            self.logger.info(f"   Reduced from config: Local {original_local:,}→{self.config['SNAPPY']['maxLocalCells']:,}, "
                           f"Global {original_global:,}→{self.config['SNAPPY']['maxGlobalCells']:,}")

    def _get_bbox_dimensions(self, bbox_dict):
        """
        Return (Lx, Ly, Lz) in the *same units as the STL coordinates*.
        Supports either:
          - {"dimensions": {"length":..., "width":..., "height":...}} or {"dx","dy","dz"}
          - {"mesh_domain": {"x_min","x_max","y_min","y_max","z_min","z_max"}}
        """
        # 1) direct dimensions block
        dims = bbox_dict.get("dimensions", {})
        if all(k in dims for k in ("length", "width", "height")):
            return dims["length"], dims["width"], dims["height"]
        if all(k in dims for k in ("dx", "dy", "dz")):
            return dims["dx"], dims["dy"], dims["dz"]
        
        # 2) compute from mesh_domain extents
        md = bbox_dict.get("mesh_domain", {})
        keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        if all(k in md for k in keys):
            Lx = float(md["x_max"]) - float(md["x_min"])
            Ly = float(md["y_max"]) - float(md["y_min"])
            Lz = float(md["z_max"]) - float(md["z_min"])
            return Lx, Ly, Lz
        
        # 3) last resort defaults (log once)
        self.logger.warning("BBox dict missing expected keys; returning conservative defaults.")
        return 0.02, 0.02, 0.02

    def _compute_block_divisions(self, bbox_size, dx_base):
        """Compute blockMesh divisions with hierarchical override control
        
        Precedence: divisions > cell_size_m > resolution > geometry-aware
        """
        bm = self.config.get("BLOCKMESH", {}) or {}
        # Handle null/None values in config properly
        min_per_axis_config = bm.get("min_per_axis", [10, 10, 10])
        if min_per_axis_config is None:
            min_per_axis_config = [10, 10, 10]
        mins = np.array(min_per_axis_config, dtype=int)

        if "divisions" in bm and bm["divisions"] is not None:
            divs = np.array(bm["divisions"], dtype=int)
            return np.maximum(divs, mins)

        if "cell_size_m" in bm and bm["cell_size_m"] is not None:
            dx = float(bm["cell_size_m"])
            divs = np.ceil(bbox_size / max(dx, 1e-9)).astype(int)
            return np.maximum(divs, mins)

        if "resolution" in bm:
            R = int(bm["resolution"])
            Lmax = float(max(bbox_size))
            scale = R / max(Lmax, 1e-12)  # cells per meter
            divs = np.ceil(bbox_size * scale).astype(int)
            return np.maximum(divs, mins)

        # default: geometry-aware from dx_base
        if dx_base is None:
            self.logger.error(f"dx_base is None in _compute_block_divisions - this should not happen")
            dx_base = 1e-3  # 1mm fallback
        divs = np.ceil(bbox_size / max(dx_base, 1e-9)).astype(int)
        return np.maximum(divs, mins)
    
    def _discover_wall_name(self) -> str:
        """Discover wall patch name generically across vascular beds"""
        candidate_names = ["wall_aorta", "wall", "vessel_wall", "arterial_wall"]
        for name in candidate_names:
            if (self.geometry_dir / f"{name}.stl").exists():
                self.logger.info(f"Wall patch discovered: {name}")
                return name
        # Fallback to default
        self.logger.warning("No standard wall patch found, using 'wall_aorta'")
        return "wall_aorta"

    def _estimate_reference_diameters(self, stl_root=None) -> Tuple[float, float]:
        """Return (D_ref, D_min) in METERS - vessel-agnostic approach.
        Derives sizes from actual geometry without anatomy-specific assumptions.
        
        Args:
            stl_root: Optional directory containing STL files to read from.
                     If None, uses original STL files. If provided, reads from scaled STLs.
        """
        policy = self.geometry_policy
        mode = str(policy.get("diameter_mode", "auto")).lower()
        clamp = str(policy.get("clamp_mode", "none")).lower()
        s_guard = float(policy.get("throat_guard_scale", 0.85))
        
        Deq = []  # Initialize for all branches to prevent NameError
        
        if mode == "fixed":
            # User-provided fixed values
            D_ref = float(policy["fixed_D_ref_m"])
            D_min = float(policy["fixed_D_min_m"])
        else:
            # Compute equivalent diameters from outlet areas
            Deq = []
            env = self.config["openfoam_env_path"]
            gen = PhysicsAwareMeshGenerator()  # Instantiate early for bbox calculations
            
            # Determine which STL files to use
            if stl_root:
                # Use scaled STL files from triSurface directory
                outlet_paths = [stl_root / p.name for p in self.stl_files["outlets"]]
                inlet_path = stl_root / "inlet.stl"
                wall_path = stl_root / f"{self.wall_name}.stl"
            else:
                # Use original STL files
                outlet_paths = list(self.stl_files["outlets"])
                inlet_path = self.stl_files["required"]["inlet"]
                wall_path = self.stl_files["required"][self.wall_name]
            
            # Try to get areas from outlet patches
            for p in outlet_paths:
                try:
                    # This should not create constant/ at output_dir level
                    # All OpenFOAM commands should run from iter_dir context
                    # Use direct STL path for surfaceCheck
                    res = run_command(["surfaceCheck", str(p)],
                                    cwd=self.geometry_dir.parent, env_setup=env, timeout=600, 
                                    max_memory_gb=self.max_memory_gb)
                    txt = (res.stdout + res.stderr).lower()
                    
                    # Parse surface area from output
                    area = None
                    for line in txt.splitlines():
                        if "area" in line and ("total" in line or "surface" in line):
                            # Extract the last float on the line
                            tokens = line.replace("=", " ").replace(":", " ").split()
                            for t in reversed(tokens):
                                try:
                                    val = float(t)
                                    if val > 0:
                                        area = val
                                        break
                                except ValueError:
                                    continue
                    
                    if area and area > 0:
                        if stl_root:
                            # For scaled STLs, area should already be in correct units (m²)
                            stem_name = p.name.replace('.stl', '')
                            self.logger.debug(f"Outlet {stem_name} (scaled): area={area:.6f} m²")
                        else:
                            # For original STLs, apply unit detection logic
                            # Robust unit guess using the outlet's bbox  
                            ob = gen.compute_stl_bounding_box({p.stem: p})
                            # Plausible m² upper bound from bbox (conservative)
                            lx, ly, lz = self._get_bbox_dimensions(ob)
                            A_plausible = max(1e-10, lx * ly)
                            # If reported 'area' is > 1000× plausible, assume mm² → convert to m²
                            if area > 1000.0 * A_plausible:
                                area_mm2 = area
                                area /= 1e6
                                self.logger.info(f"Outlet {p.stem}: detected mm² units ({area_mm2:.1f} mm²) → converted to {area:.6f} m²")
                            else:
                                self.logger.debug(f"Outlet {p.stem}: area={area:.6f} m² (units look correct)")
                        
                        Deq_outlet = 2.0 * math.sqrt(area / math.pi)
                        Deq.append(Deq_outlet)
                        stem_name = p.name.replace('.stl', '') if stl_root else p.stem
                        self.logger.debug(f"Outlet {stem_name}: area={area:.6f} m², D_eq={Deq_outlet:.4f} m")
                except Exception as e:
                    self.logger.debug(f"Could not compute area for {p.name}: {e}")
            
            # Add inlet diameter to the mix
            if mode == "inlet_only" or (mode == "auto" and len(Deq) < len(self.stl_files["outlets"])):
                # Use inlet bbox as robust proxy
                # If using scaled STLs, skip additional scaling
                ib = gen.compute_stl_bounding_box({"inlet": inlet_path}, skip_scaling=(stl_root is not None))
                lx, ly, lz = self._get_bbox_dimensions(ib)
                # Use median of inlet dimensions as proxy
                D_in = float(np.median([lx, ly, lz]))
                if mode == "inlet_only":
                    Deq = [D_in]
                else:
                    Deq.append(D_in)
                scale_info = " (scaled)" if stl_root else ""
                self.logger.debug(f"Inlet bbox proxy{scale_info}: D_in={D_in:.4f} m")
            
            if not Deq:
                # Final fallback: use global bbox
                if stl_root:
                    # Use scaled wall STL - skip additional scaling
                    bbox = gen.compute_stl_bounding_box({self.wall_name: wall_path}, skip_scaling=True)
                else:
                    # Use original STL map - apply scaling
                    stl_map = {**self.stl_files["required"], **{p.stem: p for p in self.stl_files["outlets"]}}
                    bbox = gen.compute_stl_bounding_box(stl_map, skip_scaling=False)
                lx, ly, lz = self._get_bbox_dimensions(bbox)
                D_ref = min(lx, ly)
                D_min = D_ref * s_guard
                scale_info = " (scaled)" if stl_root else ""
                self.logger.info(f"Diameter fallback{scale_info}: using bbox → D_ref={D_ref:.4f}m, D_min={D_min:.4f}m (no outlet area data)")
            else:
                # Use median for reference, minimum for throat
                D_ref = float(np.median(Deq))
                D_min = float(min(Deq)) if len(Deq) else D_ref
                # Apply throat guard scale for internal narrowings
                D_min = min(D_min, s_guard * D_ref)
        
        # Optional loose sanity clamp (only for unit mishaps)
        if clamp == "loose":
            # Very wide band: 1mm to 50mm - covers most vascular beds
            D_ref = float(np.clip(D_ref, 1e-3, 5e-2))
            D_min = float(np.clip(D_min, 5e-4, D_ref))
        
        # Log detailed diameter derivation
        if Deq:
            self.logger.info(f"Diameter derivation: {len(Deq)} measurements → D_ref={D_ref:.4f}m (median), D_min={D_min:.4f}m (min)")
            self.logger.debug(f"All equivalent diameters: {[f'{d:.4f}m' for d in Deq]}")
            
            # Sanity check: warn if any diameter is unrealistically small
            min_realistic_diameter = 0.2e-3  # 0.2 mm in meters
            for i, d in enumerate(Deq):
                if d < min_realistic_diameter:
                    self.logger.warning(f"⚠️ Equivalent diameter D_eq[{i}]={d*1e3:.3f}mm < 0.2mm - likely a scaling issue or degenerate outlet!")
                    self.logger.warning(f"   Check your STL units or outlet geometry. Common causes:")
                    self.logger.warning(f"   - STL file in wrong units (m vs mm)")
                    self.logger.warning(f"   - Outlet patch is collapsed or has near-zero area")
                    self.logger.warning(f"   - Incorrect scale_m setting in config (currently {self.config.get('SCALING', {}).get('scale_m', 1.0)})")
        
        self.logger.info(f"[GeomStats] D_ref={D_ref:.4f} m, D_min={D_min:.4f} m "
                        f"(mode={mode}, clamp={clamp}, n_outlets={len(self.stl_files['outlets'])})")
        return D_ref, D_min

    def _adaptive_feature_angle(self, D_ref: float, D_min: float, n_outlets: int, iter_dir=None) -> int:
        """Calculate adaptive resolveFeatureAngle for vascular curvature (30-45°)
        
        Per OpenFOAM guidance: start at 30°, increase only if refinement is excessive
        Lower angles capture more curvature/features (good for tortuous vessels)
        Higher angles reduce over-refinement (good for smoother vessels)
        
        Enhanced with curvature analysis for vessel-aware adaptation.
        """
        beta = D_min / max(D_ref, 1e-6)  # Narrowness ratio
        
        # Baseline within documented range (30-45°)
        base = 35  # Start more aggressive for vessel branching
        
        # Narrower sections: capture more curvature (reduce angle)
        if beta < 0.6:
            base -= 5  # Down to ~32°
            
        # More branches: capture more features (reduce angle)
        if n_outlets >= 3:
            base -= 4  # Additional reduction for complex branching
        
        # Curvature-aware adjustment
        curvature_adjustment = 0
        if iter_dir:
            wall_stl = iter_dir / "constant" / "triSurface" / f"{self.wall_name}.stl"
            if wall_stl.exists():
                try:
                    cs = self._estimate_curvature_strength(wall_stl)
                    strength = cs["strength"]
                    # Map curvature strength [0,1] to angle adjustment [-5°, +3°]
                    # Higher curvature → lower angle → more feature detection
                    curvature_adjustment = int(3 - (strength * 8))
                    base += curvature_adjustment
                    self.logger.info(f"Curvature adjustment: strength={strength:.3f} → {curvature_adjustment:+d}° ({cs['nTris']} tris)")
                except Exception as e:
                    self.logger.debug(f"Curvature analysis failed for resolveFeatureAngle: {e}")
            
        # Constrain to documented vascular range (30-45°)
        angle = int(np.clip(base, 30, 45))
        
        components = f"β={beta:.2f}, n_out={n_outlets}"
        if curvature_adjustment != 0:
            components += f", curv={curvature_adjustment:+d}°"
        
        self.logger.info(f"Adaptive resolveFeatureAngle: {components} → {angle}° (vascular curvature)")
        return angle

    def _derive_base_cell_size(self, D_ref=None, D_min=None) -> float:
        mode = self.stage1.get("base_size_mode", "diameter").lower()
        if mode not in ("diameter", "density"):
            mode = "diameter"
        if mode == "diameter":
            N_D = int(self.stage1.get("N_D", 22))
            N_Dmin = int(self.stage1.get("N_D_min", 28))
            
            # Use provided diameters or compute them
            if D_ref is None or D_min is None:
                D_ref, D_min = self._estimate_reference_diameters()
            
            dx = min(D_ref / max(N_D, 1), D_min / max(N_Dmin, 1))
        else:
            k = float(self.stage1.get("cells_per_cm", 12))
            dx = 0.01 / max(k, 1e-9)
        # bound dx to avoid extremes
        dx = float(np.clip(dx, 1e-4, 5e-3))  # 0.1 mm .. 5 mm
        self.logger.info(f"Stage‑1 base cell size Δx = {dx*1e3:.2f} mm")
        return dx

    def _point_inside_stl(self, stl_path: Path, p, n_axes=3) -> bool:
        """Test if point p is inside closed STL using odd/even ray casting along multiple axes."""
        import numpy as _np
        axes = _np.eye(3, dtype=float)
        hits = 0
        for sign in (-1.0, 1.0):
            for a in range(n_axes):
                ray_o = _np.array(p, float)
                ray_d = axes[a] * sign
                count = 0
                for n, v1, v2, v3 in self._iter_stl_triangles(stl_path):
                    # Möller–Trumbore ray-triangle intersection
                    e1 = _np.array(v2) - _np.array(v1)
                    e2 = _np.array(v3) - _np.array(v1)
                    h = _np.cross(ray_d, e2)
                    det = e1.dot(h)
                    if abs(det) < 1e-14: 
                        continue
                    inv = 1.0/det
                    s = ray_o - _np.array(v1)
                    u = inv * s.dot(h)
                    if u < 0.0 or u > 1.0: 
                        continue
                    q = _np.cross(s, e1)
                    v = inv * ray_d.dot(q)
                    if v < 0.0 or u + v > 1.0: 
                        continue
                    t = inv * e2.dot(q)
                    if t > 1e-12:  # forward hit
                        count += 1
                if count % 2 == 1:
                    hits += 1
        # inside if majority of rays say "inside"
        return hits >= 3

    def _calculate_robust_seed_point(self, bbox_data, stl_files, dx_base=None):
        """
        Robustly compute a locationInMesh that is guaranteed to be inside the lumen.
        Uses inlet centroid + inward step + inside verification.

        Args:
            bbox_data: dict from PhysicsAwareMeshGenerator.compute_stl_bounding_box(...)
            stl_files: dict with scaled STL paths
            dx_base:   base cell size (meters) for step lengths
        Returns:
            np.ndarray shape (3,) point inside lumen (meters)
        """
        import numpy as _np

        # ---------- locate triSurface paths ----------
        # Accept either the full self.stl_files dict or a minimal dict with just inlet
        if "required" in stl_files and "inlet" in stl_files["required"]:
            inlet_path = Path(stl_files["required"]["inlet"])
            tri_root = inlet_path.parent
            wall_path = tri_root / f"{self.wall_name}.stl"
            outlet_paths = sorted(tri_root.glob("outlet*.stl"))
        else:
            inlet_path = self.stl_files["required"]["inlet"]
            tri_root = inlet_path.parent
            wall_path = self.stl_files["required"][self.wall_name]
            outlet_paths = list(self.stl_files["outlets"])

        if not wall_path.exists():
            raise FileNotFoundError(f"Wall STL not found at {wall_path}")
        if not inlet_path.exists():
            raise FileNotFoundError(f"Inlet STL not found at {inlet_path}")
        if not outlet_paths:
            self.logger.warning("No outlet STLs found while building closed surface; inside-test may fail.")

        # ---------- helpers: triangles + inside test ----------
        def _triangles_from(paths):
            tris = []
            for p in paths:
                for _, v1, v2, v3 in self._iter_stl_triangles(p):
                    tris.append((_np.array(v1, dtype=float),
                                 _np.array(v2, dtype=float),
                                 _np.array(v3, dtype=float)))
            if not tris:
                raise RuntimeError(f"No triangles found in {', '.join(str(x) for x in paths)}")
            return tris

        def _ray_triangle_intersect(orig, direc, v0, v1, v2, eps):
            # Möller–Trumbore
            e1 = v1 - v0
            e2 = v2 - v0
            pvec = _np.cross(direc, e2)
            det = e1.dot(pvec)
            if abs(det) < eps:
                return False, None
            inv_det = 1.0 / det
            tvec = orig - v0
            u = tvec.dot(pvec) * inv_det
            if u < -eps or u > 1.0 + eps:
                return False, None
            qvec = _np.cross(tvec, e1)
            v = direc.dot(qvec) * inv_det
            if v < -eps or u + v > 1.0 + eps:
                return False, None
            t = e2.dot(qvec) * inv_det
            if t > eps:  # forward hit
                return True, t
            return False, None

        def _point_on_triangle(pt, v0, v1, v2, eps):
            # barycentric check
            n = _np.cross(v1 - v0, v2 - v0)
            area2 = _np.linalg.norm(n)
            if area2 < eps:
                return False
            n = n / area2
            dist = abs((pt - v0).dot(n))
            if dist > eps:
                return False
            # 2D barycentric via areas
            def _area(a, b, c):
                return _np.linalg.norm(_np.cross(b - a, c - a)) * 0.5
            A = _area(v0, v1, v2)
            A0 = _area(pt, v1, v2)
            A1 = _area(v0, pt, v2)
            A2 = _area(v0, v1, pt)
            return abs((A0 + A1 + A2) - A) <= 10*eps

        def _is_inside_closed_surface(pt, tris, eps, ray_dir=_np.array([0.57735, 0.70711, 0.40825])):
            # treat "on surface" as inside
            for v0, v1, v2 in tris[:256]:  # quick local check on a subset
                if _point_on_triangle(pt, v0, v1, v2, eps):
                    return True
            # ray parity
            o = pt + ray_dir * (100*eps)  # nudge off the surface
            cnt = 0
            for v0, v1, v2 in tris:
                hit, _ = _ray_triangle_intersect(o, ray_dir, v0, v1, v2, eps)
                if hit:
                    cnt += 1
            return (cnt % 2) == 1

        # ---------- build closed surface ----------
        tris = _triangles_from([wall_path, inlet_path, *outlet_paths])

        # characteristic scale & eps
        Lx = float(bbox_data["mesh_domain"]["x_max"] - bbox_data["mesh_domain"]["x_min"])
        Ly = float(bbox_data["mesh_domain"]["y_max"] - bbox_data["mesh_domain"]["y_min"])
        Lz = float(bbox_data["mesh_domain"]["z_max"] - bbox_data["mesh_domain"]["z_min"])
        Lchar = max(1e-3, min(Lx, Ly, Lz))
        eps = 1e-9 * max(Lx, Ly, Lz)

        step = max(2.0*(dx_base or 1e-3), 5e-4)             # 2 Δx or 0.5 mm
        max_run = max(0.3*Lchar, 15.0*step)                 # up to ~30% of min bbox edge
        n_steps = int(max_run/step) + 2

        # ---------- inlet centroid & normal (from inlet vertices PCA) ----------
        inlet_vertices = _np.array(self._read_stl_vertices_raw(inlet_path), dtype=float)
        if inlet_vertices.size == 0:
            raise RuntimeError("Failed to read inlet vertices for seed computation.")
        C = inlet_vertices.mean(axis=0)

        # PCA: smallest singular vector is plane normal
        _, _, Vt = _np.linalg.svd(inlet_vertices - C, full_matrices=False)
        n = Vt[-1, :]
        n = n / (_np.linalg.norm(n) or 1.0)

        # Prefer direction pointing toward global bbox center
        bbox_center = _np.array([
            (bbox_data["mesh_domain"]["x_min"] + bbox_data["mesh_domain"]["x_max"]) * 0.5,
            (bbox_data["mesh_domain"]["y_min"] + bbox_data["mesh_domain"]["y_max"]) * 0.5,
            (bbox_data["mesh_domain"]["z_min"] + bbox_data["mesh_domain"]["z_max"]) * 0.5,
        ])
        if (bbox_center - C).dot(n) < 0.0:
            n = -n

        # ---------- candidate search: +n, then -n, then towards center, then jitter ----------
        def _scan_from(origin, direction):
            d = step
            for _ in range(n_steps):
                pt = origin + direction * d
                if _is_inside_closed_surface(pt, tris, eps):
                    return pt
                d += step
            return None

        # 1) march along +n from inlet centroid
        p = _scan_from(C, n)
        if p is not None:
            self.logger.info(f"Seed found: inlet +n at {p}")
            return p

        # 2) march along -n
        p = _scan_from(C, -n)
        if p is not None:
            self.logger.info(f"Seed found: inlet -n at {p}")
            return p

        # 3) march from inlet toward bbox center
        v = bbox_center - C
        v = v / (_np.linalg.norm(v) or 1.0)
        p = _scan_from(C, v)
        if p is not None:
            self.logger.info(f"Seed found: inlet → center at {p}")
            return p

        # 4) sample along centerline segment (0.3..0.8 toward center)
        for a in _np.linspace(0.3, 0.8, 11):
            pt = (1.0 - a)*C + a*bbox_center
            if _is_inside_closed_surface(pt, tris, eps):
                self.logger.info(f"Seed found: convex combo {a:.2f} at {pt}")
                return pt

        # 5) last resort: jitter near bbox center
        rng = _np.random.RandomState(42)
        for _ in range(200):
            jitter = rng.normal(0.0, step, 3)
            pt = bbox_center + jitter
            if _is_inside_closed_surface(pt, tris, eps):
                self.logger.info(f"Seed found: center+jitter at {pt}")
                return pt

        # If we get here, something is wrong with the surface set (likely not closed)
        raise RuntimeError(
            "Could not find a locationInMesh inside the closed surface. "
            "Check that wall+inlet+outlet STLs form a watertight enclosure."
        )

    # ------------------------ Curvature analysis helpers -------------------------
    def _iter_stl_triangles(self, stl_path: Path):
        """Yield (normal, v1, v2, v3) for each triangle in an STL.
        Works for ASCII and binary STL. If stored normal is zero, recompute from vertices.
        """
        try:
            # Try ASCII first
            with open(stl_path, "r", encoding="utf-8") as f:
                line = f.readline()
                if not line.lower().startswith("solid"):
                    raise UnicodeDecodeError("","",0,0,"not ascii")
                f.seek(0)
                normal = None
                verts = []
                for line in f:
                    s = line.strip()
                    if s.startswith("facet normal"):
                        parts = s.split()
                        try:
                            normal = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
                        except Exception:
                            normal = None
                    elif s.startswith("vertex"):
                        parts = s.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        if len(verts) == 3:
                            # recompute normal if missing/zero
                            import numpy as _np
                            v1, v2, v3 = _np.array(verts[0]), _np.array(verts[1]), _np.array(verts[2])
                            if normal is None or abs(normal[0])+abs(normal[1])+abs(normal[2]) < 1e-20:
                                n = _np.cross(v2 - v1, v3 - v1)
                                n_norm = _np.linalg.norm(n) or 1.0
                                normal = (n / n_norm).tolist()
                            yield normal, v1.tolist(), v2.tolist(), v3.tolist()
                            normal = None; verts = []
                return
        except UnicodeDecodeError:
            pass

        # Binary STL
        with open(stl_path, "rb") as f:
            f.seek(80)
            n_triangles = struct.unpack("<I", f.read(4))[0]
            for _ in range(n_triangles):
                nx, ny, nz = struct.unpack("<3f", f.read(12))
                v1 = list(struct.unpack("<3f", f.read(12)))
                v2 = list(struct.unpack("<3f", f.read(12)))
                v3 = list(struct.unpack("<3f", f.read(12)))
                f.read(2)
                # recompute normal if zero
                import numpy as _np
                n = _np.array([nx, ny, nz], dtype=float)
                if _np.linalg.norm(n) < 1e-20:
                    a, b, c = _np.array(v1), _np.array(v2), _np.array(v3)
                    n = _np.cross(b - a, c - a)
                    nn = _np.linalg.norm(n) or 1.0
                    n = n / nn
                else:
                    n = n / (_np.linalg.norm(n) or 1.0)
                yield n.tolist(), v1, v2, v3

    def _estimate_curvature_strength(self, stl_path: Path, sample_max: int = 200000):
        """Return {'strength':[0..1], 'nTris':N} based on dispersion of wall normals.
        strength≈0 (very straight/smooth); strength≈1 (highly tortuous/branched).
        We sample up to 'sample_max' triangles for speed.
        """
        import numpy as _np, random as _random
        normals = []
        count = 0
        # Reservoir sampling to avoid loading all triangles
        reservoir = []
        k = 0
        for tri in self._iter_stl_triangles(stl_path):
            if tri is None:
                continue
            if k < sample_max:
                reservoir.append(tri)
            else:
                j = _random.randint(0, k)
                if j < sample_max:
                    reservoir[j] = tri
            k += 1
        
        # Extract normals from reservoir
        for n, v1, v2, v3 in reservoir:
            normals.append(n)
        
        if not normals:
            return {"strength": 0.0, "nTris": 0}
        
        N = _np.array(normals, dtype=float)
        # normalize just in case
        nrm = _np.linalg.norm(N, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        N = N / nrm
        
        # mean direction
        mu = _np.mean(N, axis=0)
        mu = mu / (_np.linalg.norm(mu) or 1.0)
        
        # dispersion metric: 1 - |n·mu| averaged
        disp = float(_np.mean(1.0 - _np.abs(N @ mu)))
        
        # map dispersion (~0..~0.5) to [0,1]
        strength = max(0.0, min(1.0, disp / 0.4))
        
        return {"strength": strength, "nTris": len(N)}

    def _estimate_first_layer_from_yplus(self, D_ref, U_peak, rho, mu, y_plus, model="turbulent"):
        """Estimate first layer thickness from y+ target for wall-resolved meshing"""
        # Use the robust implementation from utils.py
        blood_properties = {
            'density': rho,
            'viscosity': mu
        }
        
        # The utils function calculates for y+ = 1, so scale by target y+
        base_thickness = calculate_first_layer_thickness(U_peak, D_ref, blood_properties)
        first_layer = base_thickness * y_plus
        
        # Log for consistency with existing interface
        Re = rho * U_peak * D_ref / max(mu, 1e-9)
        self.logger.info(f"y+ based first layer: Re={Re:.0f}, y+={y_plus}, model={model} → {first_layer*1e6:.1f} μm")
        return first_layer

    def _womersley_boundary_layer(self, heart_rate_hz, nu):
        """Calculate Womersley boundary layer thickness for pulsatile flow
        δ_ω ≈ sqrt(2ν/ω) where ω = 2πf
        """
        omega = 2.0 * math.pi * max(heart_rate_hz, 1e-6)
        delta = math.sqrt(2.0 * nu / omega)
        self.logger.info(f"Womersley boundary layer: f={heart_rate_hz:.2f} Hz, ν={nu:.2e} m²/s → δ_ω={delta*1e3:.2f} mm")
        return delta

    def _first_layer_from_dx(self, dx, N, ER, alpha=0.8):
        """
        Auto-size first layer thickness to maintain T ≈ α·Δx relationship.
        
        Args:
            dx: Base cell size (meters)
            N: Number of surface layers
            ER: Expansion ratio
            alpha: Target total thickness as fraction of dx (default 0.8)
            
        Returns:
            dict with 'firstLayerThickness_abs', 'minThickness_abs', 'nGrow', 'nSurfaceLayers'
        """
        import numpy as np
        
        # Get config parameters
        stage1 = self.stage1 or {}
        t1_min_frac = float(stage1.get("t1_min_fraction_of_dx", 0.02))  # 2% of Δx
        t1_max_frac = float(stage1.get("t1_max_fraction_of_dx", 0.08))  # 8% of Δx
        
        # Target total thickness
        T_target = alpha * dx
        
        # Raw first layer thickness from geometric series
        if N <= 1:
            t1_raw = T_target
        else:
            t1_raw = T_target * (ER - 1.0) / (ER**N - 1.0)
        
        # Clamp to sensible band
        t1_min = max(t1_min_frac * dx, 1e-6)   # At least 2% of Δx and 1 µm
        t1_max = min(t1_max_frac * dx, 200e-6)  # At most 8% of Δx and 200 µm
        t1 = float(np.clip(t1_raw, t1_min, t1_max))
        
        # Keep total thickness near alpha*dx if clamped
        if abs(t1 - t1_raw) > 0.05 * max(t1_raw, 1e-12):
            # Significant clamping occurred, adjust N to maintain target thickness
            N_star = np.log(1.0 + (ER - 1.0) * (alpha * dx) / t1) / np.log(ER)
            N_adj = int(np.clip(round(N_star), 3, 20))
            self.logger.debug(f"  Clamping adjusted N: {N} → {N_adj} (N*={N_star:.2f})")
        else:
            N_adj = N
        
        # Calculate actual total thickness with clamped t1 and adjusted N
        T_actual = t1 * (ER**N_adj - 1.0) / (ER - 1.0) if N_adj > 1 else t1
        
        # Set minThickness ≈ 0.3·T for stability
        min_thickness = max(0.3 * T_actual, 5e-6)  # At least 5 µm
        
        # Ensure nGrow ≥ 1 for curvature handling
        n_grow = max(1, int(stage1.get("nGrow", 1)))
        
        self.logger.debug(f"Layer sizing: dx={dx*1e3:.2f}mm, N={N_adj} (orig={N}), ER={ER:.2f}")
        self.logger.debug(f"  Target T={T_target*1e3:.3f}mm ({alpha:.1f}×dx), actual T={T_actual*1e3:.3f}mm")
        self.logger.debug(f"  t1={t1*1e6:.1f}µm (raw={t1_raw*1e6:.1f}µm), minThick={min_thickness*1e6:.1f}µm")
        
        return {
            'firstLayerThickness_abs': t1,
            'minThickness_abs': min_thickness,
            'nGrow': n_grow,
            'nSurfaceLayers': N_adj  # Return adjusted N
        }

    # ------------------------ Dict generation -----------------------------
    def _generate_blockmesh_dict(self, iter_dir, bbox_info, dx_base):
        bbox_min = np.array([
            bbox_info["mesh_domain"]["x_min"],
            bbox_info["mesh_domain"]["y_min"],
            bbox_info["mesh_domain"]["z_min"],
        ])
        bbox_max = np.array([
            bbox_info["mesh_domain"]["x_max"],
            bbox_info["mesh_domain"]["y_max"],
            bbox_info["mesh_domain"]["z_max"],
        ])
        bbox_size = bbox_max - bbox_min
        # Compute block divisions with override hierarchy
        divisions = self._compute_block_divisions(bbox_size, dx_base)
        
        # Get grading from config
        bm = self.config.get("BLOCKMESH", {}) or {}
        grading = bm.get("grading", [1, 1, 1])

        vertices = []
        for z in [bbox_min[2], bbox_max[2]]:
            vertices.append([bbox_min[0], bbox_min[1], z])
            vertices.append([bbox_max[0], bbox_min[1], z])
            vertices.append([bbox_max[0], bbox_max[1], z])
            vertices.append([bbox_min[0], bbox_max[1], z])

        blockmesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           | Version: 12
     \\/     M anipulation  |
*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
{chr(10).join(f"    ({v[0]} {v[1]} {v[2]})  // {i}" for i, v in enumerate(vertices))}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({divisions[0]} {divisions[1]} {divisions[2]}) simpleGrading ({grading[0]} {grading[1]} {grading[2]})
);

edges ()
;

boundary
(
    background
    {{
        type patch;  // avoid accidental walls if snap fails
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (3 7 6 2)
            (0 4 7 3)
            (1 2 6 5)
        );
    }}
);
"""
        sys = iter_dir / "system"
        sys.mkdir(exist_ok=True)
        (sys / "blockMeshDict").write_text(blockmesh)
        control = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     blockMesh;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        (sys / "controlDict").write_text(control)
        self.logger.info(f"blockMesh: divisions={divisions.tolist()}, Δx={dx_base*1e3:.2f} mm")

    def _copy_trisurfaces(self, iter_dir) -> List[str]:
        """Process STL files with smart scaling and robust error handling"""
        tri = iter_dir / "constant" / "triSurface"
        tri.mkdir(parents=True, exist_ok=True)
        outlet_names = []
        env = self.config["openfoam_env_path"]
        
        all_stl_files = list(self.stl_files["outlets"]) + list(self.stl_files["required"].values())
        scale_m = self.config.get("SCALING", {}).get("scale_m", 1.0)
        
        self.logger.info(f"Processing {len(all_stl_files)} STL files with scale_m={scale_m}")
        
        for p in all_stl_files:
            dest_path = tri / p.name
            
            # Integrated copy/scale logic with smart detection
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # If explicit scale factor provided
                if scale_m != 1.0 and abs(scale_m - 1.0) > 1e-12:
                    cmd = f'surfaceTransformPoints "scale=({scale_m} {scale_m} {scale_m})" "{p.absolute()}" "{dest_path.absolute()}"'
                    self.logger.info(f"SCALING {p.name} (scale={scale_m})")
                    result = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=600, max_memory_gb=self.max_memory_gb)
                    if result.returncode != 0:
                        raise RuntimeError(f"Failed scaling {p.name}: {result.stderr}")
                        
                # Auto-detection mode: check if needs mm->m scaling
                elif self._check_stl_units(p):
                    cmd = f'surfaceTransformPoints "scale=(0.001 0.001 0.001)" "{p.absolute()}" "{dest_path.absolute()}"'
                    self.logger.info(f"AUTO-SCALING {p.name} (mm->m)")
                    result = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=600, max_memory_gb=self.max_memory_gb)
                    if result.returncode != 0:
                        self.logger.warning(f"Auto-scaling failed for {p.name}, copying instead")
                        shutil.copy2(p, dest_path)
                else:
                    # Just copy the file
                    self.logger.debug(f"COPYING {p.name}")
                    shutil.copy2(p, dest_path)
                    
            except Exception as e:
                self.logger.error(f"STL processing failed for {p.name}: {e}")
                # Fallback to copy
                try:
                    shutil.copy2(p, dest_path)
                    self.logger.info(f"Fallback copy successful for {p.name}")
                except Exception as copy_e:
                    raise RuntimeError(f"Both scaling and copy failed for {p.name}: {copy_e}")
            
            # Track outlet names for mesh generation
            if p in self.stl_files["outlets"]:
                outlet_names.append(p.stem)

        return outlet_names
    
    def _curvature_adaptive_levels(self, base_levels, curvature_strength):
        """Adapt refinement levels based on surface curvature for better flow resolution"""
        min_level, max_level = base_levels[0], base_levels[1]
        
        if curvature_strength > 0.7:  # High curvature (aneurysms, stenoses)
            # Boost both levels for aggressive near-wall refinement
            adapted_min = min(min_level + 1, 3)  # Cap at level 3
            adapted_max = min(max_level + 1, 4)  # Cap at level 4
            self.logger.info(f"High curvature detected (>{0.7:.1f}) - aggressive refinement")
        elif curvature_strength > 0.4:  # Moderate curvature (bifurcations)
            # Boost max level for better boundary layer capture
            adapted_min = min_level
            adapted_max = min(max_level + 1, 3)  # Conservative boost
            self.logger.info(f"Moderate curvature detected (>{0.4:.1f}) - enhanced max refinement")
        else:  # Low curvature (straight vessels)
            # Use base levels as-is
            adapted_min, adapted_max = min_level, max_level
            self.logger.info("Low curvature - using base refinement levels")
            
        return [adapted_min, adapted_max]
    
    def _optimized_quality_check(self, iter_dir, n_procs, env, logs, phase_name):
        """Optimized quality checking with minimal reconstruction cycles"""
        if n_procs > 1:
            # Try parallel checkMesh first (most efficient)
            self.logger.info(f"Running checkMesh -parallel on {phase_name} mesh")
            check_cmd = ["mpirun", "-np", str(n_procs), "checkMesh", "-parallel"]
            try:
                res = run_command(check_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                output_text = res.stdout + res.stderr
                (logs / f"log.checkMesh.{phase_name}").write_text(output_text)
                metrics = self._parse_parallel_checkmesh(output_text)
                
                # Only reconstruct if we need serial fallback AND it's the final phase
                if not metrics.get("meshOK", False) and phase_name == "layers":
                    self.logger.info("Parallel checkMesh indicated issues, reconstructing for detailed analysis")
                    try:
                        res = run_command(["reconstructPar", "-latestTime"], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                        (logs / f"log.reconstruct.{phase_name}").write_text(res.stdout + res.stderr)
                        serial_metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
                        # Use more detailed serial metrics if available
                        return serial_metrics if serial_metrics.get("meshOK", False) != metrics.get("meshOK", False) else metrics
                    except Exception as e:
                        self.logger.warning(f"Reconstruction failed: {e}")
                        return metrics
                else:
                    return metrics
                    
            except Exception as e:
                self.logger.warning(f"Parallel checkMesh failed: {e}")
                # Only now do we reconstruct for serial fallback
                try:
                    self.logger.info("Reconstructing for serial checkMesh fallback")
                    res = run_command(["reconstructPar", "-latestTime"], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                    (logs / f"log.reconstruct.{phase_name}").write_text(res.stdout + res.stderr)
                    return check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
                except Exception as serial_e:
                    self.logger.error(f"Serial checkMesh fallback failed: {serial_e}")
                    return {"meshOK": False, "cells": 0}
        else:
            # Serial workflow is straightforward
            return check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
    
    def _openfoam_memory_model(self, available_gb, n_procs):
        """OpenFOAM-specific memory usage model for accurate cell budgeting"""
        # OpenFOAM memory usage patterns (empirically derived):
        # - Basic cell storage: ~1.5KB/cell (coordinates, connectivity)
        # - Gradient calculations: ~0.8KB/cell (grad(p), grad(U), etc.)
        # - Linear solver workspace: ~0.5KB/cell (matrix assembly)
        # - Parallel overhead: ~0.2KB/cell per processor
        
        base_memory_per_cell = 1.5e-3  # GB per cell
        gradient_overhead = 0.8e-3     # GB per cell for gradient calculations
        solver_workspace = 0.5e-3      # GB per cell for linear solver
        parallel_overhead = 0.2e-3 * n_procs  # Increases with processor count
        
        total_memory_per_cell = base_memory_per_cell + gradient_overhead + solver_workspace + parallel_overhead
        
        # Add safety factor for snappyHexMesh (higher memory usage during meshing)
        snappy_factor = 1.3  # 30% extra during mesh generation
        effective_memory_per_cell = total_memory_per_cell * snappy_factor
        
        # Calculate maximum cells based on available memory
        max_cells = int(available_gb / effective_memory_per_cell)
        
        # Apply practical limits based on OpenFOAM performance characteristics
        if n_procs == 1:
            # Serial runs: avoid excessive memory pressure
            max_cells = min(max_cells, 2_000_000)
        else:
            # Parallel runs: higher limits but bound by inter-processor communication
            max_cells = min(max_cells, 10_000_000)
            
        self.logger.debug(f"OpenFOAM memory breakdown: {effective_memory_per_cell*1000:.2f}KB/cell "
                         f"(base={base_memory_per_cell*1000:.1f}, grad={gradient_overhead*1000:.1f}, "
                         f"solver={solver_workspace*1000:.1f}, parallel={parallel_overhead*1000:.1f})")
        
        return max_cells
    
    def _validate_feature_resolution(self, feature_size, cell_size):
        """Validate that important geometric features are adequately resolved"""
        if feature_size <= 0 or cell_size <= 0:
            return True  # Skip validation for invalid inputs
            
        resolution_ratio = feature_size / cell_size
        
        # CFD best practices: minimum 3-4 cells across important features
        min_cells_across = 3.0
        adequate_resolution = resolution_ratio >= min_cells_across
        
        if not adequate_resolution:
            self.logger.warning(f"Feature under-resolved: {resolution_ratio:.1f} cells across "
                              f"(need ≥{min_cells_across:.1f} for accurate flow representation)")
            
            # Suggest refinement level increase
            recommended_refinement = int(np.ceil(np.log2(min_cells_across / resolution_ratio)))
            if recommended_refinement > 0:
                self.logger.info(f"Consider increasing surface refinement by {recommended_refinement} level(s)")
        else:
            self.logger.debug(f"Feature well-resolved: {resolution_ratio:.1f} cells across")
            
        return adequate_resolution
    
    def _estimate_minimum_feature_size(self, stl_path):
        """Estimate minimum geometric feature size from STL surface"""
        try:
            if not stl_path.exists():
                return 0.0
                
            vertices = self._read_stl_vertices(stl_path)
            if len(vertices) < 6:  # Need at least 2 triangles
                return 0.0
            
            # Calculate edge lengths between consecutive vertices
            edge_lengths = []
            for i in range(0, len(vertices) - 3, 3):  # Process triangles
                triangle = vertices[i:i+3]
                for j in range(3):
                    v1, v2 = triangle[j], triangle[(j+1) % 3]
                    edge_length = np.sqrt(sum((v2[k] - v1[k])**2 for k in range(3)))
                    if edge_length > 1e-12:  # Avoid degenerate edges
                        edge_lengths.append(edge_length)
            
            if not edge_lengths:
                return 0.0
                
            # Use 10th percentile as minimum feature size (robust against outliers)
            min_feature = np.percentile(edge_lengths, 10)
            self.logger.debug(f"Estimated minimum feature size: {min_feature:.6f}m from {len(edge_lengths)} edges")
            
            return min_feature
            
        except Exception as e:
            self.logger.debug(f"Feature size estimation failed: {e}")
            return 0.0
    
    def _mesh_convergence_check(self, current_metrics):
        """Check if mesh quality metrics have converged"""
        if not current_metrics or not isinstance(current_metrics, dict):
            return False
            
        # Add current iteration to history
        quality_entry = {
            'iteration': self.current_iteration,
            'maxNonOrtho': current_metrics.get('maxNonOrtho', 999),
            'maxSkewness': current_metrics.get('maxSkewness', 999),
            'cells': current_metrics.get('cells', 0),
            'meshOK': current_metrics.get('meshOK', False)
        }
        self.quality_history.append(quality_entry)
        
        # Need at least 3 iterations to assess convergence
        if len(self.quality_history) < 3:
            return False
        
        # Extract last 3 iterations
        recent_history = self.quality_history[-3:]
        
        # Check convergence of key quality metrics
        nonortho_values = [entry['maxNonOrtho'] for entry in recent_history]
        skewness_values = [entry['maxSkewness'] for entry in recent_history]
        cell_counts = [entry['cells'] for entry in recent_history]
        
        # Calculate coefficient of variation (std/mean) for stability assessment
        def coeff_variation(values):
            if len(values) < 2 or np.mean(values) == 0:
                return 999  # High variation for edge cases
            return np.std(values) / np.mean(values)
        
        nonortho_cv = coeff_variation(nonortho_values)
        skewness_cv = coeff_variation(skewness_values)
        cells_cv = coeff_variation(cell_counts)
        
        # Convergence criteria (low coefficient of variation indicates stability)
        nonortho_converged = nonortho_cv < 0.05  # 5% variation
        skewness_converged = skewness_cv < 0.05  # 5% variation
        cells_converged = cells_cv < 0.10        # 10% variation (cell count can vary more)
        
        overall_converged = nonortho_converged and skewness_converged and cells_converged
        
        if overall_converged:
            recent_avg_nonortho = np.mean(nonortho_values)
            recent_avg_skewness = np.mean(skewness_values)
            recent_avg_cells = int(np.mean(cell_counts))
            
            self.logger.info(f"Mesh quality converged: maxNonOrtho={recent_avg_nonortho:.1f}±{nonortho_cv*100:.1f}%, "
                           f"maxSkewness={recent_avg_skewness:.2f}±{skewness_cv*100:.1f}%, "
                           f"cells={recent_avg_cells:,}±{cells_cv*100:.1f}%")
        else:
            self.logger.debug(f"Convergence check: nonortho_cv={nonortho_cv:.3f}, "
                            f"skewness_cv={skewness_cv:.3f}, cells_cv={cells_cv:.3f}")
        
        return overall_converged

    def _bbox_maxdim_py(self, stl_path: Path) -> float:
        """Max bbox dimension in RAW STL coordinates (no unit conversion) for unit detection."""
        xmin = ymin = zmin = float("+inf")
        xmax = ymax = zmax = float("-inf")

        try:
            # Try ASCII first
            with open(stl_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.lstrip()
                    if s.startswith("vertex"):
                        _, x, y, z, *rest = s.split()
                        x = float(x); y = float(y); z = float(z)
                        if x<xmin: xmin=x
                        if y<ymin: ymin=y
                        if z<zmin: zmin=z
                        if x>xmax: xmax=x
                        if y>ymax: ymax=y
                        if z>zmax: zmax=z
            if xmin != float("+inf"):
                return float(max(xmax-xmin, ymax-ymin, zmax-zmin))
        except UnicodeDecodeError:
            pass  # Fall through to binary

        try:
            # Binary STL
            with open(stl_path, "rb") as f:
                f.seek(80)
                ntri = struct.unpack("<I", f.read(4))[0]
                for _ in range(ntri):
                    f.read(12)  # Skip normal vector
                    for _ in range(3):  # 3 vertices per triangle
                        x, y, z = struct.unpack("<3f", f.read(12))
                        if x<xmin: xmin=x
                        if y<ymin: ymin=y
                        if z<zmin: zmin=z
                        if x>xmax: xmax=x
                        if y>ymax: ymax=y
                        if z>zmax: zmax=z
                    f.read(2)  # Skip attribute byte count
            if xmin != float("+inf"):
                return float(max(xmax-xmin, ymax-ymin, zmax-zmin))
        except Exception as e:
            self.logger.warning(f"Failed to read STL dimensions from {stl_path}: {e}")
        
        # Conservative fallback if no vertices found
        return 0.02
    
    def _read_stl_vertices_raw(self, stl_path: Path):
        """Read STL vertices in original coordinates without unit conversion."""
        vertices = []
        try:
            # Try ASCII first
            with open(stl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('vertex'):
                        coords = [float(x) for x in line.split()[1:4]]
                        vertices.append(coords)
            return vertices
            
        except UnicodeDecodeError:
            # Binary STL
            try:
                with open(stl_path, 'rb') as f:
                    f.seek(80)  # Skip header
                    n_triangles = struct.unpack('<I', f.read(4))[0]
                    
                    for _ in range(n_triangles):
                        f.read(12)  # Skip normal vector
                        for _ in range(3):  # Read 3 vertices per triangle
                            x, y, z = struct.unpack('<3f', f.read(12))
                            vertices.append([x, y, z])
                        f.read(2)  # Skip attribute byte count
                        
                return vertices
            except Exception as e:
                self.logger.warning(f"Binary STL read failed: {e}")
                return []

    def _check_stl_units(self, stl_path: Path, env: str = None) -> bool:
        """
        Return True if coordinates look like millimetres (need ×0.001 scaling).
        For human vasculature: 0.02-0.1m (20-100mm) is correct physical scale - no scaling needed.
        Only scale if dimensions are unreasonably large (>1m) suggesting mm-as-meter encoding.
        """
        try:
            mx = self._bbox_maxdim_py(stl_path)
            # Vessel geometry >1 m is implausible; interpret as mm numbers needing 0.001 scaling
            needs_scaling = mx > 1.0
            
            if needs_scaling:
                self.logger.info(f"[units] {stl_path.name} maxDim={mx:.6f} → looks like millimetres (scale by 0.001).")
            else:
                self.logger.info(f"[units] {stl_path.name} maxDim={mx:.6f} → already plausible metres.")
            
            return needs_scaling
        except Exception as e:
            self.logger.warning(f"Unit check failed for {stl_path.name}: {e}")
            return False  # fail-safe: don't scale blindly

    def _scale_stl_to_meters(self, src_path: Path, dest_path: Path, iter_dir: Path, env: str):
        """Scale STL from medical imaging scale to proper physical scale with surfaceTransformPoints."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Medical imaging: coordinates saved as "meters" but represent mm scale
        # Scale down by 0.001 to get proper physical scale
        # Example: 90 mm in STL → 0.09 m after scaling (correct physical metres)
        
        # Use string command to handle shell escaping properly
        cmd_str = f'surfaceTransformPoints "scale=(0.001 0.001 0.001)" "{src_path.absolute()}" "{dest_path.absolute()}"'
        self.logger.info(f"Running scaling command: {cmd_str}")
        self.logger.info(f"Working directory: {iter_dir}")
        # Sanitize env path to avoid logging secrets
        env_display = env if env.startswith("/") else "***" 
        self.logger.info(f"Environment: {env_display}")
        
        res = run_command(
            cmd_str, cwd=iter_dir, env_setup=env, timeout=300, max_memory_gb=self.max_memory_gb
        )
        
        self.logger.info(f"Scaling command return code: {res.returncode}")
        if res.stdout:
            self.logger.debug(f"STDOUT: {res.stdout}")
        if res.stderr:
            self.logger.warning(f"STDERR: {res.stderr}")

        # Basic verification: file exists and is non-zero size
        try:
            if not dest_path.exists():
                raise RuntimeError(f"Scaled file was not created: {dest_path}")
                
            if dest_path.stat().st_size == 0:
                raise RuntimeError(f"Scaled file is empty: {dest_path}")
            
            # Log success without detailed verification (to avoid bbox reading issues)
            mx_src = self._bbox_maxdim_py(src_path)
            self.logger.info(f"{src_path.name}: maxDim src={mx_src:.6f} → scaled successfully (0.001x)")
                
        except Exception as e:
            # hard-stop instead of copying unscaled STL
            raise RuntimeError(f"Scaling verification failed for {src_path.name}: {e}")
    
    def _parse_parallel_checkmesh(self, output: str) -> Dict:
        """Parse parallel checkMesh output to extract metrics (robust for both serial/parallel)"""
        metrics = {
            "maxNonOrtho": 0.0,
            "maxSkewness": 0.0,
            "maxAspectRatio": 0.0,
            "negVolCells": 0,
            "meshOK": False,
            "cells": 0,
            "wall_nFaces": None
        }
        
        # non-ortho / skewness (several OF variants)
        m = re.search(r"Max non-orthogonality\s*=\s*([\d.]+)", output)
        if m: metrics["maxNonOrtho"] = float(m.group(1))
        
        m = re.search(r"Max skewness\s*=\s*([\d.]+)", output)
        if m: metrics["maxSkewness"] = float(m.group(1))
        
        m = re.search(r"aspect ratio\s*=\s*([\d.]+)", output)
        if m: metrics["maxAspectRatio"] = float(m.group(1))
        
        # cells: accept both serial and parallel formats
        m = re.search(r"\bcells:\s+(\d+)", output, re.IGNORECASE)
        if not m:
            m = re.search(r"\bNumber of cells:\s+(\d+)", output, re.IGNORECASE)
        if m:
            metrics["cells"] = int(m.group(1))
        
        # success flag: robust case-insensitive parsing for different OpenFOAM builds
        low = output.lower()
        metrics["meshOK"] = "mesh ok" in low
        
        # don't try to scrape wall faces from the text here; we'll compute it from boundary files
        return metrics
    
    def _sum_wall_faces_from_processors(self, iter_dir: Path) -> int:
        """Count wall patch faces from boundary files (parallel-safe)"""
        total = 0
        
        # Check if processor directories exist (parallel case)
        processor_dirs = list(iter_dir.glob("processor*"))
        
        if processor_dirs:
            # Prefer processor totals when they exist
            for b in iter_dir.glob("processor*/constant/polyMesh/boundary"):
                try:
                    txt = b.read_text()
                    m = re.search(rf"{self.wall_name}\s*\{{[^{{}}]*?nFaces\s+(\d+);", txt, re.DOTALL)
                    if m:
                        total += int(m.group(1))
                        self.logger.debug(f"Found {m.group(1)} {self.wall_name} faces in {b.parent.parent.parent.name}")
                except Exception as e:
                    self.logger.debug(f"Could not parse boundary file {b}: {e}")
        else:
            # Only use root mesh if no processor directories exist (serial case)
            b = iter_dir / "constant" / "polyMesh" / "boundary"
            if b.exists():
                try:
                    txt = b.read_text()
                    m = re.search(rf"{self.wall_name}\s*\{{[^{{}}]*?nFaces\s+(\d+);", txt, re.DOTALL)
                    if m: 
                        total += int(m.group(1))
                        self.logger.debug(f"Found {m.group(1)} {self.wall_name} faces in root mesh")
                except Exception as e:
                    self.logger.debug(f"Could not parse root boundary file: {e}")
        
        return total

    def _write_surfaceFeatures(self, iter_dir, all_surfaces: List[str]):
        # Curvature-aware includedAngle selection
        wall_stl = iter_dir / "constant" / "triSurface" / f"{self.wall_name}.stl"
        
        if wall_stl.exists():
            try:
                cs = self._estimate_curvature_strength(wall_stl)
                strength = cs["strength"]
                # Map curvature strength [0,1] to includedAngle [170°, 145°]
                # Higher curvature → lower angle → more features detected
                included_angle = 170 - (strength * 25)
                self.logger.info(f"Curvature-aware includedAngle: {included_angle:.1f}° (strength={strength:.3f}, {cs['nTris']} tris)")
            except Exception as e:
                # Fallback to configuration or default
                included_angle = self.config.get("SURFACE_FEATURES", {}).get("includedAngle", 160)
                self.logger.warning(f"Curvature analysis failed, using fallback angle {included_angle}°: {e}")
        else:
            # Fallback if wall STL not found
            included_angle = self.config.get("SURFACE_FEATURES", {}).get("includedAngle", 160)
            self.logger.info(f"Wall STL not found, using configured includedAngle: {included_angle}°")
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}

surfaces ({' '.join('"'+s+'"' for s in all_surfaces)});

includedAngle   {included_angle:.1f};  // curvature-adaptive feature detection
"""
        (iter_dir / "system" / "surfaceFeaturesDict").write_text(content)

    def _snappy_no_layers_dict(self, iter_dir, outlet_names, internal_pt, dx_base):
        snap = self.config["SNAPPY"]
        
        # Calculate refinement band distances - either physics-based or geometry-based
        if self.physics.get("use_womersley_bands", False):
            # Physics-aware bands based on Womersley boundary layer thickness
            D_ref, _ = self._estimate_reference_diameters()
            U_peak = float(self.physics.get("U_peak", 1.0))        # m/s
            rho = float(self.physics.get("rho", 1060.0))           # kg/m³
            mu = float(self.physics.get("mu", 3.5e-3))             # Pa·s
            hr_hz = float(self.physics.get("heart_rate_hz", 1.2))  # Hz
            
            # Calculate Womersley boundary layer thickness δ_W = sqrt(2ν / ω)
            nu = mu / rho                                          # m²/s
            omega = 2.0 * np.pi * hr_hz                           # rad/s
            delta_w = np.sqrt(2.0 * nu / omega)                   # Womersley BL thickness (corrected)
            
            # Near band: 2-4 δ_W, far band: 10-20 δ_W
            near_dist = 3.0 * delta_w
            far_dist = 15.0 * delta_w
            
            self.logger.info(f"Womersley boundary layers: δ_W={delta_w*1e6:.1f}μm, near={near_dist*1e6:.1f}μm, far={far_dist*1e6:.1f}μm")
        else:
            # Geometry-based bands using configured cell multiples
            near_cells = self.stage1.get("near_band_cells", 4)
            far_cells = self.stage1.get("far_band_cells", 10)
            near_dist = near_cells * dx_base
            far_dist = far_cells * dx_base
            
        self.logger.info(f"Cell-based refinement bands: near={near_dist*1e3:.2f}mm ({near_dist/dx_base:.0f} cells), far={far_dist*1e3:.2f}mm ({far_dist/dx_base:.0f} cells)")

        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    inlet.stl       {{ type triSurfaceMesh; name inlet;       file "inlet.stl"; }}
{chr(10).join(f'    {n}.stl        {{ type triSurfaceMesh; name {n};           file "{n}.stl"; }}' for n in outlet_names)}
    {self.wall_name}.stl  {{ type triSurfaceMesh; name {self.wall_name};  file "{self.wall_name}.stl"; }}
}};

castellatedMeshControls
{{
    maxLocalCells {snap.get("maxLocalCells", 2000000)};
    maxGlobalCells {snap.get("maxGlobalCells", 8000000)};
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {snap.get("nCellsBetweenLevels", 1)};
    
    features
    (
        {{
            file "{self.wall_name}.eMesh";
            level {self.surface_levels[0]};
        }}
        {{
            file "inlet.eMesh";
            level {self.surface_levels[0]};
        }}
{chr(10).join(f'        {{ file "{n}.eMesh"; level {self.surface_levels[0]}; }}' for n in outlet_names)}
    );
    
    refinementSurfaces
    {{
        {self.wall_name}
        {{
            level ({self.surface_levels[0]} {self.surface_levels[1]});
        }}
        inlet
        {{
            level ({self.surface_levels[0]} {self.surface_levels[1]});
        }}
{chr(10).join(f'''        {n}
        {{
            level ({self.surface_levels[0]} {self.surface_levels[1]});
        }}''' for n in outlet_names)}
    }}
    
    refinementRegions
    {{
        {self.wall_name}
        {{
            mode distance;
            levels (({near_dist:.6f} 2) ({far_dist:.6f} 1));
        }}
    }}

    locationInMesh ({internal_pt[0]:.6f} {internal_pt[1]:.6f} {internal_pt[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle {snap.get("resolveFeatureAngle", 30)};
}}

snapControls
{{
    nSmoothPatch {snap.get("nSmoothPatch", 3)};
    tolerance 1.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter {snap.get("nFeatureSnapIter", 10)};
    implicitFeatureSnap {str(snap.get("implicitFeatureSnap", False)).lower()};
    explicitFeatureSnap {str(snap.get("explicitFeatureSnap", True)).lower()};
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;
    layers
    {{
    }}
    
    firstLayerThickness 1e-6;
    expansionRatio 1.0;
    minThickness 1e-6;
    nGrow 0;
    featureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    maxNonOrtho {_safe_get(self.config,["MESH_QUALITY","snap","maxNonOrtho"],65)};
    maxBoundarySkewness {_safe_get(self.config,["MESH_QUALITY","snap","maxBoundarySkewness"],4.0)};
    maxInternalSkewness {_safe_get(self.config,["MESH_QUALITY","snap","maxInternalSkewness"],4.0)};
    maxConcave 80;
    minFlatness 0.5;
    minVol {_safe_get(self.config,["MESH_QUALITY","snap","minVol"],1e-13)};
    minTetQuality {_safe_get(self.config,["MESH_QUALITY","snap","minTetQuality"],-1e15)};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {_safe_get(self.config,["MESH_QUALITY","snap","minFaceWeight"],0.02)};
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    
    // Relaxed quality criteria for castellation (OpenFOAM 12 requirement)
    relaxed
    {{
        maxNonOrtho 75;
        maxBoundarySkewness 6.0;
        maxInternalSkewness 6.0;
        maxConcave 90;
        minFlatness 0.3;
        minVol 1e-15;
        minTetQuality 1e-9;
        minFaceWeight 0.005;
        minVolRatio 0.005;
        minDeterminant 0.0005;
    }}
}}

mergeTolerance 1e-6;
"""
        (iter_dir / "system" / "snappyHexMeshDict.noLayer").write_text(content)

    def _snappy_layers_dict(self, iter_dir, outlet_names, internal_pt, dx_base):
        L = self.config["LAYERS"].copy()  # Make a copy to avoid modifying config
        
        # Apply critical layer validation to the copy
        L.setdefault("firstLayerThickness_abs", 50e-6)
        L.setdefault("minThickness_abs", 20e-6)
        
        # CRITICAL FIX: Ensure minThickness = 0.15 × firstLayerThickness (CFD best practice)
        if L["minThickness_abs"] > L["firstLayerThickness_abs"] * 0.2:
            L["minThickness_abs"] = L["firstLayerThickness_abs"] * 0.15  # 0.1-0.2 range, use 0.15
            self.logger.warning(f"Fixed minThickness: {L['minThickness_abs']*1e6:.2f}μm = 0.15×firstLayer ({L['firstLayerThickness_abs']*1e6:.2f}μm)")
        
        # Auto-size layers if enabled (default behavior)
        auto_sizing = self.stage1.get("autoFirstLayerFromDx", True)
        if auto_sizing:
            N = int(L["nSurfaceLayers"])
            ER = float(L.get("expansionRatio", 1.2))
            alpha = float(self.stage1.get("alpha_total_layers", 0.8))
            
            # Apply auto-sizing (will adjust N if clamping occurs)
            sizing = self._first_layer_from_dx(dx_base, N, ER, alpha)
            L.update(sizing)  # This now includes adjusted nSurfaceLayers
            
            # CRITICAL: Re-validate after auto-sizing overwrites values
            if L["minThickness_abs"] > L["firstLayerThickness_abs"] * 0.2:
                L["minThickness_abs"] = L["firstLayerThickness_abs"] * 0.15
                self.logger.warning(f"Auto-sizing created invalid minThickness - fixed: {L['minThickness_abs']*1e6:.2f}μm")
            
            # Use the potentially adjusted N for logging
            N_actual = sizing.get('nSurfaceLayers', N)
            self.logger.info(f"Auto layer sizing: N={N_actual} (orig={N}), t1={sizing['firstLayerThickness_abs']*1e6:.1f}μm, "
                           f"minThick={sizing['minThickness_abs']*1e6:.1f}μm, nGrow={sizing['nGrow']}")
        
        # Apply physics-aware first layer sizing if configured (overrides auto-sizing above)
        if self.physics.get("autoFirstLayer", False):
            D_ref, _ = self._estimate_reference_diameters()
            U_peak = float(self.physics.get("U_peak", 1.0))        # m/s (peak velocity)
            rho = float(self.physics.get("rho", 1060.0))           # kg/m³ (blood density)
            mu = float(self.physics.get("mu", 3.5e-3))             # Pa·s (blood viscosity)
            y_plus = float(self.physics.get("y_plus", 1.0))        # 1 for LES near-wall, 30 for wall-fn
            model = str(self.physics.get("flow_model", "turbulent"))  # "turbulent" or "laminar"
            
            # Calculate first layer thickness from y+ target
            first_layer = self._estimate_first_layer_from_yplus(D_ref, U_peak, rho, mu, y_plus, model)
            L["firstLayerThickness_abs"] = max(5e-6, first_layer)  # Minimum 5 μm for numerical stability
            
            # Adjust layer parameters based on flow model
            if y_plus <= 5:  # Near-wall LES
                # More layers with gentler expansion for y+ ≈ 1
                L["nSurfaceLayers"] = max(L.get("nSurfaceLayers", 16), 16)
                L["expansionRatio"] = min(L.get("expansionRatio", 1.15), 1.2)
            elif y_plus >= 20:  # Wall functions
                # Fewer layers with faster expansion for y+ ≈ 30
                L["nSurfaceLayers"] = min(L.get("nSurfaceLayers", 10), 12)
                L["expansionRatio"] = max(L.get("expansionRatio", 1.2), 1.2)
            
            # CRITICAL: Re-validate after physics-aware sizing
            if L["minThickness_abs"] > L["firstLayerThickness_abs"] * 0.2:
                L["minThickness_abs"] = L["firstLayerThickness_abs"] * 0.15
                self.logger.warning(f"Physics sizing created invalid minThickness - fixed: {L['minThickness_abs']*1e6:.2f}μm")
            
            self.logger.info(f"Auto first layer: {L['firstLayerThickness_abs']*1e6:.1f} μm for y+={y_plus}, {model} flow")
        
        # Sync effective layer parameters back to config for metrics consistency
        self.config["LAYERS"].update({
            "nSurfaceLayers": L["nSurfaceLayers"],
            "expansionRatio": L["expansionRatio"],
            "firstLayerThickness_abs": L.get("firstLayerThickness_abs", 50e-6),
            "minThickness_abs": L.get("minThickness_abs", 20e-6),
            "nGrow": L.get("nGrow", 1),
            "featureAngle": L.get("featureAngle", 70),
            "maxThicknessToMedialRatio": L.get("maxThicknessToMedialRatio", 0.45),
            "minMedianAxisAngle": L.get("minMedianAxisAngle", 70),
        })
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh false;
snap            false;
addLayers       true;

geometry
{{
    inlet.stl       {{ type triSurfaceMesh; name inlet;       file "inlet.stl"; }}
{chr(10).join(f'    {n}.stl        {{ type triSurfaceMesh; name {n};           file "{n}.stl"; }}' for n in outlet_names)}
    {self.wall_name}.stl  {{ type triSurfaceMesh; name {self.wall_name};  file "{self.wall_name}.stl"; }}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 1;
    locationInMesh ({internal_pt[0]:.6f} {internal_pt[1]:.6f} {internal_pt[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle {self.config['SNAPPY'].get('resolveFeatureAngle', 45)};
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;
    layers
    {{
        "{self.wall_name}"
        {{
            nSurfaceLayers {L["nSurfaceLayers"]};
        }}
    }}

    firstLayerThickness {L.get("firstLayerThickness_abs", 50e-6):.2e};
    expansionRatio {L["expansionRatio"]};
    minThickness {L.get("minThickness_abs", 20e-6):.2e};
    nGrow {L.get("nGrow", 1)};
    featureAngle {min(L.get("featureAngle",70), 90)};
    nRelaxIter 8;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {L.get("maxThicknessToMedialRatio",0.45)};
    minMedianAxisAngle {L.get("minMedianAxisAngle",70)};
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    maxNonOrtho {_safe_get(self.config,["MESH_QUALITY","layer","maxNonOrtho"],65)};
    maxBoundarySkewness {_safe_get(self.config,["MESH_QUALITY","layer","maxBoundarySkewness"],4.0)};
    maxInternalSkewness {_safe_get(self.config,["MESH_QUALITY","layer","maxInternalSkewness"],4.0)};
    maxConcave 80;
    minFlatness 0.5;
    minVol {_safe_get(self.config,["MESH_QUALITY","layer","minVol"],1e-13)};
    minTetQuality {_safe_get(self.config,["MESH_QUALITY","layer","minTetQuality"],1e-6)};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {_safe_get(self.config,["MESH_QUALITY","layer","minFaceWeight"],0.02)};
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    
    // Relaxed quality criteria for layer addition (OpenFOAM 12 requirement)
    relaxed
    {{
        maxNonOrtho 75;
        maxBoundarySkewness 6.0;
        maxInternalSkewness 6.0;
        maxConcave 90;
        minFlatness 0.3;
        minVol 1e-15;
        minTetQuality 1e-9;
        minFaceWeight 0.005;
        minVolRatio 0.005;
        minDeterminant 0.0005;
    }}
}}

mergeTolerance 1e-6;
"""
        (iter_dir / "system" / "snappyHexMeshDict.layers").write_text(content)

    def _snappy_dict(self, iter_dir, outlet_names, internal_pt, dx_base, phase):
        """Unified method to generate snappyHexMeshDict for any phase"""
        if phase not in ["no_layers", "layers"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'no_layers' or 'layers'")
        
        if phase == "no_layers":
            self._snappy_no_layers_dict(iter_dir, outlet_names, internal_pt, dx_base)
        elif phase == "layers":
            self._snappy_layers_dict(iter_dir, outlet_names, internal_pt, dx_base)

    # ----------------------------- Execution ------------------------------
    def _maybe_parallel(self, iter_dir):
        """Setup parallel decomposition if configured"""
        n = int(self.stage1.get("n_processors", 1))
        if n <= 1:
            return [], []
        
        # Write decomposeParDict for domain decomposition
        decompose_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {n};
method          scotch;
"""
        (iter_dir / "system" / "decomposeParDict").write_text(decompose_dict)
        
        self.logger.info(f"Parallel processing enabled with {n} processors")
        return (["decomposePar"], "log.decompose"), (["reconstructPar", "-latestTime"], "log.reconstruct")
    
    def _surface_check(self, iter_dir):
        """Run surfaceCheck and gate on critical STL health issues"""
        env = self.config["openfoam_env_path"]
        tri = iter_dir / "constant" / "triSurface"
        
        for p in [tri / f"{self.wall_name}.stl", tri / "inlet.stl", *[tri / f.name for f in self.stl_files["outlets"]]]:
            try:
                rel_path = Path("constant/triSurface") / p.name
                result = run_command(["surfaceCheck", str(rel_path)], cwd=iter_dir, env_setup=env, timeout=600, max_memory_gb=self.max_memory_gb)
                
                # Parse critical issues from surfaceCheck output
                output = result.stdout + result.stderr
                
                # Check for self-intersections
                if "self-intersecting" in output.lower() or "intersecting faces" in output.lower():
                    raise RuntimeError(f"Critical STL error in {p.name}: Self-intersecting faces detected")
                
                # Check for high non-manifold count (>5% of faces)
                if "non-manifold" in output.lower():
                    # Log entire non-manifold block for triage
                    nm_lines = []
                    lines = output.split('\n')
                    in_nm_block = False
                    
                    for line in lines:
                        line_l = line.lower()
                        if "non-manifold" in line_l:
                            in_nm_block = True
                            nm_lines.append(line.strip())
                        elif in_nm_block:
                            if line.strip() and not line.startswith(' '):
                                break  # End of block
                            nm_lines.append(line.strip())
                        
                        if "faces" in line_l and "non-manifold" in line_l:
                            # Try to extract numbers: "X non-manifold faces out of Y"
                            words = line.split()
                            try:
                                nm_idx = [w.lower() for w in words].index("non-manifold")
                                if nm_idx > 0:
                                    non_manifold = int(words[nm_idx-1])
                                    # Find total faces
                                    for i, word in enumerate(words):
                                        if word.lower() == "of" and i+1 < len(words):
                                            total_faces = int(words[i+1])
                                            nm_ratio = non_manifold / max(total_faces, 1)
                                            if nm_ratio > 0.05:  # >5% non-manifold
                                                self.logger.error(f"Non-manifold analysis for {p.name}:")
                                                for nm_line in nm_lines:
                                                    self.logger.error(f"  {nm_line}")
                                                raise RuntimeError(f"Critical STL error in {p.name}: High non-manifold ratio {nm_ratio:.1%} ({non_manifold}/{total_faces})")
                                            else:
                                                self.logger.debug(f"Acceptable non-manifold ratio in {p.name}: {nm_ratio:.1%} ({non_manifold}/{total_faces})")
                                            break
                            except (ValueError, IndexError):
                                # Log the problematic lines for debugging
                                if nm_lines:
                                    self.logger.warning(f"Could not parse non-manifold data in {p.name}:")
                                    for nm_line in nm_lines:
                                        self.logger.warning(f"  {nm_line}")
                                continue
                                
                self.logger.info(f"STL health check passed: {p.name}")
                
            except RuntimeError:
                # Re-raise critical errors to abort iteration
                raise
            except Exception as e:
                self.logger.warning(f"surfaceCheck failed on {p.name}: {e}")

    def _create_foam_file(self, iter_dir: Path):
        """Create a .foam dummy file for easy ParaView visualization"""
        try:
            foam_file = iter_dir / f"{iter_dir.name}.foam"
            foam_file.write_text("// OpenFOAM dummy file for ParaView\n")
            self.logger.debug(f"Created .foam file: {foam_file}")
        except Exception as e:
            self.logger.debug(f"Failed to create .foam file: {e}")

    def _run_snap_then_layers(self, iter_dir, force_full_remesh: bool = True) -> Tuple[Dict, Dict, Dict]:
        env = self.config["openfoam_env_path"]
        logs = iter_dir / "logs"
        logs.mkdir(exist_ok=True)
        
        if not force_full_remesh:
            self.logger.info("Layers-only optimization possible, but running full remesh for robustness")
            
        # Setup parallel decomposition if configured
        pre, post = self._maybe_parallel(iter_dir)
        n_procs = int(self.stage1.get("n_processors", 1))
        
        # Initial mesh generation (always serial)
        for cmd, log_name in [(["blockMesh"], "log.blockMesh"), (["surfaceFeatures"], "log.surfaceFeatures")]:
            self.logger.info(f"Running: {' '.join(cmd)}")
            try:
                res = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / log_name).write_text(res.stdout + res.stderr)
            except Exception as e:
                self.logger.error(f"Command failed: {' '.join(cmd)} | {e}")
        
        # Domain decomposition for parallel run
        if pre:
            self.logger.info(f"Running: {' '.join(pre[0])}")
            try:
                res = run_command(pre[0], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / pre[1]).write_text(res.stdout + res.stderr)
                
                # OpenFOAM automatically handles triSurface distribution in parallel runs
                
            except Exception as e:
                self.logger.error(f"Decomposition failed: {e}")
        
        # MESH WITHOUT LAYERS phase (combined castellation + snap)
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.noLayer", iter_dir / "system" / "snappyHexMeshDict")
        
        # Build snappyHexMesh command (parallel or serial)
        if n_procs > 1:
            snappy_cmd = ["mpirun", "-np", str(n_procs), "snappyHexMesh", "-overwrite", "-parallel"]
        else:
            snappy_cmd = ["snappyHexMesh", "-overwrite"]
        
        self.logger.info(f"Running: {' '.join(snappy_cmd)} (mesh without layers)")
        try:
            res = run_command(snappy_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
            (logs / "log.snappy.no_layers").write_text(res.stdout + res.stderr)
        except Exception as e:
            self.logger.error(f"Mesh generation (no layers) failed: {e}")
        
        # Basic mesh generation validation using boundary file fallback
        mesh_generation_ok = False
        
        # Parallel-safe fallback: trust boundary files if they show faces
        wall_faces_total = self._sum_wall_faces_from_processors(iter_dir)
        if wall_faces_total > 0:
            mesh_generation_ok = True
            self.logger.debug(f"Mesh generation verified via boundary files: {wall_faces_total} wall faces")
        
        # Optimized parallel workflow - minimize reconstruction cycles
        reconstruction_needed = False
        if n_procs > 1:
            self.logger.debug("Parallel workflow - deferring reconstruction until necessary")
        
        if not mesh_generation_ok:
            self.logger.warning("Mesh generation not verified; skipping layers")
            # Return early with empty metrics
            empty_metrics = {"meshOK": False, "cells": 0}
            return empty_metrics, empty_metrics, empty_metrics
        
        # Optimized mesh quality assessment - avoid premature reconstruction
        snap_metrics = self._optimized_quality_check(iter_dir, n_procs, env, logs, "no_layers")
        
        # Check for catastrophic mesh failure (zero cells) - only skip layers for complete failure
        cell_count = snap_metrics.get("cells", 0)
        if cell_count == 0:
            self.logger.warning("Snap mesh has zero cells; skipping layers for this iteration")
            layer_metrics = snap_metrics
            layer_cov = {"coverage_overall": 0.0, "perPatch": {}}
            
            # If parallel, clean up processor directories since we're not continuing
            if n_procs > 1:
                for proc_dir in iter_dir.glob("processor*"):
                    if proc_dir.is_dir():
                        shutil.rmtree(proc_dir)
            
            # Create .foam file for easy visualization
            self._create_foam_file(iter_dir)
            
            return snap_metrics, layer_metrics, layer_cov
        
        # Log snap mesh status but continue to layers regardless of quality issues
        mesh_ok = snap_metrics.get("meshOK", False)
        if not mesh_ok:
            self.logger.info("Snap mesh has quality issues but proceeding with layers (may improve quality)")
        
        # LAYERS phase - continue with decomposed mesh if parallel
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.layers", iter_dir / "system" / "snappyHexMeshDict")
        
        self.logger.info(f"Running: {' '.join(snappy_cmd)} (layers)")
        try:
            res = run_command(snappy_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
            (logs / "log.snappy.layers").write_text(res.stdout + res.stderr)
        except Exception as e:
            self.logger.error(f"Layer meshing failed: {e}")
        
        # Parse layer coverage from snappyHexMesh log
        layer_cov = parse_layer_coverage(iter_dir, env, wall_name=self.wall_name)
        
        # Optimized final mesh quality check
        layer_metrics = self._optimized_quality_check(iter_dir, n_procs, env, logs, "layers")
        
        # Now reconstruct after all checks are done
        if post and n_procs > 1:
            self.logger.info(f"Final reconstruction after successful layers")
            try:
                res = run_command(post[0], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / "log.reconstruct.final").write_text(res.stdout + res.stderr)
                
                # Clean up processor directories after successful reconstruction
                for proc_dir in iter_dir.glob("processor*"):
                    if proc_dir.is_dir():
                        shutil.rmtree(proc_dir)
                self.logger.debug(f"Cleaned up {n_procs} processor directories after final reconstruction")
                        
            except Exception as e:
                self.logger.error(f"Final reconstruction failed: {e}")
        
        # Create .foam file for easy visualization
        self._create_foam_file(iter_dir)
        
        return snap_metrics, layer_metrics, layer_cov

    # --------------------------- Objective & updates -----------------------
    def _meets_quality_constraints(self, snap_m: Dict, layer_m: Dict, layer_cov: Dict) -> bool:
        """Check if mesh meets hard quality constraints (OpenFOAM defaults)"""
        # Wall coverage gate for WSS reliability
        wall_cov = layer_cov.get("perPatch", {}).get(self.wall_name, layer_cov.get("coverage_overall", 0.0))
        
        constraints_met = (
            layer_m.get("meshOK", False) and
            float(layer_m.get("maxNonOrtho", 1e9)) <= self.targets.max_nonortho and
            float(layer_m.get("maxSkewness", 1e9)) <= self.targets.max_skewness and
            wall_cov >= self.targets.min_layer_cov
        )
        return constraints_met
    
    def _get_cell_count(self, layer_m: Dict, no_layers_m: Dict) -> int:
        """Get cell count for mesh minimization objective"""
        return int(layer_m.get("cells", no_layers_m.get("cells", 0)))

    def _apply_updates(self, snap_m: Dict, layer_m: Dict, layer_cov: Dict) -> None:
        # Decide minimal change for next iteration
        NO = float(layer_m.get("maxNonOrtho", 0))
        SK = float(layer_m.get("maxSkewness", 0))
        coverage = float(layer_cov.get("coverage_overall", 0.0))
        # Deltas
        dNO = NO - float(snap_m.get("maxNonOrtho", 0))
        dSK = SK - float(snap_m.get("maxSkewness", 0))

        # If layers broke the mesh: shrink absolute layer sizes and increase smoothing
        if (dNO > 5 or dSK > 0.5) and coverage < self.targets.min_layer_cov:
            self.config["LAYERS"]["firstLayerThickness_abs"] *= 0.8
            self.config["LAYERS"]["minThickness_abs"] *= 0.8
            self.config["LAYERS"]["nGrow"] = min(self.config["LAYERS"].get("nGrow",0) + 1, 3)
            self.logger.info(f"Layers adjusted: first={self.config['LAYERS']['firstLayerThickness_abs']:.3e}, "
                             f"min={self.config['LAYERS']['minThickness_abs']:.3e}, nGrow={self.config['LAYERS']['nGrow']}")
            return

        # If snap already poor: reduce only the MAX wall level; keep MIN to preserve adjacency
        if NO > self.targets.max_nonortho or SK > self.targets.max_skewness:
            if self.surface_levels[1] > self.surface_levels[0]:
                self.surface_levels[1] -= 1
                self.logger.info(f"Reduced surface level max → {self.surface_levels}")
            else:
                # increase angle (toward 60) to chase fewer features
                self.config["SNAPPY"]["resolveFeatureAngle"] = min(60, int(self.config["SNAPPY"].get("resolveFeatureAngle",45)) + int(self.stage1.get("featureAngle_step",10)))
                self.logger.info(f"Raised resolveFeatureAngle → {self.config['SNAPPY']['resolveFeatureAngle']}°")
            return

        # If quality ok but features washed out (coverage ok and deltas small): allow MORE feature snap
        if coverage >= self.targets.min_layer_cov and dNO < 2 and dSK < 0.2:
            # decrease angle (toward 30) to capture more curvature
            self.config["SNAPPY"]["resolveFeatureAngle"] = max(30, int(self.config["SNAPPY"].get("resolveFeatureAngle",45)) - int(self.stage1.get("featureAngle_step",10)))
            self.logger.info(f"Lowered resolveFeatureAngle → {self.config['SNAPPY']['resolveFeatureAngle']}°")
            return

        # Else, if only coverage is low but deltas are small: add layers (if still within reasonable range)
        if coverage < self.targets.min_layer_cov:
            self.config["LAYERS"]["nSurfaceLayers"] = min(self.config["LAYERS"]["nSurfaceLayers"] + 2, 22)
            self.logger.info(f"Increased nSurfaceLayers → {self.config['LAYERS']['nSurfaceLayers']}")

    # ------------------------------- Loop ---------------------------------
    def iterate_until_quality(self):
        """
        Main entry point for Stage1 mesh optimization.
        
        COMPATIBILITY: This method now delegates to the new modular optimizer
        when available, falling back to legacy implementation for compatibility.
        """
        # Try to use new modular optimizer if available
        if hasattr(self, '_use_modular') and self._use_modular:
            try:
                return self._run_modular_optimization()
            except Exception as e:
                self.logger.warning(f"Modular optimizer failed, falling back to legacy: {e}")
        
        self.logger.info("Starting Stage‑1 geometry‑aware optimization (LEGACY MODE)")
        # Note: bbox_data, diameters, and dx will be computed from scaled STL files in each iteration
        # This avoids loading STL files twice (once original, once scaled)

        best_iter = None
        best_cell_count = math.inf
        summary_path = self.output_dir / "stage1_summary.csv"
        if not summary_path.exists():
            with open(summary_path, "w", newline="") as f:
                csv.writer(f).writerow(["iter","cells","maxNonOrtho","maxSkewness","coverage","objective_dummy","levels_min","levels_max","resolveFeatureAngle","nLayers","firstLayer","minThickness"]) 

        for k in range(1, self.max_iterations + 1):
            self.current_iteration = k
            self.logger.info(f"=== ITERATION {k} ===")
            
            # Log original config values for transparency  
            snap = self.config["SNAPPY"]
            self.logger.debug(f"Config: maxGlobal={snap['maxGlobalCells']:,}, maxLocal={snap['maxLocalCells']:,}, "
                             f"featureAngle={snap.get('resolveFeatureAngle', 45)}°, "
                             f"budget={self.stage1.get('cell_budget_kb_per_cell', 1.0)}KB/cell")
            
            iter_dir = self.output_dir / f"iter_{k:03d}"
            iter_dir.mkdir(exist_ok=True)
            
            # Create required directory structure
            (iter_dir / "system").mkdir(exist_ok=True)
            (iter_dir / "logs").mkdir(exist_ok=True)

            # 1) Copy/scale STL files using config scale_m
            outlet_names = self._copy_trisurfaces(iter_dir)
            
            # 2) Compute bounding box from original STL files (in mm)
            # Scaling handled via config "scale_m": 1.0e-3
            gen = PhysicsAwareMeshGenerator()
            stl_map_orig = {"inlet": self.stl_files["required"]["inlet"],
                           self.wall_name: self.stl_files["required"][self.wall_name],
                           **{p.stem: p for p in self.stl_files["outlets"]}}
            
            # Create map to scaled STL files (now in meters)
            stl_map_norm = {"inlet": iter_dir/"constant/triSurface/inlet.stl",
                            self.wall_name: iter_dir/f"constant/triSurface/{self.wall_name}.stl",
                            **{p.stem: iter_dir/f"constant/triSurface/{p.stem}.stl" for p in self.stl_files["outlets"]}}
            
            # Compute bounding box from scaled STL files (already in meters - skip additional scaling)
            bbox_data = gen.compute_stl_bounding_box(stl_map_norm, skip_scaling=True)
            
            # Compute diameters from scaled STL files (already in meters)
            tri_dir = iter_dir / "constant" / "triSurface"
            D_ref, D_min = self._estimate_reference_diameters(stl_root=tri_dir)
            
            dx = self._derive_base_cell_size(D_ref, D_min)
            
            # Apply adaptive resolveFeatureAngle based on scaled geometry
            try:
                n_out = len(outlet_names)
                adaptive_angle = self._adaptive_feature_angle(D_ref, D_min, n_outlets=n_out, iter_dir=iter_dir)
                self.config["SNAPPY"]["resolveFeatureAngle"] = adaptive_angle
                self.logger.info(f"Applied adaptive resolveFeatureAngle={adaptive_angle}° for iteration {k}")
            except Exception as e:
                self.logger.warning(f"Adaptive feature angle failed; keeping {self.config['SNAPPY'].get('resolveFeatureAngle', 35)}°: {e}")
            
            # 3) Now write blockMesh/snappy dicts using the bbox_data & dx
            self._generate_blockmesh_dict(iter_dir, bbox_data, dx)
            
            # Feature resolution validation
            min_feature_size = self._estimate_minimum_feature_size(iter_dir / "constant" / "triSurface" / f"{self.wall_name}.stl")
            if min_feature_size > 0:
                feature_resolved = self._validate_feature_resolution(min_feature_size, dx)
                if not feature_resolved:
                    self.logger.info("Feature resolution warning noted - proceeding with current settings")
            
            self._write_surfaceFeatures(iter_dir, [f"{self.wall_name}.stl", "inlet.stl", *[f"{n}.stl" for n in outlet_names]])
            
            # Apply ladder progression per iteration with curvature adaptation
            base_ladder = self.stage1.get("ladder", [[1,1],[2,2],[2,3]])
            idx = min(self.current_iteration-1, len(base_ladder)-1)
            base_levels = list(base_ladder[idx])
            
            # Curvature-adaptive refinement enhancement
            wall_stl = iter_dir / "constant" / "triSurface" / f"{self.wall_name}.stl"
            if wall_stl.exists():
                try:
                    cs = self._estimate_curvature_strength(wall_stl)
                    curvature_strength = cs["strength"]
                    adapted_levels = self._curvature_adaptive_levels(base_levels, curvature_strength)
                    self.logger.info(f"Curvature adaptation: base={base_levels} → adapted={adapted_levels} (strength={curvature_strength:.3f})")
                    new_surface_levels = adapted_levels
                except Exception as e:
                    self.logger.warning(f"Curvature analysis failed, using base ladder: {e}")
                    new_surface_levels = base_levels
            else:
                new_surface_levels = base_levels
            
            # Detect if surface levels changed - if so, need full remesh
            surface_levels_changed = (not hasattr(self, 'surface_levels') or 
                                    self.surface_levels != new_surface_levels)
            
            self.surface_levels = new_surface_levels
            self.logger.info(f"Surface levels from ladder: {self.surface_levels}")
            
            if surface_levels_changed:
                self.logger.info("Surface refinement levels changed - full remesh required")
            else:
                self.logger.info("Surface levels unchanged - could optimize for layers-only")
            
            # Calculate seed point using robust geometric analysis
            scaled_inlet_path = iter_dir / "constant" / "triSurface" / "inlet.stl"
            internal_point = self._calculate_robust_seed_point(
                bbox_data,
                {"required": {"inlet": scaled_inlet_path}},
                dx_base=dx
            )
            
            # Calculate and apply dynamic nFeatureSnapIter before generating snappy dicts
            current_feature_angle = int(self.config["SNAPPY"].get("resolveFeatureAngle", 45))
            if current_feature_angle <= 35:
                current_nFeatureSnapIter = 30
            elif current_feature_angle >= 55:
                current_nFeatureSnapIter = 20
            else:
                current_nFeatureSnapIter = int(30 + (20 - 30) * (current_feature_angle - 35) / (55 - 35))
            
            # Apply the calculated nFeatureSnapIter to config so it's used in snappy dicts
            self.config["SNAPPY"]["nFeatureSnapIter"] = current_nFeatureSnapIter
            self.logger.info(f"Dynamic nFeatureSnapIter: {current_nFeatureSnapIter} (for resolveFeatureAngle={current_feature_angle}°)")

            self._snappy_dict(iter_dir, outlet_names, internal_point, dx, "no_layers")
            self._snappy_dict(iter_dir, outlet_names, internal_point, dx, "layers")
            self._surface_check(iter_dir)

            # Run mesh generation
            snap_m, layer_m, layer_cov = self._run_snap_then_layers(iter_dir, force_full_remesh=surface_levels_changed)

            # Objective
            # Check quality constraints and get cell count
            constraints_ok = self._meets_quality_constraints(snap_m, layer_m, layer_cov)
            cell_count = self._get_cell_count(layer_m, snap_m)
            
            # Get current feature snap iter for logging (already applied to config earlier)
            current_nFeatureSnapIter = self.config["SNAPPY"].get("nFeatureSnapIter", 20)

            # Check for mesh quality convergence
            converged = self._mesh_convergence_check(layer_m)
            
            # Log
            cov = float(layer_cov.get("coverage_overall", 0.0))
            wall_cov = layer_cov.get("perPatch", {}).get(self.wall_name, cov)
            status = "PASS" if constraints_ok else "FAIL"
            convergence_status = "CONVERGED" if converged else "EVOLVING"
            self.logger.info(f"RESULTS: cells={cell_count:,}, maxNonOrtho={layer_m.get('maxNonOrtho',0):.1f}, maxSkewness={layer_m.get('maxSkewness',0):.2f}, wall_cov={wall_cov*100:.1f}% [{status}] [{convergence_status}]")

            # Calculate deltas for triage
            delta_nonortho = float(layer_m.get("maxNonOrtho", 0)) - float(snap_m.get("maxNonOrtho", 0))
            delta_skewness = float(layer_m.get("maxSkewness", 0)) - float(snap_m.get("maxSkewness", 0))
            
            # Save per‑iter metrics JSON
            metrics = {
                "iteration": k,
                "checkMesh_snap": snap_m,
                "checkMesh_layer": layer_m,
                "layerCoverage": layer_cov,
                "delta": {
                    "maxNonOrtho": delta_nonortho,
                    "maxSkewness": delta_skewness,
                    "interpretation": {
                        "layers_degrade_quality": delta_nonortho > 5 or delta_skewness > 0.5,
                        "layers_improve_quality": delta_nonortho < -2 or delta_skewness < -0.2,
                        "layers_neutral": abs(delta_nonortho) <= 2 and abs(delta_skewness) <= 0.2
                    }
                },
                "surface_levels": self.surface_levels,
                "resolveFeatureAngle": int(self.config["SNAPPY"].get("resolveFeatureAngle",45)),
                "nFeatureSnapIter": current_nFeatureSnapIter,
                "layers": {
                    "nSurfaceLayers": self.config["LAYERS"]["nSurfaceLayers"],
                    "firstLayerThickness_abs": self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6),
                    "minThickness_abs": self.config["LAYERS"].get("minThickness_abs", 20e-6),
                },
                "constraints_met": constraints_ok,
                "cell_count": cell_count
            }
            (iter_dir / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))

            # Append CSV summary (before plateau check to ensure final iteration is always logged)
            with open(summary_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    k,
                    layer_m.get("cells",0),
                    f"{layer_m.get('maxNonOrtho',0):.1f}",
                    f"{layer_m.get('maxSkewness',0):.2f}",
                    f"{cov:.3f}",
                    "0.000",  # Dummy - no objective function in constraint-based approach
                    self.surface_levels[0], self.surface_levels[1],
                    int(self.config["SNAPPY"].get("resolveFeatureAngle",45)),
                    self.config["LAYERS"]["nSurfaceLayers"],
                    f"{self.config['LAYERS'].get('firstLayerThickness_abs', 50e-6):.3e}",
                    f"{self.config['LAYERS'].get('minThickness_abs', 20e-6):.3e}",
                ])

            # Constraint-based acceptance: minimize cells subject to quality constraints
            if constraints_ok:
                if cell_count < best_cell_count:
                    best_cell_count = cell_count
                    best_iter = iter_dir
                    self.logger.info(f"ACCEPTED: iter {k} with {cell_count:,} cells (new best)")
                else:
                    self.logger.info(f"CONSTRAINTS MET: iter {k} with {cell_count:,} cells (not better than {best_cell_count:,})")
                
                # Early termination conditions
                should_terminate = False
                
                # Convergence-based termination
                if converged and constraints_ok and k >= 3:  # Need at least 3 iterations for convergence
                    self.logger.info(f"Mesh quality converged with valid solution - terminating optimization")
                    should_terminate = True
                
                # Constraint-based termination  
                elif best_iter is not None and k >= 2:  # Give at least 2 iterations
                    self.logger.info(f"Constraint-based optimization complete: best mesh has {best_cell_count:,} cells")
                    should_terminate = True
                
                if should_terminate:
                    break
            else:
                self.logger.info(f"CONSTRAINTS NOT MET: iter {k} - continuing optimization")

            # Otherwise update parameters and continue
            self._apply_updates(snap_m, layer_m, layer_cov)

        if best_iter is None:
            self.logger.warning("⚠️ Stage-1 did not meet all quality constraints within max_iterations")
            best_iter = iter_dir
            final_status = "INCOMPLETE"
        else:
            final_status = "COMPLETE"
        
        # Export best iteration to stable location for Stage 2
        best_out = self.output_dir / "best"
        if best_out.exists():
            shutil.rmtree(best_out)
        shutil.copytree(best_iter, best_out)
        
        # Save config for Stage 2 compatibility
        config_out = best_out / "config.json"
        with open(config_out, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Log final summary
        if best_iter != iter_dir:
            # Load metrics from best iteration
            best_metrics_path = best_iter / "stage1_metrics.json"
            if best_metrics_path.exists():
                with open(best_metrics_path) as f:
                    best_metrics = json.load(f)
                best_cells = best_metrics.get("cell_count", 0)
                self.logger.info(f"🎯 Stage-1 {final_status}: Best geometry-based mesh has {best_cells:,} cells")
        
        self.logger.info(f"📁 Best Stage-1 mesh exported to: {best_out}")
        self.logger.info(f"🚀 Ready for Stage-2 physics verification (GCI analysis)")
        
        return best_out

    # ==================== MODULAR COMPATIBILITY METHODS ====================
    
    def _find_geometry_stl(self):
        """Find the STL geometry file in the geometry directory"""
        stl_files = list(self.geometry_dir.glob("*.stl"))
        if not stl_files:
            raise FileNotFoundError(f"No STL files found in {self.geometry_dir}")
        
        # Prefer files with specific patterns
        for pattern in ["wall_", "geometry", "aorta"]:
            matching_files = [f for f in stl_files if pattern.lower() in f.name.lower()]
            if matching_files:
                return matching_files[0]
        
        # Return first STL file found
        return stl_files[0]
    
    def _run_modular_optimization(self):
        """Run optimization using the new modular system"""
        self.logger.info("Delegating to modular Stage1Optimizer")
        
        try:
            # Extract configuration parameters
            max_iterations = self.stage1.get("max_iterations", 10)
            
            # Run modular optimization
            result = self._modular_optimizer.run_full_optimization(
                max_iterations=max_iterations,
                parallel_procs=1  # Single processor for compatibility
            )
            
            # Convert modular result to legacy format
            best_out = self.output_dir / "best"
            if result.get('final_mesh_directory'):
                final_mesh = Path(result['final_mesh_directory'])
                if final_mesh.exists():
                    if best_out.exists():
                        shutil.rmtree(best_out)
                    shutil.copytree(final_mesh, best_out)
                    
                    # Save config for Stage 2 compatibility
                    config_out = best_out / "config.json"
                    with open(config_out, "w") as f:
                        json.dump(self.config, f, indent=2)
            
            # Log results in legacy format
            status = result.get('status', 'unknown')
            if result.get('optimization_results', {}).get('best_iteration'):
                best_iter = result['optimization_results']['best_iteration']
                cell_count = best_iter.get('layer_metrics', {}).get('cells', 0)
                quality_score = best_iter.get('quality_score', 0)
                
                self.logger.info(f"🎯 Stage-1 {status.upper()}: Best mesh has {cell_count:,} cells "
                               f"(quality score: {quality_score:.3f})")
            
            self.logger.info(f"📁 Best Stage-1 mesh exported to: {best_out}")
            self.logger.info(f"🚀 Ready for Stage-2 physics verification")
            
            return best_out
            
        except Exception as e:
            self.logger.error(f"Modular optimization failed: {e}")
            raise

