"""
Configuration management for Stage1 mesh optimization.
Handles loading, validation, and normalization of all configuration parameters.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .constants import DEFAULT_CONSTANTS, MeshQualityLimits, LayerParams

logger = logging.getLogger(__name__)

@dataclass
class Stage1Targets:
    """Quality acceptance criteria for optimization"""
    max_nonortho: float
    max_skewness: float
    min_layer_cov: float

class ConfigManager:
    """Centralized configuration management with validation"""
    
    def __init__(self, config_file: Path):
        self.config_file = Path(config_file)
        self.config = self._load_and_validate_config()
        self.targets = self._create_targets()
        
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration from file"""
        try:
            with open(self.config_file) as f:
                config = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_file}: {e}")
            
        # Apply backward compatibility conversions
        config = self._apply_backward_compatibility(config)
        
        # Validate critical sections
        self._validate_config_structure(config)
        self._validate_layers_config(config)
        self._validate_physics_config(config)
        
        return config
    
    def _apply_backward_compatibility(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old config format to new standardized format"""
        # Convert simplified format to full format
        if "mesh" in config:
            mesh = config["mesh"]
            if "acceptance_criteria" not in config:
                config["acceptance_criteria"] = {}
            
            # Map mesh parameters
            if "base_size_mode" in mesh:
                config.setdefault("STAGE1", {})["base_size_mode"] = mesh["base_size_mode"]
            if "cells_per_diameter" in mesh:
                config["STAGE1"]["N_D"] = mesh["cells_per_diameter"]
        
        # Convert accept -> acceptance_criteria
        if "accept" in config:
            accept = config["accept"]
            config.setdefault("acceptance_criteria", {})
            config["acceptance_criteria"].update({
                "maxNonOrtho": accept.get("maxNonOrtho", 65),
                "maxSkewness": accept.get("maxSkewness", 4.0),
                "min_layer_coverage": accept.get("min_layer_coverage", 0.70)
            })
            
        return config
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """Validate that all required config sections exist"""
        required_sections = ["LAYERS", "SNAPPY", "acceptance_criteria"]
        for section in required_sections:
            if section not in config:
                config[section] = {}
                logger.warning(f"Missing config section '{section}' - using defaults")
    
    def _validate_layers_config(self, config: Dict[str, Any]) -> None:
        """Validate and normalize LAYERS configuration"""
        L = config["LAYERS"]
        constants = DEFAULT_CONSTANTS['layers']
        
        # Set critical defaults
        L.setdefault("nSurfaceLayers", 10)
        L.setdefault("expansionRatio", 1.2)
        L.setdefault("firstLayerThickness_abs", 50e-6)
        L.setdefault("minThickness_abs", 5e-6)  # Fixed default
        L.setdefault("nGrow", constants.N_GROW_DEFAULT)
        L.setdefault("featureAngle", constants.FEATURE_ANGLE_DEFAULT)
        L.setdefault("maxThicknessToMedialRatio", constants.MAX_THICKNESS_TO_MEDIAL_RATIO)
        L.setdefault("minMedianAxisAngle", constants.MIN_MEDIAN_AXIS_ANGLE)
        
        # Validate ranges
        if not (3 <= L["nSurfaceLayers"] <= 25):
            L["nSurfaceLayers"] = max(3, min(25, L["nSurfaceLayers"]))
            
        if not (1.1 <= L["expansionRatio"] <= 2.0):
            L["expansionRatio"] = max(1.1, min(2.0, L["expansionRatio"]))
            
        # CRITICAL: Ensure minThickness <= 0.2 × firstLayerThickness
        first_layer = L["firstLayerThickness_abs"]
        if L["minThickness_abs"] > first_layer * constants.MAX_THICKNESS_RATIO:
            L["minThickness_abs"] = first_layer * constants.MIN_THICKNESS_RATIO
            logger.warning(f"Fixed minThickness: {L['minThickness_abs']*1e6:.2f}μm "
                         f"(was > 0.2×firstLayer {first_layer*1e6:.2f}μm)")
    
    def _validate_physics_config(self, config: Dict[str, Any]) -> None:
        """Validate physics-aware configuration"""
        if "physics" in config:
            phys = config["physics"]
            constants = DEFAULT_CONSTANTS['physics']
            
            # Normalize solver mode
            solver_mode = phys.get("solver_mode", "RANS").upper()
            if solver_mode not in ["LES", "RANS", "LAMINAR"]:
                solver_mode = "RANS"
                logger.warning(f"Invalid solver_mode, defaulting to RANS")
            phys["solver_mode"] = solver_mode
            
            # Set physics defaults
            phys.setdefault("U_peak", 1.0)
            phys.setdefault("rho", constants.BLOOD_DENSITY_DEFAULT)
            phys.setdefault("mu", constants.BLOOD_VISCOSITY_DEFAULT)
            phys.setdefault("y_plus", constants.Y_PLUS_RANS if solver_mode == "RANS" else constants.Y_PLUS_LES)
            phys.setdefault("heart_rate_hz", constants.HEART_RATE_DEFAULT_HZ)
    
    def _create_targets(self) -> Stage1Targets:
        """Create quality targets with solver-specific presets"""
        accept = self.config.get("acceptance_criteria", {})
        base_targets = Stage1Targets(
            max_nonortho=float(accept.get("maxNonOrtho", 65)),
            max_skewness=float(accept.get("maxSkewness", 4.0)),
            min_layer_cov=float(accept.get("min_layer_coverage", 0.70))
        )
        
        # Apply solver-specific presets
        solver_mode = self.config.get("physics", {}).get("solver_mode", "").upper()
        quality_limits = DEFAULT_CONSTANTS['mesh_quality']
        
        if solver_mode == "LES":
            base_targets.max_nonortho = min(base_targets.max_nonortho, quality_limits.LES_MAX_NON_ORTHO)
            base_targets.max_skewness = min(base_targets.max_skewness, quality_limits.LES_MAX_SKEWNESS)
            base_targets.min_layer_cov = max(base_targets.min_layer_cov, quality_limits.LES_MIN_COVERAGE)
            logger.info(f"Applied LES quality presets: maxNonOrtho≤{base_targets.max_nonortho}")
            
        elif solver_mode == "RANS":
            base_targets.max_nonortho = min(base_targets.max_nonortho, quality_limits.RANS_MAX_NON_ORTHO)
            base_targets.max_skewness = min(base_targets.max_skewness, quality_limits.RANS_MAX_SKEWNESS)
            base_targets.min_layer_cov = max(base_targets.min_layer_cov, quality_limits.RANS_MIN_COVERAGE)
            logger.info(f"Applied RANS quality presets: maxNonOrtho≤{base_targets.max_nonortho}")
        
        return base_targets
    
    def get_openfoam_env(self) -> str:
        """Get OpenFOAM environment setup command"""
        return self.config.get("openfoam_env_path", "source /opt/openfoam12/etc/bashrc")
    
    def get_processor_count(self) -> int:
        """Get number of processors for parallel execution"""
        return int(self.config.get("STAGE1", {}).get("n_processors", 1))
    
    def get_max_iterations(self) -> int:
        """Get maximum optimization iterations"""
        return int(self.config.get("STAGE1", {}).get("max_iterations", 
                  DEFAULT_CONSTANTS['convergence'].MAX_ITERATIONS_DEFAULT))
    
    def update_layer_thickness(self, first_layer: float, min_thickness: float) -> None:
        """Update layer thickness values with validation"""
        constants = DEFAULT_CONSTANTS['layers']
        
        # Validate and fix if needed
        if min_thickness > first_layer * constants.MAX_THICKNESS_RATIO:
            min_thickness = first_layer * constants.MIN_THICKNESS_RATIO
            logger.warning(f"Auto-corrected minThickness to {min_thickness*1e6:.2f}μm")
        
        self.config["LAYERS"]["firstLayerThickness_abs"] = first_layer
        self.config["LAYERS"]["minThickness_abs"] = min_thickness
    
    def export_for_stage2(self, output_path: Path) -> None:
        """Export configuration for Stage 2 compatibility"""
        stage2_config = self.config.copy()
        
        # Add Stage 1 completion metadata
        stage2_config["stage1_completed"] = True
        stage2_config["stage1_targets_met"] = {
            "maxNonOrtho": self.targets.max_nonortho,
            "maxSkewness": self.targets.max_skewness,
            "min_layer_coverage": self.targets.min_layer_cov
        }
        
        with open(output_path, "w") as f:
            json.dump(stage2_config, f, indent=2)