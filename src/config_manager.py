"""
Configuration management for mesh optimization parameters.

Handles loading, validation, and dynamic adjustment of mesh generation
parameters with robust defaults for cardiovascular geometries.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigManager:
    """Manage mesh optimization configuration with dynamic adjustments."""
    
    def __init__(self, config_file: Optional[Path] = None, logger=None):
        self.logger = logger
        self.config = self._load_config(config_file)
        self._validate_config()
    
    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if self.logger:
                    self.logger.info(f"Loaded configuration from {config_file}")
                return config
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for arterial geometries."""
        return {
            "BLOCKMESH_SETTINGS": {
                "resolution": 40,
                "resolution_range": [20, 80],
                "grading": [1, 1, 1],
                "domain_expansion": 0.2
            },
            "GEOMETRY": {
                "feature_angle_deg": 150,
                "feature_angle_range": [120, 180]
            },
            "SNAPPY_UNIFORM": {
                "surface_level": [1, 2],
                "surface_level_max": [3, 4],
                "nCellsBetweenLevels": 2,
                "maxGlobalCells": 2e7,
                "maxLocalCells": 5e6,
                "minRefineCellLevel": 0,
                "resolveFeatureAngle": 150
            },
            "LAYERS": {
                "nSurfaceLayers": 7,
                "nSurfaceLayers_range": [5, 12],
                "expansionRatio": 1.2,
                "expansionRatio_range": [1.15, 1.3],
                "finalLayerThickness_rel": 0.15,
                "finalLayerThickness_range": [0.1, 0.4],
                "minThickness_rel": 0.05,
                "featureAngle": 140,
                "featureAngle_range": [110, 160],
                "maxThicknessToMedialRatio": 0.3,
                "minMedianAxisAngle": 90,
                "nLayerIter": 50,
                "nRelaxedIter": 20
            },
            "QUALITY_CRITERIA": {
                "maxNonOrtho": 65,
                "maxSkewness": 4.0,
                "maxAspectRatio": 1000,
                "minVolRatio": 0.01,
                "negVolCells": 0
            },
            "YPLUS_CRITERIA": {
                "target_band": [1.0, 5.0],
                "required_coverage": 0.9,
                "acceptable_coverage": 0.7
            },
            "REFINEMENT_STRATEGY": {
                "max_iterations": 10,
                "progressive_refinement": True,
                "surface_failure_recovery": True,
                "adaptive_layer_params": True
            },
            "SOLVER_SETTINGS": {
                "enable_warmup": False,
                "warmup_timesteps": 200,
                "convergence_tolerance": {
                    "p": 1e-4,
                    "U": 1e-5,
                    "k": 1e-5,
                    "omega": 1e-5
                }
            },
            "PARALLEL": {
                "auto_detect": True,
                "cell_threshold_for_parallel": 1000000,
                "max_processors": 8
            },
            "OPENFOAM": {
                "env_path": "/opt/openfoam12/etc/bashrc",
                "version": "12"
            }
        }
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = [
            "BLOCKMESH_SETTINGS", "SNAPPY_UNIFORM", "LAYERS", 
            "QUALITY_CRITERIA", "REFINEMENT_STRATEGY"
        ]
        
        for section in required_sections:
            if section not in self.config:
                if self.logger:
                    self.logger.warning(f"Missing config section {section}, using defaults")
                defaults = self._get_default_config()
                self.config[section] = defaults[section]
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., "LAYERS.expansionRatio")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        target = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # Set final value
        target[keys[-1]] = value
        
        if self.logger:
            self.logger.info(f"Updated config: {key_path} = {value}")
    
    def adjust_surface_refinement(self, change: int) -> None:
        """
        Adjust surface refinement levels.
        
        Args:
            change: Change in refinement level (+1, -1, etc.)
        """
        current = self.get("SNAPPY_UNIFORM.surface_level", [1, 2])
        max_level = self.get("SNAPPY_UNIFORM.surface_level_max", [3, 4])
        
        if isinstance(current, list) and len(current) == 2:
            new_level = [
                max(1, min(current[0] + change, max_level[0])),
                max(2, min(current[1] + change, max_level[1]))
            ]
            self.set("SNAPPY_UNIFORM.surface_level", new_level)
    
    def adjust_blockmesh_resolution(self, change: int) -> None:
        """
        Adjust blockMesh resolution.
        
        Args:
            change: Change in resolution
        """
        current = self.get("BLOCKMESH_SETTINGS.resolution", 40)
        resolution_range = self.get("BLOCKMESH_SETTINGS.resolution_range", [20, 80])
        
        new_resolution = max(resolution_range[0], min(current + change, resolution_range[1]))
        self.set("BLOCKMESH_SETTINGS.resolution", new_resolution)
    
    def adjust_layer_parameters(self, adjustments: Dict[str, Any]) -> None:
        """
        Adjust layer generation parameters.
        
        Args:
            adjustments: Dictionary of parameter adjustments
        """
        for param, value in adjustments.items():
            # Check if value is within acceptable range
            range_key = f"LAYERS.{param}_range"
            param_range = self.get(range_key)
            
            if param_range and isinstance(param_range, list) and len(param_range) == 2:
                value = max(param_range[0], min(value, param_range[1]))
            
            self.set(f"LAYERS.{param}", value)
    
    def get_iteration_config(self, iteration: int) -> Dict[str, Any]:
        """
        Get configuration adjusted for specific iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Configuration dictionary for this iteration
        """
        config = copy.deepcopy(self.config)
        
        # Progressive refinement strategy
        if self.get("REFINEMENT_STRATEGY.progressive_refinement", True):
            base_level = self.get("SNAPPY_UNIFORM.surface_level", [1, 2])
            if iteration > 1 and isinstance(base_level, list):
                # Gradually increase refinement
                max_level = self.get("SNAPPY_UNIFORM.surface_level_max", [3, 4])
                progress = min((iteration - 1) * 0.5, 2)  # Increase by 0.5 per iteration, max +2
                
                new_level = [
                    min(base_level[0] + int(progress), max_level[0]),
                    min(base_level[1] + int(progress), max_level[1])
                ]
                config["SNAPPY_UNIFORM"]["surface_level"] = new_level
        
        return config
    
    def save_config(self, output_file: Path) -> None:
        """Save current configuration to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            if self.logger:
                self.logger.info(f"Saved configuration to {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save config: {e}")
    
    def create_layer_config(self) -> Dict[str, Any]:
        """Create layer configuration dictionary for mesh generation."""
        return {
            'nSurfaceLayers': self.get("LAYERS.nSurfaceLayers", 7),
            'expansionRatio': self.get("LAYERS.expansionRatio", 1.2),
            'finalLayerThickness': self.get("LAYERS.finalLayerThickness_rel", 0.15),
            'minThickness': self.get("LAYERS.minThickness_rel", 0.05),
            'featureAngle': self.get("LAYERS.featureAngle", 140),
            'maxThicknessToMedialRatio': self.get("LAYERS.maxThicknessToMedialRatio", 0.3),
            'minMedianAxisAngle': self.get("LAYERS.minMedianAxisAngle", 90),
            'nLayerIter': self.get("LAYERS.nLayerIter", 50),
            'nRelaxedIter': self.get("LAYERS.nRelaxedIter", 20),
            'maxFaceThicknessRatio': self.get("LAYERS.maxFaceThicknessRatio", 0.5),
            'nGrow': self.get("LAYERS.nGrow", 1)
        }
    
    def calculate_mesh_resolution(self, extent: float) -> list:
        """
        Calculate blockMesh resolution based on geometry extent.
        
        Args:
            extent: Maximum geometry dimension
            
        Returns:
            [nx, ny, nz] resolution list
        """
        base_resolution = self.get("BLOCKMESH_SETTINGS.resolution", 40)
        
        # Maintain aspect ratio while ensuring minimum resolution
        min_cells = 12
        max_cells = 80
        
        # Simple isotropic resolution for robustness
        resolution = max(min_cells, min(base_resolution, max_cells))
        
        return [resolution, resolution, resolution]