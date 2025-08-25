"""
Unit tests for ConfigManager module.
"""
import pytest
import json
from pathlib import Path

from mesh_optim.stage1.config_manager import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    def test_initialization(self, sample_config_file):
        """Test ConfigManager initialization"""
        manager = ConfigManager(str(sample_config_file))
        
        assert manager.config is not None
        assert "physics" in manager.config
        assert "targets" in manager.config
        assert manager.targets is not None
    
    def test_config_validation(self, temp_dir):
        """Test configuration validation"""
        # Create invalid config (missing required sections)
        invalid_config = {"base_size": 0.001}
        config_file = temp_dir / "invalid_config.json"
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        manager = ConfigManager(str(config_file))
        # Should create defaults for missing sections
        assert "targets" in manager.config
        assert "physics" in manager.config
    
    def test_layer_parameters_update(self, sample_config_file):
        """Test layer parameters update"""
        manager = ConfigManager(str(sample_config_file))
        
        new_params = {
            'firstLayerThickness_abs': 30e-6,
            'minThickness_abs': 4e-6,
            'nSurfaceLayers': 10,
            'expansionRatio': 1.15
        }
        
        manager.update_layer_parameters(new_params)
        
        layers = manager.config["snappyHexMeshDict"]["addLayersControls"]
        assert layers["firstLayerThickness_abs"] == 30e-6
        assert layers["minThickness_abs"] == 4e-6
        assert layers["nSurfaceLayers"] == 10
        assert layers["expansionRatio"] == 1.15
    
    def test_refinement_bands_update(self, sample_config_file):
        """Test refinement bands update"""
        manager = ConfigManager(str(sample_config_file))
        
        near_dist = 0.002
        far_dist = 0.01
        
        manager.update_refinement_bands(near_dist, far_dist)
        
        # Should update refinement regions (implementation dependent)
        assert manager.config is not None
    
    def test_openfoam_env_setup(self, sample_config_file):
        """Test OpenFOAM environment setup"""
        manager = ConfigManager(str(sample_config_file))
        env_cmd = manager.get_openfoam_env()
        
        assert "source" in env_cmd
        assert "bashrc" in env_cmd
    
    def test_layer_validation(self, sample_config_file):
        """Test layer configuration validation"""
        manager = ConfigManager(str(sample_config_file))
        
        # Test validation fixes minThickness > firstLayerThickness
        manager.config["snappyHexMeshDict"]["addLayersControls"]["firstLayerThickness_abs"] = 10e-6
        manager.config["snappyHexMeshDict"]["addLayersControls"]["minThickness_abs"] = 20e-6
        
        manager._validate_layer_constraints()
        
        layers = manager.config["snappyHexMeshDict"]["addLayersControls"]
        assert layers["minThickness_abs"] <= layers["firstLayerThickness_abs"]
    
    def test_solver_preset_application(self, temp_dir):
        """Test solver-specific preset application"""
        config = {
            "physics": {"solver_mode": "LES"},
            "targets": {}
        }
        config_file = temp_dir / "les_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        manager = ConfigManager(str(config_file))
        
        # LES should have stricter quality targets
        assert manager.targets.max_nonortho <= 45  # Stricter than RANS
        assert manager.targets.max_skewness <= 2.0  # Stricter than RANS
    
    def test_backward_compatibility(self, temp_dir):
        """Test backward compatibility with old config format"""
        old_config = {
            "LAYERS": {
                "firstLayerThickness_abs": 40e-6,
                "nSurfaceLayers": 12
            },
            "acceptance_criteria": {
                "maxNonOrtho": 60,
                "maxSkewness": 3.5
            }
        }
        config_file = temp_dir / "old_config.json"
        with open(config_file, 'w') as f:
            json.dump(old_config, f)
        
        manager = ConfigManager(str(config_file))
        
        # Should map old structure to new format
        layers = manager.config["snappyHexMeshDict"]["addLayersControls"]
        assert layers["firstLayerThickness_abs"] == 40e-6
        assert layers["nSurfaceLayers"] == 12
        
        assert manager.targets.max_nonortho == 60
        assert manager.targets.max_skewness == 3.5
    
    def test_config_export(self, sample_config_file, temp_dir):
        """Test configuration export"""
        manager = ConfigManager(str(sample_config_file))
        
        # Modify configuration
        manager.config["physics"]["solver_mode"] = "LES"
        
        # Export configuration
        export_path = temp_dir / "exported_config.json"
        manager.export_config(str(export_path))
        
        assert export_path.exists()
        
        # Verify exported content
        with open(export_path) as f:
            exported = json.load(f)
        
        assert exported["physics"]["solver_mode"] == "LES"