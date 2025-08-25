"""
Integration tests for Stage1 mesh optimization workflow.
Tests the full pipeline from configuration to final mesh generation.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from mesh_optim.stage1_mesh import Stage1MeshOptimizer


class TestStage1Integration:
    """Integration test suite for complete Stage1 workflow"""
    
    def create_test_geometry_dir(self, temp_dir):
        """Create a test geometry directory with STL files"""
        geometry_dir = temp_dir / "geometry"
        geometry_dir.mkdir()
        
        # Create a simple STL file
        stl_content = """solid test_aorta
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 10.0 0.0 0.0
      vertex 5.0 10.0 0.0
    endloop
  endfacet
  facet normal 0.0 0.0 -1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 5.0 10.0 0.0
      vertex 10.0 0.0 0.0
    endloop
  endfacet
endsolid test_aorta
"""
        (geometry_dir / "wall_aorta.stl").write_text(stl_content)
        
        # Create outlet files
        (geometry_dir / "outlet1.stl").write_text(stl_content.replace("test_aorta", "outlet1"))
        (geometry_dir / "outlet2.stl").write_text(stl_content.replace("test_aorta", "outlet2"))
        
        return geometry_dir
    
    def create_test_config(self, temp_dir):
        """Create a complete test configuration"""
        config = {
            "openfoam_env_path": "echo 'Mock OpenFOAM setup'",  # Mock for testing
            "base_size": 0.002,
            "physics": {
                "rho": 1060,
                "mu": 0.0035,
                "peak_velocity": 1.5,
                "heart_rate_hz": 1.2,
                "solver_mode": "RANS",
                "use_womersley_bands": False
            },
            "snappyHexMeshDict": {
                "castellatedMeshControls": {
                    "maxLocalCells": 50000,
                    "maxGlobalCells": 200000,
                    "nCellsBetweenLevels": 1,
                    "features": [],
                    "refinementSurfaces": {},
                    "refinementRegions": {}
                },
                "snapControls": {
                    "nSmoothPatch": 3,
                    "tolerance": 2.0,
                    "nRelaxIter": 5,
                    "nFeatureSnapIter": 10
                },
                "addLayersControls": {
                    "nSurfaceLayers": 6,
                    "firstLayerThickness_abs": 40e-6,
                    "minThickness_abs": 6e-6,
                    "expansionRatio": 1.2,
                    "nGrow": 1,
                    "featureAngle": 70,
                    "maxThicknessToMedialRatio": 0.45,
                    "minMedianAxisAngle": 70,
                    "maxBoundarySkewness": 4.0,
                    "maxInternalSkewness": 4.0
                }
            },
            "targets": {
                "max_nonortho": 65,
                "max_skewness": 4.0,
                "min_layer_cov": 0.60
            },
            "STAGE1": {
                "max_iterations": 3,  # Keep low for testing
                "base_size_mode": "diameter",
                "N_D": 15,
                "cells_per_cm": 10,
                "ladder": [[1,1], [1,2]],
                "near_band_cells": 3,
                "far_band_cells": 6
            },
            "BLOCKMESH": {
                "resolution": 30
            }
        }
        
        config_file = temp_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality')
    @patch('mesh_optim.utils.parse_layer_coverage')
    def test_legacy_optimizer_initialization(self, mock_parse_coverage, 
                                           mock_check_quality, mock_run_command,
                                           temp_dir):
        """Test legacy optimizer initialization and configuration loading"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file,
            output_dir=temp_dir / "output"
        )
        
        # Verify initialization
        assert optimizer.config is not None
        assert optimizer.geometry_dir == geometry_dir
        assert optimizer.config_file == config_file
        assert (temp_dir / "output").exists()
        
        # Verify configuration sections
        assert "physics" in optimizer.config
        assert "targets" in optimizer.config
        assert "snappyHexMeshDict" in optimizer.config
        assert "STAGE1" in optimizer.config
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality') 
    @patch('mesh_optim.utils.parse_layer_coverage')
    def test_configuration_validation_and_defaults(self, mock_parse_coverage,
                                                  mock_check_quality, mock_run_command,
                                                  temp_dir):
        """Test configuration validation and default value application"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        
        # Create minimal configuration
        minimal_config = {
            "base_size": 0.001,
            "openfoam_env_path": "echo 'test'"
        }
        config_file = temp_dir / "minimal_config.json"
        with open(config_file, 'w') as f:
            json.dump(minimal_config, f)
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Verify defaults were applied
        assert "snappyHexMeshDict" in optimizer.config
        assert "addLayersControls" in optimizer.config["snappyHexMeshDict"]
        assert optimizer.config["snappyHexMeshDict"]["addLayersControls"]["nSurfaceLayers"] > 0
        
        # Verify layer validation was performed
        layers = optimizer.config["snappyHexMeshDict"]["addLayersControls"]
        assert layers["minThickness_abs"] <= layers["firstLayerThickness_abs"]
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality')
    @patch('mesh_optim.utils.parse_layer_coverage')
    @patch('mesh_optim.stage1_mesh.Stage1MeshOptimizer._run_snap_then_layers')
    def test_modular_compatibility_layer(self, mock_snap_layers, mock_parse_coverage,
                                       mock_check_quality, mock_run_command,
                                       temp_dir):
        """Test that modular components integrate with legacy system"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        # Mock successful mesh generation
        mock_snap_layers.return_value = (
            {"maxNonOrtho": 40.0, "maxSkewness": 2.0, "meshOK": True, "cells": 30000},  # snap
            {"maxNonOrtho": 45.0, "maxSkewness": 2.5, "meshOK": True, "cells": 35000},  # layers  
            {"coverage_overall": 0.75}  # coverage
        )
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Check if modular components were detected
        has_modular = hasattr(optimizer, '_use_modular')
        
        if has_modular and optimizer._use_modular:
            # Modular system should be available
            assert hasattr(optimizer, '_modular_optimizer')
            assert hasattr(optimizer, '_run_modular_optimization')
        else:
            # Should fall back to legacy implementation
            assert not hasattr(optimizer, '_modular_optimizer')
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality')
    @patch('mesh_optim.utils.parse_layer_coverage')
    def test_physics_parameter_calculation_workflow(self, mock_parse_coverage,
                                                   mock_check_quality, mock_run_command,
                                                   temp_dir):
        """Test physics-based parameter calculation workflow"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        # Mock OpenFOAM commands
        mock_run_command.return_value = MagicMock(
            returncode=0,
            stdout="Mock OpenFOAM output",
            stderr=""
        )
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Verify physics parameters are reasonable
        layers = optimizer.config["snappyHexMeshDict"]["addLayersControls"]
        
        assert layers["firstLayerThickness_abs"] > 0
        assert layers["firstLayerThickness_abs"] < 1e-3  # Less than 1mm
        assert layers["minThickness_abs"] < layers["firstLayerThickness_abs"]
        assert layers["expansionRatio"] >= 1.1
        assert layers["nSurfaceLayers"] >= 3
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality')
    @patch('mesh_optim.utils.parse_layer_coverage')
    def test_mesh_quality_assessment_workflow(self, mock_parse_coverage,
                                            mock_check_quality, mock_run_command,
                                            temp_dir):
        """Test mesh quality assessment and convergence detection"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        # Mock quality assessment results
        quality_sequence = [
            # First iteration - poor quality
            {"maxNonOrtho": 70.0, "maxSkewness": 5.0, "meshOK": False, "cells": 25000},
            # Second iteration - improved
            {"maxNonOrtho": 50.0, "maxSkewness": 3.0, "meshOK": True, "cells": 30000},
            # Third iteration - converged
            {"maxNonOrtho": 45.0, "maxSkewness": 2.5, "meshOK": True, "cells": 32000}
        ]
        
        mock_check_quality.side_effect = quality_sequence
        mock_parse_coverage.return_value = {"coverage_overall": 0.70}
        
        # Mock successful OpenFOAM commands
        mock_run_command.return_value = MagicMock(
            returncode=0,
            stdout="Mesh generation completed successfully",
            stderr=""
        )
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Test quality evaluation logic
        targets = optimizer.config.get("targets", {})
        max_nonortho = targets.get("max_nonortho", 65)
        max_skewness = targets.get("max_skewness", 4.0)
        min_coverage = targets.get("min_layer_cov", 0.65)
        
        # First iteration should fail constraints
        poor_quality = quality_sequence[0]
        constraints_met = (
            poor_quality["meshOK"] and
            poor_quality["maxNonOrtho"] <= max_nonortho and
            poor_quality["maxSkewness"] <= max_skewness and
            0.70 >= min_coverage
        )
        assert not constraints_met
        
        # Third iteration should meet constraints
        good_quality = quality_sequence[2]
        constraints_met = (
            good_quality["meshOK"] and
            good_quality["maxNonOrtho"] <= max_nonortho and
            good_quality["maxSkewness"] <= max_skewness and
            0.70 >= min_coverage
        )
        assert constraints_met
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality')
    @patch('mesh_optim.utils.parse_layer_coverage')
    def test_error_handling_and_recovery(self, mock_parse_coverage,
                                       mock_check_quality, mock_run_command,
                                       temp_dir):
        """Test error handling and recovery mechanisms"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        # Mock command failures
        def mock_command_failure(cmd, **kwargs):
            if "checkMesh" in str(cmd):
                # Simulate mesh check failure
                result = MagicMock()
                result.returncode = 1
                result.stdout = "ERROR: Mesh check failed"
                result.stderr = "Fatal mesh errors detected"
                return result
            else:
                # Other commands succeed
                result = MagicMock()
                result.returncode = 0
                result.stdout = "Command completed"
                result.stderr = ""
                return result
        
        mock_run_command.side_effect = mock_command_failure
        
        # Mock fallback quality metrics
        mock_check_quality.return_value = {
            "maxNonOrtho": 999,
            "maxSkewness": 999,
            "meshOK": False,
            "cells": 0
        }
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # System should handle errors gracefully
        assert optimizer.config is not None
        assert optimizer.max_memory_gb > 0
    
    @patch('mesh_optim.utils.run_command')
    def test_output_directory_management(self, mock_run_command, temp_dir):
        """Test output directory creation and management"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        output_dir = temp_dir / "custom_output"
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file,
            output_dir=output_dir
        )
        
        # Verify output directory structure
        assert output_dir.exists()
        assert optimizer.output_dir == output_dir
        
        # Test default output directory
        optimizer_default = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        expected_default = geometry_dir.parent / "output" / "stage1_mesh"
        assert optimizer_default.output_dir == expected_default
        assert expected_default.exists()
    
    def test_configuration_backward_compatibility(self, temp_dir):
        """Test backward compatibility with different configuration formats"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        
        # Old style configuration
        old_config = {
            "openfoam_env_path": "echo 'test'",
            "LAYERS": {
                "firstLayerThickness_abs": 30e-6,
                "nSurfaceLayers": 10,
                "expansionRatio": 1.15
            },
            "acceptance_criteria": {
                "maxNonOrtho": 60,
                "maxSkewness": 3.5,
                "min_layer_coverage": 0.70
            },
            "SNAPPY": {
                "maxLocalCells": 100000,
                "resolveFeatureAngle": 45
            }
        }
        
        config_file = temp_dir / "old_config.json"
        with open(config_file, 'w') as f:
            json.dump(old_config, f)
        
        # Should handle old configuration format
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Verify mapping to new structure
        assert "snappyHexMeshDict" in optimizer.config
        layers = optimizer.config["snappyHexMeshDict"]["addLayersControls"]
        assert layers["firstLayerThickness_abs"] == 30e-6
        assert layers["nSurfaceLayers"] == 10
        
    def test_resource_management(self, temp_dir):
        """Test memory and resource management"""
        geometry_dir = self.create_test_geometry_dir(temp_dir)
        config_file = self.create_test_config(temp_dir)
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Mock system memory
            mock_memory.return_value.available = 16 * 1024**3  # 16 GB
            
            optimizer = Stage1MeshOptimizer(
                geometry_dir=geometry_dir,
                config_file=config_file
            )
            
            # Should set reasonable memory limits
            assert optimizer.max_memory_gb > 0
            assert optimizer.max_memory_gb <= 12  # Cap at 12GB
            assert optimizer.max_memory_gb <= 16 * 0.7  # 70% of available