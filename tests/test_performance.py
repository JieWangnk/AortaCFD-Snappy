"""
Performance validation tests for Stage1 mesh optimization.
Tests memory usage, execution time, and scalability characteristics.
"""
import pytest
import time
import psutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import gc

from mesh_optim.stage1_mesh import Stage1MeshOptimizer


class TestPerformanceValidation:
    """Performance validation test suite"""
    
    def create_performance_geometry(self, temp_dir, complexity="medium"):
        """Create test geometry with different complexity levels"""
        geometry_dir = temp_dir / "geometry"
        geometry_dir.mkdir()
        
        # Generate STL with different triangle counts based on complexity
        if complexity == "simple":
            triangles = 100
        elif complexity == "medium": 
            triangles = 1000
        elif complexity == "complex":
            triangles = 5000
        else:
            triangles = 1000
        
        stl_content = "solid test_geometry\n"
        
        # Generate triangular faces
        for i in range(triangles):
            # Simple triangle pattern
            x_offset = (i % 10) * 0.001
            y_offset = (i // 10) * 0.001
            
            stl_content += f"""  facet normal 0.0 0.0 1.0
    outer loop
      vertex {x_offset:.6f} {y_offset:.6f} 0.0
      vertex {x_offset + 0.001:.6f} {y_offset:.6f} 0.0
      vertex {x_offset + 0.0005:.6f} {y_offset + 0.001:.6f} 0.0
    endloop
  endfacet
"""
        
        stl_content += "endsolid test_geometry\n"
        
        (geometry_dir / "wall_aorta.stl").write_text(stl_content)
        return geometry_dir
    
    def create_performance_config(self, temp_dir, cell_target="small"):
        """Create configuration with different cell count targets"""
        if cell_target == "small":
            max_local = 10000
            max_global = 50000
            base_size = 0.005
        elif cell_target == "medium":
            max_local = 50000
            max_global = 200000
            base_size = 0.002
        elif cell_target == "large":
            max_local = 200000
            max_global = 800000
            base_size = 0.001
        else:
            max_local = 50000
            max_global = 200000
            base_size = 0.002
        
        config = {
            "openfoam_env_path": "echo 'Mock OpenFOAM'",
            "base_size": base_size,
            "physics": {
                "rho": 1060,
                "mu": 0.0035,
                "peak_velocity": 1.5,
                "solver_mode": "RANS"
            },
            "snappyHexMeshDict": {
                "castellatedMeshControls": {
                    "maxLocalCells": max_local,
                    "maxGlobalCells": max_global,
                    "nCellsBetweenLevels": 1
                },
                "snapControls": {
                    "nSmoothPatch": 3,
                    "tolerance": 2.0
                },
                "addLayersControls": {
                    "nSurfaceLayers": 8,
                    "firstLayerThickness_abs": 50e-6,
                    "minThickness_abs": 7.5e-6,
                    "expansionRatio": 1.2
                }
            },
            "targets": {
                "max_nonortho": 65,
                "max_skewness": 4.0,
                "min_layer_cov": 0.65
            },
            "STAGE1": {
                "max_iterations": 5,
                "base_size_mode": "diameter"
            }
        }
        
        config_file = temp_dir / f"config_{cell_target}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def measure_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_initialization_performance(self, temp_dir):
        """Test initialization time and memory usage"""
        geometry_dir = self.create_performance_geometry(temp_dir, "medium")
        config_file = self.create_performance_config(temp_dir, "medium")
        
        # Measure initialization
        start_memory = self.measure_memory_usage()
        start_time = time.time()
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        init_time = time.time() - start_time
        init_memory = self.measure_memory_usage() - start_memory
        
        # Performance assertions
        assert init_time < 5.0, f"Initialization took {init_time:.2f}s, should be < 5s"
        assert init_memory < 100, f"Initialization used {init_memory:.1f}MB, should be < 100MB"
        
        # Verify initialization completed successfully
        assert optimizer.config is not None
        assert optimizer.geometry_dir.exists()
        assert optimizer.output_dir.exists()
    
    def test_configuration_loading_performance(self, temp_dir):
        """Test configuration loading and validation performance"""
        geometry_dir = self.create_performance_geometry(temp_dir, "simple")
        
        # Test different configuration sizes
        config_sizes = ["small", "medium", "large"]
        load_times = []
        
        for size in config_sizes:
            config_file = self.create_performance_config(temp_dir, size)
            
            start_time = time.time()
            
            optimizer = Stage1MeshOptimizer(
                geometry_dir=geometry_dir,
                config_file=config_file
            )
            
            load_time = time.time() - start_time
            load_times.append(load_time)
            
            # Each config should load quickly
            assert load_time < 2.0, f"Config loading took {load_time:.2f}s for {size}"
        
        # Loading time should scale reasonably
        assert max(load_times) - min(load_times) < 1.0, "Config loading time variance too high"
    
    @patch('mesh_optim.utils.run_command')
    @patch('mesh_optim.utils.check_mesh_quality')
    @patch('mesh_optim.utils.parse_layer_coverage')
    def test_memory_usage_during_optimization(self, mock_parse, mock_quality, 
                                            mock_run, temp_dir):
        """Test memory usage during optimization workflow"""
        geometry_dir = self.create_performance_geometry(temp_dir, "medium")
        config_file = self.create_performance_config(temp_dir, "medium")
        
        # Mock successful operations
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        mock_quality.return_value = {
            "maxNonOrtho": 45.0,
            "maxSkewness": 2.5,
            "meshOK": True,
            "cells": 35000
        }
        mock_parse.return_value = {"coverage_overall": 0.75}
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Monitor memory during workflow simulation
        initial_memory = self.measure_memory_usage()
        memory_measurements = [initial_memory]
        
        # Simulate configuration updates (memory should be stable)
        for i in range(5):
            # Simulate parameter updates
            optimizer.config["snappyHexMeshDict"]["addLayersControls"]["nSurfaceLayers"] = 8 + i
            memory_measurements.append(self.measure_memory_usage())
            
            # Force garbage collection
            gc.collect()
        
        peak_memory = max(memory_measurements)
        memory_growth = peak_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB, should be < 50MB"
        
        # No memory leaks - final memory should be close to initial
        final_memory = self.measure_memory_usage()
        memory_leak = final_memory - initial_memory
        assert memory_leak < 20, f"Potential memory leak: {memory_leak:.1f}MB growth"
    
    def test_configuration_validation_performance(self, temp_dir):
        """Test performance of configuration validation"""
        geometry_dir = self.create_performance_geometry(temp_dir, "simple")
        
        # Create configuration with many validation issues
        problematic_config = {
            "openfoam_env_path": "echo 'test'",
            "base_size": 0.001,
            "snappyHexMeshDict": {
                "addLayersControls": {
                    "firstLayerThickness_abs": 10e-6,  # Will be corrected
                    "minThickness_abs": 50e-6,         # Too large, needs fixing
                    "nSurfaceLayers": 25,              # Very high, needs clamping
                    "expansionRatio": 3.0              # Too high, needs adjustment
                }
            }
        }
        
        config_file = temp_dir / "problematic_config.json"
        with open(config_file, 'w') as f:
            json.dump(problematic_config, f)
        
        start_time = time.time()
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        validation_time = time.time() - start_time
        
        # Validation should be fast even with many corrections
        assert validation_time < 3.0, f"Validation took {validation_time:.2f}s, should be < 3s"
        
        # Verify corrections were applied
        layers = optimizer.config["snappyHexMeshDict"]["addLayersControls"]
        assert layers["minThickness_abs"] <= layers["firstLayerThickness_abs"]
        assert layers["expansionRatio"] <= 2.0
    
    @pytest.mark.slow
    @patch('mesh_optim.utils.run_command')
    def test_scalability_with_geometry_complexity(self, mock_run, temp_dir):
        """Test how performance scales with geometry complexity"""
        complexities = ["simple", "medium", "complex"]
        init_times = []
        memory_usage = []
        
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        
        for complexity in complexities:
            geometry_dir = self.create_performance_geometry(temp_dir, complexity)
            config_file = self.create_performance_config(temp_dir, "medium")
            
            # Measure initialization performance
            start_memory = self.measure_memory_usage()
            start_time = time.time()
            
            optimizer = Stage1MeshOptimizer(
                geometry_dir=geometry_dir,
                config_file=config_file
            )
            
            init_time = time.time() - start_time
            memory_used = self.measure_memory_usage() - start_memory
            
            init_times.append(init_time)
            memory_usage.append(memory_used)
            
            # Clean up
            del optimizer
            gc.collect()
        
        # Performance should scale reasonably with complexity
        assert all(t < 10.0 for t in init_times), "Initialization times too high"
        assert all(m < 200 for m in memory_usage), "Memory usage too high"
        
        # Time should increase with complexity but not excessively
        time_ratio = max(init_times) / min(init_times)
        assert time_ratio < 5.0, f"Time scaling ratio too high: {time_ratio:.1f}"
    
    def test_resource_limit_enforcement(self, temp_dir):
        """Test that resource limits are properly enforced"""
        geometry_dir = self.create_performance_geometry(temp_dir, "medium")
        config_file = self.create_performance_config(temp_dir, "large")
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate limited memory system
            mock_memory.return_value.available = 2 * 1024**3  # 2GB available
            
            optimizer = Stage1MeshOptimizer(
                geometry_dir=geometry_dir,
                config_file=config_file
            )
            
            # Memory limit should be set appropriately
            assert optimizer.max_memory_gb <= 2 * 0.7  # 70% of available
            assert optimizer.max_memory_gb > 0
            
            # Should not exceed system constraints
            expected_limit = min(2 * 0.7, 12)  # 70% of 2GB or 12GB cap
            assert optimizer.max_memory_gb <= expected_limit
    
    def test_garbage_collection_effectiveness(self, temp_dir):
        """Test that objects are properly garbage collected"""
        geometry_dir = self.create_performance_geometry(temp_dir, "medium")
        config_file = self.create_performance_config(temp_dir, "medium")
        
        initial_memory = self.measure_memory_usage()
        
        # Create and destroy multiple optimizers
        for i in range(5):
            optimizer = Stage1MeshOptimizer(
                geometry_dir=geometry_dir,
                config_file=config_file
            )
            
            # Simulate some work
            _ = optimizer.config.copy()
            
            # Explicit cleanup
            del optimizer
            gc.collect()
        
        final_memory = self.measure_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Should not have significant memory accumulation
        assert memory_growth < 30, f"Memory accumulated: {memory_growth:.1f}MB"
    
    @pytest.mark.benchmark
    def test_configuration_update_performance(self, temp_dir):
        """Benchmark configuration update operations"""
        geometry_dir = self.create_performance_geometry(temp_dir, "simple")
        config_file = self.create_performance_config(temp_dir, "medium")
        
        optimizer = Stage1MeshOptimizer(
            geometry_dir=geometry_dir,
            config_file=config_file
        )
        
        # Benchmark parameter updates
        update_times = []
        
        for i in range(100):  # Many small updates
            start_time = time.perf_counter()
            
            # Simulate typical parameter updates
            layers = optimizer.config["snappyHexMeshDict"]["addLayersControls"]
            layers["firstLayerThickness_abs"] = (40 + i) * 1e-6
            layers["nSurfaceLayers"] = 6 + (i % 5)
            layers["expansionRatio"] = 1.2 + (i % 10) * 0.01
            
            update_time = time.perf_counter() - start_time
            update_times.append(update_time)
        
        avg_update_time = sum(update_times) / len(update_times)
        max_update_time = max(update_times)
        
        # Updates should be very fast
        assert avg_update_time < 0.001, f"Average update time {avg_update_time*1000:.2f}ms too high"
        assert max_update_time < 0.01, f"Max update time {max_update_time*1000:.2f}ms too high"
    
    def test_concurrent_optimizer_performance(self, temp_dir):
        """Test performance when multiple optimizers exist simultaneously"""
        geometry_dir = self.create_performance_geometry(temp_dir, "medium")
        config_file = self.create_performance_config(temp_dir, "medium")
        
        start_memory = self.measure_memory_usage()
        optimizers = []
        
        # Create multiple optimizers
        try:
            for i in range(3):  # Limited number for testing
                optimizer = Stage1MeshOptimizer(
                    geometry_dir=geometry_dir,
                    config_file=config_file,
                    output_dir=temp_dir / f"output_{i}"
                )
                optimizers.append(optimizer)
            
            concurrent_memory = self.measure_memory_usage()
            memory_per_optimizer = (concurrent_memory - start_memory) / len(optimizers)
            
            # Each optimizer should have reasonable memory footprint
            assert memory_per_optimizer < 100, f"Memory per optimizer: {memory_per_optimizer:.1f}MB"
            
            # Test that they don't interfere with each other
            for i, opt in enumerate(optimizers):
                opt.config["snappyHexMeshDict"]["addLayersControls"]["nSurfaceLayers"] = 10 + i
            
            # Verify independent configurations
            layer_counts = [opt.config["snappyHexMeshDict"]["addLayersControls"]["nSurfaceLayers"] 
                           for opt in optimizers]
            assert layer_counts == [10, 11, 12], "Optimizers interfered with each other"
            
        finally:
            # Cleanup
            for opt in optimizers:
                del opt
            gc.collect()