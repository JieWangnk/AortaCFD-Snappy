"""
Unit tests for Stage1 Optimizer (main orchestrator).
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from mesh_optim.stage1.optimizer import Stage1Optimizer, run_stage1_optimization


class TestStage1Optimizer:
    """Test suite for Stage1Optimizer"""
    
    @patch('mesh_optim.stage1.optimizer.ConfigManager')
    @patch('mesh_optim.stage1.optimizer.GeometryProcessor')
    @patch('mesh_optim.stage1.optimizer.PhysicsCalculator')
    @patch('mesh_optim.stage1.optimizer.QualityAnalyzer')
    @patch('mesh_optim.stage1.optimizer.MeshGenerator')
    @patch('mesh_optim.stage1.optimizer.IterationManager')
    def test_initialization(self, mock_iter_mgr, mock_mesh_gen, mock_quality, 
                           mock_physics, mock_geometry, mock_config, 
                           sample_config_file, simple_stl_geometry, temp_dir):
        """Test Stage1Optimizer initialization"""
        
        optimizer = Stage1Optimizer(
            str(sample_config_file),
            str(simple_stl_geometry),
            str(temp_dir)
        )
        
        assert optimizer.config_path == Path(sample_config_file)
        assert optimizer.geometry_path == Path(simple_stl_geometry)
        assert optimizer.work_dir == Path(temp_dir)
        assert not optimizer.optimization_started
        assert not optimizer.optimization_complete
        
        # Verify modules were initialized
        mock_config.assert_called_once()
        mock_geometry.assert_called_once()
        mock_physics.assert_called_once()
        mock_quality.assert_called_once()
        mock_mesh_gen.assert_called_once()
        mock_iter_mgr.assert_called_once()
    
    @patch('mesh_optim.stage1.optimizer.Stage1Optimizer._generate_and_assess_mesh')
    @patch('mesh_optim.stage1.optimizer.Stage1Optimizer._calculate_physics_parameters')
    def test_single_mesh_generation(self, mock_calc_physics, mock_gen_assess,
                                   sample_config_file, simple_stl_geometry, temp_dir):
        """Test single mesh generation mode"""
        
        # Mock dependencies
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Mock geometry processing
            optimizer.geometry_processor.process_stl_geometry.return_value = {
                'valid': True,
                'characteristic_length': 0.02
            }
            
            # Mock mesh generation and assessment
            mock_gen_assess.return_value = {
                'success': True,
                'snap_metrics': {'maxNonOrtho': 40.0},
                'layer_metrics': {'maxNonOrtho': 45.0, 'cells': 50000},
                'layer_coverage': {'coverage_overall': 0.8},
                'meets_targets': True
            }
            
            result = optimizer.run_single_mesh_generation()
            
            assert result['status'] == 'completed'
            assert 'mesh_directory' in result
            mock_calc_physics.assert_called_once()
            mock_gen_assess.assert_called_once()
    
    def test_optimization_status_tracking(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test optimization status tracking"""
        
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Initial status
            status = optimizer.get_current_status()
            assert not status['optimization_started']
            assert not status['optimization_complete']
            assert status['current_iteration'] == 0
            
            # Simulate optimization progress
            optimizer.optimization_started = True
            optimizer.iteration_manager.current_iteration = 3
            optimizer.iteration_manager.converged = True
            optimizer.iteration_manager.convergence_reason = "Quality converged"
            optimizer.iteration_manager.best_iteration = {'quality_score': 0.85}
            
            status = optimizer.get_current_status()
            assert status['optimization_started']
            assert status['current_iteration'] == 3
            assert status['converged']
            assert status['best_quality'] == 0.85
    
    def test_physics_parameters_calculation(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test physics parameters calculation"""
        
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Mock config manager methods
            optimizer.config_manager.config = {
                'base_size': 0.001,
                'physics': {
                    'peak_velocity': 1.5,
                    'use_womersley_bands': False
                }
            }
            optimizer.config_manager.update_layer_parameters = Mock()
            optimizer.config_manager.update_refinement_bands = Mock()
            
            # Mock physics calculator
            optimizer.physics_calculator.calculate_layer_parameters.return_value = {
                'firstLayerThickness_abs': 50e-6,
                'minThickness_abs': 7.5e-6,
                'nSurfaceLayers': 8,
                'expansionRatio': 1.2
            }
            optimizer.physics_calculator.calculate_refinement_bands.return_value = (0.002, 0.01)
            
            geometry_info = {'characteristic_length': 0.02}
            
            optimizer._calculate_physics_parameters(geometry_info)
            
            # Verify methods were called
            optimizer.physics_calculator.calculate_layer_parameters.assert_called_once()
            optimizer.physics_calculator.calculate_refinement_bands.assert_called_once()
            optimizer.config_manager.update_layer_parameters.assert_called_once()
            optimizer.config_manager.update_refinement_bands.assert_called_once()
    
    def test_mesh_export_functionality(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test mesh export in different formats"""
        
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Create mock final mesh directory
            final_mesh_dir = temp_dir / "final_mesh"
            final_mesh_dir.mkdir()
            (final_mesh_dir / "mesh_data.txt").write_text("mock mesh data")
            optimizer.final_mesh_dir = final_mesh_dir
            
            export_dir = temp_dir / "export"
            
            # Test OpenFOAM export
            with patch('shutil.copytree') as mock_copytree:
                result = optimizer.export_final_mesh(str(export_dir), "openfoam")
                assert result is True
                mock_copytree.assert_called_once()
            
            # Test unsupported format
            result = optimizer.export_final_mesh(str(export_dir), "unsupported")
            assert result is False
    
    def test_optimization_summary_generation(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test optimization summary generation"""
        
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Mock iteration manager summary
            optimizer.iteration_manager.get_optimization_summary.return_value = {
                'status': 'converged',
                'total_iterations': 5,
                'best_iteration': {
                    'quality_score': 0.9,
                    'layer_metrics': {'cells': 75000, 'maxNonOrtho': 42.0},
                    'layer_coverage': {'coverage_overall': 0.85}
                }
            }
            
            optimizer.final_mesh_dir = temp_dir / "final"
            optimizer.final_mesh_dir.mkdir()
            
            summary = optimizer._generate_final_summary(120.5)  # 120.5 seconds
            
            assert summary['status'] == 'completed'
            assert summary['total_time'] == 120.5
            assert 'optimization_results' in summary
            assert 'final_quality' in summary
            assert summary['final_quality']['quality_score'] == 0.9
    
    def test_convenience_functions(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test convenience functions"""
        
        with patch('mesh_optim.stage1.optimizer.Stage1Optimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # Test run_stage1_optimization
            mock_optimizer.run_full_optimization.return_value = {'status': 'completed'}
            
            result = run_stage1_optimization(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir),
                max_iterations=5,
                parallel_procs=2
            )
            
            assert result['status'] == 'completed'
            mock_optimizer.run_full_optimization.assert_called_once_with(5, 2)
    
    def test_error_handling(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test error handling in optimization workflow"""
        
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Mock geometry processing failure
            optimizer.geometry_processor.process_stl_geometry.return_value = {
                'valid': False,
                'error': 'Invalid geometry file'
            }
            
            with pytest.raises(RuntimeError, match="Invalid geometry"):
                optimizer.run_full_optimization()
    
    def test_mesh_generation_and_assessment(self, sample_config_file, simple_stl_geometry, temp_dir):
        """Test mesh generation and assessment workflow"""
        
        with patch.multiple('mesh_optim.stage1.optimizer',
                          ConfigManager=Mock(),
                          GeometryProcessor=Mock(),
                          PhysicsCalculator=Mock(),
                          QualityAnalyzer=Mock(),
                          MeshGenerator=Mock(),
                          IterationManager=Mock()):
            
            optimizer = Stage1Optimizer(
                str(sample_config_file),
                str(simple_stl_geometry),
                str(temp_dir)
            )
            
            # Mock all the workflow steps
            optimizer.geometry_processor.prepare_geometry_for_meshing.return_value = {
                'processed_files': ['wall_aorta.stl']
            }
            
            optimizer.mesh_generator.generate_case_files.return_value = True
            optimizer.mesh_generator.run_blockmesh.return_value = {'success': True}
            optimizer.mesh_generator.run_snappy_no_layers.return_value = {'success': True}
            optimizer.mesh_generator.run_snappy_layers.return_value = {'success': True}
            
            optimizer.quality_analyzer.assess_mesh_quality.side_effect = [
                {'maxNonOrtho': 40.0, 'meshOK': True},  # snap metrics
                {'maxNonOrtho': 45.0, 'cells': 50000, 'meshOK': True}  # layer metrics
            ]
            optimizer.quality_analyzer.assess_layer_quality.return_value = {'coverage_overall': 0.8}
            optimizer.quality_analyzer.meets_quality_constraints.return_value = True
            optimizer.quality_analyzer.diagnose_quality_issues.return_value = {'suggestions': []}
            
            mesh_dir = temp_dir / "test_mesh"
            mesh_dir.mkdir()
            
            result = optimizer._generate_and_assess_mesh(mesh_dir, 1, "test_phase")
            
            assert result['success'] is True
            assert result['meets_targets'] is True
            assert 'snap_metrics' in result
            assert 'layer_metrics' in result
            assert 'layer_coverage' in result