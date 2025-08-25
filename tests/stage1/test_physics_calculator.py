"""
Unit tests for PhysicsCalculator module.
"""
import pytest
import math
from unittest.mock import MagicMock

from mesh_optim.stage1.physics_calculator import PhysicsCalculator
from mesh_optim.stage1.config_manager import ConfigManager


class TestPhysicsCalculator:
    """Test suite for PhysicsCalculator"""
    
    def test_initialization(self, sample_config_file):
        """Test PhysicsCalculator initialization"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        assert calculator.config is not None
        assert calculator.constants is not None
    
    def test_yplus_first_layer_calculation(self, sample_config_file):
        """Test y+ based first layer thickness calculation"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        diameter = 0.02  # 20mm
        peak_velocity = 1.5  # m/s
        target_yplus = 1.0
        
        first_layer = calculator.calculate_yplus_first_layer(
            diameter, peak_velocity, target_yplus, "turbulent"
        )
        
        assert isinstance(first_layer, float)
        assert first_layer > 0
        assert first_layer < 1e-3  # Should be less than 1mm
        
        # Test different y+ values
        yplus_5 = calculator.calculate_yplus_first_layer(diameter, peak_velocity, 5.0)
        assert yplus_5 > first_layer  # Higher y+ should give thicker layer
    
    def test_reynolds_number_calculation(self, sample_config_file):
        """Test Reynolds number calculation"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        diameter = 0.02  # 20mm
        velocity = 1.5   # m/s
        
        re_number = calculator.calculate_reynolds_number(diameter, velocity)
        
        assert isinstance(re_number, float)
        assert re_number > 1000  # Should be in turbulent range for blood flow
    
    def test_flow_regime_classification(self, sample_config_file):
        """Test flow regime classification"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        # Test different Reynolds numbers
        assert calculator.classify_flow_regime(1000) == "laminar"
        assert calculator.classify_flow_regime(3000) == "transitional"
        assert calculator.classify_flow_regime(5000) == "turbulent"
    
    def test_womersley_boundary_layer(self, sample_config_file):
        """Test Womersley boundary layer calculation"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        heart_rate = 1.2  # Hz (72 bpm)
        
        delta = calculator.calculate_womersley_boundary_layer(heart_rate)
        
        assert isinstance(delta, float)
        assert delta > 0
        assert delta < 0.01  # Should be reasonable for blood vessels
        
        # Higher frequency should give smaller boundary layer
        delta_high = calculator.calculate_womersley_boundary_layer(2.0)
        assert delta_high < delta
    
    def test_recommended_yplus_by_solver(self, sample_config_file):
        """Test recommended y+ values by solver type"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        # Test different solver modes
        assert calculator.get_recommended_yplus("LES") < 2.0  # LES needs low y+
        assert calculator.get_recommended_yplus("RANS") > 10.0  # RANS can use higher y+
        assert calculator.get_recommended_yplus("laminar") == 1.0  # Laminar needs y+ = 1
    
    def test_layer_parameters_calculation(self, sample_config_file):
        """Test comprehensive layer parameters calculation"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        diameter = 0.02
        velocity = 1.5
        base_cell_size = 0.001
        
        params = calculator.calculate_layer_parameters(diameter, velocity, base_cell_size)
        
        # Check required parameters are present
        required_keys = [
            'firstLayerThickness_abs', 'minThickness_abs', 'nSurfaceLayers',
            'expansionRatio', 'nGrow', 'target_yplus'
        ]
        for key in required_keys:
            assert key in params
        
        # Check reasonable values
        assert params['firstLayerThickness_abs'] > 0
        assert params['minThickness_abs'] < params['firstLayerThickness_abs']
        assert params['nSurfaceLayers'] >= 3
        assert 1.1 <= params['expansionRatio'] <= 2.0
        assert params['target_yplus'] > 0
    
    def test_refinement_bands_calculation(self, sample_config_file):
        """Test refinement bands calculation"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        base_size = 0.001
        
        # Test geometry-based bands
        near_dist, far_dist = calculator.calculate_refinement_bands(base_size, use_womersley=False)
        
        assert near_dist > 0
        assert far_dist > near_dist
        assert near_dist < 0.01  # Reasonable for CFD mesh
        assert far_dist < 0.1
        
        # Test Womersley-based bands
        near_w, far_w = calculator.calculate_refinement_bands(base_size, use_womersley=True)
        
        assert near_w > 0
        assert far_w > near_w
        # Womersley bands might be different from geometry-based
    
    def test_layer_distribution_optimization(self, sample_config_file):
        """Test layer distribution optimization"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        first_layer = 50e-6  # 50 microns
        bl_thickness = 0.002  # 2mm boundary layer
        
        # Test different solver modes
        n_layers_les, exp_ratio_les = calculator._optimize_layer_distribution(
            first_layer, bl_thickness, "LES"
        )
        n_layers_rans, exp_ratio_rans = calculator._optimize_layer_distribution(
            first_layer, bl_thickness, "RANS"
        )
        
        # LES should have more layers and gentler expansion
        assert n_layers_les >= n_layers_rans
        assert exp_ratio_les <= exp_ratio_rans
        
        # Check reasonable bounds
        assert 3 <= n_layers_les <= 20
        assert 3 <= n_layers_rans <= 20
        assert 1.1 <= exp_ratio_les <= 2.0
        assert 1.1 <= exp_ratio_rans <= 2.0
    
    def test_physics_consistency(self, sample_config_file):
        """Test physics calculations are consistent"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        # Same inputs should give same outputs
        diameter = 0.02
        velocity = 1.5
        
        re1 = calculator.calculate_reynolds_number(diameter, velocity)
        re2 = calculator.calculate_reynolds_number(diameter, velocity)
        assert re1 == re2
        
        first_layer1 = calculator.calculate_yplus_first_layer(diameter, velocity, 1.0)
        first_layer2 = calculator.calculate_yplus_first_layer(diameter, velocity, 1.0)
        assert first_layer1 == first_layer2
    
    def test_extreme_parameter_handling(self, sample_config_file):
        """Test handling of extreme parameter values"""
        config_manager = ConfigManager(str(sample_config_file))
        calculator = PhysicsCalculator(config_manager)
        
        # Test very small diameter
        small_diameter = 1e-6  # 1 micron
        velocity = 1.0
        
        first_layer = calculator.calculate_yplus_first_layer(small_diameter, velocity, 1.0)
        assert first_layer > 0
        
        # Test very high velocity
        diameter = 0.02
        high_velocity = 10.0  # 10 m/s
        
        first_layer_high = calculator.calculate_yplus_first_layer(diameter, high_velocity, 1.0)
        assert first_layer_high > 0
        
        # Should apply minimum thickness constraint
        min_thickness = calculator.constants.MIN_FIRST_LAYER_MICRONS * 1e-6
        assert first_layer >= min_thickness