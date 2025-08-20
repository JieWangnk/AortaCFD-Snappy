"""
Physics-aware mesh generation with proper boundary layer resolution

This module implements the first-layer thickness calculations and physics-aware 
mesh parameters for laminar/RANS/LES flow regimes.
"""

import math
import json
from pathlib import Path
from typing import Dict, Tuple

class PhysicsAwareMeshGenerator:
    """Generate physics-aware mesh parameters for different flow regimes"""
    
    def __init__(self, flow_model: str = "RANS"):
        """
        Initialize physics-aware mesh generator
        
        Args:
            flow_model: Flow regime ('LAMINAR', 'RANS', or 'LES')
        """
        self.flow_model = flow_model.upper()
        
        # Blood properties at 37°C
        self.blood_properties = {
            "density": 1060,  # kg/m³
            "kinematic_viscosity": 3.3e-6,  # m²/s
            "dynamic_viscosity": 3.5e-3  # Pa·s
        }
        
    def calculate_first_layer_thickness(self, peak_velocity: float, diameter: float, 
                                      target_yplus: float = 1.0) -> Dict:
        """
        Calculate first layer thickness for target y+ value
        
        Args:
            peak_velocity: Peak velocity in m/s
            diameter: Characteristic diameter in m
            target_yplus: Target y+ value
            
        Returns:
            Dictionary with calculated parameters
        """
        nu = self.blood_properties["kinematic_viscosity"]
        
        # Calculate Reynolds number
        Re = peak_velocity * diameter / nu
        
        # Skin friction coefficient (turbulent flow approximation)
        Cf = 0.079 * (Re ** -0.25)
        
        # Friction velocity
        u_tau = math.sqrt(0.5 * Cf) * peak_velocity
        
        # First layer height for target y+
        h1 = target_yplus * nu / u_tau
        
        # Calculate recommended parameters
        results = {
            "reynolds": Re,
            "friction_coefficient": Cf,
            "friction_velocity": u_tau,
            "first_layer_thickness": h1,
            "first_layer_microns": h1 * 1e6,
            "target_yplus": target_yplus
        }
        
        return results
    
    def get_flow_regime_parameters(self, diameter: float, peak_velocity: float) -> Dict:
        """
        Get recommended mesh parameters for the specified flow regime
        
        Args:
            diameter: Inlet diameter in meters
            peak_velocity: Peak velocity in m/s
            
        Returns:
            Dictionary with mesh parameters optimized for flow regime
        """
        
        if self.flow_model == "LAMINAR":
            return self._get_laminar_parameters(diameter, peak_velocity)
        elif self.flow_model == "RANS":
            return self._get_rans_parameters(diameter, peak_velocity)
        elif self.flow_model == "LES":
            return self._get_les_parameters(diameter, peak_velocity)
        else:
            raise ValueError(f"Unknown flow model: {self.flow_model}")
    
    def _get_laminar_parameters(self, diameter: float, peak_velocity: float) -> Dict:
        """Laminar flow mesh parameters"""
        
        # Calculate first layer for y+ ≈ 1
        layer_calc = self.calculate_first_layer_thickness(peak_velocity, diameter, 1.0)
        
        # Use slightly thicker first layer for laminar (y+ < 1 is fine)
        first_layer = max(layer_calc["first_layer_thickness"], 70e-6)  # At least 70 µm
        
        return {
            "flow_model": "LAMINAR",
            "target_yplus": 1.0,
            "first_layer_thickness": first_layer,
            "n_surface_layers": 10,
            "expansion_ratio": 1.25,
            "base_cell_target": diameter / 45,  # D/40-D/50
            "surface_refinement_level": [2, 3],
            "target_cells_range": [2e6, 5e6],
            "distance_refinement": {
                "near_distance": 1.5e-3,  # 1.5 mm
                "far_distance": 3.0e-3,   # 3.0 mm
                "transition_distance": 6.0e-3
            },
            "reynolds": layer_calc["reynolds"],
            "calculated_h1_microns": layer_calc["first_layer_microns"]
        }
    
    def _get_rans_parameters(self, diameter: float, peak_velocity: float) -> Dict:
        """RANS flow mesh parameters"""
        
        # Calculate first layer for y+ ≈ 1
        layer_calc = self.calculate_first_layer_thickness(peak_velocity, diameter, 1.0)
        
        # Use calculated first layer or minimum 50 µm
        first_layer = max(layer_calc["first_layer_thickness"], 50e-6)
        
        return {
            "flow_model": "RANS", 
            "target_yplus": 1.0,
            "first_layer_thickness": first_layer,
            "n_surface_layers": 12,
            "expansion_ratio": 1.20,
            "base_cell_target": diameter / 55,  # D/50-D/60
            "surface_refinement_level": [3, 4],
            "target_cells_range": [5e6, 10e6],
            "distance_refinement": {
                "near_distance": 1.0e-3,  # 1.0 mm - tighter for RANS
                "far_distance": 2.5e-3,   # 2.5 mm 
                "transition_distance": 5.0e-3
            },
            "reynolds": layer_calc["reynolds"],
            "calculated_h1_microns": layer_calc["first_layer_microns"]
        }
    
    def _get_les_parameters(self, diameter: float, peak_velocity: float) -> Dict:
        """Wall-resolved LES mesh parameters"""
        
        # Calculate first layer for y+ ≈ 1-2
        layer_calc = self.calculate_first_layer_thickness(peak_velocity, diameter, 1.5)
        
        # Use thinner first layer for LES wall resolution
        first_layer = min(layer_calc["first_layer_thickness"], 40e-6)  # Max 40 µm
        
        return {
            "flow_model": "LES",
            "target_yplus": 1.5,
            "first_layer_thickness": first_layer, 
            "n_surface_layers": 20,
            "expansion_ratio": 1.15,
            "base_cell_target": diameter / 100,  # D/80-D/120
            "surface_refinement_level": [4, 5],
            "target_cells_range": [20e6, 60e6],
            "distance_refinement": {
                "near_distance": 0.8e-3,  # 0.8 mm - very tight for LES
                "far_distance": 2.0e-3,   # 2.0 mm
                "transition_distance": 4.0e-3
            },
            "on_wall_spacing_target": 2e-3,  # 2 mm for Δx+ ≈ 40-80
            "reynolds": layer_calc["reynolds"],
            "calculated_h1_microns": layer_calc["first_layer_microns"]
        }
    
    def validate_mesh_for_physics(self, mesh_params: Dict, flow_conditions: Dict) -> Dict:
        """
        Validate mesh parameters against physics requirements
        
        Args:
            mesh_params: Generated mesh parameters 
            flow_conditions: Flow conditions (velocity, diameter, etc.)
            
        Returns:
            Validation results with warnings and recommendations
        """
        
        diameter = flow_conditions["diameter"]
        velocity = flow_conditions["peak_velocity"]
        
        validation = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check Reynolds number regime
        Re = velocity * diameter / self.blood_properties["kinematic_viscosity"]
        
        if self.flow_model == "LAMINAR" and Re > 2300:
            validation["warnings"].append(f"Re={Re:.0f} may be transitional/turbulent for laminar model")
        elif self.flow_model == "RANS" and Re < 2000:
            validation["warnings"].append(f"Re={Re:.0f} may be laminar - consider LAMINAR model")
        
        # Check first layer thickness
        h1_microns = mesh_params["first_layer_thickness"] * 1e6
        
        if self.flow_model == "LAMINAR" and h1_microns > 90:
            validation["warnings"].append(f"First layer {h1_microns:.1f}µm may be too thick for laminar WSS")
        elif self.flow_model == "LES" and h1_microns > 50:
            validation["warnings"].append(f"First layer {h1_microns:.1f}µm may be too thick for LES y+≈1")
        
        # Check surface refinement vs on-wall spacing
        if self.flow_model == "LES":
            # Estimate on-wall spacing from surface refinement
            base_size = mesh_params["base_cell_target"] 
            surf_level = mesh_params["surface_refinement_level"][1]
            on_wall_spacing = base_size / (2 ** surf_level)
            
            if on_wall_spacing > 3e-3:  # 3mm
                validation["recommendations"].append(f"On-wall spacing {on_wall_spacing*1000:.1f}mm may be too coarse for LES")
        
        return validation
    
    def generate_physics_aware_config(self, geometry_params: Dict, 
                                    output_file: Path = None) -> Dict:
        """
        Generate complete physics-aware configuration
        
        Args:
            geometry_params: Geometry parameters (diameter, velocity, etc.)
            output_file: Optional output file path
            
        Returns:
            Complete configuration dictionary
        """
        
        diameter = geometry_params["diameter"]
        velocity = geometry_params["peak_velocity"]
        
        # Get flow regime parameters
        mesh_params = self.get_flow_regime_parameters(diameter, velocity)
        
        # Validate parameters
        validation = self.validate_mesh_for_physics(mesh_params, geometry_params)
        
        # Create complete configuration
        config = {
            "description": f"Physics-aware mesh configuration for {self.flow_model}",
            "flow_model": self.flow_model,
            "geometry_params": geometry_params,
            "mesh_parameters": mesh_params,
            "validation": validation,
            "openfoam_env_path": "/opt/openfoam12/etc/bashrc"
        }
        
        # Save configuration if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config

# Convenience function for quick calculations
def calculate_aortic_first_layer(peak_velocity: float, diameter_mm: float, 
                                flow_model: str = "RANS") -> Dict:
    """
    Quick calculation of first layer thickness for aortic flow
    
    Args:
        peak_velocity: Peak velocity in m/s
        diameter_mm: Diameter in millimeters 
        flow_model: Flow model ('LAMINAR', 'RANS', 'LES')
        
    Returns:
        Dictionary with recommended parameters
    """
    
    generator = PhysicsAwareMeshGenerator(flow_model)
    diameter_m = diameter_mm * 1e-3
    
    return generator.get_flow_regime_parameters(diameter_m, peak_velocity)