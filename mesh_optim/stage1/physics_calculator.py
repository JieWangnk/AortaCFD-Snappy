"""
Physics-based calculations for CFD mesh generation.
Handles y+, Womersley, Reynolds number, and boundary layer computations.
"""
import math
import logging
from typing import Dict, Tuple, Optional

from .constants import DEFAULT_CONSTANTS
from ..utils import calculate_first_layer_thickness

logger = logging.getLogger(__name__)

class PhysicsCalculator:
    """Physics-aware calculations for mesh generation"""
    
    def __init__(self, config_manager):
        self.config = config_manager.config
        self.constants = DEFAULT_CONSTANTS['physics']
    
    def calculate_yplus_first_layer(self, diameter: float, peak_velocity: float, 
                                  target_yplus: float = 1.0, 
                                  flow_model: str = "turbulent") -> float:
        """
        Calculate first layer thickness for target y+
        
        Args:
            diameter: Reference diameter [m]
            peak_velocity: Peak flow velocity [m/s] 
            target_yplus: Target y+ value
            flow_model: "turbulent" or "laminar"
            
        Returns:
            First layer thickness [m]
        """
        physics = self.config.get("physics", {})
        
        # Get fluid properties
        rho = float(physics.get("rho", self.constants.BLOOD_DENSITY_DEFAULT))
        mu = float(physics.get("mu", self.constants.BLOOD_VISCOSITY_DEFAULT))
        
        # Use robust implementation from utils
        blood_properties = {'density': rho, 'viscosity': mu}
        base_thickness = calculate_first_layer_thickness(peak_velocity, diameter, blood_properties)
        
        # Scale by target y+ (utils calculates for y+=1)
        first_layer = base_thickness * target_yplus
        
        # Apply minimum thickness for numerical stability
        min_thickness = self.constants.MIN_FIRST_LAYER_MICRONS * 1e-6
        first_layer = max(first_layer, min_thickness)
        
        # Log calculation details
        Re = rho * peak_velocity * diameter / max(mu, 1e-9)
        logger.info(f"y+ based sizing: Re={Re:.0f}, y+={target_yplus}, "
                   f"model={flow_model} → {first_layer*1e6:.1f} μm")
        
        return first_layer
    
    def calculate_womersley_boundary_layer(self, heart_rate_hz: Optional[float] = None) -> float:
        """
        Calculate Womersley boundary layer thickness for pulsatile flow
        δ_ω = √(2ν/ω) where ω = 2πf
        
        Args:
            heart_rate_hz: Heart rate [Hz], defaults to config value
            
        Returns:
            Womersley boundary layer thickness [m]
        """
        physics = self.config.get("physics", {})
        
        if heart_rate_hz is None:
            heart_rate_hz = float(physics.get("heart_rate_hz", self.constants.HEART_RATE_DEFAULT_HZ))
        
        # Get fluid properties
        rho = float(physics.get("rho", self.constants.BLOOD_DENSITY_DEFAULT))
        mu = float(physics.get("mu", self.constants.BLOOD_VISCOSITY_DEFAULT))
        nu = mu / rho  # Kinematic viscosity
        
        # Calculate Womersley thickness
        omega = 2.0 * math.pi * max(heart_rate_hz, 1e-6)
        delta = math.sqrt(self.constants.WOMERSLEY_SCALING_FACTOR * nu / omega)
        
        logger.info(f"Womersley boundary layer: f={heart_rate_hz:.2f} Hz, "
                   f"ν={nu:.2e} m²/s → δ_ω={delta*1e3:.2f} mm")
        
        return delta
    
    def calculate_reynolds_number(self, diameter: float, velocity: float) -> float:
        """Calculate Reynolds number for flow characterization"""
        physics = self.config.get("physics", {})
        rho = float(physics.get("rho", self.constants.BLOOD_DENSITY_DEFAULT))
        mu = float(physics.get("mu", self.constants.BLOOD_VISCOSITY_DEFAULT))
        
        return rho * velocity * diameter / max(mu, 1e-9)
    
    def classify_flow_regime(self, reynolds_number: float) -> str:
        """Classify flow as laminar, transitional, or turbulent"""
        if reynolds_number < self.constants.RE_LAMINAR_THRESHOLD:
            return "laminar"
        elif reynolds_number < self.constants.RE_TRANSITIONAL_MAX:
            return "transitional"
        else:
            return "turbulent"
    
    def get_recommended_yplus(self, solver_mode: str) -> float:
        """Get recommended y+ for solver type"""
        solver_mode = solver_mode.upper()
        
        if solver_mode == "LES":
            return self.constants.Y_PLUS_LES
        elif solver_mode == "RANS":
            return self.constants.Y_PLUS_RANS
        else:  # Laminar
            return 1.0  # Near-wall resolution for laminar
    
    def calculate_layer_parameters(self, diameter: float, velocity: float, 
                                 base_cell_size: float) -> Dict[str, float]:
        """
        Calculate comprehensive boundary layer parameters
        
        Returns:
            Dictionary with layer thickness, count, expansion ratio, etc.
        """
        physics = self.config.get("physics", {})
        solver_mode = physics.get("solver_mode", "RANS")
        
        # Get target y+ for solver
        target_yplus = self.get_recommended_yplus(solver_mode)
        
        # Calculate first layer thickness
        first_layer = self.calculate_yplus_first_layer(diameter, velocity, target_yplus)
        
        # Calculate boundary layer thickness for context
        if physics.get("use_womersley_bands", False):
            bl_thickness = self.calculate_womersley_boundary_layer()
        else:
            # Estimate turbulent BL thickness: δ ≈ 0.38 * x * Re_x^(-1/5)
            # For pipe flow approximation: δ ≈ 0.1 * diameter
            bl_thickness = 0.1 * diameter
        
        # Determine layer count and expansion ratio
        n_layers, expansion_ratio = self._optimize_layer_distribution(
            first_layer, bl_thickness, solver_mode
        )
        
        return {
            'firstLayerThickness_abs': first_layer,
            'minThickness_abs': first_layer * 0.15,  # 15% rule
            'nSurfaceLayers': n_layers,
            'expansionRatio': expansion_ratio,
            'nGrow': 1,  # Allow growth across curvature
            'target_yplus': target_yplus,
            'boundary_layer_thickness': bl_thickness
        }
    
    def _optimize_layer_distribution(self, first_layer: float, 
                                   bl_thickness: float, 
                                   solver_mode: str) -> Tuple[int, float]:
        """Optimize number of layers and expansion ratio"""
        constants = DEFAULT_CONSTANTS['layers']
        
        if solver_mode.upper() == "LES":
            # LES: More layers, gentler expansion for near-wall resolution
            target_layers = 16
            expansion_ratio = 1.15
        elif solver_mode.upper() == "RANS": 
            # RANS: Moderate layers for wall function compatibility
            target_layers = 10
            expansion_ratio = 1.2
        else:  # Laminar
            # Laminar: Fewer layers, capture viscous sublayer
            target_layers = 8
            expansion_ratio = 1.3
        
        # Validate that layers fit within boundary layer
        total_thickness = first_layer * (expansion_ratio**target_layers - 1) / (expansion_ratio - 1)
        
        # Adjust if layers extend too far into free stream
        if total_thickness > 0.3 * bl_thickness:
            # Reduce layer count to stay within 30% of BL
            import math
            max_total = 0.3 * bl_thickness
            adjusted_layers = int(math.log(1 + (expansion_ratio - 1) * max_total / first_layer) / math.log(expansion_ratio))
            target_layers = max(5, min(adjusted_layers, target_layers))
            
            logger.debug(f"Adjusted layer count: {adjusted_layers} to fit BL constraint")
        
        # Apply hard limits
        n_layers = max(3, min(20, target_layers))
        expansion_ratio = max(1.1, min(2.0, expansion_ratio))
        
        return n_layers, expansion_ratio
    
    def calculate_refinement_bands(self, base_size: float, 
                                 use_womersley: bool = False) -> Tuple[float, float]:
        """Calculate near and far refinement band distances"""
        if use_womersley:
            # Physics-based bands using Womersley boundary layer
            delta_w = self.calculate_womersley_boundary_layer()
            near_dist = 3.0 * delta_w   # 3× Womersley thickness
            far_dist = 15.0 * delta_w   # 15× Womersley thickness
            
            logger.info(f"Womersley refinement bands: near={near_dist*1e6:.0f}μm, "
                       f"far={far_dist*1e6:.0f}μm")
        else:
            # Geometry-based bands using cell size multiples
            geom_constants = DEFAULT_CONSTANTS['geometry']
            near_dist = geom_constants.NEAR_BAND_FACTOR * base_size
            far_dist = geom_constants.FAR_BAND_FACTOR * base_size
            
            logger.info(f"Geometry-based bands: near={near_dist*1e3:.1f}mm, "
                       f"far={far_dist*1e3:.1f}mm")
        
        return near_dist, far_dist