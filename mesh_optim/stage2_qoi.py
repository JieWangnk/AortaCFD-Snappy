"""
Stage 2: QoI-driven mesh optimization (outer loop)

This module handles physics-aware mesh optimization based on flow regime requirements.
It uses CFD simulations to validate mesh quality for velocity, pressure, and WSS accuracy.
"""

import json
import numpy as np
from pathlib import Path
import shutil
import logging
import math
from .utils import run_command, check_mesh_quality, parse_layer_coverage
from .stage1_mesh import Stage1MeshOptimizer
from .cfd_solver import CFDSolver
from .physics_mesh import PhysicsAwareMeshGenerator

class Stage2QOIOptimizer:
    """QoI-driven mesh optimizer for laminar/RANS/LES"""
    
    def __init__(self, geometry_dir, flow_model, config_file=None, output_dir=None):
        """
        Initialize Stage 2 QoI optimizer
        
        Args:
            geometry_dir: Path to directory containing STL files
            flow_model: Flow model ('LAMINAR', 'RANS', or 'LES')
            config_file: Path to configuration JSON file (optional)
            output_dir: Output directory (default: geometry_dir/../output)
        """
        self.geometry_dir = Path(geometry_dir)
        self.flow_model = flow_model.upper()
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Stage 2 should be under patient's output directory
            patient_name = self.geometry_dir.name
            self.output_dir = Path("output") / patient_name / f"stage2_{self.flow_model.lower()}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging first
        self.logger = logging.getLogger(f"Stage2QOI_{self.geometry_dir.name}_{flow_model}")
        
        # Initialize physics-aware mesh generator
        self.physics_generator = PhysicsAwareMeshGenerator(flow_model)
        
        # Load or create configuration
        if config_file:
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            self.config = self._create_physics_aware_config()
        
        # Initialize parameters
        self.current_iteration = 0
        self.max_iterations = 3
        self.best_qoi_score = 0
        
        # Flow physics parameters
        self.blood_properties = {
            "density": 1060,  # kg/m³
            "kinematic_viscosity": 3.3e-6  # m²/s at 37°C
        }
        
        # Find STL files
        self.stl_files = self._discover_stl_files()
        
        # Get physics-aware mesh parameters with automatic geometry analysis
        self.mesh_params = self.physics_generator.get_flow_regime_parameters(
            geometry_dir=self.geometry_dir
        )
        
        # Extract geometry parameters for compatibility
        self.geometry_params = {
            "inlet_diameter": self.mesh_params.get("minimum_diameter", 25e-3),
            "peak_velocity": self.mesh_params.get("peak_velocity", 1.0),
            "reynolds": self.mesh_params.get("reynolds", 7000)
        }
        
        # Communicate expectations to user
        self._print_physics_summary()
        
        # Initialize CFD solver
        self.cfd_solver = CFDSolver(
            flow_model=self.flow_model,
            openfoam_env=self.config["openfoam_env_path"]
        )
        
    def _create_physics_aware_config(self):
        """Create physics-aware configuration based on flow model"""
        
        # Base configuration
        config = {
            "description": f"Stage 2: QoI-driven mesh optimization for {self.flow_model}",
            "flow_model": self.flow_model,
            "openfoam_env_path": "/opt/openfoam12/etc/bashrc"
        }
        
        if self.flow_model == "LAMINAR":
            config.update({
                "target_yplus": 1.0,
                "first_layer_thickness": 8.0e-5,  # 80 µm
                "n_surface_layers": 10,
                "expansion_ratio": 1.25,
                "base_cell_target": "D/45",  # D/40-D/50
                "surface_refinement_level": [2, 3],
                "target_cells": [2e6, 5e6],  # 2-5M cells
                "max_iterations": 3
            })
        elif self.flow_model == "RANS":
            config.update({
                "target_yplus": 1.0,
                "first_layer_thickness": 5.0e-5,  # 50 µm 
                "n_surface_layers": 12,
                "expansion_ratio": 1.20,
                "base_cell_target": "D/55",  # D/50-D/60
                "surface_refinement_level": [3, 4],
                "target_cells": [5e6, 10e6],  # 5-10M cells
                "max_iterations": 4
            })
        elif self.flow_model == "LES":
            config.update({
                "target_yplus": 1.5,
                "first_layer_thickness": 3.5e-5,  # 35 µm
                "n_surface_layers": 20,
                "expansion_ratio": 1.15,
                "base_cell_target": "D/100",  # D/80-D/120
                "surface_refinement_level": [4, 5],
                "target_cells": [20e6, 60e6],  # 20-60M cells
                "max_iterations": 5
            })
        
        # QoI acceptance criteria
        config["qoi_criteria"] = {
            "yplus_coverage": 0.90,  # 90% of wall area with 0.5 ≤ y+ ≤ 2
            "yplus_range": [0.5, 2.0],
            "velocity_convergence": 0.02,  # 2% change in centerline velocity
            "flow_split_convergence": 0.02,  # 2% change in branch flow split
            "wss_convergence": 0.05  # 5% change in area-averaged WSS
        }
        
        # Distance refinement (fixed units - in meters!)
        config["distance_refinement"] = {
            "near_distance": 1.5e-3,  # 1.5 mm in meters
            "far_distance": 3.0e-3,   # 3.0 mm in meters
            "transition_distance": 6.0e-3  # 6 mm transition
        }
        
        return config
    
    def _discover_stl_files(self):
        """Discover and validate STL files"""
        required_files = ["inlet.stl", "wall_aorta.stl"]
        stl_files = {"required": {}, "outlets": []}
        
        # Check required files
        for req_file in required_files:
            stl_path = self.geometry_dir / req_file
            if not stl_path.exists():
                raise FileNotFoundError(f"Required STL file not found: {req_file}")
            stl_files["required"][req_file.split('.')[0]] = stl_path
        
        # Find outlet files
        for stl_path in self.geometry_dir.glob("outlet*.stl"):
            stl_files["outlets"].append(stl_path)
        
        if not stl_files["outlets"]:
            raise FileNotFoundError("No outlet STL files found")
            
        self.logger.info(f"Found {len(stl_files['outlets'])} outlet files for {self.flow_model} optimization")
        return stl_files
    
    def _estimate_geometry_parameters(self):
        """Estimate geometry parameters for physics calculations"""
        # For now, use typical aortic dimensions
        # TODO: Implement actual STL parsing
        params = {
            "inlet_diameter": 0.025,  # 25 mm typical
            "peak_velocity": 1.0,     # 1.0 m/s typical
            "characteristic_length": 0.094,  # ~94 mm total length
        }
        
        # Calculate Reynolds number
        params["reynolds"] = (params["peak_velocity"] * params["inlet_diameter"] / 
                             self.blood_properties["kinematic_viscosity"])
        
        # Calculate friction velocity and first layer thickness
        cf = 0.079 * (params["reynolds"] ** -0.25)  # Turbulent approximation
        u_tau = math.sqrt(0.5 * cf) * params["peak_velocity"]
        params["friction_velocity"] = u_tau
        params["calculated_first_layer"] = self.blood_properties["kinematic_viscosity"] / u_tau
        
        self.logger.info(f"Estimated geometry: D={params['inlet_diameter']*1000:.1f}mm, "
                        f"U={params['peak_velocity']:.1f}m/s, Re={params['reynolds']:.0f}")
        self.logger.info(f"Calculated first layer thickness: {params['calculated_first_layer']*1e6:.1f}µm "
                        f"for y+≈{self.config['target_yplus']}")
        
        return params

    def iterate_until_converged(self):
        """Main QoI-driven iteration loop"""
        
        self.logger.info(f"Starting Stage 2 QoI optimization for {self.flow_model}")
        self.logger.info(f"Target: Physics-accurate mesh for velocity, pressure, and WSS")
        
        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration
            self.logger.info(f"=== QoI ITERATION {iteration}/{self.max_iterations} ===")
            
            # Create iteration directory
            iter_dir = self.output_dir / f"qoi_iter_{iteration:03d}"
            iter_dir.mkdir(exist_ok=True)
            
            # Generate physics-aware mesh using Stage 1 optimizer
            mesh_result = self._generate_physics_aware_mesh(iter_dir)
            
            if not mesh_result["success"]:
                self.logger.error(f"❌ Mesh generation failed in iteration {iteration}")
                continue
            
            # Evaluate QoI metrics (placeholder for CFD simulation)
            qoi_metrics = self._evaluate_qoi(iter_dir, mesh_result)
            
            # Check convergence
            converged = self._check_qoi_convergence(qoi_metrics)
            
            # Save iteration results
            self._save_qoi_metrics(iter_dir, qoi_metrics, mesh_result)
            
            if converged:
                self.logger.info(f"✅ QoI optimization converged in {iteration} iterations")
                return iter_dir, qoi_metrics
            
            # Adjust parameters for next iteration
            self._adjust_for_qoi(qoi_metrics)
        
        self.logger.warning(f"⚠️ QoI optimization incomplete after {self.max_iterations} iterations")
        return iter_dir, qoi_metrics
    
    def _generate_physics_aware_mesh(self, iter_dir):
        """Generate physics-aware mesh using Stage 1 optimizer with QoI parameters"""
        
        try:
            # Create physics-aware configuration
            stage1_config = self._create_stage1_config_from_qoi()
            
            # Save temporary config
            temp_config = iter_dir / "stage1_physics.json"
            with open(temp_config, 'w') as f:
                json.dump(stage1_config, f, indent=2)
            
            # Initialize Stage 1 optimizer with physics-aware config
            stage1_optimizer = Stage1MeshOptimizer(
                self.geometry_dir,
                temp_config,
                iter_dir / "stage1_mesh"
            )
            
            # Override mesh parameters with QoI requirements
            stage1_optimizer.max_iterations = 1  # Single iteration for Stage 2
            
            self.logger.info(f"Generating {self.flow_model} mesh with physics parameters:")
            self.logger.info(f"  First layer: {self.config['first_layer_thickness']*1e6:.1f}µm")
            self.logger.info(f"  Layers: {self.config['n_surface_layers']}")
            self.logger.info(f"  Surface refinement: {self.config['surface_refinement_level']}")
            
            # Generate single optimized mesh
            result_dir = stage1_optimizer.iterate_until_quality()
            
            # Check mesh quality
            mesh_metrics = check_mesh_quality(result_dir, self.config["openfoam_env_path"])
            layer_coverage = parse_layer_coverage(result_dir, self.config["openfoam_env_path"])
            
            return {
                "success": True,
                "mesh_dir": result_dir,
                "mesh_metrics": mesh_metrics,
                "layer_coverage": layer_coverage,
                "cells": mesh_metrics.get("cells", 0),
                "max_skewness": mesh_metrics.get("maxSkewness", 0),
                "max_nonortho": mesh_metrics.get("maxNonOrtho", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_stage1_config_from_qoi(self):
        """Create Stage 1 configuration using physics-aware parameters"""
        
        # Use physics generator parameters directly
        params = self.mesh_params
        
        # Calculate base cell size from physics parameters
        base_cell_size = params.get("base_cell_target", 0.5e-3)
        
        # Estimate blocks needed
        bbox_size = 0.094  # ~94mm aorta length
        resolution = int(bbox_size / (base_cell_size))
        
        config = {
            "description": f"Stage 1 configuration for {self.flow_model} QoI optimization",
            
            "BLOCKMESH": {
                "resolution": max(resolution, 40),  # Minimum 40 for stability
                "grading": [1, 1, 1]
            },
            
            "SNAPPY": {
                "surface_level": params.get("surface_refinement_level", [3, 4]),
                "maxLocalCells": 10000000,
                "maxGlobalCells": 50000000,
                "nCellsBetweenLevels": 2,
                "resolveFeatureAngle": 150,
                "nSmoothPatch": 3,
                "nFeatureSnapIter": 10,
                "implicitFeatureSnap": False,
                "explicitFeatureSnap": True,
                "distance_refinement": {
                    "near_distance": params.get("distance_refinement", {}).get("near_distance", 1.5e-3) * 1000,  # Convert to mm
                    "far_distance": params.get("distance_refinement", {}).get("far_distance", 3.0e-3) * 1000
                }
            },
            
            "LAYERS": {
                "enable": True,
                "nSurfaceLayers": params.get("n_surface_layers", 12),
                "finalLayerThickness_rel": params.get("first_layer_thickness", 50e-6) / base_cell_size,
                "expansionRatio": params.get("expansion_ratio", 1.12),
                "nGrow": 1,
                "minThickness": params.get("first_layer_thickness", 50e-6) * 0.4,
                "featureAngle": 130,
                "maxThicknessToMedialRatio": 0.6,
                "minMedianAxisAngle": 50
            },
            
            "MESH_QUALITY": {
                "snap": {
                    "maxNonOrtho": 75,
                    "maxBoundarySkewness": 4.5,
                    "maxInternalSkewness": 4.5,
                    "minVol": 1e-18,
                    "minTetQuality": 1e-15,
                    "minFaceWeight": 0.01
                },
                "layer": {
                    "maxNonOrtho": 70,
                    "maxBoundarySkewness": 4.5,
                    "maxInternalSkewness": 4.5,
                    "minVol": 1e-18,
                    "minTetQuality": 1e-15,
                    "minFaceWeight": 0.01
                }
            },
            
            "acceptance_criteria": {
                "maxNonOrtho": 75,  # More relaxed for QoI focus
                "maxSkewness": 4.5,
                "min_layer_coverage": 0.70  # Focus on getting layers
            },
            
            "max_iterations": 1,
            "openfoam_env_path": self.config["openfoam_env_path"]
        }
        
        return config
    
    def _evaluate_qoi(self, iter_dir, mesh_result):
        """Evaluate QoI metrics using actual CFD simulation"""
        
        mesh_metrics = mesh_result["mesh_metrics"]
        layer_coverage = mesh_result["layer_coverage"]
        mesh_dir = mesh_result["mesh_dir"]
        
        # Basic mesh quality assessment
        layer_coverage_pct = layer_coverage.get("coverage_overall", 0.0)
        cells = mesh_metrics.get("cells", 0)
        max_skew = mesh_metrics.get("maxSkewness", 10)
        max_nonortho = mesh_metrics.get("maxNonOrtho", 100)
        
        # QoI evaluation based on physics requirements
        target_cells = self.config["target_cells"]
        cells_in_range = target_cells[0] <= cells <= target_cells[1]
        
        # Initial mesh quality check
        physics_compliance = {
            "first_layer_achieved": layer_coverage_pct > 0.6,
            "resolution_adequate": cells_in_range,
            "quality_acceptable": max_skew < 5.0 and max_nonortho < 80,
            "mesh_valid": mesh_metrics.get("meshOK", False)
        }
        
        qoi_metrics = {
            "mesh_quality": {
                "cells": cells,
                "cells_in_target_range": cells_in_range,
                "max_skewness": max_skew,
                "max_nonortho": max_nonortho,
                "mesh_valid": mesh_metrics.get("meshOK", False)
            },
            "layer_analysis": {
                "coverage_overall": layer_coverage_pct,
                "layer_success": layer_coverage_pct > 0.5
            },
            "physics_compliance": physics_compliance,
            "cfd_results": {},
            "qoi_ready": False,
            "converged": False,
            "valid": mesh_metrics.get("meshOK", False)
        }
        
        # Only run CFD if mesh quality is acceptable and CFD solver is enabled
        mesh_suitable_for_cfd = all(physics_compliance.values())
        cfd_enabled = self.cfd_solver is not None
        
        if mesh_suitable_for_cfd and cfd_enabled:
            self.logger.info("🔥 Mesh quality acceptable - running CFD simulation")
            
            # Prepare flow conditions
            flow_conditions = {
                "peak_velocity": self.geometry_params["peak_velocity"],
                "diameter": self.geometry_params["inlet_diameter"]
            }
            
            try:
                # Setup CFD case
                cfd_setup_success = self.cfd_solver.setup_cfd_case(mesh_dir, flow_conditions)
                
                if cfd_setup_success:
                    # Run CFD simulation
                    cfd_case_dir = mesh_dir / "cfd_case"
                    simulation_result = self.cfd_solver.run_cfd_simulation(cfd_case_dir)
                    
                    if simulation_result["success"]:
                        # Extract QoI metrics
                        cfd_qoi = self.cfd_solver.extract_qoi_metrics(cfd_case_dir, self.geometry_params)
                        qoi_metrics["cfd_results"] = cfd_qoi
                        
                        # Update QoI status based on CFD results
                        qoi_metrics["qoi_ready"] = cfd_qoi.get("valid", False)
                        qoi_metrics["converged"] = cfd_qoi.get("converged", False) and cells_in_range
                        
                        self.logger.info(f"✅ CFD simulation completed - QoI valid: {cfd_qoi.get('valid', False)}")
                        if cfd_qoi.get("wss_analysis", {}).get("available"):
                            wss = cfd_qoi["wss_analysis"]
                            self.logger.info(f"  WSS: {wss.get('mean_wss', 0):.1f}±{wss.get('max_wss', 0)-wss.get('min_wss', 0):.1f} Pa")
                        if cfd_qoi.get("yplus_analysis", {}).get("available"):
                            yplus = cfd_qoi["yplus_analysis"]
                            self.logger.info(f"  y+: {yplus.get('mean_yplus', 0):.1f} (coverage: {yplus.get('coverage_in_range', 0)*100:.1f}%)")
                    else:
                        self.logger.error("❌ CFD simulation failed")
                        qoi_metrics["cfd_results"]["error"] = simulation_result.get("error", "Unknown CFD error")
                else:
                    self.logger.error("❌ CFD case setup failed")
                    qoi_metrics["cfd_results"]["error"] = "CFD setup failed"
                    
            except Exception as e:
                self.logger.error(f"❌ CFD evaluation failed: {e}")
                qoi_metrics["cfd_results"]["error"] = str(e)
        else:
            if not cfd_enabled:
                self.logger.info("⚠️ CFD solver disabled - using mesh-based estimates only")
            else:
                self.logger.info("⚠️ Mesh quality insufficient for CFD - skipping simulation")
            
            # Fallback to mesh-based estimates
            qoi_metrics["layer_analysis"]["estimated_yplus_coverage"] = min(layer_coverage_pct * 1.5, 1.0)
            qoi_metrics["qoi_ready"] = mesh_suitable_for_cfd  # Ready if mesh is good, even without CFD
            qoi_metrics["converged"] = mesh_suitable_for_cfd and cells_in_range
        
        return qoi_metrics
    
    def _check_qoi_convergence(self, qoi_metrics):
        """Check if QoI criteria are met"""
        
        criteria = self.config["qoi_criteria"]
        physics = qoi_metrics["physics_compliance"]
        layer = qoi_metrics["layer_analysis"]
        cfd_results = qoi_metrics.get("cfd_results", {})
        
        checks = {
            "physics_compliant": all(physics.values()),
            "layers_successful": layer["layer_success"],
            "mesh_valid": qoi_metrics.get("valid", False),
            "qoi_ready": qoi_metrics.get("qoi_ready", False)
        }
        
        # Additional CFD-based checks if available
        if cfd_results and not cfd_results.get("error"):
            yplus = cfd_results.get("yplus_analysis", {})
            if yplus.get("available"):
                checks["yplus_acceptable"] = yplus.get("coverage_acceptable", False)
            
            wss = cfd_results.get("wss_analysis", {})
            if wss.get("available"):
                checks["wss_available"] = True
            
            velocity = cfd_results.get("velocity_analysis", {})
            if velocity.get("converged"):
                checks["velocity_converged"] = True
        
        all_ok = all(checks.values())
        
        self.logger.info(f"📊 QoI Assessment:")
        self.logger.info(f"  Physics compliance: {'✅' if checks['physics_compliant'] else '❌'}")
        self.logger.info(f"  Layer generation: {'✅' if checks['layers_successful'] else '❌'} ({layer['coverage_overall']*100:.1f}%)")
        self.logger.info(f"  Mesh validity: {'✅' if checks['mesh_valid'] else '❌'}")
        self.logger.info(f"  QoI ready: {'✅' if checks['qoi_ready'] else '❌'}")
        
        # CFD-specific checks
        if "yplus_acceptable" in checks:
            self.logger.info(f"  y+ coverage: {'✅' if checks['yplus_acceptable'] else '❌'}")
        if "wss_available" in checks:
            self.logger.info(f"  WSS calculation: {'✅' if checks['wss_available'] else '❌'}")
        if "velocity_converged" in checks:
            self.logger.info(f"  Velocity convergence: {'✅' if checks['velocity_converged'] else '❌'}")
        
        return all_ok
    
    def _adjust_for_qoi(self, qoi_metrics):
        """Adjust parameters based on QoI feedback"""
        
        physics = qoi_metrics["physics_compliance"]
        cfd_results = qoi_metrics.get("cfd_results", {})
        
        # Basic mesh adjustments
        if not physics["first_layer_achieved"]:
            # Reduce first layer thickness for better success
            self.config["first_layer_thickness"] *= 0.8
            self.config["n_surface_layers"] = max(self.config["n_surface_layers"] - 2, 8)
            self.logger.info("🔧 Adjusted layer parameters for better generation")
            
        if not physics["resolution_adequate"]:
            # Adjust surface refinement
            current_level = self.config["surface_refinement_level"]
            self.config["surface_refinement_level"] = [current_level[0], min(current_level[1] + 1, 5)]
            self.logger.info("🔧 Increased surface refinement for target cell count")
        
        # CFD-based adjustments
        if cfd_results and not cfd_results.get("error"):
            yplus = cfd_results.get("yplus_analysis", {})
            if yplus.get("available") and not yplus.get("coverage_acceptable", True):
                # y+ too high - need thinner first layer
                if yplus.get("mean_yplus", 1.0) > 2.0:
                    self.config["first_layer_thickness"] *= 0.7
                    self.logger.info(f"🔧 Reduced first layer thickness for better y+ (current: {yplus.get('mean_yplus', 'N/A')})")
            
            velocity = cfd_results.get("velocity_analysis", {})
            if velocity and not velocity.get("converged", True):
                # Poor velocity field - may need more resolution
                recovery = velocity.get("centerline_velocity", {}).get("inlet_recovery", 1.0)
                if recovery < 0.9:
                    current_level = self.config["surface_refinement_level"]
                    if current_level[1] < 5:
                        self.config["surface_refinement_level"] = [current_level[0], current_level[1] + 1]
                        self.logger.info(f"🔧 Increased refinement for better velocity field (recovery: {recovery:.1%})")
    
    def _save_qoi_metrics(self, iter_dir, qoi_metrics, mesh_result):
        """Save QoI iteration results"""
        
        results = {
            "iteration": self.current_iteration,
            "flow_model": self.flow_model,
            "geometry_params": self.geometry_params,
            "config": self.config,
            "mesh_result": {
                "success": mesh_result["success"],
                "cells": mesh_result.get("cells", 0),
                "max_skewness": mesh_result.get("max_skewness", 0),
                "max_nonortho": mesh_result.get("max_nonortho", 0)
            },
            "qoi_metrics": qoi_metrics,
            "timestamp": str(iter_dir.name),
            "cfd_solver_config": {
                "flow_model": self.flow_model,
                "cfd_enabled": self.cfd_solver is not None,
                "blood_properties": self.cfd_solver.blood_properties if self.cfd_solver else None,
                "solver_settings": self.cfd_solver.solver_configs.get(self.flow_model, {}) if self.cfd_solver else {}
            }
        }
        
        results_file = iter_dir / "qoi_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"📝 QoI results saved to: {results_file}")
    
    def _load_physics_profile(self, flow_model: str) -> dict:
        """Load physics profile from configuration file"""
        
        profiles_file = Path(__file__).parent / "physics_profiles.json"
        
        try:
            with open(profiles_file) as f:
                profiles = json.load(f)
            
            if flow_model not in profiles:
                raise ValueError(f"Unknown flow model: {flow_model}. Available: {list(profiles.keys())}")
            
            profile = profiles[flow_model]
            self.logger.info(f"📊 Loaded {profile['name']}: {profile['description']}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to load physics profiles: {e}")
            # Fallback to basic configuration
            return {
                "name": f"Default {flow_model}",
                "expected_cells": "Unknown",
                "resolution_factor": 50,
                "surface_refinement_level": [3, 4] if flow_model == "RANS" else [2, 3]
            }
    
    def _print_physics_summary(self):
        """Print comprehensive physics-based summary for user awareness"""
        
        params = self.mesh_params
        D = params.get("minimum_diameter", 25e-3) * 1000  # mm
        Re = params.get("reynolds", 7000)
        flow_regime = params.get("flow_regime", "unknown")
        
        # Physics calculations
        first_layer_um = params.get("first_layer_microns", 50)
        n_layers = params.get("n_surface_layers", 12)
        surface_levels = params.get("surface_refinement_level", [3, 4])
        base_cell_size = params.get("base_cell_target", 0.5e-3) * 1000  # mm
        target_cells = params.get("target_cells_range", [1e6, 5e6])
        
        self.logger.info("=" * 70)
        self.logger.info(f"🔬 PHYSICS-AWARE STAGE 2: {self.flow_model} Flow Regime")
        self.logger.info("=" * 70)
        self.logger.info(f"📐 Geometry Analysis: D_min={D:.1f}mm, Re={Re:.0f} ({flow_regime})")
        
        # Show pulsatile effects if detected
        if params.get("womersley"):
            wom = params["womersley"]
            self.logger.info(f"💓 Pulsatile Flow: α={wom:.1f} (Womersley number)")
            if params.get("pulsatile_correction"):
                self.logger.info("   ⚠️  High α detected - y+ target adjusted for thick viscous layer")
        
        self.logger.info(f"🔧 Flow Model: {self.flow_model}")
        self.logger.info(f"📦 Expected Cells: {target_cells[0]/1e6:.1f}-{target_cells[1]/1e6:.1f}M")
        self.logger.info(f"⚙️  Physics-Based Mesh Parameters:")
        self.logger.info(f"   - Base cell size: {base_cell_size:.2f}mm (D/{D/base_cell_size:.0f})")
        self.logger.info(f"   - Surface refinement: {surface_levels}")
        self.logger.info(f"   - First layer: {first_layer_um:.1f}µm ({n_layers} layers)")
        self.logger.info(f"   - Target y+: {params.get('target_yplus', 1.0)}")
        self.logger.info(f"   - Friction regime: {flow_regime} (Cf={params.get('friction_coefficient', 0.05):.4f})")
        
        self.logger.info(f"🌊 CFD Solver: {self.flow_model.lower()}Foam")
        self.logger.info("=" * 70)
        self.logger.info("ℹ️  Stage 2 builds a NEW physics-optimized mesh from scratch")
        self.logger.info("   Uses correct friction laws, minimum diameter, and pulsatile effects")
        self.logger.info("=" * 70)