"""
Main orchestrator for Stage1 mesh optimization.
Coordinates all modules and provides high-level optimization interface.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from .config_manager import ConfigManager
from .geometry_processor import GeometryProcessor
from .physics_calculator import PhysicsCalculator
from .quality_analyzer import QualityAnalyzer
from .mesh_generator import MeshGenerator
from .iteration_manager import IterationManager
from .constants import DEFAULT_CONSTANTS

logger = logging.getLogger(__name__)

class Stage1Optimizer:
    """
    Main orchestrator for Stage1 mesh optimization workflow
    
    Coordinates geometry processing, physics calculations, mesh generation,
    quality assessment, and iterative optimization until convergence.
    """
    
    def __init__(self, config_path: str, geometry_path: str, work_dir: str):
        """
        Initialize Stage1 optimizer with configuration and geometry
        
        Args:
            config_path: Path to JSON configuration file
            geometry_path: Path to STL geometry file
            work_dir: Working directory for mesh generation
        """
        # Initialize paths
        self.config_path = Path(config_path)
        self.geometry_path = Path(geometry_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core modules
        self.config_manager = ConfigManager(config_path)
        self.geometry_processor = GeometryProcessor(self.config_manager)
        self.physics_calculator = PhysicsCalculator(self.config_manager)
        self.quality_analyzer = QualityAnalyzer(self.config_manager)
        self.mesh_generator = MeshGenerator(self.config_manager)
        self.iteration_manager = IterationManager(
            self.config_manager, self.quality_analyzer, self.physics_calculator
        )
        
        # Optimization state
        self.optimization_started = False
        self.optimization_complete = False
        self.final_mesh_dir = None
        self.optimization_summary = {}
        
        logger.info(f"Stage1 Optimizer initialized: geometry={geometry_path}, work_dir={work_dir}")
    
    def run_full_optimization(self, max_iterations: Optional[int] = None, 
                            parallel_procs: int = 1) -> Dict[str, Any]:
        """
        Run complete Stage1 optimization workflow
        
        Args:
            max_iterations: Override maximum iterations from config
            parallel_procs: Number of parallel processors for mesh operations
            
        Returns:
            Optimization summary with results and final mesh path
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STAGE1 MESH OPTIMIZATION STARTED")
        logger.info("=" * 60)
        
        try:
            # Override max iterations if specified
            if max_iterations is not None:
                self.iteration_manager.max_iterations = max_iterations
            
            # Phase 1: Geometry preprocessing
            logger.info("Phase 1: Geometry preprocessing")
            geometry_info = self.geometry_processor.process_stl_geometry(self.geometry_path)
            if not geometry_info['valid']:
                raise RuntimeError(f"Invalid geometry: {geometry_info.get('error', 'Unknown error')}")
            
            # Phase 2: Physics-based parameter calculation
            logger.info("Phase 2: Physics-based parameter calculation")
            self._calculate_physics_parameters(geometry_info)
            
            # Phase 3: Iterative mesh optimization
            logger.info("Phase 3: Iterative mesh optimization")
            self._run_optimization_loop(parallel_procs)
            
            # Phase 4: Finalization
            logger.info("Phase 4: Optimization finalization")
            self._finalize_optimization()
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.optimization_summary = {
                'status': 'failed',
                'error': str(e),
                'total_time': time.time() - start_time
            }
            raise
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"STAGE1 OPTIMIZATION COMPLETE ({total_time:.1f}s)")
        logger.info("=" * 60)
        
        # Generate final summary
        self.optimization_summary = self._generate_final_summary(total_time)
        return self.optimization_summary
    
    def run_single_mesh_generation(self, parallel_procs: int = 1) -> Dict[str, Any]:
        """
        Run single mesh generation without optimization loop
        
        Args:
            parallel_procs: Number of parallel processors
            
        Returns:
            Mesh generation results
        """
        logger.info("Running single mesh generation (no optimization)")
        
        try:
            # Process geometry
            geometry_info = self.geometry_processor.process_stl_geometry(self.geometry_path)
            if not geometry_info['valid']:
                raise RuntimeError(f"Invalid geometry: {geometry_info.get('error', 'Unknown error')}")
            
            # Calculate physics parameters
            self._calculate_physics_parameters(geometry_info)
            
            # Generate single mesh
            mesh_dir = self.work_dir / "single_mesh"
            mesh_dir.mkdir(exist_ok=True)
            
            result = self._generate_and_assess_mesh(mesh_dir, parallel_procs, phase="single")
            
            return {
                'status': 'completed',
                'mesh_directory': str(mesh_dir),
                'geometry_info': geometry_info,
                'mesh_results': result
            }
            
        except Exception as e:
            logger.error(f"Single mesh generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_physics_parameters(self, geometry_info: Dict) -> None:
        """Calculate and apply physics-based parameters"""
        # Extract geometry characteristics
        diameter = geometry_info.get('characteristic_length', 0.02)  # Default 20mm
        
        # Get flow parameters from config
        physics_config = self.config_manager.config.get("physics", {})
        peak_velocity = physics_config.get("peak_velocity", 1.5)  # m/s
        base_cell_size = self.config_manager.config.get("base_size", 0.001)  # m
        
        # Calculate layer parameters
        layer_params = self.physics_calculator.calculate_layer_parameters(
            diameter, peak_velocity, base_cell_size
        )
        
        # Update configuration with physics-based values
        self.config_manager.update_layer_parameters(layer_params)
        
        # Calculate refinement bands if using Womersley
        use_womersley = physics_config.get("use_womersley_bands", False)
        near_dist, far_dist = self.physics_calculator.calculate_refinement_bands(
            base_cell_size, use_womersley
        )
        
        # Update refinement distances
        self.config_manager.update_refinement_bands(near_dist, far_dist)
        
        # Log applied parameters
        logger.info(f"Applied physics parameters: diameter={diameter*1000:.1f}mm, "
                   f"velocity={peak_velocity:.1f}m/s, firstLayer={layer_params['firstLayerThickness_abs']*1e6:.1f}μm")
    
    def _run_optimization_loop(self, parallel_procs: int) -> None:
        """Run the main optimization loop"""
        self.optimization_started = True
        
        while True:
            # Check if optimization should continue
            should_continue, reason = self.iteration_manager.should_continue_optimization()
            if not should_continue:
                logger.info(f"Optimization stopping: {reason}")
                break
            
            # Start new iteration
            iter_dir = self.iteration_manager.start_new_iteration(self.work_dir)
            
            # Generate and assess mesh for this iteration
            try:
                result = self._generate_and_assess_mesh(iter_dir, parallel_procs, 
                                                      phase=f"iter_{self.iteration_manager.current_iteration}")
                
                # Record iteration results
                self.iteration_manager.record_iteration_result(
                    snap_metrics=result['snap_metrics'],
                    layer_metrics=result['layer_metrics'],
                    layer_coverage=result['layer_coverage'],
                    mesh_dir=iter_dir,
                    success=result['success'],
                    error_msg=result.get('error_message')
                )
                
                # Check if optimization targets are met
                if result['success'] and result.get('meets_targets', False):
                    logger.info(f"Optimization targets achieved in iteration {self.iteration_manager.current_iteration}")
                    self.final_mesh_dir = iter_dir
                    break
                
                # Adapt parameters for next iteration
                if result['success']:
                    params_changed = self.iteration_manager.adapt_parameters()
                    if params_changed:
                        logger.info("Parameters adapted for next iteration")
                
            except Exception as e:
                logger.error(f"Iteration {self.iteration_manager.current_iteration} failed: {e}")
                self.iteration_manager.record_iteration_result(
                    snap_metrics={}, layer_metrics={}, layer_coverage={},
                    mesh_dir=iter_dir, success=False, error_msg=str(e)
                )
        
        # Clean up old iterations to save space
        self.iteration_manager.cleanup_old_iterations(self.work_dir)
    
    def _generate_and_assess_mesh(self, mesh_dir: Path, parallel_procs: int, 
                                phase: str) -> Dict[str, Any]:
        """
        Generate mesh and assess quality for a single iteration
        
        Args:
            mesh_dir: Directory for this mesh generation
            parallel_procs: Number of parallel processors
            phase: Phase name for logging
            
        Returns:
            Dictionary with generation results and quality metrics
        """
        logger.info(f"Generating mesh for {phase} in {mesh_dir}")
        
        try:
            # Copy and process geometry
            processed_geometry = self.geometry_processor.prepare_geometry_for_meshing(
                self.geometry_path, mesh_dir
            )
            
            # Generate OpenFOAM case files
            self.mesh_generator.generate_case_files(mesh_dir, processed_geometry)
            
            # Run blockMesh
            self.mesh_generator.run_blockmesh(mesh_dir)
            
            # Run snappyHexMesh (castellated + snap phases)
            snap_result = self.mesh_generator.run_snappy_no_layers(mesh_dir, parallel_procs)
            if not snap_result['success']:
                raise RuntimeError(f"Snap phase failed: {snap_result.get('error', 'Unknown error')}")
            
            # Assess snap phase quality
            snap_metrics = self.quality_analyzer.assess_mesh_quality(
                mesh_dir, f"{phase}_snap", parallel_procs
            )
            
            # Run layers phase
            layers_result = self.mesh_generator.run_snappy_layers(mesh_dir, parallel_procs)
            if not layers_result['success']:
                raise RuntimeError(f"Layers phase failed: {layers_result.get('error', 'Unknown error')}")
            
            # Assess final mesh quality
            layer_metrics = self.quality_analyzer.assess_mesh_quality(
                mesh_dir, f"{phase}_layers", parallel_procs
            )
            
            # Assess layer coverage
            layer_coverage = self.quality_analyzer.assess_layer_quality(mesh_dir)
            
            # Check if quality targets are met
            meets_targets = self.quality_analyzer.meets_quality_constraints(
                snap_metrics, layer_metrics, layer_coverage
            )
            
            # Generate quality diagnosis
            quality_diagnosis = self.quality_analyzer.diagnose_quality_issues(
                snap_metrics, layer_metrics
            )
            
            return {
                'success': True,
                'snap_metrics': snap_metrics,
                'layer_metrics': layer_metrics,
                'layer_coverage': layer_coverage,
                'meets_targets': meets_targets,
                'quality_diagnosis': quality_diagnosis,
                'mesh_directory': str(mesh_dir)
            }
            
        except Exception as e:
            logger.error(f"Mesh generation failed for {phase}: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'snap_metrics': {},
                'layer_metrics': {},
                'layer_coverage': {},
                'meets_targets': False
            }
    
    def _finalize_optimization(self) -> None:
        """Finalize optimization and identify best result"""
        if not self.final_mesh_dir and self.iteration_manager.best_iteration:
            # Use best iteration if no explicit final mesh was identified
            best_iter = self.iteration_manager.best_iteration['iteration']
            self.final_mesh_dir = self.work_dir / f"iter_{best_iter:03d}"
        
        if self.final_mesh_dir:
            # Create symbolic link to final mesh
            final_link = self.work_dir / "final_mesh"
            if final_link.exists():
                final_link.unlink()
            final_link.symlink_to(self.final_mesh_dir.name)
            
            logger.info(f"Final mesh available at: {self.final_mesh_dir}")
            logger.info(f"Symbolic link created: {final_link}")
        
        # Export quality analysis report
        if hasattr(self.quality_analyzer, 'export_quality_report'):
            report_path = self.work_dir / "quality_analysis_report.json"
            self.quality_analyzer.export_quality_report(report_path)
        
        self.optimization_complete = True
    
    def _generate_final_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive optimization summary"""
        optimization_summary = self.iteration_manager.get_optimization_summary()
        
        summary = {
            'status': 'completed',
            'total_time': total_time,
            'optimization_results': optimization_summary,
            'final_mesh_directory': str(self.final_mesh_dir) if self.final_mesh_dir else None,
            'configuration': {
                'config_path': str(self.config_path),
                'geometry_path': str(self.geometry_path),
                'work_directory': str(self.work_dir)
            }
        }
        
        # Add final quality metrics if available
        if self.final_mesh_dir and optimization_summary.get('best_iteration'):
            best_iteration = optimization_summary['best_iteration']
            summary['final_quality'] = {
                'quality_score': best_iteration['quality_score'],
                'mesh_metrics': best_iteration['layer_metrics'],
                'layer_coverage': best_iteration['layer_coverage']
            }
        
        return summary
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'optimization_started': self.optimization_started,
            'optimization_complete': self.optimization_complete,
            'current_iteration': self.iteration_manager.current_iteration,
            'converged': self.iteration_manager.converged,
            'convergence_reason': self.iteration_manager.convergence_reason,
            'best_quality': self.iteration_manager.best_iteration['quality_score'] if self.iteration_manager.best_iteration else 0.0,
            'final_mesh_directory': str(self.final_mesh_dir) if self.final_mesh_dir else None
        }
    
    def export_final_mesh(self, export_path: str, format_type: str = "openfoam") -> bool:
        """
        Export final mesh in specified format
        
        Args:
            export_path: Path to export directory
            format_type: Export format ("openfoam", "vtk", "stl")
            
        Returns:
            True if export successful
        """
        if not self.final_mesh_dir or not self.final_mesh_dir.exists():
            logger.error("No final mesh available for export")
            return False
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type.lower() == "openfoam":
                # Copy OpenFOAM case
                import shutil
                shutil.copytree(self.final_mesh_dir, export_dir / "openfoam_case", dirs_exist_ok=True)
                
            elif format_type.lower() == "vtk":
                # Convert to VTK format
                self.mesh_generator.export_mesh_vtk(self.final_mesh_dir, export_dir)
                
            elif format_type.lower() == "stl":
                # Extract surface mesh as STL
                self.mesh_generator.export_surface_stl(self.final_mesh_dir, export_dir)
            
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False
            
            logger.info(f"Mesh exported to {export_dir} in {format_type} format")
            return True
            
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")
            return False


def run_stage1_optimization(config_path: str, geometry_path: str, work_dir: str,
                          max_iterations: Optional[int] = None, 
                          parallel_procs: int = 1) -> Dict[str, Any]:
    """
    Convenience function to run complete Stage1 optimization
    
    Args:
        config_path: Path to configuration file
        geometry_path: Path to STL geometry
        work_dir: Working directory
        max_iterations: Maximum optimization iterations
        parallel_procs: Number of parallel processors
        
    Returns:
        Optimization summary
    """
    optimizer = Stage1Optimizer(config_path, geometry_path, work_dir)
    return optimizer.run_full_optimization(max_iterations, parallel_procs)


def run_single_mesh(config_path: str, geometry_path: str, work_dir: str,
                   parallel_procs: int = 1) -> Dict[str, Any]:
    """
    Convenience function to run single mesh generation
    
    Args:
        config_path: Path to configuration file
        geometry_path: Path to STL geometry
        work_dir: Working directory
        parallel_procs: Number of parallel processors
        
    Returns:
        Mesh generation results
    """
    optimizer = Stage1Optimizer(config_path, geometry_path, work_dir)
    return optimizer.run_single_mesh_generation(parallel_procs)