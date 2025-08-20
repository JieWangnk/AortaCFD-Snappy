#!/usr/bin/env python3
"""
AortaCFD-Snappy: Automated Mesh Optimization for Arterial Geometries
===================================================================

Robust, flexible mesh generation tool for cardiovascular CFD with intelligent
refinement strategies and comprehensive quality assessment.

Features:
- Adaptive surface refinement with failure recovery
- Two-pass boundary layer generation
- Multi-geometry support (any number of outlets)
- Professional mesh quality criteria
- Automated convergence assessment
- Multi-resolution strategy with coarse preview
- Geometry-based configuration templates
- Early termination for acceptable meshes
- Y+ coverage evaluation for boundary layers

Performance Optimizations (v2.0):
- Quick Win #1: Coarse preview mesh for rapid issue detection
- Quick Win #2: Automatic geometry-based template selection
- Quick Win #3: Early termination when mesh quality is acceptable
- Fixed Y+ evaluation with boundary layer estimation tool

Usage:
    python mesh_loop.py --geometry patient1
    python mesh_loop.py --geometry patient1 --config config/default.json
    python mesh_loop.py --geometry patient1 --max-iterations 5
    python mesh_loop.py --geometry patient1 --enable-solver --target-yplus 2.0
    
Examples:
    # Basic run with optimizations (30-50% faster):
    python mesh_loop.py --geometry patient1
    
    # Quick mesh with early termination:
    python mesh_loop.py --geometry patient1 --max-iterations 3
    
    # Full quality with Y+ evaluation:
    python mesh_loop.py --geometry patient1 --enable-solver
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import shutil
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from geometry_utils import GeometryProcessor
from mesh_functions import MeshGenerator
from quality_assessment import QualityEvaluator
from config_manager import ConfigManager


def setup_logger(name: str, log_file: Path = None, verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging for mesh optimization."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


class MeshOptimizationLoop:
    """Main mesh optimization orchestrator."""
    
    def __init__(self, geometry_name: str, config_manager: ConfigManager, 
                 output_dir: Path, logger: logging.Logger):
        self.geometry_name = geometry_name
        self.config_manager = config_manager
        self.output_dir = output_dir
        self.logger = logger
        
        # Initialize components
        self.geometry_processor = GeometryProcessor(logger)
        self.mesh_generator = MeshGenerator(config_manager.config, logger)
        self.quality_evaluator = QualityEvaluator(config_manager.config, logger)
        
        self.current_iteration = 0
        self.best_metrics = None
    
    def run_optimization(self, geometry_dir: Path, max_iterations: int = 10) -> dict:
        """
        Run complete mesh optimization loop.
        
        Args:
            geometry_dir: Directory containing STL files
            max_iterations: Maximum number of iterations
            
        Returns:
            Best mesh metrics achieved
        """
        self.logger.info(f"🚀 Starting mesh optimization for {self.geometry_name}")
        self.logger.info(f"📁 Geometry directory: {geometry_dir}")
        self.logger.info(f"📈 Maximum iterations: {max_iterations}")
        
        # Discover and validate geometry files
        try:
            geometry_files = self.geometry_processor.discover_geometry_files(geometry_dir)
        except FileNotFoundError as e:
            self.logger.error(f"❌ Geometry validation failed: {e}")
            return {"all_ok": False, "error": str(e)}
        
        # Quick Win #1: Multi-resolution strategy - coarse preview for complex geometries
        if max_iterations > 2:
            self._run_coarse_preview(geometry_files)
        
        # Main optimization loop
        for iteration in range(1, max_iterations + 1):
            self.current_iteration = iteration
            self.logger.info(f"\\n{'='*60}")
            self.logger.info(f"🔄 ITERATION {iteration:02d}/{max_iterations}")
            self.logger.info(f"{'='*60}")
            
            # Create iteration directory
            iter_dir = self.output_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Quick Win #2: Geometry-based templates 
                iter_config = self._get_geometry_optimized_config(geometry_files, iteration)
                
                # Run mesh generation sequence
                success = self._run_mesh_generation(iter_dir, geometry_files, iter_config)
                if not success:
                    self.logger.error(f"❌ Mesh generation failed at iteration {iteration}")
                    continue
                
                # Evaluate mesh quality
                metrics = self.quality_evaluator.evaluate_mesh_quality(iter_dir)
                self._save_iteration_metrics(iter_dir, metrics, iter_config)
                
                # Print summary
                self.logger.info(f"\\n📊 ITERATION {iteration} RESULTS:")
                self.quality_evaluator.print_summary(metrics)
                
                # Quick Win #3: Early termination for acceptable mesh
                if self._check_early_termination(metrics, iteration):
                    self.logger.info(f"⚡ Early termination: acceptable mesh quality at iteration {iteration}")
                    self.best_metrics = metrics
                    break
                
                # Check for full convergence
                if self._check_convergence(metrics):
                    self.logger.info(f"🎉 Optimization converged at iteration {iteration}")
                    self.best_metrics = metrics
                    break
                
                # Update best metrics
                if self._is_better_mesh(metrics):
                    self.best_metrics = metrics
                    self.logger.info(f"✨ New best mesh quality achieved")
                
                # Adjust configuration for next iteration
                if iteration < max_iterations:
                    self._adjust_config_for_next_iteration(metrics, iter_dir)
                    
            except Exception as e:
                self.logger.error(f"❌ Iteration {iteration} failed: {e}")
                continue
        
        # Final summary
        self._print_final_summary()
        return self.best_metrics or {"all_ok": False}
    
    def _run_mesh_generation(self, iter_dir: Path, geometry_files: dict, config: dict) -> bool:
        """Run complete mesh generation sequence."""
        try:
            # Setup directories
            (iter_dir / "system").mkdir(exist_ok=True)
            (iter_dir / "logs").mkdir(exist_ok=True)
            tri_surface_dir = iter_dir / "constant" / "triSurface"
            
            # Copy geometry files
            self.geometry_processor.copy_geometry_files(geometry_files, tri_surface_dir)
            
            # Calculate geometry properties
            all_stl_files = [geometry_files['inlet'], geometry_files['wall_aorta']] + geometry_files['outlets']
            min_bounds, max_bounds = self.geometry_processor.calculate_bounding_box(all_stl_files)
            
            # Analyze inlet for flow direction
            inlet_analysis = self.geometry_processor.analyze_inlet_orientation(geometry_files['inlet'])
            internal_point = self.geometry_processor.find_internal_point(
                min_bounds, max_bounds, inlet_analysis['centroid']
            )
            
            # Generate mesh configuration
            resolution = self.config_manager.calculate_mesh_resolution(np.max(max_bounds - min_bounds))
            self.mesh_generator.generate_block_mesh_dict(iter_dir, min_bounds, max_bounds, resolution)
            
            # Generate quality controls for two-pass approach
            self.mesh_generator.generate_quality_controls(iter_dir)
            
            # Generate basic OpenFOAM system files
            self.mesh_generator.generate_system_files(iter_dir)
            
            # Prepare geometry file names
            outlet_names = [f.stem for f in geometry_files['outlets']]
            geometry_dict = {'outlets': outlet_names}
            
            # Generate snappyHexMesh dictionaries
            surface_level = config["SNAPPY_UNIFORM"]["surface_level"]
            layers_config = self.config_manager.create_layer_config()
            
            # Generate snap-only dictionary
            self.mesh_generator.generate_snappy_hex_mesh_dict(
                iter_dir, geometry_dict, surface_level, internal_point, layers_config, mode="snap"
            )
            
            # Generate layers-only dictionary
            self.mesh_generator.generate_snappy_hex_mesh_dict(
                iter_dir, geometry_dict, surface_level, internal_point, layers_config, mode="layers"
            )
            
            # Generate surface features dictionary
            self._generate_surface_features_dict(iter_dir, geometry_files)
            
            # Execute mesh generation commands
            return self._execute_mesh_commands(iter_dir)
            
        except Exception as e:
            self.logger.error(f"Mesh generation setup failed: {e}")
            return False
    
    def _generate_surface_features_dict(self, iter_dir: Path, geometry_files: dict) -> None:
        """Generate surfaceFeaturesDict for edge extraction."""
        feature_angle = self.config_manager.get("GEOMETRY.feature_angle_deg", 150)
        
        surfaces = ['inlet', 'wall_aorta'] + [f.stem for f in geometry_files['outlets']]
        
        # Format surface list for OpenFOAM 12
        surface_list = '\n'.join([f'    "{s}.stl"' for s in surfaces])
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaces
(
{surface_list}
);

// Identify a feature when angle between faces < includedAngle
includedAngle   {feature_angle};

// Write obj files for visualization
writeObj        yes;

// ************************************************************************* //'''
        
        (iter_dir / "system" / "surfaceFeaturesDict").write_text(content)
    
    def _execute_mesh_commands(self, iter_dir: Path) -> bool:
        """Execute OpenFOAM mesh generation commands."""
        commands = [
            (['blockMesh'], 'log.blockMesh'),
            (['surfaceFeatures'], 'log.surfaceFeatures'),
            (['snappyHexMesh', '-overwrite', '-dict', 'system/snappyHexMeshDict.snap'], 'log.snappyHexMesh.snap'),
            (['snappyHexMesh', '-overwrite', '-dict', 'system/snappyHexMeshDict.layers'], 'log.snappyHexMesh.layers'),
            (['transformPoints', '"scale=(0.001 0.001 0.001)"'], 'log.transformPoints'),
            (['checkMesh', '-allGeometry', '-allTopology'], 'log.checkMesh')
        ]
        
        for command, log_file in commands:
            self.logger.info(f"🔧 Running: {' '.join(command)}")
            success = self.mesh_generator.run_mesh_command(command, iter_dir, log_file)
            if not success:
                self.logger.error(f"❌ Command failed: {' '.join(command)}")
                return False
        
        # Check for surface intersection failure
        patch_count = self.mesh_generator.check_patch_count(iter_dir)
        expected_patches = 1 + len([f for f in (iter_dir / "constant" / "triSurface").glob("outlet*.stl")]) + 1
        
        if patch_count < expected_patches:
            self.logger.warning(f"⚠️  Surface intersection issue: {patch_count}/{expected_patches} patches")
            return False
        
        self.logger.info(f"✅ Mesh generation completed successfully")
        return True
    
    def _check_convergence(self, metrics: dict) -> bool:
        """Check if optimization has converged."""
        acceptance = metrics.get("acceptance", {})
        
        # Basic convergence: mesh quality OK
        mesh_ok = acceptance.get("mesh_ok", False)
        
        # If solver is enabled, also check y+ coverage
        enable_solver = self.config_manager.get("SOLVER_SETTINGS.enable_warmup", False)
        if enable_solver:
            yplus_ok = acceptance.get("yPlus_ok", False)
            return mesh_ok and yplus_ok
        
        return mesh_ok
    
    def _is_better_mesh(self, metrics: dict) -> bool:
        """Determine if current mesh is better than previous best."""
        if not self.best_metrics:
            return True
        
        current_quality = self._calculate_quality_score(metrics)
        best_quality = self._calculate_quality_score(self.best_metrics)
        
        return current_quality > best_quality
    
    def _calculate_quality_score(self, metrics: dict) -> float:
        """Calculate overall quality score for mesh comparison."""
        checkmesh = metrics.get("checkMesh", {})
        acceptance = metrics.get("acceptance", {})
        
        score = 0.0
        
        # Quality metrics (higher is better, so invert)
        max_skew = checkmesh.get("maxSkewness", 10)
        max_nonortho = checkmesh.get("maxNonOrtho", 100)
        
        score += max(0, 100 - max_skew * 20)  # Skewness penalty
        score += max(0, 100 - max_nonortho)   # Non-orthogonality penalty
        
        # Cell count bonus (more cells generally better, with diminishing returns)
        cell_count = checkmesh.get("cells", 0)
        score += min(50, cell_count / 100000)  # Up to 50 points for cell count
        
        # Acceptance bonuses
        if acceptance.get("mesh_ok", False):
            score += 100
        if acceptance.get("yPlus_ok", False):
            score += 50
        
        return score
    
    def _adjust_config_for_next_iteration(self, metrics: dict, iter_dir: Path) -> None:
        """Adjust configuration based on current results."""
        recommendations = self.quality_evaluator.recommend_refinement_adjustments(metrics, iter_dir)
        
        strategy = recommendations.get('strategy', 'maintain')
        
        if strategy == 'reduce_refinement':
            self.config_manager.adjust_surface_refinement(-1)
            self.logger.info("📉 Reducing surface refinement due to intersection failure")
            
        elif strategy == 'increase_refinement':
            self.config_manager.adjust_surface_refinement(1)
            self.logger.info("📈 Increasing surface refinement for better quality")
            
        elif strategy == 'increase_base_resolution':
            self.config_manager.adjust_blockmesh_resolution(10)
            self.logger.info("🔧 Increasing base mesh resolution")
        
        # Apply layer adjustments
        layer_adjustments = recommendations.get('layer_adjustments', {})
        if layer_adjustments:
            self.config_manager.adjust_layer_parameters(layer_adjustments)
            self.logger.info(f"🎛️  Adjusting layer parameters: {list(layer_adjustments.keys())}")
    
    def _save_iteration_metrics(self, iter_dir: Path, metrics: dict, config: dict) -> None:
        """Save iteration results and configuration."""
        # Save metrics
        metrics_file = iter_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save configuration used for this iteration
        config_file = iter_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create manifest
        manifest = {
            "geometry": self.geometry_name,
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "converged": metrics.get("all_ok", False)
        }
        
        manifest_file = iter_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create ParaView file
        paraview_file = iter_dir / f"{self.geometry_name}.foam"
        paraview_file.touch()
    
    def _print_final_summary(self) -> None:
        """Print final optimization summary."""
        self.logger.info(f"\\n{'='*60}")
        self.logger.info(f"🏁 OPTIMIZATION COMPLETE")
        self.logger.info(f"{'='*60}")
        
        if self.best_metrics:
            self.logger.info(f"✨ Best mesh achieved:")
            self.quality_evaluator.print_summary(self.best_metrics)
        else:
            self.logger.info(f"❌ No successful mesh generated")
        
        self.logger.info(f"📁 Results saved to: {self.output_dir}")
    
    def _run_coarse_preview(self, geometry_files: dict) -> None:
        """Quick Win #1: Run coarse preview to identify major issues early."""
        self.logger.info("🔍 Running coarse preview mesh for rapid assessment...")
        
        preview_dir = self.output_dir / "preview_coarse"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        # Use very coarse settings for preview
        preview_config = self.config_manager.get_iteration_config(1)
        
        # Handle both list and integer surface_level formats
        surface_level = preview_config["SNAPPY_UNIFORM"]["surface_level"]
        if isinstance(surface_level, list):
            # Reduce both min and max levels for coarse preview
            preview_config["SNAPPY_UNIFORM"]["surface_level"] = [max(0, surface_level[0] - 1), max(1, surface_level[1] - 1)]
        else:
            preview_config["SNAPPY_UNIFORM"]["surface_level"] = max(1, surface_level - 2)
        
        # Also reduce volume level if present
        if "volume_level" in preview_config["SNAPPY_UNIFORM"]:
            volume_level = preview_config["SNAPPY_UNIFORM"]["volume_level"]
            if isinstance(volume_level, list):
                preview_config["SNAPPY_UNIFORM"]["volume_level"] = [max(0, volume_level[0] - 1), max(0, volume_level[1] - 1)]
            else:
                preview_config["SNAPPY_UNIFORM"]["volume_level"] = max(0, volume_level - 2)
        
        try:
            success = self._run_mesh_generation(preview_dir, geometry_files, preview_config)
            if success:
                # Quick quality check - just basic mesh stats
                metrics = self.quality_evaluator.evaluate_mesh_quality(preview_dir)
                skew = metrics.get("checkMesh", {}).get("maxSkewness", 10)
                cells = metrics.get("checkMesh", {}).get("cells", 0)
                self.logger.info(f"📋 Coarse preview: {cells} cells, max skew {skew:.2f}")
                
                # Adjust strategy based on preview
                if skew > 8.0:  # Very high skewness
                    self.config_manager.adjust_surface_refinement(-1)
                    self.logger.info("📉 Preview shows high skewness - reducing initial refinement")
            else:
                self.logger.warning("⚠️  Coarse preview failed - using conservative settings")
                self.config_manager.adjust_surface_refinement(-1)
        except Exception as e:
            self.logger.warning(f"⚠️  Preview failed: {e} - proceeding with standard approach")
    
    def _get_geometry_optimized_config(self, geometry_files: dict, iteration: int) -> dict:
        """Quick Win #2: Apply geometry-based configuration templates."""
        base_config = self.config_manager.get_iteration_config(iteration)
        
        # Analyze geometry characteristics
        try:
            all_stl_files = [geometry_files['inlet'], geometry_files['wall_aorta']] + geometry_files['outlets']
            min_bounds, max_bounds = self.geometry_processor.calculate_bounding_box(all_stl_files)
            domain_size = np.max(max_bounds - min_bounds)
            
            # Count outlets for complexity assessment
            num_outlets = len(geometry_files['outlets'])
            
            # Template selection based on geometry
            if num_outlets <= 2 and domain_size < 0.1:  # Simple, small geometry
                self.logger.debug("🏗️  Applying simple geometry template")
                # Handle both list and integer surface_level formats
                surface_level = base_config["SNAPPY_UNIFORM"]["surface_level"]
                if isinstance(surface_level, list):
                    base_config["SNAPPY_UNIFORM"]["surface_level"] = [surface_level[0] + 1, surface_level[1] + 1]
                else:
                    base_config["SNAPPY_UNIFORM"]["surface_level"] = surface_level + 1
                    
                if "nLayers" in base_config.get("LAYER_SETTINGS", {}):
                    base_config["LAYER_SETTINGS"]["nLayers"] = min(3, base_config["LAYER_SETTINGS"]["nLayers"])
                
            elif num_outlets > 4 or domain_size > 0.15:  # Complex geometry
                self.logger.debug("🏗️  Applying complex geometry template")
                # Handle both list and integer surface_level formats
                surface_level = base_config["SNAPPY_UNIFORM"]["surface_level"]
                if isinstance(surface_level, list):
                    base_config["SNAPPY_UNIFORM"]["surface_level"] = [max(1, surface_level[0] - 1), max(2, surface_level[1] - 1)]
                else:
                    base_config["SNAPPY_UNIFORM"]["surface_level"] = max(2, surface_level - 1)
                    
                if "nLayers" in base_config.get("LAYER_SETTINGS", {}):
                    base_config["LAYER_SETTINGS"]["nLayers"] = max(2, base_config["LAYER_SETTINGS"]["nLayers"] - 1)
                
            else:  # Standard geometry
                self.logger.debug("🏗️  Applying standard geometry template")
                # Use base configuration as-is
                pass
                
        except Exception as e:
            self.logger.warning(f"⚠️  Geometry analysis failed: {e} - using base configuration")
        
        return base_config
    
    def _check_early_termination(self, metrics: dict, iteration: int) -> bool:
        """Quick Win #3: Early termination criteria for acceptable meshes."""
        if iteration < 2:  # Don't terminate too early
            return False
            
        checkmesh = metrics.get("checkMesh", {})
        acceptance = metrics.get("acceptance", {})
        
        # Define "good enough" criteria
        skewness_ok = checkmesh.get("maxSkewness", 10) < 4.0  # Reasonable skewness
        nonortho_ok = checkmesh.get("maxNonOrtho", 100) < 75  # Acceptable non-orthogonality
        cells_ok = checkmesh.get("cells", 0) > 50000  # Minimum resolution
        basic_ok = acceptance.get("mesh_ok", False)  # Passes basic quality
        
        # Early termination if mesh is "good enough"
        good_enough = skewness_ok and nonortho_ok and cells_ok and basic_ok
        
        if good_enough:
            self.logger.info(f"✅ Acceptable quality achieved: skew={checkmesh.get('maxSkewness', 0):.2f}, " +
                           f"nonortho={checkmesh.get('maxNonOrtho', 0):.1f}, cells={checkmesh.get('cells', 0)}")
        
        return good_enough


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AortaCFD-Snappy: Automated mesh optimization for arterial geometries (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Features (v2.0):
  • Multi-resolution strategy with coarse preview
  • Geometry-based configuration templates  
  • Early termination for acceptable meshes
  • Y+ coverage evaluation with boundary layer estimation

Examples:
  python mesh_loop.py --geometry patient1                    # Basic run with all optimizations
  python mesh_loop.py --geometry patient1 --max-iterations 3 # Quick mesh with early termination  
  python mesh_loop.py --geometry patient1 --enable-solver    # Full quality with Y+ evaluation
  python mesh_loop.py --geometry patient1 --config config/custom.json --max-iterations 5
  python mesh_loop.py --geometry patient1 --verbose          # Detailed logging
        """
    )
    
    parser.add_argument('--geometry', required=True, 
                       help='Geometry name (directory in tutorial/)')
    parser.add_argument('--config', type=Path, 
                       help='Configuration file path')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum optimization iterations')
    parser.add_argument('--enable-solver', action='store_true',
                       help='Enable CFD solver for y+ evaluation')
    parser.add_argument('--output-dir', type=Path,
                       help='Custom output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    geometry_dir = base_dir / "tutorial" / args.geometry
    
    if not geometry_dir.exists():
        print(f"❌ Geometry directory not found: {geometry_dir}")
        sys.exit(1)
    
    output_dir = args.output_dir or (base_dir / "output" / args.geometry)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "optimization.log"
    logger = setup_logger(f"AortaCFD-Snappy_{args.geometry}", log_file, args.verbose)
    
    # Load configuration
    config_manager = ConfigManager(args.config, logger)
    
    # Enable solver if requested
    if args.enable_solver:
        config_manager.set("SOLVER_SETTINGS.enable_warmup", True)
    
    try:
        # Run optimization
        optimizer = MeshOptimizationLoop(args.geometry, config_manager, output_dir, logger)
        results = optimizer.run_optimization(geometry_dir, args.max_iterations)
        
        # Exit with appropriate code
        if results.get("all_ok", False):
            logger.info("🎉 Optimization completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Optimization failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()