"""
Iteration management and optimization loop control for Stage1 mesh generation.
Handles parameter updates, convergence tracking, and adaptive optimization.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time

from .constants import DEFAULT_CONSTANTS
from ..utils import run_command

logger = logging.getLogger(__name__)

class IterationManager:
    """Manages optimization iterations and parameter adaptation"""
    
    def __init__(self, config_manager, quality_analyzer, physics_calculator):
        self.config = config_manager.config
        self.config_manager = config_manager
        self.quality_analyzer = quality_analyzer
        self.physics_calculator = physics_calculator
        self.constants = DEFAULT_CONSTANTS['convergence']
        
        # Iteration tracking
        self.current_iteration = 0
        self.max_iterations = int(self.config.get("optimization", {}).get("max_iterations", 10))
        self.iteration_history = []
        
        # Parameter adaptation tracking
        self.parameter_adjustments = []
        self.failed_attempts = 0
        self.success_streak = 0
        
        # Convergence state
        self.converged = False
        self.convergence_reason = ""
        self.best_iteration = None
        
        # OpenFOAM environment
        self.openfoam_env = config_manager.get_openfoam_env()
    
    def should_continue_optimization(self) -> Tuple[bool, str]:
        """
        Determine if optimization should continue
        
        Returns:
            Tuple of (should_continue: bool, reason: str)
        """
        if self.current_iteration >= self.max_iterations:
            return False, f"Maximum iterations reached ({self.max_iterations})"
        
        if self.converged:
            return False, f"Converged: {self.convergence_reason}"
        
        if self.failed_attempts >= self.constants.MAX_CONSECUTIVE_FAILURES:
            return False, f"Too many consecutive failures ({self.failed_attempts})"
        
        # Check for stuck optimization
        if len(self.iteration_history) >= 3:
            recent_qualities = [h['quality_score'] for h in self.iteration_history[-3:]]
            if all(abs(q - recent_qualities[0]) < 0.01 for q in recent_qualities):
                return False, "Quality metrics have stagnated"
        
        return True, "Continue optimization"
    
    def start_new_iteration(self, base_work_dir: Path) -> Path:
        """
        Initialize a new optimization iteration
        
        Args:
            base_work_dir: Base working directory
            
        Returns:
            Path to iteration directory
        """
        self.current_iteration += 1
        
        # Create iteration directory
        iter_dir = base_work_dir / f"iter_{self.current_iteration:03d}"
        iter_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["constant", "system", "logs"]:
            (iter_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Starting iteration {self.current_iteration} in {iter_dir}")
        
        # Log current parameters
        self._log_iteration_parameters()
        
        return iter_dir
    
    def record_iteration_result(self, snap_metrics: Dict, layer_metrics: Dict, 
                              layer_coverage: Dict, mesh_dir: Path, 
                              success: bool, error_msg: Optional[str] = None) -> None:
        """
        Record results from completed iteration
        
        Args:
            snap_metrics: Metrics from snap phase
            layer_metrics: Metrics from layers phase
            layer_coverage: Layer coverage data
            mesh_dir: Path to mesh directory
            success: Whether iteration succeeded
            error_msg: Error message if failed
        """
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(snap_metrics, layer_metrics, layer_coverage)
        
        # Record iteration data
        iteration_data = {
            'iteration': self.current_iteration,
            'timestamp': time.time(),
            'success': success,
            'error_message': error_msg,
            'quality_score': quality_score,
            'snap_metrics': snap_metrics,
            'layer_metrics': layer_metrics,
            'layer_coverage': layer_coverage,
            'parameters': self._get_current_parameters(),
            'mesh_directory': str(mesh_dir)
        }
        
        self.iteration_history.append(iteration_data)
        
        # Update success/failure tracking
        if success:
            self.failed_attempts = 0
            self.success_streak += 1
            
            # Update best iteration if this is better
            if self.best_iteration is None or quality_score > self.best_iteration['quality_score']:
                self.best_iteration = iteration_data.copy()
                logger.info(f"New best iteration {self.current_iteration} with quality {quality_score:.3f}")
        else:
            self.failed_attempts += 1
            self.success_streak = 0
            logger.warning(f"Iteration {self.current_iteration} failed: {error_msg}")
        
        # Check convergence
        converged, reason = self.quality_analyzer.check_convergence()
        if converged:
            self.converged = True
            self.convergence_reason = reason
            logger.info(f"Optimization converged: {reason}")
        
        # Save iteration history
        self._save_iteration_history(mesh_dir.parent)
    
    def adapt_parameters(self) -> bool:
        """
        Adapt parameters based on iteration results
        
        Returns:
            True if parameters were modified
        """
        if len(self.iteration_history) < 2:
            return False
        
        current_result = self.iteration_history[-1]
        previous_result = self.iteration_history[-2]
        
        # Don't adapt if current iteration failed
        if not current_result['success']:
            return self._handle_failed_iteration()
        
        # Analyze quality trends
        current_quality = current_result['quality_score']
        previous_quality = previous_result['quality_score']
        
        quality_improvement = current_quality - previous_quality
        
        if quality_improvement > 0.05:
            # Significant improvement - continue current strategy
            logger.info(f"Good progress: quality improved by {quality_improvement:.3f}")
            return False
        elif quality_improvement < -0.1:
            # Significant degradation - revert and try different approach
            logger.warning(f"Quality degraded by {abs(quality_improvement):.3f}, adapting parameters")
            return self._revert_and_adapt()
        else:
            # Marginal change - try refinement
            return self._refine_parameters()
    
    def _handle_failed_iteration(self) -> bool:
        """Handle failed iteration by relaxing constraints"""
        adjustments_made = False
        
        # Get current layer parameters
        layers_config = self.config.get("snappyHexMeshDict", {}).get("addLayersControls", {})
        
        # Relax layer constraints
        if "nGrow" in layers_config and layers_config["nGrow"] < 3:
            layers_config["nGrow"] += 1
            adjustments_made = True
            logger.info(f"Relaxed nGrow to {layers_config['nGrow']}")
        
        if "maxThicknessToMedialRatio" in layers_config and layers_config["maxThicknessToMedialRatio"] < 0.8:
            layers_config["maxThicknessToMedialRatio"] += 0.1
            adjustments_made = True
            logger.info(f"Relaxed maxThicknessToMedialRatio to {layers_config['maxThicknessToMedialRatio']}")
        
        # Reduce first layer thickness if too aggressive
        if "firstLayerThickness_abs" in layers_config:
            current_thickness = layers_config["firstLayerThickness_abs"]
            new_thickness = current_thickness * 1.5
            layers_config["firstLayerThickness_abs"] = new_thickness
            layers_config["minThickness_abs"] = new_thickness * 0.15
            adjustments_made = True
            logger.info(f"Increased first layer thickness to {new_thickness*1e6:.1f} μm")
        
        if adjustments_made:
            self.parameter_adjustments.append({
                'iteration': self.current_iteration,
                'type': 'relaxation_after_failure',
                'changes': layers_config
            })
        
        return adjustments_made
    
    def _revert_and_adapt(self) -> bool:
        """Revert to previous parameters and try different adaptation"""
        if len(self.iteration_history) < 2:
            return False
        
        # Revert to previous iteration's parameters
        previous_params = self.iteration_history[-2]['parameters']
        current_config = self.config.get("snappyHexMeshDict", {})
        
        # Apply previous parameters
        if "addLayersControls" in previous_params:
            current_config["addLayersControls"] = previous_params["addLayersControls"].copy()
        
        logger.info("Reverted to previous parameters due to quality degradation")
        
        # Try alternative adaptation strategy
        return self._try_alternative_strategy()
    
    def _try_alternative_strategy(self) -> bool:
        """Try alternative parameter adaptation strategy"""
        adjustments_made = False
        layers_config = self.config.get("snappyHexMeshDict", {}).get("addLayersControls", {})
        
        # Strategy: Adjust expansion ratio instead of layer count
        if "expansionRatio" in layers_config:
            current_ratio = layers_config["expansionRatio"]
            if current_ratio > 1.5:
                # Reduce expansion ratio for more gradual growth
                layers_config["expansionRatio"] = max(1.15, current_ratio - 0.1)
                adjustments_made = True
                logger.info(f"Reduced expansion ratio to {layers_config['expansionRatio']}")
            elif current_ratio < 1.15:
                # Increase expansion ratio for fewer but thicker layers
                layers_config["expansionRatio"] = min(1.8, current_ratio + 0.15)
                adjustments_made = True
                logger.info(f"Increased expansion ratio to {layers_config['expansionRatio']}")
        
        # Strategy: Adjust surface layer count
        if "nSurfaceLayers" in layers_config and not adjustments_made:
            current_layers = layers_config["nSurfaceLayers"]
            if current_layers > 8:
                layers_config["nSurfaceLayers"] = max(5, current_layers - 2)
                adjustments_made = True
                logger.info(f"Reduced surface layers to {layers_config['nSurfaceLayers']}")
            elif current_layers < 5:
                layers_config["nSurfaceLayers"] = min(12, current_layers + 2)
                adjustments_made = True
                logger.info(f"Increased surface layers to {layers_config['nSurfaceLayers']}")
        
        return adjustments_made
    
    def _refine_parameters(self) -> bool:
        """Make small refinements to parameters"""
        adjustments_made = False
        layers_config = self.config.get("snappyHexMeshDict", {}).get("addLayersControls", {})
        
        # Fine-tune feature angle
        if "featureAngle" in layers_config:
            current_angle = layers_config["featureAngle"]
            if current_angle < 75:
                layers_config["featureAngle"] = min(80, current_angle + 5)
                adjustments_made = True
                logger.info(f"Refined feature angle to {layers_config['featureAngle']}°")
        
        # Fine-tune minimum median axis angle
        if "minMedianAxisAngle" in layers_config and not adjustments_made:
            current_angle = layers_config["minMedianAxisAngle"]
            if current_angle > 60:
                layers_config["minMedianAxisAngle"] = max(50, current_angle - 5)
                adjustments_made = True
                logger.info(f"Refined median axis angle to {layers_config['minMedianAxisAngle']}°")
        
        return adjustments_made
    
    def _calculate_quality_score(self, snap_metrics: Dict, layer_metrics: Dict, 
                               layer_coverage: Dict) -> float:
        """
        Calculate overall quality score for iteration ranking
        
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        score = 0.0
        weights = {
            'mesh_ok': 0.3,
            'nonortho': 0.25,
            'skewness': 0.25,
            'coverage': 0.2
        }
        
        # Mesh validity (pass/fail)
        if layer_metrics.get("meshOK", False):
            score += weights['mesh_ok']
        
        # Non-orthogonality score (lower is better)
        max_nonortho = float(layer_metrics.get("maxNonOrtho", 999))
        target_nonortho = self.quality_analyzer.targets.max_nonortho
        if max_nonortho <= target_nonortho:
            nonortho_score = 1.0
        else:
            nonortho_score = max(0.0, 1.0 - (max_nonortho - target_nonortho) / target_nonortho)
        score += weights['nonortho'] * nonortho_score
        
        # Skewness score (lower is better)
        max_skewness = float(layer_metrics.get("maxSkewness", 999))
        target_skewness = self.quality_analyzer.targets.max_skewness
        if max_skewness <= target_skewness:
            skewness_score = 1.0
        else:
            skewness_score = max(0.0, 1.0 - (max_skewness - target_skewness) / target_skewness)
        score += weights['skewness'] * skewness_score
        
        # Coverage score
        coverage = layer_coverage.get("coverage_overall", 0.0)
        target_coverage = self.quality_analyzer.targets.min_layer_cov
        coverage_score = min(1.0, coverage / target_coverage) if target_coverage > 0 else 0.0
        score += weights['coverage'] * coverage_score
        
        return min(1.0, max(0.0, score))
    
    def _get_current_parameters(self) -> Dict:
        """Get snapshot of current optimization parameters"""
        return {
            'castellatedMeshControls': self.config.get("snappyHexMeshDict", {}).get("castellatedMeshControls", {}),
            'snapControls': self.config.get("snappyHexMeshDict", {}).get("snapControls", {}),
            'addLayersControls': self.config.get("snappyHexMeshDict", {}).get("addLayersControls", {}),
            'meshQualityControls': self.config.get("snappyHexMeshDict", {}).get("meshQualityControls", {})
        }
    
    def _log_iteration_parameters(self) -> None:
        """Log key parameters for current iteration"""
        layers = self.config.get("snappyHexMeshDict", {}).get("addLayersControls", {})
        
        first_layer = layers.get("firstLayerThickness_abs", 0) * 1e6
        min_thickness = layers.get("minThickness_abs", 0) * 1e6
        n_layers = layers.get("nSurfaceLayers", 0)
        expansion = layers.get("expansionRatio", 1.0)
        
        logger.info(f"Iteration {self.current_iteration} parameters: "
                   f"firstLayer={first_layer:.1f}μm, minThick={min_thickness:.1f}μm, "
                   f"layers={n_layers}, expansion={expansion}")
    
    def _save_iteration_history(self, work_dir: Path) -> None:
        """Save iteration history to file"""
        history_file = work_dir / "iteration_history.json"
        
        # Prepare serializable data
        serializable_history = []
        for entry in self.iteration_history:
            serializable_entry = entry.copy()
            # Convert Path objects to strings
            if 'mesh_directory' in serializable_entry:
                serializable_entry['mesh_directory'] = str(serializable_entry['mesh_directory'])
            serializable_history.append(serializable_entry)
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2, default=str)
        
        logger.debug(f"Saved iteration history to {history_file}")
    
    def get_optimization_summary(self) -> Dict:
        """
        Get comprehensive optimization summary
        
        Returns:
            Dictionary with optimization results and statistics
        """
        if not self.iteration_history:
            return {"status": "no_iterations", "total_iterations": 0}
        
        successful_iterations = [h for h in self.iteration_history if h['success']]
        
        summary = {
            "status": "converged" if self.converged else "incomplete",
            "convergence_reason": self.convergence_reason,
            "total_iterations": self.current_iteration,
            "successful_iterations": len(successful_iterations),
            "failed_iterations": len(self.iteration_history) - len(successful_iterations),
            "best_iteration": self.best_iteration,
            "parameter_adjustments": len(self.parameter_adjustments),
            "final_quality_score": self.iteration_history[-1]['quality_score'] if self.iteration_history else 0.0
        }
        
        if successful_iterations:
            quality_scores = [h['quality_score'] for h in successful_iterations]
            summary.update({
                "quality_improvement": max(quality_scores) - min(quality_scores),
                "average_quality": sum(quality_scores) / len(quality_scores),
                "quality_trend": self._analyze_quality_trend()
            })
        
        return summary
    
    def _analyze_quality_trend(self) -> str:
        """Analyze overall quality trend across iterations"""
        if len(self.iteration_history) < 3:
            return "insufficient_data"
        
        successful_iterations = [h for h in self.iteration_history if h['success']]
        if len(successful_iterations) < 3:
            return "insufficient_successful_iterations"
        
        # Look at quality trend in recent iterations
        recent_qualities = [h['quality_score'] for h in successful_iterations[-3:]]
        
        if all(recent_qualities[i] >= recent_qualities[i-1] for i in range(1, len(recent_qualities))):
            return "improving"
        elif all(recent_qualities[i] <= recent_qualities[i-1] for i in range(1, len(recent_qualities))):
            return "degrading"
        else:
            return "oscillating"
    
    def cleanup_old_iterations(self, work_dir: Path, keep_best: bool = True, 
                             keep_recent: int = 3) -> None:
        """
        Clean up old iteration directories to save space
        
        Args:
            work_dir: Base working directory
            keep_best: Whether to preserve the best iteration
            keep_recent: Number of recent iterations to keep
        """
        if self.current_iteration <= keep_recent:
            return  # Not enough iterations to warrant cleanup
        
        iterations_to_keep = set()
        
        # Keep recent iterations
        for i in range(max(1, self.current_iteration - keep_recent + 1), self.current_iteration + 1):
            iterations_to_keep.add(i)
        
        # Keep best iteration
        if keep_best and self.best_iteration:
            iterations_to_keep.add(self.best_iteration['iteration'])
        
        # Remove old iterations
        for i in range(1, self.current_iteration + 1):
            if i not in iterations_to_keep:
                iter_dir = work_dir / f"iter_{i:03d}"
                if iter_dir.exists():
                    try:
                        shutil.rmtree(iter_dir)
                        logger.debug(f"Cleaned up iteration directory {iter_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {iter_dir}: {e}")
        
        logger.info(f"Iteration cleanup complete, kept {len(iterations_to_keep)} directories")