"""
Mesh quality assessment and convergence analysis for Stage1 optimization.
Handles checkMesh parsing, quality metrics tracking, and convergence detection.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import statistics

from .constants import DEFAULT_CONSTANTS
from ..utils import run_command, check_mesh_quality, parse_layer_coverage

logger = logging.getLogger(__name__)

class QualityAnalyzer:
    """Comprehensive mesh quality assessment and convergence detection"""
    
    def __init__(self, config_manager):
        self.config = config_manager.config
        self.targets = config_manager.targets
        self.constants = DEFAULT_CONSTANTS['mesh_quality']
        self.convergence_params = DEFAULT_CONSTANTS['convergence']
        
        # Quality history for convergence tracking
        self.quality_history = []
        self.current_iteration = 0
        
        # OpenFOAM environment
        self.openfoam_env = config_manager.get_openfoam_env()
        
    def assess_mesh_quality(self, mesh_dir: Path, phase_name: str = "mesh", 
                          parallel_procs: int = 1, wall_name: str = "wall_aorta") -> Dict:
        """
        Comprehensive mesh quality assessment with parallel support
        
        Args:
            mesh_dir: Directory containing mesh
            phase_name: Phase identifier (e.g., "snap", "layers")
            parallel_procs: Number of parallel processors 
            wall_name: Wall patch name for boundary analysis
            
        Returns:
            Comprehensive quality metrics dictionary
        """
        self.current_iteration += 1
        
        try:
            if parallel_procs > 1:
                metrics = self._parallel_quality_check(mesh_dir, phase_name, parallel_procs, wall_name)
            else:
                metrics = self._serial_quality_check(mesh_dir, phase_name, wall_name)
                
            # Add meta-information
            metrics.update({
                'phase': phase_name,
                'iteration': self.current_iteration,
                'analysis_mode': 'parallel' if parallel_procs > 1 else 'serial',
                'wall_patch': wall_name
            })
            
            # Store for convergence tracking
            self._update_quality_history(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed for {phase_name}: {e}")
            return self._get_fallback_metrics(phase_name)
    
    def _parallel_quality_check(self, mesh_dir: Path, phase_name: str, 
                              n_procs: int, wall_name: str) -> Dict:
        """Optimized parallel mesh quality checking"""
        logs_dir = mesh_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Try parallel checkMesh first (most efficient)
        logger.info(f"Running parallel checkMesh on {phase_name} mesh ({n_procs} procs)")
        check_cmd = ["mpirun", "-np", str(n_procs), "checkMesh", "-parallel"]
        
        try:
            result = run_command(
                check_cmd, 
                cwd=mesh_dir, 
                env_setup=self.openfoam_env,
                timeout=self.convergence_params.CHECKMESH_TIMEOUT
            )
            
            output_text = result.stdout + result.stderr
            (logs_dir / f"log.checkMesh.{phase_name}").write_text(output_text)
            
            metrics = self._parse_parallel_checkmesh(output_text)
            
            # If parallel parsing incomplete, reconstruct for serial analysis
            if not metrics.get("meshOK") and metrics.get("cells", 0) == 0:
                logger.info("Reconstructing for detailed serial analysis")
                self._reconstruct_mesh(mesh_dir, logs_dir, phase_name)
                serial_metrics = check_mesh_quality(mesh_dir, self.openfoam_env, wall_name=wall_name)
                metrics.update(serial_metrics)
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Parallel checkMesh failed: {e}, falling back to serial")
            self._reconstruct_mesh(mesh_dir, logs_dir, phase_name)
            return check_mesh_quality(mesh_dir, self.openfoam_env, wall_name=wall_name)
    
    def _serial_quality_check(self, mesh_dir: Path, phase_name: str, wall_name: str) -> Dict:
        """Standard serial mesh quality checking"""
        return check_mesh_quality(mesh_dir, self.openfoam_env, wall_name=wall_name)
    
    def _reconstruct_mesh(self, mesh_dir: Path, logs_dir: Path, phase_name: str) -> None:
        """Reconstruct parallel mesh for serial analysis"""
        try:
            result = run_command(
                ["reconstructPar", "-latestTime"], 
                cwd=mesh_dir, 
                env_setup=self.openfoam_env,
                timeout=300
            )
            (logs_dir / f"log.reconstruct.{phase_name}").write_text(result.stdout + result.stderr)
        except Exception as e:
            logger.error(f"Mesh reconstruction failed: {e}")
    
    def _parse_parallel_checkmesh(self, output: str) -> Dict:
        """Parse parallel checkMesh output for key metrics"""
        metrics = {
            "maxNonOrtho": 0.0,
            "maxSkewness": 0.0,
            "maxAspectRatio": 0.0,
            "cells": 0,
            "faces": 0,
            "points": 0,
            "meshOK": False
        }
        
        # Parse key metrics with regex patterns
        patterns = {
            "maxNonOrtho": r"Max non-orthogonality\s*=\s*([\d.]+)",
            "maxSkewness": r"Max skewness\s*=\s*([\d.]+)",
            "maxAspectRatio": r"Max aspect ratio\s*=\s*([\d.eE]+)",
            "cells": r"\bcells:\s+(\d+)",
            "faces": r"\bfaces:\s+(\d+)",
            "points": r"\bpoints:\s+(\d+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    if key in ["cells", "faces", "points"]:
                        metrics[key] = int(match.group(1))
                    else:
                        metrics[key] = float(match.group(1))
                except ValueError:
                    pass
        
        # OpenFOAM mesh validation - case insensitive
        metrics["meshOK"] = "mesh ok" in output.lower()
        
        return metrics
    
    def _update_quality_history(self, metrics: Dict) -> None:
        """Update quality history for convergence tracking"""
        quality_entry = {
            'iteration': self.current_iteration,
            'maxNonOrtho': metrics.get('maxNonOrtho', 999),
            'maxSkewness': metrics.get('maxSkewness', 999),
            'cells': metrics.get('cells', 0),
            'meshOK': metrics.get('meshOK', False),
            'phase': metrics.get('phase', 'unknown')
        }
        
        self.quality_history.append(quality_entry)
        
        # Keep only recent history to prevent memory bloat
        if len(self.quality_history) > 10:
            self.quality_history = self.quality_history[-10:]
    
    def check_convergence(self) -> Tuple[bool, str]:
        """
        Detect if mesh quality metrics have converged
        
        Returns:
            Tuple of (converged: bool, reason: str)
        """
        if len(self.quality_history) < self.convergence_params.MIN_ITERATIONS_FOR_CONVERGENCE:
            return False, f"Need at least {self.convergence_params.MIN_ITERATIONS_FOR_CONVERGENCE} iterations"
        
        # Extract recent quality metrics
        recent_history = self.quality_history[-3:]
        
        nonortho_values = [entry['maxNonOrtho'] for entry in recent_history]
        skewness_values = [entry['maxSkewness'] for entry in recent_history]
        cell_counts = [entry['cells'] for entry in recent_history]
        
        # Calculate coefficients of variation
        nonortho_cv = self._coefficient_of_variation(nonortho_values)
        skewness_cv = self._coefficient_of_variation(skewness_values)
        cells_cv = self._coefficient_of_variation(cell_counts)
        
        # Check convergence criteria
        converged = (
            nonortho_cv < self.convergence_params.CV_THRESHOLD_NON_ORTHO and
            skewness_cv < self.convergence_params.CV_THRESHOLD_SKEWNESS and
            cells_cv < self.convergence_params.CV_THRESHOLD_CELLS
        )
        
        if converged:
            avg_nonortho = statistics.mean(nonortho_values)
            avg_skewness = statistics.mean(skewness_values)
            avg_cells = int(statistics.mean(cell_counts))
            
            reason = (f"Quality converged: maxNonOrtho={avg_nonortho:.1f}±{nonortho_cv*100:.1f}%, "
                     f"maxSkewness={avg_skewness:.2f}±{skewness_cv*100:.1f}%, "
                     f"cells={avg_cells:,}±{cells_cv*100:.1f}%")
        else:
            reason = (f"Not converged: CV(nonOrtho)={nonortho_cv*100:.1f}%, "
                     f"CV(skew)={skewness_cv*100:.1f}%, CV(cells)={cells_cv*100:.1f}%")
        
        return converged, reason
    
    def _coefficient_of_variation(self, values: List[float]) -> float:
        """Calculate coefficient of variation (std/mean)"""
        if len(values) < 2 or all(v == 0 for v in values):
            return 0.0
            
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.0
            
        std_val = statistics.stdev(values)
        return std_val / abs(mean_val)
    
    def assess_layer_quality(self, mesh_dir: Path, wall_name: str = "wall_aorta") -> Dict:
        """
        Assess boundary layer quality and coverage
        
        Returns:
            Dictionary with layer coverage and quality metrics
        """
        try:
            coverage_data = parse_layer_coverage(mesh_dir, wall_name=wall_name)
            
            # Add quality interpretation
            overall_coverage = coverage_data.get("coverage_overall", 0.0)
            
            quality_assessment = {
                "coverage_excellent": overall_coverage > 0.90,
                "coverage_good": overall_coverage > 0.80,
                "coverage_acceptable": overall_coverage > 0.70,
                "coverage_poor": overall_coverage <= 0.70,
                "meets_target": overall_coverage >= self.targets.min_layer_cov
            }
            
            coverage_data["quality_assessment"] = quality_assessment
            
            # Log assessment
            status = "EXCELLENT" if quality_assessment["coverage_excellent"] else \
                    "GOOD" if quality_assessment["coverage_good"] else \
                    "ACCEPTABLE" if quality_assessment["coverage_acceptable"] else "POOR"
            
            logger.info(f"Layer coverage: {overall_coverage*100:.1f}% [{status}] "
                       f"(target: ≥{self.targets.min_layer_cov*100:.0f}%)")
            
            return coverage_data
            
        except Exception as e:
            logger.error(f"Layer quality assessment failed: {e}")
            return {
                "coverage_overall": 0.0,
                "perPatch": {},
                "quality_assessment": {"coverage_poor": True, "meets_target": False}
            }
    
    def meets_quality_constraints(self, snap_metrics: Dict, layer_metrics: Dict, 
                                layer_coverage: Dict) -> bool:
        """
        Check if mesh meets all quality constraints
        
        Args:
            snap_metrics: Metrics from snap phase
            layer_metrics: Metrics from layer phase  
            layer_coverage: Layer coverage data
            
        Returns:
            True if all constraints are satisfied
        """
        # Extract key values
        wall_coverage = layer_coverage.get("coverage_overall", 0.0)
        
        # Quality constraints from layer phase (most restrictive)
        mesh_ok = layer_metrics.get("meshOK", False)
        max_nonortho = float(layer_metrics.get("maxNonOrtho", 999))
        max_skewness = float(layer_metrics.get("maxSkewness", 999))
        
        # Check all constraints
        constraints_met = (
            mesh_ok and
            max_nonortho <= self.targets.max_nonortho and
            max_skewness <= self.targets.max_skewness and
            wall_coverage >= self.targets.min_layer_cov
        )
        
        # Log detailed assessment
        if constraints_met:
            logger.info(f"✅ Quality constraints MET: "
                       f"meshOK={mesh_ok}, nonOrtho={max_nonortho:.1f}≤{self.targets.max_nonortho}, "
                       f"skew={max_skewness:.2f}≤{self.targets.max_skewness}, "
                       f"coverage={wall_coverage*100:.1f}%≥{self.targets.min_layer_cov*100:.0f}%")
        else:
            logger.warning(f"❌ Quality constraints FAILED: "
                         f"meshOK={mesh_ok}, nonOrtho={max_nonortho:.1f} vs {self.targets.max_nonortho}, "
                         f"skew={max_skewness:.2f} vs {self.targets.max_skewness}, "
                         f"coverage={wall_coverage*100:.1f}% vs {self.targets.min_layer_cov*100:.0f}%")
        
        return constraints_met
    
    def diagnose_quality_issues(self, snap_metrics: Dict, layer_metrics: Dict) -> Dict:
        """
        Diagnose quality degradation between snap and layer phases
        
        Returns:
            Dictionary with diagnosis and suggested fixes
        """
        # Calculate quality deltas
        delta_nonortho = float(layer_metrics.get("maxNonOrtho", 0)) - float(snap_metrics.get("maxNonOrtho", 0))
        delta_skewness = float(layer_metrics.get("maxSkewness", 0)) - float(snap_metrics.get("maxSkewness", 0))
        
        diagnosis = {
            "delta_nonortho": delta_nonortho,
            "delta_skewness": delta_skewness,
            "layers_degrade_quality": delta_nonortho > self.convergence_params.LAYER_DEGRADATION_THRESHOLD_NO or 
                                    delta_skewness > self.convergence_params.LAYER_DEGRADATION_THRESHOLD_SK,
            "layers_improve_quality": delta_nonortho < -2 or delta_skewness < -0.2,
            "layers_neutral": abs(delta_nonortho) <= 2 and abs(delta_skewness) <= 0.2
        }
        
        # Generate suggestions
        suggestions = []
        if diagnosis["layers_degrade_quality"]:
            suggestions.extend([
                "Reduce firstLayerThickness by 20%",
                "Increase minThickness ratio to 0.2×firstLayer", 
                "Add more nGrow iterations",
                "Relax maxThicknessToMedialRatio"
            ])
        elif diagnosis["layers_neutral"]:
            suggestions.extend([
                "Increase nSurfaceLayers for better resolution",
                "Optimize expansion ratio",
                "Fine-tune featureAngle detection"
            ])
        
        diagnosis["suggestions"] = suggestions
        
        return diagnosis
    
    def _get_fallback_metrics(self, phase_name: str) -> Dict:
        """Return fallback metrics when quality assessment fails"""
        return {
            "maxNonOrtho": 999,
            "maxSkewness": 999, 
            "maxAspectRatio": 999,
            "cells": 0,
            "faces": 0,
            "points": 0,
            "meshOK": False,
            "phase": phase_name,
            "iteration": self.current_iteration,
            "error": True
        }
    
    def export_quality_report(self, output_path: Path) -> None:
        """Export comprehensive quality analysis report"""
        import json
        
        report = {
            "quality_targets": {
                "max_nonortho": self.targets.max_nonortho,
                "max_skewness": self.targets.max_skewness,
                "min_layer_coverage": self.targets.min_layer_cov
            },
            "convergence_parameters": {
                "cv_nonortho_threshold": self.convergence_params.CV_THRESHOLD_NON_ORTHO,
                "cv_skewness_threshold": self.convergence_params.CV_THRESHOLD_SKEWNESS,
                "min_iterations": self.convergence_params.MIN_ITERATIONS_FOR_CONVERGENCE
            },
            "quality_history": self.quality_history,
            "total_iterations": self.current_iteration,
            "analysis_timestamp": str(Path.cwd())  # Simple timestamp substitute
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality analysis report exported to {output_path}")