"""
Mesh quality assessment and evaluation utilities.

Provides comprehensive mesh quality analysis, layer coverage evaluation,
and adaptive refinement recommendations for cardiovascular geometries.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np


class QualityEvaluator:
    """Evaluate mesh quality and provide refinement recommendations."""
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
    
    def evaluate_mesh_quality(self, iter_dir: Path) -> Dict:
        """
        Comprehensive mesh quality evaluation.
        
        Args:
            iter_dir: Iteration directory containing mesh
            
        Returns:
            Dictionary with quality metrics and acceptance status
        """
        # Run professional mesh evaluation
        eval_script = self._get_evaluation_script()
        
        if eval_script and eval_script.exists():
            return self._run_professional_evaluation(iter_dir, eval_script)
        else:
            return self._run_basic_evaluation(iter_dir)
    
    def _run_professional_evaluation(self, iter_dir: Path, eval_script: Path) -> Dict:
        """Run professional mesh evaluation using external script."""
        yplus_band = self.config.get("yplus_band", [1.0, 5.0])
        required_coverage = self.config.get("required_yplus_coverage", 0.9)
        
        try:
            cmd = [
                'python', str(eval_script), str(iter_dir),
                '--checkmesh-log', str(iter_dir / 'logs' / 'log.checkMesh'),
                '--yplus-band', str(yplus_band[0]), str(yplus_band[1]),
                '--required-yplus-coverage', str(required_coverage),
                '--out', str(iter_dir / 'metrics.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                metrics_file = iter_dir / 'metrics.json'
                if metrics_file.exists():
                    return json.loads(metrics_file.read_text())
            
        except Exception as e:
            self.logger.warning(f"Professional evaluation failed: {e}")
        return self._run_basic_evaluation(iter_dir)
    
    def _run_basic_evaluation(self, iter_dir: Path) -> Dict:
        """Run basic mesh quality evaluation."""
        metrics = {
            'checkMesh': {},
            'yPlus': {'coverage_overall': 0.0, 'totalFaces': 0, 'perPatch': {}},
            'acceptance': {'mesh_ok': False, 'yPlus_ok': False},
            'all_ok': False
        }
        
        # Parse checkMesh log
        checkmesh_log = iter_dir / 'logs' / 'log.checkMesh'
        if checkmesh_log.exists():
            mesh_stats = self._parse_checkmesh_log(checkmesh_log)
            metrics['checkMesh'] = mesh_stats
            
            # Basic acceptance criteria
            metrics['acceptance']['mesh_ok'] = (
                mesh_stats.get('maxNonOrtho', 100) <= 65 and
                mesh_stats.get('maxSkewness', 10) <= 4.0 and
                mesh_stats.get('negVolCells', 1) == 0
            )
        
        # Parse layer coverage
        layer_stats = self._parse_layer_coverage(iter_dir)
        if layer_stats:
            metrics['layerCoverage'] = layer_stats
        
        metrics['all_ok'] = metrics['acceptance']['mesh_ok']
        
        # Save metrics
        (iter_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        
        return metrics
    
    def _parse_checkmesh_log(self, log_file: Path) -> Dict:
        """Parse checkMesh log for quality metrics."""
        try:
            content = log_file.read_text()
            
            stats = {}
            
            # Extract key metrics
            patterns = {
                'cells': r'cells:\s*(\d+)',
                'faces': r'faces:\s*(\d+)',
                'points': r'points:\s*(\d+)',
                'maxNonOrtho': r'non-orthogonality.*?(\d+\.?\d*)',
                'maxSkewness': r'skewness.*?(\d+\.?\d*)',
                'maxAspectRatio': r'aspect ratio.*?(\d+\.?\d*)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        stats[key] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                    except ValueError:
                        pass
            
            # Check for failed cells
            if 'Failed' in content or 'FOAM FATAL' in content:
                stats['negVolCells'] = 1
            else:
                stats['negVolCells'] = 0
                
            stats['meshOK'] = stats['negVolCells'] == 0
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Could not parse checkMesh log: {e}")
            return {}
    
    def _parse_layer_coverage(self, iter_dir: Path) -> Optional[Dict]:
        """Parse snappyHexMesh layer coverage from logs."""
        layer_log = iter_dir / 'logs' / 'log.snappyHexMesh.layers'
        if not layer_log.exists():
            return None
        
        try:
            content = layer_log.read_text()
            
            # Look for layer statistics
            layer_stats = {
                'total_faces': 0,
                'faces_with_layers': 0,
                'layer_coverage': 0.0,
                'patches': {}
            }
            
            # Parse layer addition summary
            patterns = [
                r'Added boundary layers.*?(\\d+).*?faces.*?(\\d+).*?cells',
                r'Layer coverage.*?(\\d+\\.?\\d*)%.*?\\((\\d+)/(\\d+) faces\\)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    if len(match.groups()) >= 2:
                        layer_stats['faces_with_layers'] = int(match.group(2))
                        layer_stats['total_faces'] = int(match.group(2))
                        layer_stats['layer_coverage'] = 1.0
                    break
            
            # Parse per-patch information
            patch_pattern = r'(\\w+):\\s*(\\d+)\\s*faces,\\s*(\\d+\\.?\\d*)\\s*avg layers'
            for match in re.finditer(patch_pattern, content):
                patch_name = match.group(1)
                faces = int(match.group(2))
                avg_layers = float(match.group(3))
                layer_stats['patches'][patch_name] = {
                    'faces': faces,
                    'average_layers': avg_layers
                }
            
            return layer_stats
            
        except Exception as e:
            self.logger.warning(f"Could not parse layer coverage: {e}")
            return None
    
    def _get_evaluation_script(self) -> Optional[Path]:
        """Get path to professional evaluation script if available."""
        # Look for evaluation script in tools directory
        possible_paths = [
            Path(__file__).parent.parent / 'tools' / 'checkmesh_yplus_eval.py',
            Path('tools') / 'checkmesh_yplus_eval.py'
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def recommend_refinement_adjustments(self, metrics: Dict, iter_dir: Path) -> Dict[str, any]:
        """
        Analyze metrics and recommend refinement adjustments.
        
        Args:
            metrics: Current mesh quality metrics
            iter_dir: Iteration directory
            
        Returns:
            Dictionary with recommended adjustments
        """
        recommendations = {
            'surface_level_change': 0,
            'blockmesh_resolution_change': 0,
            'layer_adjustments': {},
            'strategy': 'maintain',
            'reasons': []
        }
        
        acceptance = metrics.get('acceptance', {})
        checkmesh = metrics.get('checkMesh', {})
        
        # Check for surface intersection failure
        patch_count = self._check_patch_count(iter_dir)
        expected_patches = 6  # inlet + 4 outlets + wall_aorta
        
        if patch_count < expected_patches:
            recommendations['surface_level_change'] = -1
            recommendations['strategy'] = 'reduce_refinement'
            recommendations['reasons'].append(f'Surface intersection failure: only {patch_count}/{expected_patches} patches')
            return recommendations
        
        # Quality-based adjustments
        if not acceptance.get('mesh_ok', False):
            max_skew = checkmesh.get('maxSkewness', 0)
            max_nonortho = checkmesh.get('maxNonOrtho', 0)
            
            if max_skew > 4.0 or max_nonortho > 65:
                recommendations['surface_level_change'] = 1
                recommendations['strategy'] = 'increase_refinement'
                recommendations['reasons'].append(f'Poor mesh quality: skew={max_skew:.2f}, nonortho={max_nonortho:.1f}°')
            else:
                # Try alternative strategies
                recommendations['blockmesh_resolution_change'] = 10
                recommendations['strategy'] = 'increase_base_resolution'
                recommendations['reasons'].append('Mesh quality issues - increasing base resolution')
        
        # Layer coverage adjustments
        layer_coverage = metrics.get('layerCoverage', {}).get('layer_coverage', 0.0)
        if layer_coverage < 0.5:
            recommendations['layer_adjustments'] = {
                'expansionRatio': 1.15,  # Gentler growth
                'finalLayerThickness': 0.35,  # Thicker final layer
                'featureAngle': 120  # More permissive angles
            }
            recommendations['reasons'].append(f'Low layer coverage: {layer_coverage:.1%}')
        
        return recommendations
    
    def _check_patch_count(self, iter_dir: Path) -> int:
        """Check number of patches in boundary file."""
        boundary_file = iter_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            return 0
        
        try:
            content = boundary_file.read_text()
            patch_count = content.count('type            patch;') + content.count('type            wall;')
            return patch_count
        except Exception:
            return 0
    
    def print_summary(self, metrics: Dict) -> None:
        """Print formatted summary of mesh quality."""
        checkmesh = metrics.get("checkMesh", {})
        yplus_data = metrics.get("yPlus", {})
        acceptance = metrics.get("acceptance", {})
        
        print(f"📈 Cell count: {checkmesh.get('cells', 0):,}")
        
        max_skew = checkmesh.get('maxSkewness', 0)
        skew_str = f"{max_skew:.2f}" if max_skew is not None else "N/A"
        skew_status = '✅' if acceptance.get('skew_ok') else '❌'
        print(f"📐 Max skewness: {skew_str} {skew_status}")
        
        max_nonortho = checkmesh.get('maxNonOrtho', 0)
        nonortho_status = '✅' if acceptance.get('nonOrtho_ok') else '❌'
        print(f"📏 Max non-ortho: {max_nonortho:.1f}° {nonortho_status}")
        
        coverage = yplus_data.get('coverage_overall', 0)
        yplus_status = '✅' if acceptance.get('yPlus_ok') else '❌'
        print(f"🎯 Y+ coverage: {coverage:.1%} {yplus_status}")
        
        overall_status = '✅' if metrics.get('all_ok') else '❌'
        print(f"🏁 Overall: {overall_status}")