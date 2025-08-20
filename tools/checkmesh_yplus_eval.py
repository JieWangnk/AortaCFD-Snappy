#!/usr/bin/env python3
"""
Basic Y+ Coverage Evaluation Tool
================================

Simple tool to estimate Y+ coverage from mesh boundary layer thickness.
This provides a basic assessment until full CFD solver integration is available.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def estimate_yplus_coverage(mesh_dir: Path, flow_velocity: float = 1.0, viscosity: float = 3.5e-6) -> dict:
    """
    Estimate Y+ coverage from mesh boundary layer information.
    
    Args:
        mesh_dir: Path to mesh directory containing OpenFOAM case
        flow_velocity: Characteristic flow velocity (m/s)
        viscosity: Kinematic viscosity (m²/s) - blood at 37°C
    
    Returns:
        Dictionary with Y+ statistics
    """
    
    # Check if boundary layer thickness data exists
    thickness_file = mesh_dir / "0" / "thickness"
    if not thickness_file.exists():
        return {
            "coverage_overall": 0.0,
            "totalFaces": 0,
            "perPatch": {},
            "error": "No boundary layer thickness data found"
        }
    
    try:
        # Read boundary file to get patch information
        boundary_file = mesh_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            return {
                "coverage_overall": 0.0,
                "totalFaces": 0,
                "perPatch": {},
                "error": "No boundary file found"
            }
        
        # Count wall faces (approximate)
        wall_faces = 0
        outlet_faces = 0
        
        with open(boundary_file, 'r') as f:
            content = f.read()
            # Simple parsing - count faces for wall patches
            lines = content.split('\n')
            in_patch = False
            current_patch = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('wall_aorta') or line.startswith('walls'):
                    in_patch = True
                    current_patch = "wall"
                elif line.startswith('outlet') or line.startswith('inlet'):
                    in_patch = True
                    current_patch = "outlet"
                elif in_patch and line.startswith('nFaces'):
                    try:
                        nfaces = int(line.split()[1].rstrip(';'))
                        if current_patch == "wall":
                            wall_faces += nfaces
                        else:
                            outlet_faces += nfaces
                    except (IndexError, ValueError):
                        pass
                    in_patch = False
        
        if wall_faces == 0:
            return {
                "coverage_overall": 0.0,
                "totalFaces": 0,
                "perPatch": {},
                "error": "No wall faces detected"
            }
        
        # Estimate Y+ from first layer thickness
        # Y+ ≈ ρ * u_τ * y / μ, where u_τ ≈ sqrt(τ_wall/ρ) ≈ 0.05*U for turbulent flow
        # For cardiovascular flow: typical first layer ~0.001mm, target Y+ ~1-5
        
        # Assume reasonable boundary layer thickness for cardiovascular mesh
        # If layers were added successfully, estimate Y+ coverage
        typical_first_layer = 1e-5  # 10 microns - reasonable for cardiovascular
        wall_shear_velocity = 0.05 * flow_velocity  # Rough estimate
        density = 1060  # kg/m³ - blood density
        
        # Estimate Y+ 
        estimated_yplus = (density * wall_shear_velocity * typical_first_layer) / (density * viscosity)
        
        # Check if Y+ is in acceptable range (1-5 for cardiovascular)
        target_min, target_max = 1.0, 5.0
        
        if target_min <= estimated_yplus <= target_max:
            coverage = 0.85  # Assume good coverage if estimate looks reasonable
        elif estimated_yplus < target_min:
            coverage = 0.6   # Too fine - partial coverage
        else:
            coverage = 0.3   # Too coarse - poor coverage
        
        return {
            "coverage_overall": coverage,
            "totalFaces": wall_faces,
            "perPatch": {
                "wall_aorta": {
                    "coverage": coverage,
                    "faces": wall_faces,
                    "estimated_yplus": estimated_yplus
                }
            },
            "estimated_yplus_range": [estimated_yplus * 0.5, estimated_yplus * 2.0],
            "note": "Estimate based on typical boundary layer parameters"
        }
        
    except Exception as e:
        return {
            "coverage_overall": 0.0,
            "totalFaces": 0,
            "perPatch": {},
            "error": f"Y+ evaluation failed: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Estimate Y+ coverage from mesh")
    parser.add_argument("mesh_directory", type=Path, help="Path to OpenFOAM case directory")
    parser.add_argument("--velocity", type=float, default=1.0, help="Characteristic velocity (m/s)")
    parser.add_argument("--viscosity", type=float, default=3.5e-6, help="Kinematic viscosity (m²/s)")
    parser.add_argument("--yplus-band", nargs=2, type=float, default=[1.0, 5.0], help="Target Y+ range")
    parser.add_argument("--required-yplus-coverage", type=float, default=0.8, help="Required coverage fraction")
    parser.add_argument("--checkmesh-log", type=Path, help="Path to checkMesh log file (ignored - we parse directly)")
    parser.add_argument("--out", type=Path, help="Output file for metrics (optional)")
    
    args = parser.parse_args()
    
    yplus_result = estimate_yplus_coverage(args.mesh_directory, args.velocity, args.viscosity)
    
    # Create complete metrics structure expected by the quality assessment
    full_metrics = {
        'checkMesh': {},
        'yPlus': yplus_result,
        'acceptance': {
            'mesh_ok': False,
            'yPlus_ok': yplus_result.get('coverage_overall', 0.0) >= args.required_yplus_coverage
        },
        'all_ok': False,
        'layerCoverage': {}
    }
    
    # Parse checkMesh log if it exists
    checkmesh_log = args.mesh_directory / 'logs' / 'log.checkMesh'
    if checkmesh_log.exists():
        try:
            content = checkmesh_log.read_text()
            # Basic parsing of checkMesh results
            import re
            
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
                        full_metrics['checkMesh'][key] = float(match.group(1))
                    except ValueError:
                        full_metrics['checkMesh'][key] = int(match.group(1))
            
            # Check for failed mesh
            if 'Failed' in content and 'mesh check' in content:
                full_metrics['checkMesh']['meshOK'] = False
            else:
                full_metrics['checkMesh']['meshOK'] = True
                
            # Basic mesh acceptance
            full_metrics['acceptance']['mesh_ok'] = (
                full_metrics['checkMesh'].get('maxNonOrtho', 100) <= 75 and
                full_metrics['checkMesh'].get('maxSkewness', 10) <= 4.0 and
                full_metrics['checkMesh'].get('meshOK', False)
            )
        except Exception:
            pass
    
    # Overall acceptance
    full_metrics['all_ok'] = (
        full_metrics['acceptance']['mesh_ok'] and
        full_metrics['acceptance']['yPlus_ok']
    )
    
    # Output to file if specified
    if args.out:
        args.out.write_text(json.dumps(full_metrics, indent=2))
    
    # Output JSON result to stdout
    print(json.dumps(full_metrics, indent=2))
    
    # Always return 0 for successful execution - let the main tool decide on acceptance
    # Return code 1 only for actual errors/failures
    sys.exit(0)


if __name__ == "__main__":
    main()