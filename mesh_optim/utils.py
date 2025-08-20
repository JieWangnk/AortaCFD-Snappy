"""
Utility functions for mesh optimization
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import re
import logging

def run_command(cmd, cwd=None, env_setup=None, timeout=None, parallel=False):
    """
    Run OpenFOAM command with proper environment setup
    
    Args:
        cmd: Command to run (list or string)
        cwd: Working directory
        env_setup: Path to OpenFOAM environment script
        timeout: Timeout in seconds
        parallel: Whether this is a parallel command
    """
    if env_setup:
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
        
        # Use bash explicitly for OpenFOAM environment
        full_cmd = f"bash -c 'source {env_setup} && {cmd_str}'"
        cmd = ["bash", "-c", f"source {env_setup} && {cmd_str}"]
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str) and not env_setup
        )
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout} seconds: {cmd}")
    except Exception as e:
        raise RuntimeError(f"Command failed: {e}")

def check_mesh_quality(mesh_dir, openfoam_env):
    """
    Run checkMesh and parse quality metrics
    """
    log_file = mesh_dir / "logs" / "log.checkMesh"
    log_file.parent.mkdir(exist_ok=True)
    
    result = run_command(
        "checkMesh -allGeometry -allTopology", 
        cwd=mesh_dir, 
        env_setup=openfoam_env
    )
    
    # Write log
    log_file.write_text(result.stdout + result.stderr)
    
    # Parse metrics
    metrics = {
        "maxNonOrtho": 0,
        "maxSkewness": 0,
        "maxAspectRatio": 0,
        "negVolCells": 0,
        "meshOK": False,
        "cells": 0
    }
    
    output = result.stdout + result.stderr
    
    # Parse key metrics
    if "non-orthogonality" in output:
        match = re.search(r"Max non-orthogonality = ([\d.]+)", output)
        if match:
            metrics["maxNonOrtho"] = float(match.group(1))
    
    if "skewness" in output:
        match = re.search(r"Max skewness = ([\d.]+)", output)  
        if match:
            metrics["maxSkewness"] = float(match.group(1))
    
    if "aspect ratio" in output:
        match = re.search(r"aspect ratio = ([\d.]+)", output)
        if match:
            metrics["maxAspectRatio"] = float(match.group(1))
            
    if "cells:" in output:
        match = re.search(r"cells:\s+(\d+)", output)
        if match:
            metrics["cells"] = int(match.group(1))
    
    metrics["meshOK"] = "Mesh OK" in output and result.returncode == 0
    
    return metrics

def parse_layer_coverage(mesh_dir, openfoam_env):
    """
    Parse boundary layer coverage from snappyHexMesh logs
    """
    log_file = mesh_dir / "logs" / "log.snappyHexMesh.layers"
    
    if not log_file.exists():
        return {"coverage_overall": 0.0, "totalFaces": 0, "perPatch": {}}
    
    log_content = log_file.read_text()
    
    # Parse layer statistics
    coverage_data = {"coverage_overall": 0.0, "totalFaces": 0, "perPatch": {}}
    
    # Look for "Layer thickness" section
    if "Layer thickness" in log_content:
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            if "wall_aorta" in line and "faces" in line:
                # Parse: wall_aorta: 27129 faces, 6.000 avg layers, 0.4%
                match = re.search(r"(\d+) faces.*?([\d.]+) avg layers", line)
                if match:
                    n_faces = int(match.group(1))
                    avg_layers = float(match.group(2))
                    
                    coverage_data["perPatch"]["wall_aorta"] = {
                        "nFaces": n_faces,
                        "avgLayers": avg_layers,
                        "coverage": 1.0 if avg_layers > 1 else 0.0
                    }
                    
                    coverage_data["totalFaces"] = n_faces
                    coverage_data["coverage_overall"] = 1.0 if avg_layers > 1 else 0.0
    
    return coverage_data

def calculate_first_layer_thickness(peak_velocity, diameter, blood_properties):
    """
    Calculate first layer thickness for target y+ = 1
    
    Args:
        peak_velocity: Peak velocity in m/s
        diameter: Characteristic diameter in m
        blood_properties: Dict with density (kg/m3) and kinematic_viscosity (m2/s)
    """
    rho = blood_properties.get("density", 1060)
    nu = blood_properties.get("kinematic_viscosity", 3.77e-6)
    
    # Reynolds number
    Re = peak_velocity * diameter / nu
    
    # Skin friction coefficient (turbulent approximation)
    Cf = 0.079 * (Re**-0.25)
    
    # Friction velocity
    u_tau = np.sqrt(0.5 * Cf) * peak_velocity
    
    # First layer height for y+ = 1
    h1 = nu / u_tau
    
    return h1

def estimate_geometry_parameters(stl_dir):
    """
    Estimate geometry parameters from STL files
    """
    # For now, return defaults based on typical aortic dimensions
    # TODO: Implement actual STL parsing
    return {
        "inlet_diameter": 0.0137,  # 13.7 mm in meters
        "characteristic_length": 0.094,  # ~94 mm total length
        "volume": 5e-6  # ~5 cm³ approximate volume
    }