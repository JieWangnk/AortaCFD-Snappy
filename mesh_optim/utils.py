"""
Utilities for OpenFOAM mesh generation and quality assessment.
"""
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import math

def run_command(cmd, cwd=None, env_setup=None, timeout=None, max_memory_gb=8):
    """
    Execute a command with optional environment setup and memory limits.
    
    Args:
        cmd: Command to run (string or list)
        cwd: Working directory
        env_setup: Environment setup script path (e.g., "source /opt/openfoam12/etc/bashrc")
        timeout: Command timeout in seconds
        max_memory_gb: Maximum memory limit in GB
    
    Returns:
        Result object with stdout, stderr, and returncode
    """
    # Convert list commands to string
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    
    # Prepare the full command with environment setup
    if env_setup:
        # Source the environment first, then run the command
        full_cmd = f"{env_setup} && {cmd_str}"
    else:
        full_cmd = cmd_str
    
    # Run with bash -c to ensure proper environment sourcing
    process = subprocess.run(
        ["bash", "-c", full_cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    
    return process

def check_mesh_quality(mesh_dir, openfoam_env, max_memory_gb=8, wall_name="wall_aorta"):
    """
    Run checkMesh and parse quality metrics.
    
    Args:
        mesh_dir: Directory containing the mesh
        openfoam_env: OpenFOAM environment setup command
        max_memory_gb: Memory limit for checkMesh
        wall_name: Name of the wall patch
    
    Returns:
        Dictionary with mesh quality metrics
    """
    # Run checkMesh
    result = run_command(
        "checkMesh -writeSets vtk",
        cwd=mesh_dir,
        env_setup=openfoam_env,
        max_memory_gb=max_memory_gb
    )
    
    output = result.stdout + result.stderr
    
    # Parse metrics
    metrics = {}
    
    # Extract key metrics using regex
    patterns = {
        "cells": r"cells:\s+(\d+)",
        "faces": r"faces:\s+(\d+)",
        "points": r"points:\s+(\d+)",
        "maxNonOrtho": r"Max non-orthogonality\s*=\s*([\d.]+)",
        "maxSkewness": r"Max skewness\s*=\s*([\d.]+)",
        "minVol": r"Min volume\s*=\s*([-\d.eE]+)",
        "maxAspectRatio": r"Max aspect ratio\s*=\s*([\d.eE]+)",
        "minFaceWeight": r"Min face weight\s*=\s*([\d.eE]+)",
        "minVolRatio": r"Min volume ratio\s*=\s*([\d.eE]+)"
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
    
    # Extract boundary skewness separately
    boundary_skew_match = re.search(
        r"Max skewness\s*=\s*([\d.]+).*?\((\d+)\s+faces\)",
        output,
        re.IGNORECASE | re.DOTALL
    )
    if boundary_skew_match:
        metrics["maxBoundarySkewness"] = float(boundary_skew_match.group(1))
        metrics["nBadSkewFaces"] = int(boundary_skew_match.group(2))
    
    # Alternative patterns for different OF versions
    if "maxSkewness" not in metrics:
        alt_skew = re.search(r"Mesh face skewness.*?max:\s*([\d.]+)", output, re.IGNORECASE)
        if alt_skew:
            metrics["maxSkewness"] = float(alt_skew.group(1))
    
    if "maxNonOrtho" not in metrics:
        alt_ortho = re.search(r"Mesh non-orthogonality.*?max:\s*([\d.]+)", output, re.IGNORECASE)
        if alt_ortho:
            metrics["maxNonOrtho"] = float(alt_ortho.group(1))
    
    # Check for failed mesh checks
    metrics["failedChecks"] = []
    failed_patterns = [
        (r"\*\*\*.*faces with face pyramid volume", "negativePyramidVolume"),
        (r"\*\*\*.*faces with face-decomposition", "faceDecomposition"),
        (r"\*\*\*.*faces with concavity", "concaveFaces"),
        (r"\*\*\*.*faces with skewness", "skewFaces"),
        (r"\*\*\*.*severely non-orthogonal", "severeNonOrthogonal"),
        (r"\*\*\*.*faces with negative volume", "negativeVolume"),
        (r"\*\*\*.*cells with negative volume", "negativeCells")
    ]
    
    for pattern, check_name in failed_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            metrics["failedChecks"].append(check_name)
    
    # Extract wall patch information if available
    if wall_name:
        wall_pattern = rf"{wall_name}\s+\d+\s+(\d+)"
        wall_match = re.search(wall_pattern, output, re.MULTILINE)
        if wall_match:
            metrics["wall_nFaces"] = int(wall_match.group(1))
    
    # Use OpenFOAM's default mesh validation criteria
    # Trust OpenFOAM's checkMesh to determine if mesh is acceptable
    # Robust case-insensitive parsing for different OpenFOAM builds
    low = output.lower()
    metrics["meshOK"] = "mesh ok" in low and result.returncode == 0
    
    return metrics

def _to_float_safe(s: str) -> float:
    """Safely convert string to float, tolerating comma decimals and scientific notation"""
    return float(s.replace(',', '.'))

def parse_layer_coverage(mesh_dir, openfoam_env=None, max_memory_gb=8, wall_name="wall_aorta"):
    """
    Parse snappyHexMesh layer coverage from the layers pass log.
    Returns:
      {
        "coverage_overall": float in [0,1],
        "perPatch": {patch_name: float in [0,1]}
      }
    Robust across OF versions; if a header shows "[%]" we convert to fraction by /100.
    """
    from pathlib import Path
    
    iter_dir = Path(mesh_dir)
    logs_dir = iter_dir / "logs"
    candidates = [
        logs_dir / "log.snappy.layers",             # your primary path
        logs_dir / "log.snappyHexMesh.layers",      # alt naming
        logs_dir / "log.snappyHexMesh",             # sometimes single file
        logs_dir / "snappyHexMesh.log",             # other conventions
    ]
    text = ""
    for p in candidates:
        if p.exists():
            text = p.read_text(errors="ignore")
            if "addLayersControls" in text or "Layer mesh statistics" in text or "patch" in text:
                break
    if not text:
        # graceful fallback
        return {"coverage_overall": 0.0, "perPatch": {}}

    # Identify coverage table header block
    # Typical header (OpenFOAM 8–12):
    # patch      faces    layers   overall thickness
    #                              [m]       [%]
    # -----      -----    ------   ---       ---
    # wall_aorta 16170    0.00167  2.08e-08  0.026
    #
    # We will:
    #  - detect if last column is [%] and divide by 100
    #  - otherwise accept fraction as-is
    lines = text.splitlines()
    per_patch = {}
    coverage_overall = None

    # Try to locate the header lines and whether the last column is in percent
    header_idx = -1
    last_col_is_percent = False
    for i, line in enumerate(lines):
        l = line.lower()
        if ("patch" in l and "faces" in l and "layers" in l and "thickness" in l):
            header_idx = i
            # Peek next two lines for unit hints
            unit_hint = " ".join(lines[i+1:i+3]).lower() if i+2 < len(lines) else ""
            if "[%]" in unit_hint or "percent" in unit_hint:
                last_col_is_percent = True
            break

    # If header not found, try a simpler "overall coverage" regex only
    if header_idx < 0:
        m = re.search(r"overall\s+coverage[^:]*:\s*([\d.,eE+-]+)\s*%", text, flags=re.IGNORECASE)
        if m:
            val = _to_float_safe(m.group(1)) / 100.0
            return {"coverage_overall": max(0.0, min(1.0, val)), "perPatch": {}}
        return {"coverage_overall": 0.0, "perPatch": {}}

    # Parse rows until blank or separator ends
    row_re = re.compile(
        r"^\s*(?P<name>\S+)\s+"
        r"(?P<faces>\d+)\s+"
        r"(?P<layers>[-+]?[\d.,eE]+)\s+"
        r"(?P<thick_m>[-+]?[\d.,eE]+)\s+"
        r"(?P<last>[-+]?[\d.,eE]+)\s*$"
    )

    i = header_idx + 1
    # skip dashed separator lines
    while i < len(lines) and set(lines[i].strip()) <= set("- []m%"):
        i += 1

    while i < len(lines):
        line = lines[i].strip()
        if not line or set(line) <= set("- "):
            break
        m = row_re.match(line)
        if m:
            name = m.group("name")
            last = _to_float_safe(m.group("last"))
            # Auto-detect: if last > 1.0 it's likely percentage not fraction
            cov = (last / 100.0) if last_col_is_percent or last > 1.0 else last
            cov = max(0.0, min(1.0, cov))
            per_patch[name] = cov
        else:
            # stop at first non-table-ish line
            if line and not line[0].isalnum():
                break
        i += 1

    # Also pick up any "Overall coverage: 31.5 %" style line (some builds print it)
    m_over = re.search(r"overall\s+coverage[^:]*:\s*([\d.,eE+-]+)\s*%", text, flags=re.IGNORECASE)
    if m_over:
        coverage_overall = _to_float_safe(m_over.group(1)) / 100.0

    # If no explicit overall, compute weighted by faces from table if available
    if coverage_overall is None and per_patch:
        # Try to extract faces from table rows we parsed
        total_faces, sum_cov_faces = 0, 0
        # Re-scan table for faces with the patches we recorded
        i = header_idx + 1
        # Skip separators again
        while i < len(lines) and set(lines[i].strip()) <= set("- []m%"):
            i += 1
        while i < len(lines):
            ml = row_re.match(lines[i].strip())
            if ml:
                nm = ml.group("name")
                if nm in per_patch:
                    nf = int(ml.group("faces"))
                    total_faces += nf
                    sum_cov_faces += per_patch[nm] * nf
                i += 1
            else:
                break
        coverage_overall = (sum_cov_faces / total_faces) if total_faces > 0 else 0.0

    # Use the wall patch coverage as overall if no explicit overall found
    if coverage_overall is None and wall_name in per_patch:
        coverage_overall = per_patch[wall_name]
    
    # Final fallback
    if coverage_overall is None:
        coverage_overall = 0.0

    return {"coverage_overall": coverage_overall, "perPatch": per_patch}

def calculate_first_layer_thickness(peak_velocity, diameter, blood_properties):
    """
    Calculate first layer thickness for target y+ = 1
    
    Args:
        peak_velocity: Peak velocity in m/s
        diameter: Reference diameter in m
        blood_properties: Dict with 'density' and 'viscosity'
    
    Returns:
        First layer thickness in meters
    """
    rho = blood_properties.get('density', 1060)  # kg/m³
    mu = blood_properties.get('viscosity', 0.0035)  # Pa·s
    
    # Calculate Reynolds number
    Re = rho * peak_velocity * diameter / mu
    
    # Wall shear stress estimation (Blasius approximation)
    Cf = 0.079 * Re**(-0.25)  # Skin friction coefficient
    tau_w = 0.5 * rho * peak_velocity**2 * Cf
    
    # Friction velocity
    u_tau = math.sqrt(tau_w / rho)
    
    # First layer thickness for y+ = 1
    y_plus_target = 1.0
    nu = mu / rho  # Kinematic viscosity
    delta_y = y_plus_target * nu / u_tau
    
    return delta_y

def estimate_cell_count(base_size, domain_volume, refinement_levels):
    """
    Estimate total cell count based on base size and refinement.
    
    Args:
        base_size: Base cell size in meters
        domain_volume: Domain volume in m³
        refinement_levels: List of (volume_fraction, level) tuples
    
    Returns:
        Estimated total cell count
    """
    # Base cell volume
    base_cell_volume = base_size**3
    base_cells = domain_volume / base_cell_volume
    
    # Account for refinement
    total_cells = base_cells
    for volume_fraction, level in refinement_levels:
        # Each refinement level divides cells by 8
        refinement_factor = 8**level
        refined_cells = base_cells * volume_fraction * (refinement_factor - 1)
        total_cells += refined_cells
    
    return int(total_cells)