"""
Pytest configuration and shared fixtures for Stage1 mesh optimization tests.
"""
import pytest
import tempfile
import json
from pathlib import Path
import numpy as np


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "openfoam_env_path": "source /opt/openfoam12/etc/bashrc",
        "base_size": 0.001,
        "physics": {
            "rho": 1060,
            "mu": 0.0035,
            "peak_velocity": 1.5,
            "heart_rate_hz": 1.2,
            "solver_mode": "RANS"
        },
        "snappyHexMeshDict": {
            "castellatedMeshControls": {
                "maxLocalCells": 100000,
                "maxGlobalCells": 500000,
                "nCellsBetweenLevels": 1
            },
            "snapControls": {
                "nSmoothPatch": 3,
                "tolerance": 2.0,
                "nRelaxIter": 5
            },
            "addLayersControls": {
                "nSurfaceLayers": 8,
                "firstLayerThickness_abs": 50e-6,
                "minThickness_abs": 5e-6,
                "expansionRatio": 1.2,
                "nGrow": 1,
                "featureAngle": 70,
                "maxThicknessToMedialRatio": 0.45
            }
        },
        "targets": {
            "max_nonortho": 65,
            "max_skewness": 4.0,
            "min_layer_cov": 0.65
        }
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config):
    """Create sample configuration file"""
    config_file = temp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    return config_file


@pytest.fixture
def simple_stl_geometry(temp_dir):
    """Create simple STL geometry file for testing"""
    stl_content = """solid test_geometry
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 0.5 1.0 0.0
    endloop
  endfacet
  facet normal 0.0 0.0 -1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 0.5 1.0 0.0
      vertex 1.0 0.0 0.0
    endloop
  endfacet
endsolid test_geometry
"""
    stl_file = temp_dir / "wall_aorta.stl"
    stl_file.write_text(stl_content)
    return stl_file


@pytest.fixture
def sample_mesh_metrics():
    """Sample mesh quality metrics for testing"""
    return {
        "cells": 50000,
        "faces": 150000,
        "points": 75000,
        "maxNonOrtho": 45.0,
        "maxSkewness": 2.5,
        "maxAspectRatio": 50.0,
        "meshOK": True,
        "failedChecks": []
    }


@pytest.fixture
def sample_layer_coverage():
    """Sample layer coverage data for testing"""
    return {
        "coverage_overall": 0.85,
        "perPatch": {
            "wall_aorta": 0.85,
            "inlet": 0.92,
            "outlet": 0.88
        }
    }


@pytest.fixture
def mock_openfoam_env():
    """Mock OpenFOAM environment setup"""
    return "echo 'OpenFOAM environment setup'"


@pytest.fixture
def geometry_info_sample():
    """Sample geometry information"""
    return {
        "valid": True,
        "characteristic_length": 0.02,
        "bounding_box": {
            "min": [0.0, 0.0, 0.0],
            "max": [0.1, 0.02, 0.02]
        },
        "surface_area": 0.005,
        "volume": 0.0001,
        "inlet_diameter": 0.018,
        "outlet_diameter": 0.016,
        "scaling_applied": 0.001
    }