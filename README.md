# AortaCFD-Snappy: Advanced Mesh Optimization for Cardiovascular CFD

**Independent mesh optimization tool for patient-specific aortic blood flow simulations**

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-12-blue.svg)](https://openfoam.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Purpose

AortaCFD-Snappy provides physics-aware mesh optimization specifically designed for cardiovascular CFD simulations. It addresses the critical challenge of generating high-quality meshes with proper boundary layer resolution for accurate wall shear stress (WSS) calculations and pressure drop predictions.

### Key Problems Solved

- **Layer Generation Failures**: Automatic detection and recovery from boundary layer truncation
- **Non-Physics-Based Meshing**: Replaces trial-and-error with actual y+ = 1 targeting
- **Flow Regime Mismatch**: Provides regime-specific configurations (Laminar/RANS/LES)
- **QoI Validation**: Ensures meshes actually produce reliable velocity, pressure, and WSS results

---

## 🏗️ Two-Stage Architecture

### Stage 1: Geometry-Driven Mesh Generation (Inner Loop)
**Target Users**: Novice users, quick prototyping
```bash
python -m mesh_optim stage1 --geometry tutorial/patient1
```

- Iterates on surface refinement and boundary layer settings
- Quality criteria: non-orthogonality <65°, skewness <3.0, layer coverage >85%
- Pure geometry-based optimization (no CFD required)
- Output: Clean mesh with good boundary layers

### Stage 2: QoI-Driven Mesh Adaptation (Outer Loop)
**Target Users**: Advanced users, production meshes
```bash
python -m mesh_optim stage2 --geometry tutorial/patient1 --model RANS
```

- Physics-aware optimization with CFD validation
- Monitors y+, wall shear stress, and pressure drop accuracy
- Adapts mesh based on actual flow physics
- Output: Production-ready mesh validated against QoI criteria

---

## 🚀 Quick Start

### Prerequisites
- **OpenFOAM 12** (Foundation version)
- **Python 3.8+**
- **numpy, scipy** (for geometry calculations)

### Installation

**Recommended: Use a virtual environment to avoid dependency conflicts**

```bash
# Clone repository
git clone https://github.com/YourUsername/AortaCFD-Snappy.git
cd AortaCFD-Snappy

# Create and activate virtual environment (recommended)
python3 -m venv mesh_optim_env
source mesh_optim_env/bin/activate  # On Linux/Mac
# mesh_optim_env\Scripts\activate   # On Windows

# Install Python dependencies
pip install -r requirements.txt

# Ensure OpenFOAM is sourced
source /opt/openfoam12/etc/bashrc
```

**Alternative: System-wide installation**
```bash
git clone https://github.com/YourUsername/AortaCFD-Snappy.git
cd AortaCFD-Snappy
pip install -r requirements.txt
source /opt/openfoam12/etc/bashrc
```

### Basic Usage
```bash
# Quick geometry-driven mesh
python -m mesh_optim stage1 --geometry tutorial/patient1

# Physics-aware RANS mesh
python -m mesh_optim stage2 --geometry tutorial/patient1 --model RANS

# Wall-resolved LES mesh
python -m mesh_optim stage2 --geometry tutorial/patient1 --model LES
```

---

## 📁 Project Structure

```
AortaCFD-Snappy/
├── mesh_optim/                     # Main optimization package
│   ├── stage1_mesh.py              # Geometry-driven optimization
│   ├── stage2_qoi.py               # QoI-driven optimization
│   ├── utils.py                    # Common utilities
│   ├── __main__.py                 # CLI interface
│   └── configs/                    # Physics-aware configurations
│       ├── stage1_default.json     # Baseline settings
│       ├── stage2_laminar.json     # Laminar flow criteria
│       ├── stage2_rans.json        # RANS flow criteria
│       └── stage2_les.json         # LES flow criteria
├── tutorial/                       # Example patient case
│   └── patient1/
│       ├── inlet.stl, outlet*.stl, wall_aorta.stl
│       ├── BPM75.csv               # Flow velocity data
│       └── config.json             # Patient configuration
├── tools/                          # Additional utilities
│   └── checkmesh_yplus_eval.py     # Quality assessment tools
├── docs/                           # Documentation
│   └── USAGE.md                    # Detailed usage guide
└── requirements.txt                # Python dependencies
```

---

## 🔬 Physics-Aware Features

### Actual y+ Targeting
Calculates first layer thickness for y+ ≈ 1 using:
- Patient-specific peak velocity (from BPM75.csv)
- Actual inlet geometry (from STL files)
- Blood properties (ρ=1060 kg/m³, ν=3.77×10⁻⁶ m²/s)

**Formula**: h₁ = ν/u_τ where u_τ = √(0.5 C_f) × U_peak

### Flow Regime Optimization

| Regime | Target y+ | Layers | Expansion | Cell Count | Use Case |
|--------|-----------|---------|-----------|------------|----------|
| **Laminar** | < 1.0 | 8-10 | 1.25 | 1-5M | Re < 2300, steady flow |
| **RANS** | 0.5-2.0 | 12-15 | 1.20 | 5-15M | Clinical cases, turbulent |
| **LES** | 0.3-1.5 | 20-25 | 1.15 | 20-80M | Research, high fidelity |

### Distance Refinement
- **Physics-based**: 1.5mm and 3.0mm from wall (boundary layer scales)
- **Traditional**: Cell size multiples (not physically motivated)

---

## 📊 Command Reference

### Stage 1 Commands
```bash
# Basic mesh optimization
python -m mesh_optim stage1 --geometry tutorial/patient1

# Custom configuration
python -m mesh_optim stage1 --geometry tutorial/patient1 \
    --config mesh_optim/configs/stage1_default.json \
    --max-iterations 5 \
    --output results/patient1_stage1

# Verbose output
python -m mesh_optim stage1 --geometry tutorial/patient1 --verbose
```

### Stage 2 Commands
```bash
# RANS optimization (most common)
python -m mesh_optim stage2 --geometry tutorial/patient1 --model RANS

# Laminar flow
python -m mesh_optim stage2 --geometry tutorial/patient1 --model LAMINAR

# Wall-resolved LES
python -m mesh_optim stage2 --geometry tutorial/patient1 --model LES

# Custom QoI criteria
python -m mesh_optim stage2 --geometry tutorial/patient1 --model RANS \
    --config mesh_optim/configs/stage2_rans.json \
    --max-iterations 3
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--geometry PATH` | Directory with STL files | Required |
| `--model {LAMINAR,RANS,LES}` | Flow model (Stage 2 only) | RANS |
| `--config FILE` | Custom configuration file | Auto-selected |
| `--max-iterations N` | Maximum optimization iterations | 4 (Stage 1), 3 (Stage 2) |
| `--output DIR` | Output directory | Auto-generated |
| `--verbose` | Verbose logging | False |

---

## 🔧 Configuration Files

### Stage 1: Geometry-Driven (`stage1_default.json`)
```json
{
  "BLOCKMESH": {
    "resolution": 40,
    "grading": [1, 1, 1]
  },
  "SNAPPY": {
    "surface_level": [2, 3],
    "distance_refinement": {
      "near_distance": 2.0,
      "far_distance": 4.0
    }
  },
  "LAYERS": {
    "nSurfaceLayers": 10,
    "finalLayerThickness_rel": 0.25,
    "expansionRatio": 1.2
  },
  "acceptance_criteria": {
    "maxNonOrtho": 65,
    "maxSkewness": 3.0,
    "min_layer_coverage": 0.85
  }
}
```

### Stage 2: RANS QoI-Driven (`stage2_rans.json`)
```json
{
  "mesh_settings": {
    "base_resolution": 50,
    "surface_level": [3, 4]
  },
  "layer_settings": {
    "nSurfaceLayers": 12,
    "finalLayerThickness_rel": 0.20,
    "expansionRatio": 1.20
  },
  "quality_criteria": {
    "maxNonOrtho": 65,
    "min_layer_coverage": 0.90
  },
  "solver_settings": {
    "application": "simpleFoam",
    "endTime": 500
  },
  "qoi_criteria": {
    "min_yplus_coverage": 0.85,
    "target_yplus_range": [0.5, 2.0],
    "min_velocity_stability": 0.02,
    "min_wss_stability": 0.05
  }
}
```

---

## 📋 Input Requirements

### STL Files (Required)
```
geometry_directory/
├── inlet.stl           # Inlet surface
├── outlet1.stl         # Outlet 1
├── outlet2.stl         # Outlet 2 (etc.)
├── outlet3.stl         # 
├── outlet4.stl         #
└── wall_aorta.stl      # Aortic wall
```

### Flow Data (Optional but Recommended)
```
geometry_directory/
├── BPM75.csv           # Velocity vs time data
└── config.json         # Patient configuration
```

**BPM75.csv format:**
```csv
time,velocity
0.0,0.159660
0.01,0.275651
0.02,0.361402
...
```

---

## 📈 Quality Metrics & Validation

### Mesh Quality Checks
- **Non-orthogonality**: Target <65° for cardiovascular applications
- **Skewness**: Target <3.0 for stable numerics
- **Aspect Ratio**: Monitored but allowed up to 20 in boundary layers
- **Layer Coverage**: Target >85% successful boundary layer generation

### QoI Validation (Stage 2)
- **y+ Distribution**: 90% of wall area within target range
- **Velocity Stability**: <2% change between mesh refinements
- **Pressure Drop Accuracy**: <1% change in inlet-outlet Δp
- **WSS Reliability**: <5% change in area-averaged WSS

### Output Metrics
Each optimization produces `metrics.json`:
```json
{
  "iteration": 3,
  "checkMesh": {
    "maxNonOrtho": 62.3,
    "maxSkewness": 2.1,
    "cells": 2450000
  },
  "layerCoverage": {
    "coverage_overall": 0.92,
    "totalFaces": 145000
  },
  "qoi_metrics": {
    "yplus_coverage": 0.89,
    "velocity_stability": 0.015,
    "wss_availability": true
  }
}
```

---

## 🔍 Troubleshooting

### Common Issues

**1. Layer Generation Fails**
```
❌ Layer coverage: 0.0% - boundary layers failed to generate
```
**Solution**: The optimizer automatically adjusts thickness and layer count. For persistent failures, reduce `nSurfaceLayers` or increase `minThickness`.

**2. Non-Orthogonality Too High**  
```
❌ Max non-orthogonality: 78.2° (target: <65°)
```
**Solution**: Optimizer automatically reduces surface refinement level. Manual fix: reduce `surface_level`.

**3. OpenFOAM Not Found**
```
❌ Command failed: blockMesh
❌ Error: command not found
```
**Solution**: Source OpenFOAM environment:
```bash
source /opt/openfoam12/etc/bashrc
```

**4. STL Files Not Found**
```
❌ Required STL file not found: inlet.stl
```
**Solution**: Ensure geometry directory contains all required STL files with correct names.

**5. Python Package Import Errors**
```
❌ ModuleNotFoundError: No module named 'numpy'
❌ ImportError: cannot import name 'mesh_optim'
```
**Solution**: Ensure you're using the correct Python environment:
```bash
# If using virtual environment
source mesh_optim_env/bin/activate
pip install -r requirements.txt

# Verify installation
python -c "import numpy; print('Dependencies OK')"
```

### Debug Mode
```bash
python -m mesh_optim stage1 --geometry tutorial/patient1 --verbose
```

---

## 🏥 Clinical Applications

### Recommended Workflows

**Clinical Decision Support (Fast)**
```bash
python -m mesh_optim stage1 --geometry patient_data/
# ~10-30 minutes, good for qualitative analysis
```

**Research Publication (High Quality)**
```bash
python -m mesh_optim stage2 --geometry patient_data/ --model RANS
# ~1-3 hours, quantitative WSS and pressure drop
```

**Validation Studies (Highest Fidelity)**
```bash
python -m mesh_optim stage2 --geometry patient_data/ --model LES
# ~4-12 hours, research-grade accuracy
```

### Hemodynamic Analysis
After mesh optimization, use the generated mesh for:
- **WSS Analysis**: Time-averaged wall shear stress patterns
- **Pressure Drop**: Inlet-outlet pressure differences
- **Flow Patterns**: Velocity fields and secondary flows
- **Oscillatory Shear Index**: OSI for atherosclerosis risk

---

## 🤝 Contributing

We welcome contributions! Areas of interest:
- **Additional QoI metrics** (OSI, TAWSS, RRT)
- **Parallel mesh generation** for large cases
- **GUI interface** for clinical users
- **Validation against experimental data**

### Development Setup
```bash
git clone https://github.com/YourUsername/AortaCFD-Snappy.git
cd AortaCFD-Snappy

# Create development virtual environment
python3 -m venv dev_env
source dev_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Ensure OpenFOAM is sourced for development
source /opt/openfoam12/etc/bashrc
```

---

## 📚 References

1. **Mesh Generation**: Jasak, H. et al. "OpenFOAM: A C++ library for complex physics simulations"
2. **y+ Targeting**: Pope, S.B. "Turbulent Flows" - Chapter 7: Wall-bounded flows
3. **Cardiovascular CFD**: Taylor, C.A. & Figueroa, C.A. "Patient-specific modeling of cardiovascular mechanics"
4. **Boundary Layer Theory**: Schlichting, H. "Boundary Layer Theory" - 8th Edition

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/YourUsername/AortaCFD-Snappy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YourUsername/AortaCFD-Snappy/discussions)
- **Documentation**: [Wiki](https://github.com/YourUsername/AortaCFD-Snappy/wiki)

---

**AortaCFD-Snappy v1.0** - Making physics-aware mesh generation accessible for cardiovascular CFD