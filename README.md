# AortaCFD-Snappy: Two-Stage Mesh Optimization for Cardiovascular CFD

**Literature-backed mesh optimization pipeline for vessel-agnostic blood flow simulations**

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-12-blue.svg)](https://openfoam.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

AortaCFD-Snappy provides a robust two-stage mesh optimization pipeline for cardiovascular CFD simulations across **all vascular beds**. The geometry-aware Stage 1 optimizer automatically derives mesh parameters from actual vessel dimensions, while the literature-backed Stage 2 verifier uses Richardson extrapolation and Grid Convergence Index (GCI) analysis for physics-verified meshes.

### Key Features
- **Vessel-Agnostic Design**: Works across aortas, carotids, coronaries, and other vascular geometries
- **Constraint-Based Optimization**: Uses hard quality constraints instead of penalty functions
- **Literature-Backed Verification**: Stage 2 implements Richardson extrapolation with 5% WSS tolerance
- **Hierarchical Mesh Control**: Multiple override levels for precise control when needed
- **Crash-Safe Operation**: Resource-aware scaling prevents terminal crashes

---

## 🏗️ Two-Stage Architecture

### Stage 1: Geometry-Driven Mesh Generation
**Purpose**: Generate high-quality baseline mesh with proper boundary layers
**Input**: STL geometry files
**Output**: Constraint-verified mesh ready for CFD or Stage 2

```bash
python -m mesh_optim stage1 --geometry tutorial/patient1
```

**What it does:**
- Analyzes geometry to derive reference diameters (D_ref, D_min) from inlet/outlet areas
- Calculates geometry-aware base cell size: Δx = min(D_ref/N_D, D_min/N_D_min)
- Uses adaptive resolveFeatureAngle (30-45°) based on vessel complexity
- Applies gentle surface refinement ladder: [1,1] → [1,2] → [2,3]
- Generates boundary layers with constraint-based acceptance criteria
- **Optimization Goal**: Minimize cell count subject to quality constraints

**Quality Constraints:**
- maxNonOrtho ≤ 65°
- maxSkewness ≤ 4.0  
- Wall layer coverage ≥ 70%

### Stage 2: Physics-Verified Mesh Convergence
**Purpose**: Multi-level GCI verification for physics-accurate WSS calculations
**Input**: Stage 1 "best" mesh + config
**Output**: Convergence-verified mesh with literature-backed accuracy

```bash
python -m mesh_optim stage2 --geometry tutorial/patient1 --model LAMINAR
```

**What it does:**
- Builds three mesh levels (coarse, medium, fine) at refinement ratio r=1.3
- Maintains identical physics between levels (same y+, CFL, averaging windows)
- Performs Richardson extrapolation: M_∞ = M_fine + (M_fine - M_medium)/(r^p - 1)
- Calculates Grid Convergence Index: GCI = |1.25 * (M_fine - M_medium) / (M_fine * (r^p - 1))| * 100%
- **Convergence Decision**: GCI ≤ 5% (literature standard for cardiovascular WSS)

---

## 📋 Usage & Options

### Stage 1 Commands

#### Basic Usage
```bash
# Standard geometry-driven optimization
python -m mesh_optim stage1 --geometry tutorial/patient1

# With custom settings
python -m mesh_optim stage1 --geometry tutorial/patient1 \
    --config mesh_optim/configs/stage1_default.json \
    --max-iterations 4 \
    --verbose
```

#### Stage 1 Options
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--geometry PATH` | Directory containing STL files | Required | `tutorial/patient1` |
| `--config FILE` | Custom configuration file | `stage1_default.json` | `custom_config.json` |
| `--max-iterations N` | Maximum optimization iterations | `4` | `6` |
| `--output DIR` | Output directory | Auto-generated | `results/stage1` |
| `--verbose` | Detailed logging | `False` | - |

### Stage 2 Commands

#### Basic Usage
```bash
# LAMINAR flow verification (low Re, steady)
python -m mesh_optim stage2 --geometry tutorial/patient1 --model LAMINAR

# RANS flow verification (turbulent, clinical)  
python -m mesh_optim stage2 --geometry tutorial/patient1 --model RANS

# LES flow verification (research, high-fidelity)
python -m mesh_optim stage2 --geometry tutorial/patient1 --model LES
```

#### Stage 2 Options
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--geometry PATH` | Directory containing STL files | Required | `tutorial/patient1` |
| `--model {LAMINAR,RANS,LES}` | Flow physics model | Required | `LAMINAR` |
| `--config FILE` | Custom configuration file | Auto-selected | `stage2_custom.json` |
| `--max-iterations N` | Max iterations per mesh level | `4` | `6` |
| `--output DIR` | Output directory | Auto-generated | `results/stage2` |
| `--verbose` | Detailed logging | `False` | - |

### Flow Model Details

| Model | Use Case | Re Range | Target y+ | Mesh Levels | Expected Runtime |
|-------|----------|----------|-----------|-------------|------------------|
| **LAMINAR** | Low Re, steady flow | < 2300 | < 1.0 | Coarse/Medium/Fine | 30-60 min |
| **RANS** | Clinical cases, turbulent | > 2300 | 0.5-2.0 | Coarse/Medium/Fine | 1-3 hours |  
| **LES** | Research, unsteady | > 4000 | 0.3-1.5 | Coarse/Medium/Fine | 3-8 hours |

---

## 📁 Input/Output Structure

### Input Requirements

#### STL Geometry Files (Required)
```
tutorial/patient1/
├── inlet.stl              # Inlet surface
├── outlet1.stl            # Outlet surfaces  
├── outlet2.stl            # (multiple outlets supported)
├── outlet3.stl            
├── outlet4.stl            
└── wall_aorta.stl         # Vessel wall (auto-detected patterns: wall_*, vessel_*, arterial_*)
```

#### Flow Data (Optional)
```
tutorial/patient1/
├── BPM75.csv              # Velocity vs time (for pulsatile effects)
└── config.json            # Patient-specific parameters
```

### Output Structure

#### Stage 1 Output
```
output/patient1/meshOptimizer/stage1/
├── iter_001/              # First iteration
│   ├── constant/polyMesh/ # OpenFOAM mesh
│   ├── system/           # Case dictionaries
│   └── logs/             # Solver logs
├── iter_002/              # Subsequent iterations...
├── best/                  # Best quality mesh (exported for Stage 2)
│   ├── constant/polyMesh/ # Final mesh
│   ├── config.json        # Preserved configuration
│   └── stage1_metrics.json # Quality metrics
└── stage1_summary.csv     # Iteration summary
```

#### Stage 2 Output
```
output/patient1/meshOptimizer/stage2_gci_laminar/
├── coarse/                # Coarsest mesh level
│   └── constant/polyMesh/
├── medium/                # Medium mesh level  
│   └── constant/polyMesh/
├── fine/                  # Finest mesh level
│   └── constant/polyMesh/
├── gci_analysis.json      # Richardson extrapolation results
└── convergence_report.json # Final convergence decision
```

### Key Output Files

#### stage1_summary.csv
```csv
iter,cells,maxNonOrtho,maxSkewness,coverage,objective_dummy,levels_min,levels_max,resolveFeatureAngle,nLayers,firstLayer,minThickness
1,3536368,0.0,1.76,0.000,0.000,1,1,35,12,5.000e-05,2.000e-05
2,3536368,0.0,1.76,0.000,0.000,1,2,35,14,5.000e-05,2.000e-05
```

#### gci_analysis.json (Stage 2)
```json
{
  "mesh_levels": {
    "coarse": {"cells": 3536368, "TAWSS_mean": 2.150},
    "medium": {"cells": 4621678, "TAWSS_mean": 2.086}, 
    "fine": {"cells": 6008581, "TAWSS_mean": 2.068}
  },
  "richardson_analysis": {
    "apparent_order": 1.92,
    "extrapolated_value": 2.058,
    "gci_fine_medium": 2.3
  },
  "convergence_decision": {
    "converged": true,
    "tolerance_pct": 5.0,
    "recommended_mesh": "medium"
  }
}
```

---

## ⚙️ Configuration Control

### Hierarchical Mesh Control (BLOCKMESH)

The system supports multiple override levels for precise control:

```json
{
  "BLOCKMESH": {
    "min_per_axis": [12, 12, 12],
    // Precedence: divisions > cell_size_m > resolution > geometry-aware
    
    // Option 1: Exact control (highest priority)
    "divisions": [220, 140, 150],
    
    // Option 2: Force specific cell size
    "cell_size_m": 4e-4,
    
    // Option 3: Target resolution along longest axis  
    "resolution": 80,
    
    // Option 4: Geometry-aware (default - no override needed)
  }
}
```

#### Quick Control Reference:
- **Want exact background cells?** Set `BLOCKMESH.divisions` (wins over everything)
- **Want a specific background Δx?** Set `BLOCKMESH.cell_size_m`
- **Want "cells along longest axis"?** Set `BLOCKMESH.resolution`
- **To force non-adaptive feature angle,** set `"GEOMETRY_POLICY": { "featureAngle_mode": "ladder" }`

### Quality Acceptance Criteria

Both stages use unified, literature-backed acceptance criteria:

```json
{
  "acceptance_criteria": {
    "maxNonOrtho": 65,        // OpenFOAM guidance: ≤65° for stability
    "maxSkewness": 4.0,       // Literature standard: ≤4.0 for accuracy  
    "min_layer_coverage": 0.70 // RANS wall-function requirement
  }
}
```

### Feature Angle Settings

Adaptive resolveFeatureAngle based on OpenFOAM documentation:

```json
{
  "_parameter_guidance": {
    "resolveFeatureAngle": "30-45° adaptive range. Start at 30°; increase only if surface refinement becomes excessive",
    "includedAngle": "140-160° is appropriate for vascular lips/ostia. 150° is fine",
    "mergeTolerance": "min(1e-5, 0.05*Δx) is safe. If over-merging at tight throats, cap at 1e-6"
  }
}
```

---

## 🔬 Literature-Backed Methodology

### Stage 1: Constraint-Based Optimization
- **Objective**: Minimize cells subject to hard quality constraints (no penalty functions)
- **Surface Refinement**: Gentle ladder progression [1,1] → [1,2] → [2,3] 
- **Feature Detection**: Adaptive resolveFeatureAngle 30-45° per OpenFOAM guidance
- **Layer Generation**: Physics-based thickness with 70% coverage threshold

### Stage 2: GCI Verification (Roache, 1998)
- **Richardson Extrapolation**: M_∞ = M_h + (M_h - M_2h)/(r^p - 1)
- **Grid Convergence Index**: GCI = |1.25 * ε / M_h * (r^p - 1)| * 100%
- **Convergence Criterion**: GCI ≤ 5% (cardiovascular WSS standard)
- **Mesh Ratio**: r = 1.3 (optimal for Richardson analysis)

### Key References
1. **Richardson, L.F. (1911)**: "The approximate arithmetical solution by finite differences"
2. **Roache, P.J. (1998)**: "Verification and Validation in Computational Science and Engineering"
3. **Expert Consensus (2019)**: "5% WSS tolerance for cardiovascular CFD verification"

---

## 🚀 Quick Start Examples

### Aortic Coarctation (Large Vessel)
```bash
# Stage 1: Generate baseline mesh
python -m mesh_optim stage1 --geometry cases/aortic_coarctation/
# Expected: D_ref ≈ 25mm, Δx ≈ 1.1mm, ~2-4M cells

# Stage 2: Verify with RANS (clinical standard)
python -m mesh_optim stage2 --geometry cases/aortic_coarctation/ --model RANS
# Expected: 3 mesh levels, GCI analysis, ~2-4 hours
```

### Coronary Artery (Small Vessel) 
```bash
# Stage 1: Fine resolution for small vessels
python -m mesh_optim stage1 --geometry cases/coronary_lad/
# Expected: D_ref ≈ 3mm, Δx ≈ 0.14mm, ~0.5-1.5M cells

# Stage 2: Laminar flow (low Reynolds number)
python -m mesh_optim stage2 --geometry cases/coronary_lad/ --model LAMINAR
# Expected: Faster convergence, ~30-60 minutes
```

### Research-Grade LES Study
```bash
# Stage 1: High-quality baseline
python -m mesh_optim stage1 --geometry cases/research_case/ --max-iterations 6

# Stage 2: Wall-resolved LES verification
python -m mesh_optim stage2 --geometry cases/research_case/ --model LES  
# Expected: Very fine meshes, y+ < 1, ~4-8 hours
```

---

## 🔧 Advanced Usage

### Custom Configuration
```bash
# Use custom Stage 1 parameters
python -m mesh_optim stage1 --geometry tutorial/patient1 \
    --config my_configs/high_resolution.json

# Custom Stage 2 with specific tolerances  
python -m mesh_optim stage2 --geometry tutorial/patient1 --model RANS \
    --config my_configs/strict_convergence.json
```

### Debugging & Verbose Output
```bash
# Detailed logging for troubleshooting
python -m mesh_optim stage1 --geometry tutorial/patient1 --verbose

# Check mesh quality after Stage 1
checkMesh -case output/patient1/meshOptimizer/stage1/best/
```

---

## ✅ Validation & Quality Assurance

### Mesh Quality Metrics
- **Geometric Quality**: Non-orthogonality, skewness, aspect ratio within OpenFOAM guidelines
- **Boundary Layer Coverage**: >70% successful layer generation (realistic for complex vessels)
- **Feature Capture**: Adaptive angle detection preserves important geometric features

### Physics Verification (Stage 2)
- **Grid Independence**: Richardson extrapolation validates mesh-independent solutions
- **WSS Accuracy**: 5% tolerance ensures reliable wall shear stress calculations  
- **Pressure Drop Consistency**: Verified across mesh refinement levels

### Literature Compliance
- **OpenFOAM Best Practices**: Feature angles, quality thresholds, solver settings
- **CFD Verification Standards**: GCI methodology, Richardson extrapolation
- **Cardiovascular CFD Guidelines**: WSS tolerance, y+ requirements, time-averaging

---

## 🤝 Contributing

Areas for contribution:
- **Additional Vascular Beds**: Cerebral, peripheral, pediatric geometries
- **Enhanced Physics Models**: Fluid-structure interaction, non-Newtonian blood
- **Validation Studies**: Experimental comparison, benchmark cases
- **Performance Optimization**: Parallel mesh generation, GPU acceleration

---

## 📚 Citation

If you use AortaCFD-Snappy in your research, please cite:

```bibtex
@software{aortacfd_snappy,
  title={AortaCFD-Snappy: Two-Stage Mesh Optimization for Cardiovascular CFD},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[username]/AortaCFD-Snappy}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**AortaCFD-Snappy v2.0** - Literature-backed mesh optimization for cardiovascular CFD