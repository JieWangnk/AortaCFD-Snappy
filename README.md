# AortaCFD-Snappy: Automated Mesh Optimization for Arterial Geometries

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenFOAM Version](https://img.shields.io/badge/OpenFOAM-12-blue.svg)](https://openfoam.org/)
[![Python Version](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

A robust, flexible mesh generation tool for cardiovascular CFD simulations using OpenFOAM's snappyHexMesh with intelligent quality assessment and adaptive refinement strategies.

## 🌟 Key Features

- **Adaptive Refinement**: Intelligent surface refinement with automatic failure recovery
- **Two-Pass Boundary Layers**: Professional-grade boundary layer generation with phase-specific quality controls
- **Multi-Geometry Support**: Works with any arterial geometry (any number of outlets)
- **Quality Assessment**: Comprehensive mesh quality analysis with industry-standard criteria
- **Robust Recovery**: Automatic detection and recovery from surface intersection failures
- **Publication Ready**: Clean, documented code suitable for research applications

## 🚀 Quick Start

### Prerequisites

- OpenFOAM 12
- Python 3.8+
- numpy, numpy-stl packages

### Installation

```bash
git clone https://github.com/your-repo/AortaCFD-Snappy.git
cd AortaCFD-Snappy
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run optimization with default settings
python mesh_loop.py --geometry patient1

# Custom configuration with solver evaluation
python mesh_loop.py --geometry patient1 --config config/custom.json --enable-solver

# Verbose output with maximum iterations
python mesh_loop.py --geometry patient1 --max-iterations 15 --verbose
```

## 📁 Repository Structure

```
AortaCFD-Snappy/
├── src/                          # Core modules
│   ├── geometry_utils.py         # STL processing and geometry analysis
│   ├── mesh_functions.py         # OpenFOAM mesh generation
│   ├── quality_assessment.py     # Mesh quality evaluation
│   └── config_manager.py         # Configuration management
├── mesh_loop.py                  # Main optimization script
├── config/
│   └── default.json             # Default configuration
├── tutorial/
│   └── patient1/                # Example arterial geometry
│       ├── inlet.stl
│       ├── outlet1.stl
│       ├── outlet2.stl
│       ├── outlet3.stl
│       ├── outlet4.stl
│       └── wall_aorta.stl
├── output/                      # Generated meshes and results
└── docs/                       # Documentation
```

## 🔧 Configuration

The tool uses JSON configuration files for all parameters. Key sections include:

### Surface Refinement
```json
"SNAPPY_UNIFORM": {
  "surface_level": [1, 2],
  "surface_level_max": [3, 4],
  "maxGlobalCells": 20000000
}
```

### Boundary Layers
```json
"LAYERS": {
  "nSurfaceLayers": 7,
  "expansionRatio": 1.2,
  "finalLayerThickness_rel": 0.15,
  "featureAngle": 140
}
```

### Quality Criteria
```json
"QUALITY_CRITERIA": {
  "maxNonOrtho": 65,
  "maxSkewness": 4.0,
  "negVolCells": 0
}
```

## 📊 Mesh Quality Assessment

The tool evaluates meshes using professional CFD standards:

- **Orthogonality**: Max non-orthogonality ≤ 65°
- **Skewness**: Max skewness ≤ 4.0
- **Aspect Ratio**: Reasonable cell aspect ratios
- **Layer Coverage**: Boundary layer quality and coverage
- **Y+ Distribution**: Wall-normal spacing for turbulence modeling

## 🎯 Adaptive Refinement Strategy

1. **Progressive Refinement**: Gradually increases surface refinement levels
2. **Surface Failure Recovery**: Automatically detects and recovers from surface intersection failures
3. **Quality-Based Adjustments**: Adjusts parameters based on mesh quality metrics
4. **Layer Optimization**: Adapts boundary layer parameters for optimal growth

## 📖 Tutorial: Patient1 Case

The included `patient1` case demonstrates typical aortic arch geometry with:
- 1 inlet (ascending aorta)
- 4 outlets (arch branches)
- Complex arterial wall geometry

### Running the Tutorial

```bash
cd AortaCFD-Snappy
python mesh_loop.py --geometry patient1 --verbose
```

### Expected Output

```
🚀 Starting mesh optimization for patient1
✅ Found required files: inlet.stl, wall_aorta.stl
✅ Discovered 4 outlet files: ['outlet1.stl', 'outlet2.stl', 'outlet3.stl', 'outlet4.stl']
🔄 ITERATION 01/10
🔧 Running: blockMesh
🔧 Running: surfaceFeatures
🔧 Running: snappyHexMesh (snap phase)
🔧 Running: snappyHexMesh (layer phase)
✅ Mesh generation completed successfully
📊 ITERATION 1 RESULTS:
📈 Cell count: 485,784
📐 Max skewness: 1.23 ✅
📏 Max non-ortho: 45.2° ✅
🎯 Layer coverage: 98.5% ✅
🏁 Overall: ✅
🎉 Optimization converged at iteration 1
```

## 🔬 Advanced Usage

### Custom Geometries

To use your own geometry:

1. Create directory: `tutorial/your_case/`
2. Add STL files:
   - `inlet.stl` - Inlet surface
   - `outlet*.stl` - Outlet surfaces (any number)
   - `wall_aorta.stl` - Arterial wall
3. Run: `python mesh_loop.py --geometry your_case`

### Configuration Customization

Create custom configuration files:

```json
{
  "SNAPPY_UNIFORM": {
    "surface_level": [2, 3],
    "maxGlobalCells": 50000000
  },
  "LAYERS": {
    "nSurfaceLayers": 10,
    "expansionRatio": 1.15
  },
  "REFINEMENT_STRATEGY": {
    "max_iterations": 15,
    "progressive_refinement": true
  }
}
```

## 📝 Output Files

Each iteration generates:

```
output/patient1/
├── iter_001/
│   ├── constant/polyMesh/        # Generated mesh
│   ├── system/                   # OpenFOAM dictionaries
│   ├── logs/                     # Command logs
│   ├── metrics.json              # Quality metrics
│   ├── config.json               # Used configuration
│   └── patient1.foam             # ParaView file
├── iter_002/
└── ...
```

## 🛠️ Algorithm Details

### Two-Pass snappyHexMesh Approach

1. **Snap Phase**: Clean surface capture with strict quality controls
2. **Layer Phase**: Boundary layer generation with permissive quality settings

This approach prevents boundary layer truncation while maintaining surface quality.

### Surface Intersection Recovery

The tool automatically detects when surface intersection fails (missing patches) and reduces refinement levels to recover. This prevents infinite loops of failed iterations.

### Quality-Driven Adaptation

Parameters are adjusted based on:
- Mesh quality metrics (skewness, non-orthogonality)
- Layer coverage analysis
- Cell count optimization
- Convergence assessment

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{aortacfd_snappy,
  title={AortaCFD-Snappy: Automated Mesh Optimization for Arterial Geometries},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/AortaCFD-Snappy}
}
```

## 🆘 Support

- **Documentation**: See `docs/` directory
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

## 🔗 Related Projects

- [OpenFOAM](https://openfoam.org/) - Open source CFD toolbox
- [snappyHexMesh](https://openfoam.org/releases/2-3-0/meshing/) - Automatic mesh generation
- [ParaView](https://www.paraview.org/) - Visualization toolkit