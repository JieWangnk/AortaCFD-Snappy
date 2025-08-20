# AortaCFD-Snappy Usage Guide

## Quick Start

### Basic Mesh Generation

```bash
# Generate mesh for patient1 with default settings
python mesh_loop.py --geometry patient1

# Generate with verbose output
python mesh_loop.py --geometry patient1 --verbose

# Limit iterations
python mesh_loop.py --geometry patient1 --max-iterations 5
```

### Advanced Options

```bash
# Use custom configuration
python mesh_loop.py --geometry patient1 --config config/custom.json

# Enable CFD solver for y+ evaluation
python mesh_loop.py --geometry patient1 --enable-solver

# Custom output directory
python mesh_loop.py --geometry patient1 --output-dir /path/to/output
```

## Configuration Guide

### Creating Custom Configurations

1. Copy `config/default.json` to a new file
2. Modify parameters as needed
3. Use with `--config` flag

### Key Parameters

#### Surface Refinement
- `surface_level`: [min, max] refinement levels
- `surface_level_max`: Maximum allowed refinement
- `maxGlobalCells`: Global cell count limit

#### Boundary Layers
- `nSurfaceLayers`: Number of boundary layers
- `expansionRatio`: Growth ratio between layers
- `finalLayerThickness_rel`: Final layer thickness (relative)
- `featureAngle`: Feature angle for layer growth

#### Quality Criteria
- `maxNonOrtho`: Maximum non-orthogonality (degrees)
- `maxSkewness`: Maximum skewness
- `negVolCells`: Negative volume cells (should be 0)

### Example Custom Configuration

```json
{
  "SNAPPY_UNIFORM": {
    "surface_level": [2, 3],
    "maxGlobalCells": 50000000
  },
  "LAYERS": {
    "nSurfaceLayers": 10,
    "expansionRatio": 1.15,
    "finalLayerThickness_rel": 0.2
  },
  "QUALITY_CRITERIA": {
    "maxNonOrtho": 70,
    "maxSkewness": 3.5
  }
}
```

## Geometry Preparation

### Required Files

For each geometry case, provide:

1. `inlet.stl` - Inlet surface
2. `outlet*.stl` - Outlet surfaces (any number, any naming)
3. `wall_aorta.stl` - Arterial wall surface

### File Requirements

- **Units**: millimeters (will be converted to meters)
- **Quality**: Clean, watertight STL meshes
- **Orientation**: Inlet normal should point into domain
- **Naming**: Use descriptive outlet names (outlet1, outlet2, etc.)

### Geometry Validation

The tool automatically validates:
- Required files present
- STL file integrity
- Bounding box calculation
- Inlet orientation analysis

## Output Analysis

### Directory Structure

```
output/geometry_name/
├── iter_001/
│   ├── constant/polyMesh/     # OpenFOAM mesh
│   ├── system/               # Mesh generation settings
│   ├── logs/                 # Command logs
│   ├── metrics.json          # Quality metrics
│   ├── config.json           # Used configuration
│   └── geometry.foam         # ParaView file
├── iter_002/
└── optimization.log          # Full optimization log
```

### Key Output Files

#### metrics.json
Contains mesh quality assessment:
```json
{
  "checkMesh": {
    "cells": 485784,
    "maxSkewness": 1.23,
    "maxNonOrtho": 45.2,
    "negVolCells": 0,
    "meshOK": true
  },
  "acceptance": {
    "mesh_ok": true,
    "yPlus_ok": false
  },
  "all_ok": true
}
```

#### Logs Directory
- `log.blockMesh` - Background mesh generation
- `log.surfaceFeatures` - Feature extraction
- `log.snappyHexMesh.snap` - Surface snapping
- `log.snappyHexMesh.layers` - Boundary layer addition
- `log.checkMesh` - Quality assessment

## Troubleshooting

### Common Issues

#### 1. Surface Intersection Failure
**Symptoms**: Only 1 patch instead of expected 6
**Solution**: Tool automatically reduces refinement levels

#### 2. Poor Layer Coverage
**Symptoms**: Low percentage in layer coverage report
**Solutions**:
- Increase `finalLayerThickness_rel`
- Decrease `expansionRatio`
- Increase `featureAngle`

#### 3. High Skewness/Non-orthogonality
**Symptoms**: Quality metrics exceed thresholds
**Solutions**:
- Increase surface refinement
- Improve STL quality
- Adjust `resolveFeatureAngle`

#### 4. Long Runtime
**Symptoms**: Iterations take very long
**Solutions**:
- Reduce `maxGlobalCells`
- Lower surface refinement levels
- Check parallel settings

### Debug Mode

Enable verbose logging for detailed information:
```bash
python mesh_loop.py --geometry patient1 --verbose
```

## Performance Optimization

### Parallel Processing

The tool automatically uses parallel processing for large meshes:
- Threshold: 1M+ cells
- Processors: Auto-detected (max 8)
- Override in configuration if needed

### Memory Management

For large cases:
1. Reduce `maxGlobalCells`
2. Use progressive refinement
3. Monitor system memory usage

### Speed vs Quality Trade-offs

- **Fast**: Lower refinement levels, fewer layers
- **Quality**: Higher refinement, more boundary layers
- **Balanced**: Use default configuration

## Integration with OpenFOAM

### Viewing Results in ParaView

1. Open ParaView
2. Load `geometry.foam` file from iteration directory
3. Apply filters as needed

### Using Mesh in Simulations

1. Copy `constant/polyMesh/` to your case directory
2. Ensure boundary conditions match patch names
3. Run your OpenFOAM solver

### Boundary Patch Names

Standard patch names generated:
- `inlet` - Inlet surface
- `outlet1`, `outlet2`, etc. - Outlet surfaces
- `wall_aorta` - Arterial wall

## Best Practices

### Geometry Preparation
1. Use consistent units (mm)
2. Ensure watertight meshes
3. Reasonable mesh resolution in STL files
4. Proper surface normals

### Configuration
1. Start with default configuration
2. Make incremental changes
3. Document configuration choices
4. Test with simple geometries first

### Quality Assessment
1. Always check mesh quality metrics
2. Verify boundary layer coverage
3. Validate cell count expectations
4. Review log files for warnings

### Iteration Strategy
1. Let tool converge naturally
2. Monitor quality trends
3. Adjust parameters based on results
4. Keep successful configurations