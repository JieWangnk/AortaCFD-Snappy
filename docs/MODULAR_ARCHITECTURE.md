# Stage1 Modular Architecture

## Overview

The Stage1 mesh optimization system has been completely refactored from a monolithic 2700+ line file into a clean, modular architecture with proper separation of concerns. This document describes the new architecture, its benefits, and how to use it.

## Architecture Components

### Core Modules

#### 1. ConfigManager (`config_manager.py`)
**Purpose**: Configuration loading, validation, and management
- Loads and validates JSON configuration files
- Provides backward compatibility with old config formats
- Handles solver-specific presets (RANS/LES/Laminar)
- Validates layer constraints (minThickness < firstLayerThickness)
- Manages targets for optimization

**Key Methods**:
```python
manager = ConfigManager("config.json")
manager.update_layer_thickness(new_thickness)
manager.get_openfoam_env()
targets = manager.targets  # Stage1Targets object
```

#### 2. PhysicsCalculator (`physics_calculator.py`) 
**Purpose**: CFD physics-aware parameter calculations
- y+ based first layer thickness calculation
- Womersley boundary layer thickness for pulsatile flow
- Reynolds number calculation and flow regime classification
- Optimal layer distribution (count, expansion ratio)
- Refinement band calculation (geometry or physics-based)

**Key Methods**:
```python
calculator = PhysicsCalculator(config_manager)
first_layer = calculator.calculate_yplus_first_layer(diameter, velocity, target_yplus=1.0)
params = calculator.calculate_layer_parameters(diameter, velocity, base_size)
near, far = calculator.calculate_refinement_bands(base_size, use_womersley=True)
```

#### 3. GeometryProcessor (`geometry_processor.py`)
**Purpose**: STL geometry processing and analysis
- STL file loading and validation
- Automatic mm→m scaling detection and conversion
- Characteristic length extraction (inlet/outlet diameters)
- Bounding box and surface area calculations
- Feature extraction for all surfaces
- Manifold and topology checking

**Key Methods**:
```python
processor = GeometryProcessor(config_manager)
info = processor.process_stl_geometry(stl_path)
processed = processor.prepare_geometry_for_meshing(stl_path, work_dir)
```

#### 4. QualityAnalyzer (`quality_analyzer.py`)
**Purpose**: Mesh quality assessment and convergence detection
- Parallel and serial mesh quality checking
- checkMesh output parsing (OpenFOAM-agnostic)
- Layer coverage analysis with robust table parsing
- Convergence detection using coefficient of variation
- Quality constraint validation
- Quality issue diagnosis with improvement suggestions

**Key Methods**:
```python
analyzer = QualityAnalyzer(config_manager)
metrics = analyzer.assess_mesh_quality(mesh_dir, "snap", parallel_procs=4)
coverage = analyzer.assess_layer_quality(mesh_dir)
converged, reason = analyzer.check_convergence()
meets_targets = analyzer.meets_quality_constraints(snap, layer, coverage)
```

#### 5. MeshGenerator (`mesh_generator.py`)
**Purpose**: OpenFOAM case setup and execution
- OpenFOAM dictionary generation (blockMeshDict, snappyHexMeshDict)
- blockMesh execution with error handling
- snappyHexMesh execution (castellated, snap, layers phases)
- Parallel decomposition and reconstruction
- Mesh export to various formats (VTK, STL)

**Key Methods**:
```python
generator = MeshGenerator(config_manager)
generator.generate_case_files(work_dir, geometry_info)
result = generator.run_blockmesh(work_dir)
snap_result = generator.run_snappy_no_layers(work_dir, parallel_procs=4)
layer_result = generator.run_snappy_layers(work_dir, parallel_procs=4)
```

#### 6. IterationManager (`iteration_manager.py`)
**Purpose**: Optimization loop control and parameter adaptation
- Iteration lifecycle management
- Parameter adaptation based on quality trends
- Convergence monitoring and early termination
- Quality score calculation for ranking iterations
- Cleanup of old iterations to save disk space
- Comprehensive optimization reporting

**Key Methods**:
```python
manager = IterationManager(config_manager, quality_analyzer, physics_calculator)
should_continue, reason = manager.should_continue_optimization()
iter_dir = manager.start_new_iteration(work_dir)
manager.record_iteration_result(snap_metrics, layer_metrics, coverage, mesh_dir, success=True)
adapted = manager.adapt_parameters()
summary = manager.get_optimization_summary()
```

### Main Orchestrator

#### Stage1Optimizer (`optimizer.py`)
**Purpose**: Main workflow coordination and high-level interface
- Coordinates all core modules in proper sequence
- Provides both full optimization and single mesh generation modes
- Handles phase management: geometry → physics → optimization → finalization
- Comprehensive error handling and logging
- Multiple export formats support
- Status reporting and progress tracking

**Usage**:
```python
# Full optimization workflow
optimizer = Stage1Optimizer(config_path, geometry_path, work_dir)
result = optimizer.run_full_optimization(max_iterations=10, parallel_procs=4)

# Single mesh generation
result = optimizer.run_single_mesh_generation(parallel_procs=4)

# Export results
optimizer.export_final_mesh("output/", format_type="vtk")
```

## Benefits of Modular Architecture

### 1. **Maintainability**
- **Single Responsibility**: Each module has one clear purpose
- **Clear Interfaces**: Well-defined APIs between components
- **Isolated Testing**: Each module can be tested independently
- **Easy Debugging**: Problems isolated to specific modules

### 2. **Extensibility**
- **New Algorithms**: Easy to add new physics models or quality metrics
- **Different Solvers**: Support for different CFD solvers beyond OpenFOAM
- **Custom Workflows**: Mix and match components for specific use cases
- **Plugin Architecture**: Easy to add new geometry processors or quality analyzers

### 3. **Reusability**
- **Component Reuse**: Individual modules can be used in other projects
- **Configuration Reuse**: ConfigManager works with any JSON-based config
- **Physics Calculations**: PhysicsCalculator useful for any CFD preprocessing
- **Quality Assessment**: QualityAnalyzer works with any OpenFOAM mesh

### 4. **Performance**
- **Parallel Processing**: Built-in support for parallel mesh operations
- **Memory Management**: Proper cleanup and garbage collection
- **Resource Limits**: Automatic memory limit detection and enforcement
- **Caching**: Configuration and results caching where appropriate

### 5. **Robustness**
- **Error Handling**: Comprehensive error handling at each level
- **Fallback Mechanisms**: Graceful degradation when components fail
- **Input Validation**: Thorough validation at module boundaries
- **Resource Management**: Proper cleanup of temporary files and processes

## Backward Compatibility

The modular system provides complete backward compatibility with existing code:

### Legacy Integration
The original `Stage1MeshOptimizer` class automatically detects and uses the new modular system when available:

```python
# This code continues to work unchanged
from mesh_optim.stage1_mesh import Stage1MeshOptimizer

optimizer = Stage1MeshOptimizer(geometry_dir, config_file, output_dir)
best_mesh = optimizer.iterate_until_quality()
```

### Automatic Detection
- If modular components are available, legacy class uses them automatically
- If modular components fail, falls back to original implementation  
- Same API contracts maintained for all existing code
- Configuration compatibility preserved for Stage 2 pipeline

### Migration Path
1. **Phase 1**: Use legacy class with automatic modular integration (current)
2. **Phase 2**: Migrate to direct modular API usage (recommended)
3. **Phase 3**: Deprecate legacy class (future)

## Configuration Format

The modular system supports both new and legacy configuration formats:

### New Format (Recommended)
```json
{
  "physics": {
    "rho": 1060,
    "mu": 0.0035,
    "peak_velocity": 1.5,
    "solver_mode": "RANS"
  },
  "snappyHexMeshDict": {
    "addLayersControls": {
      "firstLayerThickness_abs": 50e-6,
      "nSurfaceLayers": 8,
      "expansionRatio": 1.2
    }
  },
  "targets": {
    "max_nonortho": 65,
    "max_skewness": 4.0,
    "min_layer_cov": 0.65
  }
}
```

### Legacy Format (Supported)
```json
{
  "LAYERS": {
    "firstLayerThickness_abs": 50e-6,
    "nSurfaceLayers": 8
  },
  "acceptance_criteria": {
    "maxNonOrtho": 65,
    "maxSkewness": 4.0
  }
}
```

## Testing Framework

Comprehensive test suite included:

### Unit Tests
- Individual module testing with mocked dependencies
- Edge case handling and error conditions
- Performance characteristics validation
- API contract verification

### Integration Tests  
- Full workflow testing with realistic data
- Backward compatibility validation
- Error handling and recovery testing
- Resource management verification

### Performance Tests
- Memory usage monitoring
- Execution time benchmarking
- Scalability testing with different problem sizes
- Resource limit enforcement validation

**Run Tests**:
```bash
# All tests
python run_tests.py all

# Specific test types  
python run_tests.py unit
python run_tests.py integration
python run_tests.py performance

# With coverage
python run_tests.py unit --coverage
```

## Development Guidelines

### Adding New Modules
1. **Interface Definition**: Define clear input/output contracts
2. **Dependency Injection**: Accept dependencies via constructor
3. **Error Handling**: Use appropriate exceptions with clear messages
4. **Logging**: Use module-specific loggers for debugging
5. **Testing**: Write comprehensive unit tests
6. **Documentation**: Document all public methods and classes

### Modifying Existing Modules
1. **Backward Compatibility**: Maintain existing API contracts
2. **Deprecation**: Use deprecation warnings before removing features
3. **Testing**: Update tests to cover new functionality
4. **Performance**: Ensure changes don't degrade performance
5. **Documentation**: Update documentation for changes

### Configuration Changes
1. **Validation**: Add validation for new configuration options
2. **Defaults**: Provide sensible defaults for new options
3. **Migration**: Support automatic migration from old formats
4. **Testing**: Test configuration loading and validation
5. **Documentation**: Document new configuration options

## Future Enhancements

### Planned Features
1. **GPU Acceleration**: CUDA/OpenCL support for quality assessment
2. **Machine Learning**: ML-based parameter optimization
3. **Cloud Integration**: AWS/GCP parallel mesh generation
4. **Advanced Physics**: Womersley number-based optimization
5. **Visualization**: Real-time mesh quality visualization

### Extension Points
1. **Custom Quality Metrics**: Plugin interface for new quality measures
2. **Alternative Solvers**: Support for Fluent, Star-CCM+, etc.
3. **Geometry Formats**: Support for STEP, IGES, etc.
4. **Optimization Algorithms**: Genetic algorithms, Bayesian optimization
5. **Post-processing**: Advanced visualization and analysis tools

---

For implementation details, see the individual module documentation and the comprehensive test suite in the `tests/` directory.