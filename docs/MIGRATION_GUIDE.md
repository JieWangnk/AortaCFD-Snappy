# Stage1 Migration Guide

## Overview

This guide helps you migrate from the legacy monolithic Stage1 system to the new modular architecture. The migration is designed to be gradual and non-breaking.

## Migration Timeline

### Phase 1: Automatic Integration (Current)
**Status**: ✅ **Available Now**
- Legacy `Stage1MeshOptimizer` class automatically uses modular components
- Zero code changes required for existing users
- Full backward compatibility maintained
- Performance improvements immediate

### Phase 2: Direct Modular Usage (Recommended)
**Status**: ✅ **Available Now** 
- Use new `Stage1Optimizer` class directly
- Access individual modules for specific tasks
- Enhanced control and flexibility
- Better error handling and logging

### Phase 3: Legacy Deprecation (Future)
**Status**: 📅 **Planned for v3.0**
- Legacy class marked as deprecated
- Migration warnings issued
- Full modular API becomes standard

## Migration Scenarios

### Scenario 1: No Changes Needed (Legacy User)

If you're using the existing API and don't need new features:

```python
# This continues to work unchanged
from mesh_optim.stage1_mesh import Stage1MeshOptimizer

optimizer = Stage1MeshOptimizer(geometry_dir, config_file, output_dir)
best_mesh = optimizer.iterate_until_quality()
```

**Benefits**:
- ✅ Zero code changes
- ✅ Automatic performance improvements
- ✅ Enhanced error handling
- ✅ Better logging and diagnostics

### Scenario 2: Gradual Migration (Recommended)

Migrate to the new API for better control and features:

#### Before (Legacy):
```python
from mesh_optim.stage1_mesh import Stage1MeshOptimizer

optimizer = Stage1MeshOptimizer(geometry_dir, config_file, output_dir)
best_mesh = optimizer.iterate_until_quality()
```

#### After (Modular):
```python
from mesh_optim.stage1.optimizer import Stage1Optimizer

optimizer = Stage1Optimizer(config_file, geometry_stl, work_dir)
result = optimizer.run_full_optimization(max_iterations=10, parallel_procs=4)
final_mesh = result['final_mesh_directory']
```

**Benefits**:
- ✅ Better progress tracking
- ✅ Parallel processing control
- ✅ Comprehensive result summaries
- ✅ Multiple export formats
- ✅ Real-time status monitoring

### Scenario 3: Module-Level Usage (Advanced)

Use individual modules for specific tasks:

```python
from mesh_optim.stage1.config_manager import ConfigManager
from mesh_optim.stage1.physics_calculator import PhysicsCalculator
from mesh_optim.stage1.quality_analyzer import QualityAnalyzer

# Load and validate configuration
config_manager = ConfigManager("config.json")

# Calculate physics-based parameters
physics = PhysicsCalculator(config_manager)
layer_params = physics.calculate_layer_parameters(diameter=0.02, velocity=1.5, base_size=0.001)

# Assess mesh quality
analyzer = QualityAnalyzer(config_manager)
metrics = analyzer.assess_mesh_quality(mesh_dir, "layers", parallel_procs=4)
```

**Benefits**:
- ✅ Maximum flexibility
- ✅ Component reusability
- ✅ Custom workflow creation
- ✅ Integration with other tools

## Configuration Migration

### Legacy Configuration Format

```json
{
  "LAYERS": {
    "firstLayerThickness_abs": 50e-6,
    "nSurfaceLayers": 8,
    "expansionRatio": 1.2,
    "featureAngle": 60
  },
  "acceptance_criteria": {
    "maxNonOrtho": 65,
    "maxSkewness": 4.0,
    "min_layer_coverage": 0.65
  },
  "SNAPPY": {
    "maxLocalCells": 100000,
    "resolveFeatureAngle": 45
  }
}
```

### New Configuration Format

```json
{
  "physics": {
    "rho": 1060,
    "mu": 0.0035,
    "peak_velocity": 1.5,
    "solver_mode": "RANS",
    "use_womersley_bands": false
  },
  "snappyHexMeshDict": {
    "castellatedMeshControls": {
      "maxLocalCells": 100000,
      "maxGlobalCells": 500000,
      "resolveFeatureAngle": 45
    },
    "addLayersControls": {
      "firstLayerThickness_abs": 50e-6,
      "nSurfaceLayers": 8,
      "expansionRatio": 1.2,
      "featureAngle": 60,
      "nGrow": 1,
      "maxThicknessToMedialRatio": 0.45
    }
  },
  "targets": {
    "max_nonortho": 65,
    "max_skewness": 4.0,
    "min_layer_cov": 0.65
  }
}
```

### Automatic Migration

The system automatically migrates legacy configurations:

```python
# Both formats work automatically
config_manager = ConfigManager("legacy_config.json")  # Automatically migrated
config_manager = ConfigManager("new_config.json")     # Native format

# Access unified interface
targets = config_manager.targets
layers = config_manager.config["snappyHexMeshDict"]["addLayersControls"]
```

## API Changes Summary

### Class Names
| Legacy | New | Status |
|--------|-----|--------|
| `Stage1MeshOptimizer` | `Stage1Optimizer` | Both supported |

### Method Names
| Legacy | New | Notes |
|--------|-----|-------|
| `iterate_until_quality()` | `run_full_optimization()` | Enhanced return values |
| N/A | `run_single_mesh_generation()` | New: single mesh mode |
| N/A | `get_current_status()` | New: real-time status |
| N/A | `export_final_mesh()` | New: multiple formats |

### Constructor Parameters
| Legacy | New | Notes |
|--------|-----|-------|
| `geometry_dir` | `geometry_path` | Single STL file instead of directory |
| `config_file` | `config_path` | Same purpose |
| `output_dir` | `work_dir` | Same purpose |

### Return Values
| Legacy | New | Enhanced Information |
|--------|-----|---------------------|
| `Path` to best mesh | `Dict` with comprehensive results | Quality metrics, iteration history, timing |

## Feature Enhancements

### New Physics Capabilities

```python
# Womersley boundary layer calculation
physics = PhysicsCalculator(config_manager)
womersley_thickness = physics.calculate_womersley_boundary_layer(heart_rate_hz=1.2)

# y+ based layer sizing
first_layer = physics.calculate_yplus_first_layer(diameter=0.02, velocity=1.5, target_yplus=1.0)

# Flow regime classification
re_number = physics.calculate_reynolds_number(diameter=0.02, velocity=1.5)
regime = physics.classify_flow_regime(re_number)  # "laminar", "transitional", "turbulent"
```

### Enhanced Quality Analysis

```python
analyzer = QualityAnalyzer(config_manager)

# Convergence detection
converged, reason = analyzer.check_convergence()
if converged:
    print(f"Optimization converged: {reason}")

# Quality issue diagnosis
diagnosis = analyzer.diagnose_quality_issues(snap_metrics, layer_metrics)
print(f"Suggested improvements: {diagnosis['suggestions']}")

# Comprehensive quality assessment
quality_ok = analyzer.meets_quality_constraints(snap_metrics, layer_metrics, layer_coverage)
```

### Parallel Processing Support

```python
# Legacy: Single-threaded only
optimizer_legacy = Stage1MeshOptimizer(geometry_dir, config_file)
result = optimizer_legacy.iterate_until_quality()

# New: Parallel processing
optimizer_new = Stage1Optimizer(config_file, geometry_stl, work_dir)
result = optimizer_new.run_full_optimization(parallel_procs=8)  # Use 8 cores
```

### Real-time Progress Monitoring

```python
optimizer = Stage1Optimizer(config_file, geometry_stl, work_dir)

# Start optimization in background
import threading
optimization_thread = threading.Thread(
    target=optimizer.run_full_optimization,
    kwargs={'max_iterations': 15, 'parallel_procs': 4}
)
optimization_thread.start()

# Monitor progress
while optimization_thread.is_alive():
    status = optimizer.get_current_status()
    print(f"Iteration {status['current_iteration']}, Quality: {status['best_quality']:.3f}")
    time.sleep(10)
```

## Testing Your Migration

### Basic Functionality Test

```python
def test_migration_compatibility():
    # Test legacy interface still works
    legacy_optimizer = Stage1MeshOptimizer(geometry_dir, config_file)
    assert legacy_optimizer.config is not None
    
    # Test new interface works
    new_optimizer = Stage1Optimizer(config_file, geometry_stl, work_dir)
    assert new_optimizer.config_manager is not None
    
    print("✅ Migration compatibility verified")

test_migration_compatibility()
```

### Performance Comparison

```python
import time

# Legacy performance
start = time.time()
legacy_result = legacy_optimizer.iterate_until_quality()
legacy_time = time.time() - start

# New modular performance  
start = time.time()
new_result = new_optimizer.run_full_optimization()
new_time = time.time() - start

print(f"Legacy time: {legacy_time:.1f}s")
print(f"Modular time: {new_time:.1f}s")
print(f"Speedup: {legacy_time/new_time:.1f}x")
```

## Common Migration Issues

### Issue 1: Import Changes

**Problem**: `ImportError` when importing new modules
```python
# This might fail initially
from mesh_optim.stage1.optimizer import Stage1Optimizer
```

**Solution**: Check Python path and module installation
```bash
export PYTHONPATH="/path/to/AortaCFD-Snappy:$PYTHONPATH"
# Or install in development mode
pip install -e .
```

### Issue 2: Configuration Validation Errors

**Problem**: Configuration validation fails with new stricter checks

**Solution**: Use ConfigManager to see what's missing
```python
config_manager = ConfigManager("config.json")
print("Validation issues:", config_manager._validation_warnings)
```

### Issue 3: Path Handling Changes

**Problem**: Geometry directory vs. single STL file

**Legacy**: Expected directory with multiple STL files
**New**: Expects single STL file path

**Solution**: Update path specification
```python
# Legacy
geometry_dir = "/path/to/geometry/"  # Directory

# New  
geometry_stl = "/path/to/geometry/wall_aorta.stl"  # Single file
```

### Issue 4: Return Value Format Changes

**Problem**: Code expects simple Path return

**Solution**: Extract path from result dictionary
```python
# Legacy
best_mesh_path = optimizer.iterate_until_quality()

# New - extract path from result
result = optimizer.run_full_optimization()
best_mesh_path = result['final_mesh_directory']
```

## Getting Help

### Documentation
- **Architecture Overview**: `docs/MODULAR_ARCHITECTURE.md`
- **API Reference**: Individual module docstrings
- **Examples**: `examples/` directory

### Testing
- **Unit Tests**: `python run_tests.py unit`
- **Integration Tests**: `python run_tests.py integration` 
- **Your Configuration**: Test your specific config with new system

### Support
- **Issues**: Create GitHub issue with migration problem
- **Discussions**: GitHub discussions for general questions
- **Examples**: Request specific migration examples

## Migration Checklist

### Pre-Migration
- [ ] Backup current working configuration
- [ ] Test current system functionality
- [ ] Review new feature documentation
- [ ] Identify desired enhancements

### During Migration  
- [ ] Test legacy compatibility (Phase 1)
- [ ] Gradually adopt new API (Phase 2)
- [ ] Update configuration format
- [ ] Test new features
- [ ] Update documentation/comments

### Post-Migration
- [ ] Verify performance improvements
- [ ] Test error handling improvements
- [ ] Use new monitoring capabilities
- [ ] Share feedback/issues
- [ ] Plan for Phase 3 deprecation

## Benefits Recap

### Immediate Benefits (Phase 1)
- ✅ Better error messages and logging
- ✅ Improved memory management
- ✅ Enhanced configuration validation
- ✅ More robust OpenFOAM integration

### Enhanced Benefits (Phase 2)
- ✅ Parallel processing support
- ✅ Real-time progress monitoring
- ✅ Physics-aware parameter calculation
- ✅ Advanced quality assessment
- ✅ Multiple export formats
- ✅ Comprehensive result summaries

### Future Benefits (Phase 3)
- ✅ Clean, maintainable codebase
- ✅ Easy extension and customization
- ✅ Better integration with other tools
- ✅ Modern development practices

The migration path is designed to be smooth and beneficial at every stage. Start with Phase 1 for immediate improvements, then gradually adopt Phase 2 features as needed.