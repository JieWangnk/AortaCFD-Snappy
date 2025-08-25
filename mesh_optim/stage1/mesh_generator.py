"""
OpenFOAM mesh generation and dictionary management for Stage1 optimization.
Handles blockMesh, snappyHexMeshDict generation, and mesh execution workflows.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from .constants import DEFAULT_CONSTANTS, MAGIC_NUMBERS
from ..utils import run_command

logger = logging.getLogger(__name__)

class MeshGenerator:
    """Comprehensive OpenFOAM mesh generation and management"""
    
    def __init__(self, config_manager, geometry_processor, physics_calculator):
        self.config = config_manager.config
        self.geometry = geometry_processor
        self.physics = physics_calculator
        self.constants = DEFAULT_CONSTANTS
        self.openfoam_env = config_manager.get_openfoam_env()
        
        # Mesh generation state
        self.current_surface_levels = [1, 1]  # Conservative start
        self.current_feature_angle = 45
        self.iteration_count = 0
        
    def generate_background_mesh(self, iter_dir: Path, base_cell_size: float) -> Dict:
        """
        Generate blockMesh background mesh
        
        Args:
            iter_dir: Iteration working directory
            base_cell_size: Base cell size in meters
            
        Returns:
            Dictionary with background mesh generation results
        """
        logger.info(f"Generating background mesh (cell_size={base_cell_size*1000:.2f}mm)")
        
        # Calculate domain bounds from processed geometry
        domain_bounds = self._calculate_domain_bounds()
        
        # Generate blockMeshDict
        self._generate_blockmesh_dict(iter_dir, domain_bounds, base_cell_size)
        
        # Execute blockMesh
        result = self._execute_blockmesh(iter_dir)
        
        mesh_info = {
            "success": result["success"],
            "base_cell_size": base_cell_size,
            "domain_bounds": domain_bounds,
            "execution_time": result.get("execution_time", 0),
            "cell_count_estimate": result.get("cell_count", 0)
        }
        
        if result["success"]:
            logger.info(f"✅ Background mesh generated: ~{mesh_info['cell_count_estimate']:,} cells")
        else:
            logger.error(f"❌ Background mesh generation failed: {result.get('error')}")
            
        return mesh_info
    
    def _calculate_domain_bounds(self) -> Dict:
        """Calculate computational domain bounds from geometry"""
        # Get bounds from processed geometry
        geometry_bounds = []
        
        if self.geometry.geometry_info:
            for file_info in self.geometry.geometry_info.get("files_processed", []):
                bounds = file_info.get("bounds")
                if bounds and "min" in bounds and "max" in bounds:
                    geometry_bounds.extend([bounds["min"], bounds["max"]])
        
        if not geometry_bounds:
            # Fallback default bounds for typical vascular geometry
            logger.warning("Using default domain bounds - no geometry bounds available")
            return {
                "min": [-0.02, -0.02, -0.02],
                "max": [0.12, 0.05, 0.05],
                "padding_factor": 1.5
            }
        
        # Calculate overall bounds with padding
        geometry_array = np.array(geometry_bounds)
        min_bounds = np.min(geometry_array, axis=0)
        max_bounds = np.max(geometry_array, axis=0)
        
        # Add padding (50% on each side)
        padding_factor = 1.5
        center = (min_bounds + max_bounds) / 2
        half_size = (max_bounds - min_bounds) / 2 * padding_factor
        
        domain_min = center - half_size
        domain_max = center + half_size
        
        return {
            "min": domain_min.tolist(),
            "max": domain_max.tolist(),
            "padding_factor": padding_factor,
            "geometry_bounds": {
                "min": min_bounds.tolist(),
                "max": max_bounds.tolist()
            }
        }
    
    def _generate_blockmesh_dict(self, iter_dir: Path, bounds: Dict, cell_size: float) -> None:
        """Generate blockMeshDict with specified bounds and cell size"""
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate cell divisions
        domain_size = np.array(bounds["max"]) - np.array(bounds["min"])
        divisions = np.maximum(np.round(domain_size / cell_size).astype(int), [1, 1, 1])
        
        # Get grading from config
        grading = self.config.get("BLOCKMESH", {}).get("grading", [1, 1, 1])
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale 1;

vertices
(
    ({bounds["min"][0]:.6f} {bounds["min"][1]:.6f} {bounds["min"][2]:.6f})  // 0
    ({bounds["max"][0]:.6f} {bounds["min"][1]:.6f} {bounds["min"][2]:.6f})  // 1
    ({bounds["max"][0]:.6f} {bounds["max"][1]:.6f} {bounds["min"][2]:.6f})  // 2
    ({bounds["min"][0]:.6f} {bounds["max"][1]:.6f} {bounds["min"][2]:.6f})  // 3
    ({bounds["min"][0]:.6f} {bounds["min"][1]:.6f} {bounds["max"][2]:.6f})  // 4
    ({bounds["max"][0]:.6f} {bounds["min"][1]:.6f} {bounds["max"][2]:.6f})  // 5
    ({bounds["max"][0]:.6f} {bounds["max"][1]:.6f} {bounds["max"][2]:.6f})  // 6
    ({bounds["min"][0]:.6f} {bounds["max"][1]:.6f} {bounds["max"][2]:.6f})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({divisions[0]} {divisions[1]} {divisions[2]}) 
    simpleGrading ({grading[0]} {grading[1]} {grading[2]})
);

edges
(
);

boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 4 7 3)  // x-min
            (2 6 5 1)  // x-max  
            (0 1 5 4)  // y-min
            (3 7 6 2)  // y-max
            (0 3 2 1)  // z-min
            (4 5 6 7)  // z-max
        );
    }}
);

mergePatchPairs
(
);
'''
        
        (system_dir / "blockMeshDict").write_text(content)
        
        logger.debug(f"Generated blockMeshDict: {divisions[0]}×{divisions[1]}×{divisions[2]} cells")
    
    def _execute_blockmesh(self, iter_dir: Path) -> Dict:
        """Execute blockMesh command"""
        import time
        
        start_time = time.time()
        
        try:
            result = run_command(
                "blockMesh",
                cwd=iter_dir,
                env_setup=self.openfoam_env,
                timeout=120
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Extract cell count from output
                cell_count = self._parse_blockmesh_output(result.stdout)
                
                return {
                    "success": True,
                    "execution_time": execution_time,
                    "cell_count": cell_count,
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "execution_time": execution_time,
                    "error": result.stderr,
                    "output": result.stdout
                }
                
        except Exception as e:
            return {
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _parse_blockmesh_output(self, output: str) -> int:
        """Parse blockMesh output to extract cell count"""
        import re
        
        # Look for "cells:" line
        match = re.search(r'cells:\s+(\d+)', output)
        if match:
            return int(match.group(1))
        
        # Fallback: calculate from blocks
        block_match = re.search(r'Creating block mesh.*?(\d+)\s+cells', output, re.DOTALL)
        if block_match:
            return int(block_match.group(1))
        
        return 0
    
    def generate_snappy_dict(self, iter_dir: Path, phase: str, 
                           surface_levels: List[int], feature_angle: int,
                           layer_params: Optional[Dict] = None) -> Path:
        """
        Generate snappyHexMeshDict for specified phase
        
        Args:
            iter_dir: Iteration working directory
            phase: "castellated", "snap", "layers", or "no_layers"
            surface_levels: [min_level, max_level] for surface refinement
            feature_angle: Feature detection angle
            layer_params: Layer parameters (if applicable)
            
        Returns:
            Path to generated dictionary file
        """
        system_dir = iter_dir / "system"
        
        # Determine phases to enable
        phase_settings = self._get_phase_settings(phase)
        
        # Get outlet names from geometry processor
        outlet_names = self.geometry.get_outlet_names()
        
        # Calculate internal point (center of domain)
        internal_point = self._calculate_internal_point()
        
        # Generate dictionary content
        content = self._build_snappy_dict_content(
            phase_settings, surface_levels, feature_angle, 
            outlet_names, internal_point, layer_params
        )
        
        # Write to appropriate file
        dict_file = system_dir / "snappyHexMeshDict"
        if phase != "layers":  # Keep phase-specific copies for debugging
            dict_file = system_dir / f"snappyHexMeshDict.{phase}"
            
        dict_file.write_text(content)
        
        # Copy as main dict if not doing layers-only
        if phase != "layers":
            shutil.copy2(dict_file, system_dir / "snappyHexMeshDict")
            
        logger.debug(f"Generated snappyHexMeshDict.{phase} with levels {surface_levels}, angle {feature_angle}°")
        
        return dict_file
    
    def _get_phase_settings(self, phase: str) -> Dict:
        """Get phase enable/disable settings"""
        settings = {
            "castellated": {"castellatedMesh": True, "snap": False, "addLayers": False},
            "snap": {"castellatedMesh": False, "snap": True, "addLayers": False},  
            "layers": {"castellatedMesh": False, "snap": False, "addLayers": True},
            "no_layers": {"castellatedMesh": True, "snap": True, "addLayers": False},
            "full": {"castellatedMesh": True, "snap": True, "addLayers": True}
        }
        
        return settings.get(phase, settings["no_layers"])
    
    def _calculate_internal_point(self) -> List[float]:
        """Calculate a point inside the geometry for locationInMesh"""
        # Use center of computational domain as safe internal point
        if hasattr(self, '_domain_bounds'):
            bounds = self._domain_bounds
        else:
            bounds = self._calculate_domain_bounds()
            
        center = [
            (bounds["min"][i] + bounds["max"][i]) / 2 
            for i in range(3)
        ]
        
        return center
    
    def _build_snappy_dict_content(self, phase_settings: Dict, surface_levels: List[int],
                                 feature_angle: int, outlet_names: List[str],
                                 internal_point: List[float], layer_params: Optional[Dict]) -> str:
        """Build complete snappyHexMeshDict content"""
        
        # Header
        content = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}

'''
        
        # Phase settings
        content += f'''castellatedMesh {str(phase_settings["castellatedMesh"]).lower()};
snap            {str(phase_settings["snap"]).lower()};
addLayers       {str(phase_settings["addLayers"]).lower()};

'''
        
        # Geometry section
        content += self._build_geometry_section(outlet_names)
        
        # Castellated mesh controls
        if phase_settings["castellatedMesh"]:
            content += self._build_castellated_controls(surface_levels, internal_point)
        else:
            content += "castellatedMeshControls\n{\n}\n\n"
            
        # Snap controls
        if phase_settings["snap"]:
            content += self._build_snap_controls(feature_angle)
        else:
            content += "snapControls\n{\n}\n\n"
            
        # Layer controls
        if phase_settings["addLayers"] and layer_params:
            content += self._build_layer_controls(layer_params)
        else:
            content += self._build_empty_layer_controls()
            
        # Mesh quality controls
        content += self._build_quality_controls(phase_settings["addLayers"])
        
        # Merge tolerance
        content += "mergeTolerance 1e-6;\n"
        
        return content
    
    def _build_geometry_section(self, outlet_names: List[str]) -> str:
        """Build geometry section with all STL files"""
        content = "geometry\n{\n"
        
        # Add inlet
        content += '    inlet.stl       { type triSurfaceMesh; name inlet;       file "inlet.stl"; }\n'
        
        # Add outlets
        for outlet_name in outlet_names:
            content += f'    {outlet_name}.stl        {{ type triSurfaceMesh; name {outlet_name};           file "{outlet_name}.stl"; }}\n'
        
        # Add wall
        wall_name = self.geometry.wall_name
        content += f'    {wall_name}.stl  {{ type triSurfaceMesh; name {wall_name};  file "{wall_name}.stl"; }}\n'
        
        content += "};\n\n"
        return content
    
    def _build_castellated_controls(self, surface_levels: List[int], internal_point: List[float]) -> str:
        """Build castellatedMeshControls section"""
        snappy_config = self.config.get("SNAPPY", {})
        
        content = f'''castellatedMeshControls
{{
    maxLocalCells {snappy_config.get("maxLocalCells", 1000000)};
    maxGlobalCells {snappy_config.get("maxGlobalCells", 5000000)};
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {snappy_config.get("nCellsBetweenLevels", 2)};
    
    features
    (
        {{
            file "inlet.eMesh";
            level {surface_levels[1]};
        }}
        {{
            file "{self.geometry.wall_name}.eMesh";
            level {surface_levels[1]};
        }}
'''
        
        # Add outlet features
        for outlet_name in self.geometry.get_outlet_names():
            content += f'        {{ file "{outlet_name}.eMesh"; level {surface_levels[1]}; }}\n'
        
        content += '''    );
    
    refinementSurfaces
    {
        inlet
        {
            level (%d %d);
        }
        %s
        {
            level (%d %d);
        }
''' % (surface_levels[0], surface_levels[1], self.geometry.wall_name, surface_levels[0], surface_levels[1])
        
        # Add outlet refinement surfaces
        for outlet_name in self.geometry.get_outlet_names():
            content += f'''        {outlet_name}
        {{
            level ({surface_levels[0]} {surface_levels[1]});
        }}
'''
        
        # Refinement regions (distance-based)
        ref_diameter, _ = self.geometry.estimate_reference_diameters()
        near_dist = 4 * ref_diameter / MAGIC_NUMBERS['base_cells_per_diameter']  # 4 cells
        far_dist = 10 * ref_diameter / MAGIC_NUMBERS['base_cells_per_diameter']   # 10 cells
        
        content += f'''    }}
    
    refinementRegions
    {{
        {self.geometry.wall_name}
        {{
            mode distance;
            levels (({near_dist:.6f} {surface_levels[1]}) ({far_dist:.6f} {surface_levels[0]}));
        }}
    }}

    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle {snappy_config.get("resolveFeatureAngle", 30)};
}}

'''
        
        return content
    
    def _build_snap_controls(self, feature_angle: int) -> str:
        """Build snapControls section"""
        snappy_config = self.config.get("SNAPPY", {})
        
        return f'''snapControls
{{
    nSmoothPatch {snappy_config.get("nSmoothPatch", MAGIC_NUMBERS['smooth_patch_iterations'])};
    tolerance 1.0;
    nSolveIter {MAGIC_NUMBERS['solve_iterations']};
    nRelaxIter {MAGIC_NUMBERS['relax_iterations']};
    nFeatureSnapIter {snappy_config.get("nFeatureSnapIter", MAGIC_NUMBERS['feature_snap_iterations'])};
    implicitFeatureSnap {str(snappy_config.get("implicitFeatureSnap", False)).lower()};
    explicitFeatureSnap {str(snappy_config.get("explicitFeatureSnap", True)).lower()};
    multiRegionFeatureSnap false;
}}

'''
    
    def _build_layer_controls(self, layer_params: Dict) -> str:
        """Build addLayersControls section with provided parameters"""
        layers_config = self.constants['layers']
        
        # Apply validation to layer parameters
        first_layer = layer_params.get('firstLayerThickness_abs', 50e-6)
        min_thickness = layer_params.get('minThickness_abs', 20e-6)
        
        # Validate minThickness
        if min_thickness > first_layer * layers_config.MAX_THICKNESS_RATIO:
            min_thickness = first_layer * layers_config.MIN_THICKNESS_RATIO
            layer_params['minThickness_abs'] = min_thickness
            logger.warning(f"Corrected minThickness to {min_thickness*1e6:.2f}μm")
        
        return f'''addLayersControls
{{
    relativeSizes false;
    layers
    {{
        "{self.geometry.wall_name}"
        {{
            nSurfaceLayers {layer_params.get('nSurfaceLayers', 10)};
        }}
    }}

    firstLayerThickness {first_layer:.2e};
    expansionRatio {layer_params.get('expansionRatio', 1.2)};
    minThickness {min_thickness:.2e};
    nGrow {layer_params.get('nGrow', layers_config.N_GROW_DEFAULT)};
    featureAngle {layer_params.get('featureAngle', layers_config.FEATURE_ANGLE_DEFAULT)};
    nRelaxIter {layers_config.N_RELAX_ITER};
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {layers_config.MAX_THICKNESS_TO_MEDIAL_RATIO};
    minMedianAxisAngle {layers_config.MIN_MEDIAN_AXIS_ANGLE};
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

'''
    
    def _build_empty_layer_controls(self) -> str:
        """Build minimal addLayersControls section when not using layers"""
        return '''addLayersControls
{
    relativeSizes false;
    layers
    {
    }
    
    firstLayerThickness 1e-6;
    expansionRatio 1.0;
    minThickness 1e-6;
    nGrow 0;
    featureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}

'''
    
    def _build_quality_controls(self, with_layers: bool) -> str:
        """Build meshQualityControls section"""
        quality_key = "layer" if with_layers else "snap"
        quality_config = self.config.get("MESH_QUALITY", {}).get(quality_key, {})
        quality_constants = self.constants['mesh_quality']
        
        return f'''meshQualityControls
{{
    maxNonOrtho {quality_config.get("maxNonOrtho", quality_constants.MAX_NON_ORTHO_DEFAULT)};
    maxBoundarySkewness {quality_config.get("maxBoundarySkewness", quality_constants.MAX_SKEWNESS_DEFAULT)};
    maxInternalSkewness {quality_config.get("maxInternalSkewness", quality_constants.MAX_SKEWNESS_DEFAULT)};
    maxConcave 80;
    minFlatness 0.5;
    minVol {quality_config.get("minVol", quality_constants.MIN_VOL_DEFAULT)};
    minTetQuality {quality_config.get("minTetQuality", quality_constants.MIN_TET_QUALITY_SNAP if not with_layers else quality_constants.MIN_TET_QUALITY_LAYER)};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {quality_config.get("minFaceWeight", quality_constants.MIN_FACE_WEIGHT)};
    minVolRatio {quality_constants.MIN_VOL_RATIO};
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    
    // Relaxed quality criteria for {quality_key} phase
    relaxed
    {{
        maxNonOrtho 75;
        maxBoundarySkewness 6.0;
        maxInternalSkewness 6.0;
        maxConcave 90;
        minFlatness 0.3;
        minVol 1e-15;
        minTetQuality 1e-9;
        minFaceWeight 0.005;
        minVolRatio 0.005;
        minDeterminant 0.0005;
    }}
}}

'''
    
    def execute_snappy_mesh(self, iter_dir: Path, phase: str, parallel_procs: int = 1) -> Dict:
        """
        Execute snappyHexMesh for specified phase
        
        Args:
            iter_dir: Iteration working directory
            phase: Phase identifier for logging
            parallel_procs: Number of parallel processors
            
        Returns:
            Dictionary with execution results
        """
        import time
        
        logger.info(f"Executing snappyHexMesh ({phase} phase, {parallel_procs} procs)")
        
        # Setup parallel decomposition if needed
        if parallel_procs > 1:
            decomp_result = self._setup_parallel_decomposition(iter_dir, parallel_procs)
            if not decomp_result["success"]:
                return {"success": False, "error": f"Decomposition failed: {decomp_result['error']}"}
        
        # Build snappyHexMesh command
        if parallel_procs > 1:
            cmd = ["mpirun", "-np", str(parallel_procs), "snappyHexMesh", "-overwrite", "-parallel"]
        else:
            cmd = ["snappyHexMesh", "-overwrite"]
        
        # Execute command
        start_time = time.time()
        logs_dir = iter_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        try:
            result = run_command(
                cmd,
                cwd=iter_dir,
                env_setup=self.openfoam_env,
                timeout=self.constants['convergence'].SNAPPY_TIMEOUT_MAX
            )
            
            execution_time = time.time() - start_time
            
            # Write log
            (logs_dir / f"log.snappy.{phase}").write_text(result.stdout + result.stderr)
            
            if result.returncode == 0:
                logger.info(f"✅ snappyHexMesh {phase} completed ({execution_time:.1f}s)")
                return {
                    "success": True,
                    "execution_time": execution_time,
                    "parallel": parallel_procs > 1,
                    "output": result.stdout
                }
            else:
                logger.error(f"❌ snappyHexMesh {phase} failed ({execution_time:.1f}s)")
                return {
                    "success": False,
                    "execution_time": execution_time,
                    "error": result.stderr,
                    "output": result.stdout
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ snappyHexMesh {phase} exception: {e}")
            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def _setup_parallel_decomposition(self, iter_dir: Path, n_procs: int) -> Dict:
        """Setup parallel decomposition for snappyHexMesh"""
        try:
            # Generate decomposeParDict
            self._generate_decompose_dict(iter_dir, n_procs)
            
            # Execute decomposePar
            result = run_command(
                ["decomposePar"],
                cwd=iter_dir,
                env_setup=self.openfoam_env,
                timeout=300
            )
            
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_decompose_dict(self, iter_dir: Path, n_procs: int) -> None:
        """Generate decomposeParDict for parallel execution"""
        system_dir = iter_dir / "system"
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {n_procs};

method          scotch;

scotchCoeffs
{{
    writeGraph false;
}}

distributeFields true;
'''
        
        (system_dir / "decomposeParDict").write_text(content)
    
    def update_surface_levels(self, new_levels: List[int]) -> None:
        """Update current surface refinement levels"""
        self.current_surface_levels = new_levels.copy()
        logger.debug(f"Surface levels updated to {new_levels}")
    
    def update_feature_angle(self, new_angle: int) -> None:
        """Update current feature detection angle"""
        self.current_feature_angle = new_angle
        logger.debug(f"Feature angle updated to {new_angle}°")
    
    def get_current_settings(self) -> Dict:
        """Get current mesh generation settings"""
        return {
            "surface_levels": self.current_surface_levels.copy(),
            "feature_angle": self.current_feature_angle,
            "iteration_count": self.iteration_count,
            "geometry_files": len(self.geometry.stl_files),
            "wall_name": self.geometry.wall_name
        }