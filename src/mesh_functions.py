"""
Core mesh generation functions for OpenFOAM snappyHexMesh workflow.

Handles blockMesh generation, snappyHexMesh execution, and boundary condition setup
with robust two-pass approach for boundary layer generation.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import re


class MeshGenerator:
    """Generate meshes using OpenFOAM tools with advanced refinement strategies."""
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.openfoam_env = config.get('openfoam_env_path', '/opt/openfoam12/etc/bashrc')
    
    def generate_block_mesh_dict(self, iter_dir: Path, min_bounds: np.ndarray, 
                                max_bounds: np.ndarray, resolution: List[int]) -> None:
        """
        Generate blockMeshDict for background mesh.
        
        Args:
            iter_dir: Iteration directory
            min_bounds: Minimum geometry bounds
            max_bounds: Maximum geometry bounds
            resolution: [nx, ny, nz] cell divisions
        """
        # Expand domain by 20% in each direction
        extent = max_bounds - min_bounds
        expansion = extent * 0.2
        domain_min = min_bounds - expansion
        domain_max = max_bounds + expansion
        
        # Generate vertices for hexahedral block with correct ordering
        # OpenFOAM hex block ordering: bottom face (z-min) then top face (z-max)
        # Each face: x-min/y-min, x-max/y-min, x-max/y-max, x-min/y-max
        vertices = [
            f"({domain_min[0]:.6f} {domain_min[1]:.6f} {domain_min[2]:.6f})",  # 0: x-,y-,z-
            f"({domain_max[0]:.6f} {domain_min[1]:.6f} {domain_min[2]:.6f})",  # 1: x+,y-,z-
            f"({domain_max[0]:.6f} {domain_max[1]:.6f} {domain_min[2]:.6f})",  # 2: x+,y+,z-
            f"({domain_min[0]:.6f} {domain_max[1]:.6f} {domain_min[2]:.6f})",  # 3: x-,y+,z-
            f"({domain_min[0]:.6f} {domain_min[1]:.6f} {domain_max[2]:.6f})",  # 4: x-,y-,z+
            f"({domain_max[0]:.6f} {domain_min[1]:.6f} {domain_max[2]:.6f})",  # 5: x+,y-,z+
            f"({domain_max[0]:.6f} {domain_max[1]:.6f} {domain_max[2]:.6f})",  # 6: x+,y+,z+
            f"({domain_min[0]:.6f} {domain_max[1]:.6f} {domain_max[2]:.6f})"   # 7: x-,y+,z+
        ]
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    {vertices[0]}  // 0
    {vertices[1]}  // 1
    {vertices[2]}  // 2
    {vertices[3]}  // 3
    {vertices[4]}  // 4
    {vertices[5]}  // 5
    {vertices[6]}  // 6
    {vertices[7]}  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({resolution[0]} {resolution[1]} {resolution[2]}) 
    simpleGrading (1 1 1)
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
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);'''
        
        block_mesh_file = iter_dir / "system" / "blockMeshDict"
        block_mesh_file.parent.mkdir(parents=True, exist_ok=True)
        block_mesh_file.write_text(content)
        
        base_cell_size = np.max(extent) / np.min(resolution)
        self.logger.info(f"Generated blockMeshDict: {resolution} divisions, base cell size: {base_cell_size:.2f}mm")
    
    def generate_snappy_hex_mesh_dict(self, iter_dir: Path, geometry_files: Dict[str, List[str]], 
                                     surface_level: List[int], internal_point: np.ndarray,
                                     layers_config: dict, mode: str = "full") -> None:
        """
        Generate snappyHexMeshDict for mesh generation.
        
        Args:
            iter_dir: Iteration directory
            geometry_files: Dictionary of geometry file names
            surface_level: [min_level, max_level] for surface refinement
            internal_point: Point inside geometry
            layers_config: Layer generation parameters
            mode: "full", "snap", or "layers"
        """
        if mode == "snap":
            self._generate_snap_only_dict(iter_dir, geometry_files, surface_level, internal_point)
        elif mode == "layers":
            self._generate_layers_only_dict(iter_dir, geometry_files, internal_point, layers_config)
        else:
            self._generate_full_snappy_dict(iter_dir, geometry_files, surface_level, 
                                          internal_point, layers_config)
    
    def _generate_snap_only_dict(self, iter_dir: Path, geometry_files: Dict[str, List[str]], 
                                surface_level: List[int], internal_point: np.ndarray) -> None:
        """Generate snappyHexMeshDict for snap-only pass."""
        outlet_names = geometry_files.get('outlets', [])
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    inlet.stl 
    {{ 
        type triSurfaceMesh; 
        name inlet;
        file "inlet.stl";
    }}
{self._format_outlet_geometry(outlet_names)}
    wall_aorta.stl 
    {{ 
        type triSurfaceMesh; 
        name wall_aorta;
        file "wall_aorta.stl";
    }}
}};

features
(
    {{ file "inlet.eMesh"; level 1; }}
{self._format_outlet_features(outlet_names)}
    {{ file "wall_aorta.eMesh"; level 1; }}
);

castellatedMeshControls
{{
    maxLocalCells 5000000;
    maxGlobalCells 20000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 1;

    refinementSurfaces
    {{
        inlet {{ level ({surface_level[0]} {surface_level[1]}); patchInfo {{ type patch; }} }}
{self._format_outlet_refinement(outlet_names, surface_level)}
        wall_aorta {{ level ({surface_level[0]} {surface_level[1]}); patchInfo {{ type wall; }} }}
    }}

    refinementRegions
    {{
        wall_aorta
        {{
            mode distance;
            levels ((1.412 2) (2.825 1));
        }}
    }}

    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle 150;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

// Minimal addLayersControls (required even when disabled)
addLayersControls
{{
    relativeSizes yes;
    layers {{ }}
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.05;
}}

meshQualityControls
{{
    #include "meshQualityDict_snap"
}}

mergeTolerance 1e-6;

writeFlags (scalarLevels layerSets layerFields);'''
        
        snap_file = iter_dir / "system" / "snappyHexMeshDict.snap"
        snap_file.write_text(content)
    
    def _generate_layers_only_dict(self, iter_dir: Path, geometry_files: Dict[str, List[str]], 
                                  internal_point: np.ndarray, layers_config: dict) -> None:
        """Generate snappyHexMeshDict for layers-only pass."""
        outlet_names = geometry_files.get('outlets', [])
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh false;
snap            false;
addLayers       true;

geometry
{{
    inlet.stl 
    {{ 
        type triSurfaceMesh; 
        name inlet;
        file "inlet.stl";
    }}
{self._format_outlet_geometry(outlet_names)}
    wall_aorta.stl 
    {{ 
        type triSurfaceMesh; 
        name wall_aorta;
        file "wall_aorta.stl";
    }}
}};

features
(
    {{ file "inlet.eMesh"; level 1; }}
{self._format_outlet_features(outlet_names)}
    {{ file "wall_aorta.eMesh"; level 1; }}
);

// Minimal required sections (even when disabled)
castellatedMeshControls
{{
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 1;
    maxLoadUnbalance 0.10;
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle 30;
    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
}}

snapControls
{{
    nSmoothPatch 1;
    tolerance 1.0;
    nSolveIter 1;
    nRelaxIter 1;
}}

addLayersControls
{{
    relativeSizes yes;
    
    layers
    {{
        wall_aorta {{
            nSurfaceLayers {layers_config.get('nSurfaceLayers', 7)};
        }}
    }}
    
    // Optimized parameters for boundary layer growth
    expansionRatio              {layers_config.get('expansionRatio', 1.2)};
    finalLayerThickness         {layers_config.get('finalLayerThickness', 0.3)};
    minThickness                {layers_config.get('minThickness', 0.05)};
    nGrow                       {layers_config.get('nGrow', 0)};
    
    featureAngle                {layers_config.get('featureAngle', 110)};
    slipFeatureAngle            30;
    nLayerIter                  {layers_config.get('nLayerIter', 100)};
    nRelaxIter                  5;
    nRelaxedIter                {layers_config.get('nRelaxedIter', 25)};
    nSmoothSurfaceNormals       2;
    nSmoothNormals              10;
    nSmoothThickness            10;
    maxFaceThicknessRatio       {layers_config.get('maxFaceThicknessRatio', 0.7)};
    maxThicknessToMedialRatio   {layers_config.get('maxThicknessToMedialRatio', 0.7)};
    minMedianAxisAngle          {layers_config.get('minMedianAxisAngle', 15)};
    nBufferCellsNoExtrude       0;
    nSnappyIter                 3;
}}

meshQualityControls
{{
    #include "meshQualityDict_layer"
}}

mergeTolerance 1e-6;

writeFlags (scalarLevels layerSets layerFields);'''
        
        layers_file = iter_dir / "system" / "snappyHexMeshDict.layers"
        layers_file.write_text(content)
    
    def _format_outlet_geometry(self, outlet_names: List[str]) -> str:
        """Format outlet geometry sections."""
        return '\n'.join([
            f'    {name}.stl\n    {{\n        type triSurfaceMesh;\n        name {name};\n        file "{name}.stl";\n    }}'
            for name in outlet_names
        ])
    
    def _format_outlet_features(self, outlet_names: List[str]) -> str:
        """Format outlet feature sections."""
        return '\n'.join([f'    {{ file "{name}.eMesh"; level 1; }}' for name in outlet_names])
    
    def _format_outlet_refinement(self, outlet_names: List[str], surface_level: List[int]) -> str:
        """Format outlet refinement sections."""
        return '\n'.join([
            f'        {name} {{ level ({surface_level[0]} {surface_level[1]}); patchInfo {{ type patch; }} }}'
            for name in outlet_names
        ])
    
    def generate_quality_controls(self, iter_dir: Path) -> None:
        """Generate mesh quality control files for two-pass approach."""
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        # Strict quality for snap phase
        snap_quality = '''// Strict quality controls for snap phase
maxNonOrtho          65;
maxBoundarySkewness  4.0;
maxInternalSkewness  4.0;
maxConcave           80;
minVol               1e-13;
minTetQuality        1e-9;
minArea              -1;
minTwist             0.02;
minDeterminant       0.001;
minFaceWeight        0.02;
minVolRatio          0.01;
minTriangleTwist     -1;
minVolCollapseRatio  0.1;
nSmoothScale         4;
errorReduction       0.75;
relaxed
{
    maxNonOrtho          75;
    maxBoundarySkewness  20;
    maxInternalSkewness  8;
    maxConcave           80;
    minTetQuality        1e-30;
    minFaceWeight        1e-06;
    minVolRatio          0.01;
    minDeterminant       1e-06;
}'''
        
        # Permissive quality for layer phase  
        layer_quality = '''// Permissive quality for addLayers phase
maxNonOrtho          85;
maxBoundarySkewness  6.0;
maxInternalSkewness  6.0;
maxConcave           85;
minVol               1e-18;
minTetQuality        1e-12;
minArea              -1;
minTwist             0.015;
minDeterminant       1e-5;
minFaceWeight        0.002;
minVolRatio          0.002;
minTriangleTwist     -1;
minVolCollapseRatio  0.05;
nSmoothScale         4;
errorReduction       0.75;
relaxed
{
    maxNonOrtho          90;
    maxBoundarySkewness  20;
    maxInternalSkewness  10;
    maxConcave           90;
    minTetQuality        1e-30;
    minFaceWeight        1e-06;
    minVolRatio          0.001;
    minDeterminant       1e-06;
}'''
        
        (system_dir / "meshQualityDict_snap").write_text(snap_quality)
        (system_dir / "meshQualityDict_layer").write_text(layer_quality)
    
    def generate_system_files(self, iter_dir: Path) -> None:
        """Generate basic OpenFOAM system files required for mesh generation."""
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate controlDict
        control_dict = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     foamRun;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;

writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;
runTimeModifiable true;'''
        
        # Generate fvSchemes  
        fv_schemes = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}'''
        
        # Generate fvSolution
        fv_solution = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p               1e-6;
        U               1e-6;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}'''
        
        # Write files
        (system_dir / "controlDict").write_text(control_dict)
        (system_dir / "fvSchemes").write_text(fv_schemes)
        (system_dir / "fvSolution").write_text(fv_solution)
    
    def run_mesh_command(self, command: List[str], work_dir: Path, log_file: str) -> bool:
        """
        Execute OpenFOAM mesh generation command.
        
        Args:
            command: Command to execute
            work_dir: Working directory
            log_file: Log file name
            
        Returns:
            True if successful, False otherwise
        """
        log_path = work_dir / "logs" / log_file
        log_path.parent.mkdir(exist_ok=True)
        
        # Setup OpenFOAM environment - use bash explicitly
        env_command = f"bash -c 'source {self.openfoam_env} && " + " ".join(command) + "'"
        
        try:
            with open(log_path, 'w') as f:
                result = subprocess.run(
                    env_command,
                    shell=True,
                    cwd=work_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            
            if result.returncode == 0:
                self.logger.info(f"Running: {' '.join(command)}")
                return True
            else:
                self.logger.error(f"Command failed: {' '.join(command)}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            return False
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return False
    
    def check_patch_count(self, iter_dir: Path) -> int:
        """
        Check number of patches in generated mesh.
        
        Args:
            iter_dir: Iteration directory
            
        Returns:
            Number of patches found
        """
        boundary_file = iter_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            return 0
        
        try:
            content = boundary_file.read_text()
            patch_count = content.count('type            patch;') + content.count('type            wall;')
            return patch_count
        except Exception:
            return 0