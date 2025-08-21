"""
Stage 1: Geometry-driven mesh optimization (inner loop)

This module handles the basic mesh generation process based purely on geometry,
without requiring CFD runs. It iterates on surface refinement and layer settings
until quality and coverage criteria are met.
"""

import json
import numpy as np
from pathlib import Path
import shutil
import logging
from .utils import run_command, check_mesh_quality, parse_layer_coverage
from .physics_mesh import PhysicsAwareMeshGenerator

class Stage1MeshOptimizer:
    """Geometry-driven mesh optimizer"""
    
    def __init__(self, geometry_dir, config_file, output_dir=None):
        """
        Initialize Stage 1 mesh optimizer
        
        Args:
            geometry_dir: Path to directory containing STL files
            config_file: Path to configuration JSON file
            output_dir: Output directory (default: geometry_dir/../output)
        """
        self.geometry_dir = Path(geometry_dir)
        self.config_file = Path(config_file)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.geometry_dir.parent / "output" / "stage1_mesh"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_file) as f:
            self.config = json.load(f)
        
        # Setup logging
        self.logger = logging.getLogger(f"Stage1Mesh_{self.geometry_dir.name}")
        
        # Initialize mesh parameters
        self.current_iteration = 0
        self.surface_levels = self.config["SNAPPY"]["surface_level"].copy()
        self.max_iterations = self.config.get("max_iterations", 4)
        
        # Find STL files
        self.stl_files = self._discover_stl_files()
        
    def _discover_stl_files(self):
        """Discover and validate STL files"""
        required_files = ["inlet.stl", "wall_aorta.stl"]
        stl_files = {"required": {}, "outlets": []}
        
        # Check required files
        for req_file in required_files:
            stl_path = self.geometry_dir / req_file
            if not stl_path.exists():
                raise FileNotFoundError(f"Required STL file not found: {req_file}")
            stl_files["required"][req_file.split('.')[0]] = stl_path
        
        # Find outlet files
        for stl_path in self.geometry_dir.glob("outlet*.stl"):
            stl_files["outlets"].append(stl_path)
        
        if not stl_files["outlets"]:
            raise FileNotFoundError("No outlet STL files found")
            
        self.logger.info(f"Found {len(stl_files['outlets'])} outlet files")
        return stl_files
    
    def calculate_bbox_and_base_cell_size(self):
        """Calculate bounding box and base cell size from actual STL geometry"""
        try:
            # Initialize physics generator to compute bounding box
            physics_gen = PhysicsAwareMeshGenerator()
            
            # Collect all STL files
            stl_files = {}
            for name, path in self.stl_files["required"].items():
                stl_files[name] = path
            for outlet_path in self.stl_files["outlets"]:
                stl_files[outlet_path.stem] = outlet_path
            
            # Compute bounding box from STL files
            bbox_data = physics_gen.compute_stl_bounding_box(stl_files)
            
            if bbox_data["total_vertices"] == 0:
                self.logger.warning("No vertices found in STL files, using default bounding box")
            else:
                self.logger.info(f"Computed bounding box from {bbox_data['total_vertices']} vertices")
            
            # Extract mesh domain (in meters)
            mesh_domain = bbox_data["mesh_domain"]
            
            # Convert to mm for OpenFOAM (typical convention)
            bbox_min = np.array([mesh_domain["x_min"], mesh_domain["y_min"], mesh_domain["z_min"]]) * 1000
            bbox_max = np.array([mesh_domain["x_max"], mesh_domain["y_max"], mesh_domain["z_max"]]) * 1000
            bbox_size = bbox_max - bbox_min
            
            # Calculate base cell size
            resolution = self.config["BLOCKMESH"]["resolution"]
            max_dim = np.max(bbox_size)
            base_cell_size = max_dim / resolution
            
            self.logger.info(f"Computed bounding box: {bbox_min} to {bbox_max} mm")
            self.logger.info(f"Base cell size: {base_cell_size:.2f} mm")
            
            return {
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_size": bbox_size,
                "base_cell_size": base_cell_size,
                "bbox_data": bbox_data  # Include full bounding box data
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compute STL bounding box: {e}")
            self.logger.info("Falling back to default aortic dimensions")
            
            # Fallback to typical aorta dimensions (mm)
            bbox_min = np.array([-20.6, -40.0, -57.1])
            bbox_max = np.array([8.6, 7.2, 37.1])
            bbox_size = bbox_max - bbox_min
            
            # Calculate base cell size
            resolution = self.config["BLOCKMESH"]["resolution"]
            max_dim = np.max(bbox_size)
            base_cell_size = max_dim / resolution
            
            return {
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_size": bbox_size,
                "base_cell_size": base_cell_size
            }
    
    def generate_blockmesh_dict(self, iter_dir):
        """Generate blockMeshDict"""
        bbox_info = self.calculate_bbox_and_base_cell_size()
        bbox_min, bbox_max = bbox_info["bbox_min"], bbox_info["bbox_max"]
        base_size = bbox_info["base_cell_size"]
        
        # Calculate divisions
        divisions = np.round(bbox_info["bbox_size"] / base_size).astype(int)
        divisions = np.maximum(divisions, [8, 8, 8])  # Minimum divisions
        
        # Generate vertices (8 corners of bounding box)
        # Using the correct ordering from the working version
        vertices = []
        for z in [bbox_min[2], bbox_max[2]]:
            vertices.append([bbox_min[0], bbox_min[1], z])
            vertices.append([bbox_max[0], bbox_min[1], z])
            vertices.append([bbox_max[0], bbox_max[1], z])
            vertices.append([bbox_min[0], bbox_max[1], z])
        
        blockmesh_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
{chr(10).join(f"    ({v[0]} {v[1]} {v[2]})  // {i}" for i, v in enumerate(vertices))}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({divisions[0]} {divisions[1]} {divisions[2]}) simpleGrading (1 1 1)
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
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (3 7 6 2)
            (0 4 7 3)
            (1 2 6 5)
        );
    }}
);
'''
        
        system_dir = iter_dir / "system"
        system_dir.mkdir(exist_ok=True)
        (system_dir / "blockMeshDict").write_text(blockmesh_content)
        
        # Generate minimal controlDict required by OpenFOAM
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
    location    "system";
    object      controlDict;
}

application     blockMesh;
startFrom       latestTime;
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
runTimeModifiable true;
'''
        (system_dir / "controlDict").write_text(control_dict)
        
        self.logger.info(f"Generated blockMeshDict: {divisions} divisions, base cell size: {base_size:.2f}mm")
        return bbox_info
    
    def generate_surfacesFeatures_dict(self, iter_dir):
        """Generate surfaceFeaturesDict"""
        
        trisurface_dir = iter_dir / "constant" / "triSurface"
        trisurface_dir.mkdir(parents=True, exist_ok=True)
        
        outlet_names = []
        for stl_file in self.stl_files["outlets"]:
            dest = trisurface_dir / stl_file.name
            shutil.copy2(stl_file, dest)
            outlet_names.append(stl_file.stem)
        
        for name, stl_file in self.stl_files["required"].items():
            dest = trisurface_dir / stl_file.name
            shutil.copy2(stl_file, dest)
        
        # Build list of all STL surfaces
        all_surfaces = ["wall_aorta.stl", "inlet.stl"] + [f"{name}.stl" for name in outlet_names]
        surfaces_list = " ".join(f'"{surf}"' for surf in all_surfaces)
        
        surfacesFeatures_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaces ({surfaces_list});

// Identify a feature when angle between faces < includedAngle
includedAngle   150;  // Keep sharp lips/ostia for CoA cases

// ************************************************************************* //'''
        
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        (system_dir / "surfaceFeaturesDict").write_text(surfacesFeatures_content)
    
    def generate_snappy_dicts(self, iter_dir, bbox_info):
        """Generate snappyHexMesh dictionaries for two-pass approach"""
        
        # Get outlet names from STL files (already copied by surfacesFeatures)
        outlet_names = [stl_file.stem for stl_file in self.stl_files["outlets"]]
        
        # Calculate internal point (inside the geometry)
        bbox_center = (bbox_info["bbox_min"] + bbox_info["bbox_max"]) / 2
        internal_point = bbox_center * 1e-3  # Convert mm to meters for locationInMesh
        
        # Distance refinement settings (FIXED: convert mm to meters!)
        base_size_mm = bbox_info["base_cell_size"]
        if "distance_refinement" in self.config["SNAPPY"]:
            dist_config = self.config["SNAPPY"]["distance_refinement"]
            dist1_m = dist_config["far_distance"] * 1e-3  # Convert mm to meters
            dist2_m = dist_config["near_distance"] * 1e-3  # Convert mm to meters
        else:
            # Default: physics-aware distances in meters
            dist1_m = 3.0e-3  # 3mm -> 0.003m far
            dist2_m = 1.5e-3  # 1.5mm -> 0.0015m near
        
        # Generate snap-only dictionary
        self._generate_snap_dict(iter_dir, outlet_names, internal_point, dist1_m, dist2_m)
        
        # Generate layers-only dictionary  
        self._generate_layers_dict(iter_dir, outlet_names, internal_point)
        
        self.logger.info(f"Generated snappyHexMesh dicts: surface level {self.surface_levels}")
    
    def _generate_snap_dict(self, iter_dir, outlet_names, internal_point, dist1_m, dist2_m):
        """Generate snap-only snappyHexMeshDict"""
        
        snap_config = self.config["SNAPPY"]
        
        snap_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
{chr(10).join(f'    {name}.stl{chr(10)}    {{{chr(10)}        type triSurfaceMesh;{chr(10)}        name {name};{chr(10)}        file "{name}.stl";{chr(10)}    }}' for name in outlet_names)}
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
{chr(10).join(f'    {{ file "{name}.eMesh"; level 1; }}' for name in outlet_names)}
    {{ file "wall_aorta.eMesh"; level 1; }}
);

castellatedMeshControls
{{
    maxLocalCells {snap_config["maxLocalCells"]};
    maxGlobalCells {snap_config["maxGlobalCells"]};
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {snap_config["nCellsBetweenLevels"]};

    refinementSurfaces
    {{
        inlet {{ level ({self.surface_levels[0]} {self.surface_levels[1]}); patchInfo {{ type patch; }} }}
{chr(10).join(f'        {name} {{ level ({self.surface_levels[0]} {self.surface_levels[1]}); patchInfo {{ type patch; }} }}' for name in outlet_names)}
        wall_aorta {{ level ({self.surface_levels[0]} {self.surface_levels[1]}); patchInfo {{ type wall; }} }}
    }}

    refinementRegions
    {{
        wall_aorta
        {{
            mode distance;
            levels (({dist2_m:.6f} 2) ({dist1_m:.6f} 1));
        }}
    }}

    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle {snap_config.get("resolveFeatureAngle", 30)};  // Reasonable feature angle
}}

snapControls
{{
    nSmoothPatch {snap_config.get("nSmoothPatch", 3)};
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter {snap_config.get("nFeatureSnapIter", 10)};
    implicitFeatureSnap {str(snap_config.get("implicitFeatureSnap", False)).lower()};
    explicitFeatureSnap {str(snap_config.get("explicitFeatureSnap", True)).lower()};
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
}}

meshQualityControls
{{
    // Explicit mesh quality parameters for OpenFOAM 12
    maxNonOrtho {self.config["MESH_QUALITY"]["snap"]["maxNonOrtho"]};
    maxBoundarySkewness {self.config["MESH_QUALITY"]["snap"]["maxBoundarySkewness"]};
    maxInternalSkewness {self.config["MESH_QUALITY"]["snap"]["maxInternalSkewness"]};
    maxConcave 80;
    minFlatness 0.5;
    minVol {self.config["MESH_QUALITY"]["snap"]["minVol"]};
    minTetQuality {self.config["MESH_QUALITY"]["snap"]["minTetQuality"]};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {self.config["MESH_QUALITY"]["snap"]["minFaceWeight"]};
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

// Merge tolerance. Is fraction of overall bounding box of initial mesh.
mergeTolerance 1e-6;

'''
        
        system_dir = iter_dir / "system"
        (system_dir / "snappyHexMeshDict.snap").write_text(snap_content)
    
    def _generate_layers_dict(self, iter_dir, outlet_names, internal_point):
        """Generate layers-only snappyHexMeshDict"""
        
        layers_config = self.config["LAYERS"]
        
        layers_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
{chr(10).join(f'    {name}.stl{chr(10)}    {{{chr(10)}        type triSurfaceMesh;{chr(10)}        name {name};{chr(10)}        file "{name}.stl";{chr(10)}    }}' for name in outlet_names)}
    wall_aorta.stl 
    {{ 
        type triSurfaceMesh; 
        name wall_aorta;
        file "wall_aorta.stl";
    }}
}};

castellatedMeshControls
{{
    // Minimal controls required by OpenFOAM even when castellatedMesh is false
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 1;
    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle 30;  // Reasonable feature angle for layer generation
}}

snapControls
{{
    // Minimal snap controls required by OpenFOAM
    nSmoothPatch 3;
    tolerance 4.0;
    nSolveIter 30;
    nRelaxIter 5;
}}

addLayersControls
{{
    relativeSizes false;  // Use absolute sizes for proper physics
    
    layers
    {{
        "wall_aorta"
        {{
            nSurfaceLayers {layers_config["nSurfaceLayers"]};
        }}
    }}

    // Physics-based absolute layer sizes
    firstLayerThickness {layers_config.get("firstLayerThickness_abs", 50e-6)};  // 50µm default
    expansionRatio {layers_config["expansionRatio"]};
    minThickness {layers_config.get("minThickness_abs", 20e-6)};  // 20µm minimum
    nGrow {layers_config["nGrow"]};
    featureAngle {min(layers_config["featureAngle"], 90)};  // Cap at 90 degrees
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {layers_config["maxThicknessToMedialRatio"]};
    minMedianAxisAngle {layers_config["minMedianAxisAngle"]};
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    // Explicit mesh quality parameters for OpenFOAM 12
    maxNonOrtho {self.config["MESH_QUALITY"]["layer"]["maxNonOrtho"]};
    maxBoundarySkewness {self.config["MESH_QUALITY"]["layer"]["maxBoundarySkewness"]};
    maxInternalSkewness {self.config["MESH_QUALITY"]["layer"]["maxInternalSkewness"]};
    maxConcave 80;
    minFlatness 0.5;
    minVol {self.config["MESH_QUALITY"]["layer"]["minVol"]};
    minTetQuality {self.config["MESH_QUALITY"]["layer"]["minTetQuality"]};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {self.config["MESH_QUALITY"]["layer"]["minFaceWeight"]};
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

// Merge tolerance. Is fraction of overall bounding box of initial mesh.
mergeTolerance 1e-6;

'''
        
        system_dir = iter_dir / "system"
        (system_dir / "snappyHexMeshDict.layers").write_text(layers_content)
    
    def run_snappy(self, iter_dir):
        """Run two-pass snappyHexMesh process"""
        
        openfoam_env = self.config["openfoam_env_path"]
        logs_dir = iter_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Copy snap dictionary to standard name for first phase
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.snap", 
                     iter_dir / "system" / "snappyHexMeshDict")
        
        commands = [
            (["blockMesh"], "log.blockMesh"),
            (["surfaceFeatures"], "log.surfaceFeatures"),
            (["snappyHexMesh", "-overwrite"], "log.snappyHexMesh.snap"),
        ]
        
        # Run snap phase
        for cmd, log_file in commands:
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            try:
                result = run_command(cmd, cwd=iter_dir, env_setup=openfoam_env, timeout=600)
                
                # Write log
                log_path = logs_dir / log_file
                log_path.write_text(result.stdout + result.stderr)
                
                if result.returncode != 0:
                    self.logger.error(f"Command failed: {' '.join(cmd)}")
                    self.logger.error(f"Error: {result.stderr[:500]}")
                    
            except Exception as e:
                self.logger.error(f"Command failed: {' '.join(cmd)}")
                self.logger.error(f"Error: {e}")
        
        # Copy layers dictionary to standard name for second phase
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.layers", 
                     iter_dir / "system" / "snappyHexMeshDict")
        
        commands = [
            (["snappyHexMesh", "-overwrite"], "log.snappyHexMesh.layers"),
            (["transformPoints", '"scale=(0.001 0.001 0.001)"'], "log.transformPoints"),
        ]
        
        for cmd, log_file in commands:
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            try:
                result = run_command(cmd, cwd=iter_dir, env_setup=openfoam_env, timeout=600)
                
                # Write log
                log_path = logs_dir / log_file
                log_path.write_text(result.stdout + result.stderr)
                
                if result.returncode != 0:
                    self.logger.error(f"Command failed: {' '.join(cmd)}")
                    self.logger.error(f"Error: {result.stderr[:500]}")
                    
            except Exception as e:
                self.logger.error(f"Command failed: {' '.join(cmd)}")
                self.logger.error(f"Error: {e}")
        
        # Parse layer coverage
        layer_coverage = parse_layer_coverage(iter_dir, openfoam_env)
        if layer_coverage["coverage_overall"] > 0:
            wall_info = layer_coverage["perPatch"].get("wall_aorta", {})
            n_faces = wall_info.get("nFaces", 0)
            avg_layers = wall_info.get("avgLayers", 0)
            self.logger.info(f"Layer coverage: {layer_coverage['coverage_overall']*100:.1f}% ({n_faces} faces, {avg_layers:.1f} avg layers)")
        else:
            self.logger.warning("Layer coverage: 0% - boundary layers failed to generate")
            
        return layer_coverage
    
    def iterate_until_quality(self):
        """Main iteration loop until quality criteria are met"""
        
        self.logger.info(f"Starting Stage 1 mesh optimization")
        self.logger.info(f"Target: quality mesh with boundary layers")
        
        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration
            self.logger.info(f"=== ITERATION {iteration} ===")
            
            # Create iteration directory
            iter_dir = self.output_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(exist_ok=True)
            
            # Generate mesh files
            bbox_info = self.generate_blockmesh_dict(iter_dir)
            self.generate_surfacesFeatures_dict(iter_dir)
            self.generate_snappy_dicts(iter_dir, bbox_info)
            
            # Run meshing
            layer_coverage = self.run_snappy(iter_dir)
            
            # Check mesh quality
            mesh_metrics = check_mesh_quality(iter_dir, self.config["openfoam_env_path"])
            
            # Evaluate iteration
            success = self._evaluate_iteration(mesh_metrics, layer_coverage)
            
            # Save metrics
            self._save_metrics(iter_dir, mesh_metrics, layer_coverage)
            
            if success:
                self.logger.info(f"✅ Stage 1 optimization completed successfully in {iteration} iterations")
                return iter_dir
            
            # Adjust parameters for next iteration
            self._adjust_parameters(mesh_metrics, layer_coverage)
        
        self.logger.warning(f"⚠️ Stage 1 optimization incomplete after {self.max_iterations} iterations")
        return iter_dir
    
    def _evaluate_iteration(self, mesh_metrics, layer_coverage):
        """Evaluate if current iteration meets quality criteria"""
        
        criteria = self.config["acceptance_criteria"]
        
        checks = {
            "mesh_valid": mesh_metrics["meshOK"],
            "non_ortho_ok": mesh_metrics["maxNonOrtho"] <= criteria["maxNonOrtho"],
            "skewness_ok": mesh_metrics["maxSkewness"] <= criteria["maxSkewness"],
            "layer_coverage_ok": layer_coverage["coverage_overall"] >= criteria["min_layer_coverage"]
        }
        
        all_ok = all(checks.values())
        
        # Log results
        self.logger.info(f"📊 Iteration {self.current_iteration} Results:")
        self.logger.info(f"  Cells: {mesh_metrics['cells']:,}")
        self.logger.info(f"  Max non-orthogonality: {mesh_metrics['maxNonOrtho']:.1f}° {'✅' if checks['non_ortho_ok'] else '❌'}")
        self.logger.info(f"  Max skewness: {mesh_metrics['maxSkewness']:.2f} {'✅' if checks['skewness_ok'] else '❌'}")
        self.logger.info(f"  Layer coverage: {layer_coverage['coverage_overall']*100:.1f}% {'✅' if checks['layer_coverage_ok'] else '❌'}")
        self.logger.info(f"  Mesh valid: {'✅' if checks['mesh_valid'] else '❌'}")
        
        return all_ok
    
    def _adjust_parameters(self, mesh_metrics, layer_coverage):
        """Adjust parameters for next iteration"""
        
        # If layers failed, reduce thickness and layer count
        if layer_coverage["coverage_overall"] < 0.5:
            self.config["LAYERS"]["finalLayerThickness_rel"] *= 0.8
            self.config["LAYERS"]["nSurfaceLayers"] = max(
                self.config["LAYERS"]["nSurfaceLayers"] - 2, 6
            )
            self.logger.info(f"Adjusted layers: thickness={self.config['LAYERS']['finalLayerThickness_rel']:.3f}, count={self.config['LAYERS']['nSurfaceLayers']}")
        
        # If non-orthogonality too high, reduce surface refinement
        if mesh_metrics["maxNonOrtho"] > self.config["acceptance_criteria"]["maxNonOrtho"]:
            if self.surface_levels[1] > 2:
                self.surface_levels[1] -= 1
                self.logger.info(f"Reduced surface refinement to {self.surface_levels}")
    
    def _save_metrics(self, iter_dir, mesh_metrics, layer_coverage):
        """Save iteration metrics to JSON"""
        
        metrics = {
            "iteration": self.current_iteration,
            "checkMesh": mesh_metrics,
            "layerCoverage": layer_coverage,
            "acceptance": {
                "mesh_valid": mesh_metrics["meshOK"],
                "non_ortho_ok": mesh_metrics["maxNonOrtho"] <= self.config["acceptance_criteria"]["maxNonOrtho"],
                "skewness_ok": mesh_metrics["maxSkewness"] <= self.config["acceptance_criteria"]["maxSkewness"],
                "layer_coverage_ok": layer_coverage["coverage_overall"] >= self.config["acceptance_criteria"]["min_layer_coverage"]
            }
        }
        
        metrics_file = iter_dir / "stage1_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)