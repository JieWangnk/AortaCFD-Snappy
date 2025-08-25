"""
Geometry processing and STL handling for Stage1 mesh optimization.
Manages STL file operations, scaling detection, feature extraction, and geometric analysis.
"""
import logging
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .constants import DEFAULT_CONSTANTS
from ..utils import run_command

logger = logging.getLogger(__name__)

class GeometryProcessor:
    """Comprehensive geometry processing and STL management"""
    
    def __init__(self, geometry_dir: Path, config_manager):
        self.geometry_dir = Path(geometry_dir)
        self.config = config_manager.config
        self.constants = DEFAULT_CONSTANTS['geometry']
        self.openfoam_env = config_manager.get_openfoam_env()
        
        # Geometry analysis cache
        self.geometry_info = {}
        self.stl_files = []
        self.wall_name = "wall_aorta"
        
        # Initialize geometry analysis
        self._discover_stl_files()
        
    def _discover_stl_files(self) -> None:
        """Discover and categorize STL files in geometry directory"""
        if not self.geometry_dir.exists():
            raise FileNotFoundError(f"Geometry directory not found: {self.geometry_dir}")
        
        self.stl_files = list(self.geometry_dir.glob("*.stl"))
        
        if not self.stl_files:
            raise FileNotFoundError(f"No STL files found in {self.geometry_dir}")
        
        # Categorize STL files
        self.inlet_files = [f for f in self.stl_files if "inlet" in f.name.lower()]
        self.outlet_files = [f for f in self.stl_files if "outlet" in f.name.lower()]  
        self.wall_files = [f for f in self.stl_files if "wall" in f.name.lower() or "aorta" in f.name.lower()]
        
        # Identify primary wall file
        if self.wall_files:
            self.wall_name = self.wall_files[0].stem
            
        logger.info(f"Discovered geometry: {len(self.inlet_files)} inlet(s), "
                   f"{len(self.outlet_files)} outlet(s), {len(self.wall_files)} wall(s)")
    
    def process_geometry(self, output_dir: Path, force_scaling: bool = False) -> Dict:
        """
        Process all geometry files with scaling detection and validation
        
        Args:
            output_dir: Directory to copy/scale processed STL files
            force_scaling: Force mm->m scaling regardless of auto-detection
            
        Returns:
            Dictionary with geometry processing results
        """
        logger.info("Processing geometry files...")
        
        # Create triSurface directory
        trisurface_dir = output_dir / "constant" / "triSurface"
        trisurface_dir.mkdir(parents=True, exist_ok=True)
        
        processing_results = {
            "files_processed": [],
            "scaling_applied": False,
            "scale_factor": 1.0,
            "geometry_bounds": {},
            "validation_results": {}
        }
        
        # Detect if scaling is needed
        needs_scaling = force_scaling or self._detect_mm_coordinates()
        scale_factor = self.constants.AUTO_SCALE_FACTOR if needs_scaling else 1.0
        
        processing_results["scaling_applied"] = needs_scaling
        processing_results["scale_factor"] = scale_factor
        
        # Process each STL file
        for stl_file in self.stl_files:
            try:
                result = self._process_single_stl(stl_file, trisurface_dir, scale_factor)
                processing_results["files_processed"].append(result)
                
                # Update geometry bounds
                processing_results["geometry_bounds"][stl_file.stem] = result.get("bounds")
                
            except Exception as e:
                logger.error(f"Failed to process {stl_file.name}: {e}")
                processing_results["files_processed"].append({
                    "file": stl_file.name,
                    "success": False,
                    "error": str(e)
                })
        
        # Validate processed geometry
        validation_results = self._validate_processed_geometry(trisurface_dir)
        processing_results["validation_results"] = validation_results
        
        # Store geometry info for later use
        self.geometry_info = processing_results
        
        return processing_results
    
    def _detect_mm_coordinates(self) -> bool:
        """
        Detect if STL coordinates are in millimeters (need scaling to meters)
        
        Returns:
            True if scaling from mm to m is needed
        """
        try:
            # Analyze the largest STL file (usually wall)
            largest_stl = max(self.stl_files, key=lambda f: f.stat().st_size)
            
            # Quick bounds analysis using surfaceCheck
            result = run_command(
                ["surfaceCheck", str(largest_stl)],
                cwd=self.geometry_dir.parent,
                env_setup=self.openfoam_env,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            
            # Extract bounding box dimensions
            bounds_match = re.search(
                r"Bounding box.*?(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
                output, re.IGNORECASE
            )
            
            if bounds_match:
                dimensions = [float(bounds_match.group(i)) for i in range(1, 4)]
                max_dimension = max(dimensions)
                
                # If largest dimension < 1cm, assume mm coordinates
                needs_scaling = max_dimension < self.constants.MM_TO_M_THRESHOLD
                
                logger.info(f"Geometry analysis: max_dimension={max_dimension:.3f}, "
                           f"scaling_needed={needs_scaling}")
                
                return needs_scaling
                
        except Exception as e:
            logger.warning(f"Could not auto-detect scaling: {e}, assuming no scaling needed")
            
        return False
    
    def _process_single_stl(self, stl_file: Path, output_dir: Path, scale_factor: float) -> Dict:
        """
        Process a single STL file with scaling and validation
        
        Returns:
            Dictionary with processing results for this file
        """
        dest_file = output_dir / stl_file.name
        
        if scale_factor != 1.0:
            # Apply scaling using surfaceTransformPoints
            logger.info(f"Scaling {stl_file.name} by {scale_factor}")
            
            result = run_command([
                "surfaceTransformPoints",
                "-scale", f"({scale_factor} {scale_factor} {scale_factor})",
                str(stl_file.absolute()),
                str(dest_file.absolute())
            ], cwd=output_dir.parent, env_setup=self.openfoam_env, timeout=120)
            
            if result.returncode != 0:
                raise RuntimeError(f"STL scaling failed: {result.stderr}")
        else:
            # Simple copy without scaling
            import shutil
            shutil.copy2(stl_file, dest_file)
            
        # Analyze processed file
        analysis = self._analyze_stl_file(dest_file)
        
        return {
            "file": stl_file.name,
            "success": True,
            "scaled": scale_factor != 1.0,
            "scale_factor": scale_factor,
            "output_path": str(dest_file),
            "bounds": analysis.get("bounds"),
            "face_count": analysis.get("face_count"),
            "manifold_check": analysis.get("manifold_check")
        }
    
    def _analyze_stl_file(self, stl_file: Path) -> Dict:
        """
        Analyze an STL file for geometric properties and quality
        
        Returns:
            Dictionary with geometric analysis results
        """
        try:
            # Use surfaceCheck for comprehensive analysis
            result = run_command(
                ["surfaceCheck", str(stl_file)],
                cwd=stl_file.parent,
                env_setup=self.openfoam_env,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            analysis = {"file_valid": result.returncode == 0}
            
            # Parse geometric properties
            self._parse_surface_properties(output, analysis)
            self._parse_manifold_check(output, analysis)
            self._parse_bounds(output, analysis)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"STL analysis failed for {stl_file.name}: {e}")
            return {"file_valid": False, "error": str(e)}
    
    def _parse_surface_properties(self, output: str, analysis: Dict) -> None:
        """Parse surface properties from surfaceCheck output"""
        # Face count
        face_match = re.search(r"faces\s*:\s*(\d+)", output, re.IGNORECASE)
        if face_match:
            analysis["face_count"] = int(face_match.group(1))
        
        # Point count  
        point_match = re.search(r"points\s*:\s*(\d+)", output, re.IGNORECASE)
        if point_match:
            analysis["point_count"] = int(point_match.group(1))
            
        # Area
        area_match = re.search(r"area\s*:\s*([\d.eE+-]+)", output, re.IGNORECASE)
        if area_match:
            analysis["surface_area"] = float(area_match.group(1))
    
    def _parse_manifold_check(self, output: str, analysis: Dict) -> None:
        """Parse manifold geometry validation"""
        # Non-manifold edges
        nm_match = re.search(r"non-manifold.*?(\d+)", output, re.IGNORECASE)
        if nm_match:
            non_manifold_count = int(nm_match.group(1))
            analysis["non_manifold_edges"] = non_manifold_count
            analysis["manifold_check"] = non_manifold_count == 0
        else:
            analysis["manifold_check"] = True
            
        # Check for critical errors
        analysis["has_critical_errors"] = any(
            error in output.lower() for error in 
            ["degenerate", "invalid", "corrupted", "critical"]
        )
    
    def _parse_bounds(self, output: str, analysis: Dict) -> None:
        """Parse bounding box from surfaceCheck output"""
        bounds_match = re.search(
            r"bounding box.*?([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+).*?"
            r"([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)",
            output, re.IGNORECASE | re.DOTALL
        )
        
        if bounds_match:
            try:
                bounds = {
                    "min": [float(bounds_match.group(i)) for i in range(1, 4)],
                    "max": [float(bounds_match.group(i)) for i in range(4, 7)]
                }
                
                # Calculate dimensions
                dimensions = [bounds["max"][i] - bounds["min"][i] for i in range(3)]
                bounds["dimensions"] = dimensions
                bounds["max_dimension"] = max(dimensions)
                bounds["volume_estimate"] = dimensions[0] * dimensions[1] * dimensions[2]
                
                analysis["bounds"] = bounds
                
            except ValueError as e:
                logger.warning(f"Could not parse bounds: {e}")
    
    def _validate_processed_geometry(self, trisurface_dir: Path) -> Dict:
        """Validate all processed STL files for mesh generation readiness"""
        validation_results = {
            "all_files_valid": True,
            "critical_errors": [],
            "warnings": [],
            "file_checks": {}
        }
        
        for stl_file in trisurface_dir.glob("*.stl"):
            file_check = {
                "exists": stl_file.exists(),
                "size_ok": stl_file.stat().st_size > 1000,  # Minimum 1KB
                "manifold_ok": True,
                "bounds_reasonable": True
            }
            
            # Analyze file if it exists
            if file_check["exists"] and file_check["size_ok"]:
                analysis = self._analyze_stl_file(stl_file)
                
                # Manifold check
                if not analysis.get("manifold_check", True):
                    file_check["manifold_ok"] = False
                    validation_results["warnings"].append(
                        f"{stl_file.name}: Non-manifold geometry detected"
                    )
                
                # Bounds sanity check
                bounds = analysis.get("bounds", {})
                if bounds and bounds.get("max_dimension", 0) > 1.0:  # > 1 meter
                    file_check["bounds_reasonable"] = False
                    validation_results["warnings"].append(
                        f"{stl_file.name}: Unusually large dimensions ({bounds['max_dimension']:.2f}m)"
                    )
                
                # Critical errors
                if analysis.get("has_critical_errors"):
                    validation_results["critical_errors"].append(
                        f"{stl_file.name}: Critical geometry errors detected"
                    )
                    validation_results["all_files_valid"] = False
            else:
                validation_results["critical_errors"].append(
                    f"{stl_file.name}: File missing or too small"
                )
                validation_results["all_files_valid"] = False
            
            validation_results["file_checks"][stl_file.name] = file_check
        
        # Log validation summary
        if validation_results["all_files_valid"]:
            logger.info("✅ All geometry files passed validation")
        else:
            logger.error(f"❌ Geometry validation failed: {len(validation_results['critical_errors'])} critical errors")
            for error in validation_results["critical_errors"]:
                logger.error(f"  - {error}")
        
        if validation_results["warnings"]:
            logger.warning(f"⚠️ {len(validation_results['warnings'])} geometry warnings:")
            for warning in validation_results["warnings"]:
                logger.warning(f"  - {warning}")
        
        return validation_results
    
    def extract_surface_features(self, trisurface_dir: Path, feature_angle: float = 45) -> Dict:
        """
        Extract surface features for all STL files
        
        Args:
            trisurface_dir: Directory containing processed STL files
            feature_angle: Feature detection angle in degrees
            
        Returns:
            Dictionary with feature extraction results
        """
        logger.info(f"Extracting surface features (angle={feature_angle}°)")
        
        extraction_results = {
            "feature_angle": feature_angle,
            "files_processed": [],
            "total_features": 0
        }
        
        for stl_file in trisurface_dir.glob("*.stl"):
            try:
                emesh_file = trisurface_dir / f"{stl_file.stem}.eMesh"
                
                # Extract features using surfaceFeatureExtract
                result = run_command([
                    "surfaceFeatureExtract",
                    "-includedAngle", str(feature_angle),
                    stl_file.name,
                    emesh_file.name
                ], cwd=trisurface_dir, env_setup=self.openfoam_env, timeout=120)
                
                if result.returncode == 0:
                    feature_count = self._count_feature_edges(emesh_file)
                    extraction_results["files_processed"].append({
                        "stl_file": stl_file.name,
                        "emesh_file": emesh_file.name,
                        "success": True,
                        "feature_count": feature_count
                    })
                    extraction_results["total_features"] += feature_count
                else:
                    logger.warning(f"Feature extraction failed for {stl_file.name}: {result.stderr}")
                    extraction_results["files_processed"].append({
                        "stl_file": stl_file.name,
                        "success": False,
                        "error": result.stderr
                    })
                    
            except Exception as e:
                logger.error(f"Feature extraction error for {stl_file.name}: {e}")
                extraction_results["files_processed"].append({
                    "stl_file": stl_file.name,
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(f"Feature extraction complete: {extraction_results['total_features']} total features")
        return extraction_results
    
    def _count_feature_edges(self, emesh_file: Path) -> int:
        """Count feature edges in an eMesh file"""
        try:
            if emesh_file.exists():
                content = emesh_file.read_text()
                # Simple heuristic: count lines that look like edge definitions
                lines = content.split('\n')
                edge_count = sum(1 for line in lines if line.strip() and line.strip()[0].isdigit())
                return max(0, edge_count - 10)  # Subtract header lines
        except Exception:
            pass
        return 0
    
    def estimate_reference_diameters(self) -> Tuple[float, float]:
        """
        Estimate reference diameters from inlet/outlet geometry
        
        Returns:
            Tuple of (reference_diameter, minimum_diameter)
        """
        if not self.geometry_info:
            logger.warning("No geometry info available, using default diameters")
            return 0.025, 0.015  # 25mm, 15mm defaults
        
        diameters = []
        
        # Collect diameter estimates from inlets/outlets
        for file_info in self.geometry_info.get("files_processed", []):
            if any(name in file_info.get("file", "").lower() for name in ["inlet", "outlet"]):
                bounds = file_info.get("bounds", {})
                if bounds and "dimensions" in bounds:
                    # Estimate diameter as sqrt(2*smaller_dimensions) for circular cross-section
                    dims = sorted(bounds["dimensions"])
                    if len(dims) >= 2:
                        diameter_est = np.sqrt(dims[0] * dims[1]) * 2  # Approximate circular diameter
                        diameters.append(diameter_est)
        
        if diameters:
            ref_diameter = max(diameters)  # Largest opening
            min_diameter = min(diameters)  # Smallest opening
            
            logger.info(f"Estimated diameters: reference={ref_diameter*1000:.1f}mm, "
                       f"minimum={min_diameter*1000:.1f}mm")
            
            return ref_diameter, min_diameter
        else:
            logger.warning("Could not estimate diameters from geometry, using defaults")
            return 0.025, 0.015
    
    def get_outlet_names(self) -> List[str]:
        """Get list of outlet patch names for mesh generation"""
        return [f.stem for f in self.outlet_files]
    
    def get_geometry_summary(self) -> Dict:
        """Get comprehensive geometry summary for reporting"""
        summary = {
            "source_directory": str(self.geometry_dir),
            "total_stl_files": len(self.stl_files),
            "file_categories": {
                "inlets": len(self.inlet_files),
                "outlets": len(self.outlet_files), 
                "walls": len(self.wall_files)
            },
            "primary_wall": self.wall_name,
            "processing_applied": bool(self.geometry_info),
            "scaling_applied": self.geometry_info.get("scaling_applied", False),
            "scale_factor": self.geometry_info.get("scale_factor", 1.0)
        }
        
        if self.geometry_info:
            validation = self.geometry_info.get("validation_results", {})
            summary["validation"] = {
                "all_files_valid": validation.get("all_files_valid", False),
                "critical_errors": len(validation.get("critical_errors", [])),
                "warnings": len(validation.get("warnings", []))
            }
        
        return summary