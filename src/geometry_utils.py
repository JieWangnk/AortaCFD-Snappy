"""
Geometry processing utilities for arterial mesh generation.

Handles STL file processing, bounding box calculations, and spatial analysis
for cardiovascular geometries.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import hashlib
import glob
from stl import mesh as np_stl_mesh


class GeometryProcessor:
    """Process and analyze STL geometries for mesh generation."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def discover_geometry_files(self, input_dir: Path) -> Dict[str, Path]:
        """
        Discover and validate required STL files.
        
        Args:
            input_dir: Directory containing STL files
            
        Returns:
            Dictionary mapping geometry types to file paths
            
        Raises:
            FileNotFoundError: If required files are missing
        """
        required_files = ['inlet.stl', 'wall_aorta.stl']
        found_files = {}
        
        # Check required files
        for filename in required_files:
            file_path = input_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {filename}")
            found_files[filename.replace('.stl', '')] = file_path
        
        # Discover outlets
        outlet_files = []
        for stl_file in input_dir.glob('outlet*.stl'):
            outlet_files.append(stl_file)
        
        if not outlet_files:
            raise FileNotFoundError("No outlet files found (outlet*.stl)")
        
        outlet_files.sort(key=lambda x: x.name)
        found_files['outlets'] = outlet_files
        
        self.logger.info(f"✅ Found required files: {', '.join([f.name for f in [found_files['inlet'], found_files['wall_aorta']]])}")
        self.logger.info(f"✅ Discovered {len(outlet_files)} outlet files: {[f.name for f in outlet_files]}")
        
        return found_files
    
    def calculate_bounding_box(self, stl_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate combined bounding box for multiple STL files.
        
        Args:
            stl_files: List of STL file paths
            
        Returns:
            Tuple of (min_bounds, max_bounds) as numpy arrays
        """
        all_vertices = []
        
        for stl_file in stl_files:
            try:
                stl_mesh = np_stl_mesh.Mesh.from_file(str(stl_file))
                vertices = stl_mesh.vectors.reshape(-1, 3)
                all_vertices.append(vertices)
            except Exception as e:
                self.logger.warning(f"Could not read {stl_file}: {e}")
                continue
        
        if not all_vertices:
            raise ValueError("No valid STL files found for bounding box calculation")
        
        combined_vertices = np.vstack(all_vertices)
        min_bounds = np.min(combined_vertices, axis=0)
        max_bounds = np.max(combined_vertices, axis=0)
        
        return min_bounds, max_bounds
    
    def calculate_base_cell_size(self, min_bounds: np.ndarray, max_bounds: np.ndarray, 
                                resolution: int = 40) -> float:
        """
        Calculate appropriate base cell size based on geometry extent.
        
        Args:
            min_bounds: Minimum coordinates
            max_bounds: Maximum coordinates
            resolution: Target number of cells along longest dimension
            
        Returns:
            Base cell size in same units as geometry
        """
        extent = max_bounds - min_bounds
        longest_dimension = np.max(extent)
        base_cell_size = longest_dimension / resolution
        
        return base_cell_size
    
    def analyze_inlet_orientation(self, inlet_file: Path) -> Dict[str, float]:
        """
        Analyze inlet geometry to determine flow direction.
        
        Args:
            inlet_file: Path to inlet STL file
            
        Returns:
            Dictionary with orientation analysis results
        """
        try:
            stl_mesh = np_stl_mesh.Mesh.from_file(str(inlet_file))
            vertices = stl_mesh.vectors.reshape(-1, 3)
            
            # Calculate centroid and normal approximation
            centroid = np.mean(vertices, axis=0)
            
            # Simple normal estimation (assumes planar inlet)
            if len(vertices) >= 3:
                v1 = vertices[1] - vertices[0]
                v2 = vertices[2] - vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0, 0, 1])  # Default
            
            return {
                'centroid': centroid,
                'normal': normal,
                'area_estimate': len(vertices) * 0.1  # Rough estimate
            }
            
        except Exception as e:
            self.logger.warning(f"Could not analyze inlet orientation: {e}")
            return {
                'centroid': np.array([0, 0, 0]),
                'normal': np.array([0, 0, 1]),
                'area_estimate': 1.0
            }
    
    def find_internal_point(self, min_bounds: np.ndarray, max_bounds: np.ndarray,
                           inlet_centroid: np.ndarray) -> np.ndarray:
        """
        Find a point inside the geometry domain.
        
        Args:
            min_bounds: Minimum coordinates
            max_bounds: Maximum coordinates
            inlet_centroid: Inlet center point
            
        Returns:
            Internal point coordinates
        """
        # Use geometric center with slight offset toward inlet
        center = (min_bounds + max_bounds) / 2
        offset_toward_inlet = (inlet_centroid - center) * 0.1
        internal_point = center + offset_toward_inlet
        
        return internal_point
    
    def hash_geometry_files(self, stl_dir: Path) -> str:
        """
        Generate hash of all STL files to detect changes.
        
        Args:
            stl_dir: Directory containing STL files
            
        Returns:
            MD5 hash string
        """
        h = hashlib.md5()
        for stl_file in sorted(stl_dir.glob('*.stl')):
            h.update(stl_file.read_bytes())
        return h.hexdigest()
    
    def copy_geometry_files(self, source_files: Dict[str, Path], target_dir: Path) -> None:
        """
        Copy STL files to target directory.
        
        Args:
            source_files: Dictionary of source file paths
            target_dir: Target directory for copying
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy inlet and wall
        for key in ['inlet', 'wall_aorta']:
            if key in source_files:
                target_file = target_dir / f"{key}.stl"
                import shutil
                shutil.copy2(source_files[key], target_file)
                self.logger.info(f"Copied {source_files[key].name} to triSurface")
        
        # Copy outlets
        if 'outlets' in source_files:
            for outlet_file in source_files['outlets']:
                target_file = target_dir / outlet_file.name
                import shutil
                shutil.copy2(outlet_file, target_file)
                self.logger.info(f"Copied {outlet_file.name} to triSurface")