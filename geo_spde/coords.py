import numpy as np
import pyproj
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from geo_spde.exceptions import CoordsError
from typing import Tuple, Dict, Union, List


def is_geographic(coords: np.ndarray) -> bool:
    """
    Detect if coordinates are in lon/lat format using value range heuristics.
    
    Parameters:
    -----------
    coords : np.ndarray
        Coordinate array of shape (n_obs, 2)
        
    Returns:
    --------
    bool : True if coordinates appear to be geographic (lon/lat)
    """
    x_vals, y_vals = coords[:, 0], coords[:, 1]
    
    # Check if all values fall within geographic ranges
    lon_range = np.all((-180 <= x_vals) & (x_vals <= 180))
    lat_range = np.all((-90 <= y_vals) & (y_vals <= 90))
    
    # Additional heuristic: geographic coords typically have smaller variance
    # and more decimal precision
    if lon_range and lat_range:
        x_var = np.var(x_vals)
        y_var = np.var(y_vals)
        # If variance is very large (>10000), likely projected despite being in range
        return x_var < 10000 and y_var < 10000
    
    return False

def detect_antimeridian_crossing(coords: np.ndarray) -> bool:
    """
    Detect if longitude data crosses the ±180° antimeridian.
    
    Parameters:
    -----------
    coords : np.ndarray
        Geographic coordinates (lon/lat)
        
    Returns:
    --------
    bool : True if antimeridian crossing detected
    """
    lon_range = coords[:, 0].max() - coords[:, 0].min()
    return lon_range > 180

def unwrap_antimeridian(coords: np.ndarray) -> np.ndarray:
    """
    Unwrap coordinates that cross the antimeridian by shifting to 0-360° system.
    
    Parameters:
    -----------
    coords : np.ndarray
        Geographic coordinates with antimeridian crossing
        
    Returns:
    --------
    np.ndarray : Unwrapped coordinates
    """
    coords_unwrapped = coords.copy()
    # Convert negative longitudes to 0-360° system
    coords_unwrapped[:, 0] = np.where(coords[:, 0] < 0, 
                                      coords[:, 0] + 360, 
                                      coords[:, 0])
    return coords_unwrapped

def compute_convex_hull_diameter(coords: np.ndarray) -> float:
    """
    Compute maximum distance between convex hull vertices.
    
    Parameters:
    -----------
    coords : np.ndarray
        Coordinate array of shape (n_obs, 2)
        
    Returns:
    --------
    float : Maximum distance between hull vertices
    """
    if len(coords) < 3:
        # For < 3 points, return maximum pairwise distance
        if len(coords) == 2:
            return np.linalg.norm(coords[1] - coords[0])
        return 0.0
    
    try:
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
        
        # Compute all pairwise distances between hull vertices
        distances = pdist(hull_points)
        return np.max(distances)
    except Exception:
        # Fallback for degenerate cases (collinear points)
        distances = pdist(coords)
        return np.max(distances) if len(distances) > 0 else 0.0

def degrees_to_km_estimate(coords: np.ndarray, hull_diameter_deg: float) -> float:
    """
    Convert geographic diameter to approximate km using latitude correction.
    
    Parameters:
    -----------
    coords : np.ndarray
        Geographic coordinates (lon/lat)
    hull_diameter_deg : float
        Diameter in degrees
        
    Returns:
    --------
    float : Approximate diameter in kilometers
    """
    center_lat = np.mean(coords[:, 1])
    
    # More accurate conversion accounting for latitude
    km_per_deg_lat = 110.54  # Slightly less due to Earth's elliptical shape
    km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))
    
    # Use average for diameter estimation
    avg_km_per_deg = (km_per_deg_lat + km_per_deg_lon) / 2
    return hull_diameter_deg * avg_km_per_deg

def determine_projection_scale(coords: np.ndarray, hull_diameter_deg: float) -> str:
    """
    Determine appropriate projection scale based on spatial extent.
    
    Parameters:
    -----------
    coords : np.ndarray
        Geographic coordinates
    hull_diameter_deg : float
        Convex hull diameter in degrees
        
    Returns:
    --------
    str : 'single_region', 'multi_region', or 'global'
    """
    approx_km = degrees_to_km_estimate(coords, hull_diameter_deg)
    if approx_km < 1000:
        return 'single_region'
    if approx_km < 11000:
        return 'multi_region'
    return 'global'

def determine_utm_zone(coords: np.ndarray) -> Tuple[int, str]:
    """
    Determine optimal UTM zone from coordinate centroid.
    
    Parameters:
    -----------
    coords : np.ndarray
        Geographic coordinates (lon/lat)
        
    Returns:
    --------
    Tuple[int, str] : (zone_number, hemisphere)
    """
    centroid_lon = np.mean(coords[:, 0])
    centroid_lat = np.mean(coords[:, 1])
    
    # UTM zone calculation
    zone_number = int(np.floor((centroid_lon + 180) / 6) + 1)
    
    # Ensure zone is in valid range [1, 60]
    zone_number = max(1, min(60, zone_number))
    
    hemisphere = 'north' if centroid_lat >= 0 else 'south'
    
    return zone_number, hemisphere

def compute_albers_standard_parallels(lat_min: float, lat_max: float) -> Tuple[float, float]:
    """
    Compute optimal standard parallels for Albers Equal-Area projection.
    
    Parameters:
    -----------
    lat_min : float
        Minimum latitude
    lat_max : float
        Maximum latitude
        
    Returns:
    --------
    Tuple[float, float] : (parallel_1, parallel_2)
    """
    lat_range = lat_max - lat_min
    
    # Standard approach: 1/6 from each end
    parallel_1 = lat_min + lat_range / 6
    parallel_2 = lat_max - lat_range / 6
    
    return parallel_1, parallel_2

def create_projection_string(scale: str, coords: np.ndarray) -> str:
    """
    Create appropriate Proj4 projection string based on scale and coordinates.
    
    Parameters:
    -----------
    scale : str
        Projection scale ('single_region', 'multi_region', 'global')
    coords : np.ndarray
        Geographic coordinates (lon/lat)
        
    Returns:
    --------
    str : Proj4 projection string
    """
    if scale == 'single_region':
        # UTM projection
        zone_number, hemisphere = determine_utm_zone(coords)
        if hemisphere == 'north':
            proj4_string = f"+proj=utm +zone={zone_number} +datum=WGS84 +units=m +no_defs"
        else:
            proj4_string = f"+proj=utm +zone={zone_number} +south +datum=WGS84 +units=m +no_defs"
        return proj4_string
    
    elif scale == 'multi_region':
        # Albers Equal-Area Conic
        lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
        lon_center = np.mean(coords[:, 0])
        lat_center = np.mean(coords[:, 1])
        
        parallel_1, parallel_2 = compute_albers_standard_parallels(lat_min, lat_max)
        
        proj4_string = (f"+proj=aea +lat_1={parallel_1:.2f} +lat_2={parallel_2:.2f} "
                       f"+lat_0={lat_center:.2f} +lon_0={lon_center:.2f} "
                       f"+datum=WGS84 +units=m +no_defs")
        return proj4_string
    
    else:  # global
        # Mollweide Equal-Area
        lon_center = np.mean(coords[:, 0])
        proj4_string = f"+proj=moll +lon_0={lon_center:.2f} +datum=WGS84 +units=m +no_defs"
        return proj4_string

def project_coordinates(coords: np.ndarray, proj4_string: str) -> np.ndarray:
    """
    Project geographic coordinates using specified projection.
    
    Parameters:
    -----------
    coords : np.ndarray
        Geographic coordinates (lon/lat)
    proj4_string : str
        Proj4 projection string
        
    Returns:
    --------
    np.ndarray : Projected coordinates
    """
    # Create transformer from WGS84 to target projection
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj4_string, always_xy=True)
    
    # Project coordinates
    x_proj, y_proj = transformer.transform(coords[:, 0], coords[:, 1])
    
    return np.column_stack([x_proj, y_proj])

def remove_duplicate_coords(coords: np.ndarray, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove duplicate coordinates within tolerance.
    
    Parameters:
    -----------
    coords : np.ndarray
        Coordinate array of shape (n_obs, 2)
    tolerance : float
        Distance tolerance for considering points duplicates
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, int] : (unique_coords, unique_indices, n_duplicates)
    """
    if len(coords) <= 1:
        return coords, np.arange(len(coords)), 0
    
    # Compute pairwise distances
    distances = squareform(pdist(coords))
    
    # Find duplicates
    i_indices, j_indices = np.where(distances < tolerance)
    off_diagonal_mask = i_indices != j_indices
    duplicate_pairs = (i_indices[off_diagonal_mask], j_indices[off_diagonal_mask])
    
    # Build set of indices to remove (keep first occurrence)
    indices_to_remove = set()
    for i, j in zip(duplicate_pairs[0], duplicate_pairs[1]):
        if i < j:  # Only consider upper triangle
            indices_to_remove.add(j)
    
    # Keep non-duplicate indices
    unique_indices = np.array([i for i in range(len(coords)) if i not in indices_to_remove])
    unique_coords = coords[unique_indices]
    n_duplicates = len(coords) - len(unique_coords)
    
    return unique_coords, unique_indices, n_duplicates

def preprocess_coords(coords: np.ndarray, tolerance: float = 1e-6, remove_duplicates: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Validate, project, and clean coordinates for SPDE spatial modeling.
    
    Parameters:
    -----------
    coords : np.ndarray
        Input coordinates of shape (n_obs, 2)
        Can be lon/lat (auto-detected) or already projected
    tolerance : float
        Tolerance for duplicate detection (default: 1e-6)
    remove_duplicates : bool
        If True, remove duplicate coordinates. If False, keep all coordinates
        but report close points (default: True)
        
    Returns:
    --------
    projected_coords : np.ndarray
        Clean projected coordinates
    kept_indices : np.ndarray
        Indices of non-duplicate points from original array (all indices if remove_duplicates=False)
    projection_info : Dict
        Dictionary containing:
        - 'proj4_string': Proj4 projection string
        - 'system': Projection system name
        - 'scale': Auto-detected scale ('single_region', 'multi_region', 'global')
        - 'projected_bbox': (x_min, y_min, x_max, y_max) in projected coords
        - 'hull_diameter_km': Approximate diameter in kilometers
        - 'antimeridian_crossing': Boolean flag
        - 'close_points': List of close point pairs (if remove_duplicates=False)
        
    Raises:
    -------
    CoordsError : If input validation fails
    """
    
    # Input validation
    if not isinstance(coords, np.ndarray):
        coords = np.asarray(coords)
    
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise CoordsError(f"Expected coords shape (n_obs, 2), got {coords.shape}")
    
    if len(coords) < 7:
        raise CoordsError(f"Minimum 7 coordinates required for SPDE modeling, got {len(coords)}")
    
    if np.any(~np.isfinite(coords)):
        raise CoordsError("Coordinates contain NaN or infinite values")
    
    # Detect coordinate system
    is_geo = is_geographic(coords)
    antimeridian_crossing = False
    
    if is_geo:
        print("Detected geographic coordinates (lon/lat)")
        
        # Handle antimeridian crossing
        if detect_antimeridian_crossing(coords):
            print("Antimeridian crossing detected - unwrapping coordinates")
            coords = unwrap_antimeridian(coords)
            antimeridian_crossing = True
        
        # Compute spatial extent for scale determination
        hull_diameter_deg = compute_convex_hull_diameter(coords)
        hull_diameter_km = degrees_to_km_estimate(coords, hull_diameter_deg)
        
        # Determine projection scale automatically
        detected_scale = determine_projection_scale(coords, hull_diameter_deg)
        print(f"Auto-detected scale: {detected_scale} (diameter ≈ {hull_diameter_km:.1f} km)")
        
        # Create projection
        proj4_string = create_projection_string(detected_scale, coords)
        
        # Project coordinates
        projected_coords = project_coordinates(coords, proj4_string)
        
        # Determine system name
        if detected_scale == 'single_region':
            zone_number, hemisphere = determine_utm_zone(coords)
            system_name = f"UTM Zone {zone_number}{hemisphere[0].upper()}"
        elif detected_scale == 'multi_region':
            system_name = "Albers Equal-Area Conic"
        else:
            system_name = "Mollweide Equal-Area"
            
        print(f"Projected to: {system_name}")
        
    else:
        print("Detected projected coordinates - using as-is")
        projected_coords = coords.copy()
        hull_diameter_km = compute_convex_hull_diameter(coords) / 1000  # Assume meters
        detected_scale = 'unknown'
        proj4_string = "+proj=unknown"
        system_name = "User-provided projection"
    
    # Handle duplicates based on remove_duplicates flag
    if remove_duplicates:
        clean_coords, kept_indices, n_duplicates = remove_duplicate_coords(
            projected_coords, tolerance)
        
        if n_duplicates > 0:
            print(f"Removed {n_duplicates} duplicate coordinates (tolerance={tolerance})")
        close_points = []
    else:
        # Keep all coordinates but identify close points
        clean_coords = projected_coords
        kept_indices = np.arange(len(projected_coords))
        
        # Find close points for reporting
        close_points = []
        if len(projected_coords) > 1:
            distances = squareform(pdist(projected_coords))
            for i in range(len(projected_coords)):
                for j in range(i + 1, len(projected_coords)):
                    dist = distances[i, j]
                    if dist < tolerance:
                        close_points.append({
                            'indices': (i, j),
                            'distance': dist,
                            'coords': (projected_coords[i], projected_coords[j])
                        })
        
        if close_points:
            print(f"Warning: Found {len(close_points)} close coordinate pairs (within tolerance={tolerance})")
            print("  Consider reviewing these points based on your domain knowledge")
    
    # Compute projected bounding box
    x_min, x_max = clean_coords[:, 0].min(), clean_coords[:, 0].max()
    y_min, y_max = clean_coords[:, 1].min(), clean_coords[:, 1].max()
    projected_bbox = (x_min, y_min, x_max, y_max)
    
    # Create projection info dictionary
    projection_info = {
        'proj4_string': proj4_string,
        'system': system_name,
        'scale': detected_scale,
        'projected_bbox': projected_bbox,
        'hull_diameter_km': hull_diameter_km,
        'antimeridian_crossing': antimeridian_crossing
    }
    
    # Add close_points info if not removing duplicates
    if not remove_duplicates and close_points:
        projection_info['close_points'] = close_points
    
    if remove_duplicates:
        print(f"Coordinate preprocessing complete: {len(coords)} -> {len(clean_coords)} points")
    else:
        print(f"Coordinate preprocessing complete: {len(coords)} points retained")
    print(f"Projected extent: {x_max - x_min:.0f} × {y_max - y_min:.0f} meters")
    
    return clean_coords, kept_indices, projection_info

# Example usage
if __name__ == "__main__":
    # Test with various coordinate types
    
    # Example 1: Small region (UTM expected)
    print("=== Example 1: Small region ===")
    local_coords = np.array([
        [-122.4194, 37.7749],  # San Francisco
        [-122.4094, 37.7849],
        [-122.4294, 37.7649],
        [-122.4394, 37.7549],
        [-122.4494, 37.7449],
        [-122.4594, 37.7349],
        [-122.4694, 37.7249],
    ])
    
    try:
        clean_coords, indices, proj_info = preprocess_coords(local_coords)
        print(f"Projection: {proj_info['system']}")
        print(f"Proj4: {proj_info['proj4_string']}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Continental scale (Albers expected)
    print("=== Example 2: Continental scale ===")
    continental_coords = np.array([
        [-125.0, 49.0],  # Pacific Northwest
        [-67.0, 45.0],   # Atlantic Northeast  
        [-97.0, 25.0],   # Gulf Coast
        [-120.0, 35.0],  # California
        [-80.0, 26.0],   # Florida
        [-110.0, 45.0],  # Mountain West
        [-90.0, 40.0],   # Midwest
    ])
    
    try:
        clean_coords, indices, proj_info = preprocess_coords(continental_coords)
        print(f"Projection: {proj_info['system']}")
        print(f"Diameter: {proj_info['hull_diameter_km']:.1f} km")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
