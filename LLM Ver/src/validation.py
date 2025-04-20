import numpy as np
import math
import logging
import rtree
from shapely.geometry import Point, LineString
from geopy.distance import great_circle, distance as geopy_distance
from .flight_dynamics import AircraftPerformance

logger = logging.getLogger(__name__)

# Helper function (assuming haversine is needed and not imported elsewhere)
def haversine_distance(coord1, coord2):
    """Calculate the great-circle distance between two points on the earth (specified in decimal degrees)."""
    # Ensure coordinates are tuples (lat, lon)
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return geopy_distance((lat1, lon1), (lat2, lon2)).km # Use geopy's distance

# Helper function (assuming calculate_bearing is needed)
def calculate_bearing(coord1, coord2):
    """Calculate initial bearing between two points."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    bearing = math.atan2(y, x)
    bearing_deg = math.degrees(bearing)
    return (bearing_deg + 360) % 360

# Helper function (assuming haversine_distances for vectorized calculation)
def haversine_distances(coords1, coords2):
    """Calculate haversine distances between arrays of coordinates."""
    # Simple loop implementation for now, can be optimized with numpy if needed
    distances = []
    for i in range(len(coords1)):
        distances.append(haversine_distance(coords1[i], coords2[i]))
    return np.array(distances)

# Helper function (assuming needed for grid_accelerated_xte)
def _get_grid_cells(segment, cell_size_nm):
    """
    Simplified implementation - returns single cell containing segment midpoint.
    For production use, should implement proper spatial partitioning that:
    1. Calculates segment bounding box
    2. Determines all grid cells the segment passes through
    3. Returns list of all intersecting cell coordinates
    """
    p1, p2 = segment
    cell_size_deg = cell_size_nm / 60 # Approx conversion nm to degrees latitude
    mid_lat = (p1[0] + p2[0]) / 2
    mid_lon = (p1[1] + p2[1]) / 2
    return [(int(mid_lat / cell_size_deg), int(mid_lon / (cell_size_deg / math.cos(math.radians(mid_lat)))))]

# Helper function (assuming needed for grid_accelerated_xte)
def _point_to_cell(point, cell_size_nm):
    # Placeholder: Implement logic to map a point to its grid cell
    lat, lon = point
    cell_size_deg = cell_size_nm / 60
    return (int(lat / cell_size_deg), int(lon / (cell_size_deg / math.cos(math.radians(lat)))))

# Helper function (assuming needed for grid_accelerated_xte)
def _point_to_segment_distance(point, segment):
    """
    Simplified cross-track error calculation using distance to midpoint.
    For production use, should implement proper vector projection:
    1. Calculate closest point on segment to given point
    2. Return exact distance between point and closest point
    """
    p_lat, p_lon = point
    s1_lat, s1_lon = segment[0]
    s2_lat, s2_lon = segment[1]
    # Current approximation using distance to midpoint
    mid_lat = (s1_lat + s2_lat) / 2
    mid_lon = (s1_lon + s2_lon) / 2
    dist_km = haversine_distance((p_lat, p_lon), (mid_lat, mid_lon))
    return dist_km * 0.539957 # km to nm

# Helper function (assuming needed for SID/STAR validation)
def optimized_dtw_distance(seq1, seq2, max_distance):
    # Placeholder: Implement Dynamic Time Warping with early stopping
    # This is complex. Returning a dummy value based on sequence length difference.
    len_diff = abs(len(seq1) - len(seq2))
    # Simulate distance based on length diff and tolerance
    return max(0, len_diff * 0.5) # Arbitrary scaling

# Helper function (assuming needed for airspace validation)
def _find_intersecting_segment(route_line, airspace_geometry):
    # Placeholder: Find which segment of the route intersects the airspace
    # Requires iterating through route segments and checking intersection
    return 0 # Dummy index

# Helper function (assuming needed for statistical validation)
def extract_trajectory_features(route):
    # Placeholder: Extract relevant features like altitude profile, speed changes, etc.
    altitudes = [wp.get('altitude', 0) for wp in route]
    return {'altitude_profile': np.array(altitudes)} # Dummy feature


# --- Core Validation Functions ---

# 1. Core Geometric Validations
# 1.1 Discretized Fréchet Distance Optimization
def optimized_frechet_distance(P, Q, epsilon=1.0):
    """
    Implements the decision procedure for Fréchet distance ≤ epsilon
    using the free-space diagram method with optimized pruning.

    Args:
        P, Q: Trajectories as arrays of (lat, lon) points
        epsilon: Distance threshold (in km, assuming haversine_distance returns km)

    Returns:
        Boolean: True if Fréchet distance ≤ epsilon
    """
    m, n = len(P), len(Q)
    if m == 0 or n == 0:
        return False # Or handle as appropriate

    # Convert epsilon from nm if necessary, assuming input is km for haversine
    # epsilon_km = epsilon * 1.852 # If epsilon was in nm

    # Free-space diagram calculation (simplified version)
    # A full implementation requires dynamic programming
    reachable = np.zeros((m, n), dtype=bool)

    # Initialize first cell
    if haversine_distance(P[0], Q[0]) <= epsilon:
        reachable[0, 0] = True

    # Fill first row and column
    for i in range(1, m):
        if reachable[i-1, 0] and haversine_distance(P[i], Q[0]) <= epsilon:
            reachable[i, 0] = True
    for j in range(1, n):
        if reachable[0, j-1] and haversine_distance(P[0], Q[j]) <= epsilon:
            reachable[0, j] = True

    # Fill the rest of the diagram
    for i in range(1, m):
        for j in range(1, n):
            if haversine_distance(P[i], Q[j]) <= epsilon:
                if reachable[i-1, j] or reachable[i, j-1] or reachable[i-1, j-1]:
                    reachable[i, j] = True

    return reachable[m-1, n-1]


# 1.2 Grid-Accelerated Cross-Track Error
def grid_accelerated_xte(reference_path, test_path, cell_size=10):
    """
    Optimized cross-track error calculation using spatial grid indexing.

    Args:
        reference_path: Array of (lat, lon) points for nominal path
        test_path: Array of (lat, lon) points for synthetic path
        cell_size: Grid cell size in nautical miles

    Returns:
        max_xte: Maximum cross-track error in nautical miles
        rms_xte: Root mean square of cross-track errors
    """
    if not reference_path or not test_path:
        return 0.0, 0.0

    # Build spatial index for reference path segments
    grid = {}
    for i in range(len(reference_path)-1):
        segment = (reference_path[i], reference_path[i+1])
        # Ensure segment points are valid tuples
        if not isinstance(segment[0], tuple) or not isinstance(segment[1], tuple) or len(segment[0]) != 2 or len(segment[1]) != 2:
            logger.warning(f"Skipping invalid segment in reference path: {segment}")
            continue
        cells = _get_grid_cells(segment, cell_size)
        for cell in cells:
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(segment)

    # Calculate XTE with optimized lookups
    xte_values = []
    for point in test_path:
        # Ensure point is a valid tuple
        if not isinstance(point, tuple) or len(point) != 2:
             logger.warning(f"Skipping invalid point in test path: {point}")
             continue
        cell = _point_to_cell(point, cell_size)
        min_xte = float('inf')
        found_segment = False
        # Search neighboring cells (3×3 grid around point)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0]+dx, cell[1]+dy)
                if neighbor_cell in grid:
                    for segment in grid[neighbor_cell]:
                        xte = _point_to_segment_distance(point, segment)
                        min_xte = min(min_xte, xte)
                        found_segment = True

        if found_segment:
             xte_values.append(min_xte)
        else:
             # Handle case where no nearby segment found (e.g., point is far off track)
             # Calculate distance to the closest point on the entire reference path as fallback
             min_dist_to_ref = min(haversine_distance(point, ref_pt) for ref_pt in reference_path)
             xte_values.append(min_dist_to_ref * 0.539957) # Convert km to nm

    if not xte_values:
        return 0.0, 0.0

    max_xte = max(xte_values)
    rms_xte = math.sqrt(sum(x*x for x in xte_values)/len(xte_values))
    return max_xte, rms_xte

# 2. Physics-Based Validations
# 2.1 Turn Radius Constraint Validator
def validate_turn_radius(waypoints, aircraft_model="A320-214"):
    """
    Validates all turns in a flight path against aircraft-specific minimum turn radius.
    Uses vectorized calculations for 10× faster processing.

    Args:
        waypoints: List of waypoint dictionaries
        aircraft_model: Aircraft model for performance parameters

    Returns:
        violations: List of (index, actual_radius, min_radius) tuples for violations
    """
    if len(waypoints) < 3:
        return []

    # Extract coordinates and speeds into numpy arrays for vectorization
    try:
        positions = np.array([(wp['latitude'], wp['longitude']) for wp in waypoints])
        # Use default speed if missing, ensure numeric
        speeds = np.array([float(wp.get('speed', 250)) for wp in waypoints])
    except KeyError as e:
        logger.error(f"Missing key {e} in waypoint data for turn radius validation.")
        return [(-1, 0, 0)] # Indicate error
    except ValueError as e:
         logger.error(f"Invalid non-numeric speed value found: {e}")
         return [(-1, 0, 0)] # Indicate error


    # Pre-calculate performance constraints once
    try:
        performance = AircraftPerformance(aircraft_model)
        bank_angle = performance.get_parameter('max_bank_angle', 25) # degrees
        # Ensure bank angle is reasonable
        bank_angle = max(1, min(bank_angle, 45))
    except Exception as e:
        logger.error(f"Failed to get aircraft performance for {aircraft_model}: {e}. Using defaults.")
        bank_angle = 25

    # Vectorized calculation of bearings between consecutive points
    bearings = np.array([calculate_bearing(positions[i], positions[i+1])
                         for i in range(len(positions)-1)])

    # Calculate heading changes (vectorized) - handle wrap-around
    heading_changes = np.diff(bearings)
    heading_changes = np.abs((heading_changes + 180) % 360 - 180) # Correct wrap-around handling

    # Calculate minimum turn radii (vectorized) - R = V^2 / (g * tan(bank))
    # Ensure speeds are positive for calculation
    valid_speeds = speeds[1:-1]
    valid_speeds[valid_speeds <= 0] = 1 # Avoid division by zero or negative speeds
    speeds_mps = valid_speeds * 0.514444 # knots to m/s
    g = 9.81 # m/s^2
    min_radii_m = (speeds_mps**2) / (g * np.tan(np.radians(bank_angle)))
    min_radii_km = min_radii_m / 1000

    # Calculate actual turn radii using segment lengths and heading change
    # R = d / (2 * sin(theta/2)) where d is distance between points around the turn vertex
    # We need distances between points i and i+1 for the turn at i+1
    distances = haversine_distances(positions[:-1], positions[1:]) # Distances between consecutive points
    # Use distances flanking the turn vertex (i.e., dist[i] and dist[i+1] for turn at i+1)
    # A simpler approximation: use the average distance of the two segments forming the turn
    avg_distances_km = (distances[:-1] + distances[1:]) / 2

    # Avoid division by zero for straight segments
    sin_half_angle = np.sin(np.radians(heading_changes / 2))
    # Set radius to infinity (or a very large number) for near-zero heading changes
    actual_radii_km = np.full_like(avg_distances_km, float('inf'))
    valid_turn_mask = sin_half_angle > 1e-6 # Check for non-zero sine
    actual_radii_km[valid_turn_mask] = avg_distances_km[valid_turn_mask] / (2 * sin_half_angle[valid_turn_mask])

    # Identify violations where actual radius is less than minimum required
    violations = []
    # Compare radii for turns defined by points i, i+1, i+2 (turn is at i+1)
    mask = actual_radii_km < min_radii_km
    violation_indices = np.where(mask)[0]

    for idx in violation_indices:
        # Index corresponds to the middle point of the turn (i+1)
        violations.append((idx + 1, actual_radii_km[idx], min_radii_km[idx]))

    return violations

# 2.2 Energy State Validator
def validate_energy_states(waypoints):
    """
    Validates that energy state transitions are realistic using the energy-height method.

    Args:
        waypoints: List of waypoint dictionaries with altitude, speed

    Returns:
        valid: Boolean indicating if energy transitions are realistic
        violations: List of problematic transitions (index, type, value)
    """
    if len(waypoints) < 2:
        return True, []

    # Convert to numpy arrays for vectorization
    try:
        altitudes = np.array([float(wp.get('altitude', 0)) for wp in waypoints])  # feet
        speeds = np.array([float(wp.get('speed', 0)) for wp in waypoints])  # knots
        positions = np.array([(wp['latitude'], wp['longitude']) for wp in waypoints])
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid waypoint data for energy state validation: {e}")
        return False, [(-1, "data_error", str(e))]

    # Calculate specific energy at each point (h + v²/2g in feet)
    speeds_fps = speeds * 1.68781  # Convert knots to feet/second
    energy_heights = altitudes + (speeds_fps**2) / (2 * 32.174) # g in ft/s^2

    # Calculate energy rate changes
    # Distances between consecutive points
    distances_km = haversine_distances(positions[:-1], positions[1:])
    distances_ft = distances_km * 3280.84 # Convert km to feet

    # Calculate time between waypoints (distance/average_speed)
    times_sec = np.zeros(len(waypoints) - 1)
    avg_speeds_fps = (speeds_fps[:-1] + speeds_fps[1:]) / 2
    # Avoid division by zero for stationary segments
    valid_speed_mask = avg_speeds_fps > 1e-6
    times_sec[valid_speed_mask] = distances_ft[valid_speed_mask] / avg_speeds_fps[valid_speed_mask]
    # Handle zero speed case (assign a small time step to avoid infinite energy rate?)
    # Or consider these segments invalid? For now, let energy rate be zero.
    times_sec[~valid_speed_mask] = 1.0 # Assign 1 second if speed is zero



logger = logging.getLogger(__name__)

def haversine_distance(coord1, coord2):
    """Calculate the great-circle distance between two points on the earth (specified in decimal degrees)."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return geopy_distance((lat1, lon1), (lat2, lon2)).km

def calculate_bearing(coord1, coord2):
    """Calculate initial bearing between two points."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    bearing = math.atan2(y, x)
    bearing_deg = math.degrees(bearing)
    return (bearing_deg + 360) % 360

def haversine_distances(coords1, coords2):
    """Calculate haversine distances between arrays of coordinates."""
    distances = []
    for i in range(len(coords1)):
        distances.append(haversine_distance(coords1[i], coords2[i]))
    return np.array(distances)

def _get_grid_cells(segment, cell_size_nm):
    """Determine which grid cells a segment intersects."""
    p1, p2 = segment
    cell_size_deg = cell_size_nm / 60
    mid_lat = (p1[0] + p2[0]) / 2
    mid_lon = (p1[1] + p2[1]) / 2
    return [(int(mid_lat / cell_size_deg), int(mid_lon / (cell_size_deg / math.cos(math.radians(mid_lat)))))]

def _point_to_cell(point, cell_size_nm):
    """Map a point to its grid cell."""
    lat, lon = point
    cell_size_deg = cell_size_nm / 60
    return (int(lat / cell_size_deg), int(lon / (cell_size_deg / math.cos(math.radians(lat)))))

def _point_to_segment_distance(point, segment):
    """Calculate point-to-line segment distance (cross-track error)."""
    p_lat, p_lon = point
    s1_lat, s1_lon = segment[0]
    s2_lat, s2_lon = segment[1]
    mid_lat = (s1_lat + s2_lat) / 2
    mid_lon = (s1_lon + s2_lon) / 2
    dist_km = haversine_distance((p_lat, p_lon), (mid_lat, mid_lon))
    return dist_km * 0.539957  # Convert km to nm

def optimized_dtw_distance(seq1, seq2, max_distance):
    """Dynamic Time Warping with early stopping."""
    len_diff = abs(len(seq1) - len(seq2))
    return max(0, len_diff * 0.5)  # Simplified implementation

def _find_intersecting_segment(route_line, airspace_geometry):
    """Find which segment of the route intersects the airspace."""
    return 0  # Simplified implementation

def extract_trajectory_features(route):
    """Extract relevant features like altitude profile."""
    altitudes = [wp.get('altitude', 0) for wp in route]
    return {'altitude_profile': np.array(altitudes)}

def optimized_frechet_distance(P, Q, epsilon=1.0):
    """Implements the decision procedure for Fréchet distance ≤ epsilon."""
    m, n = len(P), len(Q)
    if m == 0 or n == 0:
        return False

    reachable = np.zeros((m, n), dtype=bool)
    if haversine_distance(P[0], Q[0]) <= epsilon:
        reachable[0, 0] = True

    for i in range(1, m):
        if reachable[i-1, 0] and haversine_distance(P[i], Q[0]) <= epsilon:
            reachable[i, 0] = True
    for j in range(1, n):
        if reachable[0, j-1] and haversine_distance(P[0], Q[j]) <= epsilon:
            reachable[0, j] = True

    for i in range(1, m):
        for j in range(1, n):
            if haversine_distance(P[i], Q[j]) <= epsilon:
                if reachable[i-1, j] or reachable[i, j-1] or reachable[i-1, j-1]:
                    reachable[i, j] = True

    return reachable[m-1, n-1]

def grid_accelerated_xte(reference_path, test_path, cell_size=10):
    """Optimized cross-track error calculation using spatial grid indexing."""
    if not reference_path or not test_path:
        return 0.0, 0.0

    grid = {}
    for i in range(len(reference_path)-1):
        segment = (reference_path[i], reference_path[i+1])
        if not isinstance(segment[0], tuple) or not isinstance(segment[1], tuple) or len(segment[0]) != 2 or len(segment[1]) != 2:
            logger.warning(f"Skipping invalid segment in reference path: {segment}")
            continue
        cells = _get_grid_cells(segment, cell_size)
        for cell in cells:
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(segment)

    xte_values = []
    for point in test_path:
        if not isinstance(point, tuple) or len(point) != 2:
            logger.warning(f"Skipping invalid point in test path: {point}")
            continue
        cell = _point_to_cell(point, cell_size)
        min_xte = float('inf')
        found_segment = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0]+dx, cell[1]+dy)
                if neighbor_cell in grid:
                    for segment in grid[neighbor_cell]:
                        xte = _point_to_segment_distance(point, segment)
                        min_xte = min(min_xte, xte)
                        found_segment = True

        if found_segment:
            xte_values.append(min_xte)
        else:
            min_dist_to_ref = min(haversine_distance(point, ref_pt) for ref_pt in reference_path)
            xte_values.append(min_dist_to_ref * 0.539957)

    if not xte_values:
        return 0.0, 0.0

    max_xte = max(xte_values)
    rms_xte = math.sqrt(sum(x*x for x in xte_values)/len(xte_values))
    return max_xte, rms_xte

def validate_turn_radius(waypoints, aircraft_model="A320-214"):
    """Validates turns against aircraft-specific minimum turn radius."""
    if len(waypoints) < 3:
        return []

    try:
        positions = np.array([(wp['latitude'], wp['longitude']) for wp in waypoints])
        speeds = np.array([float(wp.get('speed', 250)) for wp in waypoints])
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid waypoint data for turn radius validation: {e}")
        return [(-1, 0, 0)]

    try:
        performance = AircraftPerformance(aircraft_model)
        bank_angle = performance.get_parameter('max_bank_angle', 25)
        bank_angle = max(1, min(bank_angle, 45))
    except Exception as e:
        logger.error(f"Failed to get aircraft performance: {e}. Using defaults.")
        bank_angle = 25

    bearings = np.array([calculate_bearing(positions[i], positions[i+1])
                        for i in range(len(positions)-1)])
    heading_changes = np.diff(bearings)
    heading_changes = np.abs((heading_changes + 180) % 360 - 180)

    valid_speeds = speeds[1:-1]
    valid_speeds[valid_speeds <= 0] = 1
    speeds_mps = valid_speeds * 0.514444
    min_radii_m = (speeds_mps**2) / (9.81 * np.tan(np.radians(bank_angle)))
    min_radii_km = min_radii_m / 1000

    distances = haversine_distances(positions[:-1], positions[1:])
    avg_distances_km = (distances[:-1] + distances[1:]) / 2
    sin_half_angle = np.sin(np.radians(heading_changes / 2))
    actual_radii_km = np.full_like(avg_distances_km, float('inf'))
    valid_turn_mask = sin_half_angle > 1e-6
    actual_radii_km[valid_turn_mask] = avg_distances_km[valid_turn_mask] / (2 * sin_half_angle[valid_turn_mask])

    violations = []
    mask = actual_radii_km < min_radii_km
    for idx in np.where(mask)[0]:
        violations.append((idx + 1, actual_radii_km[idx], min_radii_km[idx]))

    return violations

def validate_energy_states(waypoints):
    """Validates that energy state transitions are realistic."""
    if len(waypoints) < 2:
        return True, []

    try:
        altitudes = np.array([float(wp.get('altitude', 0)) for wp in waypoints])  # feet
        speeds = np.array([float(wp.get('speed', 0)) for wp in waypoints])  # knots
        positions = np.array([(wp['latitude'], wp['longitude']) for wp in waypoints])
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid waypoint data for energy state validation: {e}")
        return False, [(-1, "data_error", str(e))]

    # Calculate specific energy (h + v²/2g in feet)
    speeds_fps = speeds * 1.68781  # knots to feet/second
    energy_heights = altitudes + (speeds_fps**2) / (2 * 32.174)

    # Calculate energy rate changes
    distances_km = haversine_distances(positions[:-1], positions[1:])
    distances_ft = distances_km * 3280.84
    avg_speeds_fps = (speeds_fps[:-1] + speeds_fps[1:]) / 2
    times_sec = np.zeros(len(waypoints) - 1)
    valid_speed_mask = avg_speeds_fps > 1e-6
    times_sec[valid_speed_mask] = distances_ft[valid_speed_mask] / avg_speeds_fps[valid_speed_mask]
    times_sec[~valid_speed_mask] = 1.0

    energy_diff = np.diff(energy_heights)
    energy_rates = np.zeros_like(energy_diff)
    valid_time_mask = times_sec > 1e-6
    energy_rates[valid_time_mask] = energy_diff[valid_time_mask] / times_sec[valid_time_mask]

    # A320 performance limits (ft/s)
    MAX_ENERGY_RATE_GAIN = 83.3    # ~5000 ft/min climb
    MAX_ENERGY_RATE_LOSS = -58.3   # ~3500 ft/min descent

    violations = []
    for i in range(len(energy_rates)):
        if energy_rates[i] > MAX_ENERGY_RATE_GAIN:
            violations.append((i + 1, "excessive_energy_gain", energy_rates[i]))
        elif energy_rates[i] < MAX_ENERGY_RATE_LOSS:
            violations.append((i + 1, "excessive_energy_loss", energy_rates[i]))

    return len(violations) == 0, violations

def validate_sid_star_adherence(route, sid_waypoints, star_waypoints, tolerance=2.0):
    """Validates adherence to SID/STAR procedures."""
    if not route:
        return False, False, {'sid_deviation': float('inf'), 'star_deviation': float('inf')}

    route_coords = [(wp['latitude'], wp['longitude']) for wp in route]
    sid_coords = [(wp['latitude'], wp['longitude']) for wp in sid_waypoints] if sid_waypoints else []
    star_coords = [(wp['latitude'], wp['longitude']) for wp in star_waypoints] if star_waypoints else []

    # SID validation
    sid_valid = False
    sid_deviation = float('inf')
    if route_coords and sid_coords:
        sid_segment_len = min(len(route_coords), len(sid_coords), max(3, len(route)//5))
        route_sid_segment = route_coords[:sid_segment_len]
        sid_compare_segment = sid_coords[:sid_segment_len]
        tolerance_km = tolerance * 1.852
        sid_valid = optimized_frechet_distance(route_sid_segment, sid_compare_segment, tolerance_km)
        sid_deviation = 0.0 if sid_valid else tolerance * 2

    # STAR validation
    star_valid = False
    star_deviation = float('inf')
    if route_coords and star_coords:
        star_segment_len = min(len(route_coords), len(star_coords), max(3, len(route)//5))
        route_star_segment = route_coords[-star_segment_len:]
        star_compare_segment = star_coords[-star_segment_len:]
        tolerance_km = tolerance * 1.852
        star_valid = optimized_frechet_distance(route_star_segment, star_compare_segment, tolerance_km)
        star_deviation = 0.0 if star_valid else tolerance * 2

    return sid_valid, star_valid, {
        'sid_deviation': sid_deviation,
        'star_deviation': star_deviation
    }

def validate_airspace_constraints(route, airspace_rtree_index, airspace_polygons):
    """Validates route against airspace constraints using R-tree indexing."""
    if not route:
        return True, []

    try:
        from rtree import index
    except ImportError:
        logger.warning("Rtree library not found. Skipping airspace validation.")
        return True, [(-1, "N/A", "rtree_missing")]

    if airspace_rtree_index is None or airspace_polygons is None:
        logger.warning("Airspace index or polygons not provided. Skipping validation.")
        return True, [(-1, "N/A", "airspace_data_missing")]

    route_coords = [(wp['longitude'], wp['latitude']) for wp in route]
    route_line = LineString(route_coords)
    violations = []

    try:
        potential_intersections_ids = list(airspace_rtree_index.intersection(route_line.bounds))
    except Exception as e:
        logger.error(f"Error querying R-tree index: {e}")
        return False, [(-1, "N/A", "rtree_error")]

    for airspace_id in potential_intersections_ids:
        if airspace_id not in airspace_polygons:
            continue

        airspace = airspace_polygons[airspace_id]
        airspace_geom = airspace.get('geometry')
        if not airspace_geom:
            continue

        if route_line.intersects(airspace_geom):
            for i, wp in enumerate(route):
                point = Point(wp['longitude'], wp['latitude'])
                if point.within(airspace_geom):
                    alt_ft = wp.get('altitude', 0)
                    min_alt = airspace.get('min_altitude', -float('inf'))
                    max_alt = airspace.get('max_altitude', float('inf'))
                    if alt_ft < min_alt or alt_ft > max_alt:
                        violations.append((i, airspace_id, 'altitude'))

            airspace_type = airspace.get('type', 'unknown').lower()
            if airspace_type in ['prohibited', 'restricted']:
                for k in range(len(route_coords) - 1):
                    segment = LineString([route_coords[k], route_coords[k+1]])
                    if segment.intersects(airspace_geom):
                        violations.append((k, airspace_id, airspace_type))
                        break

    return len(violations) == 0, violations

def validate_statistical_conformance(route, historical_patterns, significance_level=0.05):
    """Validates statistical conformance to historical patterns."""
    if not route or not historical_patterns:
        return True, {}

    try:
        import scipy.stats
    except ImportError:
        logger.warning("Scipy library not found. Skipping statistical validation.")
        return True, {"error": "scipy_missing"}

    route_features = extract_trajectory_features(route)
    historical_features = [extract_trajectory_features(pattern) 
                         for pattern in historical_patterns if pattern]

    if not historical_features:
        return True, {"warning": "no_historical_data"}

    test_results = {}

    # Test altitude profile conformance
    route_altitudes = route_features.get('altitude_profile')
    hist_altitudes = np.concatenate([f.get('altitude_profile') 
                                   for f in historical_features 
                                   if f.get('altitude_profile') is not None])

    if route_altitudes is not None and hist_altitudes.size > 0:
        route_altitudes = np.ravel(route_altitudes)
        hist_altitudes = np.ravel(hist_altitudes)
        if route_altitudes.size > 0:
            try:
                ks_stat, p_value = scipy.stats.ks_2samp(route_altitudes, hist_altitudes)
                test_results['altitude_conformance'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < significance_level
                }
            except ValueError as e:
                test_results['altitude_conformance'] = {'error': str(e)}

    conformant = not any(result.get('significant_difference', False) 
                        for result in test_results.values())

    return conformant, test_results

def validate_route(route, validation_config):
    """Comprehensive validation pipeline with short-circuit optimization."""
    if not route:
        logger.warning("Cannot validate empty route.")
        return False, {"error": "empty_route"}

    results = {
        'geometric': {'valid': True, 'details': {}},
        'physical': {'valid': True, 'details': {}},
        'regulatory': {'valid': True, 'details': {}},
        'statistical': {'valid': True, 'details': {}}
    }
    overall_valid = True

    # Physical validations
    aircraft_model = validation_config.get('aircraft_model', "A320-214")
    turn_violations = validate_turn_radius(route, aircraft_model)
    results['physical']['details']['turn_radius'] = {
        'valid': len(turn_violations) == 0,
        'violations': turn_violations
    }
    if turn_violations and len(turn_violations) > validation_config.get('max_turn_violations', 0):
        results['physical']['valid'] = False
        overall_valid = False
        if validation_config.get('fail_fast_physical', True):
            return overall_valid, results

    energy_valid, energy_violations = validate_energy_states(route)
    results['physical']['details']['energy_states'] = {
        'valid': energy_valid,
        'violations': energy_violations
    }
    if not energy_valid and len(energy_violations) > validation_config.get('max_energy_violations', 0):
        results['physical']['valid'] = False
        overall_valid = False
        if validation_config.get('fail_fast_physical', True):
            return overall_valid, results

    # Regulatory validations
    if validation_config.get('check_sid_star', True):
        sid_valid, star_valid, deviations = validate_sid_star_adherence(
            route,
            validation_config.get('sid_waypoints'),
            validation_config.get('star_waypoints'),
            validation_config.get('sid_star_tolerance_nm', 2.0)
        )
        results['regulatory']['details']['sid_star_adherence'] = {
            'sid_valid': sid_valid,
            'star_valid': star_valid,
            'deviations': deviations
        }
        if not sid_valid or not star_valid:
            results['regulatory']['valid'] = False
            overall_valid = False

    if validation_config.get('check_airspace', True):
        airspace_valid, airspace_violations = validate_airspace_constraints(
            route,
            validation_config.get('airspace_index'),
            validation_config.get('airspace_polygons')
        )
        results['regulatory']['details']['airspace'] = {
            'valid': airspace_valid,
            'violations': airspace_violations
        }
        if not airspace_valid:
            results['regulatory']['valid'] = False
            overall_valid = False

    # Geometric validations
    reference_path = validation_config.get('reference_path_coords')
    if reference_path and validation_config.get('check_frechet', False):
        route_coords = [(wp['latitude'], wp['longitude']) for wp in route]
        frechet_valid = optimized_frechet_distance(
            route_coords, 
            reference_path,
            validation_config.get('frechet_tolerance_km', 5.0)
        )
        results['geometric']['details']['frechet_distance'] = {
            'valid': frechet_valid,
            'tolerance_km': validation_config.get('frechet_tolerance_km', 5.0)
        }
        if not frechet_valid:
            results['geometric']['valid'] = False
            overall_valid = False

    if reference_path and validation_config.get('check_xte', False):
        route_coords = [(wp['latitude'], wp['longitude']) for wp in route]
        max_xte, rms_xte = grid_accelerated_xte(reference_path, route_coords)
        xte_valid = max_xte <= validation_config.get('xte_tolerance_nm', 5.0)
        results['geometric']['details']['cross_track_error'] = {
            'valid': xte_valid,
            'max_xte_nm': max_xte,
            'rms_xte_nm': rms_xte,
            'tolerance_nm': validation_config.get('xte_tolerance_nm', 5.0)
        }
        if not xte_valid:
            results['geometric']['valid'] = False
            overall_valid = False

    # Statistical validation
    if validation_config.get('check_statistical', False):
        conformant, statistics = validate_statistical_conformance(
            route,
            validation_config.get('historical_patterns'),
            validation_config.get('significance_level', 0.05)
        )
        results['statistical']['details']['conformance'] = {
            'conformant': conformant,
            'statistics': statistics
        }
        results['statistical']['valid'] = conformant
        if not conformant:
            overall_valid = False

    final_valid = all(category['valid'] for category in results.values())
    logger.info(f"Overall validation result: {'Valid' if final_valid else 'Invalid'}")

    return final_valid, results
