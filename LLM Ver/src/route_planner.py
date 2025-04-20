# \src\route_planner.py

import logging
import numpy as np
import random
from geopy.distance import great_circle
import copy
from scipy.interpolate import CubicSpline
import math

logger = logging.getLogger(__name__)

def select_sid(departure_airport, aip_data, variant=0):
    """
    Select a Standard Instrument Departure (SID) for the departure airport.
    
    Args:
        departure_airport: ICAO code of departure airport
        aip_data: AIP data dictionary
        variant: Variant number to select a different SID if available
        
    Returns:
        List of waypoints for the selected SID, or None if not found
    """
    if departure_airport not in aip_data:
        logger.warning(f"No AIP data found for departure airport: {departure_airport}")
        return None
    
    airport_data = aip_data[departure_airport]
    
    if 'SIDs' not in airport_data or not airport_data['SIDs']:
        logger.warning(f"No SIDs found for {departure_airport}")
        return None
    
    # Select a different SID based on variant number
    sid_index = variant % len(airport_data['SIDs'])
    sid = airport_data['SIDs'][sid_index]
    logger.info(f"Selected {sid['name']} for {departure_airport} departure (variant {variant})")
    
    # Validate waypoints have coordinates
    waypoints = sid.get('waypoints', [])
    valid_waypoints = []
    
    for wp in waypoints:
        if 'lat' in wp and 'lon' in wp:
            # Convert to standard format
            valid_wp = wp.copy()
            valid_wp['latitude'] = wp['lat']
            valid_wp['longitude'] = wp['lon']
            valid_waypoints.append(valid_wp)
        elif 'latitude' in wp and 'longitude' in wp:
            valid_waypoints.append(wp)
    
    return valid_waypoints

def select_star(arrival_airport, aip_data, variant=0):
    """
    Select a Standard Terminal Arrival Route (STAR) for the arrival airport.
    
    Args:
        arrival_airport: ICAO code of arrival airport
        aip_data: AIP data dictionary
        variant: Variant number to select a different STAR if available
        
    Returns:
        List of waypoints for the selected STAR, or None if not found
    """
    if arrival_airport not in aip_data:
        logger.warning(f"No AIP data found for arrival airport: {arrival_airport}")
        return None
    
    airport_data = aip_data[arrival_airport]
    
    if 'STARs' not in airport_data or not airport_data['STARs']:
        logger.warning(f"No STARs found for {arrival_airport}")
        return None
    
    # Select a different STAR based on variant number
    star_index = variant % len(airport_data['STARs'])
    star = airport_data['STARs'][star_index]
    logger.info(f"Selected {star['name']} for {arrival_airport} arrival (variant {star_index})")
    
    # Validate waypoints have coordinates
    waypoints = star.get('waypoints', [])
    valid_waypoints = []
    
    for wp in waypoints:
        if 'lat' in wp and 'lon' in wp:
            # Convert to standard format
            valid_wp = wp.copy()
            valid_wp['latitude'] = wp['lat']
            valid_wp['longitude'] = wp['lon']
            valid_waypoints.append(valid_wp)
        elif 'latitude' in wp and 'longitude' in wp:
            valid_waypoints.append(wp)
    
    return valid_waypoints

def extract_cluster_data(nominal_pattern):
    """
    Extract cluster data from nominal pattern waypoints.
    
    Args:
        nominal_pattern: The nominal pattern with waypoints
        
    Returns:
        Dictionary of cluster data for each waypoint
    """
    if not nominal_pattern or 'waypoints' not in nominal_pattern:
        return {}
    
    cluster_data = []
    
    for wp in nominal_pattern['waypoints']:
        # Extract cluster size - larger clusters indicate more variance
        cluster_size = wp.get('cluster_size', 2)
        
        # Estimate lateral variance based on cluster size (km)
        # Reduced from previous 0.05 to 0.025 to create less variation
        lateral_variance = 0.025 * cluster_size  # in km
        
        # Variance in altitude and speed (also based on cluster size)
        altitude_variance = 100 * cluster_size / 3  # in feet
        speed_variance = 5 * cluster_size / 3      # in knots
        
        cluster_data.append({
            'lateral_variance': lateral_variance,
            'altitude_variance': altitude_variance,
            'speed_variance': speed_variance,
            'cluster_size': cluster_size
        })
    
    return cluster_data

def calculate_min_turn_radius(speed_kts):
    """
    Calculate minimum turn radius based on aircraft physics.
    
    Args:
        speed_kts: Airspeed in knots
        
    Returns:
        Minimum turn radius in kilometers
    """
    # Convert knots to m/s
    speed_ms = speed_kts * 0.51444
    
    # Standard turn radius formula: R = V²/(g * tan(bank))
    # Using typical commercial bank angle of 25 degrees
    g = 9.81  # m/s²
    bank_angle_rad = math.radians(25)
    
    # Calculate minimum radius in meters
    radius_m = (speed_ms**2) / (g * math.tan(bank_angle_rad))
    
    # Convert to kilometers and add safety margin
    radius_km = (radius_m / 1000) * 1.2  # 20% safety margin
    
    # Ensure minimum realistic value
    return max(0.5, radius_km)

def generate_continuous_variation_field(route_length, num_points, variant_seed, max_variation=0.2, smoothness=5):
    """
    Generate a continuous variation field for the entire route.
    
    Args:
        route_length: Length of the route in points
        num_points: Number of control points for the variation field
        variant_seed: Seed for deterministic variation
        max_variation: Maximum variation amplitude (km)
        smoothness: Higher values create smoother variations
        
    Returns:
        Dictionary with 'lat_var' and 'lon_var' arrays of length route_length
    """
    # Use a consistent seed for the entire route variant
    np.random.seed(variant_seed)
    
    # Create control points for the variation field (fewer points = smoother variations)
    # Increased smoothness factor for more realistic paths
    control_points = max(3, num_points // smoothness)
    
    # Generate random control values for a smooth field (in km)
    # Using normal distribution with mean=0 and reduced stddev
    control_values_lat = np.random.normal(0, max_variation * 0.5, control_points)
    control_values_lon = np.random.normal(0, max_variation * 0.5, control_points)
    
    # Create parameter space
    control_params = np.linspace(0, 1, control_points)
    route_params = np.linspace(0, 1, route_length)
    
    # Create cubic spline interpolation for smooth variations
    lat_spline = CubicSpline(control_params, control_values_lat)
    lon_spline = CubicSpline(control_params, control_values_lon)
    
    # Interpolate to get variation field for all points
    lat_variations = lat_spline(route_params)
    lon_variations = lon_spline(route_params)
    
    return {
        'lat_var': lat_variations,
        'lon_var': lon_variations
    }

def apply_correlated_variations(waypoints, variant_seed, max_variation=0.2, cluster_data=None):
    """
    Apply correlated variations to waypoints to create realistic route alternatives.
    
    Args:
        waypoints: List of original waypoints
        variant_seed: Seed for deterministic variation
        max_variation: Maximum variation level (0-1, as fraction of km)
        cluster_data: Optional cluster data for variation scaling
        
    Returns:
        List of varied waypoints with smooth, correlated variations
    """
    if not waypoints or len(waypoints) < 2:
        return waypoints
    
    # Create a deep copy to avoid modifying original
    varied_waypoints = copy.deepcopy(waypoints)
    
    # Generate continuous variation field
    variation_field = generate_continuous_variation_field(
        len(waypoints), 
        len(waypoints), 
        variant_seed, 
        max_variation=max_variation
    )
    
    # Calculate average speed if available for turn radius constraints
    avg_speed = 450  # Default cruise speed in knots
    speed_points = 0
    
    for wp in varied_waypoints:
        if 'speed' in wp and wp['speed'] > 0:
            avg_speed += wp['speed']
            speed_points += 1
    
    if speed_points > 0:
        avg_speed = avg_speed / speed_points
    
    # Get minimum turn radius based on average speed
    min_turn_radius_km = calculate_min_turn_radius(avg_speed)
    
    # Apply variations to each waypoint
    for i, wp in enumerate(varied_waypoints):
        # Get variation scale from cluster data if available
        scale = 1.0
        if cluster_data and i < len(cluster_data):
            cluster_size = cluster_data[i].get('cluster_size', 2)
            # More conservative scaling
            scale = min(1.2, cluster_size / 5)
        
        # Get variation from the continuous field (in km)
        lat_var = variation_field['lat_var'][i] * scale
        lon_var = variation_field['lon_var'][i] * scale
        
        # Convert to coordinate offsets (km to degrees)
        lat_deg_per_km = 1/111.32  # 1 degree latitude is about 111.32 km
        lon_deg_per_km = 1/(111.32 * np.cos(np.radians(wp['latitude'])))
        
        # Apply variations
        wp['latitude'] += lat_var * lat_deg_per_km
        wp['longitude'] += lon_var * lon_deg_per_km
        
        # Apply more conservative variations to altitude and speed
        if 'altitude' in wp and cluster_data and i < len(cluster_data):
            alt_var = cluster_data[i].get('altitude_variance', 100) * max_variation * 0.3
            wp['altitude'] = max(0, wp['altitude'] + np.random.normal(0, alt_var))
            
        if 'speed' in wp and cluster_data and i < len(cluster_data):
            speed_var = cluster_data[i].get('speed_variance', 5) * max_variation * 0.3
            wp['speed'] = max(0, wp['speed'] + np.random.normal(0, speed_var))
    
    return varied_waypoints

def create_strictly_nominal_route(nominal_route, fixed_deviation=0.05):
    """
    Create a route that strictly follows the nominal path with minimal, 
    controlled variations. No random seeding is used to ensure consistent,
    realistic deviations.
    
    Args:
        nominal_route: The original nominal route
        fixed_deviation: Fixed deviation amount in km
        
    Returns:
        Route with minimal, deterministic variations
    """
    if not nominal_route:
        return []
    
    # Create a deep copy to avoid modifying original
    strict_route = copy.deepcopy(nominal_route)
    
    # Apply a sinusoidal variation pattern
    # This creates a smooth, wave-like deviation from the nominal path
    # without random fluctuations that could cause unrealistic paths
    for i, wp in enumerate(strict_route):
        # Use position in route to create smooth, alternating variations
        # sin() produces values between -1 and 1
        phase = i / max(1, len(nominal_route) - 1) * 4 * math.pi
        
        # Create alternating patterns for latitude and longitude
        lat_factor = math.sin(phase) * fixed_deviation
        lon_factor = math.sin(phase + math.pi/2) * fixed_deviation  # Phase shift for longitude
        
        # Convert to coordinate offsets (km to degrees)
        lat_deg_per_km = 1/111.32  # 1 degree latitude is about 111.32 km
        lon_deg_per_km = 1/(111.32 * np.cos(np.radians(wp['latitude'])))
        
        # Apply the deterministic variations
        wp['latitude'] += lat_factor * lat_deg_per_km
        wp['longitude'] += lon_factor * lon_deg_per_km
        
        # Mark as strict nominal for visualization
        wp['is_strict_nominal'] = True
    
    return strict_route

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate initial bearing between two points.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
        
    Returns:
        Bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    bearing = math.atan2(y, x)
    
    # Convert to degrees and normalize to 0-360
    bearing_deg = math.degrees(bearing)
    bearing_normalized = (bearing_deg + 360) % 360
    
    return bearing_normalized

def is_valid_path(waypoints):
    """
    Check if a route is realistic based on aircraft physics.
    
    Args:
        waypoints: List of waypoints
        
    Returns:
        True if path is valid, False otherwise
    """
    if len(waypoints) < 3:
        return True
    
    # Check turn angles and distances between points
    for i in range(1, len(waypoints)-1):
        prev_wp = waypoints[i-1]
        curr_wp = waypoints[i]
        next_wp = waypoints[i+1]
        
        # Get speed for turn radius calculation
        speed = curr_wp.get('speed', 450)  # Default to cruise speed if not specified
        
        # Calculate minimum turn radius based on speed
        min_turn_radius_km = calculate_min_turn_radius(speed)
        
        # Calculate bearings (using corrected formula)
        bearing1 = calculate_bearing(
            prev_wp['latitude'], prev_wp['longitude'],
            curr_wp['latitude'], curr_wp['longitude']
        )
        
        bearing2 = calculate_bearing(
            curr_wp['latitude'], curr_wp['longitude'],
            next_wp['latitude'], next_wp['longitude']
        )
        
        # Calculate absolute heading change (0-180 degrees)
        hdg_change = abs((bearing2 - bearing1 + 180) % 360 - 180)
        
        # Reject if heading change is too extreme (reduced from 120 to 90 degrees)
        if hdg_change > 90:
            return False
        
        # Calculate distances to check if they respect minimum turn radius
        dist1 = great_circle(
            (prev_wp['latitude'], prev_wp['longitude']), 
            (curr_wp['latitude'], curr_wp['longitude'])
        ).kilometers
        
        dist2 = great_circle(
            (curr_wp['latitude'], curr_wp['longitude']), 
            (next_wp['latitude'], next_wp['longitude'])
        ).kilometers
        
        # For significant turns, check if distances are sufficient for the turn radius
        if hdg_change > 30 and (dist1 < min_turn_radius_km or dist2 < min_turn_radius_km):
            return False
    
    return True

def find_nominal_pattern(departure, arrival, nominal_patterns, variant_id=0):
    """
    Find a nominal pattern for the route in the extracted patterns.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        nominal_patterns: Dictionary of nominal patterns
        variant_id: Identifier for selecting pattern variant
        
    Returns:
        Tuple of (waypoints, pattern, cluster_data)
    """
    # Check direct route
    route_key = f"{departure}-{arrival}"
    if route_key in nominal_patterns:
        logger.info(f"Found nominal pattern for {route_key}")
        pattern = nominal_patterns[route_key]
        cluster_data = extract_cluster_data(pattern)
        return pattern.get('waypoints', []), pattern, cluster_data
    
    # Check reverse route
    reverse_key = f"{arrival}-{departure}"
    if reverse_key in nominal_patterns:
        logger.info(f"Found nominal pattern for reverse route {reverse_key}, will reverse")
        pattern = nominal_patterns[reverse_key]
        waypoints = pattern.get('waypoints', [])
        if waypoints:
            # Reverse the waypoints for the opposite direction
            reversed_waypoints = list(reversed(waypoints))
            # Adjust headings if present
            for wp in reversed_waypoints:
                if 'heading' in wp and wp['heading'] is not None:
                    wp['heading'] = (wp['heading'] + 180) % 360
            
            # Reverse cluster data too
            cluster_data = extract_cluster_data(pattern)
            cluster_data.reverse()
            
            return reversed_waypoints, pattern, cluster_data
    
    logger.warning(f"No nominal pattern found for {departure}-{arrival}")
    return None, None, []

def create_direct_route(start_point, end_point, num_points=5):
    """
    Create a direct route between two points.
    
    Args:
        start_point: Start waypoint
        end_point: End waypoint
        num_points: Number of points to generate
        
    Returns:
        List of waypoints forming a direct route
    """
    # Validate input points have required coordinates
    if not start_point or not end_point:
        logger.error(f"Invalid input points: start={start_point}, end={end_point}")
        return []
    
    # Check for lat/lon in different formats
    start_lat = start_point.get('latitude')
    start_lon = start_point.get('longitude')
    
    if start_lat is None and 'lat' in start_point:
        start_lat = start_point['lat']
    if start_lon is None and 'lon' in start_point:
        start_lon = start_point['lon']
    
    end_lat = end_point.get('latitude')
    end_lon = end_point.get('longitude')
    
    if end_lat is None and 'lat' in end_point:
        end_lat = end_point['lat']
    if end_lon is None and 'lon' in end_point:
        end_lon = end_point['lon']
    
    # Check if we have valid coordinates
    if start_lat is None or start_lon is None:
        logger.error(f"Start point missing coordinates: {start_point}")
        # Use default coordinates
        start_lat = 1.3591  # WSSS latitude
        start_lon = 103.9895  # WSSS longitude
    
    if end_lat is None or end_lon is None:
        logger.error(f"End point missing coordinates: {end_point}")
        # Use default coordinates
        end_lat = 2.7456  # WMKK latitude
        end_lon = 101.7099  # WMKK longitude
    
    # Calculate distance
    try:
        distance = great_circle((start_lat, start_lon), (end_lat, end_lon)).kilometers
    except:
        logger.error(f"Error calculating distance between points: ({start_lat}, {start_lon}) and ({end_lat}, {end_lon})")
        return []
    
    # Increase number of points for longer distances to ensure smooth transitions
    if distance > 100:
        num_points = max(num_points, int(distance / 20))
    
    # Generate intermediate points
    waypoints = []
    for i in range(num_points):
        # Linear interpolation
        factor = i / (num_points - 1) if num_points > 1 else 0
        lat = start_lat + factor * (end_lat - start_lat)
        lon = start_lon + factor * (end_lon - start_lon)
        
        waypoint = {
            'name': f"WPT{i}",
            'latitude': lat,
            'longitude': lon
        }
        waypoints.append(waypoint)
    
    return waypoints

def interpolate_altitudes(waypoints, start_alt, end_alt):
    """
    Interpolate altitudes for a sequence of waypoints.
    
    Args:
        waypoints: List of waypoints
        start_alt: Starting altitude
        end_alt: Ending altitude
        
    Returns:
        List of waypoints with interpolated altitudes
    """
    num_points = len(waypoints)
    if num_points < 2:
        return waypoints
    
    for i, wp in enumerate(waypoints):
        factor = i / (num_points - 1)
        wp['altitude'] = start_alt + factor * (end_alt - start_alt)
    
    return waypoints

def connect_route_segments(sid_waypoints, enroute_waypoints, star_waypoints):
    """
    Connect SID, en-route, and STAR segments into a complete route with smooth transitions.
    
    Args:
        sid_waypoints: List of SID waypoints
        enroute_waypoints: List of en-route waypoints
        star_waypoints: List of STAR waypoints
        
    Returns:
        List of waypoints forming the complete route
    """
    complete_route = []
    
    # Add SID waypoints if available
    if sid_waypoints:
        complete_route.extend(sid_waypoints)
    
    # Add en-route waypoints with smooth connection if available
    if enroute_waypoints:
        if complete_route:
            # Create a smooth transition from SID to en-route
            # by inserting interpolated waypoints
            sid_exit = complete_route[-1]
            enroute_entry = enroute_waypoints[0]
            
            # Calculate distance for transition
            distance = great_circle(
                (sid_exit['latitude'], sid_exit['longitude']),
                (enroute_entry['latitude'], enroute_entry['longitude'])
            ).kilometers
            
            # Add transition points for smoother path if distance is significant
            if distance > 10:
                # Create 3 intermediate waypoints for transition
                transition_points = create_direct_route(sid_exit, enroute_entry, 3)
                
                # Add transition points (skip first as it duplicates SID exit)
                complete_route.extend(transition_points[1:-1])
            
            # Add en-route waypoints (skip the first one as we've handled the transition)
            complete_route.extend(enroute_waypoints[1:])
        else:
            complete_route.extend(enroute_waypoints)
    
    # Add STAR waypoints with smooth connection if available
    if star_waypoints:
        if complete_route:
            # Create a smooth transition from en-route to STAR
            enroute_exit = complete_route[-1]
            star_entry = star_waypoints[0]
            
            # Calculate distance for transition
            distance = great_circle(
                (enroute_exit['latitude'], enroute_exit['longitude']),
                (star_entry['latitude'], star_entry['longitude'])
            ).kilometers
            
            # Add transition points for smoother path if distance is significant
            if distance > 10:
                # Create 3 intermediate waypoints for transition
                transition_points = create_direct_route(enroute_exit, star_entry, 3)
                
                # Add transition points (skip first as it duplicates en-route exit)
                complete_route.extend(transition_points[1:-1])
            
            # Add STAR waypoints (skip the first one as we've handled the transition)
            complete_route.extend(star_waypoints[1:])
        else:
            complete_route.extend(star_waypoints)
    
    return complete_route

def create_varied_route(route, cluster_data, variant_id, max_variation=0.2):
    """
    Create a varied version of a route based on cluster data with correlated variations.
    
    Args:
        route: List of waypoints
        cluster_data: Cluster data for variations
        variant_id: Variant identifier
        max_variation: Maximum variation level (0-1)
        
    Returns:
        List of varied waypoints with realistic variations
    """
    if not route:
        return []
    
    # Use correlated variations for realistic paths
    varied_route = apply_correlated_variations(
        route, 
        variant_seed=variant_id, 
        max_variation=max_variation,
        cluster_data=cluster_data
    )
    
    # Ensure path is valid with aircraft physics constraints
    if not is_valid_path(varied_route):
        # If invalid, try again with significantly reduced variation
        return apply_correlated_variations(
            route, 
            variant_seed=variant_id + 1000,  # Different seed
            max_variation=max_variation * 0.4,  # More aggressive reduction
            cluster_data=cluster_data
        )
    
    return varied_route

def construct_varied_route(departure, arrival, aip_data, nominal_patterns, variant_id=0, max_variation=0.2):
    """
    Construct a varied route from departure to arrival.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        aip_data: AIP data dictionary
        nominal_patterns: Dictionary of nominal patterns
        variant_id: Variant identifier (0-n) for generating different routes
        max_variation: Maximum variation to apply (0-1)
        
    Returns:
        List of waypoints forming the complete route
    """
    # Use variant_id to select different SIDs/STARs and apply different variations
    
    # Step 1: Get SID for departure
    sid_waypoints = select_sid(departure, aip_data, variant=variant_id)
    
    # Step 2: Get STAR for arrival
    star_waypoints = select_star(arrival, aip_data, variant=variant_id)
    
    # Step 3: Get nominal en-route pattern with cluster data
    enroute_waypoints, pattern, cluster_data = find_nominal_pattern(
        departure, arrival, nominal_patterns, variant_id
    )
    
    # Apply controlled, correlated variations to en-route waypoints if available
    if enroute_waypoints:
        enroute_waypoints = create_varied_route(
            enroute_waypoints, cluster_data, variant_id, max_variation
        )
    
    # Step 4: Connect all segments with smooth transitions
    route = connect_route_segments(sid_waypoints, enroute_waypoints, star_waypoints)
    
    # If we still don't have a route, create a basic direct route
    if not route:
        logger.warning("Could not generate route from AIP or nominal data, falling back to direct route")
        # Use airport coordinates from AIP if available
        dep_coords = {'latitude': 1.3591, 'longitude': 103.9895}  # WSSS default
        arr_coords = {'latitude': 2.7456, 'longitude': 101.7099}  # WMKK default
        
        # Try to find coordinates in AIP data
        if departure in aip_data:
            if 'latitude' in aip_data[departure] and 'longitude' in aip_data[departure]:
                dep_coords = {
                    'latitude': aip_data[departure]['latitude'],
                    'longitude': aip_data[departure]['longitude']
                }
            elif aip_data[departure].get('navigation_aids'):
                for navaid in aip_data[departure]['navigation_aids']:
                    if 'lat' in navaid and 'lon' in navaid:
                        dep_coords = {
                            'latitude': navaid['lat'],
                            'longitude': navaid['lon']
                        }
                        break
        
        if arrival in aip_data:
            if 'latitude' in aip_data[arrival] and 'longitude' in aip_data[arrival]:
                arr_coords = {
                    'latitude': aip_data[arrival]['latitude'],
                    'longitude': aip_data[arrival]['longitude']
                }
            elif aip_data[arrival].get('navigation_aids'):
                for navaid in aip_data[arrival]['navigation_aids']:
                    if 'lat' in navaid and 'lon' in navaid:
                        arr_coords = {
                            'latitude': navaid['lat'],
                            'longitude': navaid['lon']
                        }
                        break
        
        # Create direct route with more points for better smoothness
        route = create_direct_route(dep_coords, arr_coords, num_points=15)
    
    # Ensure all waypoints have names
    for i, wp in enumerate(route):
        if 'name' not in wp or not wp['name']:
            wp['name'] = f"WP{i}"
    
    # Check if the path is valid, if not try with reduced variation
    if not is_valid_path(route) and max_variation > 0.05:
        logger.warning(f"Invalid path detected for variant {variant_id}, trying with reduced variation")
        return construct_varied_route(departure, arrival, aip_data, nominal_patterns, 
                                    variant_id, max_variation=max_variation*0.4)
    
    logger.info(f"Generated route variant {variant_id} with {len(route)} waypoints")
    return route

def generate_synthetic_flights(departure, arrival, aip_data, nominal_patterns, count=1, max_variation=0.2):
    """
    Generate multiple synthetic flight paths between two airports.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        aip_data: AIP data dictionary
        nominal_patterns: Dictionary of nominal patterns
        count: Number of different flight paths to generate
        max_variation: Maximum variation level (0-1)
        
    Returns:
        List of routes (each route is a list of waypoints)
    """
    synthetic_routes = []
    
    # Cap max_variation to prevent unrealistic paths - reduced from 0.4 to 0.25
    max_variation = min(max_variation, 0.25)
    
    for i in range(count):
        # Use graduated variation - first routes closer to nominal
        # More conservative scaling to prevent excessive variation
        variation_level = max_variation * (0.4 + 0.6 * (i / max(1, count - 1)))
        
        # Seed with a consistent, variant-specific seed
        random.seed(i * 1000)
        np.random.seed(i * 1000)
        
        route = construct_varied_route(
            departure, arrival, aip_data, nominal_patterns, variant_id=i, max_variation=variation_level
        )
        
        # Validate path - if invalid, reduce variation and try again
        if not is_valid_path(route) and variation_level > 0.05:
            logger.warning(f"Invalid path detected for variant {i}, retrying with reduced variation")
            route = construct_varied_route(
                departure, arrival, aip_data, nominal_patterns, variant_id=i, max_variation=variation_level * 0.4
            )
        
        synthetic_routes.append(route)
        logger.info(f"Generated synthetic flight {i+1}/{count}: {departure}-{arrival} with {len(route)} waypoints")
    
    return synthetic_routes

def construct_nominal_route(departure, arrival, aip_data, nominal_patterns):
    """
    Original function maintained for backward compatibility.
    Construct a nominal route from departure to arrival using AIP and nominal patterns.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        aip_data: AIP data dictionary
        nominal_patterns: Dictionary of nominal patterns
        
    Returns:
        List of waypoints forming the complete route
    """
    # This now just calls the new function with default parameters for no variation
    return construct_varied_route(departure, arrival, aip_data, nominal_patterns, variant_id=0, max_variation=0)
