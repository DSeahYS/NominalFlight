# \src\nominal_extractor.py

import numpy as np
import pandas as pd
import json
import os
import logging
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_enroute_segment(flight, min_altitude=10000):
    """
    Extract the en-route segment of a flight (portion above min_altitude).
    
    Args:
        flight: Dictionary containing flight data
        min_altitude: Minimum altitude (feet) to consider as en-route phase
        
    Returns:
        List of waypoints in the en-route phase
    """
    enroute_points = []
    
    # Find points where altitude is above minimum cruise altitude
    for point in flight['waypoints']:
        if point['altitude'] is not None and point['altitude'] >= min_altitude:
            enroute_points.append(point)
    
    return enroute_points

def identify_turning_points(flight, min_heading_change=30):
    """
    Identify significant turning points in a flight.
    
    Args:
        flight: Flight data dictionary
        min_heading_change: Minimum heading change (degrees) to consider as a turn
        
    Returns:
        List of turning points
    """
    turning_points = []
    
    waypoints = flight['waypoints']
    for i in range(1, len(waypoints)-1):
        prev_hdg = waypoints[i-1].get('heading')
        curr_hdg = waypoints[i].get('heading')
        next_hdg = waypoints[i+1].get('heading')
        
        if all([hdg is not None for hdg in [prev_hdg, curr_hdg, next_hdg]]):
            # Calculate heading differences
            hdg_change1 = abs((curr_hdg - prev_hdg + 180) % 360 - 180)
            hdg_change2 = abs((next_hdg - curr_hdg + 180) % 360 - 180)
            
            if hdg_change1 > min_heading_change or hdg_change2 > min_heading_change:
                turning_points.append({
                    'index': i,
                    'waypoint': waypoints[i],
                    'heading_change': max(hdg_change1, hdg_change2)
                })
    
    return turning_points

def identify_all_turning_points(flights, min_heading_change=30):
    """
    Identify significant turning points across all flights.
    
    Args:
        flights: Dictionary of flight data
        min_heading_change: Minimum heading change (degrees) to consider as a turn
        
    Returns:
        List of turning points
    """
    all_turning_points = []
    
    for flight_id, flight in flights.items():
        turning_points = identify_turning_points(flight, min_heading_change)
        
        for tp in turning_points:
            all_turning_points.append({
                'flight_id': flight_id,
                'latitude': tp['waypoint']['latitude'],
                'longitude': tp['waypoint']['longitude'],
                'altitude': tp['waypoint']['altitude'],
                'heading_change': tp['heading_change']
            })
    
    return all_turning_points

def group_by_altitude_bands(flight):
    """
    Group waypoints into altitude bands for better pattern extraction with limited data.
    
    Args:
        flight: Flight data dictionary
        
    Returns:
        Dictionary of waypoints grouped by altitude bands
    """
    bands = {
        'initial_climb': (0, 10000),
        'climb': (10000, 20000),
        'cruise': (20000, 30000),
        'initial_descent': (15000, 20000),
        'final_descent': (0, 15000)
    }
    
    grouped_waypoints = {band: [] for band in bands.keys()}
    
    for wp in flight['waypoints']:
        if wp['altitude'] is None:
            continue
            
        for band, (min_alt, max_alt) in bands.items():
            if min_alt <= wp['altitude'] < max_alt:
                grouped_waypoints[band].append(wp)
                break
    
    return grouped_waypoints

def resample_trajectory(flight, route_length, num_points=30):
    """
    Resample a flight trajectory to standardized distance grid points.
    Optimized for small datasets with only 20 flights.
    
    Args:
        flight: Flight data dictionary
        route_length: Total route length in kilometers
        num_points: Number of points in the resampled trajectory
        
    Returns:
        Resampled flight data dictionary or None if resampling fails
    """
    waypoints = flight['waypoints']
    
    # Calculate cumulative distance along the route
    cumulative_distance = [0]
    for i in range(1, len(waypoints)):
        dist = great_circle(
            (waypoints[i-1]['latitude'], waypoints[i-1]['longitude']),
            (waypoints[i]['latitude'], waypoints[i]['longitude'])
        ).kilometers
        cumulative_distance.append(cumulative_distance[-1] + dist)
    
    # Normalize distances
    total_distance = cumulative_distance[-1]
    if total_distance == 0:
        return None
    
    normalized_distance = [d / total_distance for d in cumulative_distance]
    
    # Create interpolation functions
    lat_data = [wp['latitude'] for wp in waypoints]
    lon_data = [wp['longitude'] for wp in waypoints]
    alt_data = [wp['altitude'] if wp['altitude'] is not None else 0 for wp in waypoints]
    spd_data = [wp['speed'] if wp['speed'] is not None else 0 for wp in waypoints]
    hdg_data = [wp['heading'] if wp['heading'] is not None else 0 for wp in waypoints]
    
    try:
        # Create interpolation functions
        lat_interp = interp1d(normalized_distance, lat_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        lon_interp = interp1d(normalized_distance, lon_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        alt_interp = interp1d(normalized_distance, alt_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        spd_interp = interp1d(normalized_distance, spd_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        hdg_interp = interp1d(normalized_distance, hdg_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Create distance grid
        distance_grid = np.linspace(0, route_length, num_points)
        grid_normalized = [min(d / route_length, 1.0) for d in distance_grid]
        
        # Sample at grid points
        resampled_waypoints = []
        for i, dist in enumerate(grid_normalized):
            resampled_wp = {
                'distance_km': float(distance_grid[i]),
                'latitude': float(lat_interp(dist)),
                'longitude': float(lon_interp(dist)),
                'altitude': float(alt_interp(dist)),
                'speed': float(spd_interp(dist)),
                'heading': float(hdg_interp(dist)),
                'norm_distance': float(dist)
            }
            resampled_waypoints.append(resampled_wp)
        
        # Copy flight data but replace waypoints
        resampled_flight = flight.copy()
        resampled_flight['waypoints'] = resampled_waypoints
        resampled_flight['resampled'] = True
        return resampled_flight
    
    except Exception as e:
        logger.error(f"Error resampling trajectory: {e}")
        return None

def identify_waypoint_clusters(flights, eps=10.0, min_samples=2):
    """
    Identify common waypoints across multiple flights using clustering.
    Optimized for small datasets with only 20 flights.
    
    Args:
        flights: Dictionary of flight data
        eps: Maximum distance (km) between points in the same cluster
        min_samples: Minimum number of points to form a cluster
        
    Returns:
        List of waypoint clusters
    """
    # Extract all resampled waypoints
    distance_waypoints = {}
    
    for flight_id, flight in flights.items():
        for wp in flight['waypoints']:
            if 'distance_km' not in wp:
                continue
                
            dist_key = round(wp['distance_km'])
            if dist_key not in distance_waypoints:
                distance_waypoints[dist_key] = []
            distance_waypoints[dist_key].append(wp)
    
    # Apply clustering at each distance
    waypoint_clusters = []
    
    for dist, waypoints in sorted(distance_waypoints.items()):
        if len(waypoints) < min_samples:
            continue
        
        # Extract coordinates for clustering
        coords = np.array([[wp['latitude'], wp['longitude']] for wp in waypoints])
        
        # Apply DBSCAN
        db = DBSCAN(eps=eps/111.0, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        
        # Find largest cluster
        labels = db.labels_
        unique_labels = set(labels)
        
        largest_cluster = -1
        largest_size = 0
        
        for label in unique_labels:
            if label == -1:
                continue
            
            cluster_size = np.sum(labels == label)
            if cluster_size > largest_size:
                largest_size = cluster_size
                largest_cluster = label
        
        if largest_cluster == -1:
            continue
        
        # Get waypoints in largest cluster
        cluster_waypoints = [waypoints[i] for i, label in enumerate(labels) if label == largest_cluster]
        
        # Calculate averages
        avg_lat = np.mean([wp['latitude'] for wp in cluster_waypoints])
        avg_lon = np.mean([wp['longitude'] for wp in cluster_waypoints])
        avg_alt = np.mean([wp['altitude'] for wp in cluster_waypoints])
        avg_spd = np.mean([wp['speed'] for wp in cluster_waypoints])
        avg_hdg = np.mean([wp['heading'] for wp in cluster_waypoints])
        
        waypoint_clusters.append({
            'name': f"NOM{dist:03d}",
            'latitude': float(avg_lat),
            'longitude': float(avg_lon),
            'altitude': float(avg_alt),
            'speed': float(avg_spd),
            'heading': float(avg_hdg),
            'cluster_size': int(largest_size),
            'distance_km': dist
        })
    
    # Sort by distance
    waypoint_clusters.sort(key=lambda x: x['distance_km'])
    
    return waypoint_clusters

def get_airport_coords(airport_code, airports, flights):
    """
    Get airport coordinates from various sources.
    
    Args:
        airport_code: ICAO code of the airport
        airports: Dictionary of airport data
        flights: Dictionary of flight data
        
    Returns:
        Tuple of (latitude, longitude)
    """
    # Try to get from airports database
    if airport_code in airports:
        lat = airports[airport_code]['latitude']
        lon = airports[airport_code]['longitude']
        return (lat, lon)
    
    # Try to get from flights (first/last waypoint)
    for flight in flights.values():
        dep = flight.get('departure', '')
        arr = flight.get('arrival', '')
        
        if airport_code in dep and flight['waypoints']:
            wp = flight['waypoints'][0]
            return (wp['latitude'], wp['longitude'])
        
        if airport_code in arr and flight['waypoints']:
            wp = flight['waypoints'][-1]
            return (wp['latitude'], wp['longitude'])
    
    # Return defaults if not found
    logger.warning(f"Could not find coordinates for {airport_code}, using defaults")
    if airport_code == 'WSSS':
        return (1.3591, 103.9895)  # Singapore Changi
    elif airport_code == 'WMKK':
        return (2.7456, 101.7099)  # Kuala Lumpur International
    else:
        return (0, 0)  # Default

def visualize_nominal_pattern(pattern, source_flights, save_path=None):
    """
    Visualize the extracted nominal pattern alongside the source flights.
    
    Args:
        pattern: Nominal pattern dictionary
        source_flights: Dictionary of source flight data
        save_path: Path to save the visualization image
    """
    plt.figure(figsize=(12, 8))
    
    # Plot source flights in background
    for flight_id, flight in source_flights.items():
        lats = [wp['latitude'] for wp in flight['waypoints']]
        lons = [wp['longitude'] for wp in flight['waypoints']]
        plt.plot(lons, lats, 'b-', alpha=0.15, linewidth=0.8)
    
    # Plot the nominal pattern
    if pattern and 'waypoints' in pattern:
        lats = [wp['latitude'] for wp in pattern['waypoints']]
        lons = [wp['longitude'] for wp in pattern['waypoints']]
        plt.plot(lons, lats, 'r-', linewidth=2.5)
        
        # Plot waypoints
        plt.scatter(lons, lats, c='red', s=40, zorder=5)
    
    plt.title(f"Nominal Pattern: {pattern['airport_pair']} (from {len(source_flights)} flights)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved pattern visualization to {save_path}")
    
    plt.close()

def save_nominal_patterns(nominal_routes, output_file):
    """
    Save extracted nominal patterns to a JSON file.
    
    Args:
        nominal_routes: Dictionary of nominal routes
        output_file: Path to output JSON file
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nominal_routes, f, indent=2)
        logger.info(f"Saved nominal patterns to {output_file}")
    except Exception as e:
        logger.error(f"Error saving nominal patterns: {e}")

# NEW VISUALIZATION FUNCTIONS

def visualize_altitude_profiles(pattern, source_flights, save_path=None):
    """Create altitude profile visualization comparing all flights to the nominal pattern."""
    plt.figure(figsize=(14, 8))
    
    # Plot each flight's altitude profile in the background
    for flight_id, flight in source_flights.items():
        distances = [wp['distance_km'] for wp in flight['waypoints']]
        altitudes = [wp['altitude'] for wp in flight['waypoints']]
        plt.plot(distances, altitudes, 'b-', alpha=0.2, linewidth=0.8)
    
    # Plot the nominal pattern's altitude profile
    if pattern and 'waypoints' in pattern:
        distances = [wp['distance_km'] for wp in pattern['waypoints']]
        altitudes = [wp['altitude'] for wp in pattern['waypoints']]
        plt.plot(distances, altitudes, 'r-', linewidth=3)
    
    plt.title(f"Altitude Profiles: {pattern['airport_pair']} (from {len(source_flights)} flights)")
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (feet)")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved altitude profile visualization to {save_path}")
    
    plt.close()

def visualize_waypoint_density(source_flights, pattern, save_path=None):
    """Create a heatmap showing the density of flight points."""
    plt.figure(figsize=(14, 10))
    
    # Collect all waypoint coordinates
    all_lats = []
    all_lons = []
    
    for flight in source_flights.values():
        for wp in flight['waypoints']:
            all_lats.append(wp['latitude'])
            all_lons.append(wp['longitude'])
    
    # Create hexbin heatmap
    plt.hexbin(all_lons, all_lats, gridsize=50, cmap='viridis', alpha=0.7)
    
    # Plot the nominal pattern on top
    if pattern and 'waypoints' in pattern:
        lats = [wp['latitude'] for wp in pattern['waypoints']]
        lons = [wp['longitude'] for wp in pattern['waypoints']]
        plt.plot(lons, lats, 'r-', linewidth=2.5)
        plt.scatter(lons, lats, c='red', s=40, zorder=5, edgecolor='white')
    
    plt.title(f"Waypoint Density: {pattern['airport_pair']} (from {len(source_flights)} flights)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Point density")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved waypoint density visualization to {save_path}")
    
    plt.close()

def visualize_turning_points(source_flights, save_path=None):
    """Visualize the distribution of significant turning points across flights."""
    turning_points = identify_all_turning_points(source_flights, min_heading_change=20)
    
    if not turning_points:
        logger.warning("No significant turning points found")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Plot a light trace of each flight route
    for flight in source_flights.values():
        lats = [wp['latitude'] for wp in flight['waypoints']]
        lons = [wp['longitude'] for wp in flight['waypoints']]
        plt.plot(lons, lats, 'grey', alpha=0.15, linewidth=0.5)
    
    # Plot turning points, color by heading change magnitude
    lats = [tp['latitude'] for tp in turning_points]
    lons = [tp['longitude'] for tp in turning_points]
    hdg_changes = [tp['heading_change'] for tp in turning_points]
    
    sc = plt.scatter(lons, lats, c=hdg_changes, cmap='plasma', 
                    s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    plt.colorbar(sc, label="Heading Change (degrees)")
    
    plt.title(f"Significant Turning Points (from {len(source_flights)} flights)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved turning points visualization to {save_path}")
    
    plt.close()

def visualize_flight_phases(source_flights, pattern, save_path=None):
    """Visualize flight phases (climb, cruise, descent) along the route."""
    # Prepare figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    
    ax1 = plt.subplot(gs[0])  # Main plot with altitude profile
    ax2 = plt.subplot(gs[1])  # Phase distribution bar
    
    # Define altitude thresholds for phases
    climb_threshold = 18000    # Below this is climb
    descent_start = 22000      # Below this is start of descent
    
    # Process data for visualization
    phase_data = []
    
    for flight_id, flight in source_flights.items():
        phases = []
        
        for wp in flight['waypoints']:
            if wp['altitude'] < climb_threshold:
                if phases and phases[-1] == 'cruise':
                    phases.append('descent')
                elif not phases:
                    phases.append('climb')
                else:
                    phases.append(phases[-1])
            elif wp['altitude'] >= climb_threshold and wp['altitude'] < descent_start:
                if phases and phases[-1] == 'descent':
                    phases.append('descent')
                else:
                    phases.append('cruise')
            else:
                phases.append('cruise')
        
        # Plot altitude profile colored by phase
        distances = [wp['distance_km'] for wp in flight['waypoints']]
        altitudes = [wp['altitude'] for wp in flight['waypoints']]
        
        # Plot with phase-based coloring
        for i in range(1, len(distances)):
            if phases[i-1] == 'climb':
                ax1.plot(distances[i-1:i+1], altitudes[i-1:i+1], 'g-', alpha=0.2, linewidth=0.8)
            elif phases[i-1] == 'cruise':
                ax1.plot(distances[i-1:i+1], altitudes[i-1:i+1], 'b-', alpha=0.2, linewidth=0.8)
            else:  # descent
                ax1.plot(distances[i-1:i+1], altitudes[i-1:i+1], 'r-', alpha=0.2, linewidth=0.8)
        
        phase_data.append(phases)
    
    # Plot the nominal pattern's altitude profile
    if pattern and 'waypoints' in pattern:
        distances = [wp['distance_km'] for wp in pattern['waypoints']]
        altitudes = [wp['altitude'] for wp in pattern['waypoints']]
        ax1.plot(distances, altitudes, 'k-', linewidth=3, label='Nominal Pattern')
    
    # Create phase distribution visualization
    if phase_data:
        # Calculate phase probabilities at each distance point
        max_len = max(len(phases) for phases in phase_data)
        
        # Use the same distance grid as nominal pattern
        if pattern and 'waypoints' in pattern:
            distance_grid = [wp['distance_km'] for wp in pattern['waypoints']]
        else:
            # Create a uniform grid
            max_dist = max(wp['distance_km'] for flight in source_flights.values() 
                          for wp in flight['waypoints'])
            distance_grid = np.linspace(0, max_dist, 30)
        
        # For each distance point, calculate phase probabilities
        climb_probs = []
        cruise_probs = []
        descent_probs = []
        
        for i in range(len(distance_grid)):
            idx_norm = i / (len(distance_grid) - 1)  # Normalized index (0-1)
            
            # Count phases at this normalized position
            climb_count = 0
            cruise_count = 0
            descent_count = 0
            total = 0
            
            for phases in phase_data:
                if not phases:
                    continue
                
                # Map the normalized position to an index in this flight's phases
                idx = min(int(idx_norm * (len(phases) - 1)), len(phases) - 1)
                
                if idx < len(phases):
                    total += 1
                    if phases[idx] == 'climb':
                        climb_count += 1
                    elif phases[idx] == 'cruise':
                        cruise_count += 1
                    else:  # descent
                        descent_count += 1
            
            if total > 0:
                climb_probs.append(climb_count / total)
                cruise_probs.append(cruise_count / total)
                descent_probs.append(descent_count / total)
            else:
                climb_probs.append(0)
                cruise_probs.append(0)
                descent_probs.append(0)
        
        # Plot stacked probabilities
        ax2.stackplot(distance_grid, 
                     [climb_probs, cruise_probs, descent_probs],
                     labels=['Climb', 'Cruise', 'Descent'],
                     colors=['green', 'blue', 'red'],
                     alpha=0.7)
        
        ax2.set_xlim(0, max(distance_grid))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Phase Probability')
        ax2.legend(loc='upper center', ncol=3)
        ax2.grid(True)
    
    # Add legend and labels
    ax1.set_title(f"Flight Phases: {pattern['airport_pair']} (from {len(source_flights)} flights)")
    ax1.set_ylabel('Altitude (feet)')
    ax1.grid(True)
    
    # Create a fake legend for the phases
    ax1.plot([], [], 'g-', label='Climb')
    ax1.plot([], [], 'b-', label='Cruise')
    ax1.plot([], [], 'r-', label='Descent')
    ax1.plot([], [], 'k-', linewidth=2, label='Nominal')
    ax1.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved flight phases visualization to {save_path}")
    
    plt.close()

def visualize_route_variance(source_flights, pattern, save_path=None):
    """Visualize the variance in position, altitude and speed along the route."""
    # Ensure we have resampled flights at consistent distances
    if not all('distance_km' in flight['waypoints'][0] for flight in source_flights.values()):
        logger.warning("Route variance visualization requires resampled flights")
        return
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Get common distance grid
    if pattern and 'waypoints' in pattern:
        distance_grid = [wp['distance_km'] for wp in pattern['waypoints']]
    else:
        # Create a uniform grid
        all_distances = [wp['distance_km'] for flight in source_flights.values() 
                       for wp in flight['waypoints']]
        max_dist = max(all_distances)
        min_dist = min(all_distances)
        distance_grid = np.linspace(min_dist, max_dist, 30)
    
    # Collect data for each distance point
    lateral_variances = []  # in km
    altitude_variances = [] # in feet
    speed_variances = []    # in knots
    
    nominal_lats = []
    nominal_lons = []
    nominal_alts = []
    nominal_speeds = []
    
    if pattern and 'waypoints' in pattern:
        nominal_lats = [wp['latitude'] for wp in pattern['waypoints']]
        nominal_lons = [wp['longitude'] for wp in pattern['waypoints']]
        nominal_alts = [wp['altitude'] for wp in pattern['waypoints']]
        nominal_speeds = [wp['speed'] for wp in pattern['waypoints']]
    
    for d_idx, distance in enumerate(distance_grid):
        # Find closest waypoint in each flight
        lats = []
        lons = []
        alts = []
        speeds = []
        
        for flight in source_flights.values():
            for i, wp in enumerate(flight['waypoints']):
                if abs(wp['distance_km'] - distance) < 5:  # Within 5km
                    lats.append(wp['latitude'])
                    lons.append(wp['longitude'])
                    alts.append(wp['altitude'])
                    speeds.append(wp['speed'])
                    break
        
        # Calculate lateral variance (avg distance from mean point)
        if lats and lons:
            mean_lat = np.mean(lats)
            mean_lon = np.mean(lons)
            
            lateral_dists = []
            for i in range(len(lats)):
                dist = great_circle((lats[i], lons[i]), (mean_lat, mean_lon)).kilometers
                lateral_dists.append(dist)
            
            lateral_variances.append(np.mean(lateral_dists))
        else:
            lateral_variances.append(0)
        
        # Calculate altitude and speed variances
        altitude_variances.append(np.std(alts) if alts else 0)
        speed_variances.append(np.std(speeds) if speeds else 0)
    
    # Plot the variance data
    ax1.plot(distance_grid, lateral_variances, 'b-', linewidth=2)
    ax1.fill_between(distance_grid, 0, lateral_variances, alpha=0.3)
    ax1.set_ylabel('Lateral Variance (km)')
    ax1.set_title('Lateral Deviation from Nominal Route')
    ax1.grid(True)
    
    ax2.plot(distance_grid, altitude_variances, 'g-', linewidth=2)
    ax2.fill_between(distance_grid, 0, altitude_variances, alpha=0.3)
    ax2.set_ylabel('Altitude Variance (feet)')
    ax2.set_title('Altitude Deviation')
    ax2.grid(True)
    
    ax3.plot(distance_grid, speed_variances, 'r-', linewidth=2)
    ax3.fill_between(distance_grid, 0, speed_variances, alpha=0.3)
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Speed Variance (knots)')
    ax3.set_title('Speed Deviation')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved route variance visualization to {save_path}")
    
    plt.close()

def generate_advanced_visualizations(route_key, pattern, source_flights, output_dir):
    """Generate all advanced visualizations for a route."""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate each visualization type
    visualize_altitude_profiles(
        pattern, 
        source_flights, 
        save_path=os.path.join(viz_dir, f"altitude_profile_{pattern['airport_pair']}.png")
    )
    
    visualize_waypoint_density(
        source_flights, 
        pattern, 
        save_path=os.path.join(viz_dir, f"waypoint_density_{pattern['airport_pair']}.png")
    )
    
    visualize_turning_points(
        source_flights, 
        save_path=os.path.join(viz_dir, f"turning_points_{pattern['airport_pair']}.png")
    )
    
    visualize_flight_phases(
        source_flights, 
        pattern, 
        save_path=os.path.join(viz_dir, f"flight_phases_{pattern['airport_pair']}.png")
    )
    
    visualize_route_variance(
        source_flights, 
        pattern, 
        save_path=os.path.join(viz_dir, f"route_variance_{pattern['airport_pair']}.png")
    )
    
    logger.info(f"Generated advanced visualizations for {pattern['airport_pair']}")
    return viz_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract nominal flight patterns from cleaned historical data")
    parser.add_argument("--input", help="Directory containing cleaned historical flight data")
    parser.add_argument("--output", help="Directory to save nominal patterns")
    parser.add_argument("--visualize", action="store_true", help="Generate basic visualizations of patterns")
    parser.add_argument("--advanced-viz", action="store_true", help="Generate advanced visualizations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up proper directories based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    
    # Set input/output directories
    input_dir = args.input if args.input else os.path.join(project_root, "data", "cleaned_historical")
    output_dir = args.output if args.output else os.path.join(project_root, "data", "nominal")
    
    logger.info(f"Input directory: {os.path.abspath(input_dir)}")
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cleaned flight data
    flights_by_route = {}
    route_files = [f for f in os.listdir(input_dir) if f.startswith('flights_')]
    waypoint_files = [f for f in os.listdir(input_dir) if f.startswith('waypoints_')]
    
    for route_file in route_files:
        # Extract route key from filename (e.g., AXM706_WSSS-WMKK)
        route_key = route_file.replace('flights_', '').replace('.csv', '')
        
        # Load flights summary
        flights_df = pd.read_csv(os.path.join(input_dir, route_file))
        
        # Find corresponding waypoints file
        wp_file = f"waypoints_{route_key}.csv"
        if wp_file not in waypoint_files:
            logger.warning(f"No waypoints file found for {route_key}")
            continue
            
        # Load waypoints
        waypoints_df = pd.read_csv(os.path.join(input_dir, wp_file))
        
        # Group waypoints by flight_id
        flights = []
        for flight_id in flights_df['flight_id'].unique():
            flight_summary = flights_df[flights_df['flight_id'] == flight_id].iloc[0].to_dict()
            flight_waypoints = waypoints_df[waypoints_df['flight_id'] == flight_id]
            
            # Convert waypoints to dictionary format needed by the functions
            waypoints = []
            for _, wp in flight_waypoints.iterrows():
                waypoint = {
                    'latitude': wp['latitude'],
                    'longitude': wp['longitude'],
                    'altitude': wp['feet'],  # 'feet' in cleaned data
                    'speed': wp['kts'],      # 'kts' in cleaned data
                    'heading': wp.get('course', None),  # Use course if available
                    'phase': wp['phase'],
                    'distance_km': wp['distance_km'],
                    'cum_distance': wp['cum_distance']
                }
                waypoints.append(waypoint)
            
            # Create standardized flight object
            flight = {
                'flight_id': flight_id,
                'origin': flight_summary['origin'],
                'destination': flight_summary['destination'],
                'departure_time': flight_summary['departure_time'],
                'arrival_time': flight_summary['arrival_time'],
                'waypoints': waypoints
            }
            flights.append(flight)
        
        # Store flights for this route
        flights_by_route[route_key] = flights
        logger.info(f"Loaded {len(flights)} flights for route {route_key}")
    
    # Calculate route lengths
    route_lengths = {}
    for route_key, flights in flights_by_route.items():
        # Get average total distance
        distances = []
        for flight in flights:
            if flight['waypoints']:
                last_wp = flight['waypoints'][-1]
                if 'cum_distance' in last_wp and last_wp['cum_distance'] > 0:
                    distances.append(last_wp['cum_distance'])
        
        if distances:
            route_lengths[route_key] = sum(distances) / len(distances)
        else:
            # Default if no distance data
            route_lengths[route_key] = 300  # Default 300km
    
    # Resample all flights
    resampled_flights = {}
    for route_key, flights in flights_by_route.items():
        resampled_route_flights = {}
        route_length = route_lengths.get(route_key, 300)
        
        for flight in flights:
            flight_id = flight['flight_id']
            resampled_flight = resample_trajectory(flight, route_length, num_points=30)
            if resampled_flight:
                resampled_route_flights[flight_id] = resampled_flight
        
        if resampled_route_flights:
            resampled_flights[route_key] = resampled_route_flights
            logger.info(f"Resampled {len(resampled_route_flights)} flights for {route_key}")
    
    # Extract patterns for each route
    nominal_patterns = {}
    for route_key, route_flights in resampled_flights.items():
        airport_pair = route_key.split('_')[1]  # e.g., "WSSS-WMKK"
        origin, destination = airport_pair.split('-')
        
        # Identify common waypoints
        waypoint_clusters = identify_waypoint_clusters(route_flights, eps=5.0, min_samples=2)
        
        if waypoint_clusters:
            # Create nominal pattern
            pattern = {
                'airport_pair': airport_pair,
                'origin': origin,
                'destination': destination,
                'waypoints': waypoint_clusters,
                'metadata': {
                    'source_flight_count': len(route_flights),
                    'total_distance_km': route_lengths.get(route_key, 0)
                }
            }
            
            nominal_patterns[airport_pair] = pattern
            
            # Visualize the pattern if requested
            if args.visualize:
                visualize_nominal_pattern(
                    pattern, 
                    route_flights, 
                    save_path=os.path.join(output_dir, f"pattern_{airport_pair}.png")
                )
            
            # Generate advanced visualizations if requested
            if args.advanced_viz:
                generate_advanced_visualizations(route_key, pattern, route_flights, output_dir)
            
            logger.info(f"Created nominal pattern for {airport_pair} with {len(waypoint_clusters)} waypoints")
    
    # Save the patterns
    if nominal_patterns:
        save_nominal_patterns(nominal_patterns, os.path.join(output_dir, "nominal_patterns.json"))
        logger.info(f"Saved {len(nominal_patterns)} nominal patterns")
    else:
        logger.warning("No nominal patterns were extracted")
    
    # Ask if the user wants to generate advanced visualizations
    if not args.advanced_viz:
        response = input("\nWould you like to generate advanced flight visualizations? (y/n): ").strip().lower()
        if response == 'y' or response == 'yes':
            logger.info("Generating advanced visualizations...")
            for route_key, route_flights in resampled_flights.items():
                airport_pair = route_key.split('_')[1]  # e.g., "WSSS-WMKK"
                if airport_pair in nominal_patterns:
                    pattern = nominal_patterns[airport_pair]
                    generate_advanced_visualizations(route_key, pattern, route_flights, output_dir)
            logger.info("Advanced visualizations complete")

if __name__ == "__main__":
    main()
    
