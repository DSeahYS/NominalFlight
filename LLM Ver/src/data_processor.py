# \src\data_processor.py

import csv
import os
import json
import logging
import pandas as pd
import glob
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_airports(csv_path):
    """
    Load airport data from CSV file with the specific format provided.
    
    Args:
        csv_path: Path to airports.csv file
        
    Returns:
        Dictionary of airports indexed by ICAO code
    """
    airports = {}
    try:
        if not os.path.exists(csv_path):
            logger.error(f"Airport data file not found: {csv_path}")
            return {}
            
        with open(csv_path, 'r', encoding='utf-8') as file:
            total_rows = 0
            valid_rows = 0
            reader = csv.DictReader(file)
            
            for row in reader:
                total_rows += 1
                
                # Use 'ident' as the key for ICAO code
                if 'ident' in row and row['ident']:
                    try:
                        icao_code = row['ident']
                        
                        # Skip non-ICAO codes (usually have less than 4 characters 
                        # or contain hyphens like "AL-LA10")
                        if len(icao_code) != 4 or '-' in icao_code:
                            continue
                            
                        airports[icao_code] = {
                            'name': row.get('name', ''),
                            'latitude': float(row.get('latitude_deg', 0)),
                            'longitude': float(row.get('longitude_deg', 0)),
                            'elevation': float(row.get('elevation_ft', 0)) if row.get('elevation_ft') else 0,
                            'country': row.get('iso_country', ''),
                            'municipality': row.get('municipality', ''),
                            'type': row.get('type', '')
                        }
                        valid_rows += 1
                    except ValueError as e:
                        logger.warning(f"Skipping airport {row.get('ident')}: {e}")
        
        # Manually add WSSS and WMKK if they're not in the dataset
        if 'WSSS' not in airports:
            airports['WSSS'] = {
                'name': 'Singapore Changi Airport',
                'latitude': 1.3591,
                'longitude': 103.9895,
                'elevation': 22,
                'country': 'SG',
                'municipality': 'Singapore',
                'type': 'large_airport'
            }
            
        if 'WMKK' not in airports:
            airports['WMKK'] = {
                'name': 'Kuala Lumpur International Airport',
                'latitude': 2.7456,
                'longitude': 101.7099,
                'elevation': 69,
                'country': 'MY',
                'municipality': 'Kuala Lumpur',
                'type': 'large_airport'
            }
        
        logger.info(f"Successfully loaded {valid_rows} airports from {total_rows} rows in {csv_path}")
        return airports
    except Exception as e:
        logger.error(f"Error loading airports data: {e}")
        return {}

def load_aip_data(directory):
    """
    Load AIP data from JSON files with proper SID/STAR extraction.
    
    Args:
        directory: Directory containing AIP JSON files (SG_AIP.json, MY_AIP.json)
    
    Returns:
        Dictionary of airport AIP data
    """
    aip_data = {}
    try:
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files in AIP directory")
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_data = json.load(file)
                        
                        # Process airport data in the correct structure
                        if 'airports' in file_data:
                            for airport_code, airport_data in file_data['airports'].items():
                                # Extract and process airport information
                                airport_info = {
                                    'airport_code': airport_code,
                                    'name': airport_data.get('name', ''),
                                    'latitude': None,
                                    'longitude': None,
                                    'runways': airport_data.get('runways', []),
                                    'SIDs': airport_data.get('SIDs', []),
                                    'STARs': airport_data.get('STARs', [])
                                }
                                
                                # Look for coordinates in nav aids if not directly provided
                                if 'navigation_aids' in airport_data and airport_data['navigation_aids']:
                                    for navaid in airport_data['navigation_aids']:
                                        if navaid.get('type') in ['VOR/DME', 'DVOR/DME'] and 'lat' in navaid and 'lon' in navaid:
                                            airport_info['latitude'] = navaid['lat']
                                            airport_info['longitude'] = navaid['lon']
                                            break
                                
                                aip_data[airport_code] = airport_info
                        
                        # Count SIDs and STARs for reporting
                        sids_count = 0
                        stars_count = 0
                        for airport_code in aip_data:
                            if 'SIDs' in aip_data[airport_code]:
                                sids_count += len(aip_data[airport_code]['SIDs'])
                            if 'STARs' in aip_data[airport_code]:
                                stars_count += len(aip_data[airport_code]['STARs'])
                        
                        logger.info(f"Loaded {filename}: {len(aip_data)} airports, {sids_count} SIDs, {stars_count} STARs")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in file: {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        logger.info(f"Successfully loaded AIP data for {len(aip_data)} airports")
        return aip_data
    except Exception as e:
        logger.error(f"Error loading AIP data: {e}")
        return {}

def load_cleaned_flight_data(cleaned_dir):
    """
    Load cleaned flight data from CSV files.
    
    Args:
        cleaned_dir: Directory containing cleaned flight data CSV files
        
    Returns:
        Dictionary mapping routes to lists of flight data
    """
    flight_data = {}
    
    if not os.path.exists(cleaned_dir):
        logger.warning(f"Cleaned flight data directory not found: {cleaned_dir}")
        return flight_data
    
    # Find all flights CSV files
    flight_files = glob.glob(os.path.join(cleaned_dir, "flights_*.csv"))
    
    for flight_file in flight_files:
        try:
            # Extract route from filename
            filename = os.path.basename(flight_file)
            route = filename.replace("flights_", "").replace(".csv", "")
            
            # Load flight summaries
            flight_summaries = pd.read_csv(flight_file)
            
            # Load corresponding waypoints
            waypoints_file = os.path.join(cleaned_dir, filename.replace("flights_", "waypoints_"))
            if os.path.exists(waypoints_file):
                waypoints_df = pd.read_csv(waypoints_file)
                
                # Process each flight
                flights = []
                for _, flight_row in flight_summaries.iterrows():
                    flight_id = flight_row['flight_id']
                    
                    # Get waypoints for this flight
                    flight_waypoints = waypoints_df[waypoints_df['flight_id'] == flight_id]
                    
                    # Convert to list of dictionaries
                    waypoints = []
                    for _, wp in flight_waypoints.iterrows():
                        waypoint = {
                            'latitude': wp['latitude'],
                            'longitude': wp['longitude'],
                            'altitude': wp['feet'] if not pd.isna(wp['feet']) else None,  # Rename to altitude for consistency
                            'speed': wp['kts'] if not pd.isna(wp['kts']) else None,       # Rename to speed for consistency
                            'heading': wp.get('course', None),
                            'phase': wp['phase']
                        }
                        waypoints.append(waypoint)
                    
                    # Create flight record
                    flight = {
                        'flight_id': flight_id,
                        'origin': flight_row['origin'],
                        'destination': flight_row['destination'],
                        'waypoints': waypoints,
                        'metadata': {
                            'waypoint_count': len(waypoints),
                            'max_altitude_feet': flight_row['max_altitude'],
                            'max_speed_kts': flight_row['max_speed'],
                            'distance_km': flight_row['distance_km']
                        }
                    }
                    flights.append(flight)
                
                flight_data[route] = flights
                logger.info(f"Loaded {len(flights)} flights for route {route}")
                
            else:
                logger.warning(f"No waypoints file found for {route}")
                
        except Exception as e:
            logger.error(f"Error loading flight data from {flight_file}: {e}")
    
    return flight_data

def load_nominal_patterns(patterns_file):
    """
    Load extracted nominal patterns from JSON file.
    
    Args:
        patterns_file: Path to nominal patterns JSON file
        
    Returns:
        Dictionary of nominal patterns by route
    """
    if not os.path.exists(patterns_file):
        logger.warning(f"Nominal patterns file not found: {patterns_file}")
        return {}
        
    try:
        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
            
        pattern_count = len(patterns)
        logger.info(f"Successfully loaded {pattern_count} nominal patterns from {patterns_file}")
        
        # Log details about each pattern
        for route, pattern in patterns.items():
            if 'waypoints' in pattern:
                logger.info(f"Pattern for {route}: {len(pattern['waypoints'])} waypoints")
            
        return patterns
    except Exception as e:
        logger.error(f"Error loading nominal patterns: {e}")
        return {}

def get_pattern_for_route(patterns, origin, destination):
    """
    Find the nominal pattern for a specific route.
    
    Args:
        patterns: Dictionary of all nominal patterns
        origin: ICAO code of departure airport
        destination: ICAO code of arrival airport
        
    Returns:
        Nominal pattern for the route or None if not found
    """
    route_key = f"{origin}-{destination}"
    
    # Try exact match
    if route_key in patterns:
        return patterns[route_key]
    
    # Try reverse route if bidirectional pattern applies
    reverse_key = f"{destination}-{origin}"
    if reverse_key in patterns:
        # Potentially need to reverse the waypoints if the pattern is directional
        pattern = patterns[reverse_key].copy()
        if 'waypoints' in pattern:
            pattern['waypoints'] = list(reversed(pattern['waypoints']))
        pattern['origin'] = origin
        pattern['destination'] = destination
        return pattern
    
    logger.warning(f"No nominal pattern found for route {route_key}")
    return None

if __name__ == "__main__":
    # For testing and debugging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    
    # Test loading airports
    airports = load_airports(os.path.join(project_root, "data", "airports.csv"))
    print(f"Loaded {len(airports)} airports")
    
    # Test loading AIP data
    aip_data = load_aip_data(os.path.join(project_root, "data", "aip"))
    print(f"Loaded AIP data for {len(aip_data)} airports")
    
    # Test loading cleaned flight data
    flight_data = load_cleaned_flight_data(os.path.join(project_root, "data", "cleaned_historical"))
    print(f"Loaded flight data for {len(flight_data)} routes")
    
    # Test loading nominal patterns
    patterns = load_nominal_patterns(os.path.join(project_root, "data", "nominal", "nominal_patterns.json"))
    print(f"Loaded {len(patterns)} nominal patterns")
