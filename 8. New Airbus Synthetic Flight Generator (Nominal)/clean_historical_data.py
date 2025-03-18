import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import re
import csv  # Use CSV module for robust parsing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_numeric_field(value_str):
    """
    Parse a numeric field that may be formatted in one of two ways:
    - If it is three digits or less (e.g., 350), no quotes or commas.
    - If it is four digits or more (e.g., "1,200"), it will have quotes and a comma.
    This function removes any extraneous quotes, spaces, and commas, then converts to float.
    """
    if not value_str:
        return None
    # Remove any quotes and spaces
    cleaned = value_str.strip().replace('"', '').replace(' ', '')
    # Remove commas if present
    cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        logger.debug(f"Unable to convert numeric value from: '{value_str}' cleaned as '{cleaned}'")
        return None

def find_flight_files(directory):
    """Find all historical flight data files in a directory."""
    flight_files = []

    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return flight_files

    logger.info(f"Searching for flight data files in: {os.path.abspath(directory)}")
    all_files = os.listdir(directory)
    logger.info(f"Found {len(all_files)} total files in directory")

    for file in all_files:
        if file.startswith("Historical_") and (file.endswith(".csv") or file.endswith(".txt")):
            file_path = os.path.join(directory, file)
            flight_files.append(file_path)
            logger.info(f"Found flight data file: {file}")

    return flight_files

def parse_flight_file(file_path):
    """Parse a flight data file into separate flight records using the CSV module."""
    logger.info(f"Parsing flight file: {os.path.basename(file_path)}")
    try:
        # Use CSV reader to properly handle quoted fields with commas
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            lines = list(csv_reader)

        if not lines:
            logger.warning(f"Empty file: {file_path}")
            return {}

        # Extract flight code and route from filename
        filename = os.path.basename(file_path)
        match = re.search(r'Historical_([A-Z0-9]+)_([A-Z]+)-([A-Z]+)', filename)
        if match:
            flight_code, origin, destination = match.groups()
        else:
            flight_code, origin, destination = "UNKNOWN", "UNKNOWN", "UNKNOWN"

        # Initialize flights container
        flights = {}
        current_flight_data = []
        current_flight_id = None
        current_index = 0

        # Process each line (now properly parsed as CSV)
        for i, parts in enumerate(lines):
            # Skip header or empty lines
            if i == 0 or not parts:
                continue

            # Reconstruct original line for pattern matching
            line = ','.join(parts)

            # Check if this is the start of a new flight (gate departure)
            if parts and "Left Gate" in parts[0]:
                if current_flight_data and current_flight_id:
                    if current_flight_id not in flights:
                        flights[current_flight_id] = {
                            'code': flight_code,
                            'origin': origin,
                            'destination': destination,
                            'data': pd.DataFrame(current_flight_data)
                        }
                date_match = re.search(r'@ ([A-Za-z]+) (\d+:\d+:\d+ [AP]M)', line)
                if date_match:
                    day_name, time_str = date_match.groups()
                    flight_date = datetime.now().strftime("%Y%m%d")
                    current_flight_id = f"{flight_code}_{flight_date}_{current_index}"
                    current_index += 1
                    current_flight_data = []
                continue

            # Look for departure
            if parts and "Departure" in parts[0] and current_flight_id:
                date_match = re.search(r'@ ([A-Za-z]+) (\d+:\d+:\d+ [AP]M)', line)
                if date_match:
                    day_name, time_str = date_match.groups()
                    if current_flight_id in flights:
                        flights[current_flight_id]['departure_time'] = f"{day_name} {time_str}"
                    logger.info(f"Found flight departure: {current_flight_id}")
                continue

            # Look for arrival
            if parts and "Arrival" in parts[0] and current_flight_id:
                date_match = re.search(r'@ ([A-Za-z]+) (\d+:\d+:\d+ [AP]M)', line)
                if date_match:
                    day_name, time_str = date_match.groups()
                    if current_flight_id in flights:
                        flights[current_flight_id]['arrival_time'] = f"{day_name} {time_str}"
                    logger.info(f"Found flight arrival: {current_flight_id}")
                continue

            # Skip taxi time lines
            if parts and "Taxi Time" in parts[0]:
                continue

            # Parse data rows (actual flight data)
            if len(parts) >= 8 and current_flight_id is not None:
                try:
                    # Extract timestamp
                    timestamp = parts[0].strip()
                    if not timestamp or timestamp == "Time (+08)" or "Gate Arrival" in timestamp:
                        continue

                    # Extract coordinates
                    lat = parts[1].strip() if len(parts) > 1 else ""
                    lon = parts[2].strip() if len(parts) > 2 else ""
                    if not lat or not lon:
                        continue

                    # Process course value
                    course = parts[3].strip() if len(parts) > 3 else ""
                    if course and '°' in course:
                        course = course.split('°')[0]
                        for arrow in ['↑', '↖', '←', '↙', '↓', '↘', '→', '↗']:
                            course = course.replace(arrow, '')

                    # --- Process Altitude (feet) using helper ---
                    feet_raw = parts[6].strip() if len(parts) > 6 else ""
                    if i < 5:
                        logger.debug(f"Row {i}: Raw altitude field: '{feet_raw}'")
                    feet_val = parse_numeric_field(feet_raw)
                    if i < 5:
                        logger.debug(f"Row {i}: Processed altitude value: {feet_val}")
                    # ---------------------------------------------

                    # Process other numeric fields using helper
                    kts_raw = parts[4].strip() if len(parts) > 4 else ""
                    mph_raw = parts[5].strip() if len(parts) > 5 else ""
                    rate_raw = parts[7].strip() if len(parts) > 7 else ""

                    kts_val = parse_numeric_field(kts_raw)
                    mph_val = parse_numeric_field(mph_raw)
                    rate_val = parse_numeric_field(rate_raw)

                    facility = parts[8].strip() if len(parts) > 8 else ""

                    current_flight_data.append({
                        'timestamp': timestamp,
                        'latitude': float(lat) if lat else None,
                        'longitude': float(lon) if lon else None,
                        'course': float(course) if course and course.replace('-', '', 1).isdigit() else None,
                        'kts': kts_val,
                        'mph': mph_val,
                        'feet': feet_val,
                        'rate': rate_val,
                        'facility': facility
                    })
                except Exception as e:
                    logger.warning(f"Error parsing line {i+1}: {e} - Line content: '{','.join(parts)}'")

        # Add the last flight if any
        if current_flight_data and current_flight_id and current_flight_id not in flights:
            flights[current_flight_id] = {
                'code': flight_code,
                'origin': origin,
                'destination': destination,
                'data': pd.DataFrame(current_flight_data)
            }

        # Log the processed flights
        for flight_id, flight in flights.items():
            if 'data' in flight and not flight['data'].empty:
                max_altitude = (flight['data']['feet'].max()
                                if 'feet' in flight['data'].columns and not pd.isna(flight['data']['feet'].max())
                                else 0)
                logger.info(f"Processed flight {flight_id} with {len(flight['data'])} data points, max altitude: {max_altitude} feet")
            else:
                logger.warning(f"No valid data points for flight {flight_id}")

        return flights

    except Exception as e:
        logger.error(f"Error parsing flight file {file_path}: {e}")
        return {}

def clean_flight_data(flight):
    """Clean and process flight data, identifying phases and calculating metrics."""
    if 'data' not in flight or flight['data'].empty:
        logger.warning(f"No data to clean for flight {flight.get('code', 'UNKNOWN')}")
        return None

    try:
        df = flight['data'].copy()
        df = df.dropna(how='all')

        if df['latitude'].isna().all() or df['longitude'].isna().all():
            logger.warning("No valid coordinates found for flight")
            return None

        if len(df) < 2:
            logger.warning("Not enough valid data points to process flight")
            return None

        df['time_str'] = df['timestamp'].str.extract(r'(\d+:\d+:\d+ [AP]M)')
        df['lat_prev'] = df['latitude'].shift(1)
        df['lon_prev'] = df['longitude'].shift(1)

        def calc_distance(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']) or pd.isna(row['lat_prev']) or pd.isna(row['lon_prev']):
                return np.nan
            lat_diff = row['latitude'] - row['lat_prev']
            lon_diff = row['longitude'] - row['lon_prev']
            return np.sqrt((lat_diff * 111.32)**2 + (lon_diff * 111.32 * np.cos(np.radians(row['latitude'])))**2)

        df['distance_km'] = df.apply(calc_distance, axis=1).fillna(0)
        df['cum_distance'] = df['distance_km'].cumsum()

        def identify_phase(row):
            if pd.isna(row['rate']):
                return 'unknown'
            if row['rate'] > 300:
                return 'climb'
            elif row['rate'] < -300:
                return 'descent'
            else:
                return 'cruise'

        df['phase'] = df.apply(identify_phase, axis=1)

        logger.debug("Sample cleaned flight data (first 5 rows):")
        logger.debug(df.head())

        waypoints = []
        for _, row in df.iterrows():
            waypoint = {}
            for column in df.columns:
                if pd.notnull(row[column]):
                    if isinstance(row[column], pd.Timestamp):
                        waypoint[column] = row[column].strftime('%H:%M:%S %p')
                    else:
                        waypoint[column] = row[column]
            waypoints.append(waypoint)

        cleaned_flight = {
            'flight_id': flight.get('code', 'UNKNOWN'),
            'origin': flight.get('origin', 'UNKNOWN'),
            'destination': flight.get('destination', 'UNKNOWN'),
            'departure_time': flight.get('departure_time', None),
            'arrival_time': flight.get('arrival_time', None),
            'waypoints': waypoints,
            'metadata': {
                'waypoint_count': len(df),
                'max_altitude_feet': float(df['feet'].max()) if 'feet' in df.columns and not df['feet'].empty and not pd.isna(df['feet'].max()) else None,
                'max_speed_kts': float(df['kts'].max()) if 'kts' in df.columns and not df['kts'].empty and not pd.isna(df['kts'].max()) else None,
                'distance_km': float(df['cum_distance'].iloc[-1]) if not df.empty else 0
            }
        }

        logger.info(f"Cleaned flight with {len(df)} waypoints, max alt: {df['feet'].max() if 'feet' in df.columns and not df['feet'].empty and not pd.isna(df['feet'].max()) else 0:.0f} ft")
        return cleaned_flight

    except Exception as e:
        logger.error(f"Error cleaning flight data: {e}")
        return None

def visualize_flight_phases(flight_data, output_file):
    """Create visualization of flight phases with altitude and speed profiles."""
    try:
        waypoints = flight_data['waypoints']
        df = pd.DataFrame(waypoints)

        if len(df) < 2:
            logger.warning("Not enough data points to create visualization")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        colors = {'climb': 'green', 'cruise': 'blue', 'descent': 'red', 'unknown': 'gray'}
        x_data = df['cum_distance'] if 'cum_distance' in df.columns else range(len(df))
        x_label = 'Distance (km)' if 'cum_distance' in df.columns else 'Waypoint Index'

        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase]
            phase_x = phase_data['cum_distance'] if 'cum_distance' in df.columns else [x_data[i] for i in phase_data.index]
            ax1.scatter(phase_x, phase_data['feet'], color=colors.get(phase, 'gray'), label=phase, alpha=0.7)
        ax1.plot(x_data, df['feet'], 'k-', alpha=0.3)
        ax1.set_ylabel('Altitude (feet)')
        ax1.set_title(f"Flight {flight_data['flight_id']}: {flight_data['origin']} to {flight_data['destination']}")
        ax1.legend()
        ax1.grid(True)

        if 'kts' in df.columns:
            for phase in df['phase'].unique():
                phase_data = df[df['phase'] == phase]
                phase_x = phase_data['cum_distance'] if 'cum_distance' in df.columns else [x_data[i] for i in phase_data.index]
                ax2.scatter(phase_x, phase_data['kts'], color=colors.get(phase, 'gray'), label=phase, alpha=0.7)
            ax2.plot(x_data, df['kts'], 'k-', alpha=0.3)
            ax2.set_xlabel(x_label)
            ax2.set_ylabel('Speed (knots)')
            ax2.grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Created visualization: {output_file}")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

def clean_historical_data(input_dir, output_dir, output_format='csv', visualize=False):
    """Process flight data files and output cleaned versions."""
    os.makedirs(output_dir, exist_ok=True)
    flight_files = find_flight_files(input_dir)
    logger.info(f"Found {len(flight_files)} flight data files")

    if not flight_files:
        logger.warning(f"No flight data files found in {input_dir}")
        return 0

    consolidated_flights = {}

    for flight_file in flight_files:
        filename = os.path.basename(flight_file)
        match = re.search(r'Historical_([A-Z0-9]+)_([A-Z]+)-([A-Z]+)', filename)
        if match:
            flight_code, origin, destination = match.groups()
            route_key = f"{flight_code}_{origin}-{destination}"
            flights = parse_flight_file(flight_file)

            if route_key not in consolidated_flights:
                consolidated_flights[route_key] = []

            for flight_id, flight in flights.items():
                cleaned_flight = clean_flight_data(flight)
                if cleaned_flight:
                    consolidated_flights[route_key].append(cleaned_flight)
                    logger.info(f"Cleaned flight {flight_id} with {len(cleaned_flight['waypoints'])} waypoints, max altitude: {cleaned_flight['metadata']['max_altitude_feet']} ft")
                    if visualize:
                        viz_dir = os.path.join(output_dir, "visualization")
                        os.makedirs(viz_dir, exist_ok=True)
                        visualize_flight_phases(cleaned_flight, f"{viz_dir}/phases_{flight_id}.png")

    total_flights = 0
    for route_key, flights in consolidated_flights.items():
        total_flights += len(flights)
        if output_format.lower() == 'json':
            output_filename = f"Cleaned_{route_key}.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump({'route': route_key, 'flight_count': len(flights), 'flights': flights}, f, indent=2)
            logger.info(f"Saved {len(flights)} flights to {output_path}")
        elif output_format.lower() == 'csv':
            flights_file = os.path.join(output_dir, f"flights_{route_key}.csv")
            waypoints_file = os.path.join(output_dir, f"waypoints_{route_key}.csv")

            with open(flights_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["flight_id", "route", "origin", "destination", "departure_time", "arrival_time", "waypoint_count", "max_altitude", "max_speed", "distance_km"])
                for i, flight in enumerate(flights):
                    flight_date = datetime.now().strftime("%Y%m%d")
                    unique_id = f"{flight['flight_id']}_{flight_date}_{i}"
                    writer.writerow([
                        unique_id,
                        route_key,
                        flight['origin'],
                        flight['destination'],
                        flight.get('departure_time', ''),
                        flight.get('arrival_time', ''),
                        len(flight['waypoints']),
                        flight['metadata']['max_altitude_feet'] or '',
                        flight['metadata']['max_speed_kts'] or '',
                        flight['metadata']['distance_km'] or ''
                    ])

            with open(waypoints_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["flight_id", "sequence", "timestamp", "latitude", "longitude", "kts", "mph", "feet", "rate", "phase", "distance_km", "cum_distance"])
                for i, flight in enumerate(flights):
                    flight_date = datetime.now().strftime("%Y%m%d")
                    unique_id = f"{flight['flight_id']}_{flight_date}_{i}"
                    for j, waypoint in enumerate(flight['waypoints']):
                        writer.writerow([
                            unique_id,
                            str(j+1),
                            waypoint.get('timestamp', ''),
                            str(waypoint.get('latitude', '')),
                            str(waypoint.get('longitude', '')),
                            str(waypoint.get('kts', '')),
                            str(waypoint.get('mph', '')),
                            str(waypoint.get('feet', ''),),
                            str(waypoint.get('rate', '')),
                            waypoint.get('phase', 'unknown'),
                            str(waypoint.get('distance_km', 0)),
                            str(waypoint.get('cum_distance', 0))
                        ])

            logger.info(f"Saved {len(flights)} flights to {flights_file} and {waypoints_file}")

    logger.info(f"Cleaning complete: {total_flights} total flights consolidated into {len(consolidated_flights)} route files")
    return total_flights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean historical flight data")
    parser.add_argument("--input", help="Directory containing raw flight data files")
    parser.add_argument("--output", help="Directory to save cleaned flight data")
    parser.add_argument("--format", choices=['json', 'csv'], default='csv', help="Output format for cleaned data")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of cleaned flights")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Set up directories based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir

    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Project root: {project_root}")

    input_dir = args.input if args.input else os.path.join(project_root, "historical")
    output_dir = args.output if args.output else os.path.join(project_root, 'data', "cleaned_historical")

    logger.info(f"Input directory: {os.path.abspath(input_dir)}")
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    logger.info(f"Output format: {args.format}")

    clean_historical_data(input_dir, output_dir, args.format, args.visualize)