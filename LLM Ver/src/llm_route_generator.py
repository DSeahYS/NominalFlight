import os
import json
import requests
import csv # Added for historical data
import pandas as pd # Added for airports.csv lookup
from dotenv import load_dotenv
import logging
import math # Added for distance calculation
from typing import Optional, List, Dict, Any # Added for type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file in the project root
# Assumes the script is run from the project root or the .env file is discoverable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f".env file loaded from {dotenv_path}")
else:
    # Fallback if running from a different structure or .env is directly in root
    load_dotenv()
    logging.info("Attempted to load .env from default location.")


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logging.error("OPENROUTER_API_KEY not found in environment variables.")
    # Consider raising an exception or handling this case appropriately
    # raise ValueError("OPENROUTER_API_KEY not set in the environment.")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-pro" # Changed to Gemini Pro

# KML Template (as requested, for potential later use)
# This isn't used by the generator function itself but included per instructions
KML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Flight Path</name>
    <Style id="lineStyle">
      <LineStyle>
        <color>ff0000ff</color> <!-- Blue color -->
        <width>2</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>Route</name>
      <styleUrl>#lineStyle</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode> <!-- Use absolute altitude -->
        <coordinates>
        {coordinates}
        </coordinates>
      </LineString>
    </Placemark>
    {waypoints}
  </Document>
</kml>
"""

KML_WAYPOINT_TEMPLATE = """
    <Placemark>
      <name>{name}</name>
      <description>Altitude: {alt} ft, Speed: {speed} kts</description>
      <Point>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>{lon},{lat},{alt_m}</coordinates> <!-- KML uses lon,lat,alt(meters) -->
      </Point>
    </Placemark>"""


# --- Helper Functions ---

def _get_aip_filepath(icao_code: str) -> Optional[str]:
    """Determines the AIP file path based on ICAO prefix."""
    # Simple prefix mapping (adjust as needed for more countries)
    prefix_map = {
        "WSSS": "SG",
        "WMKK": "MY",
        # Add more mappings here
    }
    prefix = icao_code[:4].upper() # Use first 4 chars for matching
    country_code = None
    # Find the longest matching prefix (e.g., handle WSSS vs WS...)
    for p, c in prefix_map.items():
        if prefix.startswith(p):
            country_code = c
            break # Assuming non-overlapping prefixes for now

    if country_code:
        script_dir = os.path.dirname(__file__)
        aip_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'aip', f"{country_code}_AIP.json"))
        return aip_path
    return None

def _load_aip_data(icao_code: str) -> Optional[Dict[str, Any]]:
    """Loads AIP data for the country corresponding to the ICAO code."""
    aip_filepath = _get_aip_filepath(icao_code)
    if aip_filepath and os.path.exists(aip_filepath):
        try:
            with open(aip_filepath, 'r', encoding='utf-8') as f:
                logging.info(f"Loading AIP data from: {aip_filepath}")
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {aip_filepath}: {e}")
        except Exception as e:
            logging.error(f"Error reading AIP file {aip_filepath}: {e}")
    else:
        logging.warning(f"AIP file not found or path not determined for ICAO: {icao_code} at path: {aip_filepath}")
    return None

def _load_historical_waypoints(departure_icao: str, arrival_icao: str) -> List[Dict[str, Any]]:
    """Loads historical waypoint data for a specific route."""
    script_dir = os.path.dirname(__file__)
    # Attempt to find a matching file (case-insensitive search might be better)
    # For now, assume a naming convention like waypoints_ANYTHING_DEP-ARR.csv
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'cleaned_historical'))
    target_suffix = f"_{departure_icao.upper()}-{arrival_icao.upper()}.csv"
    waypoints = []
    try:
        for filename in os.listdir(data_dir):
            if filename.startswith("waypoints_") and filename.endswith(target_suffix):
                filepath = os.path.join(data_dir, filename)
                logging.info(f"Loading historical waypoints from: {filepath}")
                with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Basic validation/conversion (adjust types as needed based on CSV content)
                        try:
                            waypoints.append({
                                'name': row.get('waypoint_id', 'UNKNOWN'),
                                'latitude': float(row.get('latitude', 0.0)),
                                'longitude': float(row.get('longitude', 0.0)),
                                # Add other relevant fields if present, e.g., altitude, speed
                            })
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Skipping row due to conversion error in {filename}: {row} - {e}")
                # Assume only one matching file per route pair
                break
        if not waypoints:
             logging.warning(f"No historical waypoint file found matching pattern: *_{departure_icao}-{arrival_icao}.csv in {data_dir}")

    except FileNotFoundError:
        logging.warning(f"Historical data directory not found: {data_dir}")
    except Exception as e:
        logging.error(f"Error reading historical data from {data_dir}: {e}")
    return waypoints

def _extract_context_from_data(dep_aip: Optional[Dict], arr_aip: Optional[Dict], historical_waypoints: List[Dict]) -> str:
    """Extracts concise context for the LLM prompt."""
    context_parts = []

    # Extract relevant info from AIP (Example: Airways, key waypoints)
    # This needs refinement based on actual AIP JSON structure
    if dep_aip:
        # Placeholder: Extract first 5 airways if available
        airways = dep_aip.get('airways', [])[:5]
        if airways:
             context_parts.append(f"Departure Region Airways (Sample): {', '.join([aw.get('name', 'N/A') for aw in airways])}")
    if arr_aip:
        # Placeholder: Extract first 5 arrival waypoints if available
        arr_wpts = arr_aip.get('waypoints', {}).get('arrival', [])[:5]
        if arr_wpts:
            context_parts.append(f"Arrival Region Waypoints (Sample): {', '.join([wp.get('name', 'N/A') for wp in arr_wpts])}")

    # Extract relevant info from Historical Data (Example: Common waypoints)
    if historical_waypoints:
        # Get unique waypoint names from historical data, limit count
        common_waypoints = list(set(wp['name'] for wp in historical_waypoints))[:10]
        if common_waypoints:
            context_parts.append(f"Historically Used Enroute Waypoints (Sample): {', '.join(common_waypoints)}")

    if not context_parts:
        return "No specific AIP or historical context available."

    return "\n".join(context_parts)


def _get_airport_coords(icao_code: str) -> Optional[Dict[str, float]]:
    """Gets airport coordinates from the airports.csv file."""
    script_dir = os.path.dirname(__file__)
    # Construct path relative to the script location
    airports_csv_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'airports.csv'))

    if not os.path.exists(airports_csv_path):
        logging.error(f"Airports CSV file not found at: {airports_csv_path}")
        return None

    try:
        df = pd.read_csv(airports_csv_path)
        # Ensure the 'ident' column exists and handle potential case differences
        if 'ident' not in df.columns:
             logging.error(f"'ident' column not found in {airports_csv_path}")
             return None

        # Perform case-insensitive matching for robustness
        airport_info = df[df['ident'].str.upper() == icao_code.upper()]

        if airport_info.empty:
            logging.warning(f"ICAO code '{icao_code}' not found in {airports_csv_path}")
            return None

        # Extract coordinates - assuming columns are named 'latitude_deg' and 'longitude_deg'
        if 'latitude_deg' in airport_info.columns and 'longitude_deg' in airport_info.columns:
            lat = airport_info.iloc[0]['latitude_deg']
            lon = airport_info.iloc[0]['longitude_deg']
            # Validate that coordinates can be converted to float
            return {
                "latitude": float(lat),
                "longitude": float(lon)
            }
        else:
            logging.error(f"Coordinate columns ('latitude_deg', 'longitude_deg') not found for {icao_code} in {airports_csv_path}")
            return None

    except pd.errors.EmptyDataError:
        logging.error(f"Airports CSV file is empty: {airports_csv_path}")
        return None
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        logging.error(f"Airports CSV file not found during read: {airports_csv_path}")
        return None
    except (ValueError, TypeError) as e:
         logging.error(f"Error converting coordinates to float for {icao_code} in {airports_csv_path}: {e}")
         return None
    except Exception as e:
        logging.error(f"Error reading or processing {airports_csv_path} for {icao_code}: {e}")
        return None

# --- Helper: Distance Calculation ---

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees) using Haversine formula.
    Returns distance in nautical miles.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers. Use 6371 km
    # Convert to nautical miles (1 km = 0.539957 NM)
    r = 6371 * 0.539957
    return c * r

# --- Physics-Based Post-Processing ---

def apply_descent_profile(route: List[Dict[str, Any]], level_segment_nm: float = 15.0, level_segment_alt_ft: float = 3000.0) -> List[Dict[str, Any]]:
    """
    Adjusts waypoint altitudes in the descent phase for a more realistic profile.

    Args:
        route: The full list of waypoints (departure, enroute, arrival).
        level_segment_nm: Distance from arrival for the level segment (nm).
        level_segment_alt_ft: Target altitude for the start of the level segment (ft MSL).

    Returns:
        The route with adjusted altitudes for the descent phase.
    """
    if len(route) < 3:
        logging.warning("Route too short to apply descent profile, returning original.")
        return route # Need at least Dep, Cruise, Arr

    # Find approximate cruise altitude (highest altitude before the last waypoint)
    # Exclude the last point as it might be a placeholder ground altitude
    cruise_altitude = 0
    cruise_index = 0
    for i, wp in enumerate(route[:-1]):
        if wp['altitude'] > cruise_altitude:
            cruise_altitude = wp['altitude']
            cruise_index = i

    arrival_wp = route[-1]
    arrival_alt = arrival_wp['altitude'] # Use the placeholder arrival altitude

    # --- 1. Calculate Required Descent Distance (3:1 Rule) ---
    altitude_to_lose_kft = (cruise_altitude - level_segment_alt_ft) / 1000.0
    if altitude_to_lose_kft <= 0:
         logging.info(f"Cruise altitude ({cruise_altitude} ft) is already at or below level segment altitude ({level_segment_alt_ft} ft). No descent adjustment needed.")
         # Ensure level segment altitude is applied if close enough
         dist_to_arrival = 0.0
         for i in range(len(route) - 1, 0, -1):
             wp_curr = route[i]
             wp_prev = route[i-1]
             dist_to_arrival += calculate_distance(wp_curr['latitude'], wp_curr['longitude'], wp_prev['latitude'], wp_prev['longitude'])
             if dist_to_arrival <= level_segment_nm:
                 route[i-1]['altitude'] = max(level_segment_alt_ft, route[i-1]['altitude']) # Don't force below original if higher
             else:
                 break # Stop adjusting once past the level segment distance
         return route


    required_descent_dist_nm = 3.0 * altitude_to_lose_kft
    logging.info(f"Cruise Alt: {cruise_altitude:.0f} ft, Level Segment Alt: {level_segment_alt_ft:.0f} ft")
    logging.info(f"Altitude to lose to level segment: {altitude_to_lose_kft * 1000:.0f} ft")
    logging.info(f"Required descent distance (3:1 rule to level segment): {required_descent_dist_nm:.1f} nm")

    # --- 2. Iterate Backwards to Find Top of Descent (TOD) ---
    cumulative_dist_nm = 0.0
    tod_index = -1
    distances_from_arrival = [0.0] * len(route) # Store distances for interpolation

    for i in range(len(route) - 1, 0, -1):
        wp_curr = route[i]
        wp_prev = route[i-1]
        segment_dist = calculate_distance(wp_curr['latitude'], wp_curr['longitude'],
                                          wp_prev['latitude'], wp_prev['longitude'])
        cumulative_dist_nm += segment_dist
        distances_from_arrival[i-1] = cumulative_dist_nm

        # Check if this segment crosses the required descent distance threshold
        if cumulative_dist_nm >= required_descent_dist_nm and tod_index == -1:
            tod_index = i - 1 # The *start* of the segment where TOD occurs
            logging.info(f"Calculated TOD point near waypoint {tod_index} ('{wp_prev['name']}'), {cumulative_dist_nm:.1f} nm from arrival.")
            # Optional: Interpolate exact TOD point if needed, but waypoint index is often sufficient

    if tod_index == -1:
        logging.warning(f"Route length ({cumulative_dist_nm:.1f} nm) seems shorter than required descent distance ({required_descent_dist_nm:.1f} nm). Applying descent from highest point.")
        tod_index = cruise_index # Start descent from the highest point found

    # --- 3. Adjust Altitudes from TOD to Arrival ---
    total_descent_distance_actual = distances_from_arrival[tod_index] # Distance from TOD waypoint to arrival
    logging.info(f"Adjusting altitudes from waypoint {tod_index} ('{route[tod_index]['name']}') covering {total_descent_distance_actual:.1f} nm.")

    for i in range(tod_index, len(route) - 1):
        dist_to_arrival = distances_from_arrival[i]
        wp_current = route[i]

        # --- 3a. Handle Level Segment ---
        if dist_to_arrival <= level_segment_nm:
            target_alt = level_segment_alt_ft
            logging.debug(f"WP {i} ('{wp_current['name']}'): Within level segment ({dist_to_arrival:.1f} nm <= {level_segment_nm:.1f} nm). Setting alt to {target_alt:.0f} ft.")

        # --- 3b. Handle Descent Segment ---
        else:
            # Calculate how far *into* the descent we are (as a fraction)
            # Distance covered from TOD = total_descent_distance_actual - dist_to_arrival
            # Fraction of descent covered = (Distance covered from TOD) / (total_descent_distance_actual - level_segment_nm)
            # Altitude dropped = Fraction * (cruise_altitude - level_segment_alt_ft)
            # Target Altitude = cruise_altitude - Altitude dropped

            dist_covered_from_tod = total_descent_distance_actual - dist_to_arrival
            descent_phase_dist = total_descent_distance_actual - level_segment_nm

            if descent_phase_dist <= 0: # Avoid division by zero if TOD is within level segment
                 fraction_descended = 1.0 # Effectively already at level segment alt
            else:
                fraction_descended = max(0.0, min(1.0, dist_covered_from_tod / descent_phase_dist))

            altitude_dropped = fraction_descended * (cruise_altitude - level_segment_alt_ft)
            target_alt = cruise_altitude - altitude_dropped
            logging.debug(f"WP {i} ('{wp_current['name']}'): Descending. Dist to Arr: {dist_to_arrival:.1f} nm. Frac Desc: {fraction_descended:.2f}. Target Alt: {target_alt:.0f} ft.")


        # --- 3c. Apply Altitude & Speed Restriction Check ---
        # Apply the calculated target altitude
        wp_current['altitude'] = max(target_alt, arrival_alt) # Don't go below final arrival alt

        # Speed Restriction Check (Informational for now)
        if wp_current['altitude'] < 10000 and wp_current['speed'] > 250:
            logging.warning(f"Waypoint {i} ('{wp_current['name']}') is below 10,000 ft ({wp_current['altitude']:.0f} ft) but speed is {wp_current['speed']} kts (> 250 kts). Profile adjusted, but speed may need manual review.")
            # Future: Could potentially adjust the profile more conservatively below 10k ft
            # or even attempt to adjust speed if logic allows.

    # Ensure the final arrival waypoint altitude is unchanged
    route[-1]['altitude'] = arrival_alt

    logging.info("Descent profile adjustment complete.")
    return route


# --- Main Function ---

def generate_llm_route(departure_icao: str, arrival_icao: str, aircraft_model: str, constraints_text: str = "") -> list[dict]:
    """
    Generates the *enroute* portion of a flight route using an LLM, informed by
    AIP and historical data, and integrates it with fixed departure/arrival points.

    Args:
        departure_icao: ICAO code of the departure airport (e.g., "WSSS").
        arrival_icao: ICAO code of the arrival airport (e.g., "WMKK").
        aircraft_model: Model of the aircraft (e.g., "A320").
        constraints_text: Unstructured text describing additional constraints.

    Returns:
        A list of waypoint dictionaries in the internal format, including departure,
        LLM-generated enroute points, and arrival.
        [{'latitude': float, 'longitude': float, 'altitude': float, 'speed': int, 'name': str}, ...]
        Returns an empty list if generation fails or essential data is missing.
    """
    if not OPENROUTER_API_KEY:
        logging.error("Cannot generate route: OPENROUTER_API_KEY is not configured.")
        return []

    # --- 1. Load Context Data ---
    dep_aip_data = _load_aip_data(departure_icao)
    arr_aip_data = _load_aip_data(arrival_icao) # Load arrival AIP too for context/coords
    historical_waypoints = _load_historical_waypoints(departure_icao, arrival_icao)

    # --- 2. Get Departure/Arrival Coordinates ---
    dep_coords = _get_airport_coords(departure_icao) # Now reads from airports.csv
    arr_coords = _get_airport_coords(arrival_icao)   # Now reads from airports.csv

    if not dep_coords or not arr_coords:
        logging.error(f"Could not determine coordinates for departure ({departure_icao}) or arrival ({arrival_icao}). Cannot generate route.")
        return []

    # --- 3. Extract Context for Prompt ---
    prompt_context = _extract_context_from_data(dep_aip_data, arr_aip_data, historical_waypoints)

    # --- 4. Construct LLM Prompt ---
    prompt = f"""
Generate a realistic *enroute* flight path segment for an {aircraft_model} flying from {departure_icao} ({dep_coords['latitude']:.4f}, {dep_coords['longitude']:.4f}) to {arrival_icao} ({arr_coords['latitude']:.4f}, {arr_coords['longitude']:.4f}).

Do NOT include the departure or arrival airports themselves in the output list. Generate only the waypoints *between* them.

Context based on AIP and historical data (use this for guidance):
{prompt_context}

Additional Constraints:
- Adhere to basic flight physics principles (reasonable climb/descent, turns). Generate enroute waypoints considering a standard descent profile (approx. 3 degrees) towards the destination.
- Incorporate the following specific user constraints: "{constraints_text if constraints_text else 'None'}"
- The output MUST be a valid JSON list of waypoint objects. Do not include any text before or after the JSON list.
- Each waypoint object in the JSON list must have the following keys exactly: "name" (string, waypoint identifier), "lat" (float, latitude), "lon" (float, longitude), "alt" (float, altitude in feet MSL), "speed" (integer, indicated airspeed in knots). Ensure altitude and speed profiles are realistic for an enroute segment.

Example JSON output format (enroute waypoints only):
[
  {{"name": "ENR1", "lat": 1.550, "lon": 103.850, "alt": 15000, "speed": 280}},
  {{"name": "ENR2", "lat": 1.800, "lon": 103.500, "alt": 25000, "speed": 350}},
  {{"name": "ENR3", "lat": 2.100, "lon": 103.200, "alt": 25000, "speed": 350}}
]

Generate the enroute flight path segment now:
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"} # Request JSON output if model supports it
    }

    logging.info(f"Requesting route from {LLM_MODEL} for {departure_icao} to {arrival_icao}")
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=120) # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()

        # Extract the content which should contain the JSON string
        if not response_data.get("choices") or not response_data["choices"][0].get("message") or not response_data["choices"][0]["message"].get("content"):
             logging.error("Invalid response structure from LLM API.")
             return []

        llm_output_str = response_data["choices"][0]["message"]["content"]
        logging.debug(f"Raw LLM Output:\n{llm_output_str}")

        # Attempt to parse the JSON output from the LLM
        try:
            # The response might be wrapped in ```json ... ```, try to extract it
            if llm_output_str.strip().startswith("```json"):
                llm_output_str = llm_output_str.strip()[7:-3].strip()
            elif llm_output_str.strip().startswith("```"):
                 llm_output_str = llm_output_str.strip()[3:-3].strip()

            waypoints_llm_format = json.loads(llm_output_str)

            # Validate structure: should be a list of dictionaries
            if not isinstance(waypoints_llm_format, list):
                logging.error(f"LLM output is not a list: {type(waypoints_llm_format)}")
                return []
            if not all(isinstance(wp, dict) for wp in waypoints_llm_format):
                logging.error("LLM output list does not contain only dictionaries.")
                return []

            # Validate keys and convert to internal format
            internal_route = []
            required_keys = {"name", "lat", "lon", "alt", "speed"}
            for i, wp in enumerate(waypoints_llm_format):
                if not required_keys.issubset(wp.keys()):
                    logging.error(f"Waypoint {i} is missing required keys. Found: {wp.keys()}")
                    return [] # Or handle partially valid routes? For now, reject all.

                try:
                    internal_route.append({
                        "name": str(wp["name"]),
                        "latitude": float(wp["lat"]),
                        "longitude": float(wp["lon"]),
                        "altitude": float(wp["alt"]), # Assuming alt is already in feet
                        "speed": int(wp["speed"])
                    })
                except (ValueError, TypeError) as e:
                     logging.error(f"Error converting types for waypoint {i}: {wp}. Error: {e}")
                     return [] # Reject route if type conversion fails

            logging.info(f"Successfully generated and parsed enroute segment with {len(internal_route)} waypoints.")

            # --- 6. Integrate Departure, Enroute, and Arrival ---
            # Create placeholder departure/arrival waypoints
            # Altitudes/speeds here are arbitrary placeholders - might need refinement
            dep_waypoint = {
                "name": departure_icao,
                "latitude": dep_coords['latitude'],
                "longitude": dep_coords['longitude'],
                "altitude": 100, # Placeholder altitude (e.g., ground level)
                "speed": 60      # Placeholder speed (e.g., taxi speed)
            }
            arr_waypoint = {
                "name": arrival_icao,
                "latitude": arr_coords['latitude'],
                "longitude": arr_coords['longitude'],
                "altitude": 100, # Placeholder altitude
                "speed": 60      # Placeholder speed
            }

            # --- 7. Apply Physics-Based Descent Profile ---
            raw_final_route = [dep_waypoint] + internal_route + [arr_waypoint]
            logging.info(f"Constructed raw final route with {len(raw_final_route)} waypoints (Dep + Enroute + Arr).")

            # Apply the descent profile adjustments
            final_route_adjusted = apply_descent_profile(raw_final_route)

            logging.info("Applied physics-based descent profile adjustments.")
            return final_route_adjusted

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from LLM for enroute segment: {e}")
            logging.error(f"LLM Raw Output (Enroute) was: {llm_output_str}")
            return []
        except KeyError as e:
             logging.error(f"Missing expected key in LLM enroute waypoint data: {e}")
             return []
        except Exception as e: # Catch any other validation/conversion errors
            logging.error(f"An unexpected error occurred during enroute waypoint processing: {e}")
            return []


    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling OpenRouter API for enroute segment: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_llm_route: {e}")
        return []

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Testing LLM Enroute Generator...")
    # Ensure you have:
    # 1. A .env file with OPENROUTER_API_KEY in the project root.
    # 2. AIP files (e.g., SG_AIP.json, MY_AIP.json) in data/aip/
    # 3. Historical waypoint files (e.g., waypoints_..._WSSS-WMKK.csv) in data/cleaned_historical/
    # 4. The airports.csv file in data/ with 'ident', 'latitude_deg', 'longitude_deg' columns.

    dep_test = "WSSS"
    arr_test = "WMKK"
    print(f"Attempting to generate route from {dep_test} to {arr_test}...")

    # Check if data files exist before running
    airports_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'airports.csv'))
    hist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_historical'))
    hist_file_exists = False
    try:
        hist_file_exists = any(f.startswith("waypoints_") and f.endswith(f"_{dep_test}-{arr_test}.csv") for f in os.listdir(hist_dir))
    except FileNotFoundError:
        print(f"WARNING: Historical data directory not found: {hist_dir}")




    if not os.path.exists(airports_csv_path):
         print(f"ERROR: airports.csv not found at expected path: {airports_csv_path}")
    elif not hist_file_exists:
         print(f"ERROR: Historical waypoint file not found for {dep_test}-{arr_test} in {hist_dir}")
    else:
        print(f"Attempting to generate route from {dep_test} to {arr_test} using airports.csv for coordinates...")
        test_route = generate_llm_route(dep_test, arr_test, "A320", "Maintain FL250 if possible")
        if test_route:
            print("\nGenerated Integrated Route (Dep + Enroute + Arr):")
            for i, point in enumerate(test_route):
                print(f"  {i+1}. {point}")

            # Optional: Save to KML (using existing templates)
            try:
                coords_str = "\n".join([f"{wp['longitude']},{wp['latitude']},{wp['altitude'] * 0.3048}" for wp in test_route]) # Convert ft to meters for KML
                waypoints_kml = "\n".join([
                    KML_WAYPOINT_TEMPLATE.format(
                        name=wp['name'],
                        lat=wp['latitude'],
                        lon=wp['longitude'],
                        alt=wp['altitude'],
                        alt_m=wp['altitude'] * 0.3048, # Convert ft to meters
                        speed=wp['speed']
                    ) for wp in test_route
                ])
                final_kml = KML_TEMPLATE.format(coordinates=coords_str, waypoints=waypoints_kml)
                kml_filename = f"generated_enroute_{dep_test}-{arr_test}.kml"
                with open(kml_filename, "w", encoding='utf-8') as f:
                    f.write(final_kml)
                print(f"\nSaved integrated route KML to: {kml_filename}")
            except Exception as e:
                print(f"\nError generating KML: {e}")

        else:
            print("\nFailed to generate route.")