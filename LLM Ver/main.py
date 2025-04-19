# Final Year Project\10. New Airbus Synthetic Flight Generator (Researched)\main.py

import argparse
import os
import sys
import logging
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from http.server import SimpleHTTPRequestHandler, HTTPServer # Use HTTPServer for clarity
from socketserver import TCPServer
import webbrowser
from geopy.distance import great_circle
import copy

# Set up logging
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_FOLDER, "flight_generator.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import modules
from src.data_processor import load_airports, load_aip_data, load_nominal_patterns
from src.route_planner import construct_nominal_route, generate_synthetic_flights, create_strictly_nominal_route
from src.flight_dynamics import AircraftPerformance
from src.route_smoother import NominalRouteSmoother
from src.visualization import visualize_flight_paths, save_kml, save_combined_kml, generate_kml # Keep generate_kml
from src.llm_route_generator import generate_llm_route # Import LLM generator

#-------------------------------------------------------------------------
# Pipeline Functions (Nominal/Synthetic) - Copied from main(old).py
#-------------------------------------------------------------------------

def get_nominal_pattern(departure, arrival, nominal_patterns):
    """
    Get the original nominal pattern without any variations.
    (Copied from main(old).py)
    """
    route_key = f"{departure}-{arrival}"
    if route_key in nominal_patterns:
        pattern = nominal_patterns[route_key]
        return pattern.get('waypoints', [])
    reverse_key = f"{arrival}-{departure}"
    if reverse_key in nominal_patterns:
        pattern = nominal_patterns[reverse_key]
        waypoints = pattern.get('waypoints', [])
        reversed_waypoints = list(reversed(waypoints))
        for wp in reversed_waypoints:
            if 'heading' in wp and wp['heading'] is not None:
                wp['heading'] = (wp['heading'] + 180) % 360
        return reversed_waypoints
    return []

def calculate_route_length(route):
    """
    Calculate the total length of a route in kilometers.
    (Copied from main(old).py)
    """
    if not route or len(route) < 2:
        return 0
    total_distance = 0
    for i in range(1, len(route)):
        try:
            # Ensure waypoints have lat/lon keys
            if 'latitude' in route[i-1] and 'longitude' in route[i-1] and \
               'latitude' in route[i] and 'longitude' in route[i]:
                distance = great_circle(
                    (route[i-1]['latitude'], route[i-1]['longitude']),
                    (route[i]['latitude'], route[i]['longitude'])
                ).kilometers
                total_distance += distance
            else:
                 logger.warning(f"Skipping distance calculation for segment {i} due to missing lat/lon.")
        except Exception as e:
            logger.warning(f"Error calculating distance between waypoints {i-1} and {i}: {e}")
    return total_distance

def visualize_with_triple_comparison(synthetic_routes, nominal_route):
    """
    Visualize nominal, strictly nominal, and synthetic routes together for comparison.
    (Copied from main(old).py)
    """
    nominal_route_with_flag = copy.deepcopy(nominal_route)
    for wp in nominal_route_with_flag:
        wp['is_nominal'] = True

    # Use the global variable for deviation
    global STRICT_NOMINAL_DEVIATION
    strict_nominal = create_strictly_nominal_route(nominal_route, fixed_deviation=STRICT_NOMINAL_DEVIATION)
    for wp in strict_nominal:
        wp['is_strict_nominal'] = True # Add flag for strict nominal

    all_routes = [nominal_route_with_flag, strict_nominal] + synthetic_routes
    visualize_flight_paths(all_routes)

def run_pipeline(departure, arrival, aircraft="A320-214", visualize=True,
                detail_level=2, output_format="kml", callback=None):
    """
    Run the nominal flight path generation pipeline with progress reporting.
    (Copied from main(old).py, minor adjustments)
    """
    result = {
        'success': False, 'message': '', 'route': None, 'metadata': {}, 'output_file': None
    }
    try:
        if callback: callback(0, "Starting nominal flight path generation...")

        # Step 1: Load core data
        if callback: callback(10, "Loading airport data...")
        airports_file = os.path.join(project_root, 'data', 'airports.csv')
        aip_dir = os.path.join(project_root, 'data', 'aip')
        if not os.path.exists(airports_file): raise FileNotFoundError(f"Airports data file not found: {airports_file}")
        if not os.path.exists(aip_dir): raise FileNotFoundError(f"AIP directory not found: {aip_dir}")
        airports = load_airports(airports_file)
        if callback: callback(20, "Loading AIP data...")
        aip_data = load_aip_data(aip_dir)
        if not airports: raise Exception("Failed to load airports data")
        if not aip_data: raise Exception("Failed to load AIP data")

        # Step 2: Load nominal patterns
        if callback: callback(40, "Loading nominal patterns...")
        nominal_patterns_file = os.path.join(project_root, 'data', 'nominal', 'nominal_patterns.json')
        try:
            nominal_patterns = load_nominal_patterns(nominal_patterns_file)
            if callback: callback(50, f"Loaded patterns for {len(nominal_patterns)} routes")
        except Exception as e:
            logger.error(f"Error loading nominal patterns: {e}")
            nominal_patterns = {} # Use empty dict if loading fails
            if callback: callback(50, "Using empty nominal patterns due to error.")

        # Step 3: Get the nominal pattern for reference
        nominal_route_raw = get_nominal_pattern(departure, arrival, nominal_patterns)

        # Step 4: Create aircraft performance model
        if callback: callback(60, f"Initializing aircraft performance model for {aircraft}...")
        aircraft_performance = AircraftPerformance(aircraft)
        aircraft_info = aircraft_performance.get_aircraft_info()
        if callback: callback(65, f"Using aircraft: {aircraft_info['type']}")

        # Step 5: Generate nominal route
        if callback: callback(70, f"Generating nominal route from {departure} to {arrival}...")
        route_key = f"{departure}-{arrival}"
        inverse_route_key = f"{arrival}-{departure}"
        if route_key not in nominal_patterns and inverse_route_key not in nominal_patterns:
            if callback: callback(75, f"No nominal pattern found for {route_key}, using direct routing.")

        route = construct_nominal_route(departure, arrival, aip_data, nominal_patterns)
        if not route: raise Exception("Failed to generate route. Check airport codes and AIP data.")

        route_length = calculate_route_length(route)
        if callback: callback(80, f"Route generated with {len(route)} waypoints and {route_length:.1f} km length")

        # Apply aircraft performance profile
        route = aircraft_performance.apply_performance_profile(route, route_length)
        if nominal_route_raw:
            nominal_route_perf = aircraft_performance.apply_performance_profile(
                nominal_route_raw, calculate_route_length(nominal_route_raw)
            )
        else:
            nominal_route_perf = None

        # Step 6: Apply route smoothing with Kalman filter
        if callback: callback(85, "Applying advanced Kalman smoothing...")
        smoothing_params = {1: {'points': 50}, 2: {'points': 100}, 3: {'points': 200}}
        params = smoothing_params.get(detail_level, smoothing_params[2])
        route_smoother = NominalRouteSmoother(aircraft_performance)
        try:
            smoothed_route = route_smoother.smooth_route(
                route, points=params['points'], variant_id=0, micro_var_level=0.03
            )
        except TypeError: # Handle potential older signature
             smoothed_route = route_smoother.smooth_route(route)

        if callback: callback(90, f"Smoothed route generated with {len(smoothed_route)} waypoints")

        # Step 7: Save and/or visualize
        output_dir = os.path.join(project_root, 'visualization')
        os.makedirs(output_dir, exist_ok=True)
        output_files = {
            'json': os.path.join(output_dir, f'flight_path_{departure}_{arrival}.json'),
            'kml': os.path.join(output_dir, f'flight_path_{departure}_{arrival}.kml'),
            'csv': os.path.join(output_dir, f'flight_path_{departure}_{arrival}.csv')
        }
        output_file = output_files.get(output_format, output_files['kml']) # Default KML

        output_data = {
            'metadata': {
                'departure': departure, 'arrival': arrival, 'aircraft': aircraft,
                'distance_km': route_length, 'waypoint_count': len(smoothed_route),
                'cruise_altitude': aircraft_info['cruise_altitude'],
                'cruise_speed': aircraft_info['cruise_speed'], 'generator': 'nominal'
            },
            'route': smoothed_route
        }

        if output_format == 'json':
            with open(output_file, 'w') as f: json.dump(output_data, f, indent=2)
        elif output_format == 'kml':
            save_kml(smoothed_route, output_file, f"{departure} to {arrival}")
        elif output_format == 'csv':
            with open(output_file, 'w') as f:
                f.write("latitude,longitude,altitude,speed,heading\n")
                for wp in smoothed_route:
                    f.write(f"{wp['latitude']},{wp['longitude']},{wp['altitude']},{wp['speed']},{wp.get('heading', '')}\n")

        if callback: callback(95, f"Route data saved to {output_file}")

        if visualize:
            if callback: callback(98, "Visualizing flight path...")
            if nominal_route_perf:
                visualize_with_triple_comparison([smoothed_route], nominal_route_perf)
            else:
                visualize_flight_paths([smoothed_route])

        if callback: callback(100, "Nominal flight path generation complete")
        result.update({
            'success': True, 'message': "Nominal flight path generated successfully",
            'route': smoothed_route, 'metadata': output_data['metadata'], 'output_file': output_file
        })
        return result

    except Exception as e:
        import traceback
        error_message = f"Error in nominal flight path generation: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        result.update({'success': False, 'message': error_message})
        if callback: callback(0, error_message)
        return result

def run_multi_route_pipeline(departure, arrival, aircraft="A320-214", route_count=5,
                           max_variation=0.3, visualize=True, detail_level=2,
                           output_format="kml", callback=None):
    """
    Generate multiple synthetic flight paths between two airports.
    (Copied from main(old).py, minor adjustments)
    """
    result = {
        'success': False, 'message': '', 'routes': [], 'metadata': {}, 'output_files': []
    }
    try:
        if callback: callback(0, f"Starting generation of {route_count} flight paths...")

        # Step 1: Load data
        if callback: callback(10, "Loading data...")
        airports_file = os.path.join(project_root, 'data', 'airports.csv')
        aip_dir = os.path.join(project_root, 'data', 'aip')
        airports = load_airports(airports_file)
        aip_data = load_aip_data(aip_dir)
        nominal_patterns_file = os.path.join(project_root, 'data', 'nominal', 'nominal_patterns.json')
        try:
            nominal_patterns = load_nominal_patterns(nominal_patterns_file)
        except Exception as e:
            logger.error(f"Error loading nominal patterns for multi-route: {e}")
            nominal_patterns = {}

        # Step 2: Get nominal pattern reference
        if callback: callback(30, "Extracting nominal pattern for reference...")
        nominal_route_raw = get_nominal_pattern(departure, arrival, nominal_patterns)

        # Step 3: Aircraft model
        if callback: callback(35, f"Initializing aircraft performance model...")
        aircraft_performance = AircraftPerformance(aircraft)
        if nominal_route_raw:
            nominal_route_perf = aircraft_performance.apply_performance_profile(
                nominal_route_raw, calculate_route_length(nominal_route_raw)
            )
        else:
            nominal_route_perf = None

        # Step 4: Generate multiple routes
        if callback: callback(40, f"Generating {route_count} routes...")
        actual_variation = min(max_variation, 0.25) # Limit variation
        routes = generate_synthetic_flights(
            departure=departure, arrival=arrival, aip_data=aip_data,
            nominal_patterns=nominal_patterns, count=route_count, max_variation=actual_variation
        )
        if not routes: raise Exception(f"Failed to generate routes between {departure} and {arrival}")

        # Step 5: Smooth routes
        if callback: callback(60, "Applying advanced Kalman smoothing...")
        smoother = NominalRouteSmoother(aircraft_performance)
        smoothed_routes = []
        for i, route in enumerate(routes):
            if callback:
                progress = 60 + (30 * (i+1) / len(routes))
                callback(progress, f"Smoothing route {i+1}/{len(routes)}...")
            try:
                smoothed_route = smoother.smooth_route(
                    route, points=100 * detail_level // 2, variant_id=i, micro_var_level=0.03
                )
            except TypeError:
                 smoothed_route = smoother.smooth_route(route) # Fallback
            smoothed_routes.append(smoothed_route)

        # Step 6: Save outputs and visualize
        if callback: callback(90, "Saving route data...")
        output_dir = os.path.join(project_root, 'visualization')
        os.makedirs(output_dir, exist_ok=True)

        # Save individual files and prepare for combined KML/JSON
        output_files_list = []
        all_routes_for_output = []
        base_filename = f'multi_flight_path_{departure}_{arrival}'

        # Add nominal and strict nominal if available
        if nominal_route_perf:
            nominal_route_flagged = copy.deepcopy(nominal_route_perf)
            for wp in nominal_route_flagged: wp['is_nominal'] = True
            all_routes_for_output.append(nominal_route_flagged)

            global STRICT_NOMINAL_DEVIATION
            strict_nominal = create_strictly_nominal_route(nominal_route_perf, fixed_deviation=STRICT_NOMINAL_DEVIATION)
            for wp in strict_nominal: wp['is_strict_nominal'] = True
            all_routes_for_output.append(strict_nominal)

        all_routes_for_output.extend(smoothed_routes) # Add synthetic routes

        # Save based on format
        if output_format == 'kml':
            combined_kml_file = os.path.join(output_dir, f'{base_filename}.kml')
            generate_kml(all_routes_for_output, combined_kml_file) # generate_kml handles list
            output_files_list.append(combined_kml_file)
        elif output_format == 'json':
             # Save a single JSON with all routes
             combined_json_file = os.path.join(output_dir, f'{base_filename}.json')
             json_output = {
                 'metadata': {
                     'departure': departure, 'arrival': arrival, 'aircraft': aircraft,
                     'route_count': route_count, 'generator': 'multi-synthetic'
                 },
                 'routes': all_routes_for_output # Store all routes in the list
             }
             with open(combined_json_file, 'w') as f: json.dump(json_output, f, indent=2)
             output_files_list.append(combined_json_file)
        elif output_format == 'csv':
             # Save each route as a separate CSV
             for i, s_route in enumerate(smoothed_routes):
                 csv_file = os.path.join(output_dir, f'{base_filename}_route_{i+1}.csv')
                 with open(csv_file, 'w') as f:
                     f.write("latitude,longitude,altitude,speed,heading\n")
                     for wp in s_route:
                         f.write(f"{wp['latitude']},{wp['longitude']},{wp['altitude']},{wp['speed']},{wp.get('heading', '')}\n")
                 output_files_list.append(csv_file)
             logger.info(f"Saved {len(output_files_list)} CSV files.")


        if callback: callback(95, f"Route data saved.")

        if visualize:
            if callback: callback(98, "Visualizing flight paths...")
            visualize_flight_paths(all_routes_for_output) # Visualize all together

        if callback: callback(100, f"Multi-route generation complete ({route_count} routes).")
        result.update({
            'success': True, 'message': f"Generated {route_count} routes successfully.",
            'routes': smoothed_routes, 'metadata': {'generator': 'multi-synthetic'}, 'output_files': output_files_list
        })
        return result

    except Exception as e:
        import traceback
        error_message = f"Error in multi-route generation: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        result.update({'success': False, 'message': error_message})
        if callback: callback(0, error_message)
        return result

#-------------------------------------------------------------------------
# LLM Pipeline Function (New)
#-------------------------------------------------------------------------

def run_llm_pipeline(departure, arrival, aircraft="A320", constraints_text="Standard route",
                     visualize=True, output_format="kml", callback=None):
    """
    Run the LLM-based flight path generation pipeline.
    """
    result = {
        'success': False, 'message': '', 'route': None, 'metadata': {}, 'output_file': None
    }
    try:
        if callback: callback(0, "Starting LLM flight path generation...")

        # Step 1: Generate route using LLM
        if callback: callback(20, f"Requesting route from LLM for {departure} to {arrival}...")
        llm_route = generate_llm_route(departure, arrival, aircraft, constraints_text)
        if not llm_route: raise Exception("LLM failed to generate a valid route.")
        if callback: callback(70, f"LLM generated route with {len(llm_route)} waypoints.")

        # Step 2: Prepare metadata
        route_length = calculate_route_length(llm_route)
        metadata = {
            'departure': departure, 'arrival': arrival, 'aircraft': aircraft,
            'constraints': constraints_text, 'generator': 'LLM',
            'distance_km': route_length, 'waypoint_count': len(llm_route)
        }
        result['metadata'] = metadata

        # Step 3: Save output
        output_dir = os.path.join(project_root, 'visualization')
        os.makedirs(output_dir, exist_ok=True)
        output_files = {
            'json': os.path.join(output_dir, f'llm_flight_path_{departure}_{arrival}.json'),
            'kml': os.path.join(output_dir, f'llm_flight_path_{departure}_{arrival}.kml')
        }
        output_file = output_files.get(output_format, output_files['kml']) # Default KML
        if callback: callback(80, f"Saving route data to {output_file}...")

        # Add flag for visualization differentiation
        llm_route_for_saving = copy.deepcopy(llm_route)
        for wp in llm_route_for_saving: wp['is_llm'] = True

        if output_format == 'json':
            output_data = {'metadata': metadata, 'route': llm_route_for_saving}
            with open(output_file, 'w') as f: json.dump(output_data, f, indent=2)
        elif output_format == 'kml':
            # generate_kml in visualization.py will handle 'is_llm' flag
            generate_kml([llm_route_for_saving], output_file)
        else:
             logger.warning(f"Unsupported output format '{output_format}' for LLM pipeline. Saving as KML.")
             generate_kml([llm_route_for_saving], output_files['kml'])
             output_file = output_files['kml']

        result['output_file'] = output_file
        result['route'] = llm_route # Return original route

        # Step 4: Visualize if requested
        if visualize:
            if callback: callback(95, "Visualizing LLM flight path...")
            visualize_flight_paths([llm_route_for_saving]) # Pass flagged route

        if callback: callback(100, "LLM flight path generation complete.")
        result.update({'success': True, 'message': "LLM flight path generated successfully."})
        return result

    except Exception as e:
        import traceback
        error_message = f"Error in LLM flight path generation: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        result.update({'success': False, 'message': error_message})
        if callback: callback(0, error_message)
        return result

#-------------------------------------------------------------------------
# GUI Application - Modified from main(old).py
#-------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget."""
    # (Copied from main(old).py - assuming it exists there or is standard)
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class FlightGeneratorApp:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.airports_csv_path = os.path.join(self.root_dir, 'data', 'airports.csv')
        try:
            self.airports_data = load_airports(self.airports_csv_path)
            self.airports_data_list = [
                {"ident": code, "name": data.get("name", ""),
                 "iso_country": data.get("country", ""),
                 "continent": data.get("continent", "")}
                for code, data in self.airports_data.items()
            ]
            self.tree_data = self.build_hierarchy(self.airports_data_list)
        except Exception as e:
            logger.exception(f"Error reading airports CSV: {e}")
            messagebox.showerror("Initialization Error", f"Error reading airports CSV:\n{e}")
            sys.exit(1)

        self.root = tk.Tk()
        self.root.title("Flight Path Generator (Nominal, Synthetic, LLM)")
        self.root.geometry("1000x800") # Increased height for new options
        self.scrollable_frame = ScrollableFrame(self.root)
        self.scrollable_frame.pack(fill="both", expand=True)

        # Load aircraft options before creating widgets
        self.aircraft_options = self.load_aircraft_options()

        self.create_widgets()
        self.toggle_options() # Set initial UI state
        logger.info("FlightGeneratorApp initialized.")

    def load_aircraft_options(self):
        """Load aircraft options from the CSV file"""
        # (Copied from main(old).py)
        aircraft_options = ["DEFAULT"]
        aircraft_csv_path = os.path.join(self.root_dir, 'data', 'aircraft_data.csv')
        try:
            with open(aircraft_csv_path, 'r', encoding='utf-8') as file:
                import csv
                reader = csv.DictReader(file)
                families = set()
                for row in reader:
                    family = row.get('Aircraft_Family', '').strip()
                    if family: families.add(family)
            aircraft_options.extend(sorted(list(families)))
            logger.info(f"Loaded {len(aircraft_options)-1} aircraft families from CSV")
            return aircraft_options
        except Exception as e:
            logger.error(f"Error loading aircraft options: {e}")
            return ["A320", "A330", "B737", "B777", "A350", "A380", "DEFAULT"] # Fallback

    def create_widgets(self):
        # --- Row 0: Generation Mode ---
        mode_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Generation Mode", padding="10")
        mode_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.generation_mode = tk.StringVar(value="nominal")
        ttk.Radiobutton(mode_frame, text="Nominal", variable=self.generation_mode, value="nominal", command=self.toggle_options).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Multi-Synthetic", variable=self.generation_mode, value="multi", command=self.toggle_options).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="LLM", variable=self.generation_mode, value="llm", command=self.toggle_options).pack(side=tk.LEFT, padx=10)

        # --- Row 1: Aircraft Selection ---
        tk.Label(self.scrollable_frame.scrollable_frame, text="Aircraft Type:", font=("Arial", 12)).grid(row=1, column=0, padx=20, pady=10, sticky=tk.E)
        self.aircraft_var = tk.StringVar(value="A320") # Default value
        aircraft_combo = ttk.Combobox(self.scrollable_frame.scrollable_frame,
                                      textvariable=self.aircraft_var,
                                      values=self.aircraft_options, # Use loaded options
                                      font=("Arial", 12), state="readonly", width=25)
        aircraft_combo.grid(row=1, column=1, padx=20, pady=10, sticky=tk.W)
        if "A320" in self.aircraft_options: # Set default if available
             self.aircraft_var.set("A320")
        elif self.aircraft_options:
             self.aircraft_var.set(self.aircraft_options[0]) # Fallback to first option

        # --- Rows 2-6: Airport Selection ---
        tk.Label(self.scrollable_frame.scrollable_frame, text="Departure Airport:", font=("Arial", 12)).grid(row=2, column=0, padx=20, pady=5, sticky=tk.NE)
        self.departure_search_var = tk.StringVar()
        departure_search = tk.Entry(self.scrollable_frame.scrollable_frame, textvariable=self.departure_search_var, font=("Arial", 12))
        departure_search.grid(row=2, column=1, padx=20, pady=5, sticky=tk.W)
        departure_search.bind("<KeyRelease>", lambda event: self.update_treeview(self.departure_combo, self.departure_search_var.get()))
        self.departure_combo = ttk.Treeview(self.scrollable_frame.scrollable_frame, columns=("Airport",), show="tree", height=8) # Reduced height slightly
        self.departure_combo.grid(row=3, column=1, padx=20, pady=5, sticky=tk.W)
        self.populate_tree(self.departure_combo, self.tree_data)
        self.departure_combo.bind("<<TreeviewSelect>>", self.on_departure_select)

        tk.Label(self.scrollable_frame.scrollable_frame, text="Arrival Airport:", font=("Arial", 12)).grid(row=4, column=0, padx=20, pady=5, sticky=tk.NE)
        self.arrival_search_var = tk.StringVar()
        arrival_search = tk.Entry(self.scrollable_frame.scrollable_frame, textvariable=self.arrival_search_var, font=("Arial", 12))
        arrival_search.grid(row=4, column=1, padx=20, pady=5, sticky=tk.W)
        arrival_search.bind("<KeyRelease>", lambda event: self.update_treeview(self.arrival_combo, self.arrival_search_var.get()))
        self.arrival_combo = ttk.Treeview(self.scrollable_frame.scrollable_frame, columns=("Airport",), show="tree", height=8) # Reduced height slightly
        self.arrival_combo.grid(row=5, column=1, padx=20, pady=5, sticky=tk.W)
        self.populate_tree(self.arrival_combo, self.tree_data)
        self.arrival_combo.bind("<<TreeviewSelect>>", self.on_arrival_select)

        self.selected_departure_var = tk.StringVar(value="Selected Departure: None")
        self.selected_arrival_var = tk.StringVar(value="Selected Arrival: None")
        tk.Label(self.scrollable_frame.scrollable_frame, textvariable=self.selected_departure_var, font=("Arial", 11), fg="blue").grid(row=6, column=0, padx=20, pady=5, sticky=tk.W)
        tk.Label(self.scrollable_frame.scrollable_frame, textvariable=self.selected_arrival_var, font=("Arial", 11), fg="blue").grid(row=6, column=1, padx=20, pady=5, sticky=tk.W)

        # --- Row 7: Options Frame ---
        self.options_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Generation Options", padding=10)
        self.options_frame.grid(row=7, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        # Options Widgets (will be enabled/disabled by toggle_options)
        self.visualize_var = tk.BooleanVar(value=True)
        self.visualize_check = ttk.Checkbutton(self.options_frame, text="Visualize Output", variable=self.visualize_var)
        self.visualize_check.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.detail_label = tk.Label(self.options_frame, text="Detail Level:")
        self.detail_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.detail_var = tk.IntVar(value=2)
        self.detail_rb1 = ttk.Radiobutton(self.options_frame, text="Low", variable=self.detail_var, value=1)
        self.detail_rb1.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.detail_rb2 = ttk.Radiobutton(self.options_frame, text="Medium", variable=self.detail_var, value=2)
        self.detail_rb2.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.detail_rb3 = ttk.Radiobutton(self.options_frame, text="High", variable=self.detail_var, value=3)
        self.detail_rb3.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        self.output_label = tk.Label(self.options_frame, text="Output Format:")
        self.output_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_var = tk.StringVar(value="kml")
        self.output_rb_kml = ttk.Radiobutton(self.options_frame, text="KML", variable=self.output_var, value="kml")
        self.output_rb_kml.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        self.output_rb_json = ttk.Radiobutton(self.options_frame, text="JSON", variable=self.output_var, value="json")
        self.output_rb_json.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self.output_rb_csv = ttk.Radiobutton(self.options_frame, text="CSV", variable=self.output_var, value="csv")
        self.output_rb_csv.grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)

        self.route_count_label = tk.Label(self.options_frame, text="Number of Routes:")
        self.route_count_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.route_count_var = tk.IntVar(value=5) # Default for multi
        self.route_count_spinner = ttk.Spinbox(self.options_frame, from_=1, to=20, textvariable=self.route_count_var, width=5)
        self.route_count_spinner.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        self.variation_label = tk.Label(self.options_frame, text="Variation Level:")
        self.variation_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.variation_var = tk.DoubleVar(value=0.2)
        self.variation_slider = ttk.Scale(self.options_frame, from_=0.05, to=0.4, variable=self.variation_var, length=200)
        self.variation_slider.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.variation_val_label = tk.Label(self.options_frame, textvariable=self.variation_var)
        self.variation_val_label.grid(row=4, column=3, padx=5, pady=5, sticky=tk.W)

        self.strict_dev_label = tk.Label(self.options_frame, text="Strict Nominal Dev (km):")
        self.strict_dev_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.strict_dev_var = tk.DoubleVar(value=0.05)
        self.strict_dev_slider = ttk.Scale(self.options_frame, from_=0.01, to=0.1, variable=self.strict_dev_var, length=200)
        self.strict_dev_slider.grid(row=5, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.strict_dev_val_label = tk.Label(self.options_frame, textvariable=self.strict_dev_var)
        self.strict_dev_val_label.grid(row=5, column=3, padx=5, pady=5, sticky=tk.W)

        self.constraints_label = tk.Label(self.options_frame, text="LLM Constraints:")
        self.constraints_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.constraints_entry = ttk.Entry(self.options_frame, font=("Arial", 10), width=60)
        self.constraints_entry.grid(row=6, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        self.constraints_entry.insert(0, "Standard route")

        # --- Row 8: Generate Button ---
        self.generate_button = tk.Button(self.scrollable_frame.scrollable_frame, text="Generate Flight Path(s)",
                                       command=self.on_generate_flights, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", height=2)
        self.generate_button.grid(row=8, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # --- Row 9: Status Frame ---
        status_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Status", padding=10)
        status_frame.grid(row=9, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, padx=5, pady=5)

    def populate_tree(self, tree_widget, data):
        # (Copied from main(old).py)
        tree_widget.delete(*tree_widget.get_children())
        for cont, countries in sorted(data.items()):
            cont_node = tree_widget.insert("", "end", text=f"[{cont}]", open=False)
            for iso, airports in sorted(countries.items()):
                iso_node = tree_widget.insert(cont_node, "end", text=iso, open=False)
                for icao, name in sorted(airports.items(), key=lambda x: x[1]):
                    label = f"{icao} - {name}"
                    tree_widget.insert(iso_node, "end", text=label, open=False)

    def build_hierarchy(self, airports_data):
        # (Copied from main(old).py)
        hierarchy = {}
        for ap in airports_data:
            cont = ap.get("continent", "Unknown").strip() or "Unknown"
            iso = ap.get("iso_country", "UNK").strip().upper() or "UNK"
            icao = ap.get("ident", "Unknown").strip().upper() or "Unknown"
            name = ap.get("name", "Unknown").strip() or "Unknown"
            if cont not in hierarchy: hierarchy[cont] = {}
            if iso not in hierarchy[cont]: hierarchy[cont][iso] = {}
            hierarchy[cont][iso][icao] = name
        return hierarchy

    def update_treeview(self, tree_widget, search_term):
         # (Copied from main(old).py)
        search_term = search_term.strip().lower()
        if not search_term:
            self.populate_tree(tree_widget, self.tree_data)
            return
        filtered = {}
        for cont, countries in self.tree_data.items():
            for iso, airports in countries.items():
                for icao, name in airports.items():
                    if search_term in icao.lower() or search_term in name.lower():
                        if cont not in filtered: filtered[cont] = {}
                        if iso not in filtered[cont]: filtered[cont][iso] = {}
                        filtered[cont][iso][icao] = name
        tree_widget.delete(*tree_widget.get_children())
        if filtered: self.populate_tree(tree_widget, filtered)

    def get_selected_airport_icao(self, tree_widget):
        # (Copied from main(old).py)
        sel = tree_widget.selection()
        if not sel: return None
        text = tree_widget.item(sel[0], "text")
        if " - " in text: return text.split(" - ")[0].strip().upper()
        return None

    def on_departure_select(self, event):
        # (Copied from main(old).py)
        icao = self.get_selected_airport_icao(self.departure_combo)
        if icao:
            name = self.airports_data.get(icao, {}).get("name", "Unknown")
            self.selected_departure_var.set(f"Departure: {icao} - {name}")
        else:
            self.selected_departure_var.set("Selected Departure: None")

    def on_arrival_select(self, event):
        # (Copied from main(old).py)
        icao = self.get_selected_airport_icao(self.arrival_combo)
        if icao:
            name = self.airports_data.get(icao, {}).get("name", "Unknown")
            self.selected_arrival_var.set(f"Arrival: {icao} - {name}")
        else:
            self.selected_arrival_var.set("Selected Arrival: None")

    def toggle_options(self):
        """Enable/disable options based on the selected generation mode."""
        mode = self.generation_mode.get()
        is_nominal = mode == "nominal"
        is_multi = mode == "multi"
        is_llm = mode == "llm"

        # Helper to set state
        def set_state(widget, state):
             try: widget.config(state=state)
             except tk.TclError: pass # Ignore if widget doesn't support state

        # Detail Level (Nominal, Multi)
        detail_state = tk.NORMAL if is_nominal or is_multi else tk.DISABLED
        set_state(self.detail_label, detail_state)
        set_state(self.detail_rb1, detail_state)
        set_state(self.detail_rb2, detail_state)
        set_state(self.detail_rb3, detail_state)

        # Output Format (All, but CSV disabled for LLM)
        output_state = tk.NORMAL
        set_state(self.output_label, output_state)
        set_state(self.output_rb_kml, output_state)
        set_state(self.output_rb_json, output_state)
        set_state(self.output_rb_csv, tk.DISABLED if is_llm else tk.NORMAL)
        if is_llm and self.output_var.get() == "csv":
            self.output_var.set("kml") # Default to KML if CSV was selected

        # Route Count (Multi)
        route_count_state = tk.NORMAL if is_multi else tk.DISABLED
        set_state(self.route_count_label, route_count_state)
        set_state(self.route_count_spinner, route_count_state)

        # Variation Level (Multi)
        variation_state = tk.NORMAL if is_multi else tk.DISABLED
        set_state(self.variation_label, variation_state)
        set_state(self.variation_slider, variation_state)
        set_state(self.variation_val_label, variation_state)

        # Strict Nominal Deviation (Nominal, Multi - for comparison)
        strict_dev_state = tk.NORMAL if is_nominal or is_multi else tk.DISABLED
        set_state(self.strict_dev_label, strict_dev_state)
        set_state(self.strict_dev_slider, strict_dev_state)
        set_state(self.strict_dev_val_label, strict_dev_state)

        # LLM Constraints (LLM)
        llm_constraints_state = tk.NORMAL if is_llm else tk.DISABLED
        set_state(self.constraints_label, llm_constraints_state)
        set_state(self.constraints_entry, llm_constraints_state)

    def on_generate_flights(self):
        # Get selections
        dep_icao = self.get_selected_airport_icao(self.departure_combo)
        arr_icao = self.get_selected_airport_icao(self.arrival_combo)
        if not dep_icao or not arr_icao:
            messagebox.showerror("Selection Error", "Select both departure and arrival airports.")
            return
        if dep_icao == arr_icao:
            messagebox.showerror("Selection Error", "Departure and arrival must be different.")
            return

        # Get options based on mode
        mode = self.generation_mode.get()
        aircraft_ui = self.aircraft_var.get()
        visualize = self.visualize_var.get()
        output_format = self.output_var.get()

        # Map UI aircraft selection to actual model (from main(old).py)
        aircraft_mapping = {
            "A320": "A320-214", "A330": "A330-300", "B737": "B737-800",
            "B777": "B777-300ER", "A350": "A350-900", "A380": "A380-800",
            "DEFAULT": "DEFAULT" # Keep DEFAULT mapping
        }
        aircraft_model = aircraft_mapping.get(aircraft_ui, aircraft_ui) # Use mapping, fallback to UI value

        # Store strict nominal deviation globally (from main(old).py)
        global STRICT_NOMINAL_DEVIATION
        STRICT_NOMINAL_DEVIATION = self.strict_dev_var.get()

        # Callback for progress
        def update_callback(progress, message):
            self.progress_var.set(progress)
            self.status_var.set(message)
            self.root.update_idletasks()

        # Run processing in a thread
        def process_thread():
            try:
                result = None
                if mode == "nominal":
                    result = run_pipeline(
                        departure=dep_icao, arrival=arr_icao, aircraft=aircraft_model,
                        visualize=visualize, detail_level=self.detail_var.get(),
                        output_format=output_format, callback=update_callback
                    )
                elif mode == "multi":
                    result = run_multi_route_pipeline(
                        departure=dep_icao, arrival=arr_icao, aircraft=aircraft_model,
                        route_count=self.route_count_var.get(), max_variation=self.variation_var.get(),
                        visualize=visualize, detail_level=self.detail_var.get(),
                        output_format=output_format, callback=update_callback
                    )
                elif mode == "llm":
                    result = run_llm_pipeline(
                        departure=dep_icao, arrival=arr_icao, aircraft=aircraft_model,
                        constraints_text=self.constraints_entry.get(), visualize=visualize,
                        output_format=output_format, callback=update_callback
                    )

                # Handle completion
                if result and result['success']:
                    msg = result.get('message', 'Operation completed successfully!')
                    messagebox.showinfo("Success", msg)
                    # Visualization is handled within pipelines if visualize=True
                elif result:
                    messagebox.showerror("Error", f"Failed: {result.get('message', 'Unknown error')}")
                else:
                     messagebox.showerror("Error", "Processing failed to return a result.")

            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                 # Ensure status is updated even on error
                 if not (result and result['success']):
                      self.status_var.set("Operation failed or ended.")
                 else:
                      self.status_var.set("Operation complete.")
                 self.progress_var.set(0) # Reset progress bar


        # Reset progress and status before starting thread
        self.progress_var.set(0)
        self.status_var.set(f"Generating ({mode} mode)...")
        threading.Thread(target=process_thread, daemon=True).start()

    def on_visualise(self):
        # This button might be redundant if visualize=True in pipelines
        # Kept from main(old).py for potential separate use
        logger.warning("Manual 'Visualize Last Generated' button clicked - functionality may depend on pipeline state.")
        messagebox.showinfo("Visualize", "Attempting to launch visualization based on last generated files in 'visualization' folder.")
        # Simple approach: try opening the default viewer HTML
        viz_dir = os.path.join(self.root_dir, 'visualization')
        index_html = os.path.join(viz_dir, 'index.html')
        if os.path.exists(index_html):
             try:
                 # Try opening directly first
                 webbrowser.open(f'file://{os.path.abspath(index_html)}')
             except Exception as e:
                 logger.error(f"Failed to open visualization file directly: {e}")
                 messagebox.showerror("Visualization Error", f"Could not open visualization file:\n{e}")
        else:
             messagebox.showwarning("Visualize", "No 'index.html' found in visualization folder. Generate a path first.")


    def run(self):
        self.root.mainloop()

#-------------------------------------------------------------------------
# Main Functions - Modified from main(old).py
#-------------------------------------------------------------------------

def launch_ui():
    """Launch the graphical user interface."""
    logger.info("Launching Flight Path Generator UI")
    try:
        app = FlightGeneratorApp(project_root)
        app.run()
        return 0
    except Exception as e:
        logger.error(f"Error launching UI: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

def main():
    logger.info("Starting Flight Path Generator")

    parser = argparse.ArgumentParser(description='Flight Path Generator (Nominal, Synthetic, LLM)')
    parser.add_argument('--ui', action='store_true', help='Launch the graphical user interface')
    # CLI arguments made optional if --ui is used
    parser.add_argument('--departure', type=str, help='ICAO code of departure airport (required for CLI)')
    parser.add_argument('--arrival', type=str, help='ICAO code of arrival airport (required for CLI)')
    parser.add_argument('--aircraft', type=str, default='A320-214', help='Aircraft type (e.g., A320-214)')
    parser.add_argument('--mode', choices=['nominal', 'multi', 'llm'], default='nominal', help='Generation mode')
    # Arguments specific to modes
    parser.add_argument('--routes', type=int, default=5, help='Number of routes for multi mode (default: 5)')
    parser.add_argument('--variation', type=float, default=0.2, help='Max variation for multi mode (default: 0.2)')
    parser.add_argument('--constraints', type=str, default='Standard route', help='Constraints text for LLM mode')
    # General options
    parser.add_argument('--detail', type=int, default=2, choices=[1, 2, 3], help='Detail level for nominal/multi smoothing (1=Low, 2=Medium, 3=High)')
    parser.add_argument('--format', default="kml", choices=["kml", "json", "csv"], help="Output format (CSV N/A for LLM)")
    parser.add_argument('--no-visualize', action="store_true", help="Disable automatic visualization")

    args = parser.parse_args()

    # Launch UI if requested or no CLI args given
    if args.ui or len(sys.argv) == 1:
        return launch_ui()

    # Validate required args for CLI mode
    if not args.departure or not args.arrival:
        parser.error("--departure and --arrival are required for command-line mode (unless using --ui)")

    logger.info(f"Running in CLI mode: {args.mode}")
    logger.info(f"Departure: {args.departure}, Arrival: {args.arrival}, Aircraft: {args.aircraft}")

    visualize = not args.no_visualize
    result = None

    # Define simple console callback
    def console_callback(progress, message):
        logger.info(f"[{int(progress)}%] {message}")

    try:
        if args.mode == 'nominal':
            logger.info(f"Detail: {args.detail}, Format: {args.format}")
            result = run_pipeline(
                departure=args.departure, arrival=args.arrival, aircraft=args.aircraft,
                visualize=visualize, detail_level=args.detail, output_format=args.format,
                callback=console_callback
            )
        elif args.mode == 'multi':
            logger.info(f"Routes: {args.routes}, Variation: {args.variation}, Detail: {args.detail}, Format: {args.format}")
            result = run_multi_route_pipeline(
                departure=args.departure, arrival=args.arrival, aircraft=args.aircraft,
                route_count=args.routes, max_variation=args.variation, visualize=visualize,
                detail_level=args.detail, output_format=args.format, callback=console_callback
            )
        elif args.mode == 'llm':
            logger.info(f"Constraints: '{args.constraints}', Format: {args.format}")
            if args.format == 'csv':
                 logger.warning("CSV output format is not supported for LLM mode. Using KML instead.")
                 args.format = 'kml'
            result = run_llm_pipeline(
                departure=args.departure, arrival=args.arrival, aircraft=args.aircraft,
                constraints_text=args.constraints, visualize=visualize, output_format=args.format,
                callback=console_callback
            )
        else:
             # Should not happen due to choices constraint
             logger.error(f"Invalid mode specified: {args.mode}")
             return 1

        if result and result['success']:
            logger.info(f"Operation completed successfully. Output(s): {result.get('output_file') or result.get('output_files', 'N/A')}")
            return 0
        else:
            logger.error(f"Operation failed: {result.get('message', 'Unknown error')}")
            return 1

    except Exception as e:
         logger.error(f"An unexpected error occurred during CLI execution: {e}")
         import traceback
         logger.error(traceback.format_exc())
         return 1


# Global variable for strict nominal deviation (used by GUI and visualize_with_triple_comparison)
STRICT_NOMINAL_DEVIATION = 0.05

if __name__ == "__main__":
    sys.exit(main())