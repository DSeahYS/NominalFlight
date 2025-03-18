# Final Year Project\8. New Airbus Synthetic Flight Generator (Nominal)\main.py

import argparse
import os
import sys
import logging
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from http.server import SimpleHTTPRequestHandler
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
from src.visualization import visualize_flight_paths, save_kml, save_combined_kml, generate_kml

#-------------------------------------------------------------------------
# Pipeline Functions
#-------------------------------------------------------------------------

def get_nominal_pattern(departure, arrival, nominal_patterns):
    """
    Get the original nominal pattern without any variations.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        nominal_patterns: Dictionary of nominal patterns
        
    Returns:
        List of waypoints from the nominal pattern
    """
    route_key = f"{departure}-{arrival}"
    
    if route_key in nominal_patterns:
        pattern = nominal_patterns[route_key]
        return pattern.get('waypoints', [])
    
    # Check reverse route
    reverse_key = f"{arrival}-{departure}"
    if reverse_key in nominal_patterns:
        pattern = nominal_patterns[reverse_key]
        waypoints = pattern.get('waypoints', [])
        # Reverse the waypoints for the opposite direction
        reversed_waypoints = list(reversed(waypoints))
        # Adjust headings if present
        for wp in reversed_waypoints:
            if 'heading' in wp and wp['heading'] is not None:
                wp['heading'] = (wp['heading'] + 180) % 360
        return reversed_waypoints
    
    return []

def calculate_route_length(route):
    """
    Calculate the total length of a route in kilometers.
    """
    if len(route) < 2:
        return 0
        
    total_distance = 0
    for i in range(1, len(route)):
        try:
            distance = great_circle(
                (route[i-1]['latitude'], route[i-1]['longitude']),
                (route[i]['latitude'], route[i]['longitude'])
            ).kilometers
            total_distance += distance
        except Exception as e:
            logger.warning(f"Error calculating distance between waypoints: {e}")
    
    return total_distance

def visualize_with_triple_comparison(synthetic_routes, nominal_route):
    """
    Visualize nominal, strictly nominal, and synthetic routes together for comparison.
    
    Args:
        synthetic_routes: List of synthetic routes to visualize
        nominal_route: The nominal route for comparison
    """
    # Flag the nominal route
    nominal_route_with_flag = copy.deepcopy(nominal_route)
    for wp in nominal_route_with_flag:
        wp['is_nominal'] = True
    
    # Create the strictly nominal route with minimal variations
    strict_nominal = create_strictly_nominal_route(nominal_route, fixed_deviation=0.05)
    
    # Combine all routes with nominal and strict nominal first
    all_routes = [nominal_route_with_flag, strict_nominal] + synthetic_routes
    
    # Call regular visualization
    visualize_flight_paths(all_routes)

def run_pipeline(departure, arrival, aircraft="A320-214", visualize=True, 
                detail_level=2, output_format="kml", callback=None):
    """
    Run the nominal flight path generation pipeline with progress reporting.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        aircraft: Aircraft type
        visualize: Whether to visualize the output
        detail_level: Detail level (1=low, 2=medium, 3=high)
        output_format: Output format (kml, json, csv)
        callback: Callback function for progress updates: callback(progress, message)
        
    Returns:
        Dictionary with results of the flight path generation
    """
    result = {
        'success': False,
        'message': '',
        'route': None,
        'metadata': {},
        'output_file': None
    }
    
    try:
        # Report progress
        if callback:
            callback(0, "Starting flight path generation...")
        
        # Step 1: Load core data
        if callback:
            callback(10, "Loading airport data...")
            
        airports_file = os.path.join(project_root, 'data', 'airports.csv')
        aip_dir = os.path.join(project_root, 'data', 'aip')
        
        # Check if these files/directories exist
        if not os.path.exists(airports_file):
            raise FileNotFoundError(f"Airports data file not found: {airports_file}")
        if not os.path.exists(aip_dir):
            raise FileNotFoundError(f"AIP directory not found: {aip_dir}")
        
        # Load airports and AIP data
        airports = load_airports(airports_file)
        
        if callback:
            callback(20, "Loading AIP data...")
            
        aip_data = load_aip_data(aip_dir)
        
        if not airports:
            raise Exception("Failed to load airports data")
            
        if not aip_data:
            raise Exception("Failed to load AIP data")
        
        # Step 2: Load nominal patterns
        if callback:
            callback(40, "Loading nominal patterns...")
            
        nominal_patterns_file = os.path.join(project_root, 'data', 'nominal', 'nominal_patterns.json')
        
        try:
            nominal_patterns = load_nominal_patterns(nominal_patterns_file)
            routes = list(nominal_patterns.keys())
            
            if callback:
                callback(50, f"Loaded patterns for {len(routes)} routes")
        except Exception as e:
            logger.error(f"Error loading nominal patterns: {e}")
            # Create an empty nominal patterns structure
            nominal_patterns = {
                "WSSS-WMKK": {"waypoints": []},
                "WMKK-WSSS": {"waypoints": []}
            }
            
            if callback:
                callback(50, "Using empty nominal patterns")
        
        # Step 3: Get the nominal pattern for reference
        nominal_route = get_nominal_pattern(departure, arrival, nominal_patterns)
        
        # Step 4: Create aircraft performance model
        if callback:
            callback(60, f"Initializing aircraft performance model for {aircraft}...")
            
        aircraft_performance = AircraftPerformance(aircraft)
        aircraft_info = aircraft_performance.get_aircraft_info()
        
        if callback:
            callback(65, f"Using aircraft: {aircraft_info['type']}")
        
        # Step 5: Generate nominal route
        if callback:
            callback(70, f"Generating nominal route from {departure} to {arrival}...")
        
        # Verify route compatibility
        route_key = f"{departure}-{arrival}"
        inverse_route_key = f"{arrival}-{departure}"
        
        if route_key not in nominal_patterns and inverse_route_key not in nominal_patterns:
            if callback:
                callback(75, f"No nominal pattern found for {route_key}, will use direct routing")
        
        # Construct route using AIP data and nominal patterns
        route = construct_nominal_route(departure, arrival, aip_data, nominal_patterns)
        
        if not route:
            raise Exception("Failed to generate route. Check airport codes and AIP data.")
            
        # Calculate total route length
        route_length = calculate_route_length(route)
        
        if callback:
            callback(80, f"Route generated with {len(route)} waypoints and {route_length:.1f} km length")
        
        # Apply aircraft performance profile
        route = aircraft_performance.apply_performance_profile(route, route_length)
        
        # Apply aircraft performance to nominal route too if it exists
        if nominal_route:
            nominal_route = aircraft_performance.apply_performance_profile(
                nominal_route, 
                calculate_route_length(nominal_route)
            )
        
        # Step 6: Apply route smoothing with Kalman filter
        if callback:
            callback(85, "Applying advanced Kalman smoothing...")
            
        # Adjust smoothing parameters based on detail level
        smoothing_params = {
            1: {'points': 50},
            2: {'points': 100},
            3: {'points': 200}
        }
        
        params = smoothing_params.get(detail_level, smoothing_params[2])
        
        route_smoother = NominalRouteSmoother(aircraft_performance)
        
        # Apply smoothing
        try:
            smoothed_route = route_smoother.smooth_route(
                route, 
                points=params['points'],
                variant_id=0,
                micro_var_level=0.03  # Reduced micro-variation for nominal route
            )
        except TypeError as e:
            # If the method signature is different, try the alternative form
            logger.warning(f"Error with default smoother parameters: {e}")
            smoothed_route = route_smoother.smooth_route(route)
        
        if callback:
            callback(90, f"Smoothed route generated with {len(smoothed_route)} waypoints")
        
        # Step 7: Save and/or visualize
        # Create output directory if it doesn't exist
        output_dir = os.path.join(project_root, 'visualization')
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output file based on format
        output_files = {
            'json': os.path.join(output_dir, f'flight_path_{departure}_{arrival}.json'),
            'kml': os.path.join(output_dir, f'flight_path_{departure}_{arrival}.kml'),
            'csv': os.path.join(output_dir, f'flight_path_{departure}_{arrival}.csv')
        }
        output_file = output_files.get(output_format, output_files['json'])
        
        # Create output structure with metadata
        output = {
            'metadata': {
                'departure': departure,
                'arrival': arrival,
                'aircraft': aircraft,
                'distance_km': route_length,
                'waypoint_count': len(smoothed_route),
                'cruise_altitude': aircraft_info['cruise_altitude'],
                'cruise_speed': aircraft_info['cruise_speed']
            },
            'route': smoothed_route
        }
        
        # Save based on format
        if output_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
        elif output_format == 'kml':
            save_kml(smoothed_route, output_file, f"{departure} to {arrival}")
        elif output_format == 'csv':
            with open(output_file, 'w') as f:
                f.write("latitude,longitude,altitude,speed,heading\n")
                for wp in smoothed_route:
                    f.write(f"{wp['latitude']},{wp['longitude']},{wp['altitude']},{wp['speed']},{wp.get('heading', '')}\n")
        
        if callback:
            callback(95, f"Route data saved to {output_file}")
        
        # Visualize if requested
        if visualize:
            if callback:
                callback(98, "Visualizing flight path...")
                
            if nominal_route:
                visualize_with_triple_comparison([smoothed_route], nominal_route)
            else:
                visualize_flight_paths([smoothed_route])
        
        # Finalize
        if callback:
            callback(100, "Flight path generation complete")
        
        # Prepare result
        result['success'] = True
        result['message'] = "Flight path generation completed successfully"
        result['route'] = smoothed_route
        result['metadata'] = {
            'departure': departure,
            'arrival': arrival,
            'aircraft': aircraft,
            'distance_km': route_length,
            'waypoint_count': len(smoothed_route)
        }
        result['output_file'] = output_file
        
        return result
    
    except Exception as e:
        import traceback
        error_message = f"Error in flight path generation: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        result['success'] = False
        result['message'] = error_message
        
        if callback:
            callback(0, error_message)
            
        return result

def run_multi_route_pipeline(departure, arrival, aircraft="A320-214", route_count=5, 
                           max_variation=0.3, visualize=True, detail_level=2, 
                           output_format="kml", callback=None):
    """
    Generate multiple synthetic flight paths between two airports.
    
    Args:
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        aircraft: Aircraft type
        route_count: Number of different routes to generate
        max_variation: Maximum variation level (0-1)
        visualize: Whether to visualize the output
        detail_level: Detail level (1=low, 2=medium, 3=high)
        output_format: Output format (kml, json, csv)
        callback: Callback function for progress updates
        
    Returns:
        Dictionary with results
    """
    result = {
        'success': False,
        'message': '',
        'routes': [],
        'metadata': {},
        'output_files': []
    }
    
    try:
        # Report progress
        if callback:
            callback(0, f"Starting generation of {route_count} flight paths...")
        
        # Step 1: Load necessary data
        if callback:
            callback(10, "Loading data...")
            
        # Load airports and AIP data
        airports_file = os.path.join(project_root, 'data', 'airports.csv')
        aip_dir = os.path.join(project_root, 'data', 'aip')
        
        airports = load_airports(airports_file)
        aip_data = load_aip_data(aip_dir)
        
        # Step 2: Load nominal patterns
        if callback:
            callback(20, "Loading nominal patterns...")
            
        nominal_patterns_file = os.path.join(project_root, 'data', 'nominal', 'nominal_patterns.json')
        nominal_patterns = load_nominal_patterns(nominal_patterns_file)
        
        # Step 3: Get the pure nominal pattern for reference
        if callback:
            callback(30, "Extracting nominal pattern for reference...")
            
        nominal_route = get_nominal_pattern(departure, arrival, nominal_patterns)
        
        # Step 4: Create aircraft performance model
        if callback:
            callback(35, f"Initializing aircraft performance model...")
            
        aircraft_performance = AircraftPerformance(aircraft)
        
        # Apply aircraft performance profile to nominal route
        if nominal_route:
            nominal_route = aircraft_performance.apply_performance_profile(
                nominal_route, 
                calculate_route_length(nominal_route)
            )
        
        # Step 5: Generate multiple routes
        if callback:
            callback(40, f"Generating {route_count} routes from {departure} to {arrival}...")
        
        # Reduced max variation to prevent unrealistic paths
        actual_variation = min(max_variation, 0.25)  # Changed from 0.4 to 0.25
        
        routes = generate_synthetic_flights(
            departure=departure,
            arrival=arrival,
            aip_data=aip_data,
            nominal_patterns=nominal_patterns,
            count=route_count,
            max_variation=actual_variation
        )
        
        if not routes:
            raise Exception(f"Failed to generate routes between {departure} and {arrival}")
            
        # Step 6: Smooth each route with Kalman filter
        if callback:
            callback(60, "Applying advanced Kalman smoothing...")
            
        smoother = NominalRouteSmoother(aircraft_performance)
        smoothed_routes = []
        
        for i, route in enumerate(routes):
            if callback:
                progress = 60 + (30 * (i+1) / len(routes))
                callback(progress, f"Smoothing route {i+1}/{len(routes)}...")
                
            # Apply smoothing with variant-specific parameters
            smoothed_route = smoother.smooth_route(
                route, 
                points=100 * detail_level // 2, 
                variant_id=i,
                micro_var_level=0.03  # Reduced micro-variation for smoother paths
            )
            smoothed_routes.append(smoothed_route)
        
        # Step 7: Save outputs and visualize
        if callback:
            callback(90, "Saving route data...")
            
        output_dir = os.path.join(project_root, 'visualization')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a combined KML file with nominal and all synthetic routes
        combined_kml_file = os.path.join(output_dir, f'flightpaths_{departure}_{arrival}.kml')
        
        # Save the combined KML if the nominal route exists
        if nominal_route:
            # Create strictly nominal route for comparison
            strict_nominal = create_strictly_nominal_route(nominal_route, fixed_deviation=0.05)
            
            # Flag nominal route
            nominal_route_with_flag = copy.deepcopy(nominal_route)
            for wp in nominal_route_with_flag:
                wp['is_nominal'] = True
                
            # Combine all routes for KML
            all_routes = [nominal_route_with_flag, strict_nominal] + smoothed_routes
            
            # Generate KML with all routes - fixed to handle the list of routes correctly
            generate_kml(all_routes, combined_kml_file)
            result['output_files'].append(combined_kml_file)
        else:
            # Save individual files if no nominal route
            output_files = []
            for i, route in enumerate(smoothed_routes):
                # Create output file name with index
                output_file = os.path.join(output_dir, f'flight_path_{departure}_{arrival}_{i+1}.{output_format}')
                
                if output_format == 'json':
                    with open(output_file, 'w') as f:
                        json.dump({
                            'metadata': {
                                'departure': departure,
                                'arrival': arrival,
                                'aircraft': aircraft,
                                'route_number': i+1,
                                'total_routes': route_count
                            },
                            'route': route
                        }, f, indent=2)
                elif output_format == 'kml':
                    save_kml(route, output_file, f"Route {i+1}: {departure} to {arrival}")
                elif output_format == 'csv':
                    with open(output_file, 'w') as f:
                        f.write("latitude,longitude,altitude,speed,heading\n")
                        for wp in route:
                            f.write(f"{wp['latitude']},{wp['longitude']},{wp['altitude']},{wp['speed']},{wp.get('heading', '')}\n")
                            
                output_files.append(output_file)
            
            result['output_files'] = output_files
        
        # Visualize all routes if requested
        if visualize:
            if callback:
                callback(95, "Visualizing flight paths...")
                
            if nominal_route:
                visualize_with_triple_comparison(smoothed_routes, nominal_route)
            else:
                visualize_flight_paths(smoothed_routes)
        
        # Complete
        if callback:
            callback(100, f"Successfully generated {route_count} flight paths")
        
        result['success'] = True
        result['message'] = f"Successfully generated {route_count} flight paths with Kalman smoothing"
        result['routes'] = smoothed_routes
        
        return result
    
    except Exception as e:
        error_message = f"Error generating multiple flight paths: {str(e)}"
        logger.error(error_message)
        import traceback
        logger.error(traceback.format_exc())
        
        result['success'] = False
        result['message'] = error_message
        
        if callback:
            callback(0, error_message)
            
        return result

#-------------------------------------------------------------------------
# GUI Classes
#-------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#f0f0f0")
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(self.canvas, padding=(10,10,10,10))
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Enter>", self._bind_mousewheel)
        self.scrollable_frame.bind("<Leave>", self._unbind_mousewheel)
        
    def _on_mousewheel(self, event):
        delta = int(-1*(event.delta/120)) if event.delta else 0
        self.canvas.yview_scroll(delta, "units")
        
    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

class FlightGeneratorApp:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.airports_csv_path = os.path.join(self.root_dir, 'data', 'airports.csv')
        
        try:
            # Load airports data
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
        self.root.title("Flight Path Generator with Triple Comparison")
        self.root.geometry("1000x700")
        self.scrollable_frame = ScrollableFrame(self.root)
        self.scrollable_frame.pack(fill="both", expand=True)
        self.create_widgets()
        logger.info("FlightGeneratorApp initialized.")
    
    def load_aircraft_options(self):
        """Load aircraft options from the CSV file"""
        aircraft_options = ["DEFAULT"]  # Always include DEFAULT as a fallback
        aircraft_csv_path = os.path.join(self.root_dir, 'data', 'aircraft_data.csv')
        
        try:
            with open(aircraft_csv_path, 'r', encoding='utf-8') as file:
                import csv
                reader = csv.DictReader(file)
                for row in reader:
                    # Create a display name combining family and model
                    aircraft_family = row.get('Aircraft_Family', '')
                    if aircraft_family and aircraft_family not in aircraft_options:
                        aircraft_options.append(aircraft_family)
            
            logger.info(f"Loaded {len(aircraft_options)-1} aircraft types from CSV")
            return sorted(aircraft_options)
        except Exception as e:
            logger.error(f"Error loading aircraft options: {e}")
            return ["A320", "B737", "B777", "A350", "A380", "DEFAULT"]  # Fallback to defaults
        
    def create_widgets(self):
        # Aircraft selection
        tk.Label(self.scrollable_frame.scrollable_frame, text="Aircraft Type:", font=("Arial", 12))\
            .grid(row=0, column=0, padx=20, pady=10, sticky=tk.E)
            
        self.aircraft_var = tk.StringVar(value="A320")
        aircraft_options = self.load_aircraft_options()
        aircraft_combo = ttk.Combobox(self.scrollable_frame.scrollable_frame, 
                                      textvariable=self.aircraft_var, 
                                      values=aircraft_options,
                                      font=("Arial", 12))
        aircraft_combo.grid(row=0, column=1, padx=20, pady=10, sticky=tk.W)
        
        # Departure airport selection
        tk.Label(self.scrollable_frame.scrollable_frame, text="Select Departure Airport:", font=("Arial", 12))\
            .grid(row=1, column=0, padx=20, pady=5, sticky=tk.NE)
            
        self.departure_search_var = tk.StringVar()
        departure_search = tk.Entry(self.scrollable_frame.scrollable_frame, textvariable=self.departure_search_var, font=("Arial", 12))
        departure_search.grid(row=1, column=1, padx=20, pady=5, sticky=tk.W)
        departure_search.bind("<KeyRelease>", lambda event: self.update_treeview(self.departure_combo, self.departure_search_var.get()))
        
        self.departure_combo = ttk.Treeview(self.scrollable_frame.scrollable_frame, columns=("Airport",), show="tree", height=10)
        self.departure_combo.grid(row=2, column=1, padx=20, pady=5, sticky=tk.W)
        self.populate_tree(self.departure_combo, self.tree_data)
        self.departure_combo.bind("<<TreeviewSelect>>", self.on_departure_select)
        
        # Arrival airport selection
        tk.Label(self.scrollable_frame.scrollable_frame, text="Select Arrival Airport:", font=("Arial", 12))\
            .grid(row=3, column=0, padx=20, pady=5, sticky=tk.NE)
            
        self.arrival_search_var = tk.StringVar()
        arrival_search = tk.Entry(self.scrollable_frame.scrollable_frame, textvariable=self.arrival_search_var, font=("Arial", 12))
        arrival_search.grid(row=3, column=1, padx=20, pady=5, sticky=tk.W)
        arrival_search.bind("<KeyRelease>", lambda event: self.update_treeview(self.arrival_combo, self.arrival_search_var.get()))
        
        self.arrival_combo = ttk.Treeview(self.scrollable_frame.scrollable_frame, columns=("Airport",), show="tree", height=10)
        self.arrival_combo.grid(row=4, column=1, padx=20, pady=5, sticky=tk.W)
        self.populate_tree(self.arrival_combo, self.tree_data)
        self.arrival_combo.bind("<<TreeviewSelect>>", self.on_arrival_select)
        
        # Selected airports display
        self.selected_departure_var = tk.StringVar(value="Selected Departure: None")
        self.selected_arrival_var = tk.StringVar(value="Selected Arrival: None")
        
        tk.Label(self.scrollable_frame.scrollable_frame, textvariable=self.selected_departure_var, font=("Arial", 12), fg="green")\
            .grid(row=5, column=1, padx=20, pady=5, sticky=tk.W)
        tk.Label(self.scrollable_frame.scrollable_frame, textvariable=self.selected_arrival_var, font=("Arial", 12), fg="green")\
            .grid(row=6, column=1, padx=20, pady=5, sticky=tk.W)
        
        # Options frame
        options_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Advanced Options", padding=10)
        options_frame.grid(row=7, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        # Visualization option
        self.visualize_var = tk.BooleanVar(value=True)
        visualize_check = ttk.Checkbutton(options_frame, text="Visualize generated flight path", variable=self.visualize_var)
        visualize_check.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Detail level option
        tk.Label(options_frame, text="Detail Level:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.detail_var = tk.IntVar(value=2)
        ttk.Radiobutton(options_frame, text="Low", variable=self.detail_var, value=1).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="Medium", variable=self.detail_var, value=2).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="High", variable=self.detail_var, value=3).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Output format option
        tk.Label(options_frame, text="Output Format:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_var = tk.StringVar(value="kml")
        ttk.Radiobutton(options_frame, text="KML", variable=self.output_var, value="kml").grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="JSON", variable=self.output_var, value="json").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="CSV", variable=self.output_var, value="csv").grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Add number of routes option
        tk.Label(options_frame, text="Number of Routes:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.route_count_var = tk.IntVar(value=1)
        route_count_spinner = ttk.Spinbox(options_frame, from_=1, to=20, textvariable=self.route_count_var, width=5)
        route_count_spinner.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Variation level
        tk.Label(options_frame, text="Variation Level:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.variation_var = tk.DoubleVar(value=0.2)
        variation_slider = ttk.Scale(options_frame, from_=0.05, to=0.4, variable=self.variation_var, length=200)
        variation_slider.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        tk.Label(options_frame, textvariable=self.variation_var).grid(row=4, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Strictly nominal deviation
        tk.Label(options_frame, text="Strict Nominal Deviation (km):").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.strict_dev_var = tk.DoubleVar(value=0.05)
        strict_dev_slider = ttk.Scale(options_frame, from_=0.01, to=0.1, variable=self.strict_dev_var, length=200)
        strict_dev_slider.grid(row=5, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        tk.Label(options_frame, textvariable=self.strict_dev_var).grid(row=5, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Generate and visualize buttons
        self.generate_button = tk.Button(self.scrollable_frame.scrollable_frame, text="Generate Flight Path",
                                       command=self.on_generate_flights, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.generate_button.grid(row=8, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        
        self.visualise_button = tk.Button(self.scrollable_frame.scrollable_frame, text="Visualize Last Generated",
                                      command=self.on_visualise, font=("Arial", 12), bg="#2196F3", fg="white")
        self.visualise_button.grid(row=9, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        # Status frame
        status_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Status", padding=10)
        status_frame.grid(row=10, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready to generate flight paths")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, padx=5, pady=5)
    
    def populate_tree(self, tree_widget, data):
        tree_widget.delete(*tree_widget.get_children())
        for cont, countries in sorted(data.items()):
            cont_node = tree_widget.insert("", "end", text=f"[{cont}]", open=False)
            for iso, airports in sorted(countries.items()):
                iso_node = tree_widget.insert(cont_node, "end", text=iso, open=False)
                for icao, name in sorted(airports.items(), key=lambda x: x[1]):
                    label = f"{icao} - {name}"
                    tree_widget.insert(iso_node, "end", text=label, open=False)
    
    def build_hierarchy(self, airports_data):
        hierarchy = {}
        for ap in airports_data:
            cont = ap.get("continent", "Unknown").strip() or "Unknown"
            iso = ap.get("iso_country", "UNK").strip().upper() or "UNK"
            icao = ap.get("ident", "Unknown").strip().upper() or "Unknown"
            name = ap.get("name", "Unknown").strip() or "Unknown"
            if cont not in hierarchy:
                hierarchy[cont] = {}
            if iso not in hierarchy[cont]:
                hierarchy[cont][iso] = {}
            hierarchy[cont][iso][icao] = name
        return hierarchy
    
    def update_treeview(self, tree_widget, search_term):
        search_term = search_term.strip().lower()
        if not search_term:
            self.populate_tree(tree_widget, self.tree_data)
            return
        filtered = {}
        for cont, countries in self.tree_data.items():
            for iso, airports in countries.items():
                for icao, name in airports.items():
                    if search_term in icao.lower() or search_term in name.lower():
                        if cont not in filtered:
                            filtered[cont] = {}
                        if iso not in filtered[cont]:
                            filtered[cont][iso] = {}
                        filtered[cont][iso][icao] = name
        tree_widget.delete(*tree_widget.get_children())
        if filtered:
            self.populate_tree(tree_widget, filtered)
    
    def get_selected_airport_icao(self, tree_widget):
        sel = tree_widget.selection()
        if not sel:
            return None
        text = tree_widget.item(sel[0], "text")
        if " - " in text:
            return text.split(" - ")[0].strip().upper()
        return None
    
    def on_departure_select(self, event):
        icao = self.get_selected_airport_icao(self.departure_combo)
        if icao:
            name = self.airports_data.get(icao, {}).get("name", "Unknown")
            self.selected_departure_var.set(f"Selected Departure: {icao} - {name}")
        else:
            self.selected_departure_var.set("Selected Departure: None")
    
    def on_arrival_select(self, event):
        icao = self.get_selected_airport_icao(self.arrival_combo)
        if icao:
            name = self.airports_data.get(icao, {}).get("name", "Unknown")
            self.selected_arrival_var.set(f"Selected Arrival: {icao} - {name}")
        else:
            self.selected_arrival_var.set("Selected Arrival: None")
    
    def on_generate_flights(self):
        # Get selections
        dep_icao = self.get_selected_airport_icao(self.departure_combo)
        arr_icao = self.get_selected_airport_icao(self.arrival_combo)
        
        # Validate selections
        if not dep_icao or not arr_icao:
            messagebox.showerror("Selection Error", "Select both departure and arrival airports.")
            return
            
        if dep_icao == arr_icao:
            messagebox.showerror("Selection Error", "Departure and arrival must be different.")
            return
        
        # Get options
        aircraft = self.aircraft_var.get()
        visualize = self.visualize_var.get()
        detail_level = self.detail_var.get()
        output_format = self.output_var.get()
        route_count = self.route_count_var.get()
        max_variation = self.variation_var.get()
        strict_nominal_dev = self.strict_dev_var.get()
        
        # Store the strict nominal deviation for use in visualization
        global STRICT_NOMINAL_DEVIATION
        STRICT_NOMINAL_DEVIATION = strict_nominal_dev
        
        # Create callback for progress updates
        def update_callback(progress, message):
            self.progress_var.set(progress)
            self.status_var.set(message)
            self.root.update_idletasks()  # Ensure UI updates during processing
        
        # Run processing in a separate thread to keep UI responsive
        def process_thread():
            try:
                # Map UI aircraft selection to actual aircraft model 
                aircraft_mapping = {
                    "A320": "A320-214",
                    "A330": "A330-300",
                    "B737": "B737-800",
                    "B777": "B777-300ER",
                    "A350": "A350-900",
                    "A380": "A380-800",
                    "DEFAULT": "DEFAULT"
                }
                
                full_aircraft_model = aircraft_mapping.get(aircraft, aircraft)
                
                # Use multi-route pipeline if route_count > 1
                if route_count > 1:
                    result = run_multi_route_pipeline(
                        departure=dep_icao,
                        arrival=arr_icao,
                        aircraft=full_aircraft_model,
                        route_count=route_count,
                        max_variation=max_variation,
                        visualize=visualize,
                        detail_level=detail_level,
                        output_format=output_format,
                        callback=update_callback
                    )
                else:
                    # Use single route pipeline
                    result = run_pipeline(
                        departure=dep_icao,
                        arrival=arr_icao,
                        aircraft=full_aircraft_model,
                        visualize=visualize,
                        detail_level=detail_level,
                        output_format=output_format,
                        callback=update_callback
                    )
                
                # Handle completion
                if result['success']:
                    if route_count > 1:
                        messagebox.showinfo("Success", f"Generated {route_count} flight paths successfully!")
                    else:
                        messagebox.showinfo("Success", "Flight path generated successfully!")
                    
                    if visualize:
                        self.on_visualise()
                else:
                    messagebox.showerror("Error", f"Failed to generate flight path: {result['message']}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Reset progress and status
        self.progress_var.set(0)
        self.status_var.set("Generating flight path...")
        
        # Start processing thread
        threading.Thread(target=process_thread, daemon=True).start()
    
    def on_visualise(self):
        # Try to use the built-in visualization module
        try:
            # This will use your src.visualization.visualize_flight_paths
            messagebox.showinfo("Visualization", "Launching visualization. Please wait...")
            
            # If a server-based visualization is needed:
            viz_dir = os.path.join(self.root_dir, 'visualization')
            index_html = os.path.join(viz_dir, 'index.html')
            
            if os.path.exists(index_html):
                os.chdir(viz_dir)
                port = 8000
                
                try:
                    server_address = ("", port)
                    httpd = TCPServer(server_address, SimpleHTTPRequestHandler)
                    
                    url = f"http://localhost:{port}/index.html"
                    webbrowser.open_new_tab(url)
                    
                    server_thread = threading.Thread(target=httpd.serve_forever)
                    server_thread.daemon = True
                    server_thread.start()
                    
                    messagebox.showinfo("Server Started", f"Visualization is running at {url}")
                except Exception as e:
                    messagebox.showerror("Server Error", f"Could not start server:\n{e}")
            else:
                # Use direct visualization function if index.html doesn't exist
                from src.visualization import visualize_flight_paths
                messagebox.showinfo("Visualization", "Opening visualization directly.")
                visualize_flight_paths()
                
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error launching visualization: {e}")
    
    def run(self):
        self.root.mainloop()

#-------------------------------------------------------------------------
# Main Functions
#-------------------------------------------------------------------------

def launch_ui():
    """
    Launch the graphical user interface.
    """
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
    logger.info("Starting Flight Path Generator with Triple Comparison")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flight Path Generator with Triple Comparison')
    parser.add_argument('--ui', action='store_true', help='Launch the graphical user interface')
    parser.add_argument('--departure', type=str, help='ICAO code of departure airport')
    parser.add_argument('--arrival', type=str, help='ICAO code of arrival airport')
    parser.add_argument('--aircraft', type=str, default='A320-214', 
                        help='Aircraft type (e.g., A320-214, A330-300)')
    parser.add_argument('--routes', type=int, default=1,
                        help='Number of flight paths to generate (1-20)')
    parser.add_argument('--visualize', action='store_true', help='Visualize the generated flight path')
    parser.add_argument('--output', type=str, default='flight_path.json',
                        help='Output file for flight path data')
    args = parser.parse_args()
    
    # If no command-line arguments are provided or --ui is explicitly specified,
    # launch the graphical user interface
    if len(sys.argv) == 1 or args.ui:
        return launch_ui()
    
    # If departure and arrival are not specified, show help
    if not args.departure or not args.arrival:
        parser.print_help()
        logger.error("Departure and arrival airports are required for command-line mode")
        return 1
    
    # Define progress reporting callback
    def console_callback(progress, message):
        logger.info(f"[{progress}%] {message}")
    
    # Run appropriate pipeline based on number of routes
    if args.routes > 1:
        result = run_multi_route_pipeline(
            departure=args.departure,
            arrival=args.arrival,
            aircraft=args.aircraft,
            route_count=args.routes,
            max_variation=0.2,  # Reduced variation for more realistic paths
            visualize=args.visualize,
            output_format=args.output.split('.')[-1],
            callback=console_callback
        )
    else:
        result = run_pipeline(
            departure=args.departure,
            arrival=args.arrival,
            aircraft=args.aircraft,
            visualize=args.visualize,
            output_format=args.output.split('.')[-1],
            callback=console_callback
        )
    
    if result['success']:
        logger.info("Flight Path Generator completed successfully")
    else:
        logger.error(f"Flight Path Generator failed: {result['message']}")
        return 1
    
    return 0

# Store strict nominal deviation setting
STRICT_NOMINAL_DEVIATION = 0.05

if __name__ == "__main__":
    sys.exit(main())
