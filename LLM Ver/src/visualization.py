# F\src\visualization.py
# Modified to handle LLM KML generation via templating

import simplekml
# from jinja2 import Environment # Not needed if using .format()
import os
import webbrowser
import http.server
import socketserver
import threading
import functools # Added import
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import LLM KML templates
try:
    # Assumes llm_route_generator.py is in the same directory (src)
    from .llm_route_generator import KML_TEMPLATE, KML_WAYPOINT_TEMPLATE
except ImportError:
    # Fallback if running script directly or structure differs
    try:
        from llm_route_generator import KML_TEMPLATE, KML_WAYPOINT_TEMPLATE
    except ImportError:
         KML_TEMPLATE = None # Indicate templates are unavailable
         KML_WAYPOINT_TEMPLATE = None
         # Initialize logger if not already configured elsewhere
         if not logging.getLogger().hasHandlers():
              logging.basicConfig(level=logging.WARNING)
         logging.warning("Could not import KML templates from llm_route_generator. LLM KML generation disabled.")

# No Jinja2 environment needed if just using .format() on template strings

def generate_kml(flight_paths, output_file='visualization/flight_path.kml'):
    """
    Generate KML file for flight path visualization. Handles both simplekml
    for nominal/synthetic routes and string templating for LLM routes.

    Args:
        flight_paths: List of routes (list of lists of waypoints) to visualize.
                      Each waypoint dictionary should have 'latitude', 'longitude', 'altitude'.
                      LLM routes should have an 'is_llm': True flag in the first waypoint.
        output_file: Path where KML file will be saved

    Returns:
        Path to generated KML file, or None if generation failed.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # --- Determine if we should use LLM templating ---
    use_llm_template = False
    # Check the first waypoint of the first route for the 'is_llm' flag.
    if KML_TEMPLATE and KML_WAYPOINT_TEMPLATE and \
       flight_paths and isinstance(flight_paths, list) and flight_paths[0] and \
       isinstance(flight_paths[0], list) and flight_paths[0] and \
       isinstance(flight_paths[0][0], dict) and flight_paths[0][0].get('is_llm', False):
        use_llm_template = True
        if len(flight_paths) > 1:
             logging.warning("LLM KML templating received multiple routes. Only the first will be used.")

    # --- Generate KML ---
    if use_llm_template:
        # Use string formatting for the first LLM route
        logging.info(f"Generating KML for LLM route using template: {output_file}")
        try:
            llm_route = flight_paths[0] # Use only the first route

            # Prepare coordinates string (lon,lat,alt_meters)
            coords_list = [
                f"{wp['longitude']},{wp['latitude']},{wp.get('altitude', 0) * 0.3048}"
                for wp in llm_route if 'longitude' in wp and 'latitude' in wp
            ]
            # Indent for KML readability
            coordinates_str = "\n          ".join(coords_list)

            # Prepare waypoint placemarks string
            waypoints_kml_list = []
            for wp in llm_route:
                 alt_m = wp.get('altitude', 0) * 0.3048
                 # Render the waypoint template string using .format()
                 waypoint_str = KML_WAYPOINT_TEMPLATE.format(
                     name=wp.get('name', 'WP'),
                     lat=wp.get('latitude', 0),
                     lon=wp.get('longitude', 0),
                     alt=wp.get('altitude', 0), # Alt in feet for description
                     alt_m=alt_m, # Alt in meters for KML coordinate
                     speed=wp.get('speed', 'N/A')
                 )
                 waypoints_kml_list.append(waypoint_str)
            # Join with newlines, maintain indentation if template expects it
            waypoints_str = "\n".join(waypoints_kml_list)

            # Render the main KML template string using .format()
            final_kml = KML_TEMPLATE.format(coordinates=coordinates_str, waypoints=waypoints_str)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_kml)
            logging.info(f"Successfully saved LLM KML to {output_file}")
            return output_file

        except KeyError as e:
             logging.exception(f"Missing key {e} in LLM waypoint data during KML generation.")
             raise RuntimeError(f"Failed to generate LLM KML due to missing data: {e}")
        except Exception as e:
            logging.exception(f"Error generating KML with template for LLM route: {e}")
            raise RuntimeError(f"Failed to generate LLM KML using template: {e}")

    else:
        # Use simplekml for non-LLM routes or if templating is unavailable/disabled
        logging.info(f"Generating KML using simplekml for {len(flight_paths)} route(s): {output_file}")
        kml = simplekml.Kml()
        folder = kml.newfolder(name="Flight Paths")

        for i, route in enumerate(flight_paths):
            if not route or not isinstance(route, list):
                 logging.warning(f"Skipping invalid route data at index {i}")
                 continue

            # Check route type for coloring (keep existing logic)
            # Make sure waypoints are dicts before accessing them
            is_nominal = any(isinstance(wp, dict) and wp.get('is_nominal', False) for wp in route)
            is_strict_nominal = any(isinstance(wp, dict) and wp.get('is_strict_nominal', False) for wp in route)
            # Note: is_llm is handled above

            if is_nominal: route_name = "Nominal Route (Red)"
            elif is_strict_nominal: route_name = "Strictly Nominal Route (Green)"
            else: route_name = f"Synthetic Route {i+1} (Blue)" # Adjusted index

            linestring = folder.newlinestring(name=route_name)

            coords = []
            for waypoint in route:
                 # Check if waypoint is a dictionary before accessing keys
                 if isinstance(waypoint, dict):
                     try:
                         altitude_meters = waypoint.get('altitude', 0) * 0.3048
                         # Ensure lat/lon exist before appending
                         if 'longitude' in waypoint and 'latitude' in waypoint:
                              coords.append((waypoint['longitude'], waypoint['latitude'], altitude_meters))
                         else:
                              logging.warning(f"Waypoint missing lat/lon in route {i}. Skipping waypoint: {waypoint}")
                     except KeyError as e:
                          logging.warning(f"Waypoint missing key {e} in route {i}. Skipping waypoint: {waypoint}")
                     except Exception as e:
                          logging.warning(f"Error processing waypoint in route {i}: {e}. Skipping waypoint: {waypoint}")
                 else:
                      logging.warning(f"Invalid waypoint format in route {i}. Expected dict, got {type(waypoint)}. Skipping.")


            if not coords:
                 logging.warning(f"Route {i} ('{route_name}') has no valid coordinates. Skipping linestring.")
                 continue

            linestring.coords = coords
            linestring.style.linestyle.width = 4
            if is_nominal: linestring.style.linestyle.color = simplekml.Color.red
            elif is_strict_nominal: linestring.style.linestyle.color = simplekml.Color.green
            else: linestring.style.linestyle.color = simplekml.Color.blue # Default blue

            linestring.altitudemode = simplekml.AltitudeMode.absolute

        try:
            # Check if folder has any content before saving
            if folder.features:
                 kml.save(output_file)
                 logging.info(f"Successfully saved simplekml KML to {output_file}")
                 return output_file
            else:
                 logging.warning(f"No valid features generated for simplekml file {output_file}. File not saved.")
                 return None # Indicate failure or empty file
        except Exception as e:
             logging.exception(f"Error saving simplekml file: {e}")
             raise RuntimeError(f"Failed to save simplekml KML: {e}")

    # Fallback return (should ideally not be reached if exceptions are raised)
    return None

def save_kml(flight_paths, output_file='visualization/flight_path.kml', name='Flight Path'):
    """
    Save flight paths to a KML file, ensuring correct input format for generate_kml.

    Args:
        flight_paths: Can be a single route (list of dicts) or a list of routes (list of lists of dicts).
        output_file: Path where KML file will be saved.
        name: Name for the KML path (currently unused by generate_kml).

    Returns:
        Path to generated KML file, or None if generation failed.
    """
    # Ensure flight_paths is a list of lists of waypoints for generate_kml
    if not isinstance(flight_paths, list):
         # If it's not a list at all, log an error.
         logging.error(f"save_kml received non-list input type: {type(flight_paths)}. Cannot generate KML.")
         return None
    elif not flight_paths:
         # Handle empty list input
         logging.warning("save_kml received an empty list. No KML will be generated.")
         return None
    elif isinstance(flight_paths[0], dict):
         # Input is a single route (list of dicts), wrap it in another list.
         logging.debug("save_kml received a single route, wrapping it in a list.")
         flight_paths = [flight_paths]
    # Else: Assume input is already a list of routes (list of lists of dicts)

    # The 'name' parameter is currently unused by generate_kml's logic.
    return generate_kml(flight_paths, output_file)

def save_combined_kml(nominal_route, synthetic_routes, departure, arrival, output_file=None):
    """
    Save nominal and synthetic routes to a single KML file.
    (Original function from file)
    """
    if output_file is None:
        output_file = f'visualization/flightpaths_{departure}_{arrival}.kml'

    # Mark the nominal route
    nominal_route_with_flag = copy.deepcopy(nominal_route)
    for wp in nominal_route_with_flag:
        wp['is_nominal'] = True

    # Combine all routes with nominal first
    all_routes = [nominal_route_with_flag] + synthetic_routes

    # Generate the KML file using the updated generate_kml
    return generate_kml(all_routes, output_file)


def visualize_flight_paths(flight_paths, title="Flight Path Visualization", validation_results=None):
    """
    Visualize flight paths and optionally validation metrics using a local web server and browser.

    Args:
        flight_paths: List of routes to visualize.
        title: Title for the visualization.
        validation_results: Optional dictionary containing validation results from validate_route().
                            If provided, PNG plots will be generated.
    """
    # Create directory if it doesn't exist
    viz_dir = 'visualization'
    os.makedirs(viz_dir, exist_ok=True)

    # Save KML file using the updated save_kml
    kml_file_path = os.path.join(viz_dir, 'flight_paths.kml')
    saved_kml = save_kml(flight_paths, kml_file_path)

    if not saved_kml:
        logging.error("KML file generation failed. Cannot visualize KML.")
        # Still proceed if only plots are needed? Or return? Let's return for now.
        return

    # Generate validation plots if results are provided
    plot_files = []
    if validation_results:
        try:
            plot_files = plot_validation_metrics(validation_results) # Call the plotting function
            logging.info(f"Generated validation plots: {plot_files}")
        except Exception as e:
            logging.error(f"Failed to generate validation plots: {e}")
            # Continue visualization even if plots fail? Yes.

    # Create HTML viewer (always create, even if KML failed, maybe show plots?)
    # Let's stick to only showing KML for now. If KML fails, we return.
    html_file = create_html_viewer('flight_paths.kml', os.path.join(viz_dir, 'index.html')) # Pass relative KML path for HTML

    # Start a simple HTTP server to serve the visualization directory
    # We use functools.partial to set the directory for the handler,
    # avoiding the need to change the current working directory (os.chdir).
    # Create a handler that serves files specifically from viz_dir
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=viz_dir)

    port = 8000
    # Check if port is already in use? Basic check:
    server_address = ("", port)
    try:
         # Try to bind to the port to see if it's free
         # Use the partial Handler to check the port binding
         with socketserver.TCPServer(server_address, Handler) as httpd:
              # If binding succeeds, the port was free. Close it immediately.
              pass
         # Port is free, proceed to start the server in a thread
         def start_server():
             # Allow address reuse
             socketserver.TCPServer.allow_reuse_address = True
             # Use the partial Handler when starting the actual server
             with socketserver.TCPServer(server_address, Handler) as httpd:
                 logging.info(f"Serving HTTP on port {port} from {viz_dir}...")
                 httpd.serve_forever()

         server_thread = threading.Thread(target=start_server)
         server_thread.daemon = True
         server_thread.start()
         logging.info(f"Server thread started for port {port}.")
         # Open browser - path is relative to the server's root (viz_dir)
         webbrowser.open(f'http://localhost:{port}/{os.path.basename(html_file)}')

    except OSError as e:
         if "already in use" in str(e):
              logging.warning(f"Port {port} already in use. Assuming server is running. Opening browser...")
              # Path is relative to the server's root (viz_dir)
              webbrowser.open(f'http://localhost:{port}/{os.path.basename(html_file)}')
         else:
              logging.error(f"Error starting HTTP server: {e}")
              # Handle error appropriately, maybe show message
    except Exception as e:
         logging.error(f"Unexpected error during server setup or browser opening: {e}")

    # No finally block needed as we didn't change the CWD
def visualize_with_nominal(synthetic_routes, nominal_route, title="Flight Path Comparison"):
    """
    Visualize nominal and synthetic routes together.
    (Original function from file)
    """
    # Mark the nominal route
    nominal_route_with_flag = copy.deepcopy(nominal_route)
    for wp in nominal_route_with_flag:
        wp['is_nominal'] = True

    # Combine all routes with nominal first
    all_routes = [nominal_route_with_flag] + synthetic_routes

    # Call regular visualization
    visualize_flight_paths(all_routes, title)

def plot_validation_metrics(validation_results, output_dir='visualization/validation_plots'):
    """
    Generate plots for validation metrics.
    
    Args:
        validation_results: Dict from validate_route()
        output_dir: Directory to save plots
        
    Returns:
        List of generated plot file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plots = []
    
    # 1. Cross-track error plot
    if 'xte_stats' in validation_results:
        plt.figure(figsize=(10, 5))
        plt.plot(validation_results['xte_stats']['xte_values'], label='Cross-track Error (nm)')
        plt.axhline(y=validation_results['xte_stats']['max_xte'],
                   color='r', linestyle='--', label='Max allowed')
        plt.title('Cross-track Error Along Route')
        plt.xlabel('Waypoint Index')
        plt.ylabel('Error (nm)')
        plt.legend()
        plot_path = f"{output_dir}/xte_plot.png"
        plt.savefig(plot_path)
        plots.append(plot_path)
        plt.close()
    
    # 2. Turn radius compliance plot
    if 'turn_radius_violations' in validation_results:
        plt.figure(figsize=(10, 5))
        turns = validation_results['turn_radius_violations']
        indices = [t[0] for t in turns]
        actual = [t[1] for t in turns]
        required = [t[2] for t in turns]
        
        plt.scatter(indices, actual, label='Actual Radius')
        plt.scatter(indices, required, label='Required Radius')
        plt.title('Turn Radius Compliance')
        plt.xlabel('Waypoint Index')
        plt.ylabel('Radius (km)')
        plt.legend()
        plot_path = f"{output_dir}/turn_radius_plot.png"
        plt.savefig(plot_path)
        plots.append(plot_path)
        plt.close()
    
    # 3. Altitude profile comparison
    if 'altitude_profile' in validation_results:
        plt.figure(figsize=(10, 5))
        plt.plot(validation_results['altitude_profile']['actual'], label='Generated')
        if 'reference' in validation_results['altitude_profile']:
            plt.plot(validation_results['altitude_profile']['reference'], label='Nominal')
        plt.title('Altitude Profile Comparison')
        plt.xlabel('Waypoint Index')
        plt.ylabel('Altitude (ft)')
        plt.legend()
        plot_path = f"{output_dir}/altitude_profile.png"
        plt.savefig(plot_path)
        plots.append(plot_path)
        plt.close()
    
    return plots

def create_html_viewer(kml_filename, output_file='visualization/index.html'):
    """
    Create HTML file with Cesium viewer for KML visualization.
    (Original function from file)
    """
    # Ensure kml_filename is just the filename, not the full path for the HTML script
    kml_basename = os.path.basename(kml_filename)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flight Path Visualization</title>
    <script src="https://cesium.com/downloads/cesiumjs/releases/1.104/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/1.104/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {{
            width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
        }}
        .toolbar {{
            position: absolute; top: 5px; left: 5px; padding: 5px 10px;
            border-radius: 5px; background: rgba(42, 42, 42, 0.8);
            color: white; z-index: 999; font-family: sans-serif; font-size: 12px;
        }}
        .legend {{
            position: absolute; top: 70px; left: 5px; padding: 5px 10px;
            border-radius: 5px; background: rgba(42, 42, 42, 0.8);
            color: white; z-index: 999; font-family: sans-serif; font-size: 12px;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 5px; }}
        .legend-color {{ width: 20px; height: 3px; margin-right: 5px; }}
        .nominal-color {{ background-color: red; }}
        .strict-nominal-color {{ background-color: green; }}
        .synthetic-color {{ background-color: blue; }}
        /* Add style for LLM routes if needed, e.g., purple */
        .llm-color {{ background-color: purple; }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div class="toolbar">
        <div>Use mouse to navigate: Left-click + drag to rotate, Right-click + drag to pan</div>
        <div>Scroll to zoom, Middle-click + drag to tilt</div>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color nominal-color"></div>Nominal Route</div>
        <div class="legend-item"><div class="legend-color strict-nominal-color"></div>Strictly Nominal</div>
        <div class="legend-item"><div class="legend-color synthetic-color"></div>Synthetic Routes</div>
        <!-- Add legend entry for LLM if using a distinct style -->
        <!-- <div class="legend-item"><div class="legend-color llm-color"></div>LLM Generated Route</div> -->
    </div>
    <script>
        // --- IMPORTANT: Add your Cesium Ion Access Token Below ---
        // Get a free token from https://cesium.com/ion/signup/
        Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5NjQ4ZDkxNC0yYzY3LTQ4N2QtYWFkMi01MmNjOGZiNzQ4ZDUiLCJpZCI6MjM5OTcwLCJpYXQiOjE3MjU4MTY1Nzd9.SQ5iqx8NBr2wcOgEgS-y9XEZgwZVcuMG76JkoOrq7tQ';

        // Initialize viewer using newer async methods and baseLayer option
        const viewer = new Cesium.Viewer('cesiumContainer', {{
             // Use baseLayer instead of imageryProvider (deprecated)
             // Use createWorldImageryAsync (recommended)
            baseLayer: Cesium.ImageryLayer.fromProviderAsync(Cesium.createWorldImageryAsync()),
            // Use createWorldTerrainAsync (recommended)
            terrainProvider: Cesium.createWorldTerrainAsync(),
            timeline: false, animation: false, baseLayerPicker: true, // Re-enabled base layer picker
            geocoder: false, homeButton: false, sceneModePicker: false,
            navigationHelpButton: false
        }});

        // Load the KML file
        const kmlDataSource = new Cesium.KmlDataSource();
        kmlDataSource.load('{kml_basename}', {{ // Use basename here
            camera: viewer.scene.camera,
            canvas: viewer.scene.canvas,
            clampToGround: false // Keep false for altitude rendering
        }}).then(function(dataSource) {{
            viewer.dataSources.add(dataSource);
            viewer.flyTo(dataSource); // Use flyTo for smoother transition
        }}).catch(function(error){{ // Use .catch() instead of .otherwise()
            console.error("Error loading KML:", error);
            alert("Failed to load flight path data. See console for details.");
        }});
    </script>
</body>
</html>
"""

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file
