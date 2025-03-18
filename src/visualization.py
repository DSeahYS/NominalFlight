# Final Year Project\8. New Airbus Synthetic Flight Generator (Nominal)\src\visualization.py

import simplekml
import os
import webbrowser
import http.server
import socketserver
import threading
import copy

def generate_kml(flight_paths, output_file='visualization/flight_path.kml'):
    """
    Generate KML file for flight path visualization.
    
    Args:
        flight_paths: List of routes to visualize
        output_file: Path where KML file will be saved
    
    Returns:
        Path to generated KML file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create KML document
    kml = simplekml.Kml()
    
    # Create a folder for flight paths
    folder = kml.newfolder(name="Flight Paths")
    
    # Add each flight path to the KML
    for i, route in enumerate(flight_paths):
        # Check route type for coloring
        try:
            is_nominal = any(wp.get('is_nominal', False) for wp in route)
            is_strict_nominal = any(wp.get('is_strict_nominal', False) for wp in route)
            
            # Determine route name
            if is_nominal:
                route_name = "Nominal Route (Red)"
            elif is_strict_nominal:
                route_name = "Strictly Nominal Route (Green)"
            else:
                route_name = f"Synthetic Route {i} (Blue)"
            
            linestring = folder.newlinestring(name=route_name)
            
            # Add coordinates with altitude
            coords = []
            for waypoint in route:
                # Convert altitude from feet to meters for KML (1 ft = 0.3048 m)
                altitude_meters = waypoint.get('altitude', 0) * 0.3048
                coords.append((waypoint['longitude'], waypoint['latitude'], altitude_meters))
            
            linestring.coords = coords
            
            # Style the line - red for nominal, green for strict nominal, blue for synthetic
            linestring.style.linestyle.width = 4
            if is_nominal:
                linestring.style.linestyle.color = simplekml.Color.red
            elif is_strict_nominal:
                linestring.style.linestyle.color = simplekml.Color.green
            else:
                linestring.style.linestyle.color = simplekml.Color.blue
            
            # Set altitude mode to absolute (respects altitude values)
            linestring.altitudemode = simplekml.AltitudeMode.absolute
        except (AttributeError, TypeError) as e:
            # Skip routes that aren't properly formatted
            print(f"Skipping route {i} due to formatting error: {e}")
            continue
    
    # Save the KML file
    kml.save(output_file)
    return output_file

def save_kml(flight_paths, output_file='visualization/flight_path.kml', name='Flight Path'):
    """
    Save flight paths to a KML file.
    
    Args:
        flight_paths: Single flight path or list of flight paths
        output_file: Path where KML file will be saved
        name: Name for the KML path
        
    Returns:
        Path to generated KML file
    """
    # Ensure flight_paths is a list of routes
    if isinstance(flight_paths, list):
        # Check if the first item is a dict (single route passed as list of waypoints)
        if flight_paths and isinstance(flight_paths[0], dict):
            flight_paths = [flight_paths]
    else:
        flight_paths = [flight_paths]
    
    return generate_kml(flight_paths, output_file)

def save_combined_kml(nominal_route, synthetic_routes, departure, arrival, output_file=None):
    """
    Save nominal and synthetic routes to a single KML file.
    
    Args:
        nominal_route: The nominal route
        synthetic_routes: List of synthetic routes
        departure: ICAO code of departure airport
        arrival: ICAO code of arrival airport
        output_file: Path where KML file will be saved
        
    Returns:
        Path to generated KML file
    """
    if output_file is None:
        output_file = f'visualization/flightpaths_{departure}_{arrival}.kml'
    
    # Mark the nominal route
    nominal_route_with_flag = copy.deepcopy(nominal_route)
    for wp in nominal_route_with_flag:
        wp['is_nominal'] = True
    
    # Combine all routes with nominal first
    all_routes = [nominal_route_with_flag] + synthetic_routes
    
    # Generate the KML file
    return generate_kml(all_routes, output_file)

def visualize_flight_paths(flight_paths, title="Flight Path Visualization"):
    """
    Visualize flight paths using a local web server and browser.
    
    Args:
        flight_paths: List of flight paths to visualize
        title: Title for the visualization
    """
    # Create directory if it doesn't exist
    viz_dir = 'visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save KML file
    kml_file = os.path.join(viz_dir, 'flight_paths.kml')
    save_kml(flight_paths, kml_file)
    
    # Create HTML viewer
    html_file = create_html_viewer('flight_paths.kml')
    
    # Start a simple HTTP server
    os.chdir(viz_dir)
    port = 8000
    
    def start_server():
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        httpd.serve_forever()
    
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}/{os.path.basename(html_file)}')

def visualize_with_nominal(synthetic_routes, nominal_route, title="Flight Path Comparison"):
    """
    Visualize nominal and synthetic routes together.
    
    Args:
        synthetic_routes: List of synthetic routes to visualize
        nominal_route: The nominal route for comparison
        title: Title for the visualization
    """
    # Mark the nominal route
    nominal_route_with_flag = copy.deepcopy(nominal_route)
    for wp in nominal_route_with_flag:
        wp['is_nominal'] = True
    
    # Combine all routes with nominal first
    all_routes = [nominal_route_with_flag] + synthetic_routes
    
    # Call regular visualization
    visualize_flight_paths(all_routes, title)

def create_html_viewer(kml_filename, output_file='visualization/index.html'):
    """
    Create HTML file with Cesium viewer for KML visualization.
    
    Args:
        kml_filename: Name of KML file to be loaded (without path)
        output_file: Path where HTML file will be saved
    
    Returns:
        Path to HTML file
    """
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
            position: absolute;
            top: 5px;
            left: 5px;
            padding: 5px 10px;
            border-radius: 5px;
            background: rgba(42, 42, 42, 0.8);
            color: white;
            z-index: 999;
        }}
        .legend {{
            position: absolute;
            top: 70px;
            left: 5px;
            padding: 5px 10px;
            border-radius: 5px;
            background: rgba(42, 42, 42, 0.8);
            color: white;
            z-index: 999;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 3px;
            margin-right: 5px;
        }}
        .nominal-color {{
            background-color: red;
        }}
        .strict-nominal-color {{
            background-color: green;
        }}
        .synthetic-color {{
            background-color: blue;
        }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div class="toolbar">
        <div>Use mouse to navigate: Left-click + drag to rotate, Right-click + drag to pan</div>
        <div>Scroll to zoom, Middle-click + drag to tilt</div>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color nominal-color"></div>
            <div>Nominal Route (Exact Pattern)</div>
        </div>
        <div class="legend-item">
            <div class="legend-color strict-nominal-color"></div>
            <div>Strictly Nominal Route (Small Controlled Deviation)</div>
        </div>
        <div class="legend-item">
            <div class="legend-color synthetic-color"></div>
            <div>Synthetic Routes (Generated Variations)</div>
        </div>
    </div>
    <script>
        // Initialize the Cesium Viewer
        const viewer = new Cesium.Viewer('cesiumContainer', {{
            terrainProvider: Cesium.createWorldTerrain(),
            timeline: false,
            animation: false
        }});
        
        // Load the KML file
        const kmlDataSource = new Cesium.KmlDataSource();
        kmlDataSource.load('{kml_filename}', {{
            camera: viewer.scene.camera,
            canvas: viewer.scene.canvas,
            clampToGround: false
        }}).then(function(dataSource) {{
            viewer.dataSources.add(dataSource);
            viewer.zoomTo(dataSource);
        }}).otherwise(function(error){{
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
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file
