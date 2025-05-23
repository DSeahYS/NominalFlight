# Nominal Flight Path Generator (Paused as of 17 may 2025)


A  flight path generation system producing realistic aircraft trajectories with three levels of variation, built on aircraft physics models and statistical pattern extraction.

## Overview

The **Nominal Flight Path Generator** creates realistic flight trajectories between airports based on historical data patterns and aircraft performance constraints. The system generates three distinct types of routes for comparison:

- **Nominal routes (red lines):** Authentic flight paths extracted from historical AirAsia flight data, representing the most common trajectory between airports.
- **Strictly nominal routes (green lines):** Controlled variations that closely follow nominal paths with small, deterministic deviations using sinusoidal patterns.
- **Synthetic routes (blue lines):** Realistic alternative paths with larger variations that still adhere to aircraft physics constraints.

All generated routes respect real-world aircraft performance limitations, including minimum turn radius calculations based on speed and bank angle constraints. (currently there's a bug where there are too many feature conflicts)

## Features

- **Triple comparison visualization:** View nominal, strictly nominal, and synthetic routes side-by-side.
- **Physics-based constraints:** Realistic turn radius calculations based on aircraft speed.
- **Kalman filter smoothing:** Advanced trajectory smoothing for realistic path generation.
- **Correlated variations:** Continuous, smooth variations instead of independent point randomization.
- **Customizable variation levels:** Control the amount of deviation from nominal paths.
- **Aircraft performance models:** Routes respect aircraft-specific performance characteristics.
- **SID/STAR integration:** Incorporates real-world departure and arrival procedures.
- **KML export:** Generate visualization files compatible with Google Earth.
- **Interactive UI:** User-friendly interface for route generation and visualization.

## Usage

### GUI Mode

1. Launch the application without parameters:  
   ```sh
   python main.py
   ```
2. Select aircraft type from the dropdown menu.
3. Choose departure and arrival airports.
4. Adjust settings (variation level, detail level, etc.).
5. Click **"Generate Flight Path"** to create routes.
6. Use **"Visualize Last Generated"** to view the paths.

### Command Line Mode

Generate a single route:
```sh
python main.py --departure WSSS --arrival WMKK --aircraft A320-214 --visualize
```

Generate multiple routes with variation:
```sh
python main.py --departure WSSS --arrival WMKK --aircraft A320-214 --routes 5 --visualize
```

## Interface Instructions

The application interface includes:

### Basic Options:
- **Aircraft Selection:** Choose from various Airbus models.
- **Airport Selection:** Search and select from 80,000+ global airports.

### Advanced Options:
- **Detail Level:** Controls waypoint density (**Low/Medium/High**).
- **Route Count:** Number of synthetic routes to generate (**1-20**).
- **Variation Level:** Controls deviation from nominal path (**0.05-0.4**).
- **Strict Nominal Deviation:** Controls green line deviation amount (**km**).
- **Visualization Options:** KML/JSON/CSV output formats.
- **Status Bar:** Shows progress and operation information.

## Project Structure

```
Nominal_Flight_Path_Generator/       # Root project directory
├── __init__.py                      # Package initialization for Python imports
├── clean_historical_data.py         # Script for cleaning historical flight data
├── main.py                          # Main application entry point with UI and pipeline functions
├── visualization/                   # Generated visualization outputs
│   ├── flight_path_WMKK_WSSS.kml    # Single route KML file (WMKK to WSSS)
│   ├── flight_path_WSSS_WMKK.kml    # Single route KML file (WSSS to WMKK)
│   ├── flightpaths_WMKK_WSSS.kml    # Multi-route KML with nominal, strict nominal, and synthetic paths
│   └── flightpaths_WSSS_WMKK.kml    # Multi-route KML with nominal, strict nominal, and synthetic paths
├── nominal/                         # Data generated by nominal pattern extractor
│   └── visualizations/              # Analytical visualizations of flight patterns
│       ├── altitude_profile_xxx-xxx.png  # Altitude profile charts by route
│       ├── flight_phase_xxx-xxx.png      # Flight phase distribution charts
│       ├── route_variance_xxx-xxx.png    # Route variation analysis charts
│       └── waypoint_density_xxx-xxx.png  # Waypoint density heatmaps for nominal route extraction
├── historical/                      # Raw historical flight data from AirAsia flights
│   ├── Historical_AXM706_WSSS-WMKK.csv   # Singapore to Kuala Lumpur historical data (Self-Extracted)
│   └── Historical_AXM713_WMKK-WSSS.csv   # Kuala Lumpur to Singapore historical data (Self-Extracted)
├── data/                            # Core data directory
│   ├── aircraft_data.csv            # Aircraft performance specifications
│   ├── airports.csv                 # Global airport database (80,000+ airports)
│   ├── aip/                         # Aeronautical Information Publication data
│   │   ├── MY_AIP.json              # Malaysia AIP (SIDs, STARs, etc.) (Self-Extracted)
│   │   └── SG_AIP.json              # Singapore AIP (SIDs, STARs, etc.) (Self-Extracted)
│   ├── cleaned_historical/          # Processed historical flight data
│   │   ├── flights_AXMxxx_xxxx-xxxx.csv  # Cleaned flight metadata (e.g. flights_AXM706_WSSS-WMKK.csv)
│   │   ├── waypoints_AXMxxx_xxxx-xxxx.csv  # Extracted waypoints (e.g. waypoints_AXM706_WSSS-WMKK.csv)
│   └── nominal/                     # Extracted nominal flight patterns
│       ├── enroute_patterns.json    # En-route segments of nominal patterns
│       └── nominal_patterns.json    # Complete nominal patterns with clusters
└── src/                             # Source code modules
    ├── __init__.py                  # Package initialization
    ├── data_processor.py            # Data loading and processing utilities
    ├── flight_dynamics.py           # Aircraft performance and physics models
    ├── kalman_smoother.py           # Kalman filter for trajectory smoothing
    ├── nominal_extractor.py         # Pattern extraction from historical data
    ├── route_planner.py             # Flight path generation with variations
    ├── route_smoother.py            # Flight path smoothing with turn constraints
    └── visualization.py             # Data visualization and KML generation
```

## General Flow
![Tech Sequence](https://github.com/user-attachments/assets/d7ebafc2-d2a6-439c-876f-6b144d3f1ced)

## In Depth Flowchart
![Full Flowchart](https://github.com/user-attachments/assets/d608ffa0-e45d-4eb3-99e7-2ef7cdff882c)

## Concise Flowchart
![Concise Flowchart](https://github.com/user-attachments/assets/a642974c-b778-46f6-bd58-6b3b097cf9ca)
  
## How It Works

1. **Data Cleaning:** Raw historical flight data is processed and cleaned.
2. **Nominal Pattern Extraction:** Extract common flight profiles.
3. **Airport Selection:** Choose departure and arrival airports.
4. **Route Planning:** Construct a route using SIDs, STARs, and waypoints.
5. **Aircraft Performance:** Apply realistic aircraft constraints.
6. **Flight Dynamics:** Smooth the path with realistic turn modeling and altitude/speed profiles.
7. **Visualization:** View the generated path in 3D or export it.

## Sample Workflow

# Sample Raw Data
![image](https://github.com/user-attachments/assets/238d9f72-31f8-4ee8-a509-5aaeeb1e45a8)

# Cleaned with Data_cleaner.py
![image](https://github.com/user-attachments/assets/07e47106-b958-47bc-8912-a4a7f6459d29)

# Extraction of Nominal Information
![image](https://github.com/user-attachments/assets/531c1ae6-8993-4c61-8c61-d4a3c9b8c5df)
![image](https://github.com/user-attachments/assets/9d1f3dc4-a225-416a-8dcb-4c2262ca61bf)
![image](https://github.com/user-attachments/assets/ba81469c-29f1-415b-8924-41b0d0467c34)
![image](https://github.com/user-attachments/assets/23f99015-a08f-4d83-81f5-4d746c54296e)

# UI Generated
![image](https://github.com/user-attachments/assets/97b1fca7-debf-4b46-bffb-5bcc659c0721)

# Sample Results
![image](https://github.com/user-attachments/assets/6b39a549-ce4e-4e0e-a0a0-e001a4f998bf)


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
