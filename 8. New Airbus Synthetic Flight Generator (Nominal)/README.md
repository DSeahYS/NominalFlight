# Nominal Flight Path Generator

A sophisticated tool for generating realistic flight paths between airports, based on aircraft performance models and historical flight data patterns.

## Overview

The Nominal Flight Path Generator creates realistic flight trajectories between any two airports worldwide. It uses a combination of real-world aeronautical information, aircraft performance models, and historical flight patterns to generate flight paths that closely resemble actual commercial flights.

## Current Status

✅ **Historical Data Cleaning Complete!** Successfully processed 48 flights:
- 28 WSSS-WMKK (Singapore to Kuala Lumpur) flights with cruise altitudes 22,000-24,050 ft
- 20 WMKK-WSSS (Kuala Lumpur to Singapore) flights with cruise altitudes 23,000-30,000 ft

## Features

- Generate flight paths between any two airports worldwide
- Interactive GUI for easy airport selection and parameter configuration
- Support for multiple aircraft types with realistic performance models
- Generate flight paths at different detail levels
- Support for multiple output formats (KML, JSON, CSV)
- Real-time visualization of generated flight paths
- Command-line interface for batch processing

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - tkinter
  - numpy
  - geopy
  - matplotlib
  - scipy
  - pandas

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nominal-flight-path-generator.git
   cd nominal-flight-path-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure data files are in place:
   - `data/airports.csv`: Database of airports worldwide
   - `data/aircraft_data.csv`: Aircraft performance specifications
   - `historical/`: Raw historical flight data files
   - `data/aip/`: Aeronautical Information Publication data

## Usage

### Data Processing Pipeline

1. Clean historical data:
   ```bash
   python src/flight_data_cleaner.py --input historical --output data/cleaned_historical
   ```

2. Extract nominal patterns:
   ```bash
   python src/nominal_extractor.py --input data/cleaned_historical --output data/nominal
   ```

3. Generate synthetic flight paths:
   ```bash
   python src/flight_generator.py --departure WSSS --arrival WMKK --aircraft A320-214
   ```

### Graphical User Interface

1. Run the application:
   ```bash
   python main.py
   ```

2. Using the interface:
   - Select an aircraft type from the dropdown menu
   - Search and select departure airport from the tree view
   - Search and select arrival airport from the tree view
   - Configure advanced options if needed
   - Click "Generate Flight Path" to create the flight path
   - Use "Visualize Last Generated" to view the flight in 3D

### Command Line Interface

The tool can also be used from the command line:

```bash
python main.py --departure WSSS --arrival WMKK --aircraft A320-214 --visualize --output flight_path.kml
```

Options:
- `--departure`: ICAO code of departure airport
- `--arrival`: ICAO code of arrival airport
- `--aircraft`: Aircraft type (e.g., A320-214, B777-300ER)
- `--extract-patterns`: Extract new patterns from historical data
- `--visualize`: Visualize the generated flight path
- `--output`: Output file name and format

## File Structure

```
Nominal_Flight_Path_Generator/
├── main.py                      # Main application entry point
├── historical/                  # Raw historical flight data
│   ├── Historical_AXM706_WSSS-WMKK.csv
│   └── Historical_AXM713_WMKK-WSSS.csv
├── data/                        # Data directory
│   ├── aircraft_data.csv        # Aircraft performance data
│   ├── airports.csv             # Airport database
│   ├── aip/                     # AIP data directory
│   │   ├── MY_AIP.json          # Malaysia AIP
│   │   └── SG_AIP.json          # Singapore AIP
│   ├── cleaned_historical/      # Cleaned historical flight data
│   │   ├── flights_AXM706_WSSS-WMKK.csv
│   │   ├── flights_AXM713_WMKK-WSSS.csv
│   │   ├── waypoints_AXM706_WSSS-WMKK.csv
│   │   └── waypoints_AXM713_WMKK-WSSS.csv
│   └── nominal/                 # Generated nominal patterns
│   │   ├── enroute_patterns.json
│   │   └── nominal_patterns.json
├── visualization/               # Visualization outputs
└── src/                         # Source code modules
    ├── data_processor.py        # Data loading utilities
    ├── flight_dynamics.py       # Aircraft performance models
    ├── nominal_extractor.py     # Pattern extraction
    ├── route_planner.py         # Route planning logic
    ├── route_smoother.py        # Flight path smoothing
    └── visualization.py         # Visualization utilities
```

## How It Works

1. **Data Cleaning**: Raw historical flight data is processed and cleaned to extract realistic patterns
2. **Nominal Pattern Extraction**: Cleaned data is analyzed to identify common flight profiles
3. **Airport Selection**: Choose departure and arrival airports from the global database
4. **Route Planning**: The system constructs a route using SIDs, STARs, and en-route waypoints
5. **Aircraft Performance**: Realistic aircraft performance constraints are applied
6. **Flight Dynamics**: The path is smoothed with realistic turn modeling and altitude/speed profiles
7. **Visualization**: The generated path can be visualized in 3D or exported to various formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.

