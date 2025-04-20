# \src\flight_dynamics.py

import csv
import math
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

class AircraftPerformance:
    """
    Aircraft performance model for flight simulation.
    """
    
    def __init__(self, aircraft_model="A320-214"):
        """
        Initialize aircraft performance model.
        
        Args:
            aircraft_model: Aircraft model identifier (e.g., "A320-214")
        """
        self.aircraft_model = aircraft_model
        self.aircraft_data = self.load_aircraft_data(aircraft_model)
        
        # Default performance values if specific data not found
        self.default_values = {
            'cruise_altitude': 35000,  # feet
            'cruise_speed': 450,       # knots
            'climb_rate': 2500,        # feet per minute
            'descent_rate': 1500,      # feet per minute
            'max_bank_angle': 25,      # degrees
            'max_climb_angle': 15,     # degrees
            'max_descent_angle': 10,   # degrees
            'approach_speed': 140,     # knots
            'takeoff_speed': 150       # knots
        }
        
    def load_aircraft_data(self, aircraft_model):
        """
        Load aircraft performance data from CSV file.
        
        Args:
            aircraft_model: Aircraft model identifier (e.g., "A320-214")
            
        Returns:
            Dictionary of aircraft performance parameters
        """
        try:
            # Try to find the aircraft data file with proper path resolution
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # Go up one level from src/
            aircraft_data_file = os.path.join(project_root, 'data', 'aircraft_data.csv')
            
            if not os.path.exists(aircraft_data_file):
                logger.error(f"Aircraft data file not found: {aircraft_data_file}")
                return None
                
            # Read the CSV file
            with open(aircraft_data_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    # Check if this row matches our aircraft model
                    if 'Aircraft_Family' in row and 'Model' in row:
                        # Try direct match on Model
                        if row['Model'] == aircraft_model:
                            logger.info(f"Found performance data for {aircraft_model}")
                            return row
                        
                        # Try match on combined family and model
                        full_model = f"{row['Aircraft_Family']}-{row['Model']}"
                        if full_model == aircraft_model:
                            logger.info(f"Found performance data for {aircraft_model}")
                            return row
            
            logger.warning(f"Performance data not found for {aircraft_model}, using default values")
            return None
        except Exception as e:
            logger.error(f"Error loading aircraft data: {e}")
            return None
    
    def get_parameter(self, param_name, default_value=None):
        """
        Get an aircraft parameter with a default fallback.
        
        Args:
            param_name: Name of the parameter to retrieve
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        if self.aircraft_data and param_name in self.aircraft_data:
            try:
                return float(self.aircraft_data[param_name])
            except (ValueError, TypeError):
                return default_value
        return default_value
    
    def get_aircraft_info(self):
        """
        Get basic aircraft information.
        
        Returns:
            Dictionary with aircraft information
        """
        # Try to get values from aircraft data, fallback to defaults
        if self.aircraft_data:
            try:
                return {
                    'type': self.aircraft_model,
                    'cruise_altitude': float(self.aircraft_data.get('Optimal_FL', 350)) * 100,  # Convert FL to feet
                    'cruise_speed': float(self.aircraft_data.get('VMO_kts', 450)),
                    'climb_rate': float(self.aircraft_data.get('Initial_Climb_Rate_ftmin', 2500)),
                    'descent_rate': float(self.aircraft_data.get('Descent_Rate_Initial_ftmin', 1500))
                }
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing aircraft data: {e}, using defaults")
        
        # Use defaults if data not found or error occurs
        return {
            'type': self.aircraft_model,
            'cruise_altitude': self.default_values['cruise_altitude'],
            'cruise_speed': self.default_values['cruise_speed'],
            'climb_rate': self.default_values['climb_rate'],
            'descent_rate': self.default_values['descent_rate']
        }
    
    def calculate_turn_radius(self, speed_kts):
        """
        Calculate the turn radius based on aircraft type and speed.
        
        Args:
            speed_kts: Speed in knots
            
        Returns:
            Turn radius in kilometers
        """
        # Convert knots to m/s
        speed_ms = speed_kts * 0.51444
        
        # Get aircraft-specific bank angle limit (or use default)
        max_bank_deg = self.get_parameter('Max_Bank_Angle_Normal', 25)
        
        # For very low speeds, increase minimum bank angle to avoid enormous turn radii
        min_bank_deg = 5
        actual_bank_deg = max(min_bank_deg, max_bank_deg * (speed_kts / self.get_parameter('VMO_kts', 350)))
        
        # Cap at the maximum limit
        actual_bank_deg = min(actual_bank_deg, max_bank_deg)
        
        # Convert bank angle to radians
        bank_rad = math.radians(actual_bank_deg)
        
        # Standard turn radius formula: R = V²/(g * tan(bank))
        g = 9.81  # gravitational acceleration in m/s²
        radius_m = (speed_ms ** 2) / (g * math.tan(bank_rad))
        
        # Convert meters to kilometers
        radius_km = radius_m / 1000.0
        
        # Add a safety margin for route planning
        safety_factor = 1.2
        radius_km *= safety_factor
        
        # Ensure a minimum reasonable turn radius
        min_radius_km = 0.5
        return max(radius_km, min_radius_km)
    
    def apply_performance_profile(self, route, route_length):
        """
        Apply aircraft performance profile to a route.
        
        Args:
            route: List of waypoints
            route_length: Route length in kilometers
            
        Returns:
            Route with performance parameters applied
        """
        if not route:
            return route
            
        # Get aircraft performance parameters
        aircraft_info = self.get_aircraft_info()
        cruise_alt = aircraft_info['cruise_altitude']
        cruise_speed = aircraft_info['cruise_speed']
        
        # Calculate profile segments
        climb_distance = min(100, route_length * 0.15)  # 15% of route for climb, max 100km
        descent_distance = min(120, route_length * 0.18)  # 18% of route for descent, max 120km
        cruise_distance = route_length - climb_distance - descent_distance
        
        # Determine if route is too short for full cruise
        short_route = cruise_distance < 50
        if short_route:
            # Adjust for short routes
            climb_distance = route_length * 0.4
            descent_distance = route_length * 0.4
            cruise_distance = route_length * 0.2
            cruise_alt = min(cruise_alt, 25000)  # Lower cruise altitude for short routes
        
        # Apply altitude, speed profile to route
        for i, wp in enumerate(route):
            # Calculate distance from start
            if i == 0:
                distance = 0
            else:
                from geopy.distance import great_circle
                prev_wp = route[i-1]
                distance = great_circle(
                    (prev_wp['latitude'], prev_wp['longitude']),
                    (wp['latitude'], wp['longitude'])
                ).kilometers
            
            wp['distance_from_start'] = distance
            
            # Determine phase and apply parameters
            if distance < climb_distance:
                # Climb phase
                progress = distance / climb_distance if climb_distance > 0 else 1
                wp['altitude'] = max(1500, cruise_alt * progress)  # Start above ground level
                wp['speed'] = max(250, cruise_speed * (0.7 + 0.3 * progress))  # Progressively increase speed
                wp['phase'] = 'climb'
                
            elif distance < (climb_distance + cruise_distance):
                # Cruise phase
                wp['altitude'] = cruise_alt
                wp['speed'] = cruise_speed
                wp['phase'] = 'cruise'
                
            else:
                # Descent phase
                progress = (distance - climb_distance - cruise_distance) / descent_distance if descent_distance > 0 else 0
                wp['altitude'] = max(1500, cruise_alt * (1 - progress))
                wp['speed'] = max(180, cruise_speed * (1 - 0.4 * progress))  # Progressively decrease speed
                wp['phase'] = 'descent'
            
            # Calculate heading if not provided
            if 'heading' not in wp and i > 0:
                import math
                prev_wp = route[i-1]
                lat1, lon1 = math.radians(prev_wp['latitude']), math.radians(prev_wp['longitude'])
                lat2, lon2 = math.radians(wp['latitude']), math.radians(wp['longitude'])
                y = math.sin(lon2 - lon1) * math.cos(lat2)
                x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
                bearing = math.degrees(math.atan2(y, x))
                wp['heading'] = (bearing + 360) % 360
        
        return route
