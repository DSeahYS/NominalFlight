# Final Year Project\8. New Airbus Synthetic Flight Generator (Nominal)\src\route_smoother.py

import numpy as np
from scipy.interpolate import CubicSpline
import math
import random
from geopy.distance import great_circle
import logging

logger = logging.getLogger(__name__)

class NominalRouteSmoother:
    """
    Creates smooth flight paths with realistic turns and transitions.
    """
    def __init__(self, aircraft_performance):
        """
        Initialize smoother with aircraft performance model.
        """
        self.aircraft_performance = aircraft_performance
        logger.info("Initialized Nominal Route Smoother")
    
    def calculate_turn_points(self, waypoint1, waypoint2, waypoint3, variant_id=0):
        """
        Calculate intermediate points for a smooth turn between three waypoints.
        
        Args:
            waypoint1: First waypoint
            waypoint2: Turn waypoint
            waypoint3: Third waypoint
            variant_id: Variant identifier for consistent randomization
            
        Returns:
            List of waypoints for the turn
        """
        # Extract coordinates
        lat1, lon1 = waypoint1['latitude'], waypoint1['longitude']
        lat2, lon2 = waypoint2['latitude'], waypoint2['longitude']
        lat3, lon3 = waypoint3['latitude'], waypoint3['longitude']
        
        # Calculate bearings
        bearing1 = self._calculate_bearing(lat1, lon1, lat2, lon2)
        bearing2 = self._calculate_bearing(lat2, lon2, lat3, lon3)
        
        # Calculate turn angle
        turn_angle = (bearing2 - bearing1 + 360) % 360
        if turn_angle > 180:
            turn_angle = 360 - turn_angle
        
        # If turn angle is too small, don't add intermediate points
        if turn_angle < 10:
            return [waypoint2]
        
        # Calculate speed at the turn
        speed = waypoint2.get('speed', self.aircraft_performance.get_aircraft_info()['cruise_speed'])
        
        # Calculate turn radius with a small controlled variation based on variant_id
        # Use deterministic variation based on waypoint and variant
        random.seed(variant_id + hash(str((lat2, lon2))) % 10000)
        turn_radius_variance = 1.0 + random.uniform(-0.05, 0.05)  # Reduced variance: +/- 5%
        turn_radius = self.aircraft_performance.calculate_turn_radius(speed) * turn_radius_variance
        
        # Calculate distance to start and end turn
        distance_to_start = max(turn_radius * math.tan(math.radians(turn_angle / 2)), 5)
        
        # Calculate points along the turn
        intermediate_points = []
        
        # Create a sequence of points for the turn
        num_points = max(3, int(turn_angle / 15))  # More points for larger turns
        
        for i in range(num_points):
            progress = i / (num_points - 1)
            turn_bearing = bearing1 + progress * ((bearing2 - bearing1 + 360) % 360)
            if turn_bearing > 360:
                turn_bearing -= 360
                
            turn_point = self._calculate_point_at_distance(
                lat2, lon2, turn_bearing, turn_radius
            )
            
            turn_waypoint = waypoint2.copy()
            turn_waypoint.update({
                'latitude': turn_point[0],
                'longitude': turn_point[1],
                'is_turn_point': True
            })
            
            intermediate_points.append(turn_waypoint)
        
        return intermediate_points
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing from point 1 to point 2.
        """
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        y = math.sin(lon2 - lon1) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def _calculate_point_at_distance(self, lat, lon, bearing, distance):
        """
        Calculate the point at a given distance and bearing from a starting point.
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        # Earth radius in kilometers
        earth_radius = 6371.0
        
        # Calculate destination point
        angular_distance = distance / earth_radius
        
        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        new_lon = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat)
        )
        
        # Convert back to degrees
        new_lat = math.degrees(new_lat)
        new_lon = math.degrees(new_lon)
        
        return new_lat, new_lon
    
    def interpolate_route(self, route, interval=10):
        """
        Interpolate additional points along the route for smoother visualization.
        """
        if len(route) < 2:
            return route
            
        # Extract coordinates
        lats = [wp['latitude'] for wp in route]
        lons = [wp['longitude'] for wp in route]
        alts = [wp.get('altitude', 0) for wp in route]
        speeds = [wp.get('speed', 0) for wp in route]
        
        # Calculate cumulative distance for parameter
        distances = [0]
        for i in range(1, len(route)):
            dist = great_circle(
                (lats[i-1], lons[i-1]),
                (lats[i], lons[i])
            ).kilometers
            distances.append(distances[-1] + dist)
        
        # Normalize distances for parameter t
        if distances[-1] == 0:
            return route
            
        t = [d / distances[-1] for d in distances]
        
        # Create spline interpolation
        lat_spline = CubicSpline(t, lats)
        lon_spline = CubicSpline(t, lons)
        alt_spline = CubicSpline(t, alts)
        speed_spline = CubicSpline(t, speeds)
        
        # Generate evenly spaced points
        num_points = max(len(route), int(distances[-1] / interval))
        t_new = np.linspace(0, 1, num_points)
        
        # Interpolate
        interpolated_route = []
        for i in range(len(t_new)):
            interpolated_route.append({
                'latitude': float(lat_spline(t_new[i])),
                'longitude': float(lon_spline(t_new[i])),
                'altitude': float(alt_spline(t_new[i])),
                'speed': float(speed_spline(t_new[i])),
                'is_interpolated': True
            })
            
        return interpolated_route
    
    def smooth_with_kalman(self, route, process_noise=0.1, measurement_noise=1.0):
        """
        Smooth a route using a Kalman filter with RTS smoother.
        
        Args:
            route: List of waypoints to smooth
            process_noise: Scale of process noise (lower = smoother path)
            measurement_noise: Scale of measurement noise (higher = trust measurements less)
            
        Returns:
            Smoothed route
        """
        if len(route) < 3:
            return route
        
        from src.kalman_smoother import KalmanSmoother
        
        smoother = KalmanSmoother(
            process_noise_scale=process_noise,
            measurement_noise_scale=measurement_noise
        )
        
        return smoother.smooth_trajectory(route)
    
    def is_valid_path(self, waypoints):
        """
        Check if a route is realistic (no loops, reasonable headings).
        
        Args:
            waypoints: List of waypoints
            
        Returns:
            True if path is valid, False otherwise
        """
        if len(waypoints) < 3:
            return True
            
        # Check for backtracking or unrealistic turns
        for i in range(1, len(waypoints)-1):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i+1]
            
            # Calculate heading from prev to curr
            h1 = self._calculate_bearing(
                prev_wp['latitude'], prev_wp['longitude'],
                curr_wp['latitude'], curr_wp['longitude']
            )
            
            # Calculate heading from curr to next
            h2 = self._calculate_bearing(
                curr_wp['latitude'], curr_wp['longitude'],
                next_wp['latitude'], next_wp['longitude']
            )
            
            # Calculate absolute heading change (0-180 degrees)
            hdg_change = abs((h2 - h1 + 180) % 360 - 180)
            
            # Reject if heading change is too extreme (>120 degrees = unrealistic turn)
            if hdg_change > 120:
                return False
                
        return True
    
    def smooth_route(self, route, points=100, variant_id=0, micro_var_level=0.03):
        """
        Apply smoothing to a route with proper turns and interpolation.
        
        Args:
            route: List of waypoints
            points: Number of points in the output
            variant_id: Variant identifier for deterministic randomness
            micro_var_level: Level of micro-variations to apply
            
        Returns:
            Smoothed route
        """
        if len(route) < 3:
            return route
        
        # Validate the path first - if it has unrealistic turns, reduce the variation
        if not self.is_valid_path(route):
            logger.warning(f"Detected unrealistic turns in route variant {variant_id}")
            # If the path has unrealistic turns, smooth it more aggressively
            measurement_noise = 3.0  # Trust measurements less
            process_noise = 0.05      # More smoothing
        else:
            # Normal smoothing parameters
            # Higher measurement noise for variant_id > 0 means trust the measurements less
            # This makes synthetic routes smoother but maintains their character
            measurement_noise = 1.0 + (variant_id * 0.2)  # Scale with variant id
            
            # Lower process noise for more consistent trajectories
            process_noise = 0.1 if variant_id == 0 else 0.15
        
        # Apply Kalman smoothing for a realistic trajectory
        try:
            kalman_smoothed = self.smooth_with_kalman(
                route,
                process_noise=process_noise,
                measurement_noise=measurement_noise
            )
            
            # If Kalman smoothing succeeds, proceed with further processing
            # Create a new route with turn points
            smoothed_route = [kalman_smoothed[0]]
            
            for i in range(1, len(kalman_smoothed) - 1):
                # Get adjacent waypoints
                prev_wp = kalman_smoothed[i-1]
                curr_wp = kalman_smoothed[i]
                next_wp = kalman_smoothed[i+1]
                
                # Calculate and add turn points - use very small micro-variation
                turn_points = self.calculate_turn_points(
                    prev_wp, curr_wp, next_wp, 
                    variant_id=variant_id if micro_var_level > 0 else 0
                )
                smoothed_route.extend(turn_points)
            
            # Add the final waypoint
            smoothed_route.append(kalman_smoothed[-1])
            
        except ImportError as e:
            # Fallback if Kalman smoothing isn't available
            logger.warning(f"Kalman smoother not available: {e}. Using traditional smoothing.")
            
            # Create a new route with turn points (traditional method)
            smoothed_route = [route[0]]
            
            for i in range(1, len(route) - 1):
                # Get adjacent waypoints
                prev_wp = route[i-1]
                curr_wp = route[i]
                next_wp = route[i+1]
                
                # Calculate and add turn points
                turn_points = self.calculate_turn_points(
                    prev_wp, curr_wp, next_wp, 
                    variant_id=variant_id if micro_var_level > 0 else 0
                )
                smoothed_route.extend(turn_points)
            
            # Add the final waypoint
            smoothed_route.append(route[-1])
        
        # Apply interpolation for final visualization
        interpolated_route = self.interpolate_route(smoothed_route)
        
        return interpolated_route
