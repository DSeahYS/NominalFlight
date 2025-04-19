# Final Year Project\8. New Airbus Synthetic Flight Generator (Nominal)\src\kalman_smoother.py

import numpy as np
from scipy import linalg

class KalmanSmoother:
    """
    Kalman filter and RTS smoother for flight path smoothing.
    
    This implementation uses a constant velocity model with small accelerations
    modeled as process noise, which is suitable for commercial aircraft.
    """
    
    def __init__(self, process_noise_scale=0.1, measurement_noise_scale=1.0):
        """
        Initialize the Kalman smoother.
        
        Args:
            process_noise_scale: Scale factor for process noise (lower = smoother path)
            measurement_noise_scale: Scale factor for measurement noise (higher = trust measurements less)
        """
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        
        # State transition matrix for constant velocity model
        # State vector: [x, y, altitude, vx, vy, v_alt]
        self.A = np.array([
            [1, 0, 0, 1, 0, 0],  # x += vx
            [0, 1, 0, 0, 1, 0],  # y += vy
            [0, 0, 1, 0, 0, 1],  # altitude += v_alt
            [0, 0, 0, 1, 0, 0],  # vx stays constant
            [0, 0, 0, 0, 1, 0],  # vy stays constant
            [0, 0, 0, 0, 0, 1]   # v_alt stays constant
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # measure x
            [0, 1, 0, 0, 0, 0],  # measure y
            [0, 0, 1, 0, 0, 0]   # measure altitude
        ])
        
        # Process noise covariance matrix
        # Acceleration is modeled as noise
        self.Q = np.eye(6)
        
        # Measurement noise covariance matrix
        self.R = np.eye(3)
    
    def _prepare_data(self, waypoints):
        """
        Convert waypoints to measurements format required by Kalman filter.
        
        Args:
            waypoints: List of waypoint dictionaries with lat, lon, altitude
            
        Returns:
            measurements: numpy array of shape (n, 3)
        """
        n = len(waypoints)
        measurements = np.zeros((n, 3))
        
        for i, wp in enumerate(waypoints):
            # Convert latitude/longitude to a local Cartesian frame (x, y)
            # For short distances, we can approximate using:
            # Earth radius is approximately 6371 km
            # 1 degree latitude = 111 km
            # 1 degree longitude = 111 km * cos(latitude)
            
            # If this is the first waypoint, use it as our reference
            if i == 0:
                self.ref_lat = wp['latitude']
                self.ref_lon = wp['longitude']
                x = 0
                y = 0
            else:
                # Convert to kilometers in a local Cartesian frame
                lat_km = (wp['latitude'] - self.ref_lat) * 111
                lon_km = (wp['longitude'] - self.ref_lon) * 111 * np.cos(np.radians(self.ref_lat))
                x = lon_km
                y = lat_km
            
            # Convert altitude to kilometers for consistent units
            altitude_km = wp.get('altitude', 0) * 0.0003048  # feet to km
            
            measurements[i] = [x, y, altitude_km]
        
        return measurements
    
    def _convert_states_to_waypoints(self, filtered_states, original_waypoints):
        """
        Convert filtered states back to waypoint dictionaries.
        
        Args:
            filtered_states: numpy array of states from Kalman filter
            original_waypoints: original waypoints for reference
            
        Returns:
            smoothed_waypoints: list of waypoint dictionaries
        """
        smoothed_waypoints = []
        
        for i, state in enumerate(filtered_states):
            wp = original_waypoints[i].copy()
            
            # Convert x, y back to latitude/longitude
            x, y = state[0], state[1]
            lon_km = x
            lat_km = y
            
            wp['longitude'] = self.ref_lon + lon_km / (111 * np.cos(np.radians(self.ref_lat)))
            wp['latitude'] = self.ref_lat + lat_km / 111
            
            # Convert altitude back to feet
            wp['altitude'] = state[2] / 0.0003048  # km to feet
            
            # Add velocity information if you want to use it later
            wp['velocity_east'] = state[3]  # km/timestep
            wp['velocity_north'] = state[4]  # km/timestep
            wp['velocity_vertical'] = state[5]  # km/timestep
            
            smoothed_waypoints.append(wp)
        
        return smoothed_waypoints
    
    def smooth_trajectory(self, waypoints):
        """
        Apply Kalman filtering and RTS smoothing to a flight trajectory.
        
        Args:
            waypoints: List of waypoint dictionaries
            
        Returns:
            smoothed_waypoints: List of smoothed waypoint dictionaries
        """
        if len(waypoints) < 3:
            return waypoints
        
        # Prepare measurements
        measurements = self._prepare_data(waypoints)
        
        # Extract time intervals (assuming equal spacing if not provided)
        dt = 1.0  # default time step
        if 'time' in waypoints[0]:
            times = [wp['time'] for wp in waypoints]
            time_diffs = np.diff(times)
            if np.any(time_diffs > 0):
                dt = np.mean(time_diffs[time_diffs > 0])
        
        # Adjust state transition matrix for the time step
        A = self.A.copy()
        A[0, 3] = dt  # x += vx * dt
        A[1, 4] = dt  # y += vy * dt
        A[2, 5] = dt  # altitude += v_alt * dt
        
        # Adjust process noise based on dt
        Q = self.Q.copy() * (dt ** 2) * self.process_noise_scale
        
        # Adjust measurement noise
        R = self.R.copy() * self.measurement_noise_scale
        
        # Initialize state and covariance
        n_states = 6
        n_meas = len(waypoints)
        
        # Storage for filtered and smoothed states
        filtered_states = np.zeros((n_meas, n_states))
        filtered_covs = np.zeros((n_meas, n_states, n_states))
        
        # Initialize with first measurement
        x_0 = np.zeros(n_states)
        x_0[:3] = measurements[0]  # position from first measurement
        
        # Estimate initial velocities if we have at least 2 measurements
        if n_meas > 1:
            # Simple velocity estimate: (pos2 - pos1) / dt
            initial_vel = (measurements[1] - measurements[0]) / dt
            x_0[3:] = initial_vel
        
        # Initial covariance estimate
        P_0 = np.eye(n_states) * 1.0
        
        # Forward pass (Kalman filter)
        x_pred = x_0
        P_pred = P_0
        
        for i in range(n_meas):
            # Measurement
            z = measurements[i]
            
            # Kalman gain
            S = self.H @ P_pred @ self.H.T + R
            K = P_pred @ self.H.T @ linalg.inv(S)
            
            # Update
            res = z - self.H @ x_pred
            x_update = x_pred + K @ res
            P_update = (np.eye(n_states) - K @ self.H) @ P_pred
            
            # Store
            filtered_states[i] = x_update
            filtered_covs[i] = P_update
            
            # Predict for next step (if not last measurement)
            if i < n_meas - 1:
                x_pred = A @ x_update
                P_pred = A @ P_update @ A.T + Q
        
        # Backward pass (RTS smoother)
        smoothed_states = filtered_states.copy()
        smoothed_covs = filtered_covs.copy()
        
        for i in range(n_meas - 2, -1, -1):
            # Predict
            x_pred = A @ filtered_states[i]
            P_pred = A @ filtered_covs[i] @ A.T + Q
            
            # Smoother gain
            G = filtered_covs[i] @ A.T @ linalg.inv(P_pred)
            
            # Smooth
            x_smooth = filtered_states[i] + G @ (smoothed_states[i+1] - x_pred)
            P_smooth = filtered_covs[i] + G @ (smoothed_covs[i+1] - P_pred) @ G.T
            
            # Store
            smoothed_states[i] = x_smooth
            smoothed_covs[i] = P_smooth
        
        # Convert smoothed states back to waypoints
        smoothed_waypoints = self._convert_states_to_waypoints(smoothed_states, waypoints)
        
        return smoothed_waypoints
