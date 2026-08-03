"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    """Return the closest waypoint in a bounded window ahead of the car."""
    waypoint_count = len(waypoints)
    candidate_indices = [
        (current_idx + offset) % waypoint_count
        for offset in range(min(120, waypoint_count))
    ]
    distances = [
        np.linalg.norm(location[:2] - waypoints[index].location[:2])
        for index in candidate_indices
    ]
    return candidate_indices[int(np.argmin(distances))]

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.path_xy = np.asarray(
            [waypoint.location[:2] for waypoint in maneuverable_waypoints],
            dtype=np.float64,
        )
        self.waypoint_spacing = 2.0
        self.target_speeds = self._build_speed_profile()
        self.previous_steer = 0.0

    def _build_speed_profile(self) -> np.ndarray:
        """Build a cyclic center-line speed profile from path curvature."""
        path = self.path_xy
        waypoint_count = len(path)
        curvature_step = 4
        previous_points = np.roll(path, curvature_step, axis=0)
        next_points = np.roll(path, -curvature_step, axis=0)

        first_side = path - previous_points
        second_side = next_points - path
        chord = next_points - previous_points
        twice_area = np.abs(
            first_side[:, 0] * second_side[:, 1]
            - first_side[:, 1] * second_side[:, 0]
        )
        denominator = (
            np.linalg.norm(first_side, axis=1)
            * np.linalg.norm(second_side, axis=1)
            * np.linalg.norm(chord, axis=1)
        )
        curvature = np.divide(
            2.0 * twice_area,
            denominator,
            out=np.zeros(waypoint_count, dtype=np.float64),
            where=denominator > 1e-6,
        )

        # A local maximum is safer than an average at chicane entry.
        curvature = np.maximum.reduce(
            [np.roll(curvature, offset) for offset in range(-3, 4)]
        )
        lateral_acceleration_limit = 25.0
        speed_profile = np.sqrt(
            lateral_acceleration_limit / np.maximum(curvature, 1e-4)
        )
        speed_profile = np.clip(speed_profile, 17.0, 80.0)

        # The two tight Monza chicanes define the stability boundary.  Keep
        # them at the proven-safe v4 speed while allowing faster medium turns.
        critical_corner_mask = curvature >= 0.045
        speed_profile[critical_corner_mask] = np.minimum(
            speed_profile[critical_corner_mask], 17.0
        )

        # Propagate each corner's limit backwards using the braking equation.
        segment_lengths = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        maximum_deceleration = 18.0
        for _ in range(4):
            for index in range(waypoint_count - 1, -1, -1):
                next_index = (index + 1) % waypoint_count
                braking_limit = np.sqrt(
                    speed_profile[next_index] ** 2
                    + 2.0
                    * maximum_deceleration
                    * max(segment_lengths[index], 0.1)
                )
                speed_profile[index] = min(speed_profile[index], braking_limit)
        return speed_profile
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location data and locate the car on the cyclic path.
        vehicle_location = self.location_sensor.get_last_gym_observation()
        distances = np.linalg.norm(self.path_xy - vehicle_location[:2], axis=1)
        self.current_waypoint_idx = int(np.argmin(distances))


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle without jumping backwards.
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # Increase geometric lookahead with speed for stable high-speed tracking.
        lookahead_metres = np.clip(7.0 + 0.50 * vehicle_velocity_norm, 8.0, 30.0)
        lookahead_waypoints = int(round(lookahead_metres / self.waypoint_spacing))
        target_index = (
            self.current_waypoint_idx + lookahead_waypoints
        ) % len(self.maneuverable_waypoints)
        waypoint_to_follow = self.maneuverable_waypoints[target_index]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Heading controller with mild smoothing to avoid steering oscillation.
        raw_steer = np.clip(-1.7 * delta_heading, -1.0, 1.0)
        steer_control = 0.75 * raw_steer + 0.25 * self.previous_steer
        self.previous_steer = steer_control

        # The precomputed profile already includes the required braking distance.
        target_velocity = float(self.target_speeds[self.current_waypoint_idx])
        speed_error = target_velocity - vehicle_velocity_norm
        if speed_error >= 0.0:
            throttle_control = np.clip(0.35 + 0.30 * speed_error, 0.0, 1.0)
            brake_control = 0.0
        else:
            throttle_control = 0.0
            brake_control = np.clip(0.08 - 0.20 * speed_error, 0.0, 1.0)

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": brake_control,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control
