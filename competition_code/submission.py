"""Self-contained ROAR Monza racing controller.

Competition constraint:
Only this submission file is deployed, so the controller logic and its three
track datasets are packaged together without runtime file dependencies.

Provenance:
The controller evolved through discussions and simulation trials with assistance
from AI agents with ideas from https://roar.berkeley.edu/past-results/, and
includes tuning and contributions adapted from publicly reviewed repositories.

Design overview
---------------
The file is organized as seven cooperating logical modules:

1. Asset decoding reconstructs the maneuverable waypoint loop, dense racing
   path, and Section 3 radius table from Base85 constants at the end of the file.
2. Geometry and progress helpers maintain cyclic waypoint indices and compute
   planar distances and three-point circumradii.
3. Lateral planning selects a speed- and section-dependent lookahead target,
   snaps it to the dense racing path where appropriate, and applies pure pursuit.
4. Longitudinal planning converts previewed curvature into grip-limited target
   speeds, projects those targets backward through a braking-distance model, and
   chooses the most restrictive throttle/brake recommendation.
5. Section state divides the lap into ten calibrated ranges used for steering,
   friction, preview, and brake-recovery gain scheduling.
6. Recovery and diagnostics rebuild position-dependent state after a teleport
   and optionally capture control telemetry without affecting normal execution.
7. ``RoarCompetitionSolution`` orchestrates these modules once per simulator
   tick and is the only interface required by the competition runner.

Data flows from sensors -> cyclic progress -> steering target and curvature
preview -> lateral/longitudinal commands -> section-specific safety overrides ->
``vehicle.apply_action``. Speeds are expressed in km/h inside controller models,
distances in metres, and planar geometry uses the first two world coordinates.
"""
import atexit
import base64
import io
import json
import math
import os
from collections import deque
from functools import reduce
from typing import List, Tuple

import numpy as np
import roar_py_interface

# Runtime policy. Constants keep competition behavior deterministic: diagnostics
# are opt-in, section lookup normally uses non-skippable index ranges, and a
# single-tick planar displacement above 26 m is treated as a runner teleport.
DEBUG_ENABLED = False
ENABLE_RESPAWN_RECOVERY = True
USE_RANGE_BASED_SECTIONS = True
RESPAWN_DISTANCE_METERS = 26

# Diagnostics are accumulated only when DEBUG_ENABLED is true. Keeping separate
# streams makes post-run inspection possible without coupling control decisions
# to file I/O or formatting work on the normal competition path.
_DEBUG_FRAMES = {}
_DEBUG_VEHICLE_LOCATIONS = []
_DEBUG_TARGET_LOCATIONS = []
_DEBUG_SPEED_LINES = []
_DEBUG_CONTROL_LINES = []
_DEBUG_STEERING_LINES = []


def _decode_float64_array(encoded_data, shape):
    """Decode a Base85-wrapped little-endian float64 table without copying.

    The explicit shape is part of the payload contract and catches accidental
    truncation because ``reshape`` fails when the decoded element count differs.
    """
    raw_data = base64.b85decode(encoded_data)
    return np.frombuffer(raw_data, dtype="<f8").reshape(shape)


# Shared geometry and controller value objects

class SpeedRecommendation:
    """Safe-speed estimate for one previewed turn.

    ``target_speed_at_turn`` is the curvature-derived speed at the previewed
    feature. ``safe_speed_now`` projects that constraint backward and is the
    maximum present speed that can still decelerate over the available distance.
    ``speed_excess`` is cached because the actuator uses it every tick.
    """

    def __init__(
        self,
        distance_to_turn,
        current_speed,
        target_speed_at_turn,
        safe_speed_now,
        preview_id=0,
        radius=0,
    ):
        """Store one curvature constraint and its derived present-speed margin."""
        self.current_speed = current_speed
        self.distance_to_turn = distance_to_turn
        self.target_speed_at_turn = target_speed_at_turn
        self.safe_speed_now = safe_speed_now
        self.speed_excess = current_speed - safe_speed_now
        self.preview_id = preview_id
        self.radius = radius

    def __str__(self):
        """Return a compact diagnostic summary in controller units."""
        return (
            f"{self.preview_id} d {self.distance_to_turn:.0f} "
            f"sp {self.safe_speed_now:.1f} tsp {self.target_speed_at_turn:.1f} "
            f"r {self.radius:.0f}"
        )


def wrap_angle_positive(rad: float):
    """Normalize an angle to the half-open interval ``[0, 2*pi)``."""
    return rad % (2 * np.pi)


class PurePursuitSteering:
    """Stateless geometric path tracker using a preselected target point.

    Lookahead selection and section gain scheduling remain outside this class;
    it only converts current pose plus target geometry into base steering. This
    separation lets the orchestrator change racing strategy without duplicating
    the pure-pursuit relation.
    """

    def compute_steering(
        self, vehicle_location, vehicle_rotation, target_location, current_waypoint_index
    ) -> Tuple[float, str]:
        """Return normalized steering and optional corner diagnostic text.

        The command uses the pure-pursuit bicycle relation
        ``atan2(2 * wheelbase * sin(alpha) / lookahead, 1)``. The outer
        solution applies section-specific gain scheduling and clipping.
        """

        # Pure pursuit operates in the horizontal plane. The path payload keeps
        # three coordinates for API compatibility, but Monza elevation is not
        # used by this controller.
        waypoint_vector = np.array(target_location) - np.array(vehicle_location)

        # Normalize the vehicle-to-target vector before comparing its bearing
        # with the simulator yaw angle.
        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        if distance_to_waypoint == 0:
            return 0  # Prevent division by zero

        waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

        # alpha is the signed angular error in the simulator's yaw convention.
        alpha = wrap_angle_positive(vehicle_rotation[2]) - wrap_angle_positive(
            math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
        )
        debug_str = ""
        if 813 < current_waypoint_index < 840:
            v_angle = wrap_angle_positive(vehicle_rotation[2])
            d_angle = wrap_angle_positive(math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0]))
            debug_str = f"a{alpha} rot{v_angle} {d_angle}"

        # 4.7 m is the calibrated effective wheelbase and 1.5 maps the bicycle
        # steering angle into CARLA's normalized steering command.
        steering_command = 1.5 * math.atan2(
            2.0 * 4.7 * math.sin(alpha) / distance_to_waypoint, 1.0
        )

        return float(steering_command), debug_str

class DenseRacingPath:
    """Locally searchable dense path used for steering targets.

    The dense path supplies a smoother and wider target than the maneuverable
    waypoint loop. The previous match anchors a bounded search, preventing the
    nearest-point query from jumping to a geometrically close but topologically
    distant part of the circuit.
    """

    def __init__(self):
        """Decode the dense line and seed its cyclic local-search cursor."""
        self.last_match_index = 0
        self.line_locations = _decode_float64_array(
            _DENSE_RACING_PATH_B85, _DENSE_RACING_PATH_SHAPE
        )

    def snap_target_to_path(self, target_location):
        """Snap a coarse target to the nearest local dense-path sample."""
        index = self.find_local_nearest_index(target_location)
        self.last_match_index = index
        return self.line_locations[index]

    def find_local_nearest_index(self, target_location):
        """Find the nearest sample inside a 50-behind/49-ahead cyclic window.

        Distance is expected to decrease and then increase across this local
        window, so the first increase identifies the preceding minimum. Respawn
        recovery globally reseeds ``last_match_index`` after a teleport.
        """
        loc_len = len(self.line_locations)
        previous_index = self.last_match_index
        previous_distance = 10000
        for i in range(100):
            ind = (self.last_match_index + loc_len - 50 + i) % loc_len
            loc = self.line_locations[ind]
            dist = np.linalg.norm(target_location[:2] - loc[:2])
            if dist > previous_distance:
                return previous_index
            previous_index = ind
            previous_distance = dist

        print(
            f"return t_loc {target_location[0]}, {target_location[1]} "
            f"ind{previous_index} dist{previous_distance}"
        )
        return target_location

    def point_at_distance(self, current_location, distance):
        """Return the first forward path sample beyond a Euclidean distance."""
        loc_len = len(self.line_locations)
        for i in range(300):
            ind = (self.last_match_index + i) % loc_len
            loc = self.line_locations[ind]
            dist = np.linalg.norm(current_location[:2] - loc[:2])
            if dist > distance:
                return loc, dist

        # Fall back to a modest forward target if the bounded scan is exhausted.
        loc = self.line_locations[(self.last_match_index + 20) % loc_len]
        dist = np.linalg.norm(current_location[:2] - loc[:2])
        return loc, dist


def planar_waypoint_distance(
    p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint
):
    """Return XY distance between two competition waypoints in metres."""
    return np.linalg.norm(p2.location[:2] - p1.location[:2])


def advance_radius_table_index(vehicle_location, current_idx, locations) -> int:
    """Advance a cyclic Section 3 table cursor while preserving topology.

    A full scan first accepts a sample within 3 m. If no sample satisfies that
    tolerance, a bounded forward fallback chooses the closest of 20 samples so
    transient tracking error cannot move the cursor backward around the lap.
    """
    for i in range(current_idx, len(locations) + current_idx):
        ind = i % len(locations)
        if np.linalg.norm(vehicle_location[:2] - locations[ind][:2]) < 3:
            return ind
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(locations)
        d = np.linalg.norm(vehicle_location[:2] - locations[ind][:2])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind

class CurvatureSpeedController:
    """Stateful curvature-preview controller for throttle and brake.

    Generic sections sample several forward waypoint triples. Section 3 uses a
    denser prerecorded radius table, historical vehicle positions, and a
    calibrated prediction delay for its rapid direction changes. Every preview
    becomes a :class:`SpeedRecommendation`; the lowest safe present speed is the
    active constraint.

    The actuator is deliberately stateful. Recent commands, speed change, and a
    bounded brake-hold counter prevent noisy curvature estimates from producing
    alternating full-throttle/full-brake commands near a threshold.
    """

    debug_enabled = False
    debug_messages = deque(maxlen=1000)

    def __init__(self):
        """Initialize fixed preview geometry and per-run actuator history."""
        self.straight_radius = 10000
        self.maximum_speed = 305
        # Generic preview points are sampled by traveled path distance. Multiple
        # overlapping triples detect both tight nearby bends and broad turns
        # whose curvature is visible only over a wider baseline.
        self.preview_distance_targets = [0, 30, 60, 90, 120, 140, 170]
        self.sampled_distances = [0, 30, 60, 90, 120, 150, 180]
        self.near_sample_index = 0
        self.middle_sample_index = 1
        self.far_sample_index = 2
        self.tick_count = 0
        self.previous_speed = 1.0
        self.remaining_brake_ticks = 0
        self.recent_brakes = deque([0]*20, maxlen=20)
        self.recent_throttles = deque([0]*20, maxlen=20)
        self.recent_locations = deque(maxlen=20)
        self.radius_table_index = 0
        # Section 3 uses its own dense topology because center-path waypoint
        # spacing is too coarse for the calibrated chicane preview.
        self.radius_table = _decode_float64_array(
            _SECTION3_RADIUS_TABLE_B85, _SECTION3_RADIUS_TABLE_SHAPE
        )

    def compute_actuation(
        self, waypoints, current_location, current_speed, current_section, section3_preview_waypoints
    ) -> Tuple[float, float, int, SpeedRecommendation, str]:
        """Evaluate preview geometry and return throttle, brake, gear and plan.

        This is the longitudinal module's public entry point. It selects the
        generic or Section 3 preview model, applies section-specific post-brake
        throttle recovery, and updates histories only after the tick's primary
        command has been selected.
        """
        self.tick_count += 1
        self.radius_table_index = advance_radius_table_index(
            current_location, self.radius_table_index, self.radius_table)

        if current_section in [3]:
            throttle, brake, speed_data, debug_str = self.compute_section3_actuation(
                current_location, current_speed, current_section, section3_preview_waypoints)
        else:
           throttle, brake, speed_data, debug_str = self.compute_preview_actuation(
                current_location, current_speed, current_section, waypoints)

        # Gear selection is coarse because this CARLA vehicle's automatic
        # drivetrain dominates acceleration; reverse is retained for parity
        # with the inherited actuator contract.
        gear = max(1, int(current_speed / 60))
        if throttle < 0:
            gear = -1

        # Optional trace point for the selected throttle and brake command.
        #             + " steer " + str(steering)
        #             + "     loc x,z" + str(self.agent.vehicle.transform.location.x)
        #             + " " + str(self.agent.vehicle.transform.location.z))

        self.recent_locations.appendleft(current_location)
        self.previous_speed = current_speed

        # Recover throttle progressively after a braking sequence. Each section
        # has a different recovery envelope because its exit geometry differs.
        consecutive_brake_ticks = self.count_consecutive_brake_ticks()
        speed_excess = current_speed - speed_data.safe_speed_now
        if current_section == 3:
            if 0 < self.remaining_brake_ticks and self.remaining_brake_ticks < 5 and speed_excess < 8 and consecutive_brake_ticks > 5:
                throttle = 0.2
        elif current_section in [0, 1]:
            if 0 < self.remaining_brake_ticks and self.remaining_brake_ticks < 5 and speed_excess < 12 and consecutive_brake_ticks > 3:
                recent_throttles = max(0.3, self.recent_throttles[0])
                throttle = recent_throttles + 0.05
        elif current_section == 4:
            if 0 < self.remaining_brake_ticks and self.remaining_brake_ticks < 5 and speed_excess < 12 and consecutive_brake_ticks > 3:
                recent_throttles = max(0.3, self.recent_throttles[0])
                throttle = recent_throttles + 0.05
        elif current_section == 6:
            if 0 < self.remaining_brake_ticks and self.remaining_brake_ticks < 4 and speed_excess < 8 and consecutive_brake_ticks > 4:
                # A gentler history-based recovery was tested here; the fixed
                # value below is the validated Section 6 behavior.
                throttle = 0.35
        elif current_section == 9 and current_speed < 160:
            if 0 < self.remaining_brake_ticks and self.remaining_brake_ticks < 8 and speed_excess < 20 and consecutive_brake_ticks > 4:
                recent_throttles = max(0.3, self.recent_throttles[0])
                throttle = recent_throttles + 0.06
        elif 0 < self.remaining_brake_ticks and self.remaining_brake_ticks < 5 and speed_excess < 8 and consecutive_brake_ticks > 5:
            recent_throttles = max(0.3, self.recent_throttles[0])
            throttle = recent_throttles + 0.03

        if self.remaining_brake_ticks > 0 and brake > 0:
            self.remaining_brake_ticks -= 1

        # throttle = 0.05 * (100 - current_speed)
        self.recent_throttles.appendleft(throttle)
        self.recent_brakes.appendleft(brake)
        return throttle, brake, gear, speed_data, debug_str

    def count_consecutive_brake_ticks(self):
        """Count the uninterrupted braking run ending at the previous tick."""
        count = 0
        for b in self.recent_brakes:
            if b > 0:
                count += 1
            else:
                return count
        return count

    def compute_preview_actuation(
        self, current_location, current_speed, current_section, waypoints
    ):
        """Build and actuate generic near-, middle-, and far-turn speed plans.

        Adjacent triples provide local curvature. At high speed, two wider
        triples span more of the path so large-radius bends can become limiting
        before the car reaches the short-baseline preview.
        """

        preview_waypoints = self.sample_distance_waypoints(current_location, waypoints)
        r1 = self.estimate_waypoint_radius(preview_waypoints[self.near_sample_index : self.near_sample_index + 3])
        r2 = self.estimate_waypoint_radius(preview_waypoints[self.middle_sample_index : self.middle_sample_index + 3])
        r3 = self.estimate_waypoint_radius(preview_waypoints[self.far_sample_index : self.far_sample_index + 3])

        target_speed1 = self.target_speed_for_radius(r1, current_section)
        target_speed2 = self.target_speed_for_radius(r2, current_section)
        target_speed3 = self.target_speed_for_radius(r3, current_section)

        close_distance = self.sampled_distances[self.near_sample_index] + 3
        mid_distance = self.sampled_distances[self.middle_sample_index]
        far_distance = self.sampled_distances[self.far_sample_index]
        speed_data = []
        speed_data.append(
            self.build_turn_speed_plan(1, r1, close_distance, target_speed1, current_speed)
        )
        speed_data.append(
            self.build_turn_speed_plan(2, r2, mid_distance, target_speed2, current_speed)
        )
        speed_data.append(
            self.build_turn_speed_plan(3, r3, far_distance, target_speed3, current_speed)
        )

        if current_speed > 100:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            if current_section != 9:
                r4 = self.estimate_waypoint_radius(
                    [
                        preview_waypoints[self.middle_sample_index],
                        preview_waypoints[self.middle_sample_index + 2],
                        preview_waypoints[self.middle_sample_index + 4],
                    ]
                )
                target_speed4 = self.target_speed_for_radius(r4, current_section)
                speed_data.append(
                    self.build_turn_speed_plan(4, r4, close_distance, target_speed4, current_speed)
                )

            r5 = self.estimate_waypoint_radius(
                [
                    preview_waypoints[self.near_sample_index],
                    preview_waypoints[self.near_sample_index + 3],
                    preview_waypoints[self.near_sample_index + 6],
                ]
            )
            target_speed5 = self.target_speed_for_radius(r5, current_section)
            speed_data.append(
                self.build_turn_speed_plan(5, r5, close_distance, target_speed5, current_speed)
            )

        # Treat every preview as a constraint and obey the most restrictive one;
        # selecting by safe present speed naturally trades bend radius, target
        # speed, and remaining braking distance in one comparable quantity.
        update = self.select_limiting_speed(speed_data)

        self.log_speed_candidates(
            " -- SPEED: ",
            speed_data[0].safe_speed_now,
            speed_data[1].safe_speed_now,
            speed_data[2].safe_speed_now,
            (0 if len(speed_data) < 4 else speed_data[3].safe_speed_now),
            current_speed,
        )

        throttle, brake = self.choose_longitudinal_actuation(update)
        self.debug_log("--- throt " + str(throttle) + " brake " + str(brake) + "---")
        debug_str = ""
        for i, sd in enumerate(speed_data):
            debug_str += f"R{i}={speed_data[i].radius:.0f} "

        return throttle, brake, update, debug_str

    def choose_longitudinal_actuation(self, speed_data: SpeedRecommendation):
        """Convert the limiting safe-speed plan into throttle and brake.

        Hard braking starts only after the speed ratio exceeds one predicted
        braking tick of tolerance. ``remaining_brake_ticks`` then makes the
        command stateful, avoiding alternating full-brake/full-throttle ticks.
        Below the hard threshold, light braking or estimated steady throttle
        damps speed error while preserving momentum.
        """

        percent_of_max = speed_data.current_speed / speed_data.safe_speed_now
        # Empirical speed response is expressed in km/h per 20 Hz simulator tick.
        avg_speed_change_per_tick = 2.4
        true_percent_change_per_tick = round(
            avg_speed_change_per_tick / (speed_data.current_speed + 0.001), 5
        )
        speed_up_threshold = 0.95
        throttle_increase_multiple = 1.35
        # v13 retains the validated global hard-brake trigger. A larger value
        # delays braking but has previously destabilized the first chicane.
        brake_threshold_multiplier = 1.10
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        speed_change = round(speed_data.current_speed - self.previous_speed, 3)

        if percent_of_max > 1:
            # Consider slowing down
            # if speed_data.current_speed > 200:  # Brake earlier at higher speeds
            #     brake_threshold_multiplier = 0.9

            if percent_of_max > 1 + (
                brake_threshold_multiplier * true_percent_change_per_tick
            ):
                if self.remaining_brake_ticks > 0:
                    self.debug_log(
                        "tb: tick "
                        + str(self.tick_count)
                        + " brake: counter "
                        + str(self.remaining_brake_ticks)
                    )
                    return -1, 1

                # if speed is not decreasing fast, hit the brake.
                if self.remaining_brake_ticks <= 0 and speed_change < 2.5:
                    # start braking, and set for how many ticks to brake
                    self.remaining_brake_ticks = round(
                        (speed_data.current_speed - speed_data.safe_speed_now) / 3
                    )
                    self.remaining_brake_ticks = min(8, self.remaining_brake_ticks)
                    self.debug_log(
                        "tb: tick "
                        + str(self.tick_count)
                        + " brake: initiate counter "
                        + str(self.remaining_brake_ticks)
                    )
                    return -1, 1

                else:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.debug_log(
                        "tb: tick "
                        + str(self.tick_count)
                        + " brake: throttle early1: sp_ch="
                        + str(percent_speed_change)
                    )
                    self.remaining_brake_ticks = 0  # Target reached; clear the brake hold.
                    return 1, 0
            else:
                if speed_change >= 2.5:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.debug_log(
                        "tb: tick "
                        + str(self.tick_count)
                        + " brake: throttle early2: sp_ch="
                        + str(percent_speed_change)
                    )
                    self.remaining_brake_ticks = 0  # Target reached; clear the brake hold.
                    return 1, 0

                # TODO: Try to get rid of coasting. Unnecessary idle time that could be spent speeding up or slowing down
                throttle_to_maintain = self.throttle_for_steady_speed(
                    speed_data.current_speed
                )

                if percent_of_max > 1.02 or percent_speed_change > (
                    -true_percent_change_per_tick / 2
                ):
                    self.debug_log(
                        "tb: tick "
                        + str(self.tick_count)
                        + " brake: throttle down: sp_ch="
                        + str(percent_speed_change)
                    )
                    return (1, 0.6)  # light break, while keeping throttle on.
                else:
                    return (1, 0.1)  # light break, while keeping throttle on.
        else:
            self.remaining_brake_ticks = 0  # Target reached; clear the brake hold.
            # Speed up
            if speed_change >= 2.5:
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                self.debug_log(
                    "tb: tick "
                    + str(self.tick_count)
                    + " throttle: full speed drop: sp_ch="
                    + str(percent_speed_change)
                )
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.debug_log(
                    "tb: tick "
                    + str(self.tick_count)
                    + " throttle full: p_max="
                    + str(percent_of_max)
                )
                return 1, 0
            throttle_to_maintain = self.throttle_for_steady_speed(
                speed_data.current_speed
            )
            if percent_of_max < 0.98 or true_percent_change_per_tick < -0.01:
                self.debug_log(
                    "tb: tick "
                    + str(self.tick_count)
                    + " throttle up: sp_ch="
                    + str(percent_speed_change)
                )
                return throttle_to_maintain * throttle_increase_multiple, 0
            else:
                self.debug_log(
                    "tb: tick "
                    + str(self.tick_count)
                    + " throttle maintain: sp_ch="
                    + str(percent_speed_change)
                )
                return throttle_to_maintain, 0

    def select_limiting_speed(self, speed_data: List[SpeedRecommendation]):
        """Return the preview candidate with the lowest safe present speed.

        This is equivalent to intersecting all preview constraints: satisfying
        the lowest permissible present speed satisfies every looser candidate.
        """
        min_speed = 1000
        index_of_min_speed = -1
        for i, sd in enumerate(speed_data):
            if sd.safe_speed_now < min_speed:
                min_speed = sd.safe_speed_now
                index_of_min_speed = i

        if index_of_min_speed != -1:
            return speed_data[index_of_min_speed]
        else:
            return speed_data[0]

    def throttle_for_steady_speed(self, current_speed: float):
        """Approximate feed-forward throttle needed to hold speed in km/h."""
        throttle = 0.75 + current_speed / 500
        return throttle

    def build_turn_speed_plan(
        self, name, r, distance: float, target_speed: float, current_speed: float
    ):
        """Project a generic turn target backward to a safe present speed.

        This calibrated closed form is retained exactly because changing its
        constants changes hard-brake branch timing by whole simulator ticks.
        """

        projected_distance = (1 / 675) * (target_speed**2) + distance
        safe_speed_now = math.sqrt(825 * projected_distance)
        return SpeedRecommendation(
            distance, current_speed, target_speed, safe_speed_now, name, r
        )

    def sample_distance_waypoints(self, current_location, more_waypoints):
        """Sample the supplied forward path near configured arc distances.

        Accumulating segment lengths makes preview geometry insensitive to the
        nonuniform spacing of competition waypoints. Actual reached distances
        are retained because the threshold-crossing sample can lie past its
        nominal target.
        """
        points = []
        dist = []  # for debugging
        start = roar_py_interface.RoarPyWaypoint(
            current_location, np.ndarray([0, 0, 0]), 0.0
        )
        # start = self.agent.vehicle.transform
        points.append(start)
        self.sampled_distances[0] = 0
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            # print("start " + str(start) + "\n- - - - -\n")
            # print("end " + str(end) +     "\n- - - - -\n")
            curr_dist += planar_waypoint_distance(start, end)
            # curr_dist += start.location.distance(end.location)
            if curr_dist > self.preview_distance_targets[len(points)]:
                self.sampled_distances[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.preview_distance_targets):
                break

        self.debug_log("wp dist " + str(dist))
        return points

    def estimate_waypoint_radius(self, wp: List[roar_py_interface.RoarPyWaypoint]):
        """Estimate planar path radius from three waypoint locations."""

        point1 = (wp[0].location[0], wp[0].location[1])
        point2 = (wp[1].location[0], wp[1].location[1])
        point3 = (wp[2].location[0], wp[2].location[1])

        return self.circumradius_from_points(point1, point2, point3)

    def circumradius_from_points(self, point1, point2, point3):
        """Return the triangle circumradius used as Menger path curvature.

        Degenerate, nearly collinear, or overly close samples are represented by
        ``straight_radius`` so numerical noise cannot create a false sharp turn.
        """
        # Quantization suppresses sub-millimetre noise in embedded coordinates.
        len_side_1 = round(math.dist(point1, point2), 3)
        len_side_2 = round(math.dist(point2, point3), 3)
        len_side_3 = round(math.dist(point1, point3), 3)

        small_num = 2

        if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
            return self.straight_radius

        # Heron's formula supplies triangle area from the three side lengths.
        sp = (len_side_1 + len_side_2 + len_side_3) / 2

        # Calculating area using Herons formula
        area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
        if area_squared < small_num:
            return self.straight_radius

        # Circumradius is the reciprocal-scale form of Menger curvature.
        radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))

        return radius

    def circumradius_from_locations(self, loc1, loc2, loc3):
        """Adapt raw location arrays to the shared planar radius calculation."""
        point1 = (loc1[0], loc1[1])
        point2 = (loc2[0], loc2[1])
        point3 = (loc3[0], loc3[1])
        return self.circumradius_from_points(point1, point2, point3)

    def target_speed_for_radius(self, radius: float, current_section: int):
        """Convert radius to a section-calibrated lateral-grip speed limit.

        ``sqrt(mu * g * radius)`` is evaluated in m/s and converted to km/h.
        Here ``mu`` is an empirical combined grip/controller factor rather than
        a direct tire-friction measurement; each section absorbs local path and
        steering-model differences.
        """

        mu = 2.75

        if radius >= self.straight_radius:
            return self.maximum_speed

        if current_section == 0:
            mu = 3.2
        if current_section == 1:
            mu = 3.0
        if current_section == 2:
            mu = 3.4
        if current_section == 3:
            mu = 3.3
        if current_section == 4:
            mu = 3.05
        if current_section == 6:
            mu = 3.3
        if current_section == 8:
            mu = 3.1
        if current_section == 9:
            mu = 2.1

        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6

        return max(
            20, min(target_speed, self.maximum_speed)
        )  # Keep targets within the controller's physical speed envelope.

    def log_speed_candidates(
        self, text: str, s1: float, s2: float, s3: float, s4: float, curr_s: float
    ):
        """Format generic preview speeds for optional longitudinal tracing."""
        self.debug_log(
            text
            + " s1= "
            + str(round(s1, 2))
            + " s2= "
            + str(round(s2, 2))
            + " s3= "
            + str(round(s3, 2))
            + " s4= "
            + str(round(s4, 2))
            + " cspeed= "
            + str(round(curr_s, 2))
        )

    def debug_log(self, text):
        """Record a bounded diagnostic message only when tracing is enabled."""
        if self.debug_enabled:
            print(text)
            self.debug_messages.append(text)

    def compute_section3_actuation(self, current_location, current_speed, current_section, waypoints):
        """Use dense spatial samples for Section 3's fast direction changes.

        Triples separated by three 10 m samples estimate overlapping curvature
        constraints. Three historical vehicle locations extend the preview
        behind the current position so a turn remains represented during exit.
        """
        locations = self.sample_section3_path(current_location)
        speed_data = []
        num_radiuses = len(self.sampled_distances)-6
        radius = [0] * num_radiuses
        distances = [0] * num_radiuses
        for ind in range(num_radiuses):
            l1, l2, l3 = locations[ind], locations[ind+3], locations[ind+6]
            distances[ind] = self.sampled_distances[ind+3]
            radius[ind] = self.circumradius_from_locations(l1, l2, l3)

        break_early_d = 3
        for i in range(len(distances)):
            if distances[i] > break_early_d:
                distances[i] -= break_early_d

        debug_str = ""
        for ind in range(num_radiuses):
            target_speed = self.section3_target_speed_for_radius(radius[ind], current_section, current_location)
            speed_data.append(
              self.build_section3_speed_plan(ind, radius[ind], distances[ind], target_speed, current_speed, current_section))
            debug_str += f"r{ind}={radius[ind]:.0f} "

        update = self.select_limiting_speed(speed_data)
        throttle, brake = self.choose_longitudinal_actuation(update)
        return throttle, brake, update, debug_str

    def sample_section3_path(self, current_location):
        """Build a signed-distance sample window around the Section 3 cursor.

        Forward table points are spaced by approximately 10 m of accumulated
        path length. Recent vehicle locations provide negative-distance samples
        behind the car, yielding a continuous curvature window at corner exit.
        """
        increments = [10] * 19
        points = []
        dist = []  # for debugging
        start = self.radius_table[self.radius_table_index]
        points.append(start)
        curr_dist = 0
        total_dist = 0
        self.sampled_distances = [0] * len(increments)

        for i in range(1000):
            ind = (self.radius_table_index + i) % len(self.radius_table)
            end = self.radius_table[ind]
            d = np.linalg.norm(end[:2] - start[:2])
            curr_dist += d
            total_dist += d
            if curr_dist > increments[len(points)]:
                self.sampled_distances[len(points)] = total_dist
                points.append(end)
                dist.append(curr_dist)
                curr_dist = 0
            start = end
            if len(points) >= len(increments):
                break

        prev_points = []
        prev_distances = []
        for d in [30, 20, 10]:
            point, actual_dist = self.historical_location_at_distance(current_location, d)
            if actual_dist > 0:
                prev_points.append(point)
                prev_distances.append(-actual_dist)

        points = prev_points + points
        self.sampled_distances = prev_distances + self.sampled_distances
        dist = prev_distances + dist
        # print(f"wp{len(points)} dist{len(dist)} {str(dist)}")
        return points

    def historical_location_at_distance(self, current_location, target_distance):
        """Return a recent vehicle position at least ``target_distance`` behind."""
        if len(self.recent_locations) < 10:
            return current_location, 0
        accumulated_distance = 0
        start = current_location
        for i in range(len(self.recent_locations)):
            p1 = self.recent_locations[i]
            accumulated_distance += np.linalg.norm(p1[:2] - start[:2])
            if accumulated_distance > target_distance:
                return p1, accumulated_distance
            start = p1

        return current_location, 0

    def build_section3_speed_plan(
        self, name, r, distance: float, target_speed: float, current_speed: float, current_section: int
    ):
        """Project Section 3 target speed through the calibrated brake model.

        The delay term removes distance travelled before measurable speed loss.
        Its locally validated value is 24 ticks in Section 3. Speed-dependent
        braking gain approximates the stronger deceleration observed at higher
        approach speeds.
        """
        if distance <= 0:
            maximum_speed = target_speed
            return SpeedRecommendation(distance, current_speed, target_speed, maximum_speed, name, r)

        braking_gain = 170
        if current_speed > 230:
            braking_gain = 200
        elif current_speed > 210:
            braking_gain = 185
        prediction_delay_ticks = 6
        if current_section in [3]:
            prediction_delay_ticks = 24

        braking_distance = distance - (prediction_delay_ticks / 20) * (
            current_speed / 3.6
        )
        if braking_distance <= 0:
            safe_speed_now = target_speed
        else:
            safe_speed_now = math.sqrt(
                target_speed**2 + 2 * braking_gain * braking_distance
            )
        return SpeedRecommendation(
            distance, current_speed, target_speed, safe_speed_now, name, r
        )

    def section3_target_speed_for_radius(self, radius: float, current_section: int, current_location):
        """Apply Section 3's independently calibrated radius-to-speed factor."""
        mu = 2.75
        if radius >= self.straight_radius:
            return self.maximum_speed

        if current_section == 3:
            mu = 3.6

        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6
        return max(20, min(target_speed, self.maximum_speed))



class SectionTiming:
    """Measure section ticks and distance independently from control state.

    This observer is diagnostic: the competition solution maintains its own
    range-based section state for control. Keeping timing separate ensures print
    reporting cannot alter steering or braking decisions.
    """

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        # vehicle: roar_py_interface.RoarPyActor,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
    ) -> None:
        """Initialize timing counters without sharing mutable control state."""
        self.maneuverable_waypoints = maneuverable_waypoints
        # self.vehicle = vehicle
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.section_start_indices = []
        self.current_waypoint_index = 0
        self.tick_count = 0
        self.section_start_tick = 0
        self.current_section = 0
        self.lap_number = 1
        self.previous_location = None
        self.section_start_distance = 0
        self.current_distance = 0
        self._initialize_progress()

    def _initialize_progress(self) -> None:
        """Seed timing progress from the first available location observation."""
        self.section_start_indices = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]

        # print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        # print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        # Section boundaries are shared with the competition solution.
        # print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_index = 0
        self.current_waypoint_index = advance_waypoint_index(
            vehicle_location, self.current_waypoint_index, self.maneuverable_waypoints
        )

    def update(self) -> None:
        """Update distance and section timing for the current simulator tick."""
        self.tick_count += 1

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        # vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        # vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        # current_speed_kmh = vehicle_velocity_norm * 3.6
        if self.previous_location is not None:
            self.current_distance += np.linalg.norm(vehicle_location - self.previous_location)
        self.previous_location = vehicle_location

        # Find the waypoint closest to the vehicle
        self.current_waypoint_index = advance_waypoint_index(
            vehicle_location, self.current_waypoint_index, self.maneuverable_waypoints
        )

        # compute and print section timing
        for i, section_ind in enumerate(self.section_start_indices):
            if (
                abs(self.current_waypoint_index - section_ind) <= 2
                and i != self.current_section
            ):
                print(
                    f"Section {i}: ticks "
                    f"{(self.tick_count - self.section_start_tick):4d}  distance "
                    f"{(self.current_distance - self.section_start_distance):6.1f}"
                )
                self.section_start_tick = self.tick_count
                self.section_start_distance = self.current_distance
                self.current_section = i
                if self.current_section == 0 and self.lap_number != 3:
                    self.lap_number += 1
                    print(f"\nLap {self.lap_number}\n")



def planar_distance_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    """Return XY distance from a raw vehicle location to one waypoint."""
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def advance_waypoint_index(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    """Advance a cyclic waypoint cursor without allowing backward jumps.

    The normal path scans forward for the first waypoint within 3 m. The bounded
    20-waypoint fallback tolerates lateral tracking error while preserving lap
    topology and avoiding a full nearest-neighbour search every simulator tick.
    """
    for i in range(current_idx, len(waypoints) + current_idx):
        if planar_distance_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(waypoints)
        d = planar_distance_to_waypoint(location, waypoints[ind])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind


def find_nearest_waypoint_index(
    location, waypoints: List[roar_py_interface.RoarPyWaypoint]
):
    """Globally reseed waypoint progress after a detected runner teleport."""
    nearest_distance = 100
    nearest_index = 0
    for i in range(0, len(waypoints)):
        dist = planar_distance_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < nearest_distance:
            nearest_distance = dist
            nearest_index = i
    return nearest_index % len(waypoints)


@atexit.register
def _write_debug_capture():
    """Persist optional post-run telemetry; return immediately in race mode.

    The exit hook is registered unconditionally, but the early guard guarantees
    that the validated competition path performs no diagnostic file writes.
    """
    if not DEBUG_ENABLED:
        return

    print("Saving...")
    fname = "\\debugData\\line.txt"
    with open(
        f"{os.path.dirname(__file__)}{fname}", "w+"
    ) as outfile:
        outfile.write("\n--- Debug steer\n")
        for line in _DEBUG_STEERING_LINES:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Locatons\n")
        for line in _DEBUG_VEHICLE_LOCATIONS:
            outfile.write(f"{line}\n")
        outfile.write("\n--- wpsToFollow\n")
        for line in _DEBUG_TARGET_LOCATIONS:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Debug str\n")
        for line in _DEBUG_CONTROL_LINES:
            outfile.write(f"{line}\n")
        outfile.write("\n--- More Debug str\n")
        for line in _DEBUG_SPEED_LINES:
            outfile.write(f"{line}\n")
    print(f"Saved. {fname}")

    print("Saving debug data")
    jsonData = json.dumps(_DEBUG_FRAMES, indent=4)
    with open(
        f"{os.path.dirname(__file__)}\\debugData\\debugData.json", "w+"
    ) as outfile:
        outfile.write(jsonData)
    print("Debug Data Saved")


# Competition interface and per-tick orchestration
class RoarCompetitionSolution:
    """Competition entry point coordinating all controller modules.

    ``initialize`` reconstructs embedded assets and seeds cyclic progress.
    ``step`` then executes a fixed pipeline: observe, recover, update progress,
    choose the lateral target, preview longitudinal constraints, apply calibrated
    section overrides, record optional diagnostics, and send one vehicle action.
    """

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        """Store runner interfaces and create stateful controller components."""
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.steering_controller = PurePursuitSteering()
        self.speed_controller = CurvatureSpeedController()
        self.section_timing = None
        self.section_start_indices = []
        self.tick_count = 0
        self.current_section = 0
        self.lap_number = 1
        self.previous_location = None
        self.total_dist = 0
        self.racing_path = DenseRacingPath()
        self.emergency_brake_latched = False
        self.section3_steering_scale = 1

    async def initialize(self) -> None:
        """Decode the embedded waypoint loop and initialize lap-relative state.

        The first 35 source waypoints are intentionally excluded to match the
        validated loop alignment used by section boundaries and embedded racing
        path assets.
        """
        # Use the embedded maneuverable path so the submission is self-contained.
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(io.BytesIO(base64.b85decode(_MANEUVERABLE_WAYPOINTS_B85)))
            )[35:]
        )
        self.section_timing = SectionTiming(
            self.maneuverable_waypoints, self.location_sensor, self.velocity_sensor)

        self.section_start_indices = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_start_indices}")
        print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_index = 0
        self.current_waypoint_index = advance_waypoint_index(
            vehicle_location, self.current_waypoint_index, self.maneuverable_waypoints
        )
        self.previous_location = vehicle_location


    def section_for_waypoint(self, idx: int) -> int:
        """Map a cyclic waypoint index to one of ten calibrated sections.

        Section ``k`` runs from ``section_start_indices[k]`` to the next start.
        Section 0 wraps across the lap boundary (2611 -> 2626 -> 0 -> 322).
        Range membership cannot be skipped when high speed advances progress by
        more than the old boundary-proximity tolerance in a single tick.
        """
        starts = self.section_start_indices
        wrap_start = starts[0]          # 2611
        if idx >= wrap_start or idx < starts[1]:
            return 0
        for k in range(len(starts) - 1, 0, -1):
            if idx >= starts[k]:
                return k
        return 0

    def reset_after_respawn(self, vehicle_location):
        """Reconstruct every position-dependent cursor after a runner teleport.

        The runner can move the vehicle back to spawn without recreating this
        solution object. Waypoint progress and dense-path search anchors must
        therefore be reseeded globally before bounded forward searches resume.
        Corner-specific brake and steering latches are also cleared so pre-crash
        state cannot leak into the restarted lap.
        """
        self.current_waypoint_index = find_nearest_waypoint_index(
            vehicle_location, self.maneuverable_waypoints
        )
        self.current_section = self.section_for_waypoint(self.current_waypoint_index)
        # DenseRacingPath normally searches only around its last match. A
        # teleport invalidates that local-search anchor, so recover it globally.
        line = self.racing_path.line_locations
        best_i, best_d = 0, float("inf")
        for i, loc in enumerate(line):
            d = np.linalg.norm(vehicle_location[:2] - loc[:2])
            if d < best_d:
                best_d, best_i = d, i
        self.racing_path.last_match_index = best_i
        self.emergency_brake_latched = False
        self.section3_steering_scale = 1
        self.previous_location = vehicle_location
        print(
            f"[RESPAWN] recovered -> wp {self.current_waypoint_index} "
            f"section {self.current_section} "
            f"lineidx {self.racing_path.last_match_index}"
        )

    async def step(self) -> None:
        """Compute and apply one control command from the latest sensor snapshot.

        The competition runner has already advanced all sensors, so this method
        reads their last synchronized observations and never requests another
        observation. Ordering is important: teleport recovery precedes bounded
        path searches, and ``previous_location`` is copied only after every
        diagnostic and recovery consumer has used the old value.
        """
        self.tick_count += 1
        self.section_timing.update()

        # Phase 1: read the latest synchronous simulator state.
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        speed_mps = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = speed_mps * 3.6

        # Phase 2: recover position-dependent state before any local searches.
        # At 20 Hz even a 300 km/h car moves about 4.2 m per tick. The 26 m gate
        # is therefore well above normal motion and brief simulation jitter but
        # below the runner's hard-collision teleport distance.
        if ENABLE_RESPAWN_RECOVERY and self.previous_location is not None:
            position_jump = float(
                np.linalg.norm(vehicle_location[:2] - self.previous_location[:2])
            )
            if position_jump > RESPAWN_DISTANCE_METERS:
                print(
                    f"[RESPAWN] jump {position_jump:.1f} m detected "
                    f"at tick {self.tick_count}"
                )
                self.reset_after_respawn(vehicle_location)
            elif position_jump > 5.0:
                # normal motion at 300 km/h is ~4.2 m/tick, so anything above
                # this is worth seeing while we debug
                print(f"[BIGSTEP] {position_jump:.1f} m at tick {self.tick_count}")

        # Phase 3: update cyclic progress and exact section membership.
        self.current_waypoint_index = advance_waypoint_index(
            vehicle_location, self.current_waypoint_index, self.maneuverable_waypoints
        )

        if USE_RANGE_BASED_SECTIONS:
            detected_section = self.section_for_waypoint(self.current_waypoint_index)
            if detected_section != self.current_section:
                if detected_section == 0 and self.lap_number != 3:
                    self.lap_number += 1
                self.current_section = detected_section
        else:
            for i, section_ind in enumerate(self.section_start_indices):
                if (
                    abs(self.current_waypoint_index - section_ind) <= 2
                    and i != self.current_section
                ):
                    self.current_section = i
                    if self.current_section == 0 and self.lap_number != 3:
                        self.lap_number += 1

        # Phase 4: choose a steering target. Most sections snap a coarse
        # center-path target to the dense racing path. Sections 0 and 9 retain
        # the maneuverable target because their lap-wrap/Parabolica calibration
        # was validated without dense-path snapping.
        next_waypoint_index = self.lookahead_waypoint_index(current_speed_kmh)
        steering_target = self.select_steering_target(current_speed_kmh, vehicle_location)
        steering_target_location = steering_target.location
        racing_path_target = self.racing_path.snap_target_to_path(
            steering_target.location
        )
        if self.current_section  not in [0, 9]:
            steering_target_location = racing_path_target

        # Phase 5: compute geometric steering before section gain scheduling.
        base_steering, steering_debug = self.steering_controller.compute_steering(
            vehicle_location, vehicle_rotation, steering_target_location, self.current_waypoint_index
        )

        # Phase 6: preview curvature and choose longitudinal actuation. The
        # generic controller begins at the steering lookahead; Section 3 also
        # receives a nine-waypoint backtracked window to preserve nearby bends.
        preview_waypoints = (self.maneuverable_waypoints * 2)[
            next_waypoint_index : next_waypoint_index + 300
        ]
        preview_backtrack_count = 9
        waypoint_count = len(self.maneuverable_waypoints)
        throttle_preview_start = ((next_waypoint_index + waypoint_count) - preview_backtrack_count) % waypoint_count
        section3_preview_waypoints = (self.maneuverable_waypoints * 2)[
            throttle_preview_start : throttle_preview_start + 300
        ]
        throttle, brake, gear, speed_data, speed_debug = self.speed_controller.compute_actuation(
            preview_waypoints,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
            section3_preview_waypoints,
        )

        # Base pure-pursuit steering is gain-scheduled by speed, then calibrated
        # per section to compensate for different line geometry and understeer.
        steering_scale = round((current_speed_kmh + 0.001) / 120, 3)

        # Section 3 entry guard: select steering authority from entry speed and
        # permit one full-brake pulse when the fast-entry threshold is crossed.
        if self.current_waypoint_index in [800, 801]:
            self.section3_steering_scale = 0.85
            if current_speed_kmh >= 162:
                self.section3_steering_scale = 0.95
                if not self.emergency_brake_latched:
                    throttle = 0
                    brake = 1
                    self.emergency_brake_latched = True
            if current_speed_kmh < 160:
                self.section3_steering_scale = 0.75
            print(
                f"spd {current_speed_kmh} "
                f"mult{self.section3_steering_scale} sec={self.current_section}"
            )
        if self.current_waypoint_index in [802, 803, 804]:
            self.emergency_brake_latched = False

        if self.current_section == 2:
            steering_scale *= 1.2
        if self.current_section in [3]:
            if self.current_waypoint_index < 813:
                steering_scale *= self.section3_steering_scale
            elif self.current_waypoint_index < 845:
                steering_scale *= 1.45
            else:
                steering_scale *= 1
                self.section3_steering_scale = 1

        if self.current_section == 4:
            steering_scale = min(1.45, steering_scale * 1.65)
        if self.current_section == 5:
            steering_scale *= 1.1
        if self.current_section in [6]:
            steering_scale = np.clip(steering_scale * 3.2, 3.1, 7)
        if self.current_section == 7:
            steering_scale *= 1.75

        if self.current_section == 9:
            if self.current_waypoint_index > 2580:
                steering_scale = max(steering_scale, 1.7)
            else:
                steering_scale = max(steering_scale, 1.5)

        steering_command = np.clip(base_steering * steering_scale, -1, 1)
        # Prevent a small wrong-direction steering command during the calibrated
        # Section 3 switchback while retaining full authority in the turn direction.
        if  820 < self.current_waypoint_index < 837:
            steering_command = np.clip(base_steering * steering_scale, -0.007, 1)
        # High-speed Parabolica approach guard. The latch limits this emergency
        # correction to one command rather than a multi-tick brake sequence.
        if self.current_waypoint_index in [2381, 2382] and current_speed_kmh > 257:
            if not self.emergency_brake_latched:
              throttle = 0
              brake = 1
              self.emergency_brake_latched = True
        if self.current_waypoint_index in [2383, 2384, 2385]:
            self.emergency_brake_latched = False

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": steering_command,
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,  # Gears do not appear to have an impact on speed
        }

        if DEBUG_ENABLED:
            _DEBUG_VEHICLE_LOCATIONS.append(f"{vehicle_location[0]}, {vehicle_location[1]}")
            _DEBUG_TARGET_LOCATIONS.append(
                f"{steering_target_location[0]}, {steering_target_location[1]}"
            )

            self.total_dist += np.linalg.norm(vehicle_location - self.previous_location)
            self.previous_location = vehicle_location
            s = f"{self.total_dist:.0f}, {current_speed_kmh:.0f}, {speed_data.safe_speed_now:.0f}, {speed_data.preview_id}, {brake*10:.2f}"
            _DEBUG_SPEED_LINES.append(s)
            wp_ind = (self.lap_number-1)*3000 + self.current_waypoint_index
            s = f"{wp_ind:.0f}, {current_speed_kmh:.0f}, {speed_data.safe_speed_now:.0f}, {speed_data.preview_id}, {brake*10:.2f}"
            _DEBUG_STEERING_LINES.append(s)

            wpl = steering_target_location
            d = np.linalg.norm(steering_target.location - vehicle_location)
            s = f"d {self.total_dist:.0f} t {self.tick_count} ind {self.current_waypoint_index} \
sp {current_speed_kmh:.2f} rec {speed_data.safe_speed_now:.1f} dif {(current_speed_kmh - speed_data.safe_speed_now):.1f} \
r={speed_data.radius:.0f}: {speed_debug}, \
t {control['throttle']:.3f} \
br {control['brake']:.3f} \
st: {control['steer']:.10f}, \
{base_steering:.6f}, {steering_scale:.6f} trgt wp:ind {next_waypoint_index} {next_waypoint_index - self.current_waypoint_index} {d:.1f} \
loc: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) wp({wpl[0]:.1f}, {wpl[1]:.1f}) {steering_debug} section {self.current_section}"
            _DEBUG_CONTROL_LINES.append(s)


        if DEBUG_ENABLED:
            _DEBUG_FRAMES[self.tick_count] = {}
            _DEBUG_FRAMES[self.tick_count]["loc"] = [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ]
            _DEBUG_FRAMES[self.tick_count]["throttle"] = round(float(control["throttle"]), 3)
            _DEBUG_FRAMES[self.tick_count]["brake"] = round(float(control["brake"]), 3)
            _DEBUG_FRAMES[self.tick_count]["steer"] = round(float(control["steer"]), 10)
            _DEBUG_FRAMES[self.tick_count]["speed"] = round(current_speed_kmh, 3)
            _DEBUG_FRAMES[self.tick_count]["lap"] = self.lap_number

        # Must happen every tick regardless of DEBUG_ENABLED -- teleport recovery
        # at the top of step() depends on it.
        # np.array(..., copy=True) is essential: get_last_gym_observation() may
        # hand back the same underlying buffer every tick, in which case storing
        # a reference makes previous_location track vehicle_location exactly and
        # the teleport check can never fire.
        self.previous_location = np.array(vehicle_location, copy=True)

        await self.vehicle.apply_action(control)
        return control

    def lookahead_count_for_speed(self, speed):
        """Return the validated discrete center-path lookahead for speed.

        A table is used instead of interpolation because waypoint count is
        discrete and earlier interpolation introduced unstable boundary targets.
        """
        speed_to_lookahead_dict = {
            90: 9,
            110: 11,
            130: 14,
            160: 18,
            180: 22,
            200: 26,
            250: 30,
            300: 35,
        }

        # Interpolation method
        # NOTE does not work as well as the dictionary lookahead method, likely to cause crashes.

        # speedBoundList = [0, 90, 110, 130, 160, 180, 200, 250, 300]
        # lookaheadList = [5, 11, 13, 15, 18, 22, 25, 28, 32]

        # interpolationFunction = interp1d(speedBoundList, lookaheadList)
        # return int(interpolationFunction(speed))

        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
                return num_points
        return 8

    def lookahead_waypoint_index(self, speed):
        """Return the speed-selected cyclic waypoint target index."""
        num_waypoints = self.lookahead_count_for_speed(speed)
        return (self.current_waypoint_index + num_waypoints) % len(
            self.maneuverable_waypoints
        )

    # def get_lateral_pid_config(self):
    #     """
    #     Returns the PID values for the lateral (steering) PID
    #     """
    #     with open(
    #         f"{os.path.dirname(__file__)}\\configs\\LatPIDConfig.json", "r"
    #     ) as file:
    #         config = json.load(file)
    #     return config

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py (Summer 2023)
    def select_steering_target(
        self, current_speed: float, vehicle_location: np.ndarray
    ) -> roar_py_interface.RoarPyWaypoint:
        """Choose a section-aware pure-pursuit target.

        Critical fast sections use physical-distance lookahead on the dense
        racing path, making target distance independent of sample density. Other
        sections average center-path waypoints before the target is optionally
        snapped to the dense path by ``step``.
        """
        if self.current_section == 3:
            distance_gain = 0.25
            distance = distance_gain * current_speed
            distance = np.clip(distance, 44, 70)
            location, _ = self.racing_path.point_at_distance(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [5, 7]:
            distance_gain = 0.25
            # The 0.24 coefficient is intentionally local. Validation showed the
            # shorter target restores lap-2 stability after the Section 3 braking
            # horizon was reduced to 24 ticks, while a global change loses speed.
            if self.current_section == 5 and 1320 <= self.current_waypoint_index < 1360:
                distance_gain = 0.24
            distance = distance_gain * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.racing_path.point_at_distance(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [6]:
            distance_gain = 0.28
            distance = distance_gain * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.racing_path.point_at_distance(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.build_average_waypoint(current_speed)
        else:
            new_waypoint_index = self.lookahead_waypoint_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def build_average_waypoint(self, current_speed):
        """Average a section-sized center-path window around the lookahead.

        Averaging suppresses waypoint-to-waypoint heading noise. The result is
        displacement-limited relative to the original lookahead so smoothing
        cannot move the target too far across the track; Section 1 uses a tighter
        bound because its chicane is especially sensitive to lateral target shift.
        """
        next_waypoint_index = self.lookahead_waypoint_index(current_speed)
        lookahead_value = self.lookahead_count_for_speed(current_speed)
        num_points = lookahead_value * 2

        # Section specific tuning
        if self.current_section == 0:
            num_points = round(lookahead_value * 1.5)
        if self.current_section == 3:
            next_waypoint_index = self.current_waypoint_index + 22
            num_points = 35
        if self.current_section == 4:
            num_points = lookahead_value + 5
            next_waypoint_index = self.current_waypoint_index + 24
        if self.current_section == 5:
            # num_points = round(lookahead_value * 1.1)
            num_points = lookahead_value
        if self.current_section == 6:
            num_points = lookahead_value
            # num_points = 5
            next_waypoint_index = self.current_waypoint_index + 28
        if self.current_section == 7:
            # Jolt between sections 6 and 7 likely due to differences in the
            # lookahead values and steering multipliers.
            num_points = round(lookahead_value * 1.25)
        if self.current_section == 9:
            num_points = 0

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        next_waypoint_index = next_waypoint_index % len(self.maneuverable_waypoints)
        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location

        sample_points = [
            (start_index_for_avg + i) % len(self.maneuverable_waypoints)
            for i in range(0, num_points)
        ]
        if num_points > 3:
            location_sum = reduce(
                lambda x, y: x + y,
                (self.maneuverable_waypoints[i].location for i in sample_points),
            )
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if self.current_section == 1:
                max_shift_distance = 0.2
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv * max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(
                location=new_location,
                roll_pitch_yaw=np.ndarray([0, 0, 0]),
                lane_width=0.0,
            )
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location)
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint = self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint


# Embedded track-asset module
#
# The competition accepts one Python file, so runtime track data lives here at
# the end of the module rather than obscuring the controller implementation.
# ``_MANEUVERABLE_WAYPOINTS_B85`` is a byte-for-byte NPZ archive consumed by the
# competition waypoint loader. The two numeric tables are raw little-endian
# float64 buffers with explicit shapes. Base85 is transport encoding rather than
# compression; repeated ``0000000000`` groups in the dense path represent the
# eight zero bytes of each float64 Z coordinate (the track is controlled in XY).

_MANEUVERABLE_WAYPOINTS_B85 = (
    "P)h>@6aWAK2mk;8ApnZUu?@m#008*;000dD6aZ{*V_|e@Z*FrgZg6=401yE9`2YX_0002OXaE2J0001-on>4W+xPazZbh+kY)tHqQ7n#%h^W|OU<ZN?$`FbO"
    "2zKjO*xkn(TTIMjJ9c*~>hqg*&D{5_|I6pR;p6wS&b8KFd(ED>cg*tY*w(vCDq)NeT(+NoP~X6^^~ACb`qwF2Q7qekMBu2v-oty12<+!?@D{y?2KoPw2Mz2U"
    ";QzmWyH%@It7=8Ddbx_?MDhRczs!OlXy*=$)c*HZtR6g0qQBK2HU8gU^}s3O^0fs9_xAgJ<k?~4-)%30cijtq1zb9Lhq(0py215=Pr!xw&xq^(Uk$F-JpiuP"
    "{|MeT&-nlGfV&$_`%=G&t0^@Mo;-9OxRN$wAo-o+3;RxHJZEzeSNuj9e5*@;)84BHaedS*gG*O>gNsMY5?AHz1{adLoAJ0-BQ88khK`@9bEpAvZPfem)UFO@"
    "d&65ozm3ZT((y9m_x2$!dQ>rZK>F6E-}F5Y=cjfCPoC4#)G0au?MgMm;0b-2f~%e*!COu@xa{7<<coq4kFeR`?>{sG*D_B*yIe0AT=Z&S#`7_rxD@%?;9H;7"
    "HSM!3MEnU^f=R!&)i0}%hezduDbAW!dpCoR`v<(bng3^d5$BM>!PMTWW_t%5gZ}nu2G<U`oB4cy9(l;OfpmseGVSAUpk43JLcg3DPu+*`JM(2Q^`pt9Y_GxN"
    "Glftd+)VwWU*LD6o57WxrA+&Vf8qC7%Mi*#X_H^c7(_Y+2QaQI$wgc%6cs}Gv~+w56BqwW454<FH2cN13~~MA$q<UOnAxs4cj8L#XCc(TqS-I$>JnFHWtb=#"
    "*M-T&pr(jtRoRJ@A5%w9+XnXKTNzwXi<&w!JHqesffFhIA|_AiNnAP}ZSa7*1<g1s3`DzHttWez!lwPh0O%A=nn>*`WaeSsc;fQ=X9m{`nq24+jy(KIKk0wm"
    "Wvr>M<%vQ3w@Xi=JQOhbxLJt*Q?p5=pU;fv-Xhq4>kFOyW*)Asgx~xTlc=upn0eT?0lfAK(qC`t$niUn=Wj=#?`ry;z90Ja9vM6#Ca>9#kB)-JrVcgww{$L3"
    "Ki3(wt5*qw>mHV0=@Rmjt!XIfn_LLI0l#N^hf;ej`;GUJ|7~Fgm)$L%?iuvO6`|BGxy?M3dkg#3M+`0`S-k5P*bjODeY0J9#uV@*K_Q;gY?pWHQIwzOE(+<m"
    "nA|rbc-`U#S9WGM_0wf1E{v<HP#v0i(Av4e{z?<*=P>)FRDRfJ_5sgk=3#ST@TPrOJo-PyiED|03iYq0A5xmQR4c;Z??0IFC^O0t7thWzc=AwFM_yQoxKe4A"
    "!M8p$xfWCv`pb4I)W4>VRIMiB8FNCR{9Ez(*CDRoxMpzC%d{85>%+d!Q^aHDO<mmxac2BxaBn|TM;P7&I(svO{f}2kvGS0;8RG1bKaA?f%$w}ig1Gv`&ENs)"
    "O|Bhj3H$DK!>C=Rj`+6~^#6Lo-sF1re-P)Et_Dw@V{%mkm&yz=xObT4H=;H2`6k%l34P3VX<xy$xv^pP{A-!pAkHRBDE@-ktotDBmA#u-JW@$;dB`DyYlo~j"
    "JDBklxEMxx&T8s-c%of7bm*ILO7+lQ^}&0}gNv2_n#hkdG*!5vQ`+L?kx#)ZSGe6?E7=@zek&0!+3Hp(jD8W<RS%~)O|InfK>QP%g;PGQI?vP;_MRPy54ChI"
    "Nr*@HhkX{S&KrYAk1=?_-OOgYIyOXmCqx<CWvt1y;;1*h_<Y1~)$Lx?r~GIwbgaDTsAoNXUpUp9=~sGH4Se_+g9}O4b^XE}@zl9(aAl{dBh9D?{hP0tU!fVU"
    "J3TTb++GjLPh6)$(QFZtt!~A2CD32@ibUA`DsnN{2UaGoT0ASp2f1NmgFCKMVL!$b{bxJkeM~Mk&x870DZ{VT-Z44Rzxzi<P`_C9Ry!No8x|g6Z?C)o<DOo6"
    "7W7Rn=1z-v+OLGY)n9IZF|Ov^86nxOYyI61@SVpEuGTem)Lx%)9|^q{L3L%-LjcBe^~Dp$#p=&7&iDFiaIv)2FCQP^zI;A?q&;uau-nidm@m?v2YKsN+;_6O"
    "McU&Ohn$Dbo;s1t@23;Ee<gZGQv6n)PaiVl>=McHKY9=Dvr&T?SF>-0{p$&c)2iokYr#9jL{j~j?b7cpL*5Q7BKssOZ#(884=EeCewTQ(S3W@PwXP#|))d6Q"
    "?Ode2fAt*`p}+MG*Dp2-dHC}h@t8V7>>%W=+HZr)?wQQ`9Eay6;d|D}RG(J;9QT1vRAGaAhne<jObf){sUmSdGymF4JlE-28%(z6Uw&N?e%H62EZMGKeSI<L"
    "6z(?JUWe-Kobam-Hn?Nlkng5Jo^OtyY;Tvm9nZZ&-e`k6?rZX}XLzn2Fdy+)amL|!TVA<lvSb@)bgv}noY`&gtuEI1vwa_)t9PG(PA2O<wHEIer>`1Z_pq*)"
    "!}IYz67Yz(_evD_-w%`R{iR<XgLpOxQzYB-iN47n@n_62MY3IYS~*|Xk0>&Q;<WNmzbWEbU15qnPUWsUbe`2UxZ}A+ABguv<xfk4JMMS-uFTMP?KFkz#Nune"
    "(fg_JwC@yqov4Q&<GpqHNP|1C-+7maE3u)BD<2QxJ$g%=!5!DNHhUv<t}U7(*{*9P>q5LQ-&!}tUROf)D8zGh58^TNpu3EM{^}D37fV~$*DV?E>C3Ocul1a8"
    "u`TQ;Jz#N)v+E(AiSHQKeaqszLEYb^V_m1Z>`GiKkR^)h!K#x@e*(yF<^l%y4zr$H+Px&MoGcS1*`6!J=xfCF-_;H7sB>k!2HvWP!5z;_>ckDiwJ5JB7H64x"
    "#MPDEq9prxdoCP)&kl@|Y~!KSeK_pPkBPGPi%_YnsUHz#uOBJ7IdL(~j3|4a^<q__zjLX<9rdh?D?(h@y@A<FZ!$q=)m|3A;Qo39^-IKwD0{t0myZz_x?N^@"
    "R@Fts_2>5t?x<TO-ALkM!`Cct%Ci>4r9nTSZ`FxZn7F2;pDG#Sk5vz=KMbdKU2&aiuQ%!3N#bg<IMrUC%A|$hXDdO+y8osOC9YqtH<jYC#>q?)_>`6gPo878"
    "OH1oYT=>+%;L>`N3(KAiqjoiur%JYQO!D4KT%I%3;Lhjs<Y~l}&*P@r`(2;W9lXutsnjo4yE0WKE``i6xO1Gp@@FV<b;(qFzi4h3iR&TjD9$TZ{q$W%T=v?9"
    "JXqID_u<5aEJqCP7!Q?24T)>3&ryC{9M@M?=x4lPaL0J1zP~ku;tzks`b%iBj=22wE%IaOv>!`ct^AX?b=@f`&7k9xE}G?eZBF9C1ea*Z{+yq9Z!qbE7BaZ="
    "IxV$<xYD^yw7p$=s<Ffcm#PMLycg&nJc!Hl>PJhq=Tbd32Y8_t(bQk2zL@9MAj<zHZ^UW!i@b`s@~x}Ebq{M?O&La9E9D<;uPe229pYk|5e9dROR{S!;;KFl"
    "?X~LjcG5u7i3*Rl_oEy*lekbgmc=h#?MhtUKgZyX=Nvs}N!a&V7H#h@W!b9%WdCI?v)96Q5?3c}=RC_q*ylUIxa!`VxN_tK@@)B?kd?Sr^CD}P`tQ~LWFK;a"
    ";&;4<epp0Ye)E8}OK8!DxKRBS{94!bj&j7s_MZ*zxbEcL@BOL2ynZ7d>v?Rz9^!h9^wX$LEWUpt_@(UA?E3N)58^`YeAB4EtoMREnTU&{icPcEp-|>RKhhar"
    "cADL<5<io;($(GI&ga0BorueyY8gCvs8v5d3qYrPqiK?D9Fy+c=}S7X%_$x~t4`7^C9XedV{pefBh2hcT+ZoB?Nv=3VOTNf)anLq)x)m)eTWz9gZQmH_gY3="
    "%|FQCj{A|gxi@h=dIaLJu8aO9!GDfM{MK{($;Z8^Uy6qyZ`SxTC6TyNH{9Tk>qtL30QSwNl3&Ml7w<+~Ej8WX&ga<SFJ$ujd=`&WTt{3DUjS~6lg9@V7xpfl"
    "CfUcmL^tB%;+4p!#np?wD9&+fnZB~do48zm6XLP1$FK`M$#0?Us5k3+x!#tz{&qLB*K3^bL0sF<`z5A5am_=c{1mj}A9b-i#sBU&i$@*Umbg4KiP~k|@5H+2"
    "x>37+pJ)B0U-*Z(5^$ON-Fi-tx1H`veve!?xO3dD<3U{bdyC~$cRSLB{1&}$aOd?{uMu&z*(2oH%0sdJe#D19L%mt!%;8$Z1?3gHj@14;IurMM$NEcZT7|gu"
    "{v)e5_07gk)UJ_VS^WBxa>T{&KiKu6Jx=UM`s04{covr+E)`7`BiXKZQC;dw`MHzU;EwMD($|8-mH3P?lI=NAxIDK5wYNrAgFBx~lX4MPZsmxPY~!JJXSxsZ"
    "{;trs#vk`A;E(e%9i`{g_SCL!1!L^}C>2XbT)JL_adEohP4(8bIOFo&AMHs0MyVKke@V~Zd69ijw-|f>DuZ9NCH-3!V(fLOPrcuU@+rF;+<70*d80Mu;a)Yy"
    "CBIAmkl(JgNXNR~#dRk<N#|Kz#^s;;T2Xt08d83&?~|JA_LijoQDXTKAFpjeI%AqKE(9%aPIddGCF@5u&4Om6Ke{#J+WrI&%EK2g(y^{1Wp6}N(x2+XxVURv"
    "6XLl#8Ql5aI(39Ze&=>!Ty#6$i2Ro8!MIj#cLTCdlnw59-qyOTt4Dbg`!TLhSz3qcbKU^bv7X=LYcp$6JT5~R7e_9sL4NxVXI!5XSB>l+k2JV*9Pk@mg?PYd"
    "#>K126)FEO$1yHFc<M%ZYZ$_~p8sSiYS-aVTwm7xZrVDL^czGlE(ByNK<!;K1?@HOr&7svE>wqiq79zV#~L3REfPrQUM%XzdS4jzdR1fN@6LE;uay5FNK}V2"
    "nZB^<vWvmRceBaf@jjkAYXO5x<LB|b1#cCJXIsenMd*C16mfMi@AsSO%NbnFyA0!i<+o(5N)%7fa`?6GN9T4`p?E@9!mnvBH@#HN;6jJhxW26W#m*-+$lh%Y"
    "?-%bswTP!$$F46;%u|=*ys@6wdEfH&$$ryD7Qb|^QA3I|d^6Kg3ivjrcsg%o_QKBH66sXg&hjI=pJ`%nEzeF~=YOs>HMo3Y7p`}!A5%Z_ApPJyyw1nGZAS4|"
    "+Q+Y>_#e%QC;!XxEIvuo(%^c70|s|||B+X^v?9M-4q`mC`u%26PvU<L(fDaSFG)=*|3mTnYTWOex~)ln^%0g&?U-j9;zf_KI#h!?wWa*5JZ^BucSvzbA1{i("
    "^hu_#6bfiZe%GGj<5-yFO*&PRm|tPWDdO_(Gu*y*+xDbi=Nz+_4qhRy>_5-AzM#7g#oyv0*WdGqxOP67_j{ee9mucGWuBjpZ;9)duCRWW7L4*Gza6eIuDJdr"
    "E?&LPxH?<uNPautV0Eqv89I^Q+c$ar--#nG^|{T*v17SAll{{>Z2VDsFCwlExyQIxrHCKJ|L{KNtyJPtj|Ys49mFnVf8indwZ10^r<N1fJRWl%QmQM(bMOh{"
    "YOz(s#Y)e(eY5i2D9)|V4eod^7UeC(wJI+emsZv6PVpRlg?h8bnXCth%RSzpKCSoY)lv_G3!mPSj^q8J!U^J9)O%L9%DaDhQXWctWcG5COT@K<pLjggI`^V@"
    "{J!vZCEg(}efY|_{<W%1amIgVTwE}kxKjBi&)bZ>;HQ5vzjC2p#D!kJc|7wQ^`<=k{=>Ky7EWAUB*ZeEJtx3xr)FHsnW+!Oe>n~3W7`lH{nHuTG0x~e<B98E"
    "(;M7*-|2j%5AvKb)}BwXwQFDcciPO1OQ$*!*XL$oT>HHM_GPohvi>S`7kp24t{+sqAI0hI!g(rx;_Bm^Oh=!-7JQ5=*KhioxSlx=w;%5APkeD+#)V>m;6?Hq"
    "-1#0|Vi$3Fd;x<y?hEps@5F^W1zFzoIraOKj!$8OS4pwH`_&I6F5E6c`M2Jiq>)F7%RP&+c+|*r14!qA$of&3)rz<{pak#Vuvp^iyONxTo+GaHF3ok0x(p=0"
    "ugY-y&pyPJ!ETJJU*-_k-;`r<${TLLenbTxf49PeNdId^#--$5;FBuzetEotxRkmI*N=EgT#Tv8b;g$;OgwjW-tR8MiR%k%Fs|L%N?a;Zi`A99?Gw0Kn{nk~"
    "ts(U9Qgu0RG=aFXz8>R(&mrP^jRuVCxl#|MI1e`D*URqa#Pz0)`So&VDslOw#JIZkEOE`VDUat(_F+_47d==V$~k<7p$?lft{j<7T)EnUadFK}*!OA0xb~*V"
    "aPs@wliUB0iK_vvx&5Hk#D$bLj0+22fKTz_*Y)3uBPc&vy*aNPKwO&9p6Bh_4&qwA4!j>LeS`gCU(Rb537~%$>%_0qjg5%wD>^eSZI+2E6}#{}`-DShYggVc"
    "_g50vs(0u0@aY6``A`p@|KrbyOU-(5UMJH?vOgvBIP+B?E_(Oj`M=tRxN^NO@Au&0#D(tuTqh+SI`{i?zH1wC&3~Z59p3|${#T&?dJyA6>MyVl7{a*hl{b*$"
    "|1gyEGIfY+<A!tnf?bI#sRP(}sO}0TuIeKV?)d#e&9emhw}LRPW-#Mleu%i_HpbwN?^kLT9r|J8;5Vb$za3Kr5pNWX@xgj8yH|p^mT?k|^FyuozM{>DtFB=N"
    "cYLQ6=J*pAibfjT@!qGcn*#mBD2$($Uu_L^uEtRQ`<PrgdJ^_O;&ER#x%S^v=-;1haJ8<<g-Pj05%*ijdG|8J6*m=mu<ko&n-iCMCZgZpYN(%W{fO(6H=(_6"
    "z!fnZ{GVOmKfs0RD*P5bNd4mYe*54cak<M0<k{5MVs1g_z<KmzYBL_6uf)~Z>lCL)8dGOn&e4?r=l9XSAIvz1Rv<2JeSv(MI&#rwuy_3o{TpWdt-67a_-pWh"
    "^q0(diUklCR%VGK`|sd#=1B0K`3<fe`ewFw<vilb;4*PkC#kKvT~A!@P$Q1=`P$UkwI6Xd^nm?)GoBo0iR(%2<0zk>z@>Dz;CF0qgM0g>GV@dQ8FA@ufWf6J"
    "e@*>rpNOjo3i0)3{i`m&p%a>bcm$K5OE-q>=c>@RzM~$>PFy*^)!>f#obtW=#N|gC^i8gXlq9ZyOh)|aOrFD?xE89DewbM&^6&c4x%$rF?>~Gs^LfP+@nro="
    "`S<z*u8Up3uVkAp8TD+|kFtIc^gRksr~XPY?c*kZCzOZ1IbT=a5euD^+6Le1lHR=TTByXufz78&w(kaN>Ro785ufQ)C)RjU?<{dCSI_Abzp0~oJ|r&e95h|B"
    "&7V{Qf5Lv!sOeOnX8wg+u45?=rzcIPx-#v>jg^QinWGI}CB>ZQtS7c4uIHI$@Pt0*ylDMgAL7!<#fa0KN3EuvL|k)EgkN+1v;KAgc=Q&7>jf?QI=i8BdGB=U"
    "cauxrmx+tHk5YcD`QGZgx5R}ZXQ5-xmzGv!q4gD7@Ku8+&#~rF&vGNKcfX7NvgTtiY))J){FLm6n)9edm%hY>D{tY~n*aPv!TOE1-=JgmulQvt>_h(|KY6X^"
    "<b%*3k|Cbz!I~F6<_>YCbPj`WeP-!z_=Whl=8dO*vF2}YEJ*7pl)Odbsa@7}QK=qr!M8N@&H2pY-7dtXS{37|AI<s9%9^p@{i~7AP;377xH-gi|2hVD%pcZQ"
    "?IJFRHG;my(_JMlW^5KO+2)N4%RVBWm7ej`ZnM2gnw+$LL#oj(-ku+^xjS+7l`q<5#v{kHC9bdQ8c+Re=|3AnTy8EK+&j#)mo`oV|LkvYWv7`3{qkDqM-3sr"
    "F;;#mog%K37-?|P%d(&H1onr<P`|k3w(^`dnA$rq1pQ*o4-Y6wT&@z%@+p^VN?gAiWpLTu<U+UJuwM~p@b@3gd`fv0;!^urh~MPu#ih{sJs){9xv=9PbapI7"
    "oL0ZccZdr;SD~(~aUkLsakbcbgG=j8u2d})Lj30z#BYrQz3LIy5ALG;dssUCyAapL?MM7B*7Y@xxKRHH^}976UJIHBo%bhE4;J6Go4B~=9O}*F%BgG6A9Wf1"
    "V$DPT{uy!BzCqlYpDY*8J&|;N-9?>S^M?mlCoUa*g#I$Q{=kR0F#84S&9YA&NL=-O$GARh263h27pgZui!a<tTz~o#`LT5NUnDO33JFxV);!+Q@1S2ljlmuB"
    "!Sr?6Cy{>X^u+Hv<^xwEE@#e^K>4(OkGs~ExRjD5L9+keT5K3}^z3AB{eCKsizP1Z%b6hAe;+xq0enX81j#m^LEdzhxEho%L9+j@H{%s?!M7l@S5q?6ItsO1"
    "kp#*1yP@)^JaH|%m>}7{i^|{sA+CQfks#TBU)wzx{AK9`$@aV|cbEp9J8l$@^*fjRU>$Mgd<D{R{B9VQL|o7+CrGy6>C_F+!Pivf{8t)UzoE{kL4F;-Bd>8K"
    "E={b>xYD&G_~3d8lKuBPsXB4FM?>_Z_5PLCgSgmAV)pVbAK1HlpuHCN=>zWCg0)u?X<f(v{_|v9eG&%#sx{-n=^5~Q+l%sV{obb!Urt=p+B2>_-wK_jzO3KX"
    "IT~>#vNPk#!er!aa98-X-aFdq#Pvo!sQw+_y-U9bFCepe(4YT>{)av+9$|HkFw#He&-@CNiV@dm4P-h}I(OpYh#}0ccB~O`q4{ve^%yVc6duX#>+~cpzYF5m"
    "i|cUW%9$~YD<3C<CyvMUZuR4_IQUgU7}vMcdJjSC8Ope{a1+{FKb&!G<U!(6-pLp@Ec+_w5&!Eb#+AHx!IP#juDyH>omFv+%coM{S4l9q<NLgPC2Kgv?>~!i"
    ";Xq;HTGP3lFRe&i78fus?yV1ga}m2Pw3yb=*|C&yVQ@F%di-+6b>AU~XZR|{rDDOv)fQ{G{)1`IDYBk%@zg@Z^LZoc)_f1uXRSv(d$+LhM_>Lg{Py0C@z$D$"
    "HXG|e<S#pE+$(Lp7cad9`%!x^ZdmW#uU~>c-Dhye`@UY}7vgVt06JFuWwOw^4kh6bs|R6VA>#T?4da+Kzb>f))`1i|M&q;N`$^^o(CK}G*MIsp#HEF&`1Kgw"
    "9sEfW<66K_@Pg-<j=VDjyvGH0T}a8X#N~y_{JO}z1o0oc3~v2yTVNA$;l@?u#~MEu9)!;Q>jrnuODuN*?LBq_<A(Kn@xgo0kGzFCx4tt}cnAAZcaR_JyT{zW"
    "$j{Qd=wFMsaiMh%;@5kq6YKr!goyPRId#^*VyCLGcYDC%5py;ruIGJ-`;_(FW~~o#>CQv+qxE}kjlRT%s7JVtGFs2`LBy53kC~1b7YV;p9wYyjzV96PJ^L8<"
    "YpdSUCnEmmk6Hfpjk{rg&*awk|Fx$O=b^{^`trX)Tvi|Rap~uC*mru&>QIdSf%tzu0=Ir&Yn7SSC&-bH(C^l~h3^H4tLYxG`=*{)fw(g8A@Xn4PjOlYB8bZ#"
    ";C^A{XIC5O#OSPF)Y?6WYvt~vo~`<MG>o_&c$eizm^z8LFz7a}3#+cSPltV>o812BQuqzM&g*>IX82unh4rtJ>k#ZmBr~qfI}iQW=QvNggS?eWLcdr#V_w0&"
    ")CoKXTHoJXe!~8u#_QiVBds@(Y97FIrloTu4{>qg9%iqeC`nwex1H<Ps6t$>u!-5ro(-{{VfY%RFJ|y0uDo21@z&~>Yre#_7K>TC^yM<*Y&;jwnO6Pe9}2$}"
    "6WF*aR2hpnzeVFY)6%IJj&?1HVAqS{6$`)pCgOUr=Cw|p3;X+Hct1v~&=&)kU%A?P*tZ+R@}ouW03XtaT^Gu<1JD`Yjn%DEhSrk^LRVk*e5IZ|kGz#_i+Zre"
    "!wlD9f3-QgkLacE!EaC_<iWac-g*L^618!CS^01F27IPF@@!qdV?Tpua%1-$A>bGCKUqZlj&UqCt*_9o<j1(`_+2<7aiL63exG`g9Xuoxt}pBPzfd0H^3GIz"
    "y!}>?xN`Ut-e(=_dPLY?e1!K(tG%yFgFnB5_d#oXODPYY>NwuhE&Yq`;5B!~+uJ2ht${ex(EGJ>o`2f9#HE+Bn2y|`A@XoF0`JAv^TV<xSbs7w2=DvWd(h72"
    "#Pvpf@g8c8dkLQKd&LLu(T@2xZIOqbP4OOTJ-0mfM*KUf!msuIIMEkz_9%|;C)T>8JbuK*6c_fME3WGX{Vf98Wv%C}*9-9veUI;cmd@QiXz!i-_&#a9CkFNh"
    "54%YA)_bn{b`auRb$Gfx4{F>n=oh8$gwAz38Ac*M(lYwa=2(Zmb`<n?PM<E>*Jacg2c7*B@tx3mP97T!zpaN&xA&vEb`tXQqC39RTJN)m!=Mw>2Ju+^emW9<"
    "ztpGiu8#MY15wZqsDST;)_ra2G;pmTzH?ghZCb~nf4gPIcVjDl*96#K|AFt)R(o&F1n>HSadE^P@B%mS-P@|e%=4k&?=-&KTXnT!A^Kh06KD4;=UD>#ENh@|"
    "#UHo~`R_V6&R(C&>E+P*Z?eIi^Pfwuf=;8+_+7(lS7;*gwxA!g*KVwZ{pSvhEBQB|4m;BC7tZ;U-kV@QyIPz*&q9YS&_7-rzdKmZlNGm_b(ItCvc{#K+hJc^"
    "Fu3D7QugdbdxyV^wYOL8yBoYG{T}4Vr&a{(%7l{_@VlY)T=aS$>gxJI{BC9CXU~4*E#oF;uP#3bUVkzB{ZU?U7<nie!{U*`j({gj;C=&-p&s@PVqEEV0`V8<"
    "!no*p%4`??&hC8fs+$Czv9%c&%ASS&wla*1uIG`@bb0anqZQAe3$SmKj>WH^PDb9Qe#X2_>$!CFW$<&4_`D<ctI#ib1@jWE-%0abH|y;vo3EvP{||l#ZR7Kk"
    "|GSBP`B!E0sD<6P!N0__`O?yYyYQPkl+Ujodmnt@a6X^V_W|PB-{b$xGkk>pz0!tpy}%RX`B!~D@ATJG*cY$B<_)TMpF{tjf_y&V@t0_C(@c!Zt6w9|Vc)SH"
    "$MPHg7JiRD!#WGg?|}E_{p&j8LW_^cPnna9Yb8I!@2FjD-GP|yE8<_figESPH^l#H7GGy^^auR9M=-9bzi|Bq1@d(bVZY%wu{Yy#pTDsG&zr5g5n89Bb%5g4"
    "#$@mKo=`CjxT`y}mold#uJ{z?>vulUx=lf=nT>JnQbyuJd`h&PtJ^X||JW<WrP*1b|MFI}JsvS6JM`})F|PT$5SNbZW$XT=LAi+QU)F$I_qAcUiL37OXdRhj"
    "KFO%O&>1|1>nQmV|ESRhcg|CqToCqK`*A+CF!X<PU|gA56govc7*`jI&?#Sy{5s~Ps3nNY6N)n~Cze8fwzx1Z?kEGl+y73rb9JX1{+<0D<KoKl_;=}hSg&e5"
    "&-bqg{hsHs?$!GJ`ferUf7Jo3-?ZNQzE?(`KWt#@bHxHx(5`<LO=bO?w<_ATeHzx2TF=iX!G-h_XdUiQ)30(BTrW2WahhC609RW0!QT4rS{+>N)Dr7_t?vMP"
    "5s%od7F*{l-F1ik;L=P-|A}@9BXYC+h?9{Ac?GTy$9vzM3W(><N49QSdsGhgn;u5l?bYsOq3?Q`ult^4_HUUZ(6Q{(7YARp73;gL_pIzipkGxbeXGvJv<1LV"
    "#7EiltaV5ID}zI$?ER<~%Ypv#8iDoc*85tyOt5$D#r#SqQp0{?JJv4cBE}`bO``SYURL~RFdiz|t42}&ADHti*1a;{FG>&}YR;pOs^Y$&?{$UWhbEtm`;=TV"
    "z46yO?29>{VZ%kl@%}sQOVG!h51_f>{;qv~Mf)vS&ogT6e=!dCzEArVcsZVn@Vun_y*x#-&m;PGCEAti6!tf;t|M(8?8okB`y|L^V-V-DEe3b4Q+_)Uep|01"
    "eaC(60G{WyR|~)$arT4#@9EI7p2NdC!|&I~DU$v9bLT&>e=-4aTE82%!TXn(5MXfUd%;V*->K34!L9LXY6-M=Rd=>OhMETNq3UNJ?EhdrZ{Ns(_*Z+feI%sr"
    "Kk;5%R>FQ7mfwF~<9o=0T8yg=@SZMruEh4u5DF*5zGf-(i>04(1iV=x#Ob*2;5&m7l8g51u$~*00jt4}X2Sj(78mB>yFtrTEDzF*so>wgPqyb@Ei?h&E0(-B"
    "xU}ASPmCOb_?tY(cPuNP`@18~LOSi2;#g;w&kNsM8s4D&PaMy^tr{cFrpeg<#MD>4@qJ0FdYb%N`%TGRisSo+s4;teX)gG6-;4cZES))N5a-t|Z2uPV#b<oq"
    "`%L?^INo!$mrw9r@%S?AyJGpR^dG)M?wQBzmF}luuf}toxP9Q!QM3<?<M-cd>&@?kq1YeBdQbne2;cGi$4<7_p-?Lhar%zHel=FV#N+#<(qRDOV$ulcH0{OO"
    "tACgA9kOC)+K0!nAJO;r;3-~=D|tN-=baXetNCjXmo_%yI<BRmGp?5L*SXHLVP3@9t1{!#$PC~m%3wbnYu)tr@A#g4t_a`%C;dx&$BxRw>P@SD8{fm5XG8w2"
    "=g;2f@cp@1I`pIUy+}C-`;-*y<7DYy*^KYoNuMJn+qk4uQ{i_8?RVt-T_bS@?EN3JeWIk-;jkC)Fs@w1_k2C^3iB(M^N0PQv&LWNdsDwo&@Xa~ab;mk=v?@h"
    "^XGNY-uUelzvFk_>g9=RUh5fGf(pSt?TScy-lXeU5l`5HNXfqbtNI`O?vr{3*WZQTS;To$xxV-SzwbN?g}!yam~t7vzf>I?$@(|rar|BrH=Oe9SSJ{~3;Opd"
    "PtN!5BZ-J7Uk_%l7n_gYgSz{&{D`}vVSlVO<8roO_)X`*b-LsCFRf`k*6+gU9{7DKp(<;yUalQ<Zn!ZnES6wjUSv9wOEt7>Vm@Xs4k`(*Wyd~U*7wp|xslIY"
    "=@}P#r$?M4en;53yz5&y<$25J2z$HK&M$~-MP3`+`JVC|zhjD19?|}9*7s-0{Ty-i@2v>Q{yS;hLFf#j{&YNlDjzmO=U5Wk7fvgv5*L1I5%zwU0%s7{Tknmq"
    "*Wv$OP@%tJ3*%zfQD|4LHLSh*K7ZH`TFSWU-5L51=0w=@uN`YeTy7S}@+0J}2m6_kEI-2V3TW4pV73pQ`X7EDmzo4Idp&!0v{xC-b;kU~{DE7&8JADL$M5y!"
    "yP%FNp5Y<pCk*q(etlLRnqI;0{+nAeE*v=yodS&+R|@Yob!zcCnY;$`D3U7iyk%X0c?-2lV_!jwFPMgT4(kdtdm%X(`p;Y$*Sv=z9-l0{zZPTOg|av`)7Rg$"
    "gWs1wD1VOcKP{VJ-a?}foKLF;9`T%UX=_RFJ@*+`ujNKOudf^2xo=X(^oXa^g>ZX3((Z4VKePEnxIKQc+6(Z!2bsN)@iykYY~M-Tv3}6&9PEE?;QaAH%&Qr>"
    "GTdHQLbENHcXWCovlq<yJi@%0oX?$!c{S&wnT{qzLZ?EQ!JWVF$fGguC~O?%$5C&K`Xl}$BN$h*`a!?M0Hz~&dcwY6Pj0`s9@@2|Blnw75jwrwGA^er3Vys9"
    "x9^ezdCT2^wO3mvz;E~JyuEIZF|SN3&$wRn6y}4umEiX2)?;4Tv;vGP|HeY+t_$PpnGw)wlacq=Z6D&Y|6lCOYW?nWt0wd_eKojrC4*I01)!7o8vDDNT#WyX"
    "`FQUhV*gm{y;6UG`FO={Vjo)T`K0<u*mtJ=V4d|DyB>VT3AW#?etJ4|j_x<ObG^A|5OF>8cKEgOwy-mC?Kh1R&fmrU)`!lYC4Ar7#Uku;&1PIHn-23B#c51m"
    "4ty~Y_DYz&f5mPWh>NdBv-V1jc7P`iK|ZbT75C>6m)(1Zu{z8!k+^!f6YCeXWgqatZNgYT`m`jjziEQ@I_4o%B(6-a&A6~XC+zc7VqE(CmFBN18%j|8*8NT`"
    "cn9-Z>*Z%$DtrX<WA|s{_T3UOpSFD()?f0v7|fe}|6Q?jz063=%N_m}`!-wOPa1YcJO>`pKGKfgPoFnHJTGoA9c^9-*grpSaL0TJZ9qooUpPwpMLT|9zV;UL"
    "mRIe8j-@~P8s<^Y*a#i#el+6%<}LSMj{UK%ep$Q{^PG#!RqXYw$4<k1=A>z~f3@RxnK2`w9~7$C^RI>ZVSaV_QJkl14Ew}^EFSGqX>iXTjO(kjqFwjfbKdq7"
    "%^Meov|wC%e-rb}Kh<Mg+@fK=`W|=MAKS5xtmGQZlOItE{bgMj@p0yS^#YU!$9+G`D9n?;n@zFTiSn&0;z>%)`cVmM0-ap%LRtLf+@Q1ZMkuqtk{xljKFWE}"
    "7pzCPw}Ek?=pC#J2${pU=zWy9wq;VNJ$|{^I^yckffSEppS`{Duuo~v?4`70z<1UQm2B^2La!e1+pJWmy?(UO&9L6VC!4{Y_t^y%u-+l?>m<8hVM8w1huxoK"
    "&ySG!gVtq86HZUE$D>}?vA!d58{?XI8vOY}rY|?zf^{FMBMt5xSN-NfzrZlY)xb$uw^EqKU1$9-_lJEeiE;5-8>~YaQ;u;pdkw5xS(l63|0oFixxXgbxs;p="
    "`Xe7Ru77-vbwlyz8P{?qV|~!N-Hgk5_keF&#`&`)&<UH$`G!bv&yk!T9)fj4eqFiW!#-Fy<mWNbo;R^rJ?Lzz$a#h0;JfoME_r8!&W%4I%>LXvtgCAEoN@i%"
    ">sVJ+>k^HBj_1rT2eHm;$iE>h&v#eB-ghPAx(n8YiIZa)SMvv9{a5U0#+Bh+vA*n1Pv%#TZGv@k-&!&*u62Wcj;a{f9ly)ufc=jGoFDs&^@C;7Fs|Rai}i-}"
    "UI*K`xa2t25B|9t%>0IIz<R)Yv`)qOyV&fR(D{_exR_x)bcV$<uK(zT^^(2E1+#u1&=U5?X+4c|eo>>!uwV2K<HF0_h{vlsx6hP9>pitWg*pHD5bIKX(=jeh"
    "O9KD=hR%_3&g0sO^{1b&Gp;4f!@AcWhdEz53HBM*F|JSX$9mPbGZ>fWwZXd7KNA>N&sE1d){6ZYml6xYZ^pJ<e{L!>o|=pcfzN3@uYSEKZ`a_9#H9-vxK6L#"
    "#N}gjUXPQvT7vb$8*ehMRE&iEjH8UpzC(zMyEbs0CO+URXLFr;b+PVv;Y7yO5hB({C-vvNNd~k_Z8x6PbAz|E{#sa6hjF3YHR9sb;*4w64-nTAGIO0jE3odn"
    "*QarIu2h%?{ir*Pi|Ix}r}qiQ)q9=68*XM?KG+Dn$UMgN(9&2x{wS35ty#fy4B~v!C#<i3(4O--x3C`nb3Mj|Tt~2ueq2c&f9kccACi^x-P2)T^2=D3x8Tvl"
    "r8jpOSDJMvuI@X@xY*GH>-tY^VO$?s9y<DbZXb{nI(Ngk{jl!=6zA>1+`jsK;^OKKjLXGN5?5z5U|c=1iMU>_43EFXZ0PLD&g~lq6W2C=8^d_RKG0vQGp>~K"
    "BreZLVqC9W1@;TJbDkp~+7-Nraj8`b?JuEEk7Qibl87r_!?@1PdDy4IuM@Yg>yLdZvNh)V#j6unQ_=mz@ts7gn~J!QH7Da@x{KIP<I|7PcCIvELR`-Ih;cdH"
    "5ZHe_%eeZy4sq$pPJ=tI>oOULD}R?5+&OOhUZZ_N#N1JwS6E40N)^DkxHAAe$&Yb8pb_>VS=(eZtCI>@iK|iNSv<m(o7fknhbxO;i(89*QMS^3#`*i&__5$c"
    "o-!_8Yerm2zQDNP;!0e2wP%#wue|I5_Is(YoN;B!7UEjbY23bL2yrncka20`Kd@iYjq7;k$38RB%^24zKBawW<lsu9Sp78Kf&Fc=<{f3vr{EDrT+Q}3$j&9L"
    "9dWJ2bH<hKMTiUeFEK6_c}4r}C_ncF+5HNM`>_vC$yJQ&(_*lnPwMH6t9!Z;7gCQ4Vs#Z&fw*?PH{(i`U$l>qx}Y`V(#A93Eo(9^c+UqfRFrW!Uw`8NyYX}V"
    "v^9u}-`)i>?vWb%AmzHrxc=nw0Qfz|xcFc>aqZuYjLTyJVgF(-*AJG6>n}qYSL<a5KQoZ=|GnQ~Kd1xk8JD`R$G%ek)ni;6KaRNexFpw?TM$?O%gXtCSMZgz"
    "kBc+@^N(piEA`WT#)a2Au`kx^B*wL4;l!1j+c^*R2H&%Y^UFnvYxg1<mnN9|bnP6*{jT|!xNy1?=VhabE2$;M^^cv1OE291PyD5<eZv@6rhKIR$E2M<1DO3j"
    "jkt2=k-?qgTZVY-Z}#RK<JyC6(CNR6aX~CkTnJmr`HgRVD4t$Xxc^u>*Nzhxo{wPh$cJVTmwdV~F3##lTwdzI`NwL|PpZhdl8U&PDKF#N<%_hBoYw8n2*w2!"
    "`^$B|%(!rR7;&Y{D#rD>^<n>YG`GK+3H#brYQ?yc?gs6HCw3{w?R^r7YaL%3_aDc4BCXdb@YsXk)^~}=O|j2jqj-bc&gaUB{rb+E`&3!_YY)@@ep+<3;r4z}"
    "!@Cn#N~UG@dZ{0@f1tPu_dSQskU7|I@Mj|9dSX@VH`rzz<7&$bJ*a=rc``099!y*+Ux@4cNe>?PdMNYTauxPb?0SfCHD43#qxdnNaV6yr?Z+q=>dm-RVm$V5"
    ">|1T9J^$+LJj9jcG@L)$PWwO#N3Y_3Wz9Q!*ADwbu3Bwy=eZ<L-q1diO7K{NJH8jmd1Hvn^*o2z`%Av&MqH{_$l%WR^fD*0?`8TogYEq#kL*KS^gG13aN{rS"
    "pD9&I7|im}XbJY)Oy7rb>0@2+=<1BCX|K_~oocJJj0-JB64&Qk8)VOq9G{)I)@u#p`qB-wU#L)W+#q`%gl3-LnLQ2ed=6>$jP@PX3Kbk=Z<jbSlDK;6rE$M<"
    "zQ1oQPFx7tKhT~xIlD&toof4I2ikSCf!&F#X?rj(to}*+sS4FA545*ebD2+EZv5A{UpdxM%EfAe7fd#|<GY`7@)GUiDt4s%r1RXIU&Dy&Pv|~ro5z@mxa!r6"
    "?uU-w4JRgg6Yra6fW6-2k{-l`u21{h{Yq1g(Eha2$vyq;dC*s+(0;ahx#<3OeK}Jd;^MNd%wDcNiuT1-npI?6P&c<B{Yk(5?Ocp`N&D_<t``}Xo0KB1lvEiP"
    "ru3lwd*ys18JAAZqy2uRR!zBm=q1`8Sn2G_xY{BMaeeUPeoSY3%a)X{BD*<X7~X>NHY$p7Vc-GUuUL%sV_Z4?mG(E*>Xq+D{b>CTb-8LY^m|HQd!4B52Gc&t"
    ";)e?ccg!ymvZ_rf{)Wr@+WTD`{U7#Wei_u)UWfXQEW}m4d0&>#2Tf@|XGzI#aOeE+(9ex1p7O8y*txW%Uqi~z?}LmhAI{PK(#pGdZvVM%J>moVFfL43TZh_}"
    "ttRI`%ho3S#hEzYIkP6|w7%V&@?h;-HLO4l^vlNHluv6u>cknepSJQnq&MZk+CNSDU4`;9+{@tJe%9}pna@@xotWYV4@htBS0-(pS%Kmy{6&^*^Fr0t!`!If"
    "%bb?UuepDh7|@|K`K`7P_U3+N^6f4qh?f{*aL0ab^26svk+*sVcdW-33Q4p-xjr>Bbj<m&`k@`J)Grga^&-Eo%>CTbKFdn&de_?EYF%^xG^ys>)D-9GM|A&G"
    "t$DWU>ko|$ztux}+S?^n&9a8#s+6-Q<;QH7k~-lN`S(ujLG@;x)B4OUjb`X3)HAqun7Qwq{&;T|gX<H{b*DJ3{oi^X$!Tz{OIL%-?&f}U>hs}w3@$af+l~A_"
    "Gvn_#EI-+g?r(6>>$$m)S+P2W46Yn_ME6P6+Ar?IvLeKP_3A3wt}iv^R561K0k@6&p<d98N1bpT`_*sm*hRAKx2ArdTb%sL*Zrv8tbOg)wkScoR7XF`o3;N^"
    "txLqEF;|THo^!t3%g!YYE;ni4nc_G1j}odLA+9G~>O^_?ZMJuJKH67bI@-?Qj(MS4v0lW*!bu&ef6a4{rOoS!3-y|GB>lqHeeDTxCG|dEYOl5LQ|=<Ai6>Su"
    "xU$nampR&#xUg(9-Nzi?!ygPLt|pc0K=o;zzq}}pxK?bDalcCFW9=sv8%<nqoX>~)(YzkToD*PwEY`R$InP1%9s<5Q4e|BnK4#LS?uh64cyF>dxxCK{@tpb8"
    "j`Xebqg|R1ms3BW`;WDMmm1X-yx;w{ls9YKD>nlC!#(5v<1*H?*Sbd%7vrwCrhYfuD;Hgec2zm`5B0CPznOmYU&M2LhbQIJtOx1BE3_+Wxp6;n><^}e7Ar&f"
    "e-+o#UJttF2mU_TxSu%oB~$t>BCh=%-dwWnqa}a+NL==jjr&HxUF#g@(H+WC-U@pe_YKFsVRD%*#3kQWO{ssax?PgdjoPKRYGUVVw?W`*Ye;q-edGz^^5|}j"
    "D4&*nOR*gJJ?h&)vhCX@$sxq$>n-Zp^B^p}L0rjQ$G8vZ9#(#qH7id#IoxVVwt2tGq?5#znt6@!-f>>DxV%LLit}5lYGj|wx(}WwuH^dYF4^|Q(jt1&ITzaE"
    "^TxRD{5`#D=1Syu%=WU9?Ya|uE)v(4&M9fHAHC1P%A{ZG2aVs3=UCr!?v&>isq#toc?G#6>D&#a!hhLF-x`;?yH%z7NiLbvSa8m}7OS5nu1~sfhU~2U$2v`-"
    "b3K%PD^h7TeerO!>Kgrjn~d``oa^rA7pg(}7yOKI)Va<*Yr2}$FC$+V<Dz3;y}Tk7oiieSIBSeg&h@svQ`IKFg{M&+InH&hn1RkKQI?f6^6gk(pllWEQr_m="
    "raU>;bER#Cb5DNPFxuzXmqxoku|DPh<C|)Bt~@wO=cgzi7uK-Hsl{c+c`Hj=)FOMwxs$$)DgT*Y)wcUpHu~V)mY_HzF2_1?WzBEm!n-Plf9HKL!!|nqMSXSJ"
    "@b5f-^UxqVFGl(1W%zgO`ymW=^PqV4oizMA=8Xt%pAc8BE|xU=Ic}d<(m6H4y-rOuTfa-qWIESIte>N)W;<6}IwQ8Ac=nuaYPVN{bmHQmX@(!?K0N6b(|J1D"
    "$5zek{UzM)(2DZ2Hb--NyX3K%=)4{!_*!#&zY8@E5Elbhw$N<r#pDtF=-eN%LQqS4y~&5O(m6r;2@k`MbKd^HJBiC_3V7P%5nKDvc|+2=51yKBUX$?i8F8_C"
    "vf;=1+&h09om-^D>}pMMT7J7_pmUDIBlC>^&$*8K*IeTA;o!EKeIK@u#oAFE8~b@N{V@-T%LfOv(`@V7)HUvO?vmWJl=0U)%-UCH&Jg0F+gxwWw%?A{b`{QP"
    ">XW%W(;0t*xU@Rd;Ewfr(#))M-jlK-#rW&ofA~-X;_81Bd^Fp<JH29m;&Q{!K9o0WeqE1v;zD?UaXyvfJyFtk;GC&huMF;7kEUM3IaAYy``UGsTc6=K=!36j"
    "+uuWL?Ly~R3GQR)d@XAonLN^sxSI5<qh_mrIZY#+dljkB`CX2ETK2bxeWmoBHT(KL7k}b<o!HKr{d@bR@zD7vr=Mn@H&`&9xD>G1PqTdoQ0A)O%S&|8Y}bWS"
    "bUSfz>KcPP*PE|C4*hS{yK1)j*8;AAx7fkB)a)tv)oR`BT+H(g`SICqaOXO7DI=X1rX|-g&Uth0(^9qoapC9*gFDwRhr1CMYc%g+uUl<N9pduMGd=A3QgSPt"
    "N7k@?PsU68!SBfAo>XsE{kIzkzq32_vU8#9IO6)++r8|5)uU0c*DA?&dnM;Q*gp)CHQTx^efw(g^m}B@zMi7lPUt-RA=~xEG{?ZB8}`<0_hoV3CA4dp!noW("
    "?|}Y2#<+g#E$sKEH_kJ3&OgkPf;_}~8r-@6N&QT|l+P@2;8wjI$phZ<yuoej)=PjVxb(GirA%ewN@3r=_WD#)>Jk@jE$nNLNB^%mac#$gzV`g+k>1b=E#8mi"
    "ZDUvXt?A!SvyHc6*?!P(y4v8j`Dt`L3+m$M{aBm@#=&p8ivIR^v>D;hKOD%oTrdv&?LKbrH4lED{qeW!%STn{A8XX#&V{Y(5zp<g{wxpbPUxpP!R*DAhhVQ}"
    "9Kd+kY2>G18-qLNi#EATT&fdK=a)L>)&IOjT>O05;LiJFgU7`6$N~d3+jXZcc@6u1T?X3qg`h9+yL&O`*MB1qYwmDfKRu0)+S<Z{>|Dv51N>?)#&zHPILB~H"
    ";vjoGQu*TGSza=(M3g74w5dAS-tS`BY8aO%jWoD(e~JBdaUNo(T?TjlPW@RTuDANabhI`t5oeDEI0x8zPCM2H_M<}#?wr?j%?I%;JT`>YLpwj{oJ}*7#h<+="
    ";w;~Cs67u_WMAmiiW<t|X*mdfkDTGW=5Xj-%xs*)?A#A(Xdv2KqP4*t-&cjrxNm74V}`N#Ur&I2@3X_~`m%p0aq)ci;dU-PkAUCfK8!1Ur^3F+T*mdsvFOK6"
    "w}!L+?J<M6tQ8qy=km=t&>!5JadpN*aCI%?iq|seWPU$_#hJ7c@ti0S!0JR=i#!afZ*b>4x9=N~=M23V7dCH0zr@nH=+3;A-wobl732Egf5GMBbZ)uhd&0Uy"
    ";JqI+t_sHxPl<FR?K)D(DYW-lnUQvTxz;(@Pyff@&fhQACBwe>P{yU)*N~r)ahylr1kbvKaV5<?<n3ZI<EqC)^y8iPobP)I`-!;%?Odq$3i%AG!MGUw4)r$0"
    "mvQOPC-CK?7}wmtL#Oy$#+BzOh)3JU{Z>eYb5x(+<n}|-5trZo;PwwQ5m#pw3ZlPOoN{*J+OUR<3$Cuvxh4m(evHot`?cYmHz|yG(kBMl{pz1Z@W3R_`<I5!"
    "kC%+gXU%h9^JN{y?8{X~{23}SuBp|COFO(qv3A|BMO@!8oN;MkedznmG`RD3)eI8!_wL}lO*6#*@QT5m^M&TOf_>Z1jH|9~k+-6GN87nDtv&dO+Kh{7I)RVv"
    "!nn4-E8@>Hk#W6BFZkWKlyP-QU*xl!W^l)QlR9qzbh1A*xa0eVwtNV5wxt=v^0sLN^3%8U7<-<@Cqc-=^j2e7zwaN5_+th#d;LH#bRI`DE}sfTJd-ytuBM4Z"
    "KVHAUb#hLH{*w1RPTyF>lQ;KRJ6C!pfNM1w7XoJ^Kf5|}9z5U7+xW5eeisfcM*MddGkf)@3N9bwey^-TyBg_?Yxmc}{#&YXc75pu&Oz1|m1JC~whg>vOU6al"
    "UEpg5aC`54@cVum<7&@?$fvfE>#RCLT<Dg}xVG*Dxc-50{dp4d*)-31rt|$g>L+h)#+AR9;CF=|<7(sU(9a*j`Q%&3&yHmVcdkd4@1ecPhYc=Wi8Jl{KZJd^"
    "2jCyfc+{uhacL)ze)gkiSHw%?=R+xj>jiUytLNTAe^o1k|L;07dE!U3_uqjAfBzx9sk7iK;$IbQa3M(mSJa>2?i;{=nsEmIfxh=y^6T-%dfrG)=VGg`UrnI+"
    "-<kOzPw&Zs{3Khj5r3bTW<FnK27gs8nDif;`MH`Me0&FkZ*|d4{Y$xU9{1i+!IaP2W<OraOI*o0*Wdx^|1;yPUWm9h`(J~LUYAYYp%`)f^Bv-)FPM3*UJ~{V"
    "e;Zt_d&cA;Wx+p)A;eFb?HX5sxLB%b2<7>x8GnR3^w0G*c);C5rv3cti1WY62KV;cZ^q+N2i$do!KL+kOr1szkcS87;diHbeI-jcfBgHW5X$#fGe54)5$E`P"
    "6E)-W<t9@nr4{1LRA(aPdA;fPZ5!yk^_xijwZ`Q4z2W!7c!SICt4$r<7jYI{G?D6Xg=zoM5B=_Pko+E6X0~@&59s(moJjSy*o<>hZ}@GRb`s@nq1lf={SoKd"
    "a+4?z^GyBHgW<Pln@JSU95WBz!{N8^h)I;^nWlY2Ao7rJ<|K+U!Hnm~80e4KHHq>$-K;|)7&^%}3|=KA#<c%33HIlHfJdA0KMP0yt}hsB)aSD()Bg4pw0Bv3"
    "@X2Po>c*gM|Mm){{6v_3tH&cwF)Wny!%UrWv*5S%icqRg#cWsAd5H7SG2&j6OkRHx^4b4MD781l%+K*<$lJVh3h@ahzp)bW&&sP%-o}~nyk28oFJ%?V^H{Ur"
    "Gi-$YmO2La4jXN@%WW&{pL#0P-XQaO>9-SpCwEgQ&Onow+K2dCk5H)J1I+l(9)!-;aE0n~xEbgBBgn(2Sq9fVhMD;<c@lXJSglaq4l(l<a0WVE_fUKL3^v=f"
    "`vT%GpM>@fH2wa#4E<NP44yo6fOX&h5B3M&qFw%GJQ?r6Z=w)J`Rr%fd+G2yI#(Fwt*>cc?lJTymk6W&?QPoUcn<sH)xszq*^Kk^E95Pusllc7y-fS&?-9>8"
    "-!RHsPgB3@XV|y*52O6_F#D_Qcf@&RbQslncQgNuQ&5Kwr;vTYZe~7L3Vvi?c|jPpw~Og_Zd&5%>U9Qx|H049bB;{J#g+%csGd7p<5)K0+UIiyS9W$XdG?&p"
    "pK}-X9nE+u=7IgcZ(;9i+V3bpT$!6Hobu4Y^t-DFap9j_;biY)=3!rP;!^68WPhc-^}JUaagMDXPI>k=+grUnalJ!}aB6Qm)4p|O#M7%Y*{3(Ta;zHS|2v56"
    "UA(O4`C5o4X+k*lds{RAx9Y*~mY8s=!#3vi^0X1`;};t|c}{DyUBMpEU%4fm^8JsgGruL`Ij*5yo~GYwZJ_h%3gR@mcGDYq@Oy?hTbb?p<%{;leK)wbUrUoO"
    "?gIPkSt5)$V_KN`Kh^{Blr0iL{ngy`JFE}dHK$4h<+GWoGrK?f>t2%x%BP3fFLj5Ye(H6Mpn9|R+4CL&-n@SV)#nOx{y?W7#FJrc1l3y`b3Q}2vB<-zD1*!C"
    "UCntE7lOgtErib7!?52Jig*Io8(bWF4_x{gfjo3QK)mz~lS@-!U;cu@l^BapjzyfG?}Og~SIW)+_xOZ(9)Zi-=73j9A4&1N1lJEPK)bT$i=;aF0IqFWf_@As"
    "YjEw^H#2^9Ir!7s23L0e0T*IcgV$<haD8YxGjILZArH;`$gfLQbAFNMCbW0Vz(|TKmpP9ma2xtDX}rOuumYz2%w6Cq(+n<piRQecW&7aw_9D{xSjOy^bqCSj"
    "O&blKJhYOjzvu{b3Lm6)9jb2fh!d!*KIg&fn(+@zg3jW*;EhfFU*{43ruWE03sa}mW#lte>dD66B(qM$`q!bK-E}hA`<V7EZo&R(NrSumubR*<=^l7U4TD!n"
    ">0$bv{t)pGZDH`O&-$ABqo2ZlV<&^_1qYe-yIvyxz60TRxXCNLgZ+;Q@H@)%yZjUSD?X0&T_%|Ga}vM7Z|kK7fB#{U>38oh=r`PAaOp~f*<Y!JE(TYM9zh<a"
    "nmUcrf*-z2_Uq$Jzr!*TS1UfAO#M64<ndXF3;n)8XRfI;#f7-`BHa{<XQ3I-vfSW5^BP=sUuNno&JUf1Weu)-tTg%U!mw{$*WliMYs~zd7ZJ~1Pv~qguj?J9"
    "h>M@QOrd;kHtXcF8`@QDh`|&3Y&ZM&bw$|M4Vgmywad(pP?fk`FAje9neFOO6TIOvg9qF_VCr<J3;txw6sm{AX8f%h!f#)V?9(4Jxmy$BQfxB(o-lRVHAlNH"
    "KY;#elaKO5eq29Ip?*1Q@|kU6Ungx8)yV}jKM&f&zH;6u%BOX1Y4J|5e^@q({9ZBpWk6Ttd28J$%FlH(58r#jerRj5ce!qzi`ozIU+)%0@!T}?6FLa({W>Iy"
    "`u&a>=dt17)j|-@eKXEfqloLhW5Lb+HkCT#!AC77Uht9Gk0U1`p16(Be{7vU8;O1#bkN|*bDo$w&!(YYs$MX-V;-F}c?NVc>S)(XGe3Lgfot#4E_0rt`e+Gs"
    "`lp&o@tEh^3V&824|j7K+<6}EsP)MIKP9J9J-jpRXKh6u&QzaDb!E*1T)P|juigwgAI*639W>k9(csb*^SoDO+%fP&{ijl0S^J9yCn28iW8l~1YVbwGUp5MU"
    "&F3O@?KS9JoM&+D&{>mbxnt&M-Bil+ck_Pj`4Dm5*=KNNXKK9f%k7?{zY3p$ekzj(zlHsgThK`{pJ#@BhW)vh@cYfYzrXp3e)LU2{ML7jVyU`PyE0~rrnp|4"
    "&pB^15ZC;RMN_{#GwttZ>x#N{kEVWqXxhJYh0fl_(Ui}-rhUeO#O1o)q~meZ<TXX&!Xw$>0e7#O@#ifI9vEbBwQjO$A5;lC3nHVbuFjeKM|JqEHaD8ub;@jS"
    ">3YQVn~BglZpL{}g8k3Eh+i|G(`L6syYikjxXaiBrq12Ah$rhEgUjyQ%ziB8i+Fy$rufsF&tpQruEfP_zoV%iuUqFV%kUeJeVXC7^nA14*7Zl6Z;KjSS|4ok"
    "?L*<Wmb<}|hmJJsJTVa5qwzHA-w|egW*JXh3G;@2plL6MqFu9lgO4%CpLLUA9}{Ts|2;07_1rfW`dcCmzV%tCsXucj+SP3i*{hLeyJpRYPW<X=l!s{ZI*nb5"
    "cCFuq_^tch_f^Q-pOd8TV%_g<Z-C!fHx2&&gL%IbL$<^I*h}avF!M8fAM)w{YZ~>}Qgi(2ry<V&vcwqmR?xiP$)ispp1{H})Q@YdcrHMvK;;<HH}7}qoomo("
    ")hLGIG4FTE<~y+O-!6vozs-v05%Lfr$56X=n|Ut(68T?0GKTusyx-}iKH&Nq8G(2Xn(f_?r5ow5ogPE~pY`1}Di?V4GRC#O1&AvHw=yoZ5s7O9k8obREO_0^"
    "jO#BefxmgkxcFZU;^LQ2-2Oy;;zF8qu`JHnO^K_A^Dr*?c|vDE8OF8R-iW_lZN`;+onc?Q731QU9>nD$UAW)me$Yu7#O>z{LEctOU|jbPM0*FsaQm9$p<jM6"
    "=f6VH-m4pVoNuN;=k-Cx)njq6KYW34Ibt^Q)Ac^jPy0oPGyi)YXTz0f*J~lp_-oB~60&Y2uJ6ieaK}6u?eI>-->-zhoxcZ7I*5FhtRBbm=6w=zW^QJ1=kI#u"
    "FQQ#SM`o`*{||nT_;a24I^yp?hH?Go3$!;r3UONVzE*yQ-`M#Ecg(97O8iCq@;Vm3G&OT~^vl1b<2c9abUxyGnX?9W%!AVJmL{%Ux@~ahzV10|5SOODhK@CF"
    "aC|e^H~7u+tgY+>{w3RV&A$Kb*}>pviyGXye|uVmxGK9F+&TaA#$3ezyYY0&v&DaJ2A}55xN!J1anYr>!JX&kym>@ii5_Wi=RCLXf5BgdO}Dp8O)T7l+FNbr"
    "bbJ2_8S8^bt{{EKdQ6|L#FZD@nT~vJ0&%g)G3Hk&wir57uNd69U+&FA#PwB=U~lbH*Z(1L`QRsmJI*^()@SHR`8<*)UbC$yRW4N`uFiBdxMQA`_^l&xC8}h+"
    "z2C+23UMi<I=EFoft!fSwLRiB+x$Zz%N^oE2_J(y_RSMAWbZ{fb9={AyR3O2-X6sD1|#F`@r#{<iEBYhJgbxSJBh1{<KrpMRzGffM_gI9l<Difd1ca%-ehp+"
    "`3Q$<5tnlvFu3D75@z~=FE|sg+4svz8%$g%do$kNU&^m##HH2G4DLK{vF-`t>iRD%9&ynt=*&-*pxNd-i9We{Q+xd~CfMsl&)5Jwe~tv^cd|cmVNsq0&3>Lj"
    "r5VJ<Y=sjv`+DJq2Z-yfOBmdF4o=1w#N}YO1bchsRC)VQyLMJi&}{Qqgf||<waqmQ?wHRhB?l7M<Lf2Z>sHrR5EuJN3HJWgGhcyz-WCb={*}6>>r4H(wl%X?"
    "U)Lcne(_FV{Z)4aakXS8mItlXD&j(^ZajYPTg2s5GV*DS^V3}Vk>7j$SRMqa6>&LkP=dYQ<is%I(%)e$pVIFA#KjVUXqOdF<TvPK8bk3o_Bs4n-JkNgVm#+J"
    "MiN(kge2JOM>wz@_FWa&JI<5(NL&~Y!R@`O^`|)VPhni09|%5aD)Xx+?I12JoW|^x&fkcOqhby2oG0A6F3smtQcZ`wHScj>FmZWiJnGP@hn$Cri{BDh{?(nS"
    "29i$8nXJyms~*JV=Cc@A?@uMJ3$qcAHUG`?EOBY&Y+i4ZvJIkk{hW<>Ed3B~;(C!e26yfY+GGyw%gkYUkeA&euKXq)=RCCk3J<1s-6ZauFI1%`aVcgt>c`UW"
    "zLL1^I-BKRTK)ulJhj&`4@6&47W4d8B0rAv9McRTE-#qD;!%Xn#C0Ws<xNk1M_loVM}Dk%ddXFXQhT#bXYtE%LBy5yvCy~b&0_~~y?zYKr*Qu>ad}}huS4%z"
    "!zljKQMkUW=Z!yOiA$F!BM(;mzxNW?^azZHj_1Z7;OoMu-yQ2F*3=tL@yMYDcb@keGJ&{oHH3{1LZf}K|2v+ICvut}#KkXT*tns&)*M0nSr8wWB1aLI+6VAF"
    "=ig3TUoaGLTKNol2d)id<EOA8FU_}8&h^81V%_)4)FdveGU~yqlQTZV)hgXt{-xBziECdwQ-3+uHLKI0)1(9LJJ$WB^BUrMZ!dmdm~)J{(99F<wd%0JUE<P&"
    "W_)~j@E!J58Y7?9JVwvlG%ry3To?6YolkMYow&5E2IER#YvSq<chre>?!(po#PurWalKpTJNyVEuKibvu3yKzmum~bPZwi#t9tJsE-Ws{$CGsDVK3xi<F=aS"
    "3F2Rs1MRZvf5#u#U&zGncT)L$ffQ$<v@9N>Rb}F8uHW(Yd@3$2h|ASJ#M|}tH$A}LKaRKeqcCtZ^taxKr#x8i5p80Li;?H>+-2!sS^>XJH9WUi_q#xvk0@xf"
    "cHlY38b9}6BrcpxWLy~agt(MyA$v}fpZ^5^9?PD`v~n&qzfd_78gI|PD3&HJ%@0KU)^l}eUE-RjKYQNO7q=xYAL|@%uTQC3Z`kMmhx}UgEcPBnTp!iI;Ldew"
    "?W5p#MP-9K&!eol5c&&5o@ad{adA^FJU3eF4euR>PGnj<S6k}=w_e8l%g$fv`P_Oxk_$dT9;!ZP&+TfCZ;12rO?nS<oWGNhk>)$fYtFLgfBkm>@Vp17+vC*I"
    "Rv@m-**smdjaTyJ`ox7#%cfI%t@}}lHsCd8u=f<bMR(|Yq4yo<JjPr@!AA$;{mbI<6EH9HaNp_nI+SKa6W3GwvUaIe<`I`FH)lH1zEy~)cCG1}ZQK*aZHIkG"
    "8Kxurdl>P2%!_<l`rFRIJ}JX=d%x&6Zi4s!9!L6Ce-(cUo#+>F_WX!vKf*r$tvJm#{%FtsfJ^jV>^N^o+m~fD>4fiR_F{BC;?kUrahm-(d|pZ7^76$7cU~9m"
    "-H8kFvF!a?n_7>!b}+=?j`s_7Q*+|_oT2=EysJ0jnbDnn2he|Zg?&I9ypLPw{w(u{y;lRWx9VIi6+m3|tH}Av324`_!i>u$CPQDzigsD+d3VRdeqc(hy}#sl"
    "3*fiUYlA!1XDL5dB0pvB7~JtZCjZ?8`v+%ZHTycvA$yRwGy6%$8ds(B8tkWSFu3EqUD5Llc(FyXEDtTOAkNv-VyRu0PU<_zf2j#vC*U#Kdv_4xw8pVTuZfG%"
    "U0A!M=bw>J!PDT5b(dP&-{AA>#M=Aoe=nSvFYD@tc&z(rg{;J-tNB<S^f|8J12e?h>q@^-khuKhYm7br;`!plm7Y)OJEY^ByMAuad44TMvp+9gsSKU^Czy`@"
    "um*8q(oTar&)4>>2mWv+zV}-1O)VS4Z{u0aUOMT4_C|yo+%ayeS6dR-&y2)(UF-d5Y-{lRz46`HYFFuY;Jv+L?ENd<@j?8djTskXIuRE~)Awq}`?&DE3*rne"
    "!rH5K>j9n1Sz}oK56ZA#Lf_9F<B9sWFZj-v`2KI{)E{8xnZD0E&%GKo7`$TAG<%&08~$Ha=N+Fz_WkkEt=^)nU=R@{dW$-0B9e#_DcS_9L@H^BWI~V-620$^"
    "wR-d}!HB+MMG1DJjF2V1Iw7p``^|jrJkOlx&-d%R&OP^Y&&+e@)*$E;cZ_iPNhowywxL{uJA{M(wF>hThMx&h;N|9HzQE97*G$A&K4PI_-k<xw+2E}vkoct*"
    "@z7!Lu!Uy*)qn)VAKM4>F9tudK<CQXwuE!<7sAgfo^Yl05|n$cvc|3dkG4zEzNLR)UdiakE?f?KuUAQmdAw`ZO8D<_KS?qD|4QMj5szn9lG#5Ouon8nq$io{"
    "%Z*=;a%b#JGS^q}-UvTC))LNn{01Ge5(($sHX+WvQAy_d%KbOPK4W5%Ip4%SThQ(&ya<=NY=iy%emb|FlduE!dF=`3TJ3@k4;m3JwAzjOrdOqY9QMNg0;lU^"
    "*{9iPKjP1Ot<Nu8_Wjj82z`nk5U$u9Mw}n760Uwb0^Q`3gma&cfv2Tm{@T#t?Fl@m2iFrW{VxM?rY|I%zkCYuWXvR7IjbO^b5kf!&qTdcAIwu5`b;?oKY7lW"
    "?>6$S<OTTI(J4{a+j!o8y$C&jZK82&|IO(#cw`MyuAK8HxOWN6I~)C?pe*?Jd$+(Gk6hv}v~R$p1;l>MHRv`kn{c(wb?_Od2^Y`bfd8d?bpMuq@0eTAA$cR="
    "!si^6J1LoPb>|(#KW&!AE&mttlE2|c3c$Jr<2y7s7xB#XCGkie_ZfE~_2t_<g#OW;7m$3Y{0KbRp4xxNLp-7ljq~mk)K@4;IG_G6{MhU3Rjm7BHatUpZSpj3"
    "9nXn)4(@%OaDMm;@QM2R8EZZ`y+Yph+Do`(i~TTChhHh@-XWhWEhb!f^dA0;#}F=_ErkEi0kquhpHQ#szBK-MpP}=67s90p|DoJ(ohf(z20t(CDYyRtKSef#"
    "i=~TU|4dOy!sWNcbuO%XKaa$7yQI#!iFt%`>7~K@T&H|kS)HqnXRz+e$hRjy!Jglzulur`$2g|E&XwA-#;yBSUQ~deeksJioKQ*UeC1fe)q={1|5+f`(HY|;"
    "?W*cr`g0`l&&{j`{{^n|%y}hcR)?Q_C&JYVHt=)7LF3l*-~4Jq&jq#Sk#^r)3-wJYO}O&3Hh5ZL0?CK=KZDPDLbwuBSLfo`n+fLlh1s^?Q!+Jfx!0Hz>LH%x"
    "`xDGMi{<Ns_mvY!p4<RedZr|hdaZ8&|E*#PSNwVC?-E4$WpLSdl*X;+y0mPFdd(jMZaj}E;8IN21fu_Y#>Y1!{^b#kQ0}1G#GXISxO*95&)YUe`;PjEb-ae3"
    "IB>x$pKvaZ@z`60OMUIY)6NpkZvp3u4p2X3n?Q%$O>@ohNK?W28>xiz*T9wYadXY}65H8>2L%(Zt^gM!M$aYr_7R-l;zqdY+Z6Hd>_)h98l0crocd|v0Q=f?"
    "2<H}oOIOPfF1=#*Ss&+^^GX@g4F0q8=a70G1?O^a5w6-cXZdiJ_>tqmg;56xSMwN`Hc|WjEudReD&ayJxY{9(`l;9w+&x(1)_bw*G9ETs<JRxZE{wamY22F6"
    "!x<mdZ4S{V6kPOZuCK$l><3%~&Uw}$oZk*EcPLA^bdK?=pW=xhm2ownaQ-WcC;t}VQf)`*yx}bM(*c}Ya)5Bz6I|W3iEuR#oc}#F-mHU?#Qev`5iW0G_DR9i"
    "kHX4bHJWfS7hFE%M*Q%fz=e!%gmcwfvHWjNxatTlU#dgn`ON%em({pszQ%}CzVQkBm5lG)Crr=T|6*T~F)q{*Tv(YS5&ah+-?-!FB$A)Ek+<^i2lahGmU9>!"
    "!TG_PHEw<X#IWb3?=s4>@!Y9P<FTK}(4jti9>;_dE>FkvuD%>gxH1dx4f)Psja%nCCbIX{m7dr?WjseO@Sfxg91)K(emjl5hyS+4{wbp!3-Q0eZ7omA6(jNg"
    "!i5%U`^BvH`u&0bFRAxU>{BzwTU7iHDg*b&k@}|Me^aWP5~rB{cVv$`-~)o<%=!x%_+OSw4${lD+&kYA|Kq%EvpBQP%4+O?=69FYxb?fK4EhQGpBS-bE<~cg"
    "p-jlcK0%{DoL`we@3L-hneQHk{)|v8Hr8COkcs{eUuqQLN^A6&<ULNYM2Fwdzv8FVB3!6cmh}?~XA>Ta{u}RelW_SZ`g@Y|{@Lbs<irxF*W?spuUupO$)Q2B"
    "N&I~{)Hikz;o_0+Y<#R4;Yu^~W2JUw370mb-z!vn8e`^MtuN@`isxfU`z98F`)np${qhn0<TdfsJ`DYHv8_lrmxum5*S#mT_d@@l^R%V>9@{_f`E{0=^Zu_9"
    "Pl<b!|NRnvQce&q_~1K5xwDpV=_bBwggw!O%N_;bTfHeyd&<6FI_UeFE&YKy?0f27Rl>O#_WgD8)l4%NALDyX`u7U<eOvm0?EB7s7va(cd=E+u6RCZZ2k?Jj"
    "0^#zud$4cbk8rL&zC-!#ji{ewe77p`9NSmU^!$PETxsWn8D=g<-iDv<Cka=+;yYTk-AFh$;U@fym`VNoiSKkN(wlJEJsbH`qBFI>fpGx!P)%y@brtc5AEV7W"
    "%Q;tIe=LV`UyMIU(+<<~2OeF54x?An^9m+ld_#H{PPq8|JZtyi^jv~ajGsuSI#ByujJI%It3?xi?9U+2zrWU=Z_7Na<a!!@imqzhdY)m(Nj5&bD~jY{YC7^E"
    "A}Pw8w?fu&`1w9j<JS4^-fZ0IzkV9GjNeKJF-|4a!2gnEzSR!nT2goY{bc>$S%q;hd8D4F*7wqPjGL*M`goP~eK=(wxZgU$#oK$}r&=`Od@qdG@kw5h=6We<"
    "7~hj!+Y>I<+6n*Xst_(DY=?c)s|e!f@K*5fEX+$7>nIBUKzli-k#k5S2aH?FEf?zNmKgId{A7%C%F<+V&WR*rTvTj0AcE+SgK<=DPh-NR${S$6uvmmyPrm;;"
    "v{!|P;btz!u0ebKlR^23Rq%6dUAWo5@OcI7UqutHIAJ_l?B`AVsNooY7OQm(H|K+}Ck6GrUXAh>i;*W^-q7=Q9FkGr>enb2F&?g5OCy|Lp9nu$3+XvLIrHJ?"
    ")MUcBO7l>!<pb#XJcH&U&SiFlD+w6Smt%_)E-P`U?=+R1N2GqmynsCB6xJ6Q{=3E?e!mTb3kjHC5JG1WE@ws~Kll04^OU|uqF$+;iM{F+0sD0}8n?ceBB#Uu"
    "+V^2*E*%JiZfma7^Pt{_AkO3TTq&_tFzS1GaTqB#Fc7?CAmP$B%-2YN4kTQBJQ;CjG$mZH!@Q5wsI<nd>#_ZQL47kHhnhLJ9P>w9z!}2D%l@cusow}!OJW{M"
    "{uD!b&oSWl{WNY}=N2*w@igcXYSvBJ<%e=R*Cbqe<^$b!JPR>%zJWL1OJ6q=E)Dg9Zn<MZh;Avv5YMHb3Fj^jVegaM!6Z-0c%t0gWrXwnJiyaD2$$mAz&Dp8"
    "oIB{s-b?3$NV~kjxk%!fc*3O-n12-a^#C{4vDX;@{T;t(>)ovDYHs#p@8KgFw|?(0!ThGQY+4}6s{wtW+pt!F<~$U>3&@{61^RkD%X{D$=1<k+EmO_*Vn|QK"
    "+1Q`*W<B8Fw;tg_UN`9T<8A=4UyFHLF@6Q*L!40GMxKOA<+_0PtPnuzmDvg94!<;o=sdF{bZC(<#hfQ%f6OBbTtC9)&+Xu+#E;2@?{ACzOguq2Ke-L;i$xO7"
    "HENCezHUpndfO4@et9v8_(^SvcI>i)aM`5=%B}UwB$9_*bCi3KCtNw{!2aLwPb7Jm)fDjz-l%cQ_<;Js9_4oM)3{|lm~0EKde)q1jz{U$1bVi=@e66M@ph<J"
    "^kTx51&v|vGm!FKjZkjWQooqvl(Vr;KrC}+0%^y0JnAdWB3!lsSMxhfAo=Fh0QG(FZam4KQT4%(?IB!{>LH#F0fZ}CZDC)%DdkypVZZ9}IO69W);}oyWWrS&"
    "a4CHx;bPZ1i2r;o!j)0Ap=X7g{)8vgg8lv!!lhj`!EN09iEcM+5NA|b!sVhG&^hOvNb<+F2K(QX2p4)|y@wLgop33*8tgX~jwODURfYYxK*G6WRiH!L{4pe+"
    "+{%c5;;1ns-->}t{c}c}xny4n&xM~m<*pSGkGp<;vh_I{#@4YUcGI|Z9)Cc2#OZcW+qYy{@5Fa2hj?OJj3n{2`3Zi)Hv5^m*r+V*t6{&a<vFch2Ju%&^)+*~"
    "Y-!fdEK9iZ1M6^<0?CJzTZnV_#7>{RNxce+!#+BcaOGh!=))I`FmwJ6$8_+g{OS+%zuo`wGIQ?ccdSQh=0~`C8S9AnYx+66)^anxVm;M|frJZ(|HJyFwK&(="
    "qR$^+;OAWz!sWG}!K?2dO7bVA2=Ux#M!1mh3G1v}Hw_^?;v?#{yB6W%)I!8FDAm)<B@yeU_^o9Km%ZM@e@$*M;coBPy0W+Kin$-v>n(KX`@qf2)pl>-|JKz("
    "X3jgj2A5B|5_`K>;4Vi7nz>N(CDxDa``v};UjggE)X^)P&0PBN9DcaD0|<Xt06o1z`xE}`8FUL4`w^b|6y+B8=u7xt|DwKCyZ0e^bvhsZ4>#>?=HfxDXH#0#"
    "5{RGF$KZ#`YxgNw*13tkd5CA^8|_{O>-v1VM=1BzZ6}f^B~<9}=tyV9v`$$3`~ddOYqa|=EYG{J<v#Sujn?kXu&z&b$9g=qmX~%vh4sFeGP#KJ`pQ=3a+ULc"
    "BOi9IXioAi?k>u8oM*3?znkptpkCu!>-Pm&=8v*+;3qk|u9-`VZlT<p7pt2&*9q$h<-*j`X0GI3N4b;6=btulac(y16}<CPFB2D9UIQPX-ydZi-@AJi<@S10"
    ")6BUze7B2E3mcfZ)c6Yc!y0zP{;$iZm-8?OVjp`6_H7ed5^jGH<*u06hVYx`QQysT+6m@%k2weKu&slcD-LJTzMC>T5&IiwV4v~03*ob{o>Od8x|^Bv4yO^1"
    ")Ubz{3%5?<x$qQv3FiEtn*shjO)zu0V>;H6b~>uvvtxY^JUota%U$eawpSC6A^y>K`x4#;>rutm&-<DENDmIfeq~XAGv|^ILI2IAor!(-gW$1MUCjQ)g8hhp"
    "NrQoA&Mn&q9UR-bnz_<rFRPa@h}b{HdRoEJ)6L9<)HLLqm%lsFt@|$6XM}nX|Ic=yzDwc;o8woPZAZBuoITB4>9GwuL}e2$KivZV8+?Wk{Zs#d|AvnU7kgoy"
    "u-a?VP{Lntf_>0S!qrtW?B_%dBjtAg4fz(r4JUpIHbTz|iG-_Z8{mKBPuhK6)_Z4;u19@?7i-+Qf8oSB`0rLlyD!Z0-OU}t`e$za$`Pbpj;#TA{@I)8pS~J?"
    "Vm1)2Dyxtu2kQBdJUPA+_P4eWt{hzfoxe2qHFJ6Ya+JII0O9=cW#A>+`<ZnRjxB{gtB?ATc+yhQzG=Nj5<i=<4qVRsb0pD!{Sx?jI&_qobITT^+}8I9mlh>M"
    "|NQZzNuJDIgmO#0AY6!9h;l21k0JKqNw7csm2fdAk@W`>#u7et0d#IuN+j`2oDcg17s7=xSPw7VO(vZ4PCy=V*Qx!Gx$v{ChChkZWe(ym<4d^MCm!~B8wgi>"
    "NU--%3FkV+q1|2V#u58gvCzNE<Z&cVn$L!xQ@b>7?I+vCfNy+FdA(U^ujuyVzDuqq_6vxsqbM&o1NKWZ#uGhDL_@c(oOb`F<#|!QMS}0^OSteh0`c!nr1meu"
    "p--JG!o~dQDECF>U(B3)I1PODFv7*$F!1zM)c#f|d+z=rT)rHF_DyU+?rD`WgJB;so^Uxm2>$16qxZV*4@7&leL*<CZ7S^lZ8OQNn<59W_jv@lmsVaq1@_6u"
    "2^W`5MmtvcM!1kX32|N!CY!lBZzAe-H-T_28T&_M=ktV12@_bq_opeO+}QE3U*}GF)Ht+b_)^08Fn`1!o<q1iRYd!?trI}(Cya&txY2|Q{$o(DrZVBG&uHXl"
    "t0x+_u0!w~g?Jns$bHSS%SdpWK*D(e`)kCx`v_P1`l4N4y`4((v!@U2T{{L69XfhLx4&ml-f9HuyXz$3Qd2MZ`Mp>W@t;2o<@)yz5=`IM{K}!Mf0d+h>pIDS"
    "Lr||KSA$4iH5?54`p;osoW)b<j&hH-)b0&WUBozl$qhQM3(@$Sw~TWe1|c5dfX1@}**&DZzbnceRHSj`l)<|UM17}r)$U2(SEVH56<yH2uVO;<I2l*72Ov(@"
    "vl<V*&$zrD`=lhPxOV@$*t(R_PwodE&^T0&r<BoNeOZ5dK&T#%ac`_`AMi$#LiKpqJ+0CQ0sNVMuYKw`#>Er8zz^)iefP$_uFHC&UP)JUe%auGJz!twg~l!C"
    "h$?-%qh1v&hH1R7alUB%ZqQ$L)VSqdGwGcZ^t|8^rpIsGYvzP<)!mcB^m2`SF3ZYIOC(%44F8Hxn$9iXP5rSSjoWf9OxK^aukf)Gc<Hwq7k09HT9u9M!QE<1"
    ")BPLYaS?66*L0jFnD-+LbVNK2yr&7K{VD3ZW{AgSrpB%JPM)@ho>ewz+`6xG9roEtFVppMWtI=zv-<E~?Y_o6eTp&vY8~*S-=~@LP?}!@I$vxsU5n?baju<P"
    "MU*>Gn6AffoNMP@0`+?9uW{+nEp~3$fg<dS{3=b?{j+;l)xsCpNBQG7jkoz~#3QLX=ZBw~uIri0>~~++xl%TFx}GQOJVS2K8SJB6@)7<Y8s|ptMY%hxhU@yU"
    "`;O&N>)|K0mBz)^#yNyX=3`&vUoPQ#d#UUktvk4fLpm}pT#tv9E1&T~xt`JCdV76jIyA=qQ{~<w-H*)9ZIdHxQQzG48h`WFIKNT)hJCX8wt_R=gp6G5t4-RY"
    "alXkH7H73|?9WX&r18-E|FAd<*1&%HQH^J>G<5hi8vAy+BjLI}xyE^k-q?qmd_b?)O{Tw6yOYj2*WKZQdEW524g6Hu8LsQkbdzTk>hqM+?H!1RJ=fgG9PFpQ"
    "mIi(Dn0@?V?BkAR<r?b)SFgmr?@ShdJ@#DGjl}-$I;V9%Pg(vb(cZAXeo40vWal@k^SZ!)P>vo?X?6~zT-_G?wd+06c<Q1D#`$#|;=J_{_Dl!<<r97WQk-8Q"
    "LXVT3Z^=bw>6{<jI6~LqKc@e!!`K%;u3Lm)-uL%G#{T{1!y^RK{oP`>C9uC8qVxOg9%t#^Z1DKS8s|M&-b%J%h{rW8Lf5B&#s6v&_Vv47iO}oCo_A3cG5@;!"
    "fAC|p%X2^Ye_uXQ&rf!LtGv+*_9t6L>N*?u{Kk5MzjTk(b#B7i<%k<}xEQQ)PapOk5Zbwd-%E+q^R_<IbAdC?6F9n0_p_4u;cxVZ|1;Sdx9%VG2baFT*SO^#"
    "UNwN(C)h-3?Xt5jtJgVZU%Ru$J>%K^%iPxi(7&Fy#>Lj`{U+YTyfxo2DoQW+JIgCO59r)@b(9{z@xNAX7<3+UG)j-3<)K^?dJ2{9M(O#$bWnrGL1(w3D7}5z"
    "`$5@)d351<t!PdE%O6-iY>YxY(Ve69`qnf4ujim%<-IkYI?j&i^E4UtN{E1cWA^-JWx!ASS-O9|ef!6~G`q*0kmn8mnW1sx$2;}L^A%7;xU^&to|l6)XIl9G"
    "0Z>Z=1QY-O00;m803iTa1#SzfVE_R5`2YY702BanZ**aFX>V?GE^csn0RRvH`1t?;00000s$l>C00000?7eqX70tHj3oN2!BuN%TKnapTpo=6?3`7+XOc)R_"
    "VnjexFv0>A6C$EwLIgoju)7fj6cr`sEIH@gZ+Tf=>z@6McfWh@J7Mqh<Uf9+N3E(^vu4eznzJf;jf=y&jm%5|Oed9g>^!vnpwdb?C1Y;`B~>{kZ~udb4{r7I"
    "^gp;`=WlMcb>E?#G=6CJ*8Mwa|0`?h>1nFUX{)Hp1<L*3{Vibnw|~LT)v?ZK#vDl6k(PWzcp+F(`>KQR(Zu!%&SC}k=0LU4lv~PCU5p4T>nFf+?BzaH&c&zZ"
    "fTHfyo~(rf2vZ631i`I>wS|PcP594&1#uGN6>KTU)xx|QVt6U!>nwtX<Gx%&gykOA5Y~#2|FRMpY`8#0_9*DJ5;Y2t^Jol?#IZyn<ByB`h!Q!-2e@Zf{NxhD"
    "-Y$RMMTjlN)Q_Yrys>c(%pQ;0sPMQGiF{!{Lb%Cc?3|LzRcQQjk>aRN%15O7;<j=kP6l$PeSQP{cJYXoL`)gMFAe$I?Arb~Whx>wQancVOF-^&BBrxJ|0-f{"
    "$J|4>ErOiwXlz~F*hOUKX=5c}Ee`q8)4%G7W0w%WE8hl*WKqcFdAg2#juFHhC+>9;iNe_Kl-HZ@ESm#L={`OOM>~)OyyiWGzA(0_ebUpC#?!4F%53j_Lxv8n"
    "Xd}WFLN4nk(mS5^44JZ&YA3>kAm3xUcc|fc48kJD*+GN}Vku8kg;vpcYKg=kAfgd=rqdk+2ZM`xxJ=|nQ4uceTr)xN!*Y1mIkfKzr6PjV11$u>2l@0DqCX>s"
    "itw+S9wGX9U^x>_$$h(eA`wyf_EDmr8}fYQW`|orB(gLQ9Vhy^U^y8QjRqb)2H4ugKe`Bl9g96=5F0H;)5D}IEm>9#s296T4Z+WWMT<^b<J@yVGrsIsg!Uzb"
    "{C-_$#@=@{zPw%ZfqS<(R)$v%5wa9)m6YgMN$)H$jCFiq)zE}Clo&M=8m!olsr`F?G|hs#v9dwW{r3>>nCM=DpkU)48^0C)FL?d0wXKP<HxS=FL45?l0(oeJ"
    "Q>ES&L(C>7c!2OEP%rSP^L@!IDDOTw-1&V9otGzp`iV4p^`_5)=IUGhs~U1p1MdC91V0nzBowY*N8<_C*jlX{;!&TiQa_0x0wwvCj?ylT{!7RSwvl2YX!dWp"
    "(89|~+dY&q0o$@#B5oW#Y*Z|$dTAEKxE_l9Er(kU3FSo4BpSO+m#Oapqdt79NrMEKL^s9;)b>IikM93&(MVtuf6KRQ8fz8nzk&oElx!xjG02~a#Cz2DNMh_l"
    "id{s*5c*=2!==_^7VJEvX5g69h{nb~m?At!(4F`8?slVb<KWM{r{s+>^XBYo0_#IpxW0W#k(vdM0@Es(q9;)O?faXExFNJswrCHB_$=^Yy1$OA%Ytr~y3rl&"
    ">Z7caGr+sFa^~x;Su~K>ZGa#K(E9F#+veRfVA;-1i6)z@v19i&nh2~N)vepOY&d5InE7b!P<h6R&C9<Z^$_JAL{n$L@DkhL;stf+qPT-Sgy#TyA~oLU6^(!W"
    "`N&V)Ef00RIzCAN2A`Kt3ccU^|F3yfv5&L64iC1XO0JzCx=^LCZB~{kGjx4Z!E~)`#t6`giig`PmHEzq0=Y|r2a{J|$Cotq6G2U=LD<bD3w36IT_lwSRkg>g"
    "*S2;NL<M@`c+%U`)6=kg*Ws*gg7}4wYDLGc>7E99VtnlD98Iu!y{bbc*34{@dpZr$Zwf~}2(icJ$3+R+w7t^H&VL%@ukq{3Q|5#9nL~-xdt0V~{}sx61w&f7"
    ">l9~+pf>ai2(7wA<1149KJ0vFjFoc+OcFuWXr{Myzkt*<_<VauQhFJIlI1)TUH_y@Y#ONF)Rbk8u0+)iZyqJUFVv{=(3*A{*B6*;D%fL=WnOozCW6XPUJTW1"
    "9Gar*6%%IJB|AZYX4G(j^#-%fDZpZS`)Y+O0_$(Ezt_MmaSGI(Uheogh8cT&`Ar*vHKF{weNWdsm;$$tMLYktKl9_g89jmBVXvU^gF|`>M|q8~d0vc~ez_T;"
    "wQmaS4rhuIyRU)GkFP3JDeX@9>8(@Xkr!|GjJzxc0w+2Mq7vQ38rYDgHU(^*V%L0sp@khW^64jtpJ+N)Gh1ZGB)IUwaOOswF{b~VhxYwvRNSa)=jB(EU}x#)"
    "bt`ZqY+j##pjr3Vs~^8N2{h6xx!EW}P@de!US&3)o&?rwMUqBeb)q3-Uk8YwMzsFe;>k=JSJ`PleLq+P>O;g&Yw@nllRzf$=MK+@x|p1<?Jt7(ih6Io`lxcr"
    "Bv8}l-D!7IhhF~zRB~I~c>LG|m>a$(yZY8LY<^z&jtZ6<oXRYj0JQ~c#5H;?>Gj0XyGT{GRq6y#U2U{ksEr-g^QVXY@oxze&<@O(Q;%weT^7S={PZ%xJS&=A"
    "`bYjDqo~u0W`8p9yip?PKaUT6Us@(%VosM^2D*9c3hm876JX5!qM7bx3v7P;#-Zokl0=a$6Tnk*729x#B}V#JF2*4h)*2IFfeU*_>2U$r?}yI6tXCAD0FCmK"
    "JFqv(^m#B3HJCMD^kRG*a0<2F+NiAuf7kN@hkg+{4wgg<+|9Bg=;Ny#t=s;NQ!slRbT2rQ7`tyJeO!M+C6B2my-OMgvi3&;iZ}(~@5-#V>v?cx9F9{`AHp_q"
    "b|0nj`FS(vFWm8S+P>fN`H$j?T8BgHtS#yCj6f4EX}M+jj)RG++M(B_i!kx+9<2oN4R!x$cgoyjoUWfs=<UiulOIIfx}q=+*dxV<?-eN1$I)9f!8b2sY-9|4"
    "^K(w=dTmNCKM2(n6!^Tld<+yAzU^2!%L2#O2S-1t$ZunSc_j7dxz*~hoXt;yn3?X6f%iH$caFYNfxKb=^sPOCV?cbr+gR0AO?vssXjWyszL>=rUCxU1`oBky"
    "ojPD!CN~DoO$>yoJR3n-M`z}Upi(rx-TdgN>=@{{EN6Gd&K&C1#zI}S`l(S^PqH8P7Q}cQ9vlVwI_X1BAN1(<?KK*l8W+?5d=ym07!A5<F=6?J%W8?B60~_B"
    "s}DRK1qP`c4}Z^V^LkQ-nr@O{GNEyc3}vO?^V&ah)~@SD8`-R(K5yCE)w?}x6ue|wkap?t5~#P#dED&2UZda=-wN#9Y8iTca?$SkisQ?)M(O=Nh?4c;vs(5I"
    "(;NkDkHGaLl+j)h3%=pOMWaAkJn7k!r;F+OpO1!w8$NUz8KL{ni?J=A5_<^ZBkCUb^MFn92uR+k?q}|A0QE<g<<yA`>ImpG^piMc?nLSd5?jdocoF*u5K?_7"
    "v;6ros6SID14}pO4g;fl%>H1=dQ6!^w~ZjO5FS+9Vd=VI5P#!iK!lnYw0l9DkSjY(hQXf4tgl3-G~qa1WjS5&fq57_kJ-n1%T0wW2Mt(!Hc_H@2xP1n@@Zi<"
    "p?@EP&JhyZBW@1?wZlH!s=UnUa!Wu~EnaakXzLJI+~z0jm}3gLNm}q5IjteOUa`UX=xbDBe*N+xAbjjq)1nQkbU75E%_cQB0u+ZpcG02d$Qwl{PeiQX<D%9<"
    "u=SD8_a{#^p+2WMe_n0%bP(iu54`!T&XB`I0S8y4>mX1*eSB#5cO_Vk!|TsUHv|X4x^?eY64gfZ-^I{6LM}O<We^;}I~zmo8TM^S%Fj0svj@QO#45AVSCdfR"
    "jJ7Bot<D+%KNc@3X&_d>c|^xXYyGw}1Hh5GtaVwSJ-xqDkXthm-$R`SKzzJIq@#f?tY-sv+8#`709fzKzBMVg5SyO|f1q1t^Vm+$_Jgy$k1wQ`$-;SU_4}4<"
    "l+1qmye|vq!Nt-@mq}PZ$Y@>QLRD9z%lQL3Su}Qd!*m}wIb6ir>M;%Le?^k(LGMH#-T$wK_RYdt&byn&A4~4s)41Orkp14v@z}8s?FY@9q+ee@BLIYNmC{C6"
    "F`8YdGIhSZpYE^f)9raWdg9W0;q}4&;Gw;(NvX6s9FLzItXHY=)5<Z9Hk2t~=tDtA;(^c1{m}oPmw!2mOW1vXOXH-Se=>NYk_f8!2Oj^I?*%^FaeZ_)BlQq{"
    "y!OG1oLe-L;;-hYOIZ!&zntQrv{<u`?!SrCc_Esa++%*AxflG@>j>s#Q>Xj4Ur=;`<qUIKFDRa5Ii9}78jjzkEBC&?*4hJZ)!lkyaFw6#-~L2}h4t+Amh}Lh"
    "T2;P$<vx`31I2ZoNN=R^wdPY6XJb_XX>T)j-xbkT?*(gyR%V+bqX=n#?rU(J4A<-hZWnw_#4eiy@^{~zx$l~BvKM^Y6!?C?ofVMtf?Sf>amk2Ykkk0S)@OS&"
    "GH-uM(Kn|Gb+u{ScvFTy`|3Q%IVU#$(4=vmtw~)^sZam6xwx98^Yr3H(0*#}b*Or7(+gr$ul~}&?V+5v`X+|0{@DXww%26mE)|0EoL(4SKqU8ob$hxsf6Ir|"
    "hl4)Uv)^bu%(C6?x1A^DK=S#1Px%Yq_^`OgXrmd_H`9-{M=Gs*!0_SQjk!y;pxsD);UKnVIawchXb0Fo_9c2}bc2}dQk~~0iqKBRHI~hkt?dR8TV5(SjM~6)"
    "t(7z7es-)21X_Uid}@}EJFmE{Ht@I$RKI`Y_}iZ${bZJu_d9<a>jHAq3ZnHr-LU=vi^bR7pz-;6=O>(JNuIOIzb>=W4wf(R$t+KSy9?~iD)2xE8E79y7Drt^"
    "@~ab+eLf!cQerXe7gVAt#OHn|*xa+l`_g+gIR9>KJME(@(g{qKz2V!cZUOD~jZGmHi(5NDdwiKaUw|~Uhu-c<_w_?M08gAX`{6Uz@O$UL1)8iT9YD0nfy#44"
    "47OLc=z|B}P&<8oA3(`<&Bv!bw-N{2LFu;p%1=%*=2Mjz-bJrIw1bGoy)r)Mg`u9<&K?^-aiSe`geN*%DKqTqqT|OqWO&;FpR&Tw^fG?vcU?CLcs$$H26*$2"
    "@)*sU!*S~D(idl$)dtv;u!aW)@~~Y$W*40j-`@tjO4j86UI&qK(@-}~)mYdDc3}$BrhY@Pzp`y_aIF<;1L7qao)5Wgp&WMWRIZ*3`32TUkNQ=f<bn45qbX;c"
    "w$Crn?2@yhO~in%SBdDs7i<A#4Xt3=1;NaMRK|E|4@!3a?$ZjmPlnj8UMC3Kg?B_X*4ed!5nU#o#kDJ-z0C}=?Qy7V0cI0Nk1eob%*$cJOE<stX#sO8dnw{>"
    "?9gv4|Iod_dP@tmx1?RFY4)hs-9moP@UKWtj=TRTo>}DL#6}_4n<*%FJJ^`>y_vp#pF_#|hdKAw@f0?L3t?QhqGJW1y>;YUqt8^(4DQ`{EaQ4Y0gmh2$Jk0O"
    "otlB-Y4_V2k|QYj{egRX?lrA%ru%PZ41MnZ{&-(|6Wu>ChW?MYS6#}J_e}sl%ON8dEDz`Xu&)Kaxpqxp!L?*l;d|O}-U;OSoN_+95$qh?Z=$n87y9$<RxjO{"
    "yc$6{r$S>d!mu0WC3%~^r!;^n)d`zayVcMSj4zWyPc+s8dHd+^oY^{X{Q7WM?_yh452U8q(1(V~u)WLF7PcHxr~?Vj?+i~@8^HQp*CdLf7t{g;t(_I7ek^eQ"
    "4VG%CoSUlwi|o2h3939?mvMjE^byytq3g*YO4@&+ko+1+%^GkBky-h<k>QuFJvPD0qg4%-&Lr8t+%5tA|4(^aHvU>t4RUv#zTlE81?`o8%7@n-K2@M^iPWi?"
    "K!!bu+=xyrZmtB$Hen&Z?Hg%NM6&q5Hi=e(Vg7T)PuWc2`bDE&Uo-VX1!x?&8=xV{3ibSZf6<n^5fvcyFi@#KVgc<ymV(!^EmzAyn?mUA2sI8^PdACpr*|GG"
    "2XV<QIq%~2p&rJj#H^p(R0gu}19!oN)v&+PE`|A3d@2D!XJ2@)EV5wC3r8IvvVAP3`&SE~erAVXND6vV4BY*A(s_=w!*%hCo^Re>G)}H-NPmXp<T~_!msjbB"
    "Y&+tDz<#&7-_*T_r5LPf{>WW+oC(Tj3ujY>Sl3UGlCnfoQ)(rgA9<Yb2wzC~3HSd=JFMr>k$PU^C%v9+&>mu#-G*=KKS6AAsfbL02=upi8>-oGJ^2A_e>OV3"
    "l2n4_TbI2U6Zla?Ul)v`q(9lw;w>mvP(;@^#yp5c<u2x<anj%WN1m;7;q8VBhW-R_s4|5f{0?5VckT9KVvMilg=Vh7df$M7m~?8-asa<yAKtfM-LpavLroaA"
    "ZQ_Oc785%A=GlTmx;<pH_ehg`ViQ*(w3B3gNIuSqF|}_s2c#deTjJLdE8VX^Vfn(Fmu^r1xh`|<GP{5A%vZYJ>cDmP({_`ab8=tl`%z;kxjy>(`fgfPNg<RE"
    "Ieup}j^+l_xSKqORY;#UoL?L(KPVln`$G4l80Ac@sD2fd^W`t)xb&<Z@}qIGUH{1EzmNaJ|L}h)&Uaop%ZAUI^!Jd}PwdVVzWNL_B>EP~F42K@VBoRs(+741"
    "V4HDIKtGUx_OsmJ`L1h39ys^bTBGZnCbTDBd`}nXEcpbkF9`hRrezA}>yd-(o%_mjfxydMzAv`1GS)}0h=Z>Za_RddjVQU^6gU?C>i~_D_McqWlbp1Bq#sD~"
    "fQt%Gop&LyKKxw`53{^J0M#$2w~ja)!FHW3xF}YO=K#w6?VnxC&7d5TJa+qJPom&(g84{LjSQSmeUPF<ox<55P2xl2Z~IKv(|cCRX_h}5u2V>RLh^a}r~K3K"
    "dAlbH{bUMNI>x8#Eqz|@LP<McXQ9@*yp`l%p!_F-Y{V64eBNI&hjO-Naq8dWn+d+0OMb5MOc?H)g~vAWRvEtm*sW~Y7tht;`qBBD-im}D=^*;q>!{!B0@81k"
    "wr8GlkV=F5+oYVAWXC!7ElPv+A^Ux5@65JeG+v<3sn2O;0LcE`n6^YHA^#09%;EWQe`ppV=ih_7w^i71W`H#|WlfvQ>JYMB9?A(bH)&k!UecWJrF?`epX4Lq"
    "t$eH(U;g2L`2Ug@udmOldanxooU|`TY-5cx!Qm2?;a?X95OQ4*vFnnv&F)OF*Y2ETxONFbt}nvYcn$^9cxLC}B^FH|5pq3FavSxh4_A~tg!=r4|688=y--^B"
    "G6mWn$vXLuj>Q>tyH}2q^G?M;!|fa2GwAl@9ZJey|J~c6EB}SuU{B`ZcY)Cec|P**%{iR3yY1!8;k^7dWS>UDmN(#ZHMY|wmj|xzb}Gv&l@6wXJKO6WH6&Z1"
    "+@illdr$SJfnNKE(u0ZG0NVe{CK`g(Z$O0{``z_w;|RI#eyrkY?<$l5<h)MZ!y`+fp1)AcdKgRNmp0m&)EVbPJtujf{Pr}>lox;aAO45`;s3JykkVM;iBvv7"
    "+QZuO4NuIMXMkP<<^q&k2inh5Z&Rk1*kyxNPwZ~GMcUK(Ph^jf{LCp8473Pk4a-k0hUZ*j`|eqvG0XvhhAZ<Qe6j}g{V1gH{89&YyANRUT8yi-IKvO|3w!P{"
    "`uHQTeZQx^#l-@U_P>NvOvq!`Cop9DIjL`48j$-WT~kf0JKOTW+ASk2_Oa?@J<&~G6Wh|H^FhPI(TEp|oZ#<1uTD(*wYw0=dWfG(9d@MGvj};+Xs5t>z8`>p"
    "usmaGDWhFiu#>%E%|F0#<2_^dr=8&U!$)PJ-l~=Wqmypu-d%Er-&YFBX|$Xz1A6VLpL9peVEq-GC(g$0EC+7sh0lA-wtzs*N5e!iE3nHkSipR^61*=fU2FDc"
    "1E_nmaGWS&11GQM8r^$V1HLRgSaQV76mV?H86dzE;=W}=ZN$rZ@VUdQ{cM8}Sw4DsuZii=^m-63)AIGi4^J>3&&CD<=0a0HMmK_=ITxbbTWrAY8Mg)k>qNZT"
    "B6!EA8-cLcBj*lFGcbQX+m3Aba%h9fNE6sIz-lLWWdnG>w4jIZX93gb(&%k2EnqJ7%lI~59azpN|Mj}YgRMZc>$~2#xgWSvc50MJ76KxV!z~=U+d$!v;L$oA"
    "A8@;6x{VMM0ncuXP~?|%0__Ik`B3j#5PmtKlaQT72GqDR)wgtm)244PCuTZ=ydC$(iJ%!|Mn~hJXnzm*^}ZqI+<}!~J}$Ha*>j`BnLVlxY`!YTG~B-fh)d?w"
    "5pnE5k^g(fxuJfbZ*}O$>oHFt{X%bmh@gOVhntRV#s+}YjYDb^0q%geRiK+_U<Pt3LnVo)2f?#zi)V^!Z9yYg+CzYT<hbgA1Kq+yV9uiFZOyW^^zR1|QHgPR"
    "L2L-@zjvKq?u;3r?*k*vA#S3l9EL&iR(TD@m8*f^xAWbE=OD6?X?nKJaRkipJZSu`sRze}aI3thRP6}Rba&6S%d?=zd-S1V#}1q-E*PPo_eo=9{AX>kaaP#I"
    "5x{}Tqi@!`U@x~kXeY2n<hXU!oMh`Th;z^gP2cB<k$N)s;utYwH4ILr2EQKk^Te7ruc;v7W|4;JUvge|hQNx|8`JD>ZK9VmfD9eK$lY%^1QvF1%_;oYfsy)o"
    "Ko?yd>pBR~64MP`J2zqTdNqnDbiM3+_I3c=nleo=o)gFB<v)fJc1BSKaRZ>n6Bn*}N}l&MAy?Y(ZnU@R2bI3ow;viBz<umIm8}|0s(m167uyLvPdoa4bu}V%"
    "LA#h`e-F?vifc8Dw}jmKf}PE!V_o!p6;1j+NejCCRp!rcdpp6sH!rLG%3U$CANSi<99y!k9msvoS6Z`P35yUNZXvK%)Y_w6z>dES9B_YgV#QH&Z2kFbwFEYU"
    ">TKuH9NP5@<VSpMcNw%N<%Z^!ah~5b*b3~@Y!=#{TZ^S|QyPheS=4K^#5wd!3&`Tx_?o(TH9byv8r4m_e$Ugq1+>4TSd`~#!0#WmMeFGdGy|W<3pb~hP_VUo"
    "_6-uAbLgC`9is8L8La*}_x$(1!Mr}qq1SIqGtE3|26IQ&OYB`{g`IX89wR&_&_k0wB>_8{K~Q(VqKDU3U}Su?zyTAr%|gw980QrU&k}^(!^<<#GPwy%;qMc}"
    "y$rE1OAeYGM$xXK&76%^O`!C=o1u286+Mn?7%lWRy6!mJ2+R!2kKE#u!{+sT3|;gz=grgGjo_XpRls<o8U1`@5KS>W`lENa0X*2);Ab3WO6nDw$u|ASKB*qC"
    "G|V2ElvBVS`d^(UJcm(mQITzZK^@o{KUBFdOoD!%HH2QhsKRU9TMJG;{HpRhPPj)ec9f7EK&v)~xlOFE1@f<+)XLXcVdLqYeFW%1?S&=l6dh~8T`tEhm8_O9"
    "p857J<%(?9Y7nBEsc#W$jnVA}`b;OgLwL3l<OM`n4t{3DX=`%D+UKpS1P=ml1m!naV)Jq4ZD`&e?nm!+Dgdj;I;CTK7Gd*on|-L7z`F+Z6Xo<cy-u`sN6H8x"
    "JBT(UE)-HYUJfLVuN3Q|^&=S{PI85N4ZKd}%Sii)?oO`L4^*xIq3FsS-<dvyj7#^{$j)+6uK*)MJydglag6+)Q!wkun&;&(ez?D7wwD0C=+!PCAB(5u^!>CP"
    "$Sp-fHOy#y-tMe5A?5a;#m85z?^nu?LPY0o4iTW^KaF2=T{%>1K}E=T?|)XV@ny}P^KFsHyq$0SZ_lYAmfJ?wL?ORkI7|~kmH+PCd0kD^?>H`UzqB~5b^6iz"
    "au|<Fp7*)-YAXk<D+fKUd(eKc0LFV*y6>heUS9qW`!FRa&HQ-3QYnY~A!Ip%ul<(2r17z8(D>;CC+Q!e7ebD{Yj-FIAEFbCw`0W!d9E?5F|*uZZ8^yNj=X)~"
    "sszY$RnN5Y)2n`zgR{;)Ux@wzgq$DO52{m3%gVtbVA0o?QUc34GM%`)hQ_^%B`$7Wn)es3etNa*m6E6bVf?_@{e7aY0`R<etl(jWRBAarPNNj==PwZD{L-FW"
    "PCs`@K}k95D?PCK^)JM!t0F2FhbY5%669I+?rY1+VZ1RJ7ZJM0F6jCahW?P}0%{z*3nOTpjGrO(^AG>S|0DUq?5*JWS^(qI44&Jr%q=PdZylbnp0yW;@kmFk"
    ")KZ78mw^OfF=f60HAXyDgMYDCP$@`9!ei}y?8$g<q;sNv<9I;{m@+@{;PId^Jf}NoLD|;5r36S=g^YXJI*{=vi1C5;+^f#T^z%3&7%w4|P}*n0Urdh^>x1Wd"
    "C(+nvZG6RGy8;@dGz8;$kdKq~k*B<Wg1Wcv>A&;u$bH(E8?S2qaQzA6qRDX=Z^<5j(l{9xLCS~Z<aqy&;yQ_5TIH*);XZavti;i<T|dB%@MgpM9&UJ^$g(Fv"
    "Tf*oE-4EA;?TuT}y7)u<cfiJBqU>>maZY{U)SWvSd%uDX`wy#s=K+!D!iw?n`yYcZ;9N%3#oJ9*kaPJZ#2!Ce0G91<!I^g2!hSi}=_wK7kO$`GD%Fe!*1`U5"
    "(YnaaQ<zKl|5w1cl$rf~HryULfGNN$kDb*9>XqnZQY2W7gDpdMUcB}&gY|Dpc^%+;J{wTCyf5nfz6|c$ADKvSwN}jpNm*Js_=UBwUFuKR`w5rVV3$^Xn`f~l"
    "EWg$5r7+uvSHO3OOEx}u75u(xQmTYo_Zb-Bmb|dwy&Y^<<>6m9$C&Q}y{l&J&nm4L=Lp{A)AqM6f+1{4o}_>wj4QhsWjJ+7WFI!4hbRf-=;AKj&&fT12}_!Z"
    "MAtg1!FVdO6+c#He!q$}usI&fveJP4=sg(!=DtBZ_V7`e=5ifI{Cr-x;AY3nL~P%aiWm3RGx99-_6D<sg(qQe(z8xK9%O~_Y$2apqx$V$VnVKQpUu9kgYtZT"
    "ICQU(Od58_O5c0=b353s#%RAWjrW<@)plKd*(@h0H;$)rZVv}=tR_@Tg1W|;p@+!|#KHJa82i4h%?tN4xLxMm-TvNRFqiA8=P1V+yui@Ey;!>l%e;G6`(z-4"
    "vu*q~`EKMV#@)VODkI1l>YJ{U`vX76(!cP>C#*JFCzWIK`D+X=eMDd{2S*h)&RHp_ahAdD(OQv#JvD#fl7R!7*q1uY-s#2E-t!E8Lieoi0>Q?=aNfKZnn%o<"
    "uzRt=Esw()e5-(2SJQ!J$VvS<_6=B_zSV*yZMyNK`67c~sI57=uekLu{GI2b!Scahn3a}Fy37>@_mw@jwn3#Gn_uVJ!G5WIlawE<(t&mPhjU4$G5WE%D%#g7"
    "qyw9e&$osBzW@ER?CP*i?CwM3b!f|aSiWFiqOGAp7nV6Fl;%rejNk0QE9p09yRf;UI=&oJhW=}ZD#ctp(Tz<U&TqZ-nh|$r74>%ex4k{sMR|(8sOviTeUpx`"
    "+82dhEYXlPV=}`L{%%5LUys>bFSd(Msl4>~I><e$gi7Wf_hIwnO9ke$6kU7aZ>iFcSse;XO>%LBa^QCxdsRHukJ)RCb~dP)LOYXi`1*0gZvfLce1b1vr6r8N"
    ")#ni3HyJ*F<-}U%oK4h&=cjI7Wtq91G){F6FXCk6NnLv6&SAJ{5c8Hh$+`M1BhTonZ(pDPkwI*&uggjA1KM!>e)Z`TF!?cv?T8o9e7tfElz)H==YzcqhcM$`"
    "^^!L%5EyTx=r&l~t2Km?dDG;+diB=Kz!S%Zu+JfSZ+gSjVEO4~Rq6)mLm116j$I7|v;*jDm3#eiwqb1P*4&K+XN_UIHn=V1%@!NRm}-uc$|<gd^Q^Jj@aHAl"
    "hp`nO6a&&Ot$=a7wze^o`6<KLvM>JWZYK@l{C=nQ<|*TeVT@&}QBZIp<9FA_3(2;bj9`Tn7LO-GE1~?ei4}2v`$n*OI|;1Q&72WOeZXiu{ox2E9K&<ScORpi"
    "H1xLm!jcipw>_Bccf2n2laoVaq~=Dj@B89-?(nn2eCm{Q2c^$)jAHB0N68f46odPLHPaH!1}dZ2iOp*_$L4Fn_$xWemHyHFqnM|-7~gNZO4d`>mFQDV9L47I"
    "j@d9WE;sApPNyG9qZk<{Pv!+H7{{)upm8!k<sUhjpZl-M=i|^sVVw3u9;<H}C8OA;p0)4#T8!X)$GXjB=_;-<Y(9^Mk(avbq#ZSqcMN+-Y+cI84-G;yeJhD6"
    "%`uGJw<Om`THSsgo<?I>ys^O<12Kj@DQBnd;`A89$aNy=7wH>!0PcV>jEtiu*VpFM<9^CCPR7xZ@#-We{VQ_aP4Y&rLf`S-dQcxMLbL0SJRHO3?}eC#c?XY0"
    "-M<$nlI54fc;GnybpLW1e<b1TrzgtDKjJKk-{V^|hLLd^WZZ9?LbNxfnXFF}j8}}dxo~blGo$`w`6MUf-2U<V|2IA#|IZ26*H*#VX$BKxm~E1Zg<>uvPbAQ-"
    "?S`JfIEGLoe0P0Y%*a#L)?vS=JC4oAr-?v5(-}}z>N<`s(tK~LZ@B`F-$CV~%uh$h>GfyC8$UD-oSZm5j*;tnGT!^<nxr#RG*0%*Kl0>WM0VO?3dSAH%-*@C"
    ";y;eH$1Sxws!#&=H!8mFJ;qPt=|7ihY8_Srq`eBquif@rGmbgsXNDQ&3?QWZBa!!$3mnEV&SP&5Pn{`(<@l5q)Ns*wr73>~tMq$>jNAWb<;z#AzB<CQ3eHEW"
    "5qZoH0>`mS%PXE=$mW1~$3Lm&w`Ib|v6{`5GPx4w@ErDg!o)&!|2QUHmRI?Ji5rmfEY&BV=JCdHEYWSlLxBe^Q2uCo$;L}GUeh(>Vtu3#A^Vr)+dTX3t666J"
    ";eYtQ6ldmA0~J~FFn*v7u^$M~AIHwGw6E&Y8$if$aWPr&**U#&EMOT2_sab5F#h4#2{FmvT--`aawInUza3Y#`QZKg{E@U9Pjl}XE_NHou%#?ISF9QPIk$-M"
    "B?jlmG5fOJqv2wvaDB9Ao26S)7CFB#;^$|)zw`L>PGA#bp$!?96zHc7Fpt-K;hVtL`|ePp*fZAY)H9iHD@-P^m6?`ZQ_slt#ZTl3+hnQF<q2%$yKYE^s57h&"
    "G5n>=Uw)Eq50&6NC4Nr2s%zsUW-OUTY-L^v>*H#6-)j8qBz?ZNfWO->wCu<J`;*vwUcfTwCs?0M*PH5`#EuqC-H~-<<bUaIYTUMlWeO8MrBdN?!U@V@uIG$^"
    "ck~n%y!4H-<$f9H&#OM-ycPOt3ghGamF^wq2)`H7PCL0>ej0NRTlu2&gCU%+^KJI^C4^67>(Z~7SlKYvQ|{Ae?Zi8$u^N@fiYpBm>qnMZEA6_V8LZtHH~F9Q"
    "0$zR!$Y0Vig9YZC`7m(D2DaB<K;ZjR?pZ9F^X)00G{$`t`_F0VO5UHvW{SRVTe6E${vCBb$6@_B?AEE-*i8`(-n2V@UtjSYmLMJ}ZFlp3`aW&JzTrItmRk0x"
    "`P^*=?>{b;agd1#Pu;(qHOzY#78S=<MQHE=wvEzbvnQDFtiq{umTL$`#ziPz!89bEG2s(;$Gg1Lw`2419z0-9R@%?CnF*hdLvW`1nM^=D@@mHG!_2q<=g8LI"
    "`x0c_#jN=5=!fr_@ym}3t_+Ji!*z#ZRYU(BP8Ph~>*$)Gq_uDz`uJ^Qme*Pq+?i?^neXZV*NKIOk7QbXSn$Gc?Y>+gP;RL3MRuY;jRmj!7?+zp0QXhU7`DN?"
    "xdaPN85Wt`T)z(f?sn!c&yK26a9-bTvDP+oxL+jd@O-83J_@cbdSpopvjbemRiuO7!E+S+x@wSNbejrXzvw-=x`jW9f;SI5m;X9K=ARa!`;Tt*xKv2Nck4XR"
    "U%if*exFza8o%J>OY|!R-)C}-b#jj?yoc(h$D%oz77Bi0-_qHW&n@6MP1f0wV#CUclko%Oy3DiTR41{36`#-hmV@=#-jFdZtjLP{RQlVQ`zvFlzvq^<c9>V6"
    "73Y|IJT}6ug^_;y6|iXWJ0n&+?}wzJO`{S<`t2L9pXVYru;P6fyI=EWMOe<Sz{Gg109HHzF^*!nrVjmWmU^{lwkxc-ho9uur)v!1zFg{kgOSMxtoVFffd$-8"
    "&YRh2yfuRrm+<0|lMk_h<NdtO?zpYrSaBX@VZ}u=CQuF=0uS3%H?rbAhXkGvCohEU{e-*EDGjpX^XG=WD0yF5SbUJa35`EE$m`8oW)AB!rP6lNbCwmK&!bd?"
    "_u)KH-fEdH%7*WA{V<JZ$iVh$dQmF(sj=bn_frkf_p58rib9*31R9^uD=ek&C)Yy$Z}*x1o=fk4axU!vJ0Sbz(i4}tXYy>g)uIo=N@<M<d7qdz*2*43;}>K&"
    "eIo_GBJ=z3Rp?H!r+049xL9(_HIvnufB3%>@9kczYbLM?`tv6>4d$d5u;IrYjO8O8ImmcLG<xoV`yeYDK8Cb7Yf3GHddqG2p-zgE4JYr1A-~_ytvGRxpA9e6"
    "-C}QWRRhkOY<D-#waSuuHG-1kOX~9H?JSGg@M{MP6U6Um!FywR65JA2aI@iyx<AW(4xNDOi0O!Jp3Pir`0KMAb_SpvA@{4PH<P8CXxwD{?BRnA8UOUWPmJ;;"
    "Jd+2I`x39VFW#a;<8{xYas%QNU>xbG>w-v94=XOG$hYvt<q?EDS32YMO>{kt3vsDxJ+Cf<>(k};gm<i?@nc7a1pOnI!Z=m^g}L&|Wvuu|3#mY(&%>~sYpLcw"
    "_e)vv4vw(RZdfTSCnV6OHjc*GADc?tTa^#}UXy_fnZNmG);PzrL=xoW_kZ~R9p}lN6D6vb!+E)B+w^oz1uIV8XGrS98n#^-zq(m*;S=sY)|-t1+|M_3(OuoZ"
    "iof+5b6Is^2Ify)IVE&YxP}$Kyjw`D!?_kA_vMzWvSsnp_=$><(|fh^5OVyI{PxRCoi7)k{^5W4AO45`)AP4Ek#*v3TF{?ysrIkQq50v@XD^Ce)LRepBR+kq"
    "*1a0Sia+}Di>tBAmcHLxg1Gl>N%!8xig#a()qZu$6p-s^y}BKnmO8NF)>n!H13t+B^1dmiYUV%_JyyKb)TTK}Zxrr(NGH2biqQCqT(L=q4OOt7Brjp|;|vgd"
    "{}-M-i2A+2`2e}!9-4oq%TbpVcPuDmcDpDE$n(A*PXi~7WLWVqHh)BNRX@z@+<ar^A}ftQpFB7#va$@8f3**n{mm&elZKX)`G4W6!&xiNomB<oI<#B(>XN4t"
    "thm$t_-z|b%0mC)<(e0AK}@W;d$bIX_H6*?oxq3;Pu?C1J{idtnEROpkpAJLOX#_-4hoJP5vjSxVF>fh6Bk{7ezBT@o35})m%2Fv^BjGipuZX_D7Xl-W)gp7"
    "J@ik`o!wGWOXH#k+&^wO{RQTsliYi`$M&U}Y0y9XujV@)gxiE(Izat7f8DCYH;aPjA|kjUi#+T{@oW1n^6yjd{OHW9m)=jnd0EJFeaB52cX%3p;@BtBKhHt<"
    "A8oJq4x`{7e3r{a-{ge*?Pgxb+ZLRm;I=ok$GxkT0rI?TMZ|`fTRSND#)8*|(o$2f{0~;q%<MEiIUFs*d=G)+MaoY~x73`1`xqn`?O(tK^-zuX+R@Xd6x_b-"
    "M3CliD{Sv38TFBqG%lzRv44x)H|RH#e3;LA5&ksupN2~%%OpB_8^C?Ga`gImw+n^7zGlR6nY`*cIOayd$@q41ya>jv8=IhU^1O`H!~gEQ!S0)@%JRj~?(Mzf"
    "e6$cyaDnX;Vtflsp&e+H{qU1Xmx3=|X}oF4M^>1}$J>67IbNNDcQ4&76Sk)r>S2_G@6lT{{`Km1M3rFppCmug3*6;Opa0>1_#gg<|KWf5AO45`;eYra{)hkJ"
    "fA}B%{{w#S`;}fK7{Gn7@Kn9Sl?oKR@tvi}*c=Ot*R9H48|W=f!K;}(&j|?D!}G891$S0@(s*-7h+}a^K0@wKl3ZJpN$BL}<UjoXko!In<&YZ`h4;xEj1g&k"
    "$4$ZG_odur%jiMK__RY@cU!_~eAT&{)+N`q;r<)j&E{baNeWKhQ%%OtOujtM@<ED%&*zi=M9Da0i_mSzN61s)e%-$sXRoBhSWBwH{cts%UuwtLDfsSktNTj="
    "$DrKoEAOc!u~G2a4Xr;!)vDn>`G`nczzZ5b)MwAhUWmhUhyN@-pBFZXlKXq5)i<+q7m)jhKjHm{T<;=lf6(~9z5g}`8K34k^_iwW<o(8EJX`mDit7z|3O@ht"
    "TLYAv`gbesyEMMkTh2)6oD!5<f>6ULRv`)=;?Pkn;Wh~Mb9otsa1f;6trOK+bwWSj`G|*FLG*7fv?+M(-8TFW|3BpO?@b89c(#Ibk=l!B`pj!QX7f7_g_Qq`"
    "ioR*5-~4~Pp90_GKt#NX)>Z-vKL0Kifs*z)m%8>yjVZ1CfAgMHKB{(sC8L(c$$XZ7<m7%exu5*c%G3BVwj!yHfPUW*q8OaiylOcGcch3_rtQ^%_jLK+no03$"
    "Wx-FdEzx(6-Hd(f>m4LyS%Lkon>>*VDfoPTx(tjvb!;_=Gh0TpGy8dzB}>=C`?YGtR+@?IrQo)ishc~#3B!CVqwr^2C;Tb+;NybMJqry0d4E^AG}TSlhSo1`"
    "LML@UYytP)efdRTEMQU1(dn%s6uiNrC;FVe2fQzi-LkxvshS19uAgpb`AQL<tAd({_f;7zxS1Vmd}**Dpx=vv>^{onSV3jMhh{HhF4@~afyc21LW2XWXfO`e"
    "Sj~b*G|UCXoZJGgOn#XlJlTP0f6#j|J{H_Ooy$vAR-GP)R*f9FA*{h&!Hlo&D|c_{Uj^gsRdQR@MZPlQ<>j}(JdxA`<b7W^uYa&9J;{uBOh0_5#?K05ew-a9"
    "Ks&1U(UDVy#)C|D$m)$c0{VS8h~+{36&rRk;}2MQ7rlDz3h#wGps;S_D;F~^@!B<B|L{ioxf=_xd9%ee_68IF<X)NEy#rprO>99wk<0}wqF1Z-2{YkyFG7#M"
    "J>dlsOC5U&SuU{6=+_e_8v?VMv%R`ur6<^3$yH16vxAkl<)7f*bJ$S(glX}w<*;3&j@>TLzRhAwPyQsvB|O0Vy-ah6NHxH|vd&_Ifp5}3H+j&@pF{?%+ME_e"
    "&0x>tBsxPQwgB%2jS<3g8j;5)-S4x{U}rKEDnf<j!F(KPFDfwAta_q$8uM8xBIOsV4aj^t4U?HiOzWpHU4g(i4?@)dd5>1iS8?{a<<r>djfbpKiZ=r#>}CTI"
    "H-dyjo$x~Ar!a*d=QscQz6y|Ws&ewGZaRmiuxZx`=L;?Z;6_J%D}fE8J6|MW`;jTE>{f3B9#D_`a7t|<8YWTc1C+08X}oA#a@P+XJwV=zwP;z<ng^qknC*{S"
    "E<@{kk@<T9$I(6fHFxcXC$ZSn{Fww5M|fVDt9iAYCwmfWI{me-jfoAA_kH<SPG{&OOk%xM|2qpcYU%UZ7`jR7d+bsgKlCdmEl=Hv-oHc0wH-s5OU5R!OX@Re"
    "aoacn`98AHVWHzgT@zT4+XI!HC(ZPEc@*8ab#+S*jj!78-Da{XA0h86BYEd+b=xP_mk4SnIzY%W_`*{X%irGD#F|bTjSz8z=m^n%i84HaosItb?e}}f<o!HH"
    "ZnM<S9h}6v795)X{Z2G_-<Lu7D`|nlli2+G(UmCK-kb+&JAd=}Jk1<bgL<}+h+}Z0v)XQwKAIS*58AP+6TcozV)Jpx+L+g}ef0#^g`QudqO_@j%xf4%Mc&MG"
    "6Ac4sYGRppSkojnA3rQWkALn(#n1M9e>6Tx&$BSa$oG+bJH5yJRj07hRg*@)-@Tf*GtFpy_iY!yz$t7#ZkdgK4_*%{Xe5}I6gGvC{s37%>dEmdjmB3Ouhz7t"
    "wTrAj$;rIb`E$&ke|P?fQ+Md+VJiI`vjcLExauDQr`BQQyHuj-0aCn`Qy7-oC3<*LmL7NAjDBaj^Kd=SG?XX#KJ{0Z^>KG;d`e~Y^ZEoWjEu)V`Z=)Xp8hm;"
    "yh9(HJ#G!(1KMrEl=9Vmnx5af0=w|Eu#^a@LHpE`be^S5V*%x#(n`)d!u(A)MSaKnDl^y^f5NhsOedJ%7rH6`l?Oh9&Amw$n;de2c_r%(hh}bfp2g<xwR47k"
    "n%?K_ujPAYG4h>FGJc!=iHzdd;yG;oJ-#`Ve5bn;f7@*<KZiY)`{nc9!V$&~N9?%ub@S~xtmoU^IJ-P2sOPrBNnBN$1UA{9nMV;IfA=~1Ez!PCwt&Ep3QEJK"
    "rDUE?9D1Sly|Awa6CUj!s!?`)1<WJ<7%rS5xRD8W6FaT?`@RCwZ)vyaI${*gglBccKVWB7hIs+4rJ8T+o-pB4p_OsE?;PRxhVS-kZ*6137dvig|DAt9{;syS"
    ">?=<#X58>i@GU+yXPCEJ{-STRJCGUwxA!5$p#qy~-dKEM#*-EqJa_saijnV8Dyx;+Y7{c#uV>3TR*S5I@zU-)a)oyHGUM~_R*S&AfQ+*ClPjm0@ra|f5+WA%"
    "P(BZL)NPj(V!^e>oVR{E?F92d_Lf@L@p`b}U6(b+WR^I?{2iV*iKyEX7Mv;K>ih5O?cn#91@@UbbhF@itggs!|AEZQFmPAcuF}tflka(vdEd96udq_5aq?Xc"
    "`g{#J`Tod1et&3L%tW}iJ|Ok2%;TngQ56e*_tTRYs}uIn&%dp(euGaA3!YG7Dbek34f(QOjX~;p+V|&0C3mIDl70v}z^~cIe})Bb_R=`1THy%4|Ni!S@G4Ul"
    "ypzdY$}!uL<Y@Ve;&qEzSa7aE)@4ha1<80Y^j>HCrWk@5@2hFr%fECTEJru^?w96TW_%TgMa|EACm28Tvo`6FQxr4K@hp06ue~#ji%`ni)4kl984qdavwZbs"
    "B_R7Nn|e=KU?DTkZibBJMe_hMUxOdNWy>MNj2{`gOMSaf4Zimz+3Z-B%g2m6IwxH$=w`I*toLbO-Y-me&cH*xg)ELRo~?hEpL$6X6Fvisb{|;d40)k8tDVqo"
    "CS25$UEBJ(738hm$))%KCj3~Cu#WhQ6TDZ}IUnt#s4?N$Mw*)!#XCbj9k%?*rY{5*|Kyyf;wB4tFR@o9VJ93wU>jNY<o0(u!Sai)yE=t;%wY?3XU0n09HHN6"
    "ZFTVM$IWw?@G;41ii|mon;Lk?%r<Q_hsCcAeY5N~<KAP{iRN>DX|q_a#)eM)AjZAN#DXA6Ch1u$^plMEHCt1-U*w~zSnu63gE=1L+Op^sqnv|YhU!eNGgzEM"
    "FZWI%P3U(W*|R-+lx+qZNpMcSsj?2<+kE;Yt6t56X)NK)6Hsh~!16m=p4$o^oW{m0Y_e|^vXk{k?OxB`T<klIZRF~ha)Iv&CZkBbaz*H!X)I>SNa({yGuV%L"
    "Y&A{x=V<dn&w91mcglc_18<87%3?O2#%{KD-Eb<NhH>NEE%hSWM$;J1&0W>{x*q1ChZJo6u0`XN0mDAMszR7|K=Q@e3zuN`GGIRV-*KbC+A0%1I9`yp4O&NK"
    "Wu~#@?yZMR*ck1d*ivxYm3JERTW-&ip56@WsUxEzX-(r)wkTU|gU<-*XOcWUwfvE%^y~k09v3sIqyBL*jK^E4Y|d)dFohAOFJiE$LAVc@K68Wra{Uyx%+BIS"
    "*TZrc|3>njJ*?^-%<o{l*}pfx?S9Nn;g%ZI&x77voY5&$*y)1iHBa7-!F77ym9<W_$y1o1t;?ycGgUBta{8p~muea}S#^4cjX^GqpCtK`*c01$a+bpQF6lIp"
    "n~le&u&#4!o~aj&!gwtarZ%mPqf=P$2G?C(_)l1luEHyWb{f~hUCf@l=Kh7RmJm1L{HzGeUp<p<mb!il6Uo5AQkPL!e~nL)AEVYyVYeD5o$6}KVL9W9YGPCx"
    "cRnj++GU>kFU#i!l#Vw38i4K1Z`>Q4uzCvPC@<QT$R-8jIhVF9=8m_W!qV%fxcAv`!}~c$^lXoNsZU|u^}f7pV@<G}-uRal`ZON0eedl;_gsX$zwYn&=ChA|"
    ")pKJ1X}E3Y-IUMq25>%o(QRc=!ZU>h5v|1?4vc&jzk|EvSvjY$ucGOv*H~0TIh^6B8=vE#$!G8R1*p^tXs-l=k5oGsPhwZTC`;abI0N?yV_nXuP=8Efxw{g*"
    "1#j0wJ^$h@_4q1{PbZ|NE6II@^(6TN>tLR!?2P~KJpS`*J1aBjC%n2O^X5+aBv$(C%xpyq8+_OMNx8FO$y3@oZqOrd_;w3y@7w{GM4ZOi8gA_L-tZOHpX48&"
    "&b3|^&V;}FhyUS!_#ghS#($nQ&_3I;h_sWa!m`pJj>Jjqg--C+Dt|H`HVge2$wY0rJBi8otC(zk^$p&;BL4MUW;u;3epP<3@X<SH|Nrnm{J(;$6s;ZyvQdbW"
    "RBsRAri_t%U%muWutO9gb?ovOaa<Sjdzt!t!FEwd!iU$jgrEWB{p!28m=vOr4ugR~!kxh{L=H0t`$Qo|J{v2EL4C+SSi8?*(ox8Rfs9EakipS9Ng<}~QOK#b"
    "W8FlH9^~V1+QftPqL8l%Ih}+PgPYi1Vqs#ALZXzTMu=~WdO98`;$v#0BC@O+y@U&c_g*q*WuitQRSyrh5ji@L+s>Yr4OWXnsBgtOiS-Pw>X1cYGKxaxfMF-m"
    "pbfbuk|Afx5rr_dc=Qnm7<|KyEfjol6ms*yxe0=U!KdYxvY8H25&iA^n}|3q$Zu~}<zZSEh3o)#1_(n2FRJPmU{Z-fR&L)tK$K}hzG-reOz<!jSyB6{gYaSS"
    "g|A%snP~F4BNft1a5H$-M4u2I8imZCo2-C*_~JqCV4D2r*P#sVQ-7B)m=cAA-WnVw1R4BLoL7VFnkd9|V0jl2tpRyW-J{t!^C;wT(%}w*!r<6sXj@PV6;aCO"
    "tS7Y98S=j$+2W}gh49{RZXy!ZARmhsYnOdWMWmlT?jsZ!{NS7BURl;CB*@;Rj!002eDgi_X<~|sL~e=bB4inS!c4lu^Dz~{ldGGEhUJX*S|p8oPDUb+FD(5<"
    "I5D_!)3P4fosme!-D5*U(K5(YTn76I{Yd0)z{gR-g26wCiBHBQM<R3L?yW?w3glsN{llP)iX7*E)<YOEcoOIJ7A!mp5w;wvC&n4$tD=y-H%^C&D7snp5Qi9i"
    "?DFg&Q9wn`PgYkG!%7VOaT4gjl&FYzR?`5H%-}0Bruwl_DpLMNp^Dhf-~}tYhJrk!knBkRTB2ep<m>j>4q?$$MAQHHI1$6(G6#hE<K9r|<BP#19vci{U#W=A"
    "ZGj=8P!aN1sU>v{H2JrD-_S~gF?edJ`naqk6%m$uSVLGdxR&*f-iFOl$P(?b4kC96<Op_XDCh?jaXD+&NQ5%D;j@t8AYUr-Wt;spVZq>^F8msb3#QWTm;&Ta"
    "j}Q0C7E+ObXTPS1lMK#oacY#fPep_p)Orbh1|Mp`CcqIYlC&_Off$yDT#tHqj^K_$=HKmN@B>%X#=r|IGXMS;gKwi=@5hXz5YDfKvxE|ZyNHH$2eng?1Sa7D"
    "qGd7U#L%K~kVQpG*L$`T2@I~gCSZ^_NJZw)-x=KM<7itPt-t2a*BIP%Pijw`ITev_tehZf<RFi)+%h41nTj}RE*T*1FgUlvl?m)76<PXu%_QM2_m_6%%--lG"
    "#QrDFq|(?g8HFe>5A7t9Wg(A??VIrAib5`aTU1GCFt|={dXt8K6jFV0TR9OY^SAu9^3}TJ<xz-BNJ>2+#^66H*H3E5L?P3!*awLiX~<{M+@3h`D8$1lwt*01"
    "@NJ2KZT&PkQ$mMm-%I^1_er`n)R0a^413O$6Z{NbeLHa=j%I%vB`Z6K2+6<5-wgGuAo%~DJ3a}Xh>NEp?Y+63M1%yR{@<v5$uxUt8@Z{H;QxEBt@&Wo^E?&#"
    "c>ZZ45w-|&U$cSHqEl34@0+<Hg5&Rb%~WD%|2NvW-f+E%2oq<_cgvewJtwKi!SJqOg5&S`_UC(hFl#DOvMi*T2or<+=cnueBA<$!*i=43aQr>Lte(*kv5Sh-"
    "%<O9+!bJa;UoCqul6;nmgs6TPB{&%Tkg@i7&{NX>i2N-tk5uW^m!gvI;~>y(^PzolAE`*wmF7|6r~)9lmwC5n@V-bSZKkD<c&Z6`mj52U;K)eiV~5)~!KDwm"
    "x~jN<DQ$l7GVALhj;w@SKg?;7X>cU+HG1bT(Q5>G=iNMEJTwyV<h2|oJdGh|)^1b?W{yC_Mw%;$PzFCQvy?OVHWjJv5$GirGdOnEoi}(m5=od1?IVsd$`2Y?"
    "D$G<)MfUNH_Y>b3^|4>T#~dsWfz--nwGsLZ{)xCdL_CQ=)b~EFAwmrySI!$=#AFhQoL1_uB8CkhPmwDUGj)zcwqBMPCCnK7=wTZ%Q(h{fCiJqNc+Zf}An!*J"
    ")26@5Z7y>UACov0IrK=Vf!L)7`NlPOxlQe<NNON&FOjbc`NKWoJa};=a*U;ZoG@eX1<#cin6^bC2OT7uiFh5z6FnL@nb1gt`s3C(!NK6~PS`ELry~(nsT~u<"
    "ac#(LljB59Rj7zU&1^R@s|EQhJt{xGjEcm2^mh^K8Ql8TFMgb+SK)z|JBil}Jt>aQ<1^KyB5Q44_7jo}zJ8q!kLhkIa`(9NIFY;pa`zpsLbw|hnSbAn!7W_-"
    "hcO2#!pBzBLnvrK9@^J6>^T-mx1Z{e=j>PQ4T_3Hnl3ty5Zr2zv+vgG17f7zQ-z%LLq3fv^bnNgjQTfFdkLDH5163cgwZm{Bi0`4B3?%#YswRv32ha~_1;_d"
    "cy6O2QF$$OguOE4(sfN;o*yHT5396?h@(o3@sc(@AS+Dv`%=gsSgQ>Ooune^T(4S)1VzZF>y#$I1}ZY<>(D_oEP?!y*UAo}HWD!??&={V8T=-Gq&rBE=8x%x"
    "4-#&Sas6z&=2+ZFB$9Poag2zTXUNU;`ULSR65-z9H$qSt+{iP&v%!vvtj^}1B*0?GBQ0b4Fq=puj%~b`xFiSp<^4z<k#iNPKNnO<jLSmql5?e3mO@2F>Lmt<"
    "7#YY-8+AtdY5HUDx3-_)kY?yf`>#HYHE^DSJhv-p*q_GtjUR3z#3Uh)tGwJ%MAOeWmR&uBtOVrzHglb_J7|7FVf-i&BM!OJp<N^XG<&ry_R1I$BgT+V&&$5G"
    "5>#aV{Yp{DukiMb%C0BJqX^`cOL_;2j!}{Mcsyaq0}sdy_$Slm3wG@WqGBQB?^A2`R>&gQ{5zxz0W~z*OYr3+_LbwLW2>+{=33<2L97)5YTuaC*5=dn<ff)9"
    "b2KM*GqQe+&=3U5V#C|IY5e+vz;A9^18C$6`w_xT05}y0-s+=qCgEMD)Fn&N`S&jQAy2r*)@t35g3iY!^8N#lN9W^wcp$G2Bl3Q~<0&fNK1%d+gKO>P3A_z7"
    "d-D0&?S3#tf0u^~EDw3E{EWuG{c!7iXVrj?Q}1*W5uBiBrI_>!8c#~P(voFWgwFfz9FUX0OU4<JT=10h2*!Q@A>XSd`Ny>rW1e{zkl5G<Q-mig<k6MAgUS4n"
    "2>Grn$v@lH4+A0+A>XSdx$$y~K2RTtsIB80A|jX}Z#vV_akMLvzD{I<T(rusbNS^+WL};G0_*ud{8TD3{O)TH(Jq0(?~iHnnj%!>yP?Ds(X9!2?~*klrdz29"
    "S5?dqu~-js(?>U0gQsc!Ipx#<(azwmu@+KHlYiyv&Cg|Uo+!jqM}C4h$>5?2t+K&wG(YE^#RS2{;HRqh@dkgUB7LbXqr`Px$a5d6bK*I)`ZqlrBq$6{?qi5n"
    "+w~I>I*><1C~z@NP?1-PA_D|7gMYoYO**)hio7l`93`T)A-{G@hX?PcjqC8O1fjy<8|-bRO~<H6r_JgPB1a4ISYn+ZQ!}mpH%3Q@jSOy-${~Qir6N0Ar3s>+"
    "A<r*uo25)?{%v#4<$mG{gCD5-D#8>=ML?8B4`IRJK3DdNnLeYfr<6{15_KyW{Sv&D6IYBvz96&XL^Oj78nR4y($;rR4aFLW26f1*Uq!SRIWzn#lIKesk44b@"
    "Lvmb`yxZ*bAh;BX5HDYk5blbQ1M%Qd&(28X?f=u>na5N0y?fko97FXjbF)Nbo*R&}cA=4siKH2&GL(uE3Js)a5<(P}lFF3Q*+-L5LQ&?ZWJrb#XX4!B9&7FI"
    "@B4k-`@8q|y7%|`4bGqM*LkhI_S$Pb`&ny0&+}24+0MPAhWL~p9fL_pQB>WP@qOGQ8i?O|U&IKCjG|UOUDeCYB6yBuVLyKS?4SNG`FZKDrtm20EF+CC+?(o%"
    "KR?m>i<^k{`@}ku;_Y*vMo}m26XxyHRT1|YX7?n;M^l&duIu9}68ugjm$$ErLhn1sQ=R00HF1dd9pu^(fxWmSw5!U9d*WyMz#`&(2f4Cdi1uOLI^pX?D}j*M"
    "xPr7U-uyr3k{+os7jJrR+``qK2g*w8H?*;sc;4;W-bf7zywYG1hij?`n(!Vs8k>njT`+#Up8{~yxS3wc<4Nq2PosJhGbVQpaqH$HUVg)-?t57YjqkS*{3Y#I"
    ";T8Xxpv`jr05?e<q=z|w=5#Xgt#%EMgKQ|E@V&}6?zuVOc6j~e4;@TAv{A2dNn9V5Ik>rjYb6I1bR6nm^LS;#Z|w#qb?AKuIgHDKc@~0J$|+xZ$N9f2ch2;9"
    "7;j>Z>S>?&y!oYo0t@MYz&yK+Q4U>28Z5lh?%dwd`3%3yZ<-qJSH<J`SMM}UUH=N@gFp`Rq5i+PLuH!bFUAs-Z!#(OVc`}N7XI2Wzs+b^62)`gwQysv+`__P"
    "U1d~9>x)LNKaZb$re@80N$9w3z06b#ie%w1p9RGgL{R5_zsJHGZ=_|7>IuTQK6SHB`=C-5F6o!N;Ao%``u~moHwD=>Ed1CB?>uEoqQ3q6fG5_rGHg6`$BEC2"
    "&6QEUMosH$Cv!74&X508LOM8{bmzR|N7=Z1%YdA0k}7Kd-A2`KNA9w5SFh}4qa4NyEpHxvTT;iyi{o}=<W!-0ywrG={_6^29DMSWfts$9CTQF?x6SJ^Sjxe-"
    "(OqOm_5NUdLgAp^z#&f#F6ybIFYpsX|KBE>tn}_Q2M?=T6|ZD86UFy>y_R`8{|W~Wti@f|J)ec@W9M`+%Y9yO@SJVnUiEW;`g86<vyD|32Tu>w-fEyj=#nTk"
    "F1hhUav1052Tj1?_rd*D_4es=hw+P-MxrdNb<y)ZDY^BRi1RS6M-MtmDqMluSw0+?bv|$yf3!>-%yTDn{SLTN_e9+r#*;M{?l<f(LH#k?vR!xG`(gaBkJN-$"
    "Z-{=XayDvT6dW1Tw^BuM(H!Zej-OYK;66cnBc8k?^v=Yat`4{+jNscIYX?-^TZ^un^{I7w)@&}m@4+XfHZw~?&ray($a)7ZZgnDLpB7Gx!#&HkB&|za{1syj"
    "nBr`T=EMB%fWBOc06ok___Ar~a`apzR^82cEH6Npu3S>>?P-D9F}V8H_XZ|Ff1nr2vVU%Yu3I*LyT+N#0(7X~2iv(NQ?a<~mH>S>zSQpUTxF!|)RK!ow0<E#"
    "ca%%uG^Y|eWyX45TIS6H^zXu*PAV|~)vax6n5UmSlSE%Wo&D4S5c+GfBI-Q~79={OroJ!h1ECjoYVGyX&w(U*eDhpKiSU(Z{MFmM)Sg9==mj2|66ReZ`gwd<"
    "%*L{gM9)k+v%6vuF%E^Cdk&ihNc8fHjkJ;NrYIiURo5k1T33*MFevU^^hOKx{~pW!)=R4d=_8Y#%SwOOKymVNzWS6{FF|^4$mXcgczYP%1&-##Z{06Q57g}p"
    "kLj{S{nW`n$}Y_nr1SNUlu#Z1J(S$TUn>OZD+TKJsh8TI>!ya**KQUd(|6pYmg<rSU9)3`Kl+c!km>yAPTa3!fSRV|Ml#(`q2SOneH5QhHQc(te)df=U8V5M"
    "#TCYcZsP^hTSbq$$#j0+fg%p$7`tnJ&59Z%)2os$*2l1o(D+x0y=)j#6r$5Y89pItI%wTxtZAb@RuQ7}?J9A<*CYG&IeUaqoH^_d*F{M8#v>v6GGPtDyW0re"
    "MzhC~qLE|@eM9boQmcIgSEltG*0iF~{dBgjD>_K<$Oyw`<2VZa&EW;q-|IrowzPQi;v0p&dLd=9yR0#?1F0-MH3cSR?70(ql&rwg$6A>1SUw}sj;BgZt$Tzp"
    "-Sc8u)yKUA-!SzXGp<^gKIf&_d~k%|_a9cpODl-b|EMn+x$PS*#!G~*^<cAX@n<7s&+k<XguG4>p`W>}>LN;!LwX9k#OuDq-VmYRiP<9*yWSAR3A;xQgk@BU"
    "(2EvezeZ};p#F#38Qa-*h|uk?9!<>4Anq%tU=izqz9_xT%ikrimRK(@lI82f{6y*RnG#C2i&vrlU$&1D{_Tt?-Mq=Y=ExnQoqO4()7O6xrJp`27JDH<2iZ5@"
    "R4Hm4Ta+G_ze2ksjL<PGoGs}jDJO>AY_&Q0D^ros3!C6A_Dw(xJ2E>blNF$g_&v>ohk7`o80DF7v(^qxLO0O)hHv>NQOuy)*~W%Lw11&GB|q$@C^mAyW`>8_"
    "LR25SDOQw{bWRlW*gE7Qy@2Td>o*sc$y<nGF-3kKWzhTaIc>@nkCa+9QS6D*$#kc5!d^9Blyw%I%xnKY^?7B{b5ZtsIoD4>6f4Lz&px<Q5$P;k-fQ;KxLpLJ"
    "%n$Eg83vF&u@&oGms=!)t*!XY@x(2{-ufMr@Ozpef|btFvG>LZJ0Ji4217Gk1lxFWm7~wqHHZ%cb)1{ASp<W6YS7+3?+C%%%tf%pmVK8VwHu*vn||j`oFi2P"
    "i|N#EyEbHr=9?S7`FH?Z7(0^N!HLQ<L;ir2cKqYgc3}+a%fNAvyC!oqIawIHeDynNRR0C`f3s`JrUnOLtfSXV$k3dy|3BBH21FPOV=lT2og8l2qIQxWHh5ka"
    "6vjq8UJKluL$ovf*;oCXYznr(<A~a*J%M>2^c3G&L1!q~o%?a@EvakJyiF*{o}IOYf|Wc8JvZPkkMeqkq#wSO+d#o)sgWAyv=jEnCZqCpt}F%9p7C?~oZs0W"
    "8Rz8vR>eYC?2FyFH>Qc~`JM7aWAQ8@tl0jJPWNlV4i7KuKKjf@2s=%_!c^Kx=%w!8_HjX$gAgV(|A^_`JIG(6&B84n?VK%yMYm{qtrsHZZ2~<t<kJutQ`k}X"
    "vRsVNmlQc&tJ2#}#`yXvy)>8yL=n81@P)_Yw>R33>TdAY3&CN2@&A-BD|e9ld_x7z|5Ewl<_o1{Y;oy=_&7&^=1IxNqUW1$k}<w6r#0&Tsx7AiW(JV4kLFV^"
    "HqZz^;f-2m;zJiQCf!W^<`usH>DTP5ku~~hLB=$9wXBqNkjG*Cd8b>VL#hcG6ZboR`uF`gpy#PPdmb5c?e#V>&{=@iwfIjNd2InQMyX5rH26Rp`Hk`uGbNJh"
    "1u?<8(BP*d>S+F)(^{A8P7}mrU(bm%o3DhPuhPr&xk?#=82^1_fbz?t4W3wDxh#k&>G_5<tS~^=4eLL){X~!;mOaI=Xp5F98lUNIrmH4y62x5gU3B>7jiEfL"
    "3wMj9d&~tfdA6c#<Bv6H{+yUn(d4Nih+S8lzUa5>%hCK~NvDRpPZ7kRUk&YsPUN0IK{g5F>m?EXiCclViFhXov(q_SY7t22(smsXd)QY>!t{LtQugK$`n>OJ"
    "-rJo?Ct;J!pGeH!vjXw8%l#~}Pm-|x&M%eUS`#|9Q`Y46Mf#GkJkrCZml9=>F3inX`@3hhkucBG;YHtNwUFJG&VIErdkqQW>t+s(<+*>Nc?@~>cC030{Jhjp"
    "DF2HkYVSMB*EPppU1wp2<_YW8v+gb<5|&@uDdzT=(3yQR-{!|24HDM)^h3I6!93)jc&j$5ev>9)5t6sF3q6&P{Yg!*nxMoG7`v_tvd=#o548Dx7C`O5{+p<b"
    "SS))dfR%}aj|BhJ$6@`7XBLwq6CVj+P+tu4A?7=gb#Ve1Jyb+aK3EU&GbR~33l9ijZi@9Ex1_40`PO-sEEVJ+fZ1~b7Vo9%BE6?*yK*04V*#x5)XTd;b#h2o"
    "=x|)XAzVQKGh42A=9{+;>c3WErN;@f05;jY<Y~V7bYwSn$y!I}{^SCfPXzr>?FHnmQUzS#f3s%kz<uKXA9hq&NXBx(q<W3XMM&4ahW7r{_gDE>xL}6A(bSHz"
    "5*o}Kg<N*|_9W%YhUopW@WUk6x0|^D3;ympTDJk(t43RUH`$F`z}Huriu9iD56PC*+HwJZf8BuQ$)zgQPb?cQc;9lRvgYhk<Y!jQDT;NP%>~ch15dAez6ANb"
    "4NQMylco`{R;cKZxE7%c>^!YECG!3VP?^5t>)H&$Zb-<?bg()*0<xXu8y}Jg{oktV76%($M*xTZ#!ktI@DB$H7Al_vBY>YbDuD7FC)Vd=85)fMSdRks-=?^G"
    "wgHdx&mRhDFz*#|m@oXt|NmF~*l(wF0~9onzfm@rvE_pP$e2$31f*~KPFa0k|I`ssC6wOrV3sDbhgY8sd_LDP4ETP_Akw8ZkhZwaX&VLy!xeAayHnBq-kz>v"
    "pq(}hpq?z$dzJpR<X2+y@YsDd(_sGci;t>FaXh|g=1F#huNfN8M$LrDOKgV${7wY(%S5dnbdRha1_vfp3{}bfZok&0n0!)x7}Qkmws$s|gzoD{^Vz$r#fJgZ"
    ";f4B8X`0i_7V-GHYVnZsOOehjEn`)*rXiC9dQ6<=Y)zes?7*~$V9uu3902oD;XEv!k(V)r#~-P-tk^hx8M5d35s`Mv*Ezr>bQ$@P8KEQGi`lDxjpcxF5sB`t"
    ";e-zF({p%vt}E~V?dv5fUCohQt#A=ITM0OTud6-**{k3t4clrR4v2VFm@xW&gy%xD($o~q7;J#GAJ{c<x*@W|OX8-CVE5U8voKcXtOjw85gtRIsu9cv%<HmG"
    "LVEz}Py4`v6`4kC@G)H@WvFQ_dM~eetLj)Z%mPq<8QK}ME$xgaEi3?azTora%mb8o9)~(#{QXNc;*sjD>wj+tUN=xy&_1w0Nm%9`)pQ{=Zgz?FJ69I6KoiB*"
    "Z!}LIK1T~3)<k_C=dWKiWDjc(3FVh$umE2lxfAIQyVRfg9CioxpU_{H61{OFgvbA=zdX+WlEd>!IIm8Ga20%lSYV->MDrIHW8~MZ9gN#QVI>Pp-sY43G)Do="
    "w;0oRJ6?&iz+cwmdP}q0?@G4fasK<F7U@|-{-z?;bM$%rk@NeRJkI|`IrNL5o%g$N<FUq9CV=%xpr6lb3EFV0nK|}et3~yZGAGB#rSUji@9=yH^4$Zni$<^e"
    "&&m&vNS>Uc4(k>^pcWk02x}c=f>$r6eee7#i1Jz7>gEn*kXfK-0{9eW(}3!&sm&@p_>{+E-c(Oz=@p^8Kgg9o1hiVLyNmMv#`(WD=kFH`iF5uZQ$P9e)+2v^"
    "%BI2jcJzETCfCFSu&xeVr=q{TE1SmS@cjb!BY%wtsmbsA@`$MGXyRh#R*o~#dj#&AWKIhYZ7*Ps`K#5aowU`e30n$?_Tl{hN9D)LQ&d$0%+Y!8v&brWN(2+Y"
    "@1*dYAuLV7Og9+Xw^nq&A#=JPDe^eqUoAxKLk{-~NC)b#aL!q8*Gq}Y$PP1$wk(@z%LGto6V*ku*i=cef&HmQ*PS+Xhv_JX_Y2z}=i~hUlk;`VC!_tvo6^X@"
    "04pZA!QHBSe6=Ln_a%jV@#1PS0slMNH}t*UF3$4qQy%B<(@RjDSjhiqug3X528a6;_@1iT-v9kNi3#kj>&}nr1*5u-IW|Ui3<eNB)WNPkEQ`)Xv!(o7m%U?v"
    "!Z)FvqwyQizU@@|qCnztScewcH^`x%_K(5~&G-5oU7(BJn`Y**vkg-ifM3Utf#PcFU9C0?USo`{3k5Vj+}Czt2YncTAAiz^#^I$;^qc42jIsUrJLGRe{%u{;"
    "sX4{!==?3_b+P|44+dB}d%*M4kxA%0t+XDOI%UZK*)yJf9$j~^|G7@<*jspfV#7|c02iXphKO@=lKUwJxF$6<g*}l?h3j$N^ST7Ja0bBEMjRTA4?*W|v7bY("
    "x-rK5lTKtGUd3cGh24nf2)~>EB`>7G?+cK_xTbMF&d2#U|5xMD&np|qJxkDe@A0em;_j_sjP2vNH28f@khFW=EmH=7`jyaMnr(8zG07C#Pa=L2a-&vTgIs??"
    "*R|X4XEDc-0j~H*CrVBuqk423uW|Ayy#5GZ@RBpnYC!c3ZnLKa$?&+De)U1mwMD32J>)u}mU=Z;?)}qp{&$M`XrFbdGCzKjIRo(Hp!#U=d$dYv?Wya$@tmV&"
    "v77N$8tGH)ar?+VYrz0#ly7CK?yf`QhR@GR^ya-!gAa03*bkng>m~`cU-9{GI6E`wevRQZbY3*h$N4xP=YKEWr&V1b+p`?iN6pdcSl;T%0N$ZqdCOcRkZySU"
    "ZI^_FynR9Cm;S;lF<q#R-=#}Yj~#eiJE?B)wn90whmcG8c@=MurjPS~Gp>11EL&y*QNLHwj$vY*%>e5~Nhc!o$VjJtTLrb|<6{O`g=b;(h!)h2+oP`ZCLT|<"
    "jh@<gmyY(K<9wX|Gw?*+Xz?XZy2#H=N;m7<bBh7^I!b-$JC5a((2{*Oh;tI?=MN|O9Q3^LXFq?OkMnW<f5-p0PQBa158gY^0PeUV`}gbARsQwkOJ^DAyBl1m"
    "D_s29Q9N$YJXA74Mi-r<vaKAp&$z|_p(DGJWP|&tFy2qjGwhfG@476d_c<xYW}<W8MYkVpkU37AqeB0xgzi2u;1~lyT?S}pAphgJ^f({q<9wX|t=w7YN02;8"
    "3&1#G;r1C<V|o6(xop-dDLHYZx3Y5BmQj+*0Nf4!jLCai(SC5LSxP;N$1R2SUWxCjK>j@BZ6eAGgnOU=iObY*eLo$#HqOWSI3MTZe4PKceDB1?vb)NN?>Hjf"
    "I~`wzF#ycRg#PwiP9@1Y^v`~~R!558S{~mV=GGH$6+Kp`tD7or)U>fIn%7R3Q_qTnQPeU0#J`yrOPT$QI3GQxclrHq=1k^{-s_IhWA*S`{?B<q|HpRWpy)B("
    "qQ<`)x3P-r_i`cXi~g;gWkm8VTOUn@ajJ^)In(n`|L^(UlA39?l-2l7&iW=U-cG$de{ImnUIwst5Yv2NzZ8Fc>{Sc5iACKw&19Qe^3d46OajIKL}a~kr6&%7"
    "ni%8W(K>tZ`|r}G22~o5!@OV^hYvZdhXeCB{~E6pI&<D{*D4(9-X1$~&YvDK1ZL^hC3_F)<8VIvDrxm^8XO$U<1>Lep40||VP~DZLC{>7ygqxk(^wvm0MJ?O"
    "t6+a{5Ht<BpZj8EhhN)PGQ`yu027`T+*x&h031Ks!&LBf!^6nY{oG6`&`%Bj;NaB{>hdkbbm+$Tn9eLUiIKgey!#h$cepgvp$6lbJE@ubYxY~X{sR0u2D@;q"
    "TX2y3T^#I5l&zJ!-3J5~2o@hQ_Q3h|_awogicqamr(Qss|Ln>7iLUtKiQd1sQ^bLt>AkMF9X(*bP#h^jc@w^N_}F)@6&W<&Kl1HVLpNYID_n|CbH(}kxf8*y"
    "bo0YhnV%rfeuvK}Kc7UY;c8C?2X^W`ynp=%sF=9^jz;26Jnc+wD;JjlJ~Q4+SLt<tWyJwb=Ps|uHzhl@bFD}~CPhzYsHGJst(1-ZG07XRqSy3s=SYFf9pp2D"
    "^e;f&R8w`6m=7K?v$~n<E(Nye9rvl5)Cj`N$Pebs@EOyCmIieb`@Hr@Re^enf&B3cYjBYbg*{xrqGnkMiCkJz2EGnyKmSm?7u9dfaLD&O(ewfw-MBZq`mozr"
    "Uf5*72_{!6+2w$u@JUw;<Gk_QWeYmEF%m#uqUih{t91N`YnM;0oi*-VX8M_nw^7rx8cklv=i@&%v}CDo(?xmQtLNNb{_EK@Ji2+fYZG}7Sg~KPi<>zSlqC*5"
    "4VB5knf`fePX$>3e!glw_3i#i=Y!ux<MP)EdIc!n0M?CEe<HAPzW8G>y<PcemD6H${_kOEZg{x+1+Z<XG6<Z|H>UT&roCOUbnqkd1$Ya}7oLcB2K@2qr*1Wp"
    "cHbvo1`OOI=8COw2MxJWL)>>1pfB^7zI59Mz_6HfL|wog=}`Nx8ZtCJUjsxGJ#I<}djbYIjln%64&sg5v)<S?fg`KK7jC840RB4aqf+bVc6+t805?-1jjbM@"
    "KyOA&6?e*SK<`QSjL5ccAa&_m^~B%};Cu0bR&LWERfMHrAX)VtXg|VJf=5=ObDmnEkb`v}+5y)>J|jSJGl2TtnOu+e)~kPjUgd@PnAJw4<LtXKV?QPBCwQB+"
    "VwS@^SER??YvcRlsZ=lU_!X3^r{oSkDkXm9Hi-for_^}Onm+J+f_Itw3u^$MW6!3*gyt>%;I5gN%EMS|qyrvw&QkBr&H+$X=h*(PSRdrkUkz}B8fn%aZ%0kF"
    "83g);!HS)fm0<VQSry!%ZrWl?`hJzvK~U>q+#KskqmHd3d~ey?)Kzx}!H3?j)a^YxL9)AgBe#i5eNyBvk{8LdW6@b2p2Na`uNT0kDX)!uoLw~pT54mv&6^5o"
    "V|6w05V?$yavopS?k}0-0(BeD(R5c>`+eHZ0HJeF7%nn07}FnYrJdilb&G5_1C+}+txvMo1ozUYKe=-lw2c;(Cr_y{fn(_$R&wof@M@MIZ=MX$zNGAWYP*pM"
    "V*ET0v{-CH<CE+1HLvqC6HIacq8DVb0qI=d`_kTZ>Lv?ZQ7!k-w_k&F#uYe5-DF!f$O$kor7hY7^|Gl26CKVjyv+t?Ur+87tltKpJ#0#INs@@?fR?<o!TNi*"
    "q37PZPjFAK^)Q%_Fc27*zkN&>RRD;^J60dF7y--ONGdb0bH@Ax7R?Z^G+gX60$^R!vAU!9*APoda~|j0?b-N-M_<~w?@0KuZt02BwvRv^LHG-{uT8M0<^7rK"
    "8-6|hYRCVc{|8V@0|XQR000O8001EXV;<!BjsO4vs8av{4*(PZY+-I?Uw3I_bZB!fZg6=401yDEQvd(}0001v000000002($gvK=U;u^DT0KR#zmhHtjflZy"
    "BNl^68f{|`q!o)uyo2{eS#1W>xy#A@?(^krvFb$U=n{&u+0<bahjBFwX&kCUeX8^Q`cN0;yHE37Q?|X?=Et)A$so)6X)leh@vrMfpCuELBuSDaNs=T<k|arz"
    "BuSDaNs=T<k|arzBuSDaN%BwL9?$UQ3s6e|0u%rg000080000X0E)-44Z>&u0QmU;01W^D00000000000Du7i0001NZ)0I}X>V?GE^csnP)h*<6aW+e000O8"
    "001EXSOsnis$l>C`1t?;4FCWD0000000000fB^w%0044tbYXO9Z*FrgZg6=}O928D02BZK00;m803iTl9_0Cs0001}Qvd)D00000000000001h0qV*C0Bm7y"
    "WnXt`WOQhAE^csnP)h{{000000{{a6vj6}9xyt|m000"
)

_SECTION3_RADIUS_TABLE_B85 = (
    "00000&b)EJ0000WGkbVI000000Eojt00000&b)EJ00000GkbVI000000Eojt00000&b)EJ00000GkbVI000000Eojt0000W&b)EJ00000GkbVI000000Eojt"
    "0000W&b)EJ00000GkbVI000000Eojt0000$&b)EJ00000GkbVI000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt"
    "0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt"
    "0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt"
    "0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt"
    "0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt"
    "0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt"
    "0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0000$&b)EJ0002sGJAMH000000Eojt0001>=)7^j0001hdwY05000000Eojt"
    "00000QoV7&0000WoqTve000000Eojt0000$^u2Mw0001BT7Gyy000000Eojt0001>y}oh40002Mgn)QJ000000Eojt0000$q`z^%00000|AKfx000000Eojt"
    "0001>p}=v#0000$xP^E?000000I-ih0001hvcYk{0002srigez000000Jx7p0001>)WUJV0000W#fx}A000000Jx7p0002M2E=i|0001h507|20000006>sH"
    "0002sN5yf#0001>gOYea0000006>sH0002sm&S3x0000WA(nVR0000006>sH0001B_Q!F+0001h=9zdv0000009b-R0000WW65#A00000)SY-h000000APYZ"
    "0001B-pX;n0000$>7jT)000000APYZ0002sXUuWH0002sC8l^l000000ML6t0001h|IKm00001himG@(000000I+jF0001hrO$D|0001B7q56g000000N`&x"
    "0002MS<!L800000%(Hkv000000HA3=0000$9Mf^Y0002MsJM7Q000000Dxsc0000W?bLC=0001Bt-W|a000000Eojt0001h%+_(h0001>+QN81000000Eojt"
    "0000$y4Z2R0000WFvxg70000000@FW0000$w%T#P0001huFZHr000000Dyu(0001>z}#`b0000$RnvGt000000Eojt0001B+1_!$0001>B-waC000000Eojt"
    "0001B0O4`K0001h9N&0A0000003d}x0001>HREx>0002sJmq*m000000Eojt00000dF64y0002sh3j}g000000Eojt0001h%I9&w0001B_wjf@000000Eojt"
    "00000D(Z2-0002skob5&000000Eojt0001>m+W!C0000WQ~!8C000000Eojt000006Yp`r0001>JqCF|000000LY0z0000$obhqM00000Pz`xN00000004_X"
    "0001BGxTx500000ixhc400000062_500000*!FS200000?i+bP000000FaD80000$iurNC0000Wc_Vp1000000MLy<00000O8jxa0000WEGv0H0000003eP)"
    "0000$7yog<0000W2Qzs<000000Eojt0002M@d0wc000003Oji~000000AP?n0001>*#vUH0000WHA8to000000DO=@0001>&IfY90002Mhf8@t0000004R|_"
    "0002M&<b+E000000#tcG000000Eojt00000;0<!X0000Wq+EGG000000Eojt0000${19@$0002sZDn~t000000Hl&Y0000WB@}YN0001BUv7Co0000009=zm"
    "0002MS{HJ_0000$cz1a~000000Bn;$0002soEmb#0002sxPW;;0000003?(^0002M>m72y0002MAc}cF000000N|8B0002sMj>**0000$u#<T}000000HBpX"
    "0000$t|W560001BXPtRK000000H~Ef0002sAt-Xd0001hMW=Z{000000A!Xx0001hp(}F00000WOR;%C0000002Giw0000WEiZDw0000Wc)NK(000000C<l;"
    "0002M#4>Wg0001>%*A;?000000Kktx0002sXEt)c00000NYHse000000Pv4M0002M7CLgk0001>>e_ig0000005Fh100000(LHj&0001hwB>m~000000N9B@"
    "0001>mO*mB0001BrSN$`000000GN+J0001BXGL<r0000$z597U000000N{^60001BLrHSL0001h{04eJ0000001%Ks00000Dot|00001hViI~l000000I-ih"
    "0002M8&Pt=0001h?;UzT000000N{^6000007*%q>0001>qbYho000000Hl^c0002s9$9k00001hel>bP0000001%Ks0000WFkN!M0000$fI)gd000000Kkes"
    "0001>OJQ=r0002ss7-o6000000HBXR0001Bab<GA0001h_*r^D000000N9B@0000Wp=ol!0001hZf1Hw0000003eV+0001>+HG>c0001h3v+rv0000007!~J"
    "000009&&QP00000&wqMA000000LY3!0002MYIbtK0001Bxr};1000000C0~$0002sz<P4P0000$%9(mV000000059c0001>Ab)be0002M0jGLE000000N{^6"
    "0000WiGy;$00000V6=Kb0000005p(50002M`iFAB0001><iUDC000000DO=@0000$b&PVr0001>kI#BQ000000AP?n0001h_>gkI0001BVcmK_0000002qou"
    "0000Wgq3o@0001hS?hX0000000N{y00000$7@Bgx0000Wc=>uj0000008olR0001hw4QRn0000Wy$5?h000000Jw@k0002sSfg^l0001>C>DD_000000C0*x"
    "0000W1*mes0001Bx+Hr*0000006>R80001BxvX-)0000$ax;5C0000003eD$0000Wb+K~50001hP(ph^000000GN+J0001hIJR=Y0002MR8f0C0000005FO`"
    "0000W1G{p-0001Be_(q+000000Fa7600000)xUDU0000W&~JM{000000ECc00002st;BM`0002sMt^%i000000Fa760001hjmdJr0002M<Boek000000Pu=H"
    "00000bk1_X00000r=EL20000004Rz;00000VbgNJ0000Wkgj_`000000Dy`>0001>RM>LB00000pT2uQ0000000@de0001>PTg|A00000)XsZA0000004$I|"
    "0002sPU3RF0001>F5i1V000000Kkes0001BRp)ZR0000WukU+6000000GN<K0002sVeE3i0001hRRDZI000000ML*?0001hbn$Y)0001hArX8)000000Jw=j"
    "00000jrMZD0001B6C!*-000000N{^60001htNU`m0001hDl>dQ000000APwh0002M&j5440001BXGDBJ0000009cSf0002s`2};p0002M$yIzn000000BDy$"
    "0002sDGPJJ0001hQD=NW0000004$I|0001>Ul4P^0001>{&;*q0000004SG00000Wn-z1w0001>(29IO000000FaPC0001>+!}Mh00000$eesY000000K||$"
    "0002sBOr6Y0000W<gR=`000000Nj>900000aV2xW0001BCBb|^0000003eq@0000$#42;Z0001>i_?5S000000N{^60001B9WZmi0001h7Uq0F000000347&"
    "0001hd^K~x0000$#`t_d0000008EfT00000;yQD{0001hnhSkE000000C<o<0001>O+a(N0000WlOBCQ000000H}~a0001>zeRJv0000$urYl=000000N9W~"
    "00000I7@TD0001h@kD(<000000Jw@k0002MwNP`w0001hS6F>O00000004_X0000$I97AO0002M;cI<B000000Kktx0001Bzgu&_0002MkbiwZ0000001S{o"
    "00000N?~)r0000$WRrbB00000063690000W*k*IU0002sT&I0N000000A!Fr0001hXl!%90000$db@o<000000BDCm0001h`f+o>0001Byw80=000000H}vR"
    "0001>jdydv0000WB;<WS000000C0*x0000WA$@be0002su=ssI000000I-Tc00000w1RWM00000V-0>l000000N{i`0002sMTm320001hIwF2R0000002qZp"
    "0000$){S$(0001BH#dGj0000007!*E0000$WRr8i0001BSxkOF000000APeb0000$@0fGI0001Bp<#YN000000FZ=00002McAj&<0001>4tIV)000000Puo9"
    "0001>`J;2d0002sp^Sb&000000Jwob0001>cd2v00002sTB3eH000000Puf60002M@2+#e0001hIk<j60000003d)s0001BVYG9=0000$KF)qX000000I+^Q"
    "0001h%eiyF0002sXykrC000000Puc50001hEx&WX0001Bx%z%U0000006=^|0002Mg~W5f0000$FcE)10000008o2C0002M)5&wd0001B%_o0A000000C0Lh"
    "0001h7SD6Q0002MkUoDv000000I+&M0002MOw@D00001>dR2cw0000001$aV0000Wb=h;k00000i)?>D000000Kj)Z0002sj^1;?0000$!h?T6000000I+vJ"
    "0001hnB#N60002MADe$b0000001$UT0000Wk?3>40000Wq_TfN000000N`~%0000WckOe)0001BPs)El000000C06c0001hN%C{R0001BA>w~P000000DyEr"
    "0002s2KRHo0000$8v1`g000000I+jF0000$uKRPq000009SMLy00000004790001hJOFgS0001BKox*M000000B~|Z0001huLN|!0001BcOig40000006=m;"
    "0001>1qpP(0002M!7YG5000000KjoT0002MJPmZf0002MA3A_Q0000006=j-0000WQxSB)0001Bkw$<(0000003dNd0000WM-_Cy0001B7E^#g000000KjlS"
    "0000$7a4TG0002su3vya000000Dy2n00000!W?wK0002MT5Nzo000000AO%H0001hMj&*+0002s8F_#}0000006=g+0000Wq$6~|00000?1g|p000000AO%H"
    "0001h+9q_s0000W(UO2b0000006=g+0001B=_z!;0000W$)A8g000000AO%H0001h&@6Po0002M)2x6%000000Dy2n0002Mk1uq<0002M@VS6M000000FZD%"
    "0002MCo*)v00000Ajg0}00000004170001>m^5_20001>Vbp*?000000AO)I0002M;x=@^00000wc&t30000006=j-0001B203)V0000$8t;HW000000HAR|"
    "000001Uq!V0000$k^F!_000000KjoT0002M+dOo@0001>847_w000000FZJ(0001hjy`n20001hvloFt000000I+gE0000W96)ry0001>UL}D*000000I+jF"
    "00000g+X+{0002s8Z&`F000000MK(l0001B$wG9%0000W=s<x$000000Pu4_0000$=|gnD0000$#!i7i000000FZP*0000W<wSJA0002sv|NEe0000005Ekx"
    "0002syhU`t0000Wv}%Ds0000008n*60001>a7J{%0000W#CU-~000000DyHs0000$07rDd0000W<c5Jj0000001$OR00000Y)5p!0000W6qSKM000000C06c"
    "0000$vqyBm0002sRHK1F000000FZS+0001h)<<-}0002sq_KfO000000AO}N00000)<<-}0002s1i*nn000000N`~%0001Bu}5^k00000b<Tl6000000HAh2"
    "0000WXh(Fw0001h^xT0!000000MK?o0001h`9^fW0001>gX@7n0000008n;70001BWkz(s00000BKm<q0000001$UT0001Bt3`Cc0001h&j^A*0000006=#@"
    "0000W%S3d*00000j2D7H000000AP1O0001h#6xtz0001>S0;i$000000DyNu0000WmqK*F00000GBtug000000HAk30002sLP2!E0002s8bpFX000000Kj)Z"
    "0001>$3Jwy0001h5>tXe000000Kj-a0001BB|db(0001>7-E7z000000PuJ~0000WTs(BZ0002sEOCNB0000001$aV00000Y&vwn0001hPk@3z0000005Ew#"
    "0001BRXB9O0000$fRBPe000000AP7Q000007&dgk0002szMg_W000000DyTw0001Bv@>+T0001B3$B7d000000I+#L0002MDlv4x0002MWxawx000000ML0r"
    "0000WdoFaq0002M&CG&8000000PuQ10001Br7Lv60001hL)?Nu0000003dom0002MsVH>70001>#_NJW0000006=>{0000Wh$VEu0000WSNnoM000000APDS"
    "0000$Kq7R&0002s^a_JO000000I+&M0002M(H?Zb0002MpBjTe000000ML3s0001BJR5Yt0001hS1N-*000000PuQ10000We;0JX0000W96EzQ0000003dll"
    "0000$nG<xt0001B?MZ_`0000005Ez$0001>i4SzZ0002s%2|Uz000000AP7Q0000WPYZOw0001>v}uDt000000HAn40000$=mvDa0000$s(FJz00000004ME"
    "0001>Qv!6r0001htBHd^000000AP1O0000WkNtDN00000xS4}M000000N{2&0001BnfP<S0001h&#8kz000000C06c0000WaPxD(0001>@wtOQ000000FZS+"
    "0000$5$<!q0001>9m<110000001$OR0001>dgyb&0001>Q`&<-000000I+jF0000$s^W9N0002MlInv%000000KjuV0001Bp4@Z50002M+WLb)000000HAU}"
    "0001>RMvCA0000$D+`1`000000AO-J0002M%FlDa0001BgByfE0000003dQe0002M{>XE{0002s;wywe000000FZG&0001B?!j}w00000Nj!u=000000MKwi"
    "0001Bm%4Mn0001BwM>LS000000AO%H0001>_p)=q00000CtZX<00000003}60001h46Ad%0001Boo$3b0000006=d*0002s)1q_0000007=DC60000006=a)"
    "0001BOPX`Q0001>m5zi!000000HAF^0001BaFKJs0001>6`zDa000000KjZO0001>K!|g|0001Bm#>6C0000006=U&0001>xqfrN0002M8Nh@<000000HA9?"
    "0002s*mQHi0001>o6v+n000000N`ps0002Mn`(2w0000$9N~mO0000006=O$0002M|6g;!0000$n(>4{00000003%00001>0#$Rs0002s6as}n0000006=L#"
    "0000WqDgbW0002siV}rD0000003c~V0001h+&y!^00000`yz!v000000HA0<0002MuQ7AL0002MWHW_80000006=I!0001>8YOeU0002M#zTcb0000006=I!"
    "0000092j%J0002MA5?`v000000N`gp00000vkG&-0001hZ)Al)00000003w}0000$-~4jG0001>v~-0)000000KjKJ0002Mpzd<O0002M?u3Ov000000N`gp"
    "0001>_uq2B0002M9F>JY0000003c{U00000<j``!0001hKBa{~00000003w}0001>Wx;a500000Q?-Ra000000HA0<00000ez9`E0002sS;d7w000000Dx#f"
    "00000E}?S30002MQq_e(00000003z~0000WcaU<x0001>J>`W!000000Dx&g0001>T7PoD000008uf)h000000AOlB0000$+HG>c0000$=mds900000003)1"
    "0000$_*-(o0000$s1$}k000000KjWN0002swn=ip0000$S|o-*000000N`vu0001>7&mgj0001>{xgO^000000KjcP0001hAS80Y0002sl|zO=000000HAL`"
    "0000W(GYUL0001>AXA1w0000003dKc0002sE&Xx80001>pJIkU0000006=j-0000$I_Yu10002M6LN+?0000008ny30001>`P6a100000eu0KS0000003dTf"
    "0001BZ@_WD0001>-;jnt0000001$LQ0000WpQ~}e0000$IiQ9>000000HAe10000$jFxf00000WjIV}300000004JD0002sIe~G&0000W*}jHA000000FZb<"
    "00000t7~z<0000WA<l+C000000DyWx0000$;#G0L0000$W!;8A000000I+*N0000$;y-b~0002srR;`40000001$mZ0000WttoN90000$<okv|000000HA(A"
    "0002sL=$nq0000$APa{;0000005E_+0001Bs{C-k0001hS{jEy0000006>610002s-sNz>0001BlPQNl0000008oKI0000W<IZrv0002s$~cEW000000APVY"
    "0001Bx3_S>0000W0Z4~H000000C<5w0001BT%d5k0002MHCKl~000000KkGk0001>(TH%s0001>Xl92%000000GNS50001h7IARE0002MnRSOi000000H}dL"
    "0001>Dp_#A00000$b^SL000000Jwla0000W4MA|g0001B^OJ`_000000LXwq0001By(n<N0001B8>5Fn000000N8*)00000IuLNc0000WKeC5F0000001$vc"
    "0000$1n+Oa00000UcrYz000000Kk4g0002MD7|mM0000WchHAG000000Dyf!0001>-imL)0002MiQk7n000000HA$90002MBUo?10002Ml<tQ>000000FZk?"
    "0002s^dxV<0001hm;Hx8000000I+&M00000Pw{TR0000$kqn4H000000ML3s0000WFTHNS0000$fE$QF000000HAn40001hl8A1=0000WWGaY2000000I+yK"
    "00000c2jP^0001hJ2{9z000000MK|q00000*Bx%a0001>1xScM000000AP1O0002svgd8U0002Mz*dMr000000MK?o0002s1hH+v0002MYh{Q)0000005Ekx"
    "0000$$$4$S0002s1ayc%000000KjxW0000W{6THN0002sih_th000000Pu4_0000Wn+a{e0000${E&!10000003dTf0002sqR?!>0001hSD%PL000000KjrU"
    "0002M50`Af0000Wo2`gI000000B~|Z0001>-&}0K0001B#k+_=0000003dQe0000$5F%{A0000$)yarJ000000MKzj0000WpXO`80002s$k&KK0000005EYt"
    "0000Wi>hnD0001hpXG=^000000Pt`?0001h(`{?O00000So4TL00000004170000Wcra_g0002M@BoQG000000KjlS0000WdGcz&0001>X%2}%000000MKwi"
    "00000*RyKC0000$!W)S|000000FZD%0001>l5}do0002M`6!7%000000N`*y0000$tu$)D0001>4>pNF000000Pt`?0000$DD-K-0001>14D^G000000KjlS"
    "0000$1+!_u0001B)lZ2)000000KjlS0002sMR94s0002Mgj|V00000001$CN0000W>@I1*0000$5ow7)0000003dNd0002s`Ri!F0001Bdv%FG0000005EYt"
    "0001>aHwd&0000$!hwlE000000Pt`?0000$Qe<er00000=8cI!000000FZG&0000WpdV<!0001h=9!5=000000HAR|00000T-s;A0000W!=;Hp000000AO)I"
    "0001>hLmT(00000e6fi@000000I+dD0001>B2H((0000W5WR^&000000B~_Y0002MFb8MA0002Mf5?eI000000Dy5o0002svA<@(0002M$<v8I000000FZG&"
    "0000Wr*~$+0002M?cRw%0000008nv20002s5HDuH00000>*<L=000000AO)I0000$@a1K|0001h!1Reg000000AO)I0001BMx1590001hZU2ct000000Pt`?"
    "0000W6jEit0001h@d=7R000000Pt`?0002sSqEjn0001BOB9Mg00000004170002s7`$Y_0000$dLD{E000000MKwi00000QgCFz0000$eJF}R000000MKwi"
    "0001h1|(#_0002MQ!<J`000000HAO{0001BH`imp0001>{5y(3000000I+aC0001B=80p#0001BcSVXo000000Pt`?0002s6g*?V0001h!%m7o000000Dy2n"
    "0001h#O-3h0002M-&cx2000000FZD%0000W_nu<F0000W%wUQ@000000MKwi0002su1{jX0000$h-r#I000000N`*y0002M?f+rG0000$6mp6{000000N`*y"
    "00000wy$Bp0000WaC(YB0000005EYt0001>314Br00000oPvr#000000Dy5o0001>>kVPR0001Bmx_u&000000MKzj00000Uc6wy0002MW0HzM000000MKzj"
    "0001>V`^Z)0002M0Gf(G000000Dy8p0002s{uf}t0001hZ=s4n000000Dy8p0002MWQt$F0001hu&9ba0000005Eev0002s0pwo50001>#IK4#000000HAX~"
    "00000+B{ys0001Bt+t9l000000C03b0001B@|0b`0001>YrTp<000000AO`M0000WQ0!d50001>0K|$w000000N`~%0002M_(EL300000Zpw;4000000DyKt"
    "0002M@R(b`0000Wwa|({0000008n^90001hJ?>h-0001h*Vc+a0000003dll0002M<3d`%0000W*W8Lg000000PuN00000W=$2W)0000Wwc?6E000000ML6t"
    "0002sPU={|0002MbLfgd0000003d!q0002M96eaT0002s6z+;Z000000FZw`0000$R*zS}00000n)8Z4000000GNP400000|KL`@0002s1o?_U000000FZ$|"
    "0002s8ZuVE0001hSN@7X000000I-5U0001>t%6m+0002MlLL!D0000000@LY00000yV6v^0001>x(JIv000000I-EX0000WMj}+e0001B&JBw|000000KkPn"
    "0000$QgKtj0001h&=QM40000005FI^0000$;=fYB0002s!551_00000004|Y0002M_zhCP0002Mqa2Gs0000002quw0000$m|Ic60001>cOi>F0000004R(="
    "0002M!l+Qd0001hJ|>Gm0000008EiU0001>^y*K*0002s_$rG)0000009cVg0001hfrC!K00000s4t5^000000BDgw0002sE*nk20000$Of-u?000000FaMB"
    "00000|FKKJ0002s<v5E#000000Eojt0000W^hZj-0001hbv=tf000000Kktx0000W5Zy??0001h`$3C9000000ECu60000$RB=bZ0001>cSegq000000F;(M"
    "0001>!30LY0000$>Pm}1000000Hl^c0000$S(ik>0002sQcsIO000000JxSw0001B8!1D;0002Mu~Umc000000LYd=0001>2)aSQ0001h23U(g000000N9p5"
    "0000WAxc2N0002MP+W^Z000000O*!L0001>%CbGc0000$kYI~I000000Qi<b0000$s?R#W0002M#AJ&=0000000@^r0002M;OaNP0001h?P!ZY0000002r4*"
    "0002sZUi;J0000$3T=x(000000AP?n0002sQyenD0001h8gYw200000063RG0000$AZ0GV000009(9XA000000C<o<0001BM~*1K0000W6nTq40000009cnm"
    "0001>9ke3A0001>{CtZ*000000F0180000WN39va0001h)_{vZ000000GN<K0002MXV(kB0001hp@fS-000000HlyW0000$C>-Ix00000UWki80000006>dC"
    "0001BgH{7T0001>3XF?D000000JxAq0000WoPZTT0001Br;m$3000000K||$0001BLarb{0002MGn0!z000000ML*?0001BdY>jh0000$u$GHJ000000Emk~"
    "0002s@{=q;0002M9GZ(j000000O*iF00000)I~8s0000Wd7X<u000000Pv7N0002MN<%b20002M!=Q^m000000Kkht0002sT0%EK0001h{-cXP0000000@yl"
    "0000$0z*1L0000$DW{7-0000002Glx0002sJw-i00001>L#m5F000000Pu@I0001Bg(N^g0000$POXbT0000004R|_0000WGp9j70001hNU)1Q00000092Sj"
    "0001>rYb`~0000$G_#990000006>vI0001B-mXMI0000$5w?p!0000008EiU0001h*fT~z0000W;JAxG000000EC!80001>letGg0001Bp}UJf000000F;<O"
    "0002s4M0gi00000R=$fs000000Bn&!0001BMaN1&0001B{=kbs000000Eojt0002MI8RJK0001ho5PDh000000Kk|)0001h=GRR?0002MEyjyL000000MM8~"
    "0002sOkqzz00000w#bV>000000N|KF00000wRKQH0001BHOq@Y000000Eojt0000$9y?J$0000$tj&u+00000005al0001h;Q&%V0001>8_<hD0000001%l#"
    "0002M{l-#20000Wgwl&Z000000Eojt0002Ma*b0!0001>=G2Qo000000Eojt0000WKU7pe0001>L)VKy000000Eojt0001hVjERJ0001>o7sy%000000Eojt"
    "0000$+~QS00002M@7s$&000000Eojt0002MtEyH&0000$K;DZ$000000Eojt0000$&~jHm0002sjo^zw000000Eojt0000WNjq3T0000$+2V^p000000Eojt"
    "0002M6$Dv80001hBIS!f000000Eojt0002MHO*N-0001hYUhhU000000Eojt0002MsFqql0002su<46H000000Eojt00000ZeUwL0001B_v?#5000000Eojt"
    "0001Bg(_S?0002MJ?@J@000000Eojt0000$?(|$h00000gz$?%000000Eojt0001BsJ~r60001B%JPdq000000Eojt0001Bvx;6o0002M5cP{d000000Eojt"
    "0001B4_0450000WSNDrR0000002rD;0001>-4I|v0001ho%xGE000000Eojt0001>+jd|;0001h<ok<200000063aJ0001>0o`Cg0001>EdGl?000000Eojt"
    "0001>PeWlq0001>bO4M%000000Eojt0000$#i?OH0000$y#kCt000000Eojt0001BVi00L0002s1_g{j000000Cbu_0000$CwF2%0001hPzQ`a000000Eojt"
    "0001>6W(G#0001>nhA_R000000F;_Q0000$Cq-jG0002s<qM2J000000Eojt0000$Vyt680000WF%FDC000000Eojt0001B#1mvd0001BeGrU5000000Eojt"
    "0001BO?zZO0000$$`Xt~000000Eojt0002M{o-Um0001>7Zr>^000000N|QH0001>)JkPQ00000W*3Y=000000Eojt0000$(z0bh0000$wHb^+000000Eojt"
    "0001B_8MkD0000$1sse(000000Eojt0002sK!avL0001>RvwH%000000Eojt0002svFm0)0000WsUVC%000000Eojt0001BNmOS*0001h{347%000000Eojt"
    "0002s2EAuM0000$Q6-E(000000Eojt0000$>LqAE0001hrzea+0000009czq0001h^Nwgh00000|0#?>000000Eojt0001>BKT-P0000WSS*Y{000000Eojt"
    "0001>cVTHj0001hvM!83000000Eojt00000^2%vI0001h4l#^C000000Eojt0002Mku+*R0000$Y%`2N000000Eojt0001BQ=V!-00000%r%Ta000000Eojt"
    "0000W69{WS0001>D>#fm00000005gn0002MgK=v>0000Wi#m)z000000Eojt0000WQ`&1l0000$=sb)-000000Eojt0001hMM7*q0001BK|hQ@000000Eojt"
    "0000$GO27p0001hl|hU^000000Eojt0001>`VMVC0000$<U@==000000Eojt0000$h;wZ~0000WEk=w$000000Eojt0002s$kuH@0000WZ%B+l000000JNAu"
    "0001>uQ+Z%00000t4fSO000000Eojt0000$F_~^a0000$;7p7_000000Eojt0002sMfh$&0000W4^NCh000000Eojt0001>=2dS%0000WI8ls1000000PL7R"
    "0002s4zh1R0001hTT_fc0000009=?r0000WT?TMK0002Mc~y)+000000Eojt0002s@-lEh00000lvj*E0000007RES0000$$X;+j0001ht67Xd000000Kk?&"
    "0002s-HC8O0001B!CQ<#000000Em%50001>HMDR*0002s)?JK10000000@jg0002M+TCzK0002s?q7^R0000005FF@0002M-Ue|%0001B4PuNy0000006>F4"
    "00000PA_pl0002MH)M=J000000I+~S0000WH&<~$00000b7qV{000000C0Rj0000$t$uMp0001h!)S~^000000PuN00001B!=-UR0000$Eo+QG0000005Eqz"
    "0002siOg|80000$wrz|+0000001$OR0000W5A$(A0001hU~r5;000000MK(l0001hUm0>h0001BFLR7Q000000Pu1^0001Bg+FpY0002sBzBBI000000KjoT"
    "0001>hGlX<0000WLwSrq000000Pt`?0002sSBr8$0001hihPVf000000Dx~m0000W+puy#0002M_<xK*000000N`vu0002s?$vTY0002siGz$l0000003d5X"
    "0001>dirue00000Jco=x000000KjNK0001hY#wt!0002M4~vXI000000N`do0000Ww?T720000$29JzD000000N`Xm0001BPGoaH0002MAd`$h00000003h^"
    "0002MDTZ@E00000V3&+Q000000AOH10001hL8o&-0000$#G8yj000000Ki>900000jmL990001>P@s%J000000DxLR0001>1?O`>000002c?Wa00000003D)"
    "0001>s|9pG00000?x~DG000000Dx9N0002sg(Gx80001h53h_s000000Kim00001BnLu<v0000WbhL~>000000Kif}0002s>Rfa{0001BBf5-0000000Dw_I"
    "0002Met2|10000WBfyM60000006<Vc0000WSCe!=0000Wc*cxC00000002)w0000$bggti0002MB+ZOK00000002%v0001h)W&o`00000E7XiY000000Dw$D"
    "0000$df#+F0001>irb7p000000DwzC0002MW%qPI00000MB|J=000000DwwB0001BlMZ!20002MRqKpE00000002rr0001h{v>rk0002szVnPg000000DwtA"
    "0001Br8;#$0002seEf_+000000DwtA0000$g->-r0001Bjs=ZC000000DwtA0001hoMUxB0001B@D7bY00000002oq0002s=yr8L0000$pcsun000000DwtA"
    "00000YKe6~00000n<9-s000000DwtA0000$Ae(hS0002s-7Jkj00000002rr0002s46JoP0000$W;cyM000000DwwB0000$F}!s^0002MFhY$$000000DwzC"
    "0000$kjr&I00000I!%p000000002!u0000WFxquM0001Be^`w`000000Dw+F0002M80d9C0001B{bP+l000000Dw?H0001BQT25|0000Wv2Tq)00000002@z"
    "0002s<N<a-00000mwSys000000Dx3L0002M;SF{`0000$tcQ(2000000Kiv30001BSs8Xf0000$?URi_000000Ki;80001BUL<xv0000$R-lbQ000000DxjZ"
    "0002s3NLm*0001>;;fB8000000AOlB0002semQnP0001hjk%3L000000MK_p0001>*+X_f0000$QO1ox0000004SG00002sJx+E&0001>Cew{T000000LX+u"
    "0000W&scUq0000$1>lW9000000Pt}@0002MtYdaS0002M<m`<=000000N`do0002M@NRZM0001By!nkl0000006<|t0001>uX=Vs0001>ga(d400000003M-"
    "0001h`-XNv0000WH4~0N00000003A(0001h;FESh00000#vhJA0000006<kh0000WXrXpM0000$F)WTi000000Dx0K0001BnXh(00001BZ#a%X00000002@z"
    "0000$c)oT(0002Mc|?vt00000002=y000002hMgt0002sN>GkK0000006<Vc00000M&5Qn0001B-CK@8000000Dw<G0002MGw*gl0001BEN6~D000000Dw+F"
    "0001>&H#5n0001BGjfhV00000002%v0001>5)*eo0001>?tG3w00000002%v0001h`z3ck0001hTZfK700000002%v0001Bg*kUX0001>cae@j00000002%v"
    "0001Bs!ex50002ML7R?100000002%v0000$WMX$f0002Mx1^3h000000Dw+F0001hu6K7p00000+O3X2000000Dw<G0001Bf{k}T0002MsJ4zk0000006<Vc"
    "0001>*`s$r0001BCBKe9000000KiZ{0000$t-5zW0000WQOAxz000000Dw_I00000_tAGi0001hFVBuZ000000Dw|J0002su<3U|0000W#MX{L000000Kii~"
    "0001h(*bxu0002M6W)$M000000031$0000$SQ>ah0000WBjt`j000000Dx9N0000$I5T)a0001B`0I{9000000DxFP0001BZA^GT0000WnevW600000003G*"
    "0001>@nv{G0001>3;B*f000000Ki;80001hzJGW?0001hRsW7a000000DxaW0001h&Y5^X0002sd<Bm{000000H9$&0000$9kh5r0001hh6|5C000000Kj8F"
    "00000rp|ak0000WcM*?300000003q{0000$W$Ac80000$QWuXv00000003z~0002MR|I)L0001>8y$~8000000N`st0001>dmni~0000$)gq5T0000006=d*"
    "0001B%{qBN0002MfGCeZ000000AO)I0001hOjdb70000WATEzU000000DyBq0000W_i=eZ0001Bv@?%D00000004JD0000W%#C?K0002sJ~@v-000000I+#L"
    "0001>%Bguk0001Bz&?*a000000N{E+0002s@x^&S0002sIYf^@0000006>300002MK;n5o0001ht4WVQ000000I+~S0001Bw*GlQ0000$7Eg~r000000Pul8"
    "0001hQX6_e0000WdsL4=0000009b`U000006gYZ70002s+F6f4000000C<H!0002s`BZv90001hHeZiG0000004Rt+0002M19Eym0001hj%1HO0000001%8o"
    "0002MFpqjb0000W<Y|vU0000007#KQ0001>f2?{y0001>IB$<Y000000A!Is0001>@5y>V00000jC7Ac000000KAw$0000$f#-Ta0002M;CYWg0000003eq@"
    "0001hF$H@-00000H-C>n0000007RES0000$0V8`r0002MkA#mv000000Eojt0001B??8J%00000?1_&+000000Em}B0002s`d)iL0001BOplL1000000Eojt"
    "0001hBYk^70001hv6PQM000000L+&_0002MXq$UL0001B9GZ_n000000Eojt0002s$+>$#0001hjh~M|000000Eojt0001>M%a5m0000W1*MNb000000Eojt"
    "0001>-u8Py0001>gQ|}}000000Eojt0002skrjME0001h39yeq000000Eojt0002MUo?C`0001>mbH&S000000Eojt0001hNK<@30001BE4z<C000000Eojt"
    "0001hOLBZb0002M$H0$30000004$e40002sXpwwC0001>YsQa2000000Eojt0002Mps;*E000007tD`9000000Bn~)0000W^Ur)h00000%F&NN000000Eojt"
    "0002MU+;WC00000gx8Ni000000Eojt0000W=MH^90002sMBR@-000000Eojt0001Bi7tIW0001>4C9YL000000Eojt0001hMoxV|0001>+USo!000000Eojt"
    "0001h9&UX=0000WukMdP000000Eojt0002M5srO80001hiS&;^000000Eojt00000Ag+Bt0001hY5I>q000000Eojt0002MNzQ#h0001>CjgK@000000Eojt"
    "0001>j_!Rx0000W9RrX+000000Eojt0002s?hbxH0000$6$X$%000000Eojt0001hYA=33000005DAb#000000Eojt0000$0Z@KG0001B4GfS#000000Eojt"
    "0001Bvv7Vu0001h4G)k&000000Eojt0000WfRTPc0000$5E76;000000Eojt00000XtI7l0002s6%~*`000000Eojt0001hYSVr|0000W9vF~7000000Eojt"
    "0002MhxC3x0001BDI1VL000000Eojt0001BzZHK#0000$Hy)5c000000Eojt0001B5jTH80000$NFk6w000000Eojt0002MeOP}$00000TqKY{000000Eojt"
    "0001B1bTl!0002Mawm{L000000Eojt0000$rJH|100000j4F^o000000Eojt0002MV7-4p0000$s4b8{000000Eojt0001>Hs60h0000W$1spU000000Eojt"
    "0001>lK_A~0000$=rfQ&000000Eojt0001BnGt|M0002s3pbEJ000000Eojt0001htRjFw0001>F*=Yy000000Eojt0000W%`<>N0001BSv`<I000000Eojt"
    "0001B`$T|10002sf<TZ!000000Eojt0001BH&%c^0000$u0xPO000000Eojt0000WfoFg~0000$+eVN;000000Eojt0002M*m!_I0001>2}+Pa000000Eojt"
    "00000K8k=q0001hIZcp2000000Eojt0000Wv73NE0002sYEY0s000000Eojt0000$GOd6>0001Bol}rN000000Eojt0000$!M=b%0000W(N>T^000000Eojt"
    "00000UeJI*0000023n9n000000Eojt0002s2;zW100000JYA4L000000Eojt0001B!SjGX0002sa$t}^000000Eojt0000Why{T_0002Ms$-Bq0000007#fX"
    "0002sTo-{r0000$<7SXR000000Eojt0002MKPiDg0001B9BPn2000000Eojt0002sFgt-j0002sRc(+!000000GOCS0000$Fin9#00000kZ_Pd000000Eojt"
    "00000K3{=A00000%5#uF000000L+*`0000$T5*9u0001h19y->000000OXiJ0001Bg@b`W0001>JbI8o000000Eojt00000zm<VN0002sbbXLO000000Eojt"
    "0002s2dIHS0000$tbmX}0000005q9E0000WV7h@o0002s;e(Js000000Eojt0001h$IO910002M6^D>O000000QiwW0002sKHq^r0001BM~aX@0000000@#m"
    "0000$#_@qb0000Wc8!og00000034D)0001hT?B$a0001>p^%V3000000I-cf000001s8%q0001>$&-*k000000LYC%0000$y(xk~0002s?3R!~000000O*N8"
    "0001>hCG5m0000W3!0EX000000LX_x0001BVNZfU0002MBb|^y000000APhc0002MO=5yT0002MHK33{000000C0ss00000Om%`l0002MK%<aA00000004nN"
    "0002MT!?}|0001hL#B{G0000001$yd00000fSZCq0000WJ*kjD0000004RV!0001>wyuIe0000$EUb_~0000008o8E0002M0K$Sm000005U-Fx0000005E(&"
    "0001BUe$s?0001B=dqAL000000N{8)0001B(CC6d0002Mu(Xgs0000006=#@0002MR{VlM0001hYPXO;000000FZV-0001>@ehMQ0000W6uOW=000000KjxW"
    "0001Bp(BGp00000th|sw000000AO=K00000W;cUC0002MFu#yM00000004790001BKTCr^00000p}~+q000000FZG&0001BEnkB`0001h`@@hx0000000417"
    "00000Fmr=I0002MKE{wh000000Dy2n00000M~8zz0002sX~>X3000000HAL`0002Mahroc0001>d&-bN000000AOxF0000Wu&;wa0001>a?FrG0000006=X("
    "0000$0mOqq0002sP0o-&00000003-20002sWY~j1000004A7830000003d5X00000+U$cs0002stI?1^000000KjQL0001>Ujc+b00000EYpxc000000N`jq"
    "0001B^%aCb0002siqw!m00000003w}0001hnJR=p0002M#?_EO0000003c^T0000WOFx7_00000-qw&n000000AOZ70000$2vmeX0000W&)1MZ000000KjEH"
    "0000W&S-={0001Bm)MX%000000KjBG00000lYN9h0000WHQA6r000000Kj8F0001hLz09*0000$r`eD|000000Kj5E00000%BF-s0002s?b(n(00000003e@"
    "0001h7Q2K%0000W3EGfA0000003cyN0001B9nXY70001>`Pq;_0000006<_s0001>+v0>k0002Mz1fgJ000000DxXV0001hQ}%>F0001hQ`wL}00000003P;"
    "0002MhY5v10002MxY&?D0000006<$n0001BdmM#80001B>erA!0000006<zm0001>FfWBb0001h=GKrv000000Kiy400000s6mB50001BtJRP|000000DxCO"
    "0000W;Zub`0001BH`I_o0000006<ni00000++~G80002siPDfj000000Dx6M00000mUo3f0000WqR@~)000000Kim00000$4vB?80000$fX<LW00000002}#"
    "0001BLYjp@0002sBg~LM000000Dx0K0002MFsp??0002MjmeNe00000002`!00000*t>;50000WzQ&M1000000Kif}0000$HOz%T0001>wZo7=0000006<be"
    "00000M%;x!0002Mbij~600000002@z0001B2keDF0001B{JfAr000000Kic|0000$cKd}u0001>O}UUj000000Dw_I0001>kP3!C0000WX|<3*0000006<Yd"
    "0001>QyGRp0000$P_d9e0000006<Yd0002syeEc0000001g(%j00000002=y0001>$~A^S0001hg{Y7~000000KiZ{00000dqajm0000$)}xR>000000KiZ{"
    "0000W%ut3v0001B_nweI000000Dw?H0001>yIqDr0000$>Y0#0000000Dw?H0002MNNR>a0000$u#}KM000000Dw?H00000a(0G50002MN{^60000000Dw?H"
    "0001>HiCvg00000yNQrM000000Dw?H0001>m5qi#0002M0)&u2000000Dw?H0000$j+ll(0001BBYu!T000000KiZ{0001>Aftvr0001h9(a&I000000KiZ{"
    "0001>Os$4M0000$^>L6u00000002=y0001h61Rpx0001hscVox0000006<Yd00000bHIi`0001BJY|qT000000Dw_I0002sZpnr~0002MuU?Qq00000002@z"
    "0002s2GWK=0002s1Xz$j000000Dw|J0002sJlcjp0000$JyDQA00000002`!0001h6XJ$I0000$T1t>W000000Dx0K0001hjOm6z0001hT|<yS0000006<hg"
    "0002ss_=$D0000$NIZ~0000000Kim00002MarcHm0001>8#RzX000000Kip10001h<o$*~0002s*e#Gi000000037&0001h2Ly*e0001hfhLeZ0000006<tk"
    "0001>-wB660001h79Ws600000003G*0001BaSw+;0002sm=};h000000Ki*70000$!W4%<0002s2@sG#00000003S<0001h*BOUE0000$Y6y@(000000AOH1"
    "0001hwjPH-0000$y8w_t000000N`Oj0002sWFv<_0001>`|*!J000000H9+)0000W=qHCj0002sWa5uN000000H9_-0000$MJ$Iv00000w9=120000006=L#"
    "0002sfG~$Z0001>>cWpe0000006=U&0002MpEQR+000003bT(u000000HAL`0001>r8tK`000006Qhqn000000MKzj0001Bl{|+)0002s29u9K0000008n&5"
    "0000$azKYb0000$=7Ntv0000001$RS0001BK17E=00000vT~0=0000006=&^0001B`$vaB0001BYGRK-0000006=>{0000$s!NAJ0000W5K@mo0000001$pa"
    "0001>OizbE0001>qd|{A000000MLIx0002s;!=k|0000$B{7dc000000Qi7F0001BZB~as0002slpv2l00000062m`0001>>{*9E0002s@(+(d0000008oNJ"
    "0002sU|okm0001BKmU$E000000O*5200000%3y~;0001>dhU)u000000O*B40002sC}f8~0000WrQMD|00000004$S0000$eP@S20001BzRZq50000003e1y"
    "0000W$7+W_0002s#=DL{00000062y~0000W2X2Qz0001BzN(Hu000000QiVN0000$J#mLX0002srJ0UE0000002qlt0002MX>^A`0000$e29)f00000062+2"
    "0000WjdzDY00000LU)cp000000BDCm0001Br+SA#00000_+^ej000000EmY`0002sxP6B}0002Mom7rM000000HB9J0001>!GMQA0001>GeeF*000000BDOq"
    "0002s!GniD0001BxiOAF000000N95>0001>xrT>80001BFCdOU000000I-Wd0000$sELO_0000$l@5+T000000MLs-00000j*N#u0000W>->#C000000Pu@I"
    "0001BYLACN0000WG3$*$0000001%8o0002sJd%e%00000XW5NF0000001%2m0002M1(k<D0000WjmM2Z0000005FO`0000$#FvLa0001Bq_vGe0000008olR"
    "0001hc$$Ym00000tfP%U000000N{o|0001BBAtgo0001hqmhk300000004(T0002s!Jmgf0002sihqqj000000H}&U0000WSE7eN0001>V{MH<000000BDdv"
    "0001>;-rT_0001>EL)90000000C<o<0001>WT%Hf00000=17e|000000F0180001>-KmE_0001BkT#7#000000HBaS0001hPOOJO0002MD<q9T000000I-li"
    "0001Bx2}gk00000xDky&000000K||$0000$8L@{z0001BHUEr2000000N9W~0000$bhC#*0000Wr0k49000000PL4Q00000%C(0;000001lo*1000000Eojt"
    "0000$8n}l*0001hRmY4#0000003et^0002MWx9tz0001hn6!*Q0000006dsL0001Bt-Oao0000W&7q7y000000Eojt0000W@xF&Z0001h^Nx%_000000C1Q<"
    "0002MF~NsG0002M3w(?~000000Eojt0001>Z^MT`0002s6={q>000000Hl~e0001BtHp;v0002s5mt;q000000Eojt0001><j03V0000W07HyG000000Eojt"
    "0000W9Lk440001h;4h3o000000Eojt00000Qp|@y0002svK)*+0000001TNx0001>hR%mT0001hcL|I@0000004SM200000y3mI}0000$F7}H+000000Eojt"
    "0001h?9zuo0001>*W!yn000000Eojt0000$9@U3H0000WbI*%F000000Eojt0002MPuGV)0001h0lbSq000000Eojt0001>f!T*Z0000$f~bo?000000Eojt"
    "0000Wv)hM20001B^^}W1000000K}O<0001><lTor0001BTY-x}000000N|NG0000$7T||K0001>vu%q&000000Eojt0002MNaBY;0002s{#lDa000000Nj^A"
    "0000WeB_5f0002sJVuK^000000Eojt0002Mu;zzA0002sZ8D2M000000Eojt00000=jew(0002Mksgac000000Eojt0002s9_xod0001hs0xcf000000Eojt"
    "0001hSnY>E00000vi6EV00000092Sj00000l<$W?0002suHuS7000000Eojt0000W(eZ~s0001>p3aIu000000Eojt0002M5A=sX0001hfx3!7000000Eojt"
    "00000Q1*vF0001hSEh<U000000Eojt0001hkobo{0001hAd-qe000000Eojt0000W(fWr#0001h+<l5b000000Eojt0002s5&efi0000Wi)o5L000000Eojt"
    "0002sQ2&QO0002sELDm?000000Q8nX0001>jRA;20002M!9a>Y0000000@^r0000$#si2z0000WNi2#$0000003eq@0002M`2~nT00000!xxG`0000005q3C"
    "0002MB?yQ>0001hF9V7|0000007#cW0000WNeYNS0001Bj_!#-000000ECc00002sU<`;r0001>-`R;k000000C0yu0001>YYvD&0000WBgKh8000000EmY`"
    "0000$W)O%#0001>Sg?se0000003d}x0002MO%jMe0001hfSZXx0000009b=S0000W9~6i{0001Bnudu$000000MLLy0001>*A<9B0001hrg4ct000000Kk7h"
    "00000bQg#~0000$rCf<X000000HAw70002s@EC|d0001hmPd&|000000HAq50001>OB#qk0001hdNPSX0000008n>80001>fE$QF0002sQ5}gu00000004DB"
    "0000$jU0$T0001>90-X(0000001$LQ0000$ZXJj~0002M+wzD&000000MK$k0002MA0CK60002MkKKqs000000Dy5o00000o*sxm0002sImn1W000000AO%H"
    "00000<Q|AX00000+_H#3000000KjfQ0002M?H-6g00000fSrgy000000HAC@0000$xE_c=0000WIf#fr000000N`ps0002sLLP`f0002s9CL_30000003c{U"
    "0001>iyeqS0001hJ6?!D000000AOZ70001>ksOFX0000Wph<{8000000H9<*0001>QX7as00000R5ge|00000003h^0002s%o&J40001BULlA;000000AOK2"
    "0001>`xl5n0001hzzv8%00000003Y>0001h*A<9B0002scKU}v000000DxUU0001BSrdpr0001Bf9Hol0000006<$n00000dJu>}0002M*VKnV00000003G*"
    "00000G!2MA0001>e!_=900000003A(0002Mg9(U00002sa<GR$000000Dx9N0001BW(9~q0002MwVsDS000000Dx6M0002s*#L+@0001hM~#O-000000Dx3L"
    "0002M-TQ|?0000$DSU@O000000Kii~0000$a`uNn0000$U2BIx00000002`!0001hm+*%`0000$<ywb8000000Dw|J00000PwIz20002M!%Bxh000000Kic|"
    "0000$n&XE+0001>`#Og}000000Dw_I0001>dfbOV0001>jw^>i0000006<Yd0000$^3;bw0000WfE<TF000000KiZ{0001B1I>p(0002s(G7<{000000Dw?H"
    "0000Wu*8Qz00000hW~~@0000006<Vc0001B`@4rg00000pYeu300000002-x0001h=CX%C0001>9^{5V000000Dw<G0000Wb*YCy0002s2G@o_000000Dw<G"
    "0000Wsh@{H0000$SjvV#000000Dw<G0000$g_Vau0001h6uyQ)000000Dw<G0000042y?A00000JhFyB00000002-x00000K!Arp0001>&ZdSy000000Dw?H"
    "0001hBXx&B0000$%$tTl00000002=y0001BxoL+$0002sGLeQr00000002@z0000$1YU<g0002M|AvM@00000002`!0001B2U3SX0000$Eq#VS0000006<hg"
    "0000$#zluf0002sv~z|)0000006<ni0001>L^+2*00000j%tQL000000Kiy40000Wh%ARd0001>tYL;g00000003M-0002smLG>e000000$GMY0000006<_s"
    "0002scoK&|0001Be^7=%000000AON30001BHU)=30000W6-b6a0000003c>S0002s*7t@$0002MwLgYH0000006=R%0001>Vd{oJ0001BS2u=0000000N`*y"
    "0001B+uVjg0001h@-Buz0000003dZh0001>M$d*o0000$geHbS0000008n~B0001hq``(j0000$3?7C+000000MLIx0001h^s|OQ0000Wj1`7J0000003d=u"
    "00000G^U0?000002MvZm0000001$yd0001BVVH(M0000Wfdqy?0000004RY#0001BdWwcX0002M{`-YM0000005E|-0002se0+vL0001hgY$(z0000005E_+"
    "0001>XKsc-0000W6YGUQ000000I+^Q0000WIbeoB0001>u;PV4000000MLFw0000W@ll3A0002sT-t>|000000C0Uk00000kVA$*0001>7t)150000005E?*"
    "000007Bz-I0001>;K_wR000000HA<C0001hgeHbS0000$w!wu!0000007!#C0001h+!uyG0002smAQpL000000BD0i0001>9|?v)0000$e6fW<000000N{r}"
    "00000Q2K>H0001BW~qfh000000QitV0001BZ|j9X0000$P@#oD0000001S~p0001hf82#Y0001hHkpM$0000006dsL0000$g3X0M0001B7LkQO000000Eojt"
    "0002Mc)o=|0001B?TCdy000000Eojt0001hWUqxl0001>xqyX0000000Eojt0002sMWBU10001>d3c3D000000Eojt0001>9g&4V0002sEO3QD000000Eojt"
    "0002M>w$$p0001B(`SW1000000Eojt0001huyTb!0000WZC`~z00000034Y>00000Yh#5#0002s`c{QN000000Eojt0000W8&rir0002MeNKfy000000BDy$"
    "0000$z(j>W0002s^+kn1000000ECx70001BS~Z100001>WIcsI000000HBvZ0001h=p}_f0001h$~1*Q000000Eojt0002sY8Hh+0001hC@h6Q000000Eojt"
    "0000$;Rc030000$ej|lI000000Eojt0001>OZS980001>%o>G20000002G)&0002sspo`10001h5)y?#000000Eojt0000W{MUp)0002MPzi-V000000Eojt"
    "00000L&$_c0001hh5&^?000000F;(M0001hez=4{0000Wv-gBR000000Eojt0000Wuc(AT0001h*6xHr000000L+#^0002s)R=@o0001>@8yI*000000Eojt"
    "0002M@Q8#!0000${@jE>00000005Uj0001>0(pc$0002s0n~&*000000Eojt0002M3TcEu0000$`pSer0000005q3C0000$30Z_d00000=fQ+P0000008p1e"
    "0001B{z!yC00000$hd?+000000Bn~)0000W>Ntcz0001Bov(yI000000ECx70001h%qWCF0002sW~PKd000000Eojt0001>r5J=j0002sB%Opn000000Eojt"
    "00000bqIt&0002M*OP=m000000MwU20002sI{1S?0001>e~N@a000000PvSU0000W_veE^0002M9D#&D0000000@{s0001Bsn>%*0000$uXlt%0000003?_|"
    "0001>QOAQo0001>H*SPL0000006>^P0002s@3w<L0002swPb`q000000Eojt00000gr<W)0001hDqDm=000000Eojt0002M4VHsI0001Bl~II1000000F0PG"
    "0001BjfI0i0002s_D6(3000000Eojt0001>0(FBx00000Pd<b|000000Eojt0001hZDfN$0002Moiv0%000000Eojt00000&QpUx0001h<0^ze000000Eojt"
    "0000$BtwHh00000AR&Z6000000Eojt0000Wa595H0001hQx}9l0000004$k60001>u_1#%0000Wehq{`000000Eojt0001>=n#WI00000paX<J000000Eojt"
    "0000W6##=k0000$x%q=Y000000Eojt0000WH}8T#0002M$?t<e000000Eojt0001hPTqn*0000$(&d9d000000Eojt0001BTg`$%0002M(%XYT000000Eojt"
    "0000$UA%%o0000W%h7{C000000Eojt0001hRI7qO0001hyT^k-000000Eojt0002MK$?O;0000Wq`iYc000000Eojt00000B8h@P0001Bg|mY|000000Eojt"
    "0000$_;-Rp0000$U#WvY000000Eojt0001>!)Ag&0002MF`t7#000000Eojt0002MgH?h+0001B{gi`1000000Eojt0002sI7EU#0002s!HR=G000000Eojt"
    "00000<1&Ik0001Be}IEP000000C1Q<0000Wf+2!H0002sHFkqQ000000Eojt0000077&6!0001B<!ggL000000Eojt00000pZ<YB0000Wj$wm9000000Eojt"
    "0002s9PNQX0002sFIR&=000000Eojt00000klTSk0002M%1nbm000000Eojt00000_{o7l0002sUP6OF000000Eojt0000WRk(pb00000>^Flz000000Eojt"
    "0001Br>22G0001haxH^E0000001THv0001>?v#N*00000@*;yk000000Eojt0000WD}#YR0002MY8Zn+000000Eojt0002sTycRw0001h+zo?3000000Eojt"
    "00000gkOO`0000WM+1XE000000Eojt0001>piO~50001>s`!FH000000Eojt00000vpa!60001>2JV7D000000Eojt0000$yD5P{0002MTH}I1000000Hl{d"
    "0002sxfg*z0000$rr3f&000000Eojt00000uLXfX0002M=*@ya000000Eojt0001BoAZD`0002MA;W?|000000Eojt0000Wf8l^Y0001hP`H9X000000Eojt"
    "0000WThD+%00000bghCw0000001%i!0002MFTH?30002si=u)+0000003?_|0001h{i=XK0002MmY0G+00000063UH0000W#h8FV0000$l#GHv0000008E%b"
    "00000hlYSa0000$gMorT000000A!dz0002sLv(;Z0001BV|Ic-0000008EiU0000${9%AV0000WGi!oC0000009=tk0002Muup(M0000$@?U~L000000Bn&!"
    "0002MVLgCA0002MpHzZC000000C<r=0000W5G#N{0002MI!S^+000000Em%50000$ycmE$0002M!aagO000000N{u~0002sW(I&j0002sHZp=h000000Pu)F"
    "0000WA?SZV0000$mL`Hg000000O*WB0000Wbi;o@0000$;2MHJ0000002qlt0001h$DV&c000006c2(x0000005FL_0000$9eRI20002MFav@>0000006>XA"
    "0001hc2$2s0001hHTQu)000000H}sQ0002M)G&WQ0001>A?txa000000KkMm0000$IShY50001B^xlC$000000MLX$0002MrRRP?0002stkQu%0000009b=S"
    "0001>8^eA;0000$O2&ae000000APba0000$oSuF_0000$%(;O;0000005F0;0002sD0+TC0002MF|C0>000000Dyx)0002s!c~4i0001hdZ2+o0000008oNJ"
    "0001>YB7F50000Wrjvm{0000008oKI0000$A`N~(0001Bw1$B|000000BC|h0000$=;(bw00000rFnrs000000BC_g0001BzQlb%00000cx-_{000000Dyu("
    "00000rl5U50001BEnk5^000000Dyr&00000o_u{k0000$#ZiGk000000FZ$|0000$rdWMI00000KtzE+000000H}gM0001hzBGM60001>oi>3$000000Jwrc"
    "00000=Ma5B0002s-Y9`U000000I-2T0001>AMJcV0002M0~>)r0000003d`w0000WXvus)000004Gn=n0000005F6=00000z@>aZ0002s`u>1G000000LX(t"
    "0001>CW3rG0001>&G3Lg000000N8^-0001hn_hfC00000hT?!g000000O*520001h9y@$M0000$B-DUF000000N8~<0002st`~ej0002MsKtOl000000O*B4"
    "0001>O!Rv|0002s6Ssgs000000QiMK0000W_t1Ml00000W~hKb0000000@Ra0002st*v`N0001>pP7I_000000MLm*00000aEg0C0001>!HR%D000000N{y0"
    "0002sJZF1A0002s%X@%8000000C<K#0002M6GVGJ0001Bzifa&000000N{^60001h^B{Xb0002soLzuG0000002qou0001h+WmS!00000Wln%V0000000@vk"
    "0002M$J%;80001>7(akO0000002Giw0002sxwm>i0001>xG#V}000000L+#^0001BuatT~0002MMIeAc000000Nj>90001hrg3^e0002szz~2y0000005p(5"
    "0002solbf{0002sDgl5%0000006>sH0000$l`DEc0002M3FCi20000001THv0001BiU@i@0002Mox^`X00000034S<0001>eC2sS0002M6QqAY0000005F$8"
    "0001>YQlLy0002MY=nP6000000Eojt00000RG)c30000$s$_pa0000008p1e0001>H+*?O0000$%tU`c000000AQCu0000W6<K*e0001B)FgjE000000Eojt"
    "0000W>@|5n0002s!UTUn000000Eojt0000Wyb^gp0001BnB;yy000000Eojt0000WgztDj00000R>Xcl000000HBvZ0000WM9g?V00000{G)zA000000Eojt"
    "0000W`>1$80002sii3VY000000Eojt0002MsD*ey0000W1Y>?c000000Eojt00000Ok#LI0000WXhMEK000000Eojt0000$<UV*n0001>w;_H&000000Eojt"
    "0002Ma2t3)0000$^Z<T9000000Eojt0001>@b`B>0001B9^idI0000002G)&00000W7Ky*0002sHotv9000000Eojt0001h$FO%m0001BKc0O+000000Eojt"
    "0001>9gTND0001>H+_9T000000Eojt0000$WNCLm0000WAY6Sw000000Eojt0002Mn?-j(00000`Z|3;000000Eojt00000!y$J-0000$#Tk7-000000Eojt"
    "00000+x>Px0002Mf%tqt000000EC!80002M<JopV0000$FxPxQ000000Eojt0001h+_iQ<0001h(6xL(000000Eojt0000$#gTSE0001BVw8M9000000Eojt"
    "0002soo#kN0000$=5l;M000000K}L;0001>W=VEH0000$Tv2>L000000Eojt0001>A0>7`0000$#4mh6000000OXiJ0001>$N+Xg0001>9u9m!000000Eojt"
    "0002MVcc~<0001hYwUYK000000Eojt0002s>bG@30000WtIB&o000000Eojt0001>W0Q420002s-Kcv&000000Eojt0000W%Wic*0000W1c!S-0000005F+A"
    "0002sB1&~Y0002M9c6n!000000Eojt0002MXeD((0000WD?@ug000000Eojt0000$o&a?~0001hEFpV9000000Eojt0001B!rOE}0002sA^&<n000000C1T="
    "0001B*0yv&0002s3f+1@000000Eojt0002M*pYNV0000W=(>79000000Eojt0000W%WQN&0002sxR`oC000000Eojt0000Wtw(e~0001BeRg_4000000I-=r"
    "0002Mej;>00000$HB@>)000000Eojt0002MK>c$-0000W;4ykY000000Eojt00000^Vf4g0001Beh+#;000000Eojt0001hm9ld{0001>5AAtC000000Eojt"
    "0002MD2;PK00000l*)NP000000Eojt0001htY>pT0000W3#fTO000000Eojt0001>AVPCM0002Mb%l9A000000Eojt0001>g&cE00000W(_(o*000000Eojt"
    "0000W+xBum00000BtUsU000000Eojt0002sAklI_00000XdQV#000000Eojt0001BSF3VB0001hpZa(}000000Eojt0001heuZ*C0000$%GY>6000000Eojt"
    "0000$mS1u}0001h=(Ko1000000Eojt0001ho;h+r00000`jL1*000000Eojt0001>mlJY800000|896d000000Eojt0002Mf$ecX00000_e*#{000000Eojt"
    "0000WUB_`i0001B;wX4P000000Eojt0002MDWP#d0002Mzy)|f000000Eojt00000=z4KL00000lHzwj000000Eojt0002smsD{;0002sRlj#Y000000Eojt"
    "0001BIWBQP0001h44ijB000000Eojt0002M%m;Bm00000ws?0y000000Eojt00000QsHnw00000QdM_A000000Eojt0000W$+~br0002M;4ybV00000034e@"
    "0000$GL&#Y0001BVh(pe000000Eojt0000Wjcss100000*Xnja000000Eojt0002s*+y_c0001>KF4-I000000Eojt0001B7$9&!0001BnWJ_<000000Eojt"
    "0001Bk@9aq0001B=YMuU000000Eojt00000)}(Jh0000$C|q_x000000Eojt0000${#I{50001hTsU?>000000Eojt0001>3JGsO0002MgA{f^000000Eojt"
    "0002s`nqmF0001hp73=*000000Eojt0001h&unf$0000$tju*l000000Eojt0001Bh#qc00000$t*CWC000000Eojt00000CeUp_0000Wp@eln000000Eojt"
    "0001BsDW)j0000$h+uU<000000Eojt0002M4>N5*0001BVmx&~000000Eojt00000T;yy(0001hFBo+|000000Eojt0000$jg@Rb0002M@AGs(000000Eojt"
    "0001hqeg5%0001BqtA3e000000Eojt0001hp7(1&0000$N~?50000000Eojt0002seyD3e0001h<c4%W000000Eojt00000L|JP<0002sabk2p000000Bn{("
    "00000?g?u^0002s@;!7w000000Eojt0002se7b5t0001hXBl)r000000Eojt0000W@oH*70001>&-8OZ000000Eojt0002sNgHZF0001hE6{U5000000Eojt"
    "00000h|6g}0002sdaH9l000000Eojt0001ht9xlc00000zJ_x^000000Eojt0000Wv@2;q0002M^kH*A000000Eojt0000$qT6Ud00000A3bwG000000Eojt"
    "0000$c8h300002sJQ#C8000000MwR10001>FFR;J0001hPV;g=000000Eojt00000&+2DD0001>Q_gZg000000Eojt0000$RGMc%0001hOsR4}000000Eojt"
    "0001hze#660002sID~RQ000000Eojt0002s5cy_60002s7+-Qg000000Eojt0000$NT_B&0001>>^X8k000000Eojt00000W>;oF0000Wv=nkc000000Eojt"
    "0000$Y6fON0001BZ|`wH000000Eojt0001>RJCP50000W9?Ee*000000Eojt00000CS+wm00000!K86O000000Eojt0001>-x6g&00000SbuRq000000Eojt"
    "00000g27}!0000W<5_V)000000Eojt0001h4sc{Z0000WV>EF<000000Eojt0000$h8|=<0001>*A8(&000000Eojt00000>da$60002sK<RKm000000Eojt"
    "0000WJ9}e50000WpTlrK000000Eojt0002MdMRT-0001B@||!%000000Eojt00000s@Gyb0001hI(TqE000000Eojt0001B%7tP;0001>cT#Xb000000Eojt"
    "00000-7{i90001hsVs0n000000Eojt0002s;^ARH0000W&<1co000000Eojt0001>-H%~F0002s*1>N;000000Eojt00000&OKp30002s_H}PS000000JN7t"
    "0001Bwd-I&0002M|0!=k000000Eojt0002sl$l^a0002M@!f7f000000Eojt0001>ZAV~00000W%#m(D000000Eojt0000$f$CpC0000Wk40`k000000Eojt"
    "0002M9Y$Y30002sI{9rt000000N9s60001hvY}o;00000&#Y}g000000Eojt0002MK>b}n0001hOJHq4000000Eojt0001h%2-`M0002MuoG=S000000Eojt"
    "0002sP_$e?0002s{>E%T000000Eojt0001>)DK)h0002sHF|77000000Eojt0000WS8H290000$RxNBm000000Eojt0001>*v48w0000$VBl*&000000Eojt"
    "00000S|VCN0001hRFi8!000000Eojt0001h+kII;0000WGe&Db0000000fvo0000$UD;Sb0000${rGA?000000Eojt0000W-!xc20001Bv#V-A0000001%i!"
    "0001BVUSlq0002MRbFa9000000Eojt0001B<Lg#H0002s<q&E>000000Eojt00000WJgv&0000$WW#Ad000000Eojt00000<DykS00000(sgM-00000005Xk"
    "0001hUj9@-0001>Feqt2000000Eojt0002M+F4XU0001BgW707000000Eojt00000hP6{b0000$;*4lO000000Eojt0001h#t~CM0001hfInzJ000000DzZ3"
    "0002M3U5+C0002Ml=5dl000000Eojt0000Wq|H%400000NTz2%000000Eojt00000$}Leq0001hwpwRE000000Eojt0001hrH@cR0002s^bco1000000Eojt"
    "0001htL9HY0001h6UJsh000000Eojt0002M<grdb0001h9(`s&000000Eojt0002s9Dz+h00000AT?${000000Eojt0002Ma8yh{0001BAMRy9000000Eojt"
    "00000_bW?40000$BB*6R000000H~Qj0001h!3Rn}0001BEn{Us000000Eojt00000;O9s{0001BLmy>8000000Eojt00000V9ZBA0002sWZYyx0000004SJ1"
    "0002MK($6d0000WkD6pa000000Eojt0001>P@+XZ0001Bu~}q5000000Eojt0001hHj_j^0000$uo+}P000000Eojt0002Mt%pNE0001hblPJ;000000H~Ni"
    "0002seSbng0000$=9^<c00000063UH0002sUwT160001h<y~Vy000000ML*?0001h7<fQH0002MPatDJ000000QibP0002sRPjDQ0000W5#wS&000000MLd&"
    "0000$F8Dk^0001B6Q*K7000000C0pr0001>LIpcO0000$QfXp9000000APVY0001>yBRq^0001>-z;K40000001$mZ0000W%{Dhc0001>(ePnF000000PuD|"
    "0002MX<aoy0001hHML<t0000003dWg00000VvaLF0002M4|-uh0000003dNd0001BaLO@20000WVLxF&000000Dx{l0001>I}$KJ0001BDFtCb0000006=O$"
    "0002MwahF)0000$XUkwf00000003t|0001hu8t`{0000$7Ls5<000000AOT50001BIcp|B0002s99dvM000000AOK20000W_+leK0002sP#|DH0000006<_s"
    "0001>nRXyR0000W70+Kl000000DxUU0001hR81Q|0000WNL^n*00000003J+0002MGUpXQ00000{OVpn00000003D)0002M!MhMZ0002M0C-+N000000DxCO"
    "0002Mjh6>N0002sGz4Bi0000006<ni0000WVPE_|0000$f{$H5000000031$0002swgcrq0001h+ZkOz0000006<hg0001>$er200002sI-^`b00000002}#"
    "0001>om=+50000WpetNJ000000Dx0K0001B8?6Gs000005VKoA0000006<ef0000WS+NPg00000lQ~;J00000002`!0001hf*laR0001hGQV0t00000002`!"
    "0001BjwKVo0000$`9oSj00000002`!0002MbMY0x0001h^2b>~0000006<ef0001h9D*3Y0001>DN9*E000000Dx0K0002s(byWm0002MsLfbF00000002}#"
    "0000WH1{090002sc~4kC000000Kim00001>Jl`I`0000WrO;PE000000Kip10002st4Sch0002MF;Q1Q000000037&00000Gz%fX0002MB+*tt000000Kiy4"
    "0000W=d~fg0000$f>2gK000000Ki&60001>u}dPr0000$OwUz7000000DxUU0001Bw9X>H0002MgiTdI000000DxdX0001>IYA@900000EX!0t000000N`Rk"
    "00000rlljm0002sLP%6V00000003q{0002sW&R_;0002M%EVJZ000000N`mr0000$>`f%V0000$#6VL(0000001$9M0000$be1H*0002sF}zYh0000001$LQ"
    "0000W6xt-f0001B7B^Br0000005Ez$0001ht{Nr40001hbg@xD000000Pul80001B#aSi50000WP%BYD000000N95>0002Mo0uiQ0000Wt)x&u000000Pv7N"
    "0001hL)az20000Wj~!4z0000002G)&0000$jTt7u0001B@+(h3000000Eojt0000Wb6h6C0002M*&9wk000000Eojt0001h+Mp)D0002s)e22O000000Eojt"
    "0001>$L1!$0001h==e)O000000Eojt0000W|1BrL0001h5#~xj000000Eojt0002sP<AK40001BQPN33000000MwX30000$lD{Xw00000sJ%!)000000Eojt"
    "0001hp9v_y000007N|!+0000002G-(0001>gj6WN0000WosmXB0000002qxx0002MQJ^Tm0001BK6*t!0000008oiQ0002sjO{4E0001>{b58v000000C<N$"
    "0001>3Oy;n0001>;YdS30000006>F40002MwT~&l0001B@h(C@000000FZw`0001hvgRqk00000HxxlY000000C0Ig00000G(#%D0001BeDOa(000000MK+m"
    "0000$m8L4d0001>EV(^E000000MKzj0001h{t_#|0001hzkfSG000000AOxF0000$fQl=?0002sdPO-v000000Dx*h0000$;SMao0001>qzX4c000000Dxye"
    "0002s0-`Ly0002Mlg~6j0000006=6w00000r$sHm0002ssFX55000000H9((0001hdiyQF0000WQCcuS00000003b?0000$1jR1E0001h2tq7C00000003Y>"
    "0001houe<n0000WgVHBJ0000006<<q0001BKTI&d00000C}$%;000000DxRT0000$OwusG0000W0RkRC000000DxOS0001BTVyf70000WKtdHj0000006<zm"
    "0002sx&ku500000QX&aJ000000DxIQ0002Mo~<&#0000$iG}Y!00000003D)0000$DrGak0001h_;m8W000000DxFP0000$ZYeat0002swk-<40000006<tk"
    "0001hYx*?60000$gnAXg00000003A(0001>6WBGt0001B!p0rI000000Kiv30001>QNlLB0000W6a6B<000000Kiv30000$`nWg10002M)d?rS00000003A("
    "0002s)Vnyq0002MPYEl)0000006<tk0000WqslqJ0001>H~TKY000000Kiy40002M8Ra^_0001BwQn)N000000DxIQ0000W=L$T)0000$?piaz000000DxLR"
    "0000Wr8_;q0002MWIr{)00000003M-00000+jc&{0002sRUJ3L000000Ki>90001>1H3=L000007WFv50000003cvM0001hMukAY0001>4#zsc000000Kj2D"
    "0000$<1j(M0001B)ssBH0000006=6w0001Ba^OM00001hC0sth000000H9|;0001Bf|EkP0001hY7juc0000003dBZ0000Wl}SUu0001BYVkn8000000C00a"
    "0001h8UjSX0002Mn$bbP000000PuW30001BbG$^r0001hj<G_(0000006dUD0001>^>;<U0001B*^xuQ000000Eojt0002s=`%*a0002M4s%4n0000009=qj"
    "0001hpzlV&0000WuT@3B000000Puc50000$da6gj0002MBRWRF000000AO`M0002M<YP#{0002M#2`n&000000Dy2n0001BUmHol0000$)(J?!000000HA9?"
    "0000$aL`G>0000$ko8Ew000000Dx#f00000s)9<u0002s9OOyB000000KjEH0002stuaf$0000$nAJ+a0000003c*Q0001BBi&2D0002s70OG%000000N`Rk"
    "0001hzJyG`0001>putSQ000000N`Oj0000WJSa`T0002sI=oH50000003cyN0000WU&l?r0001B;ki!0000000H9z%0001>?_o~B0001BjJi+20000003cvM"
    "0001>zV}YR0001B7hh1o000000Ki{B0001>q=-+z00000T;fo`0000006<?r00000J_=C40002s3TIKk000000Ki^A0002slTA>-0002s80}HO000000DxXV"
    "0001hNRCjz0001hdUR62000000DxXV00000RLoGo0000$BmGjq000000DxXV0000WuLn`U0001h28UC?000000DxXV0001BS3ps~0001B7ZOy!000000Ki^A"
    "00000Om<Pg0001hNS;){0000006<?r00000kf~9?0002sj4M^Z000000DxaW0001>DcVuM0002M*SS@|0000003cvM0002sAP7>x0001>97$Hd000000DxdX"
    "0000WfiqIT0001>PuW($00000003b?0001>SzJ=U0001BWolQz000000H9$&0002MyMj`{0002sQv6rI0000006=0u0001>yrxpX0002s4UbsB00000003h^"
    "0002Ma>!D^0000WjUrjV00000003k_00000`R7u=0001>$F^C(0000003c;R0001>Yz0%m0001hwM$yS000000Dxvd0000$>>yLX0001>PvKg?00000003w}"
    "0001hm^xFy0001>k9S+Z000000HA3=0002Ml2TK^0000WZw_3*000000N`st0002M{b*Cb0001h=%rl10000006=a)0001>`+ZZu0001B_dQ*}000000FZD%"
    "00000u8~u~0001>meyUs000000Dy8p0002MHKbF(0001h!f#%{00000004AA0002Mv9(jc0000WcL`s>0000001$UT0001BKgLtQ0000$v7=wW0000006=*_"
    "00000`PEav0002MHXmTX0000001$gX0001h_T*E*0001h)NEkD000000I+^Q0001>P4iR00002Mu)1Kt0000001$#e0002s5CK%c0000$$_Qb=000000APba"
    "0000$P!3eU00000AXj0)00000004zR0000$8XHu=0002su%%(Z000000N9B@0000WdnZ)D0001heDq<!000000Pu-G0000$eKS<R0002sfkR@z000000N9W~"
    "0001>DL+)e0000Wz>;FX000000PK)J0001Bh)Ptz0000WH{xQz000000Q`_Z0001>pjA}B0002s<uPNx00000034V=0000$d0<q)0001B%z<OT000000Eojt"
    "0001h7;99(0002s>Ct1r0000008E%b0002MgLPEE0000WJ|bkm000000A!dz0002sz<*T00001>$Z}-B000000Eojt0002M)QMEU0000WiNj>T000000Eojt"
    "0000W#FJFO0000We-vfF000000Eojt00000lbuw+0001hs%2%s000000I-%o0001hM5k210002s3b$pz000000Eojt0000W+^|%@0000Wpa^Eb000000Eojt"
    "0000$T)I@i00000YFcK%000000Eojt0000W$HP>>0000WXsl+y000000Eojt0002s9nDn00001hn*L_M000000Eojt0001>VAfQ?0002M08wYa000000Eojt"
    "0002MkKa_l00000o1<sI000000Eojt0000WtLRj~0000WY4&Hp000000Eojt0001BvGG*E0001>YDs9o000000Eojt0002sp!!t60002MpPXpG000000Eojt"
    "0001hcmq|y0000$2=HjY000000Eojt0001BHVjq30000$q(f=I000000Eojt0001h*c4U30001>bC+qr000000Eojt0002MTpd-w00000cI#=t000000Eojt"
    "0001>!6j9|0001ht3PVM000000Eojt0000$1TIy;0002M6O?Me0000000@{s0000$BsNvR00000ujp#P000000Eojt0000WAU;*V0000WeLZWy000000Eojt"
    "00000_eE8}0002MeUfXy000000Eojt0002ss7+PC0000WvFB^R000000Eojt0002MHC0u>0001>7Cmgh000000Eojt00000o?KPH0001>u99rP000000Eojt"
    "0002M;$v070001hc<5}v000000Eojt0002s1Z!2m0001>bUtms000000I--q0001B269!v0001BqLgjG000000Eojt0001B>3CJZ000000_$zS000000Eojt"
    "0001>uYXm*0000WlR<94000000Eojt0001BTZUD@0002MRhVwT000000Eojt0002s?u=Ey0001hN$+mJ000000Eojt0001hY?D>M0001>Zbomw000000Eojt"
    "0002M)|pkn0000$#GP-z000000Eojt0002sE}&Jw0001hOZ9KS0000009=_s0000Wc&1gr0001BfhKUk000000Eojt0001>v#eFX0001Bby9G^000000Eojt"
    "0000W<FZx20002Mfqrnn000000Eojt0001h2)R|j0000WrmAqj000000Eojt0002sB)?U_0001h;?{7$000000Eojt0002MImK1L0002sHUn|M000000N9y8"
    "0001hN6S^f0001Bqc3s5000000Eojt0000WPtjGt0001>CtPvB00000005dm0000$QrA_$00000#E5ag000000Eojt0002MP~BC)0001>ceQcA000000Eojt"
    "0001BOXF3*0000WL*jA30000006>~R0001BLg`h&0001hCJ%DJ000000Eojt0001BHt$uy0000$Av|)x000000Eojt0002MC-qgp0001>Gih?b000000Eojt"
    "0001h7W-Ae0001>U6yjd000000F;_Q000000|8dR0002Mp2Kp$000000Eojt0001>?FLrB0001B_waJS000000Eojt0002M)eKg^0001BXC8CF0000007RHT"
    "0000Wyb@Nx0002M?n`sP000000Eojt0002spcq!b0000Wjd^px000000Eojt0002MgC17E0002sLZ@@U000000Eojt0002sWF%I=0001>57l$P000000Eojt"
    "0001>L@HLm0001h^8$3h000000HBya0001BA~06K0001h?l5$~000000Eojt0001h{xw#>0000$0AF;#000000BDv#0000W**jLi0000WDUEc%000000Eojt"
    "0000$vO!kB0001>XuEX4000000Eojt0002MiAPqz00000zv*<q000000Eojt0001>UrkoP0000$EEsjb000000Eojt0002MGgDT;0000$uts&j000000KAq!"
    "0000023c0X00000OLcX?000000L+#^0000$)?QY?0000W{G)Zi000000Eojt0001>q-0jW00000#L{)Z000000Eojt0000WacWk;0000$q5*cm000000Eojt"
    "00000JaAUP0002smN0g}000000Eojt0002M1$I`z00000qF;8v000000Eojt0000$&U;qC0001h!;W^q000000Eojt0002sm4Q~k0001B`@MF+000000Eojt"
    "0002MU58e{0001>N$qyP000000Eojt0001BC5=|V0002stsHm2000000Eojt0001B?vhr(0000$C`@<2000000Eojt00000xR_SJ0002sxO;cN000000Eojt"
    "0002Mfu2^t0001hU#oY(000000Eojt0001BO{7-90000W9NTxm000000Eojt0000W8LC#m0001>?h1Io000000Eojt0002s<gZr10000W*E)E>000000Eojt"
    "0001>u(eje0001>)oOUa000000Eojt0002MeY;k`0002M>6&=J000000Eojt0000WOTkva0001>6U%tO000000Eojt0001>8OK(@0001BQ~P+p00000005Xk"
    "0001B>C9HZ0001>s4IEE000000Eojt00000yV6#`0001h6I^+~000000Eojt0002sj@VYf00000l#F@6000000Eojt0002sWZqW50002sDZP2X000000Eojt"
    "0001BJ>*uv0000W)$Mt}000000Eojt0001>80uEQ0000$mL7V*000000Eojt0000$_U~4}0001>Y)^W?000000Cbo@0002s)%8}u0000$Sb%!K000000Eojt"
    "0001BxBFJW0002sSh0G*000000Eojt0001>ngLh90002sZsK~t000000Hl~e0002Me+O5<0002snG<`!000000Eojt0001hW(`-s0000$+C+Q6000000Eojt"
    "0001BPZL+b0001hFLryt000000Eojt0002sIT=^L0002snWuZe000000Eojt0002sCLdS80001h8QFWl000000Eojt0001B7A04}0002stqOd=000000Eojt"
    "0001>2rE~>00000S3G>c000000Eojt0002s{V-R+0000W6>faM0000003ew_0000$_BL0*00000=%0MR0000005F+A0000$@jO?+0001h&(VCq000000Eojt"
    "0000W??P9=0001>%L9GD000000Eojt0001h??_j`0001B+%<i`000000Eojt0000$@=jO4000000%(1}000000Eojt0001>_EcBE0000$JeqyL000000Eojt"
    "0002M{90GQ0002sip+h$000000FaqL0001B1z}gf0002s?f!kh000000Eojt0002M4`)}v0001BWifuh000000Eojt0001>9Bo&?0001B@ne3#000000Eojt"
    "00000EOS@D0001hk(PeI000000Eojt00000K6zKb0002sMah1^000000Eojt0000WQ-4>$0001>4*Y(=000000PLAS0001hYlT<90001h>n?x5000000Q{Li"
    "0002shKpCg0001>++lyf000000Eojt0000WrIA;_0002s;go;C000000Eojt0000$#+O&X00000`^SI4000000Eojt0002M>77@=0000$DEoiF000000Eojt"
    "0001>52RPX0002MG#7xt000000Eojt0001BH>y{_0001B-!Oo{000000Eojt0002sV6a!f0002MlS+WV000000Eojt0001>jJ8+60000WQe=R@000000Cbu_"
    "0000$y1ZAw0001h8h(Jl000000Eojt0001B>cUsR0001>?3aMR000000Eojt0002M9LZO}0001>$gzOH000000Eojt0002sQO{Sv0001huFHVH000000Eojt"
    "00000iq%)Z00000p5=hR000000Eojt0000W#oJfF0001Bm;Zpk000000Eojt000001L9Y}0001hni+w>000000N|QH0002ML+Dq)0001hr!#@T000000Eojt"
    "0000WhwfLv00000zD<F^000000Eojt0002s%=A~l0001B-e-Zp000000Eojt0001>6#G}e0000$2!esY000000Eojt0000$UIAFZ00000Je+~R000000Eojt"
    "0002MsRvlV0000$dA5PT000000Eojt00000_YGLU0000Wz|euf000000Eojt0000$Mif}U0001>5bJ@!000000Eojt00000m>O8X0001hY6XJ8000000BD;)"
    "00000>>yac00000&L4un000000Eojt0001hLMK?j0000WJ34~E000000Eojt0002Mn=M$t0001>u~dS<000000Eojt0002M_cK_)0001>F>ivv000000Eojt"
    "0002sRytU~00000yNQCp000000Eojt0000$xIkFI0002sO{9Xr000000Eojt0000$97kBd0000W>b`=&000000Eojt0001hf=yV!0002MkJ*C2000000Eojt"
    "0001B>Qh+20002sKJ<dX000000Eojt00000R9aZT0000$_YQ-=000000Eojt0002MzhGFv0002sxG96c000000Eojt0000$EoWH300000ghYeD000000Eojt"
    "0002MoNZXZ0000$SYLy{000000Eojt0001>4s=++0002MHF|@<000000Eojt00000gL+uN0001h9F>E>000000Eojt0002s`G8o!0001h46uX1000000Eojt"
    "0001hbB9>K0001>1<QlL000000ECu60001h?~Pc%0002s2j+vo000000Eojt0001hZj@NS0001B69I(4000000Eojt0002M@0wV^00000C>(^q000000Eojt"
    "00000bD>zk0002MMK^@N000000Eojt0000$_@`LF0001hY*K{4000000Eojt00000f38@-0001hoNa`_000000Eojt0001B2enwh0001B)rf?^000000Eojt"
    "0001>k-J#H0001h7o>#1000000Eojt0001B9l}_^0001BV!wpI000000PL1P00000tH@Zu0001Bw%UZi000000Eojt0000WJI`3a0001B6ZV9_00000005Uj"
    "0001>&DB`I0000Wc@c%c0000001THv0001hV%%8300000=q!c6000000Eojt0001B`r=r?0001>Ur2?(000000Eojt0000WmFZZ(0000$;A4fr000000Eojt"
    "0001hGVoZy0001>X@7;l000000Eojt0000W)Am@v0002M`<jKo000000Eojt0000$c>P$w0000$mbQh!000000Eojt0000WBm`N&0001>IMRi{000000Eojt"
    "0001B*$P>}00000<L!mO000000AQCu0000Wn-N*S0000$l?jHx000000Eojt00000ZWvj>0002sP9=uG000000Eojt0002MRUcWv0001h4?%{&000000Eojt"
    "0000$SSDG(0000W*Ib6c000000Eojt0002Mcr97L0000$rFe$G000000Eojt0001hx-?n90002Md6b60000000Eojt0001hAUs*X0001hRIrA?000000Eojt"
    "0001>tV3DA0001BHOz*<000000Eojt0001hUrSlQ0002M8|a3>000000Eojt0001>H&a=_0002M2Ly+}000000Eojt0000WG+SA~0000W`5%YC000000Eojt"
    "0000$RAX7d0002M?>mRU000000Eojt0001hmTOtS0001h>sN=s000000Eojt0001>`g2*p00000>~n{}000000Eojt00000fP7iN0000$@sEeV000000Eojt"
    "0002sCWTqR00000`>Th*000000Eojt0001>?~Pf&0000$3CM@Q000000N|HE00000+m>0t0001>8{~(;000000Eojt0000$>7QA^00000G60Cc000000Eojt"
    "0002s8L3&o0001hOdN>7000000Eojt00000Y_eIv0002sYB-3%000000Eojt0001h-n&`A0002sj8urg000000Q{Fg0001>ZpB%^0002svT%sN000000Eojt"
    "0001B8qZn40000W+>D68000000Eojt0002s-q=~d0002M2&ss`000000Eojt0002sxZ+vB0002sIK_y-000000Eojt0001BrtDe30000$Y~hH&000000Eojt"
    "0002sruJFD0001>qWy@#0000002r7+0001Bxd2+g0000W+!=|$000000Eojt00000-U?d40001h7&eK(000000Eojt0001h6%|^*0001>S5k?<000000Eojt"
    "0000WVINw+0001>nQn=|000000Eojt00000!zo(800000-inF9000000Eojt0000WJTzLs0000$C8vqN0000004$h50001h&OTbe0001>Zo`Sd000000Eojt"
    "0000Wfk|4x0001hy5EVw0000005q6D0001hT2xxV0000$2>gk_000000Eojt0000$U|(9m0000WSQv`H00000063UH0002MlxkYQ00000sx*qg0000006dsL"
    "0000W_;p&q0002M{ZNX)000000Eojt0001>h=E$b0001BQ*DaC000000Eojt00000LycO%0001hs)>rg000000Eojt0001BBAHsi0001B1Ez|=0000007RHT"
    "0001hAEjEr00000Uc!pN0000007#fX0001>FR@y{0001hy55Sw000000Eojt0001hOubsb0001>82gIA000000Eojt0000$a>-i20001hco&Pn0000008p4f"
    "0002MoYh*u00000+B1v5000000Eojt0001B%i&tU0000$Jy46l000000Eojt0001h0Pb4A0002sp=^u5000000Eojt0001hLHb(2000002#Jfp000000Eojt"
    "0001Bl?GeD0001hZ>5XC000000Eojt0001>1`}Jr00000*ujgy000000Eojt00000n;%=i0002MLEekN000000Eojt0001hRxDe<0002ston<<000000Eojt"
    "0001>KRH{#0001>7#EDd000000Eojt0000WTt-{K0001Bg)@x6000000Eojt0000$wNqQb0001h@=uJw000000Eojt0001hT47tj0001hUu=xP000000Eojt"
    "0001BS8rRu0000W#E6W*000000PL4Q0001hw0&E^0000$4yBC1000000Eojt0002McZ^%W0001hD#47v000000DzZ30000$t(#lG0001B3f+vr000000C12%"
    "0002sW~*Dk0001hr1*@$000000GN$H0001By1QGz0001>@Dq)|0000006>XA0001hw#-|=0002s=q`=G000000MLj)0001BYTjGG0001>h)IpW000000LX<v"
    "0000$;qY6)0001>%wvtf0000001$#e0000WDFj@=0002Mv3-re000000Jwob0001hR2W>q0002sG?$IQ0000006>300001>cP(7N0001BRIrV}0000008o8E"
    "0001ho<m%~0000$4#|zc000000I+#L0000$&RSf+0001BVd0Iy000000HAk30001>40Bw-0001>PWg?%000000MK+m0002MdyHJb0001h+7OPw000000HAR|"
    "00000Pp4eK0001>3@DDk000000N`&x0001B!oys^0000$={}CY000000HAC@0001>6yjXK0002saaE4M0000006=O$0001Bdjeg+0001BsB4bE00000003w}"
    "0000$EhSyR0000Wk%5lD000000AOZ700000R7zdI0001>FPDzM0000003c;R0001B2y$J(0001>h^&sl000000AOT500000W}IEX0001>p23d5000000Dxma"
    "0000$fXQ9J0000$b=Ho+000000Kj5E0002MT=`wV0002s4(pD<0000006=0u0000$0xVv@0001BZU2tI000000Kj2D0000$c41z?0002skrR)=0000006<|t"
    "0000$wwYeQ00000ekYH>000000Ki~C0001>!_;2D00000GdqvK000000AOH10001>suEwo00000uTGD^00000003Y>0000WcT-=$00000?_rO?0000006<?r"
    "00000GnZe$0000$@^p{D0000006<<q0002s>D^zz0002sw1<zt000000DxUU0002M(i3370000$G@6gV000000Ki;80000$HacLy0002sY^;yK0000006<(o"
    "0002sAYx#^00000T)vOM000000DxOS0001hl8a!#0001B{m+lU00000003J+0001hiMC+C0002MOyQ5f0000006<zm0001B0pno60000WNAi!r000000Ki#5"
    "0002M^bldd0001BbODgS0000006<wl0002MTtH#K0001BmkE%-00000003D)0002sF>ztQ0001BixH5(000000DxFP0001BYolSn0002sO&O5D0000006<tk"
    "0000$0M}u_0002M-5`*_00000003A(0002M<_luL0002sIVh08000000Kiv30002s6hdOa0000$VK0!t0000006<qj0001>etBZR0000$QZ|sl0000006<qj"
    "0000$7_wr(0000$3q6p)000000037&0001h*6U)x0002si$jpW000000Kis20000WttMl@0000W(@BuP000000Dx9N00000kz!-O0000W;7^di0000006<ni"
    "0001BbDv|t0002su~m@3000000034%00000Mcre-0000WM_Z7<000000Kip10002s_#0%v0000WpJ0%{000000Dx6M0002sf?H(30001hxn+>R000000Dx6M"
    "0001>)th9%0001Bm1>Z{0000006<kh0002M<lJPy0000$GH;N;000000031$0001Bq8??y0001hk8_Z~000000Kim00000W0bymp00000tap&W000000Dx3L"
    "00000@}p(I0002MhkKB~0000006<hg0000WY3F6Y0002sA%Bp-0000006<hg0002MSSx100002sd4iC@00000002}#0001>vTtU<0001>kA;xH00000002}#"
    "00000YP4p+0002MVu+By000000Kii~0001ha`|S!0001B^NNtb000000Kii~0001By*_8a0002sK#h>V000000Kii~0002sL4;?(0000$OplPj000000Kii~"
    "0001B^~q<z0002s6_Jp@000000Kii~0001>%@1h60002sosy8i000000Kii~0000Wzg1|!0001B<&%)W00000002}#0001hz?*2m0002M?39qe000000Dx3L"
    "0002M$Khze00000wUv;-000000Kim00002M%qMBU0002sK9-Qc0000006<kh0002M$8BlA0002si<XeU000000Kip10001huCr;t0002MoR*Nl000000Dx9N"
    "0001>d--X=0000WbC!_6000000DxCO0000$I6rE@0001B6PA#`000000DxFP0000W?}Tc=00000f|ZcK000000DxIQ00000pUP^$0001>!IY4|000000DxLR"
    "0001BNf2wm0002s)RU0F000000Ki*70002s@l|WU00000zmkx^000000Ki;80001hmYZw90001>f{~EG000000DxUU0001BGU02$0001>Adry20000006<<q"
    "0001B!X|9M0000$nvRgb000000DxXV0000$L2PWm0000$@Qjea000000Ki^A0001>rm<|m0001>A&Zc}000000Ki^A0001>==N;D0000WEs2o80000006<<q"
    "0000$`8#dE0001B5r>e#0000006<+p0001B%Ybdb0000$%!H7@000000DxRT0000$PQ`7&0002sTY`|l000000Ki*70000WZ3%9`0000$zJHLw00000003J+"
    "0001>7ENxz0002M@qCcL000000DxLR0001BI*@L_0001B^?8uM00000003G*0001h$<l7X0002s$aavx0000006<wl0000WwG?l_0002MYjcpm0000006<wl"
    "0000$@KtZX0001>-EWY<0000006<wl0002sZI*Aq0002M9&C`n0000006<wl0002MFV=6s0002sFldm#000000Ki#50001ho(gcl000007G#jX0000006<zm"
    "0002M$|-Qb0001B&tH(h00000003J+0001BnMiQJ0002sU0aaA00000003M-0002M3TJS@0001>!d8&L0000006<+p0001>C53Rn0000W08)^^000000DxXV"
    "0001>@1k(P0002M7)_AC00000003Y>0000$ZoqKB0001h4oHx|000000N`Li0000$r`>SC0001h<3f<Y000000H9$&0002MpZajX00000nmv%f000000AON3"
    "0000$TNiP_0001hGdPgI00000003h^0002s+cR;%0002suriRq000000Dxma0002sAy9F^0001B6D^Ry000000N`Ul0001hENpSW0002MS|^ae0000003c*Q"
    "0001B`G;}90001>har%_0000006=6w0001Bgrjl50001BnH!M60000006=6w0002s$-Z&G0000WkQI=?0000006=6w0002s#o2Md0001BYY&jX0000006=6w"
    "0000WbM$e*0002sDhZIl0000006=6w0001>%?@(F0002M&H|9Y0000006=6w0000$(<pMl0001ht@w|?0000006=6w0002sf<kh@0001BhwG2P000000Dxpb"
    "0000$+gozL0001BE8dU5000000KjBG0000$+;nol0001>n$VBH0000006=9x0000WhmCT;0002s*20g#000000H9?+0002M+@*5B0002M;k1vy0000006=Cy"
    "0002s-o0|b0000Wyrz%90000003c^T0002skkxX)0001>XPA$`000000KjKJ0000$_Uv-N0001B=82EM000000AOf9000005(IO=0000$I(v`6000000N`jq"
    "0001><r;Iq0001BWNeSX000000HA3=0001hb1`$k0000$W?hfK0000003d2W0001>yhn4u0001>Ku(Xq000000KjTM0002Mz+H2|0000$^gfTk00000003)1"
    "0002Mesgoc0001hfiI81000000N`ps0001>^@wx80002M=pT>3000000Dx;i0000$Bc5}>00000DiDvr000000AOoC0001>1+#O&0001hMF5Y$000000AOoC"
    "0001>na6X$0000WJMfOc00000003)10001h-P?1(0001h3*e5x00000003)10001h&+l`<0001Bw$F~h000000AOoC0000$Z3A?`00000Jim^>0000006=R%"
    "0001>wH9>10002MoUV?*000000HA9?0002srYUs50001B*`1ES000000AOoC0000$K|FN80001h@{Eqa000000KjWN0001>gidt80000$=zEU9000000AOrD"
    "0002MabR@700000ylRfX0000003dBZ0002s337D60001BZds1N0000003dBZ00000PJ?v70000W|3{9%000000N`yv0001>Ka_O90002Ma5j#>00000003`5"
    "0002M;G}fG0000W!z7Nt000000Dx~m0000WGqrTU0001B_Y#i4000000AO%H0000WImL9q0002s4FQh8000000HAO{00000^we~~0002M2k(u*0000005EYt"
    "0002MW#n|g0001h<=l<H000000Dy5o0001>kMnfE0000WsLGAN0000001$FO0000Wa{_h10002MPq>Z10000003dQe0002M5D|620000$-KCAd000000AO-J"
    "0002MX&-gK0002MP?L?o000000MK$k00000el2yt0002ss(_8a000000N`>!0002sO*?hK00000>}`#}000000Pu1^0000W*+_N30002M6kCnJ000000AO=K"
    "0002M9aeR~0002MBuI_G0000001$IP0002M8e?_90002s95s!=0000003dTf0001h&Tn<V0000W{UMFO0000005Eev0001BH+*%#0000$#SV?Y000000KjrU"
    "0001BQ;2oI0002sb^DCK000000KjrU0000$A(VB%0000W59o}*000000B~|Z0000WpPzNW0000$k<^U9000000Dy8p0001h%&K+30001h{J@OB000000FZJ("
    "0001hr?qv!0000WQLT)?000000HAU}0000$Fu--d0002MkC}|X000000I+gE0002MX3BNI0002sxP^?s000000FZM)0001hOVoA10002M%yEps0000005Eev"
    "0001>-`#b<0001>%Uz7X0000003dWg0001hBj<I%0000$wn>b@0000005Ehw0000$8Sr($0001>j5UnF000000KjxW0001h!}xW;0001hPa%xJ00000004GC"
    "0001hBms870001>{|t=40000001$RS0000$J_&Zf0002socN2t000000C0Ce0002M6A^a60000WDCCR40000008n^90000$s2Fy@0001Bq0fuJ0000006=*_"
    "00000{vUS00000W3A>BH0000008o2C0001h7bkYW0001>U#5$|000000HAz80001B_bqn70001hrICxk000000Kj}e0002MpEP#B0001h+kA__0000006=~~"
    "0001h6FYXm0001h0ceZB000000APMV00000R6%yY0001h7gLMC0000000@9U0000WV@Gzt0000$9X^Y|0000001$yd0001>LQQtS0000W6Df<p0000008oNJ"
    "0001h^HO%e0001B`V))50000005E|-0001>c35`600000(*BCT000000O*520002M&0TiD0000Wo9T+c000000Dyu(0000$`C@j!0001BRn&^W0000000@LY"
    "0001h`e$~)0000$0l$jC0000002qWo0001B(rk9X0001ho~eq!0000002qcq0001BfpK=g0001hEtHDD0000006>I50002s2zGYB0001hu6~NZ000000N{u~"
    "00000YI=6S0002sA!&-h0000008oZN0001>rG9q600000hf|8c0000000@ad0002syn=SX0001h-aU%I0000002qlt0000$v4(cQ0001hC@6}+0000002qxx"
    "0001hgo<{+0000$WfF?O0000005FX}0001BH;#6|0001hl>CXn0000007#8M0002M%aL}#0001hw&sbz000000Q`|a0001BLzQ;G0001B%g~9y0000000@#m"
    "00000pqO^R0001B)4GYk0000002q=$0001>;+uBB0000$&ZLRJ000000H}>X0002s4WD+v0001hypD;$000000Kl0*0002sBBFM{0002Mop_1B000000Mwa4"
    "0000WBc*o00001ha$<?V000000PLAS0002M5U6&*0002sI!uYc00000005dm0001>?5cLa0001>_BDyX0000002G=)0001>xUF`;0000$rXPvG0000004SP3"
    "0000$cCdE90000$NePL-000000Eojt0001>CbM?H0000W;PQyT000000Eojt0000W%e8jE0002sYTSsx000000Eojt0002sWVm*~0001B>BWe^000000Eojt"
    "0002M^tyJy0002MTd#<~000000FatM0002Md%bqR0000$!kCD_000000Eojt00000`@eR;0001>9D<0z000000Eojt0001hbHaAO0001hYio$W000000Eojt"
    "0001><-~Tt0000$u2hJ?000000Eojt00000Qpa|{0001><vobN000000Eojt0001>xyg3G0002s5GRPh000000Eojt0001h9L#pW0001hFc65q000000Eojt"
    "0001Bd(L*i0001hMEQrn000000Eojt0000$*3fpq0002MOyY;Y000000Eojt0001BEz@?u0001BNy~@8000000Eojt0000$fz@`v0001hI<$wt000000Eojt"
    "0001>(bsms0000WAf1Q6000000Eojt0001>9@=)m0001B`h|zU000000Eojt0000$Xxw(d0000W$!>?h000000Eojt0001>uHJUQ0001hj8})i000000Eojt"
    "0000$@!)pA0000$L_mkY000000Eojt0001hFynT>0002s@F<7C000000Eojt0001hYvp#p0002skr9W$000000PL1P0001>qUUzN0001hC;EoK000000Eojt"
    "0001B)#-M?0002svEqin000000Eojt0002M1MGId0002sFw2I(00000034S<0000WE$()}0002Mq_c*=000000Eojt0001>QSf%a0000$4V;F+000000Eojt"
    "0001>aPoG*0001hYJ`Ts000000Eojt0002siS%~B0001hyljTR0000009=<q0000$o%VLX0001B1678=000000Eojt0001Bs`z%m00000KRt%P000000Eojt"
    "0000WvHEtv0001hZzYDn000000FakJ0002su>5ww0002MlnsW!000000HBvZ0001>ss47r0002suJwh$000000Eojt0001>ngDme0002sz1@Yt000000Eojt"
    "0002Mg93NJ0002s!NrBZ000000MM5}0000WWdwJ?0000WyRC)5000000N|HE00000J_dKd0000Wsg;Gm000000Eojt0001B4hVO^0000WjD3Z`00000005Xk"
    "0002M)d_dN0000WWMzfG0000001%i!0001BlM8pi0000WF-?WQ000000Eojt0001BM-6wt0000W^E8FP000000Eojt0000W@eX&u0001BsvL#D0000006>^P"
    "0001>kPvsk0002sRs@B>0000008p4f0000$B@%bQ0001h`0Iqg000000Eojt0001>t`m2_00000kJE&}000000Eojt0002sEERXa0002M8oPwR000000Dzc4"
    "0001hpB8t(0002Mo1%ok000000FanK0000$1{in10000$5{iVt000000Eojt0002sU>SG700000esP4r000000I--q0000WuNrs20000W-&lme000000Kk|)"
    "0002s@Edo)0001hH9&;G000000Eojt0001BCLMRc0000$fhUB(000000Eojt0002sP9Ar_0000$!VZMM000000Eojt0001>Y9DvN0002M`1OOp00000005al"
    "0000$c_4Sd0002MB;A9+000000Eojt0002sdLehf0000$M#O`_000000Eojt00000Zz6ZV0000WU8{q@000000Eojt0002sRU>!60001>X_AA$000000Eojt"
    "0001BFC=%s0002MYk7mf000000Eojt0001B`y_Y30001BV_<{8000000Eojt0001>xg~eN0002MP)CEn000000Eojt0002sX(o5T0002MGcJR`000000Eojt"
    "0000W3@3NM0001B3>AaG000000F0SH0001hpeJ|00002s+WUgQ000000Eojt0000$CMb8n00000pW=eR000000Eojt0000$ohWy}00000S;>OH000000Jxbz"
    "0002M1u1vH0002M2(W^{000000LYm@0001>UMY9L0001hu9kwp000000Eojt0002Mrzv;900000N_~RB000000Eojt0001h;wg8)0001h+hc;j000000Qi|e"
    "0000$4JvoQ0002sVoHL*000000Eojt00000Cn|Tq0000$;4p%~000000Eojt0000WFe-Pz0002MQx}53000000Eojt0002sC@Ocr0001Bz5Ri}0000005qCF"
    "0001h5Gr@T0002s9OQw(000000Eojt0001h<|%i;0002Ma>{|g000000Eojt00000t0{ND0002MzOaG7000000Eojt0002MT`70K0002M0GENl000000C1W>"
    "0000${wR0A0000$Ieme^000000Dzi60002sj3{@&00000Xk&rE000000Eojt0002M2q<^J0000WjY)yP000000GygY0001hawm7d0001hs4s!Q000000Eojt"
    "0001B$tHKe0000Wx)p)I000000KA$&0002M4JLQM0001B!ux>0000000Eojt0001hKP7j-0000W!Qp_w000000N9#90001BUL<$G0000$w#R_L000000Eojt"
    "0001BY9n{R00000qOE|y000000Qj0f0001>V<LCJ0000$gp+{40000000^5v0001BN+EZ^0000WU3q}O000000Eojt0000$9w2wX0001>EMI`Y000000Eojt"
    "0001>-yV0s0002M@<f2Z0000005qFG0001BjvaTv0001>uPT7R0000006?2S0002sDjavf0000WV-SGA000000Eojt0001BwHkN80000W8|;6;000000AQOy"
    "0001>Eg5&f0000$UAKS0000000BoB;0001hloxlv0000Wje>u`000000Eojt0002s=@oat0001hs!e~u000000Eojt0002sEfjaa0001hwHSZE000000GOLV"
    "0001>VG?)10001huH$~d000000Eojt0000Wg%EeZ0000Wm9Bok000000Eojt0001>m=1Tq0001hX?cFY000000Eojt0002MoD6rs0002MD@1<4000000MM8~"
    "00000k_vag00000+zx)g000000Eojt0001Bc?fsF0002sdD(ry000000Eojt0001BQwDdy0002s1*Cnz000000Eojt0001hAOv^70001>e{Oxi000000Eojt"
    "0001>;Q@ER0001h=Qw@9000000Eojt0001Bm;ZLa0000$Jp+Bf000000Eojt0001>Lj88Y0002MfX#fs000000Eojt0000W<ob5N0001BvX^|o000000Eojt"
    "0000$eE4?200000(qVkS000000Eojt0002M3-)%v0001h-z$8;000000Eojt0002Mlk;}K00000+x2_E000000Eojt0001>6!CVz0001>#J_vM000000Eojt"
    "0002sj_!8A0000$oQZqD000000Eojt000001MGId0002sVN-j+000000Eojt0002sap`uz0001B6&-uP000000Eojt0001h+vaw_0000Wx8{1l000000L+#^"
    "0001>K;(A700000N3eRp000000Eojt0001Bq2YGG0000W#d&(b000000Eojt0001B{@!-L0000$F++O5000000Eojt0002MSKM~M0001hiwt_e000000Eojt"
    "0002Mt=V?K00000)YW;w000000Eojt0001>0M~ZF0001>3ZHqv0000001%fz0000$P}Fw70002MFKBtd000000Eojt0000WozZr{0000WLos>4000000Eojt"
    "0000W=gxM(0002sMEZEZ000000Eojt0001BF3fho0000WHN$wo000000Eojt00000bIEqV0000$6pVPl000000Eojt0001BwZ?Y90001h<5YOS000000Eojt"
    "0000$^}}|+0001>q8)g^0000009=<q0002sGQoDh0002sQRa8R000000Eojt0001BZoYQF0002M@ve8k000000Eojt0001hr@MB*00000gLikp000000Eojt"
    "0002s-ne$a0000$20(Yf000000Eojt0000W6t;H20002MdkA;H000000Eojt0002MN3(Xo0000W<j{7&000000Eojt0001>d9ZfC0001>KbdyG000000Eojt"
    "0002MsjYUv0002sj$wAd000000Eojt0000W*s6BG0002s(J6Mo000000Eojt0000W1*mqw0000$3G#Ko000000Eojt0001>Fr{|D0001BHM(`c000000Eojt"
    "0000WTcUQr0001hR)TfF000000Eojt0002Mgr9c60002sY)W;&000000Eojt0002stekeh0000$c@lNN000000Eojt0002M)R}g`0001>d)jos000000MwU2"
    "0001>{FZjW0001hbfR>?000000Eojt0001>B9wN(0002MV`_B3000000N|HE0002sNRf8H00000N-}i7000000Eojt0001>Z;p1r0001>C;D^1000000Eojt"
    "0001Bmy3450002M{K0d;000000Eojt00000zle6g0000W$%%8o000000Q8qY00000=!JH`0001>jZt&J000000Eojt000005`%WY0002MN*Qy&000000Eojt"
    "0001BJ%Dz=000000O4}L000000Eojt0001>YJGOV0000$uBdXr0000000@{s00000ntFD?0000$Rd8~^000000Eojt0002M%XfCb0000$^*3_B000000Eojt"
    "0000$0Cjf20002Mj{kAM000000Eojt0002MH*$8s0001>A;)pR000000Eojt0001>a&LCP0000$u8wiQ000000Eojt0000$v21q00001>G*)rI000000Eojt"
    "0002M^J#X#0000WwH<N50000003?_|00000J7;#l0002sFXeE+000000Eojt0001>hGcfY0000WrL1tk000000Eojt0002M*I{<R0002s6m)RF000000Eojt"
    "0000$E?;)Q0001Be>-r%000000Eojt0000$id=TU0001h<N|QO000000Eojt0000W>{)ie0002MiIi`^00000063UH0001hR9AMu0001>KqGI!0000006dsL"
    "0000W!&G*_00000?y+vb000000Eojt0002sI8t`N0000$l0a_2000000Eojt0002swNG}y0002MEzfPh000000Eojt0002sIZbxJ0000Wz+G*?0000007RHT"
    "0002s#Y%R-0001>O6+XF0000007#fX0000$Sx9!k0000W&3bIW0000007#fX0000$^hI{S0001>M+t1e000000Eojt0001hmqT{I00000x|eIf000000Eojt"
    "0001>LP2)G0000$B_?aY000000Eojt0001h_C9vN0000$iL`3K00000092Sj0001hvOIRc0000$=0a+~00000092Sj0001hb~<*z0001hJke>u0000009cqn"
    "0001>LO6E70002Mi(hHL000000Eojt0001h7dCdl0002M)a_`%0000009=?r0001h^fPwA0002M7kp^I000000Eojt0001B*)ev&0001>R0?Ro0000009=?r"
    "0000$#xHij0002Mi<oD?000000Eojt0001Bx-E9V0000$y(VYC000000AQFv0001hwkvkP0000$>9l6R000000A!dz0002MxhZzQ000005JP6b000000A!dz"
    "0000$#3y#Z0001>G0|nf000000Eojt0002M)g^Yo0001BPG4of000000Eojt00000?jv@<0000$X6<Ca000000Eojt0001>3?g>G0000$d3$8Q000000Eojt"
    "0001>F(7up0000WhzVrC000000Eojt0000WTpo770001>l9yw^000000Eojt0001>j2w2r0002sm?dMt000000Eojt0001h!WwqK00000o3moT000000Eojt"
    "0000W{TOz@0001BnL%Q}000000Eojt0001hJQsGr00000l+R(n000000Eojt0000$fE9MY0002sid|vA000000Eojt0000W$P;$J0000$e(PYs000000Cbo@"
    "0001B6B2g70000WZh2t9000000Eojt0002sVGwq}0002MS_fdj000000Eojt0000WvkrE^0001>h$>&e000000Eojt0002s1`T$=0001hRYqRG000000Eojt"
    "0001>Tnl!<0002M9AaI-000000C<={0000Wv<Y^=00000-hEua000000Eojt0002s3<!3>00000nweX`000000Eojt0002MW(Ib^0002MP_<gX000000Eojt"
    "0002s!31`|0001h0?}E(000000Eojt0000W9Rqg20001Buk2XB000000Eojt0001Bc>#970001hR|r_Z000000Eojt0000$*8g?D0000W`y*Gt000000Eojt"
    "0000WGyZkJ0001hnm<;+000000Eojt0000Wko<MP0002MHCt7{000000Dzc40002M?D}=U0002s%yv}3000000Eojt0000$NcnZZ00000V3Je7000000Eojt"
    "00000qxW^d0001h@2yh6000000Eojt0000${PlIf0001Bd&yD2000000Eojt0002MQ}lJf000001m#e`000000Eojt0001>sPc8d0001hi2zW*000000Eojt"
    "0002s`tWtY0000W7&uPA000000Eojt0000$N$+*Q0001>7jI0!000000Eojt0001>lkIiD0002M5vEGO000000Eojt0001>+3R({0001B2i-`(000000Eojt"
    "0000$8tQex0001h_!CFK000000Eojt0000WRp@oV0000$<w`}s000000Eojt00000isp5|0001h&VfU~000000Eojt0001>xa4)f0000Ww75aQ000000Eojt"
    "0001h;No?_0000$E#E)D000000Eojt0002M0pWGP0000$>`Xks000000Eojt000009N%@o0002MrMfu4000000Eojt0002MFWq&(0002sStvEY000000En1C"
    "0001>Jll1^0002s2$nIx000000Eojt0001hL)mq}00000W(O?5000000En1C0002ML)Ue{0002Mu;L`Z000000Eojt0001hKGk)=0002M<9i#x000000Eojt"
    "0002MGt+gz0002sreq4h000000Eojt00000B++%i0001>@O05X000000En1C0000W56^YL0001>2Eq<N000000Eojt0000$_RMv_00000%Bmee000000Eojt"
    "0000$*~)dm0002MQTio7000000Eojt0002MxX5+D0001hM;|Rf000000Eojt0002Ml*V<y0001BBO)?D000000Eojt00000Zp3xK0001hr?fRd000000Eojt"
    "0000WMZ$H!0001hFh@B+000000Eojt0002s8NhYG0001BzS%rL000000Eojt0000W?Y(ut0001hC^kSq000000Eojt0001Bzq@t70001hbpJs>000000Eojt"
    "0001hkhyih0002s!pK8F000000Eojt0002MVYhX_0000W6p}?i000000Eojt0001>GqrWV00000Xj?}>000000Eojt0002s2D5d*0002sz9dON000000Eojt"
    "0001>+pu-O0001>8170y000000Eojt0002MvaWT&0000$b+=4F000000Eojt0000$jI4FQ0000$)qqYw000000Eojt0001BX{vR=0000$93@ad000000Eojt"
    "0001BN~m?f0001B&k0dL000000Eojt0001BFQ#?C00000gziy5000000Eojt0002M7^HQ;0000$I@3}>000000Eojt0002s2BLMq0000$^0!k!000000Eojt"
    "0002s`JZ*b0001>tDICo000000Eojt0002M@tt+R0002MW`R{e000000Eojt0001h?wfVM0001BBW6}W000000Eojt0001h@R@bM0001>;YwFP000000FanK"
    "0000W_m_3R0000Wp)XiK000000Eojt0002M0+w~a00000ViZ|G000000Eojt0002M5|nko0000$B>7oD000000Eojt0001BCz5r*0002M=iORB000000Eojt"
    "00000K#+C70001>tioGB000000Eojt0001hT#j|X0001>a;IEC000000Eojt0001>e2jI#0002MIgDLE000000Eojt0000Wpo(?C0001h0dZbH000000Eojt"
    "0002M#)x&m00000%T-@M000000Eojt0000$@P>830001hN*!Q8000000FanK0000$9EEkj00000u#aFs000000Eojt00000OM`X50000076M^F000000Eojt"
    "0000We1Ubq0001hd~{(z000000Eojt0001BuYYyG0001B<K|&N000000Eojt0001><b8F(0001hN?Bq+000000FanK000009DH@a0000WvddyX000000Eojt"
    "0002sRC;y50000W8a`t{000000Eojt0001>k9c*!0000$g0N#i000000Eojt0000W%yxCa00000>mp=8000000Eojt0000$3UzhB0000WRFz~v000000Eojt"
    "00000N^^C<0002My$5AL000000Eojt0002sig9(o0001BCwgT-000000Eojt0001>%x`tT0002sknCkZ000000Eojt0001h4sLb80001h`d(&0000000Eojt"
    "0001>P;7O;0001BVbNwl0000006>^P0002sl4^Cp0000Wr9)>x000000Eojt0000W)o69V0001>f3;^o000000Eojt0002s7-x0B0000Wr6y=V000000Eojt"
    "0002sUS)N_0000$-Ii!T000000BDy$0002Mrek%$0002M0S0M6000000Eojt0002s@L_ep0000$^>b-J000000Bo5+0001>Jz#af0001BqTp#j000000Eojt"
    "0002Mj9zuX0000W_e*L(000000Eojt0001h-duIS00000=e24;000000Eojt00000Gh21Q0001>XC7-n000000Eojt0001>iCJ~P0002saD{6?000000Eojt"
    "0001><5zXS0000W{pxE#000000Eojt0000WK~{CZ0001h22X52000000Eojt0002sp;UFi0002Mi?M7#000000Eojt0002s1ygmv0002sj1p}?000000Eojt"
    "0001>Y*BT<0001h4RUQj000000Eojt0000$)lYT60000W6VPoy000000Eojt00000Ku&eQ0001hqbzPf000000IZlm0001ht4wvk0000W$A@k}000000Eojt"
    "0001B6-#x%0000WnBi_f000000Eojt0001hdP#M_0000WJ34Pb0000007RES0001h*GF~00000W!;x=5000000Q8YS0002sDMod`0001>I_Pge000000GyFP"
    "0002MY(#ay0002s+8%H~0000007#KQ0002snL>5I0002MD@kxb000000Q8VR0001Bt3Y+Y0001hAaig)000000MLd&00000oIZ8H0000$yPR-9000000O*21"
    "0001>Xgqbm00000?8R_E0000009b)Q000004mx$f0001hobGTy000000APMV0000$iZ^w@0001ByB2Xk0000001$dW0001>-86N;0000WKR$6l000000PuD|"
    "0000$1u}KO0001hA7ybs0000003dWg0001B`Yv_A0000WRf};z000000B~_Y0001BwkvhO0001B+OTmz000000AO!G0001hGAMPx0002ssnu~n0000006=U&"
    "0002MXC!sN0002szWH%L00000003z~0001>O(1o^0000$6diIv000000AOW600000;v03q0000Wr$2H)0000006=3v0001BAs2PP0001hbYXHp0000006<|t"
    "0000$1QK<?0002scY<<200000003S<0001Bg$#AT0002stD<s100000003M-0000$o(6Tm0000$3c_+g0000006<wl0001BNdR@g0000$jNo!W000000034%"
    "00000g!y#90000$D*kdn00000002}#0001BP4aZW00000-5PU100000002@z0000Wr|NXT00000oi=ko00000002-x0000$nc{T70001BX;E`P00000002%v"
    "0001>DB5(u0002sM{9FH000000DwzC0000WOVM<|00000I)rmT000000Dwq90001B@yB$)0001>Kb&(w000000Dwk70000$5WaN40000WRI_tH000000Dwh6"
    "0002sp|f<r0002sb;)x;000000Dwb40002Mpr~}f00000sNQox000000DwY30001h2c2}l0002M_wsW<000000DwV20002M(~xw)0002sVg__T000000DwS1"
    "0000$2!wRN0001h;23m300000002Kg0001hs&;h10001hY$|j>00000002Hf0000Wz-V;90001h0y}g-00000002Ee0000$R9bYv0001>nM!m(00000002Bd"
    "00000a7%Q+0000WEm?Fx00000002Bd0001B96WTu0000Wxn^`g000000028c0002MWGr;R0002sICFGB00000002Bd0001>PaJf>0000WtbTMr00000002Ee"
    "0000W;R|%Y0001B6^V2}00000002Ee0001>Bl~l}0000$a*}jF00000002Ee0000$AnJ3#0002M!J2eH00000002Hf0000W;n;J)0001>1EO?500000002Nh"
    "0001BZO3!K0002MJE(L(00000002Qi0000W%e8aB0001>aIJJe000000DwY30002M{-bli0000Wt+8}K000000Dwe50000W3X*fc000001hsTP000000Dwk7"
    "0001h?SFH?00000jJI?^00000002rr0001hrfqY;0002sX}NSj00000002-x0000WF<W!M0001B%DQww000000031$0002Mib!+70002s;=6P}00000003V="
    "0000WwKa3V0001h^SpFG0000006=a)0000$xFU1F0001BHNJE}000000Eojt0000Wnh$fp0002M#lLhw000000MKzj0001BUi@;v0001>tig0Z00000003q{"
    "000004e4^g0002s{KIrW0000006<$n0000$vek0H0001>&&G5>0000006<hg0001BRKjw=0002ME6Q|000000002%v0000W`>t}p0002s7|wJ+000000DwtA"
    "0001>w3~9k0001BrP6dj000000Dwk70002Mh=_8)0002s*4K1E000000Dwe50002sfpv1g0000Wv)yz+000000DwY300000uwrt+0000WMC5cp000000DwV2"
    "0001h9Z+(>0000$j_Y(l000000DwS100000*gbN<0001BmGg8!00000002Kg0001h@+)${0001BU;K1H00000002Hf0000$dKz-T00000>;-i|00000002Ee"
    "0001hehG5G0001hI}mk100000002Bd0002M4Eb@u0000WOB;1S00000002Bd0002sGwE@_0001B7bkT<000000028c0001h1=?}I0001Bnlp7k000000028c"
    "0001Bk;`$w0002M%RqHN000000028c0000$;k|Ld00000q)v4}000000028c00000`>=7q0000$8eMfj000000028c0001><)m@I0001BC2Vy-000000028c"
    "0001>shM%W0000WyL)v&000000028c0000$N{?~C0000$(2I3I00000002Bd0001>$%S#i00000Tby-300000002Ee00000CVg?h0000WQLS}A00000002Ee"
    "0002MTy=540002MsJ?YT00000002Hf0000$Yi@DC0000$SkQGq00000002Kg0000$NN91u0001>SK@U*00000002Nh0001h=wfle00000p!9V>000000DwS1"
    "0000WKVEUb0000WD+hK!000000DwV20002sIa+bR0001>^BHzP000000DwY30000W!&Y&?0002M^(=Nk000000Dwb40002s##3>?0000$EkJfa000000Dwe5"
    "0002sE>Us70001Bms55?000000Dwh60002s=}vLL0002sFlTl^000000Dwk70001h@=S5S00000_j-0f00000002lp0001hMoV$P0001h=8kqi00000002oq"
    "0001h=}B?G0002s{GxV1000000Dwq90000W<VbPA00000IJb5{00000002oq0001>K1gxE0000WmCJTO000000DwtA000000!VSd0001>5#V+}000000DwtA"
    "0002sGe~j30001BtMqn200000002rr0001h+emT100000UkP_W00000002rr0001B`AKoW0001>Djs)000000002rr0000Wk4tgD0001B3o~~>00000002rr"
    "00000piObW0002M{YQ5|000000DwtA0000WGf;8B0000W`&@TG000000DwtA0001hPg8Ng0001>{c(3d00000002rr0002M>Q-^U0002s28VY*000000DwwB"
    "0002M{919q000007@T)N000000DwwB0002sgI{sL0001BFR^z(00000002rr0001hd}MLJ0001hOU8FV00000002rr0000$-f3~b0001>ZQFN100000002rr"
)

_DENSE_RACING_PATH_B85 = (
    "WLU`zFlckY6lslmMyh>40000000000M%JWUEo^hZv8D5$!?=Aw0000000000x0aO^CvbDX=J6fbLC1YS00000000005)sP@9dvWRKD#6?!PR|00000000000"
    "wY+o-4|sFHO>ZElLF9cv0000000000{1C$I{Cji2p?fJz!}5JV0000000000Mup;k=6`d*%}rP(Mge|60000000000?T`hR%Y$>kT~oz6$`5`(0000000000"
    "O{-c2tcP>J;Q5G@Pab|i0000000000$4e2Khl_K-?5?{y)+~NN0000000000x9M0aUXOFY6X10<UORq30000000000b^Kt5FOze?+WOCo=ShA*0000000000"
    "VEb5u`<8RS*KFcQa#(&q0000000000*j;e&!J2cxo}<Na{$_qa0000000000Hk;O<f}V50xeSsIjCFoM0000000000*DAYaJ)(2Kx)oq58-#v90000000000"
    "8iY+0@}+aYGAu~StCD^|0000000000TOPf8ps91fvb|#WJ)wR;0000000000^l&+ENUd|g#y7~h(XW0$0000000000Mf35%>9BLa{|Av7XTE+w0000000000"
    "s~i?XgtT+O^qbWl{>*+r0000000000fMnIj7r1l4D`BqDmfL<n0000000000B0^swq`Py#JjF9QG3tIm0000000000?=0gXDZg{Tvo(HT&G>#m0000000000"
    "LY$DBrowZ;9w4SnYX^To0000000000cn6nY9L96N6!OCc3l@Js0000000000?&T#zipg`p9~tnLtR#Ox00000000000<UZp@XT|-(gH&TP&0o(0000000000"
    "589fkP0(||y&Tn9^+A6?0000000000ao*q2qSJH0Y}CG-oKSy20000000000h%;+o@78m`be9;`MPGkF0000000000u25_zGum^&V$=E!@oj%V0000000000"
    "K&=w_ZQXOg%JqFro_v2m0000000000oWPBzpWt)AHf*|?O^bg(00000000009|8>p$m4UsJfz6|{+WM30000000000BI&>5=H_$2XbZuBvZ#MR0000000000"
    "2<_Ds{ONPRP7k~)Xt#er0000000000Cxw1(2kmpfcpcIMAjW?{0000000000-P_>22=H^jHWnHb+SGqQ0000000000PI&ss|MGLdWx!-%mg9dw0000000000"
    "4l8Xu?DccN>hsm>RPuj70000000000UN|zh&G>V`=So@GiU5E>0000000000hvY)tqx*Bfi^ZQ>><55A0000000000$#a-dZvJz?X53);P7i=U0000000000"
    "YAP(ZECF=D`2jz*vKD|q0000000000u1xo^-UD>N%&6Rr7ao8>0000000000*&g%Zg9dcKUqjr2eI<ZE0000000000FXb5R8wqs4H$>o*<Sc+d0000000000"
    "^y6W5rVDhy*By(%Of-N%0000000000ZePb8BMx-H#zKDxwmX180000000000-<v{-kq~sige6~VAVYva0000000000jor*z@e*{vl{n4li%Nh%0000000000"
    "y=Y%GLKSqtf9`95_)&mB0000000000uYvQsgBNtb#JSxuW?6th0000000000sBD6fwHb84=M2UC)L?)>0000000000@@)lK*Bf-eZ(79QL}-9O0000000000"
    "cwi5{=pA&xKid`EwQqnx0000000000tD1=0=^u2!cYm(yCwG8A0000000000^|JRF+aYwokBZg;n|^>m00000000003Eg+&yd!kLVi?014~Bq10000000000"
    "luRU|k0o@##6+nhgpGhe0000000000PBj?7Qzvx5kt@C#`jmh`0000000000=gMxU2`O~Ioai?3aGZcZ00000000006hTA6u_|=H#Ft2*=%au@0000000000"
    "hRd*eOe}Q3+f*nWVXJ^Z0000000000@}1H;+AVazygLhC+OmK^00000000002<*dbTQ79LK4>OcRJwpb0000000000e(9fj&@gnsJoc^x&%uB|0000000000"
    "1j>QVIWly>iUHbIO38pg0000000000Ou>ZGmos$00H)kU#?XL300000000004FkmR>NIq~d;jalLfC*n0000000000_QG0|Gd6U<&M7_9zu$mB0000000000"
    "%7I@bayN9qkx~U-J?DTx0000000000+|`lYrZ{xKY$Y&XyY7HM0000000000G`4wP(K&R$^1JKLIre}+0000000000od#Ac@j7(CPled2xcz`Y0000000000"
    "<EmxG1w3@Xuke=UHwA$}0000000000*oNn?4n1_hKyAl=w+w+m0000000000M}8V>3qEwfENTjGHxz+D0000000000nJt}s`#yBQg#iq%w;O>#0000000000"
    "I1q%d-amA}Fr3miI3s~T0000000000)gF5vwLo;hPrdL6xhjD`0000000000#Iu*|enE7=*Om?*Ix>Mk0000000000SSjx|JVJE9zgpCDyE%bC0000000000"
    "<=K`V?Lu_G{W|IfJwbs$0000000000!5^f<kwbLAg=zTHzDR*U0000000000KUFM7DnxX^Q5dJ!Kv01|0000000000wbVq$wnTKmQ+b{a!&iYo0000000000"
    "boc^fH$`;7g1&}zMPGqH0000000000(;9p~tVMLd*BJ;2$Yy~+0000000000Czi!{6-IQxMZL<!N^XHb0000000000zQpt|b4GN)!rFGI&2@o50000000000"
    "_5dG+#YS|%Mx?Z@P<??w00000000008Kqg63`cap#7ivF(}aOQ0000000000gbTWfNJn(QHv@(eSB!x`0000000000hK^Nvc}H}>ko;+D+LM7m0000000000"
    "b%;TSpGS1S$Z3GtU7LYG0000000000rBjpBx<_=t(|mDR;i7>+0000000000Xi8vy%SUv;sb}TqWU7Hc00000000007stJ)(MNQ^KwN}@=&^x70000000000"
    "0wlq2%|~>=jg0mpY`K9z0000000000e!GA6zDIPxg(~F8@4$gT0000000000;^z)rrbl$ZAZYb(bI5@}0000000000e*=WTgGY40Qg_l6_s@Yq0000000000"
    "tZ4K@S4VWf6^bLYde?zK0000000000x^Z;|AxCt;Thfq9{@#H=0000000000{&*ZH;YM`89D>5sf#!ig0000000000&e;CQmqv8JPRn3V1@3`B0000000000"
    "e(Ny_MMiYM<lhswi1mR$0000000000VhhU%=|yzF+VO)34E=#X0000000000%#z>dgGF?}ATwD&j|75100000000006WoNx6-9KwuhlzS5)6Vs0000000000"
    "ib2rCo<wxOehbu9l@o$M0000000000h8@=eA4GJ(fd<Mf7#o5>0000000000S<0@UmP2&FuK2~=nj(Tg0000000000TB2;e1w(Ye{S$Cv9V&uA0000000000"
    "-Tx(gY(jLvV~TLPpD}_!0000000000sPF6d%0YC%-KT}&AvuCT0000000000{p(yjAwhJ&IoCqfqd<Z{0000000000x!*QPZ$Nawing+oB}jrm0000000000"
    "m^B$5wLf&gATr?^r%!@F0000000000Ram3C@;-FHNH*GNC|80&0000000000ud;L=C_Z$+Q?IFesa}FW0000000000Uau@kRXudTjfq=PDrSN}0000000000"
    "BDrchdOUQ%MX~$ms%?Tm0000000000v^vi5mOFI7%@US9D|LcE0000000000)t>flt2%VRbec9$t9*h$0000000000JWzwNw>fmchBWCND};hT0000000000"
    "sksxmyEt^fRL!}hs*8d^0000000000)I~>yw>NaaEK#)LDU*Ug0000000000fM2~8t2T7NUaaKRsG5R60000000000VsRB}mNj(1ILlUiC!&Ht0000000000"
    "Jm&6!c{Fsu3Yr$wrKy5I0000000000$0+GsR5NtIBn2?rBe8-&0000000000zN4f1CNgxu(fF>9p}2xT0000000000;=#RI@i272WMQB09>9V?0000000000"
    "?d{flvoCbOD=Q88o5zAc0000000000nv#xSZ7y`cbV3BG7|()00000000000r5c0<A1!pikCTu9mDYkl0000000000%kWlf$SZWf%L=~w5Z;170000000000"
    "%73PjX)1KUbd;EujOBtr0000000000Trv`P0x5LB-5e~^2knAD0000000000J##r8lP7e*Q;>wKg7ktw0000000000EUmkP8zywXDBG<d{riGI0000000000"
    "<w`iHn<R9=sOhsWcLRe!0000000000A-7J86C-rMA*iC(@Ct)K0000000000om`$QgCTUl<N}5hY7&D$00000000008K%aI>K}B#LFQ@X;u(WL0000000000"
    "PX1=7Ngi~-g+5wGTOor$0000000000`OhALpd56-0Qh1$(<p;L0000000000)rE~1?;3Q#0RNA>N-%>!0000000000pNdjzH5qik)<5mY!8e0I0000000000"
    "6iE$wa~E{L&8CK6I6s3x0000000000@GD(JsTFj<Hy~($u114E0000000000@15bv))REVWtg2NBu;}s0000000000&IvjS{1J4(qx&*ZnN@>80000000000"
    "NZJJj84z^9KsGe~4PApk00000000008+zoqEe>?Ri!`k(f@Fh000000000000917-It+Bc)Z(ev^=pGb0000000000C<+)*J_>ZeRgf(JX>)@>0000000000"
    "c@+&EItX;YYkQ`k+j@gQ0000000000N3=m(Ee3SJzgs1+O@f0!0000000000bzX$i7X);`Dhf{}zlnoD0000000000<cp(D`2lpmfM3Y=FOh>l0000000000"
    "ceHJl(f@P6nKU*VpqPU|00000000006i~5np#5{ePGUWY51@lU0000000000nASY0W&3l$a>0T(ey4*#0000000000?!Vp#A^CH_;3}&e?5=}A0000000000"
    "^Mhqo()M$}Zg*QRS+;{f0000000000jhdfYdGvF@@qajG#l3?-0000000000oqJp074mbyLLKL~FvWvF00000000002%l%Kr|)yXHz*7no6Lhi0000000000"
    "v3!z{F70!`sCi6=1k{5-0000000000c#Nt{s_JvVYwAq!Y}<oC0000000000LPmb98t8MtRa3H0)Z&9c0000000000^da1Mf8}$)L5(4sI_iT!0000000000"
    "Z<msK*y3}*25~>hpYel00000000000mL70=CE#<wch|7c1NnnM0000000000PA@}2X5DkZZ<!jXW&ngh0000000000c0t;gn%Z-~!H*$E$OnW#0000000000"
    "^UMQB!q;=aN0VxyC=Y}{0000000000sY#9(-PCiy*L<3=hZclD0000000000d@o4S>d|w+N+nim<sF1T0000000000P4@pk>&|n)bY()tK_!Gh0000000000"
    "{uP8N-pg~qD;5N}oGgSu0000000000bq1qX!^m^MOb|dh_A`V)0000000000lRqnyn#FU#r!1R4PCJA^0000000000J2~s5W5RR57eu<jq(X#10000000000"
    "QfskQ9=~(IbT2}-`ALL80000000000yG=}P%DZ#Gk;m5)O;LnE0000000000RalC*XSj2~N97;Uo>+uH00000000001_L_!^|W)qZehRV?q7sJ0000000000"
    "w8bg^bg^^5+!R|yJ!gbK0000000000I~XXB<gIhSX4seMiEe~I0000000000gmMFOM5=SZ=!6=`)OCbF00000000006MOx0l%{jQG|XAo9espA0000000000"
    "<uWyA)S`31)>aD%WrT!40000000000?x~M11fO%j<%Pycs*8j`0000000000t-X`1Bb#%;P37T*?UIB*0000000000oJ+{VGM96}`T(@4E}Dcu0000000000"
    "HZinTF_Uw^*kqHTZJ~rf0000000000_e<VKACGgu(wx>=s;GoO0000000000TY1fS{EBnH)il(;<gbK40000000000;Fbu_$cA&k&$!~e9Jho(0000000000"
    "|A7`Ngo1OxsI*u>Qoe*h0000000000H5h(vE`D>sP_&FSg~fzG0000000000{WivX$9Z$WutU{|wakP-0000000000)Vp;zPjz#^vhIi`<I{vd0000000000"
    "HiQ#{#c*@LM?wDj4cml30000000000nr$XQDr|GWSjUU<HR6Oo0000000000gjF?EeP?sP(F}<BTIqy90000000000W*NtxzG8F0o}EejeDH)o0000000000"
    "#g<|N?p<@hs|6?LoA-o300000000007OReA3t4l(<<NGfxBi4c0000000000*)!6l7F2V<H;a!Z&;^A+0000000000i4L@;4^MNzj=;xL<qU;D0000000000"
    "p%}YN_DOTV)ckcC_Y;Lc0000000000n=fv8%0zR(^~>;P1sjDx0000000000^)OTqj6ZY0-MNk>5F&*@0000000000Ec)~CIy!T}d{U-D7b%560000000000"
    "xM<V>*EDm$xgn0K8Zd=G00000000007+QQGU@vpPfwP=R8aIVN0000000000$E$KR)+uwq!ILNq7CwbQ0000000000K07`iIV5wyV@`<P4Ml}O0000000000"
    "{^e!Ch#qsmR&Nrm0ZfHJ0000000000eu;4Y#29nHgn}Jp@lu6A0000000000I|Jpo?Gkgq*b@No+*yS{0000000000v@lB30u6J(Laz`##9)O$0000000000"
    "V(%s}1P61#s^FzJr)Pyg0000000000QJloU@c?tcKH2Thg>HpF00000000006vr#U%ldM_Lo<vCV0DE+0000000000a|!^3lJs)G2((G=HGG9Y0000000000"
    "u~X+?M(%RJo4fCa2ZMz`00000000009FT{l=jU?352{Dt)QN>a0000000000^{j@ecHwfsb>A8Ro{)t=0000000000ftQNE^V)L2-McdMVwZ(L0000000000"
    "3q3hIU(|BISf5zKBc6po0000000000*NQlBxy^FG`RP@7;G~5>0000000000D{d~b0>^T|%=_~knX83B0000000000Pt?FvJHT?l=QRAfO|peR0000000000"
    "g{qaSW4Us`SM@hs{<wue00000000007~WC}e6n)DE}d`+tG|Un0000000000OkAGCgsXDEeT;9<Q^tir0000000000XA&MRex!21RU`hi_{@bs0000000000"
    "u<du6Xq|Gv$HC>qnbU<p0000000000ZQD^@M3!>E<;U*`H`;|j0000000000<j7IV5sz}fyp`pO(BOqY0000000000T%#Q|(T8%tUyG1bX6J=K0000000000"
    "8(3TMf`D?s;Rw2K`0a&20000000000Xw<M$CwX$fP;M*chV+F%0000000000hXur}zHxHEy&JB<5&MNe0000000000z}PxdNosPyIdKaNnE{4C0000000000"
    "Sv$L#$6|88)y@ph90-O$0000000000m>TQTIa_kTqm+9ToeqXT0000000000#A9$(pi^?dvLYeo85M>=0000000000Aq4Sp`%7}b6xqg2lN*LW0000000000"
    "`REdTOGI+O+7~ue2_l9-0000000000kpFwikUVn06hAlyd?<!M0000000000DmBur%QSMp(yo?j>n?^s00000000005_h|L{48?7C&wC$S2cz}0000000000"
    "jI!N?BPDXcBIu2Dz&nOO0000000000-PGP6K^=0y)N#2jCPIcl0000000000PRnGURTXl;P{d}zh)9M&0000000000B7+rIVGVM?qd#9c=uU<}0000000000"
    "Og3(sWCe1-9h708L{)}B0000000000vhZ;vU;S~wPgx_?om+-L0000000000cL0a`PW5rX@k+Ky^I?WS0000000000PD^aqG3{}{Tz1G;L}-RU0000000000"
    "=(gZ<1?6$T<LC!kkZy)R00000000007G}e&$J}wi-d)JH)^vtI0000000000S-6gJc+_#gZ_L?s7kh?40000000000JcU6w8O(9ObT1$kQ-Ov+0000000000"
    "QLL|ZtHW`?Has#Fiid_k0000000000E)A`yFuQTU0;s0;yp4uH0000000000V}}h(s<Lsw9Viq!?30E-0000000000g8HP_8LDx>&~Xu97@CGa0000000000"
    "97adUfT3}~WI8qJL7;{}0000000000$w|nX;Fxj1A~zORX{Lri00000000007v=lLI*@U|QyohukF1730000000000n)kKHkcV-=LZWKjv$BRj0000000000"
    "=3vSi<bH9$GkXj9*SLm20000000000ivSHcHg<8qa$Usw`@V)i0000000000)G-K~h;4De=Eo0XAjO72000000000074RxF++=aUq-|(@Mazaj0000000000"
    "Eh(9dFkErKs+ngCYte>40000000000a&^e_gi~?A{qKXgkk^Jl0000000000Gi@%p*h+E0pcFJPw%vw60000000000!>B+kD?)L=nyc;|+vA2n0000000000"
    "cEEI;dpU8y^jIn_|LKN60000000000olHPk$uM!iuE1RhBJhSl0000000000h~*fv6DV=O(T4wBL-vM100000000006)>juSs!u0AW5OtV*G|c0000000000"
    "id~YVn-+1vG#8;afC7g=0000000000DPV=7*bQ;O*{kz?n+S(M0000000000zjnIi5Cw6-#ocJGv<`<r00000000009N=tYMEr2TvxCsh%M^z|0000000000"
    "4%^odbMtV(TXg#1;2MWO0000000000V5T<Xoa%7EZ-%z)^B{*n0000000000;ef=?!QpVgr;yb31Sf|;0000000000U#Yyy;n;A%ylEr>6fK890000000000"
    "s9svk{LgT}W9W4gAv1?S0000000000iQm}9631}BQsbC0EIEfj0000000000&}wtABffCJL@`}pH9v<y0000000000Oh09QFST&M?x>QaJVl2<0000000000"
    "$(!jVHmq>K1R)6nLQ98000000000006~8;2H=}UCLU53PMNx-90000000000{^yp~Gn#O~T5b3tM^}eH0000000000QB^j|Dv@x&2=5T=MqP(M0000000000"
    "-bGbx9Efnh{>!Q3L}Q0Q0000000000^$jPo2!C+Eb})MlL1~9T0000000000>_y6s?{#p%k(LB#J8y?T0000000000*17=l&}?wPy!^N4Gj)eR0000000000"
    "u<Sj{tYdJ%Q%N0wDtm`O0000000000a9i~WgIaLFtzx_+AAyHJ00000000005QmzbQ&DihBQG7v5r>CB0000000000iSM9w9!PM&36;lS0*!}20000000000"
    "))ah!<34b}zIsF7@REl>0000000000?RKBAqBU^9m&O=4-I#|!0000000000%9VcdTP$$E>|W1a$DW5k0000000000WAThO4<m5E6e{mVucU`S0000000000"
    "wFy3Kycux7X$}D7l&Xh70000000000wO$2-V-Rq_L-L6!c(8{+0000000000U~pDy1P5@y`MVSgS+<8j0000000000TzyF+KKpOL=us$qH@t^H0000000000"
    "E|bU5X6SFgV46&D6T*i;0000000000Dk2WNg4J)p!FvhY?8k>d0000000000MJfcuk-~4lU)<)e!_9|40000000000ZL5&Sldo^Ul%j0nm(zzp0000000000"
    "m!+Deh@5Z0wX`B;Y1xNB0000000000vlt0IaEWigAo-9kINygr0000000000wgkg$N_KC+Cssc%1m%Z70000000000l4-9K7h`Y0C$hC$&FY6h0000000000"
    "H%GM))=zK1ck_FvlkkT?0000000000nw`5shdgh<Z2%euSN4ZM0000000000tZb=>Dk*QkW4{+`7yO4n0000000000U$kYxzZGx5u@?Qw)B%V<0000000000"
    "t8~}PMgniZvg}b7j|YfA0000000000cdg}yzwU0pxhI}OMh%ES0000000000zg~CIDcf$qBCcsg`Vxph0000000000a)@(^g~x8dLuNb=s~Ctt0000000000"
    "gDSz0(z9;BcQ&4sSRRN#0000000000<U~iq5TI_r6a^Q_0wjn)0000000000hu7=?K#Xp{a=0#_sVRs+0000000000U*ec3V0do8>Naf+O)rQ*0000000000"
    "Uu7~Gab#}5)@E7+?KFr$0000000000bzq>ea!_u-iY|V5i8_ct0000000000n$yY9WIS%bVsv0#B0z{h0000000000yuOuHM<{N<wH}G4w?&9R0000000000"
    "&>`y{8We88+!KIDNlS=80000000000!~C!d-T!UCD>}_H*HDN+0000000000j!(||kLzu~0C@ObVOEGh0000000000AQ~#mGS_Xuv9pQ1=Ua$C0000000000"
    "ZCtBc#lmgCm(vJ9YGH^#0000000000W)?Y%N3Lza2fqE>=w^sO000000000007$wxxtVRiURm&lV{C{(0000000000DDf3T8ij4Z@{4FO+Hr_L0000000000"
    ")e$p^YH)4989eUKOm~Pt0000000000_Nn;ksatKpEKmtnyL^a10000000000g1JpZ*hX!@impMoC4z`R0000000000XFwTu^)YS0fjuARi-(9n0000000000"
    "m_9;b0Ud3?Z{sQ1?Tm;(00000000003be1``v+~ntcK};OOl8{0000000000uAoTQ<nnC5kPYV1q?d?50000000000cLekxyxnZTb6%&v`J9MA0000000000"
    "Q{R}KfyivYt6#M}N}`BB0000000000Ix!wBHM4BMllPZDmZyk700000000009r|jk)}3s?f(+NV-K>Z}0000000000>p5onWr%FR(uDe)AhL)+0000000000"
    "neg`a;&N=j*+z=rUbl!q00000000009v%dPO<ioj@Hh^1n7oKU0000000000V#0KKrAKVQa6t}E&B2I30000000000U)?Xg>M?A<utpML{l<tu0000000000"
    "25%fH9UN@H2rr_bD9ngJ0000000000Np~TjJO*sQ(%?k^Ptk}#000000000081^OpNbqaGW<rH?aMp-G0000000000YU>aqLfUJ<8an9hjN6Dn0000000000"
    "E?3FzC&X*ONv`%{q~M4^0000000000Ryj63`mJlgN`*Maw&jRG0000000000*pWW)xR-0dbfq;5#OjDZ0000000000p<v17V}WbHAoX=6%<qUm0000000000"
    "sinRA`f6*ys#<{v&-92u0000000000;9vq2epG9~WW3*}%=w5w0000000000HIjY`?LBM2ssI`H#Qumt0000000000o;ex6M<;8*)D5-iw*!el0000000000"
    "5LdYnju30WILbS7r3i^Y0000000000cj`snzxiswFdQ|9jSY!F0000000000%JZcL-QsG%7V0P&a1x0?0000000000OCPjo<;!Zo$2gE2O&5tk0000000000"
    ")D1|$*tBZE^$?(1BpiuA0000000000H0yL|ww-Fg2fMr{^&p8r0000000000asaxkeTHhl(qBi!z9or30000000000jFy(vEpKYTEhwRIfGUYV0000000000"
    "?{-8)##d^<KZzcEIWLJo0000000000n}Yy#M?h-8zaH%F=`)Ey0000000000{#c@(vM6f6U0d!=k~oP#00000000000kDt!2oY+)8d+c<Gd+nw0000000000"
    "f<Z3IOZjQQ@G*Vu%t47j0000000000SjOW*eBo)p+y8$&U`B~R0000000000X!jMOo5^Xw*xaL(?MjJ20000000000l8q`OtFdXo<KiQIbx(;v0000000000"
    "u|bCcteI)R`YStq_*02M0000000000rOc6dpMhz>7SbdfcUXx)0000000000Q0K41hG}WQIPV+b@mz^O0000000000jY>xzV^L|qSe;u_X<>;#0000000000"
    "I<LLMH92X(b@m?m-DQbD0000000000JI;9T{~&3=h!mk*P-=-l0000000000Y+&Bp!U$==k`WyO!ET8_0000000000s2)_ufADC)j0etfFms7O0000000000"
    "6hS+OH`r*vY*`-#pLdBs0000000000i^nZO>Az^ePWV(z419?|0000000000e=T{Ylc#9FQv86RcYujN0000000000gBWUJHIHb(sv5Og;Dm`l0000000000"
    "CoWAS&30(Og$M)GMTv<(00000000000u*nWSzc(s6?=STr;Uj~0000000000rEeT_*+po;h$k~>1d@qB0000000000rsknsOD$-?5yLn&T$YJI0000000000"
    "IY>+et`um%{uZAlu9}HJ0000000000>+m&P0Q_gbbveVa`JRbD0000000000l3qJQLgQz^i(@^<KBI|10000000000oShEUbINDHp7$4TeW!^)0000000000"
    "ce-ILm9b~Q5m;TawX2Ch0000000000gu&)xrkH2IKIa#j=dX!C0000000000Grv!kr+;U_hX+zG6SRpy0000000000@V`#pm}Y0dNgEb>IJk*G0000000000"
    "9MF|kdQE4*<AiR8SG<Wp0000000000Cj8bJOf+Y}wWeiSaKMQ`0000000000dr-te4jX5{9ODH5gT#qI0000000000f!Po2zyW8#c$`^)kI0EY0000000000"
    "pQCFqWawtVC^c*3mCT7i0000000000M7xP9_|9g)hz~_Ime7en0000000000+AJLPe6?o4{*Cc_kkpAl0000000000#01r<@|$MB;1YYmgxHBd0000000000"
    "a!+PvS%PN3kO1rsbKHqQ0000000000OBsSfv1n$%Z@ErwTi}U60000000000yKdT9`c7uR*`KZDJ>-c%0000000000EL<VSG&N?xFcuSm8t92Y0000000000"
    "1MUc=U>jz@&whzW@#~2|0000000000Fv#%@egS5{7j1V#!taSd0000000000g;p;bjpt>+ko>N0jq`~>0000000000ScF93jm>4i5);qDQum2K0000000000"
    "54N*<fU{-5d){+668nij00000000004^aR8WSM2as*E|v%l?T#0000000000w6>{&JAY-sfJ?k;e*%g?0000000000XHGRB1Z8Ew*hrWdEe47}0000000000"
    "ihCsPzDi}lm9lBd)Cr0}0000000000eWLjTY%pcOleWxwb`6R^0000000000rYARf3l?R-wSrw75)q0(0000000000Wp7_)o%>|K&_uYcrxc1n0000000000"
    "9^UldA>m}e&MJ8&HyDaQ0000000000GT3<lnZ{(mhGIN?z#ED{00000000003BqvR0<2`f+8LgzLm!Gj00000000000RUW)Uy)?MtbRJFz#@u30000000000"
    "fcCvGuXbd>*S@WAHztZe0000000000=m{Cr@mplTI0%&arzwg+0000000000mY(mDCqZPuuZW2}5-o~A0000000000^%%s2P$p!+9&Rc!b}))S0000000000"
    "X5ARUYzt(-VdfIk(=&=d0000000000OHgoMeDGtySs&jQD>sTj00000000001J+D;f7N5a;YKU@dpe3h0000000000_|h;ScDiH0+lrHK#yyHa0000000000"
    "hpV1(V4q{aBpI}G3PFlM00000000006T}#UK7?byogXRwM?{K100000000001A!rI5NTt;Bg~Eme@BWy0000000000xkfug)=gu;l}v$%uS$wQ0000000000"
    "lzHePkTPSy($1WD*-eT-0000000000`hBY(J{M!axqyAs{7{NO0000000000PPhhA;QM00D3HI78B~fu0000000000@Qi%XcHm;b|0@5ZFIS2{0000000000"
    "L|~7#0mWj#8h_XeKU#`F0000000000w3fjYfvIA^U%KbMM_r0Q0000000000mz0$W^^Ib{qzcKkNnnaV0000000000Rlbd=U2<Z;&Yc+PL}Q9T0000000000"
    "Q+O;2yH;Yrx1n-TIcADL0000000000@+R~}4Lf4MK}2)sCTWU50000000000k}hX+R3Bo%O0%}34Qz@)0000000000obv~&kON}CvK-ld?QV)d0000000000"
    "Z`xP^!slVYSFUz(#&L>300000000003v*nN=*wZie#Yc<m~@Ij0000000000^)%Ug1h8SifXbARV|R)`0000000000F<E!(6_jDXwPnrNCwhuN0000000000"
    "`{*$X9e82D^6E@t<$Q`j0000000000mhA%u8eCz(1+CFRoPUZy0000000000H*n|%4M1VQ!~PkJOoEC)00000000009(vk1_9J1y|1#uN_JoQ+0000000000"
    "iWJqx)dyj~gPiWOnTLu%0000000000vq;n1tLtFEDUSELHj0Ws0000000000)*sNKc+X(KyIQ?}&5Vjb0000000000HlxNyJhNcH5>@^dUXO}E0000000000"
    "2i&pp_LgA4{?B-o=#h#*0000000000iqnkIsCr<)OdjhGZIp^Y0000000000{*qD-QeI%d)RGgM>z0Z@0000000000mm~hu@<Cw0Ux2VnWSNRU0000000000"
    "n~sW2izQ&d$T@&E*PDt!0000000000Lf!JE8wp^**%3i-M4pO30000000000kTGNsNZDV&Xj`!jte}cO0000000000)s%)lNuXcAM`@2E4x@@d0000000000"
    ")b>IwJ7-_OMvj@rYNd)m0000000000_i2wW9xh+NG#!o$!l#Nr0000000000zyM;N^7LN7=V&Vj6sn3q0000000000s?D&@xVv7!F2pa+V62Kj0000000000"
    "C*yGzae`jJ+JRYhsIH1Y0000000000v{ReQ8%bWk!1QJj>#&MI0000000000(lH}Mxes2zuWuH&D6@({0000000000?2rG|N!DG!dqpODVYP}t0000000000"
    "fPf~~&757p@W@nhmA8sO00000000005*Ch~MPpsS=1gF*#JP$<00000000002xBaNu_;}^DkKL@@4JdX0000000000=y~_i5%65Vkf8@!6~2l<0000000000"
    "EN$iMX0}|w=@%gFHo%HN0000000000L{ID@v3y*>1=<c9RKkit0000000000uQCG7@IqX`x;m2hY{ZH{0000000000>?s{UCJ9`?*NkYIfyRnJ0000000000"
    "W=Av9QO{ezDkD%8lE{ic0000000000pPf3{bCp}bi%<uCp2~_q00000000006hA4ej9Xj4$7f9Grp$^!0000000000G(Pp6oFQAlvb)`DtImo*0000000000"
    "fUT4Vq~}_|9iMp`tk8-;0000000000Z=d}=qpe!N)@0rZsnUu-0000000000btWd4nsZvfwO`6kqST5&00000000005>*6giaA=qhvNJFm)43v0000000000"
    "!aa@&bN^YuAwV%Kir9)k0000000000^76UPRK;1qP|GOpc-o3U0000000000B!_(tFpF8h>eirGWZa5C0000000000&<xKW1yNbRytP+=Ox}t>0000000000"
    "@#<<+)D~I5=kfA+G2n_o0000000000+ZI~ro!nT!CB)}f6XJ?M0000000000Bf2&CVWL>TL2Q63^5cp?0000000000blPg5AZS>?@N(|~&gF_g0000000000"
    "jD2SP*)3SW>tJW@rss-40000000000Dyfw#j`LT*@4gxUed&ro0000000000`eJ2AJ-JuFuoXN<Q0t070000000000y0AiR=YChf>+Sv1Anl4k0000000000"
    "CzzVKj6_$!R?w2R?(T{}0000000000?H<}lEecn_szojTx$ufW0000000000%SvuH$<J26pw|Drg7S($0000000000d?Cc4V3k(D@e4fzN%V?90000000000"
    "u&79+^IBHGS8f;I4EBmZ0000000000Bj#vZf*@AFh=##o&i9Hy0000000000l;4r#4CYn9LEUYejroc|0000000000teP-cldDz0Hv3MTO8bgH0000000000"
    "EFH4g6>(L-B59X&1^tRZ0000000000(QJc$lr~ksz^{2UzW$0p0000000000NRJU`5ByZX##bopbO4J$0000000000ONz;^hQU<8?djF4DFTZ?0000000000"
    "oMRYr`i4}%?(VdM+yje10000000000<{)d$YD`qXg75c(jRlK90000000000=*KSI*AY~}V|3BHJO_(F0000000000Twc(<LDy5jO9pg4>IjQK0000000000"
    "^Xa^csGL*4?JW`+mI{kN0000000000XmDX;3}REj2Xx+CKn#mO0000000000c=sw{Z6{N}P+<!i=?#lO0000000000(u9Vg%j{CXf_}?jkPnMM0000000000"
    "G9KYTC$dt&R8&zxHW7<J0000000000R>`MZfOk^Bfospp+7gRE0000000000>WfDX*E~|dzf5=#eiVy90000000000u4^rID*{r$$$)+zAQp>20000000000"
    "U2ev#eaBJ2SCAA5!551_0000000000q?v!w&5TjNBLr>TU>S=*0000000000Li5G{8&Xlg<Yc<6{~C)x0000000000^MD3aXctkyR68$+og9lm0000000000"
    "Wk{~-vfNO>ED-*DIv$Ha0000000000R8K14`=L<4BWzio)*p*N0000000000f1Bq}LuXLH^4D|waUqL90000000000kup<Ri7ZgSQ`~=~3?qv`0000000000"
    "mo&61oa0Zx0|KzArX-6%0000000000#<0>wBW_Q?w9z&jKqiYo0000000000D&^ZbsQ6C6Ak;<)*(ZxY0000000000#0LyGD}zqJ#$IB9aVd*H0000000000"
    "<ZQ(~tPM`U@K7#+2`h_00000000000Nu1L)E0|5d5Y~bKp)89)0000000000HxK&?s3J|k6XrGdH!h1n000000000089xB1BCSlo;dmWg&M%8V0000000000"
    "Ql+Q&nm0_qUWN}UWHF0C0000000000D!aAh5W-8qbq3;9`7(<@00000000004EpX_giK4o67g!-jWmlu0000000000UE3Ez_0~$j7(RfhAvTLZ0000000000"
    "W7zbEV_{0baOL;Fv^R@D0000000000hiXa|&+AFR0pF(<M>&f?0000000000E(_h)H+D(DwjySt*gA_q0000000000r89B2od8L|d8i0nX*`QS0000000000"
    "N?psJ0E<Y#E@)*t`8|t30000000000jRCNGU=>Kfz*$l_h(C)!0000000000u$kjEy`M+G6?LUb7D0<Z00000000009G@z=7Ai-;7qT>Jqe6>70000000000"
    "I*Up0Y_vweun5+UFGPz#0000000000P1gj*zdlC5!f;=(x<!jX0000000000!~7s656MNqKH{dvL`RE20000000000{B&aYT2n>91yzE{%}9$t0000000000"
    "L%EGAquoTn0>Q_yRZ5FM0000000000|3YV`=VwH~93D+{+)Il<0000000000mW#7iDDp$VJ!I$$Voi%c0000000000Q4BXGWqd=xO^a`D=1z-10000000000"
    "mOi%sod!d|G>f>FX;6zm0000000000&YShk(vU*H*jt%->rsn90000000000KApLH0~<oXCH3bIY*ULs0000000000PrSzSFQq}i08e^R>r{(C0000000000"
    "W^&;mST8}qRbeJJX;zCs0000000000%{b2Od$~Zs2qeO_<yVV90000000000>xfZBn?pdr1&<@hU|EYm0000000000OSoSoY@0v8GMXS|+FFZ10000000000"
    "h*w*Fn6Ey-c0G@TQe2Bb0000000000V&WcdyTd)ezHmhz$z6*;0000000000ol7&x)Ym+~>?;{6K3|JK0000000000(1jso<m)@Y?q=Yhv0#fp0000000000"
    "!-H}c>i;^xrx|f~BVvm{0000000000OgOK#<r6x<1U#XrlVgiO0000000000^Oax~*C#o^>%~SZ0%eOp0000000000i;oWszB@R;M6WmZZf1)>0000000000"
    ";1@-2nol>t`0q&s+GmSE0000000000gCU9(Y-2XS@qbfFLurdZ0000000000Kc8pOGIuq=5?3n0s%ncs0000000000<b!5=?us<PM{T5F5NwM;0000000000"
    "0m2XHo}M$nc)FJ$bZv`30000000000WlZesL$Wf!k2c!))ozPG0000000000-W7#k-NiA$aoIuPHE@eS0000000000`VeCkYuPZs2b-kPlW~hc0000000000"
    "6k#67+1W3^Idj;-?{bSk0000000000u+Ahx#Q-kA@L98|Npy=q0000000000?oaDOm?td27Do$#p>>Nu0000000000Xv%|lQcfzskTSME_jZdw0000000000"
    "$C?0(^K>Y{=^BdNN_dMv0000000000D4xtddz&V}U+K3~oOz2t0000000000VtTWz>A@txsUb0->w1eo0000000000>1Q<nKj<RB&C>C~IDCsh0000000000"
    "xSrBvdk!JM#tZhhgMEuY0000000000fvy$|K4~7nl-oIh%zleN0000000000@{_+FRJ9twH&MA96M&0A0000000000x?P}HIRO{It$T2ZSAmN_0000000000"
    "!F;(8?o1QF_i0<knSzTz0000000000)Wa4$<uDJx5i54m*@KHf0000000000UivJrj>iYU`^~(p7ln&J0000000000=zg{+*<Ap@xM3?)Q-+H`0000000000"
    "oMvdPis$jaLiN4JjE9Rr0000000000Xm@XNK9$J7oR@G0#E6SP0000000000XiLH0>4oP%#8*EL`H71_0000000000bdw=5N}2jUx<n@7D~pRj0000000000"
    "RE@bzE4TzecWohiT#SoA0000000000*t=<F3?vLd1>r2*ij9jv0000000000wK>q+BghdzS*}C_w~mWI00000000005^63<ZhjR&dE{j8;E#(y0000000000"
    "8~#ot<un;UUy_NR2$73G0000000000<%BU7i0d3c567+&E|QBt0000000000Kga50D^Vaog^|!lQIm^600000000000#zMxs4OBt!T{?%a+Hfe0000000000"
    "J~J=}I|w8|#UuLjkd=!-00000000001uCQZ;^QSij0yT^t(J>G0000000000*Z}q)puZ<T6v5`3$Cryh0000000000Mt`VFa+@hYV_Ce5-<XR)0000000000"
    "TWva0S$8WybK!P3^_h!60000000000my#<8RZ}fMLueA72%C#Q0000000000kGA_kWiKy4)BK9Q8Jvqi0000000000qEHp;#Dy?GB(B_-D4mNy0000000000"
    "A`P5L;d3!SGK&K#HJ*z=0000000000jP%i|2Vycn{i6GDKc9<00000000000=IZwJH&8P`hy=lIN1%&90000000000^V3Qday~Rb&c06}O`(fG0000000000"
    "c#{qFwJbG2(cq(pP@;=K0000000000dJTVm0~<C#i+Q=AQKO4M0000000000z$M?CS_wBm0ho+yP^61M000000000048UzSyZ1LhE=w8bOr?uJ0000000000"
    "Wx#4XCFVFl7NT1cN2ZHF0000000000jtUl%nAAByv#U?+KBtR70000000000O@gl_6~j6}2zT*wG^mR}0000000000rVDrAnz1`T5)(bBC#j1-0000000000"
    "U5c!%DWE(+&W<jK7^;gv0000000000JK~RY!HzvZL;1lH2dj%f0000000000K=)uGVtqb9WnuG1^Q((M0000000000G_7Bt3v53?KGwb?-K>j10000000000"
    "8sSdrUM)aCR|-d}#I1`!0000000000Xf^;!yre)th316hsjiDa0000000000`u>5}9UMVG`!TlOjIWD800000000004Pz2}gOfo(2ecTaZLo_#0000000000"
    ";EL}Y?FvFb05s}DOtFhV0000000000&7N1KT7p7AF?ndmD6)${00000000007RQME%lSe;?n==+1G9@j0000000000^((;&L2*Msg>Yi6+p~*60000000000"
    "$MrtAx#&YcMX^~Zvb2jp0000000000%}hY;HDE+Qgw*u7hP8`80000000000K3M~jwbw*IilK39S+<Kn0000000000eM82MI8Q}DuHOwzDz}S30000000000"
    "#WTmtzQ#pBJyr*2`L~Ne0000000000Qc^@QNIphDiOieG#<+_>0000000000gvWcQ)U`%H<p~LWlDUgO0000000000mb<GKWGzQPnAMDsT)K-u0000000000"
    "$W{_(_M}HZ|Mp}LCA*720000000000biwU_iyTNmUp3DK?7NFV0000000000*&}>RBa}!$5?Bd<vAl~w0000000000HKzNqy$eY|U?x$Db-jx~0000000000"
    "<8Z2(T7yYIooD1QIKGQO0000000000BCN*i`T9vf8RGze`M!%l0000000000FMy7an{rA(Ct&1*xxb4*0000000000XB9wjKk7<A6Q;~ac)*K50000000000"
    "2b39e<zY)eFnssgHNlHO0000000000P$XG_joC{;&#9tD^1+Kh0000000000neIG;H&IMLJ@yNjt-^~y0000000000BO3!@<H$@v&M0x)XTys?0000000000"
    "D$?l!kw8sA&<GG6A;gP700000000004LmPVK)6jnlU*ih*~E)L00000000002!Rna@h?t5YVPsNkj0BY0000000000Sjo<lqo+<lqk^zeN5+dl0000000000"
    "T0Ny$S07J6j^UpM{l<$x0000000000P6g9y43|$pf71Q-vB!%*0000000000wj@hgVFyq^!5j!LX2^>`000000000044D=TJws4HsfYZ!8Oe)40000000000"
    "#`Eey8GlefhuRpY%*l&D000000000007>WW^}0|%rretUe#(nL0000000000?W`&X)Amq6TUEf&Fw2WT0000000000sx#Dnurg6V_Y21+;>(Lb0000000000"
    "Rq+B!j&4yv$H(jqlgx`i00000000009{SZ>Y^qT}A5d?SL(Pjo0000000000B4TW-O6E~OPHc?8^UaGu0000000000fIa0YDI!upq2qz1qt1&!0000000000"
    "XrN!r2VPP@ZGDYOQ_qV)0000000000_n!KO=9yAJ00Hg30?>;<0000000000O-<EJ#nn<ktu@{ave1h_0000000000n7<Acq!Cj<z~Y`mVbO~~0000000000"
    "^$Et%f=yFEjL&Rr5Ymf40000000000yxrM#VTn^feh$u)ztW390000000000#VvOLK*du)0-r~|ZqtiE0000000000P7mvF9syK9xbtc39n_0J0000000000"
    "Cxwp3{5(`ZrVqLv&D4uP0000000000H<G#p+jvw!*7dwqeASCU0000000000O*zOYxwKS3OsV&gE7prZ0000000000J$(=~m+(|T6fN)5+SZFe0000000000"
    ">|pFBbu3jtI)o@3ir0%k0000000000Dd)5SQ)pE{!LcZ7IoOLp0000000000$`ZrGFr-yLw77J^=-7)u0000000000vF2}Z4&YTl8k~_InAwX!0000000000"
    "uVoYk>>E}<{9XixN7{=(0000000000o018G$yZiDX9QC9_S%a;0000000000RdA^4rju4cUXTB6rrV1^0000000000t^rCsgwIw$=>n4QRosg}0000000000"
    "e@RDpVGCD48Dfro1>K840000000000q)PdoK1WwT?5}whwcU$A0000000000^ar`H8-rIsaPQc-WZsKF0000000000Mugw3_`X*_uOl&N6yJ+L0000000000"
    "a>B)y)%sUJt4{|c#NUfR0000000000N5#ZzvNl*ibavqEbKr|W0000000000u#Yt@k8)T*5t~87BjJlc0000000000eI#+)Ypz&8hQP0<)8UIi0000000000"
    "hKA5~Na|QX-|4TKgW`)o0000000000pt!96BqmuvCK|__GvkXu0000000000r@dZw0b*G|VpMvr<Kv4!0000000000W|%+Q-JMxLnww70ljMs)0000000000"
    "&P%8+y4hJk+49H&MCFS>0000000000soL>ymK9n+CtYqv^yP~{0000000000&PR2ka#319irvGTq~?o200000000007?z8}PK{bX4R=lhRp*O90000000000"
    "V#tluD#=<vyDXD^1?Y=F0000000000a^cj~1_fI{mH`(Fw&;sM0000000000J><E^;y_zKt@vK7X6cJS0000000000jMJ5>zI<Ci1r_jN7wU^Z0000000000"
    "IkLfinz&m)s##Ag$m)wg00000000009fOuccJy07p~(RZdFzWn00000000004M}bMQ7~LU@@I?zD(s6u0000000000*#@7cEo@vst2cBF+w6-#0000000000"
    "Zoh;>38-8^%{x9YjO~j+0000000000n00s4<l|gGWOr|BJ?@J@0000000000Gc3?nz#m;eeC>?9?(T~~00000000006z1p6n_FE#8lf8}pzn)70000000000"
    "5~de9cb8p2MU|kcQSggE00000000000&lpOQqx^P3eBKL1M!PM0000000000$U`ITEe~EmZ%d*6wDF5T0000000000Hz%<+2}@o;eG@0pX7Y<b0000000000"
    "Cmg$V<c3~AKPb1n81suj0000000000YbBbhzrtQXwv5)q%JYjr0000000000MjpM@n*Lru{(iIQeDsSz0000000000*M#x+bvj=_$X5g>FZGK+0000000000"
    "Uv?f6QFdQIWUCo|;q{9^0000000000A`mMjEV5rfw+%$>llF^10000000000ZEyZD2ku`$q!QF`M)!+A0000000000!`QzZ@f2V{2~HXZ`S*)J0000000000"
    ";rU9Q-ZEf7%Ts5ttN4pR0000000000t0@#7%urxJ#J3A-Uipha0000000000*7lB^xo%)U-3?eW5&DZj00000000000q`ywr;K1g_UW7l#QKXs0000000000"
    "(4%LUl&WAr?%yitcKeG!0000000000?4Ya^g2-S%rh`(_Dg28-0000000000`E_xUZ{}b?25Dr(-296`0000000000n6W7lT?Ao3<|DDZkNt~40000000000"
    "dE!WsN+MxEE=Y;FLjH?D0000000000I+sEdI6z@QyJ<bS_5O=M0000000000hhB=8C0=1bbQ~VKsQ-&V00000000000LPOa6MSJnIP`nETL6qe0000000000"
    "PG8cf0GeSy<%FlW4grin00000000005VpBC?zdq;Ut;mH!2ygw0000000000_F28i+tpz}h#;z`bOMY(0000000000q1jMh$@5`AKT{%=Cj*Q?0000000000"
    "x5MZ9wh>}LYmTaU+5?P00000000000-=_7Oq%UGX=w*{rjRcH90000000000y%Fw0l1*YjnQbj0Kn09I0000000000^O8jHe`{huUpdL(@&$}Q0000000000"
    "F}ZA{ZHZz)9$mMPr3Q>Z00000000006BF8BTc=_`vTR5%SO<(i0000000000MlS#(NyTD70{qOn3J8oq0000000000YMfH;HsfMI?4Fe|y$Fmz0000000000"
    "Dd|MQBmrYURO#}OZwZV*0000000000A*4#55+7qg8E-Y(AqtE@00000000000$kUH06k+s9mK5&)C!D10000000000Z=YUv?pk9&Lwu|oh6{{90000000000"
    "4E<Aa+<0R^YAb>fI1G$H0000000000fCDRc%9dk5cx8X+=?siO0000000000Zb4UwxU^$HN51r)n+=RW0000000000ctuR5rqW|Ty~#2&O%9Ae0000000000"
    "P@a9xl<;FfyBmt5{tk>l0000000000mfBtfgAQasB}hx|uMdns0000000000?}%eZaV%s&*y6?!VGxW!0000000000{x|=ZUrJ;^wtr0w5)q6*0000000000"
    "Zg!>iO=x65q_QvE!V!!>0000000000+|tQoJceXIe-FNYa}ta|0000000000`rF^%Dx_pUEh<U%BNL230000000000XCPZ^8Ny^hiIyQd)Dw(A0000000000"
    "$}X=42;gKud+3)>gcOWG0000000000$(2vC_Wfi);j{cOG!=|M00000000002;{GB<r`%{o{XT~;}wiR0000000000I{-aA)H!89FLUctlNO9X0000000000"
    "5TxuH!&hZM<<aPqLKloc0000000000HJ(ihvUO!ZOhpc#@fVCh0000000000c3bNSp_64mJW<_tpcsrm0000000000qDWC0kg;Vzn0eP6PZ^9r0000000000"
    "e$om!f6rw=H9~%p{TYlv0000000000*=3_;ZtZ12_AXP|s~U_z0000000000eQ&0rT?=MFxWm`-SsRQ%0000000000JzSsbOetnSRD_M-2ONw*0000000000"
    ";FuXhJV$0gtn~GxwH%B<0000000000G?JjFD`jRtnwXM4V;zh@00000000000=2&w8-r#*{qy9p5FU&`00000000008EyiW3ZZ5|vx;v2y&jA}0000000000"
    "Ju37b`o3mBl;E%^Yafh20000000000NLC=T>D*>Ogn2D57$A&50000000000{9Y1L+4^QcTiIn4#UPA80000000000DX~@i$QWlp{Da%saUqOA0000000000"
    "nz?nZxHe}%JpB)W9wLlD000000000080GzHs8nY_{>X$2%OZ?G0000000000ZlHNEm~v-8A8WgMcO#5I0000000000Zdi2whmdDLcV{)wBP5JK0000000000"
    "<av42cdlnZ=s|Q1&?JmN0000000000mT?raXUu0nPJa0_d?k!P0000000000R;<vLSL$a#h@eJAC?<?R0000000000?Up%$M+j&@aV*_J)FzBT0000000000"
    "A)eZEHzsI6<)uC@fG3PV0000000000$Syx?Cq!sK!I10-EGUdX0000000000tF`iH7h-5Y=5T1#*C>oY0000000000mY;xW2Y_fmF3K~KgDH$a0000000000"
    "Q(1Ix_nl}!cOcM1FDi^c0000000000uTNQd=euY?p7*}$+A54d0000000000aPPW?*V$-5fYX79hAWIf0000000000Z>1!c$M<MJ{WqE(GAxWh0000000000"
    "Z{8uQw-sqX?x=~b-7Jhi0000000000JqJF&r!#3lE~?Qvi7kvk0000000000sEh>Gmr-dzq`&jQG%k!l0000000000dD{T)hi_>>C2es?;4X|n0000000000"
    "c3{2+c#Ua5jyBBBi!Y2o0000000000c!<a!Xsc;Jz@LRyH!zGq0000000000M8K*zSjlNXl`J3O;xLRr0000000000sC4j9Natxl?O+;kjxmft0000000000"
    "YZ4`BI0b4zp}_bAIx>tv0000000000U<TBEC?je>l+Qb%<uZ&w0000000000NWF-W7(r@4pb=$8ku!`y0000000000{G-mI2w!SIq6+r)JT#0z0000000000"
    "L?<z^_<U+Wc0O>i=roK#0000000000=d(M!=$dLk`Y=^`lr@Y%0000000000wzhf0*tlvy3}z-$KsJm(0000000000e4KX1$ku8=i(zFt>^6)*0000000000"
    "|Ixj~xb$j3P^XhLm^X|-00000000006~x-Ts1j>HJi9VNL^zB<0000000000gx^@Pm@sQVCn3aS@HmV>0000000000lb6Dth)!!jL9w)*oH>j@0000000000"
    "Gk~6jcx-Dx10)~rNIHx_0000000000C^vFtXo_n<d(8<~^*W3|0000000000G+M(wSg312zX5yMp*xH~0000000000C5IOdNXBbG+!T3sPCSf20000000000"
    "z+3dyH{@$T=voRd`#g+50000000000$93P7Cjx9h?<X|`s6C88000000000019M(V7a(jv2hjiaR6dMA0000000000Hu3iL20m;+LZ@&D0zZsE0000000000"
    "B#thZ_FHT~t^tlYuRn}H0000000000o#%=z=6P&DT{B*bTtJLK0000000000WSfz=)t78QW{~Ip2|<iN0000000000J4eVj#kFie)NC-7w?T|R0000000000"
    "=ZLhbw9{-rxg0-FWkQTV0000000000GXC2Tr15M(Ec6*J6GMzZ0000000000-9*G`lMih`I!DJc!9$Ed0000000000xcc(6f-P-8_-RdEZ$ykh0000000000"
    "f4%MTaZ7DLaImtm9Yu^l0000000000{wg{kVQFnZy>9(2%teeq0000000000_uySWP={?m>G(6$dPa;u0000000000GkxApKc#Iz3!_}6DMySz0000000000"
    "ah*X<F2ij=F~NPJ*hh>&0000000000fZq%~9pP<2Zl|5hhe(V-0000000000CE>*y4F7FF)v3ZHH%W{@0000000000BS7-z`W$XRbR6@m=1Gh|0000000000"
    "K$;h!={jyeVN*3@mP(930000000000Lyfyg*jR2rsG^TcMoWx90000000000?6zaw#&&K%UtS1i_DhUF00000000004YieCwUll^m)fPXrc8`L0000000000"
    "W+m6lqq1&5U~6tQS51sS0000000000yrr!qlF)8I&17io2TqJY0000000000)zhA6fbMQU>>8TcxK4~f0000000000drdHxZVYcg*Wn5NX-|wm0000000000"
    "X0Ye4Tq<uslnuUG8&Hfu0000000000Z<<Q7N=R=&J@UNp%utL#0000000000RjJ;cIA(7^;o}U!eo>4-0000000000+IlH+CWLQ5kT$f;Fj9;_0000000000"
    "$E>g}6ryiHT|T)K<5G-30000000000<J|eu0l;rST~8vKl~asB0000000000?x>1h@7-@eo4xg2NK}kK0000000000x31{W-1~1pE!{a``&5iT0000000000"
    "0LIJ|<qU8@A_DcKu2qac0000000000NN*9e+Z%8|j({yHVpfbm00000000006(R*j(kgI3fvoBA6<3Tv0000000000Os#6l$T@I84cagL$yba(0000000000"
    "%ijb+zesRELtO?<eprk^0000000000dx(~)wO4RJHHvfMGFgm30000000000K6c^*tY&aP^t64x=UI$E0000000000_IImKqIGaUlQApeoLY=P0000000000"
    "hFr9an1paZB2<D-Qd^8b0000000000((=E$j+1aewQz|92waRn0000000000w+WHigraajS=&YhzFdqz0000000000AaLsFda-aoBIv+TbX|-<0000000000"
    "?(~l6aKLatB<ua^Dqf6000000000000%t+kX3uayZ8!wd;a-eD0000000000K$XO|T;6a%5G2|MnO}@R000000000053n1DQtfa+S^iUuP+*Ke0000000000"
    "{WOqDNc?a>8>{q62w{vs0000000000YkK|(J_~U`0G&b}z+sF)0000000000tM^{bGa7L~J(E7*cVdh{00000000003nZ$XDJgM42~Fv0FJp{A0000000000"
    "-gel3A2@M9pov46=3|UN0000000000acAOw6-RMEItWfeon(wa00000000006>Y$w3s!MJ42TIsQ)P@m000000000082@VU0cCMOQpk362xg2x0000000000"
    "&Xn46_;YbUL5P;Hyk?9*0000000000hNTBF?}Bka6o+ZiZ)c1^0000000000k*Ml2=8<tg1cdOgA!v+10000000000JQaL^-Jo$mMLsMk(rAo80000000000"
    "CRwy0)vs|t2BIAlfoY6D0000000000L(9Y!&AxF!5A|+zE^3TG0000000000DoboT#m#X*QX(1S+G>nI00000000001Jmkhz1(p?Hy_R$hii;L0000000000"
    "_FRB?wd-*}C*eQZGi;1N0000000000G2?7Lt@?35l3gI$;cSdR0000000000++XmLqzQ6B;)x;Gk!_4X0000000000AI(Zln;3FHeOvX6L~e{g0000000000"
    "D<!-{ktcFM)W=P?`fiLs0000000000Ct`$9hBk6QQN-L4wQr0-0000000000K7QjzdPQ<TUPe?{ba0G70000000000n{ycgZB%kVX+vg4IdP0Y0000000000"
    "W;|wAUSo1V*RaHw1agc(0000000000$I=NLPI7WU7la`Q*K&+N0000000000@GwLwJb`jRld>*Tv2%<-00000000002vJp7D3EeMxiOwJlyr<h0000000000"
    "fo*hs5}$HFEGM>SfOU*O0000000000G&$pJ`mJ(6BqQ^7c6N+F00000000002gXYB-@I}_8&fuqc6W?G000000000090GXQ!pm|%O(e+DfOw2R0000000000"
    "n%g((qS|sm_5<-_m3fRn0000000000pFmggf9Y~SP+$3Ew0ev{0000000000P9LMYS@?25n4>B9-Fu8c0000000000%4O&fFb8u$1U||?5`Bz70000000000"
    "H3Ia!0v2;X)|(?<QGSd-0000000000uHur4(Ij&~MJj%En}3Wz0000000000UiFbgoHKJkkX9n7@PLd!0000000000Vko|~VnTC3?~Pa(QG$#>0000000000"
    "uOKvuBvErf%Tqijy@QNE0000000000_@RgE;9qk<Z1}}6Hie8p0000000000NkX{kmTq%EwJ3?ty@rfH0000000000s+Yx(MSXKXijf(fRfvp00000000000"
    "EjX|3>x*+h(ccO;|A~x10000000000+qrZNiJEgjaKP|1yo-!L0000000000y|NYk8>w?ZO5k2Hj*W~!0000000000-nfy;qqlQFLhR_tc8`oe0000000000"
    "GoR#tAI5V)`2HD|cae-h0000000000%Jb3>j?{BNzpm`=k&}!-0000000000_K3s|?&5Pm7rA;~#+8gf0000000000M4h$(KJjxvo|P|06_|`b0000000000"
    "NN=WTeE)Mm?z?sif0~Ry0000000000k3Esirww#KqNBSF1f7gP0000000000uts#Sz8iEvRm5qqq@RpH0000000000G|5bszbSM;o|WuXVWNyc0000000000"
    "wF4LfsW)^$9FMPHI;D(20000000000z%<i(c|~+UXeb^tFsO_`0000000000=9+tfFH>|t9oeHVL#vEH0000000000xO5Ji$zXIq)flX}bFPd)0000000000"
    "z!B*BMs9RKuVwa^!Lf`$0000000000lpW*Ls(W-maco~3DYcA10000000000r-#nZ`G|Btt$x!@rMQeh0000000000w3GajIF@ukUzyA~F}#dG0000000000"
    "*q#%qYol~P`gLCg$iIw00000000000I+Mk{ldyC^W(Ty8Xv2&^0000000000)X~Jati5zVRbYxz7RQW00000000000exs>2vdVNo*(C7^)yj-O0000000000"
    "7f}q<pVxFi1NruKsLqT)0000000000e<Ou0apZJB?0~9Ol+uhq0000000000kYlXhA@OuTreGRqoYss$0000000000^?Ms!t^IUB)}bUk#M+EN0000000000"
    "93iQ_4GMKY;PQLm4d0AF0000000000YeO)5JQj68DX#~tdgF{h0000000000%-n?vGa_|B&XtMu3+RkM0000000000DCd6R>MeCZ<YmJT!|aSe0000000000"
    "WnA!eT{(3?j^1`tq4A7B0000000000Yc{VvghX{f-M(cNsP>FN0000000000ByW(eSWk68^TkoC*!zq?0000000000PP1%3)LC^vi8PXGGy#o30000000000"
    "7714;^J8^DDw?*3y9SLw0000000000WI2Z`xoveo&I~RdYYdG*0000000000=YUpPD|dB327z$GKN5{V0000000000P#hk@PJnelW9#(@HW-aS0000000000"
    "Oh`9zD2a7IYCtxIN*;|s0000000000h(Y^xy^(c5v2>}<c_fWM0000000000w?xC<5}I{D!+ghfz$%SE0000000000hmPpEEuwWmFHHy&8!?SQ0000000000"
    "c=_`K6RLGUQBzqzhBu8s0000000000^a4Fg#Ibci@bz2=06vXC0000000000mQa2$KDl*3Xd(56h(wJ*0000000000O326Ze86=;0-)#uAWMxv0000000000"
    "1yCu`e8+V_7(D*J#!-zx0000000000xy`Z)I?i=K{lEMIeOQe^0000000000SSNJeu+()x0`{s9Ltl+R0000000000(L#NT-P(0Pd8p4V8D@<@0000000000"
    "7zW-@yx?^}wxKbf|7?vw0000000000X6>|`N#=DxB09UF_H&Iu0000000000Wj$;LiR*Phde9R<{d$c+0000000000qVPO*hwybkd5fs_5Q2?B0000000000"
    "Y-sCmQuTE}p?5zgFNuvn00000000005(Yed_xW`|f}x%>RgjH90000000000=qmFgg#L9vp%sjMfR~Lx0000000000I5^en1p{_KgW%0(uAYrR0000000000"
    "S&ybziU)Q;whk&$+@*~`0000000000mXRy;9t?Isz^rDT2d#}j0000000000;R{-%&=7V&p-A4#Ewqh500000000005hdnjr4@ES-1l#YO}mXi0000000000"
    "Fqop}o*H&QQ-dWzX2Oj?0000000000sd2u?z8`i#_S)xVcgc-F0000000000-@Rz01|@bttnQy?fY6OV0000000000J&Dj8cq(>4Vy-r`fY*&c0000000000"
    "E#M@~6fkx`1PYbAcixRa00000000006VH@W-ZgeWeet93WaW)O0000000000SYl65);o4UxKvEPN9>J20000000000W~PH+{y}y?p(A&iAM=es0000000000"
    "p{HOXSV(q2AWjYO>-mj90000000000baE9m<xX}$BgFSItN@Nc0000000000MibF9r&V@9mUSP{Uk8pr0000000000fR@6)pIml8WKwN71`duu0000000000"
    "hX@&V&|`K$cS_3Jo)nHi0000000000z?K8`I%{@7ea{{$DI1PJ0000000000e#q+Y;Bt0A##RnOq#=$#0000000000&>HDy#Cmo>&@H<d4k(U50000000000"
    "!zbTw=Yn=XMXfkrWG;?C0000000000ZHcWKOpA6v(Jh+Bq%@8|0000000000+}SjH^pkc#83^<L%sP%h0000000000CbLzf;+%Fs(Uexa+CYv#0000000000"
    "TCeb^7^ZeWoGjR^%SMhs0000000000h-l`Un6Gv~Bq)o5olK5E0000000000u}tBJWVv=g)thPZO;V0P0000000000hC<Nad&71>Ql_UG+*gi30000000000"
    "N@|Q@+0Ax9*Fq+uNnMUW0000000000`03<Pci4772nTC(m|~7V0000000000f#<07Pvdq#goyii%4d#10000000000;)th>UG8>3?p$4+;cSjT0000000000"
    "4)m{Jn)!A>;~5sq-*JvW0000000000{!Wpx00nnI1XZi-#&(WC0000000000sJ|M-iV$}|^P{5dn0t;u00000000004#7hRGaPq7wULjjSAdQ{0000000000"
    "BrPgL_9u5he05EO1ci=30000000000R=!p2(ld8JLC?l)poxw^00000000005Xi-T$3J&K)k4HxERK#q0000000000zMXk_(o1(h5_AqwsgjOB0000000000"
    "$&zqd@>h32#|CaU7?+Mf0000000000qxz|zC1iI%&Yr6adYg_w0000000000u_z_SYH)Wz`rnVk&!3J!0000000000WN_h5z<hT=B5vbV8l;Xu0000000000"
    "DJ9;tB#C!G9PMtoTBwdd0000000000YnmE&mX&uvya4;;k*tnD0000000000lp~fC6ry)P)=W^(z_5-$00000000006BJMgoUV63L3KB8=(LVN0000000000"
    "Qol#jEW39=(j9rb3Av6y0000000000zj%D|#K(6)Uh3VdCB2S70000000000$CK^jV$^p)y?!n<J;9DZ0000000000)nl3^1>tu<z<x7MQN@lx0000000000"
    "SUiF?s_l0`MUJ)aV#$s`0000000000q5>EQQTlg4I8}kSa?OrF0000000000eCAQ%`UZGF-5Ujsf6<OW0000000000X5?zUsS|iWx5m%Tht-Zi0000000000"
    "6GW|;Um$ouFy3U_iP?@o0000000000F->D{9xZr4qv}Tygx!un0000000000Z%5ZX=s9>mbaedAbm5Lb0000000000e|l2@zeadK1Ox_iTIG&F0000000000"
    "3@#$Gq*Hi6vmxkPGwF^%0000000000&Z&)3n_zfA;#DT-{p^lF0000000000)<jQlq-}UWDSar-xbTiZ0000000000?o&2Ozj}B;7gc<oWb}?e0000000000"
    "YI}`Y>W6qhytzk{0{D(V00000000000se6lB$aqT*e})MllzW90000000000XC49WYNB{RB|LA$8UKzz0000000000S|3<ny{~vcXk%b+l>(1I0000000000"
    "ibIs$8NGNwRVwou2nLTp0000000000v(3`pe#&@2uoZItatV(>0000000000i~up&>eqNcHmWC4*9?z90000000000aFXP9T;+H`(%NqpHV}_M0000000000"
    "cX1UM)be;h0C^OhjuVeS0000000000jpqdTQUG~CWgm7~-xiNR0000000000@D-I)+6{R?jNnziCK``G0000000000*d3^6Ya4k$>CSXMV;zq`0000000000"
    "fFK}o2Pt_#|NN6olOT^k0000000000B|qP+uQqu=QGIk<wIh!}0000000000`kKdAWkh*ERbtoO$0m<J0000000000LRGu2DN=brjlO+G$tjOO0000000000"
    "T^Ebb{a<-NZ`VdGxGaxA0000000000&F|&w;B9$8OZW$Nl`oG#0000000000<W4cH(0h46gerB~V=|9G0000000000wA$6J%87YEe)(b{CN+;h0000000000"
    "P&qt~%$IpUqVS~=-#3py0000000000&JfM&)1`SpTALDbkvfk+0000000000Z*%cr-m`f?QU`qrKs}E@0000000000C0fM#>A`tGX>e>H?LUt|0000000000"
    "vnone_Ro1hN>@B9nL>|10000000000ZIIn+1KxQ+z}V9<L`9E40000000000nl{-r5$t(Dlo1*{?njS60000000000lE{~^ANqMfk{<R^mr9R70000000000"
    "p7C?lE(m%+f*enKKTVH700000000005nafhJ{EdFF(?ec=1-460000000000Hn#5uPb7LkY+e*7jZ%+50000000000U#LU~U^03@_}fmqGF6X20000000000"
    "*WMwWaX@-NrM;qo*H@1~0000000000@dV<(gHC!tI3V|eds>e`0000000000{KemDmRx#3e;_l*9$k+>0000000000M9p=WscL#aM%z7B!C#L+0000000000"
    "8^-=Zy?A;+S8e_`V`7g$0000000000&Rf-F(S~|Jdh=U!1Z9sv0000000000t~pfr<&=6re2P62rDu;o00000000001Pj*p`l5P3Cr^qTMQV>g0000000000"
    "Bzv`85V3kdM_wtI<ZO>X0000000000Uo}l6Cck<>qtpS8gKv*O00000000000_FEbJk5GQ3eWllB65#F0000000000S~U>8Qrmh!Lu&g6z;us50000000000"
    "w$bo|Y3X`E8pG<FU3ZT_0000000000Wo7h~fcJVpTK2EW`gxB)0000000000xtn$Bmj!!3&>%9Imwb;v00000000000U*zQuM>MfL|<4EG=Gmk0000000000"
    "hn-tY$02(_ea;Cz(1DLY0000000000pTBiY-!FSWOp7fcY=n<M0000000000moa#J_dR<+e6p&>2ZxV90000000000xSqY!4@`SN&|qjaqlu3|0000000000"
    "S+bx_C|Y|!8naDxK8%k*0000000000#TR_!KxlhFBE~_E*^ZAu0000000000PkO47S$2Csv$iXUbdirh0000000000KZS-;a)f(8mZ9=$50sBU0000000000"
    "?Bi-4i;{alnWKF^s+NyH0000000000nL)Dfqo8|0gxRwSMVXI40000000000q)E5KysmpdBVc^j-<yv>0000000000QK!_A)x3K^JXF4{dY+F!0000000000"
    "^#x^B?#g>Wq|#uL6`_wn0000000000+z`VE2ibc-9r#L+u%nMa0000000000RC7tDALn~Oa^YjHOQw%N0000000000s@PH}IQ4r#aI2E^=BSTA0000000000"
    "g4^VYPy>8GHg(Qvf~$`}0000000000$3K$GXc2rsY!O2Z9<Gl-00000000006FaW(fFFE7I7&a-xv-Bw0000000000?hujum@Rxj7|{yMRkM#k0000000000"
    "+XR#HuseJ}hF9&~@wJaY0000000000Us?>-$Vq%a{f;gRj<}CN00000000002vt?8;8%P=_b3)mD!Y$B0000000000S~$jX_+@-R^w#U7#=Va~0000000000"
    "+ef!D5OjP%YfASAW5AC<00000000004#$A)CxU!H)ii)}|H6+z0000000000fcbi+KahMtwMWk4o5hbn0000000000vm*XrSDt)8fix*mILMDc0000000000"
    "GKOLxZ>)SkwXE2=)XI-Q0000000000erEvahPr$}%wZ%Lam|lF0000000000BMLCPp2&PaK?PK24bYE30000000000q_Ihzw$^+=kVvnhsnU-?0000000000"
    "j-jQI&g6VRH{E^3Mb(c$0000000000TOFL6=JI?%t8@p;;n$Bq0000000000plF4@{s4VIZNQVRecF#e0000000000+Qzy477u+u^l@ry8QqUS0000000000"
    "mV1$0FCBeAzB<$VwBL_F0000000000R?1}FM=O0mM$8pnP~wk30000000000p!SP~U^#t22H1Iu>*S9>0000000000{zG3_c}IOfdk;r_hUbq!0000000000"
    "_VqSbl2v^`9VEpuA?lAn00000000005>RV~t7CmYZO1#KyX=oZ0000000000(Unx`#BqH<;HvV>Rqu~L0000000000z6wi&-hX{S^nWp`@9~d70000000000"
    "Wb&q0_>Fx)DEL1%iS&;^0000000000Kajv_5}bWN@5E(nBKMC#0000000000-Y4+9EUJA#%9rX#yZMhm0000000000$o!2|N4R}JHX;_pQ~ZxW0000000000"
    "ep2!?V#a+ys1-2W>;8{G0000000000TjM5PebjwG0xI{#!2pmz0000000000mLQGLnBsjv+h2QY4FZrr0000000000a`TLXwD5gE<^}cRSObti0000000000"
    "*4<)n(Efcug6PsTqy&&a0000000000reRBr><oTD4%CBV?*))R0000000000!2pBO2pfJt=#+MKItP$I00000000004LwFWB`JPDaQ7K-g$R&90000000000"
    "Z<x`$K{tLt1jyW2&<T)00000000000#*|@CUPXRD0@E!u8w-#?0000000000`?E6bdQ*Nt%jA0sW(<%(0000000000?012UmtlTDyva4*unmwv0000000000"
    "g<^^*v~PYuHBQR4`woym0000000000p!JEi(0zVDls>4IMi7ud0000000000C$qdS?TdauI%htHkr9wU0000000000_#9u137URDf|UD-+!ByL0000000000"
    "{<N*cC8&Nt&yr53ClruC000000000082#kXLAHKCelDitautw30000000000D0e}rUBrGs>oajgz7~)`0000000000BNvWFdD4DB<A1r%2^f$-0000000000"
    "rrUw1l;D0qkgZmORT+>#0000000000Zb)a&ukL<8^PHDpp&F1t00000000007mt;_%KUynBQnBd?HiCl0000000000BH?3q<qCg5HBlUnIUSHd0000000000"
    "AaE)E{}_KjKi=!wg&vSV0000000000l5IUy8Yh21RlRme(jSmO00000000004-5BuG&O%fkkSI!9wCrG00000000008G5aEPD6h{0|M5BYa)<90000000000"
    "LhQFmXi<Ma$1gTcxFe8200000000006O(Q9fnR?>?>}iW1tpL`00000000006rd$~n{9tUk`}cuQYMf<0000000000*F-nnw0nO*$CqD3pC^z&0000000000"
    "-m>{0&WV3Opr@{G>?n{x0000000000xf{ep=a+v#HY(_>IVzAq0000000000^04hq0H%LHodZ(+hAWUj00000000007^KZT8MJ>u?iVs%)GUxd0000000000"
    "vL?G0GQxjAJ4?>ZAuf<W0000000000HnbhvO3;5m-~(hzZ!eHQ0000000000r$z;iW8Qy2XwulxyfBbJ0000000000R%YcgeC&TfjbYwl3NnyD0000000000"
    "lIAkal=^=_gs&y|S2K`60000000000OieMcRs?`Rg1TRlr8JO00000000000*Q)gQVheyky?{L~^EHq_0000000000hV`94ZxVn&W&m)>KsS&;0000000000"
    "1efQCdl`U0xHqU_jyRA&0000000000>-dY+haiAJ;3Svy+c}Ux0000000000-h@RblqY~d7{={?DLarr0000000000fuzoVpe}$wlNntBcRY|l0000000000"
    "b;04_tTli@h#i)8#66He0000000000T17cbxjcYCD9}mW5kHVX0000000000)R~XN#Y2EVtS>1uUqFyR0000000000f+(Lz(n^3pL^|bltU-`K0000000000"
    "2U{ZD-cf)+D()_$`9hFD00000000001e?=>>sWw5k4d+<MnsT600000000008&YjM_+NlPsY)-jlSPm~0000000000@ad@t1!sUjuJ1>a;6{)@0000000000"
    "A<RSU5pIA$&^EGDEl7|+0000000000QDXJy9(8~~M;;&Zc}b8!00000000009~OE2D|~=JM5I4&#!8St0000000000F;R&wIfH;e|Hj0{5=@Xl0000000000"
    "D1G33Mv8zyt%P6nUQLid0000000000qXRkWQ;~o`fZM?Ms!otV0000000000Movw5VVHnGu*sRw_D_&N0000000000vBuXZZ=ZlbZcUqcLQ#-F0000000000"
    "h1Qb&e5Qav@?>`LjZ%<60000000000X6HEgimiY^a1Pux*;9}}0000000000^kRe>nY4gF97ugcBvp_=0000000000(y}jRr@MeaD-a7CZ&r{%0000000000"
    "pg{H9wZecv(quNbxmS=t00000000000d|jo#L0j_M?|m|1X++k0000000000l)Si7)6jrGysRlVPFj#a0000000000{(UP|;@5ycX`E9Qm|KuQ0000000000"
    "-y@=l@!o(yff3)a;areF0000000000*caFF0Oo)|I0}6SD_)R50000000000jl+bG5bc0K#Dv}ybYGA_0000000000oRDQ?AoPGhT7R_HykL+(0000000000"
    "sOU3tF#CW&F8Hra1!9mu0000000000RW+)*Kmvh4b?t$4O=FNi0000000000L*RlxQ3-)SWL?`ym1K}W0000000000Q1f&GVi18qkt}h`++~nJ0000000000"
    "Yg$VLa~FX?x+)3yBWI960000000000XgN+qgdTxFYd9#-YG{x^0000000000Uf`j@l_h~dg`a#>v1yP%0000000000UY}}vr!0X$=Bc!W_iB(p0000000000"
    "e#Xpjxif)4YoUH*J#3Ib0000000000%%$w0%Q}HT?$3zvf^CpM0000000000VU6wz-a&ytQ-6Ui$Zn880000000000P0jYL@JN9`bNLn-4RDY^0000000000"
    "p|vTA15klLFNpWFQE`w!0000000000bC#={7FU5lVRg{~mU56l0000000000*uS9UDPDm<>Izc;+H;UV0000000000-GMD*J!OGFq>?JJ9d(dE0000000000"
    "nk;q}QEY)ga8?EtVRn!}0000000000AP}GZWOIQ)DWH@rq<4@&0000000000f_%&ud3u3BuDOHw=6H}m0000000000)xjcUje&tc-t2*SDSD7V0000000000"
    "Dlo?3p@@M%m4iL7YkQDD0000000000m>1iRwvT~8vBzYct$dI`0000000000#Ri6L%a(ya;J2kg?|qO!0000000000#vWaJ;GKa$W03NjFn^Fh0000000000"
    "c%o0f^`wD7QuqX>ae$CO0000000000J&zqc3#@@a<Mo?ZvVo950000000000YT}0aAhUr$NRM``@`8{+0000000000V3|w*HoAd8xXM$SGK7#o0000000000"
    "b+Ic@O~HXcXRh)Za)ppU00000000002Po(AW5|I(i$D@5v4)U90000000000aQZ#%de4DCSBG)8?}v~;00000000003nuVHlGcGh|A^1TEs2mo0000000000"
    "GN`@(sojA<xl8LuYl@IS0000000000Lw;r>!sLNKyVZj^sEd$50000000000m!+z;+UtQpI4xL`<cyF&0000000000i5p71^YVc}WmqRmAdZkg0000000000"
    "a@_<y4f=sVc3rSsTaS=H0000000000vgn#UCjo*$qC&F2m5`7?0000000000pgPRMK?s6C8V>Ai&ykQo0000000000n$TF)Tn~ak7sec22$PUN0000000000"
    "`Mm&mcou>{op}L?K$MU`0000000000beNN|lpTUVtbg?9ca@Mp0000000000kx>GOu_S^(S2<N~u9lEM0000000000hb}?^&ntpJkr5ye<d={@0000000000"
    "fpkbC?J|NuOLI}d7@3ej0000000000w`dCV3p#>8ddcN=Oq!5D0000000000R#1I;D?x%l3V+!lf18j%0000000000lW>-~OGtu1@ct;du$+)U0000000000"
    "q7F;CY)^tfAGFXf;hm5`0000000000v2T!pj#h#|g$d1h5TB4h0000000000_1Ie#v0Z{d7-|}eKA@050000000000pO2z@)MSD{#Xz-JYoU-p0000000000"
    "<Jeic_-ledg!q}$mZFeA0000000000^7(ex9dm*}LHxw>z@v~r0000000000{x|Q~LwbTh@wPIl>7<ZA0000000000I@q_wYJq}4gp=Ru5T=ko0000000000"
    ")E~y6l8Ayp?EJ^8Hm8t400000000001ym1nyN`lE8(!$&TBwjf0000000000`?-ig<(7g#10p?peW{Q@0000000000=$0oE5T1fSl(WKdo~n>Q0000000000"
    "|1aU{JEej^!hLnOy{nKw0000000000bIPjGXsm)jf9p{|+pLg40000000000cB^^Jma~FDyJu7Y_pOjX0000000000Iv7LY#kzt(XkB^h5U-Fx0000000000"
    "@>Tp5_Q8Tcc{jcFD6o(~0000000000%&D4iC&_|9-C!IOKe3QN0000000000{#X<DS<r$&heVS%Q?igi0000000000x~N;Pjn{%eYfT<kWwVe#0000000000"
    "Gu~*Q!rp>Fdgi%hb+nK`0000000000m_9Jg_~n8?qIzOigSC)A00000000009uKEJFztds*>n^jkG7CN0000000000_msl<X!L?W4?xVbn75EX0000000000"
    "R26sxqx*tDHb;gAptz7g0000000000WYxT6-U5R_KlMHgrMZwm0000000000U>b%X8w!I!9JTVer@D|p0000000000b}7j{SrLOk!10{`sJoCr0000000000"
    "+DCiBm>7dV92Py-rM!?p0000000000zPgdL*&l;IA&ty1p}mkm0000000000y?+oI94CW7f?^v1n!b=g0000000000NtwU^U@n6|)bs@;kiU>X0000000000"
    "`7QT_r8R><pUiwkgTRnL0000000000IeFvA>^y@&Uo^W5bHR{60000000000ZML-NHAI6zg?9WLU&4?;0000000000`yiSGflGrxiQRi2N5han0000000000"
    "Q1rq(&QgOw<Fj<RD#VaL0000000000&TC$+9a@7x2Sg%~3B`~=0000000000;=-GFZ()N#ZP>p|<He9b0000000000D`G&q!)Sv*X^O#>x5kh_0000000000"
    "aI!x|8gPR^mS+LShR2XV0000000000CL+jja(9D3`znb=Qpk`%0000000000f|YEN%YK7Ez)VSu8_AGB0000000000u<*b5B!+`P3q>*(<jIgg0000000000"
    "=NDD>eT{=a1&lOat;&!<0000000000QSZLo)s%xk(3wErc*~GL00000000004iHflES!Tt9ixomN6e5w0000000000mSMsJfun;!Z^9d;8O@MD0000000000"
    "z*Xd-)vAL)2r#iJ?9Grs0000000000E<cg$Dzbw>E~w3Jz0Qz80000000000U+_xYfw_Y~#1oNoi_efi0000000000-d5cs+rWcBa0y4hRM3z>0000000000"
    "Eobz{HpqiOxLh_18PSkH0000000000Pag-jl+S}e+8kp#*wK(c0000000000L4Azh^wxtw1X<B9lG2bs00000000001b_Q4SKfm_aUg}xM$?c$0000000000"
    "oFdDXyXAvGRf2-~_0y0*000000000024->sBJG1f<&DjgpVW{*0000000000L`P_0iu8j(S5j(#L)DN#0000000000R>UKs^!tNA;@(`M;nk2p0000000000"
    "KieC#U;~6eyF#hldDf6X0000000000`-#Vb&I*J;7mZUA4A+oA0000000000j?SI`JraaKEzMshnAea%0000000000_ofvXuNj0uF%r`I9oUdS0000000000"
    "IT5D1At8i7Tx+g%o!F2-0000000000QVYh_l_-Qj;OF>u7TJ(M0000000000K8F=p3owL0gqjfJh}n=p0000000000pJawQf;WUfEdWHc^x2R=0000000000"
    "8f$lk`aXm}kB(vqTH2660000000000SV6u<bVh_gQrLV1x!RCG00000000001&%O=?@fe28*rn?5!;YJ0000000000YJDO}Y*mCnM!%75WZRHH0000000000"
    ">7tKs>Rg0C()SDKu-lM80000000000FwfP@X=H>z@&}$8_uG&_0000000000Sf5Wc>T85R6++PHHr$Xv0000000000cCc4@YjcD^s`x54aNLkU0000000000"
    "rk_Y}?RtbjC4M&MqTG-`0000000000{Rw^PZ-RtC^dx)S&fJhd0000000000lZ;Qh@`;2%hd+V*^W2a?0000000000eZ(5Lb&-TXPax4L65WtM0000000000"
    "+D<vM_?U!1z8UjJDcz7j0000000000yT}EAd!U3sLw;{MIo*&z0000000000e;_{j{ilRLLnwjuLEVr*0000000000eXE!zfUkr=FX;u-L*0--0000000000"
    "o^`h%0=I-fikKNwKi!Z(0000000000y<$WFguaA8^T+A+G~JLu0000000000v{vce1;&Ix1PP=&CEbuf0000000000TQyM9hs}gQTkUnE5Z#bK0000000000"
    "iQ*#m3Dtx^l?IKl_uP;`00000000008A1zUjNF7kP39Kh+T4&p0000000000<TtY}59EYEARa-py4;XJ0000000000!l%S^lIw&(tUo!bm)wv*0000000000"
    "jdkfv7W0Hairg$ka@>$W0000000000;qJS>n)-x5$7_nvO5Bh@00000000000kRQp9|DCy6+k9xAl#5Z0000000000E~A&)qX~sT&5<a=@!OC<0000000000"
    "QNrzmClQ4}Jb#jiz1xsL0000000000RS&f3sThSot|nz|g4>Wl0000000000DvFAWDj<bGZl<9iKiiN%0000000000x-t>zs3(O$${a~j^4gF<0000000000"
    "?+ECDBrk<P45(bPn%a;+0000000000w+ye<oHm6(fo9$CHrkLt0000000000{5=L_5k7@LcXrIZ#o3TR0000000000s+^6ifklNtIQA;ZMcI%*0000000000"
    "6Q^4;?M#J0flWy6wb+nA0000000000e8SL+Q&fdO=qarE71)qK0000000000ae(?&v|EKh$-Z0*X4jBF0000000000?-}SV4`YQuIPY{rrq+-^0000000000"
    "@yDpJV`_y!N}dnG*42<e0000000000bHht}v2le!8LvmD_tcO;0000000000cSYbT_;`gtx&$~A3Dl540000000000^@ep^H-LpeK!W=v3)7H50000000000"
    "?7$kWZij_H#0std{n3y>0000000000TrA(YosNY-UV&%8;n0vk0000000000JOa`m#Fd3WA<0FEw$G420000000000-}Urs;hcp)50O-He9n+S0000000000"
    "2spIK^rMA9=JL>RGtH1d0000000000joHsX{i=mPXrCf(+slwZ0000000000`wsWi`LTsSBkQVAbjpxG0000000000;0)Q1>9~bJqbke){m76&0000000000"
    "#RnyK%fE#{c(UYTc*l@H0000000000JT!`tp2mehFV|RM<He9b0000000000(#E&tVa<g=SnY`7KE#kf0000000000Q<GwP6x4-4yzAJ_io%dU0000000000"
    "$U!rWwc3S1R2OR`$H0(40000000000QYk_SMd5`&%XgtT^uCZl0000000000^=--f!smrRd~(}t61|W>0000000000wKD^9FYbjv!)$mXBD;`40000000000"
    "lAbAfiu8p*`ODY7Be{@30000000000lhSq`)cS=$dG%PK7q^f=0000000000xbq}$3;~8foK9OU{<M%l00000000001}1KJGY5u1zdSz0*Rqg60000000000"
    "evOPrNezZTI3;1rq_B`c0000000000BTl;MO%sMdWo;v(Wv-Aw0000000000_=_=YK^cZXpnBx18my2&00000000000c$MWBOiu9Lb3x(#Ho-#0000000000"
    "LJ#Fe^(2NssCnSwVyBQm0000000000xSW-_wJL@{CpOyB_N0(N0000000000Y&Up8WiW<7AWcw!fTEB<0000000000UIWML12%?0=p&*`0HBaS0000000000"
    "k0iXskvoP!+aE8QcAbzw00000000002fvZr4nc-MRi;+U<C>5_0000000000#`b|ddq##pt{iwmNSKg70000000000&+o?d)=P#!Lg?~gq?M3B0000000000"
    "B+i-=B2k7wYi=ft_>z!70000000000|0ao|TvmoaK)lu=MUap{0000000000H$9EZhFgX}_BJ+Wij9y!0000000000*&E5npkRhT2CL?W$cm6a0000000000"
    "K-~t?s%3^iCI`K)|A&x30000000000$(s-*rD}#i3_6YmFolpn0000000000(Q&Z*k8g%RZZa>rS%Q#20000000000vw5(PYjlP{|K4=;e1DKY0000000000"
    "&du5$IC+LZcP=#GnS78y0000000000eu(+t^?imwiVs4luz8R`0000000000AjH-YrGkb)>~M2;!FG^900000000004x2peMTdq!RY9G2%yW=H0000000000"
    "rziEQ*o%fhcW;!|(QuGJ0000000000M}*3VU66)A4C*bs(QS}G0000000000O3Q}8)Rcxm!=FA>%xaK800000000004;S=|KADC<Sbv+z!Do;_0000000000"
    "@PXVYo1KP0e0Lc~vSg4z0000000000NhS)<>Y;`}<#vttoneqb000000000008ebVEvAM*poUxpg<g<A0000000000SfDb(W2%Ngk8DhZXIqd!0000000000"
    "sIiYljIM@2arg&qMOctP0000000000OgDLvrm}`W0aDPf9#xP)0000000000lI7&|v9^Xl`=I7P^HGpM0000000000(s0PWt-6Ll84ZjR!%mPu0000000000"
    "XN?&?nZAZV7QO)+j!Te00000000000q3_MXbHau|uiJ=9R7j9O0000000000z=_+KJjRAVs+nSy6-AIh0000000000l6#?G^T~!lbh;G&(Ls<v0000000000"
    "Q>oS3n9YVjSVIxOi9V1(0000000000*_=xaF42ZSo0<(IJv)#<0000000000{NL97v($z_cqYGc>^G1=0000000000nEXnzE7*oWE=}9?mo$(-0000000000"
    "f_!^0mfMCv`*aucKQWL%0000000000m+3ox^xlR*9|S43<Smdu0000000000tj%AcN8*M+*<i(nhANOi00000000006cM_ekmZIztHdp>B`1(T0000000000"
    "mkWy!(CCIhC6r`Z!XuDC0000000000Me}h12JD7F1XF{ZTp*A@0000000000UD!8lGVg{!ZqW3C^c;{t00000000007t5tgRq}>Fh}$C<iW!hV0000000000"
    "wA+N9ZuN#hbVGY)9Tt#40000000000Y#N?je)xt!Sl4BJt`d+y0000000000e3!a~g!_g-S!PaXJr9sT0000000000BV(fyf&PX-m%pDg$_tP{0000000000"
    "n2jL-bODDzJhH^=R0xnj00000000008z=rrT?B_fXbmWx-2{+80000000000<v#-KJO_tBLZCEMWC4&s0000000000H0bu>5(<Yv*>kEo()*7<0000000000"
    "L@=c%-wcO97*pv9*z}J;0000000000Q7G29p$~^ZjC}hp+U<`(0000000000x%E<!S`vpqqKdY{*yfKw0000000000EZixy2^EJxeEgl<)8CIk0000000000"
    "ah-NPuNQ|v{;heB%h-=V0000000000pl2#mOB#njG%Nz`ztN9C0000000000(`Yso-W-QODg^KbvC5A?0000000000CeMqfXCH?^>%#l)p2Lqo0000000000"
    "tjHTJ=plzdef8b8iM)?M0000000000frHn`UnGY>?kG=iakY;?0000000000y)Cev&L)RIN9qSHR<4gg0000000000d-ayiG%1Hbmd3#TH>Zz40000000000"
    "(FlEhk}HQm-AXd|7NCzn0000000000+m%z$=PidoF$5JT^O%o70000000000u*wVPG%$xilz$<k%#n{k0000000000Y1NkHc`}DU7D^C$ql%9}0000000000"
    "9^~cjwls%8yzrU6cY}{W0000000000>m=$W>^6r$m;iTzNqmn$0000000000=*5(+899eQuLfY(7<G?80000000000EA`f>KRbs&3y=-Z<!z5Z0000000000"
    "&ETOrUOk6ExhXh%ux5`y0000000000?Y8EHbw7tdzF?T`cwdh|0000000000oTZkng+YfvF2d$FKUj}I0000000000_}Ze>jzfn)58K;o15uAa0000000000"
    "8io8Fkwu3<XE-pM#YvAq00000000008Pb%jj7Nt+M`xVZg+h-&00000000005Aa;0fk}rzwDj0AL_3c_00000000005)DQra7%|ky=|Vp05p$40000000000"
    "L03UlSWSmOXkVp_yDX1D0000000000uKYh%I!}i{#a5=SbR~~K0000000000cPleX7Ey;l+u&I?EFO<Q0000000000vk?qe>{5q7vWj>$;ueoU0000000000"
    "dEL95yi|ukSmC<8mJg3W0000000000=DagMhgOF`)^XY3NeGWX000000000065eECOjw6NI?0T%`v8wX00000000006K_B!3|fakh;)rUtoe>W0000000000"
    "1D?0A##@I#&j6I1Tk?)T0000000000{Fu8Vd|ih?8v9t>3G0qP0000000000d*<GuEMJE}-rhYAx8sgL0000000000kukVN*kFf1)|EFvW898F0000000000"
    "wgaOle`1F~p1*%|4%Lo80000000000lJde?A!LU@0)!l~xXq400000000000&XYUxzGa6%$O;tuVaAR?00000000007kw}WSZ9Yox_fL_3crp(0000000000"
    "4@AYP>}ZETpZ+V$vA2#u0000000000Xnq~_d}@b4M7vm9Sg?*j0000000000#lc+q2yBNybJ_R|{-};X0000000000&ZDN!kZp%R?0^#GqM?pJ0000000000"
    "FuER!6mN$>gW0R-Mw*U50000000000lA7o~lyHYY`U=Y&>XME?0000000000o!Tb@5ORk=8$EW0jEjyy0000000000{SN@_hjWKOwU@IhE`*Li0000000000"
    "SkROX{B(yvhC+!B&wP$R0000000000Rj~<xY<7o0VP*X`Z*`790000000000p-6&N*>{IP3@8?<4sMP>0000000000<|`tkLV1TkPO{l)t!9ou0000000000"
    "$nXq-sCtJ$@heV#OJ9ya0000000000^ekop3w(z_zPx$s=vR(F00000000006n0TMYJG=5f{bFeg;0(_0000000000%wh6C#(sxD{lX69AxVxv0000000000"
    "%7ttyAApBI0*Hipy+MvZ0000000000w0GkRb%BRKRs+g=SUQeC0000000000G|hVY$AX7Iyj<n)@-mJ<0000000000^cz+S7=(vF3B26ej4O^n0000000000"
    "p81$lWrc@8{7AMzCM1qP0000000000*!yPauZD*}WGi?nzZ{N00000000000Ot|CT_lJi-3X3S0R~3#x0000000000ro=K#J&A`vu#e4;?+uPX0000000000"
    "inez}fQpAeCt+<EhX#&70000000000r6bngz>9}KGbau<9{-I%0000000000o}qCt|BQz~pb56^wD*lb0000000000OSo)IJC27y1_EnDOYn_A0000000000"
    "s?hC0b&rQYxbgr#;pmM(0000000000HZju)t&oR68X>3JcHxac0000000000DLnC*;*p0yG}qf14BCxA0000000000zf&CK6_bZR7hfIwpwf*%0000000000"
    "G1@5LMwEv@$pY$qHOq}a0000000000yfN#6b(M!eS`5IU$-|960000000000o4hx^p_Ye0(ZwBRUA&Dz00000000003Ekp}%a?~hI(dQ0@U)FU0000000000"
    "LiNDe@|cG}qi|`?gRPA~0000000000e`ltm7n+Aa682MZ7N(6r0000000000{gr3gIh%(-ms7v8sGf~L0000000000_Tt!bS)7MJInybUI+u+=0000000000"
    "rm)UqcAbYn2fq;y%#V#g0000000000NMRkJke-J?4q^2uU5Jf900000000007+0AesGo;GPy<un?SPFy0000000000Q1lkmyr73b;qRz1etC^R0000000000"
    "ESqYx&Y_1u$Z&NZ4swk^0000000000k0*Pw-J*v;KioK|oobCh0000000000Or`M9>7$20Sg54TEMtv800000000004<3dj^Q4DB+mYOlyIYMw0000000000"
    "3}vjv`K5<IUO7?qNmPwM0000000000ZcWt8{icUNVLx*K*i4N;0000000000W~y#d|EGsQc0{wZWkroZ0000000000Bv0&a{-}pQA?bk?^F56~0000000000"
    ")(*Zp`>BUO^UY5tfHsXl0000000000t-Msu^{R(Ja+8hf3@?p90000000000*efAI?yHADCEBZHnJ0}v0000000000f-jV=<gAB4oI)F$B_NGJ0000000000"
    "<C-2U*{z2_Uu{f|u^5d&0000000000B|$OM%dUq&{Q9p#JrRvS0000000000b@e>Fysw8q_lvf>$O(-=00000000002#d1+tFVVa+`!QMQv!`Z0000000000"
    "3Dp9&nX!jJH#%<#-};O|0000000000t)`y$g|df0l9&SJY4eOg00000000008`JjOaI=R%cYOPj^y`d30000000000i7kA7TC|5iZidPhf8>ln0000000000"
    "CW)W)LbZoL2v^{93Ehl900000000009l6n0Dz=9}$d0Y8lhuqs0000000000p^$^X5VwawJI<oV9nOqE0000000000<KiVp_P2*X?zJ$*r^bvx0000000000"
    "41*~I+qj27W9G-HF~5vJ0000000000Qbhj`zqyA%E&$nhySI!$0000000000;Q1GWqPmAb)uYrkMX-!O0000000000>2~CGg}aA9;c+YE&Zvw)0000000000"
    "FE610XuO9&dzD6qSfPwS0000000000w}RU-O1+0b3oI@n;+c#<0000000000Bd~D_EWU?8jc)g`Ym$sW0000000000I1)LE4Znv#sN9h`^@@x@0000000000"
    "wF3vS?!SjX1@`Q>euIoa0000000000SKqLH&%lR3MmqQ`2z-n{0000000000;r|Qsufc~v6e**ak#vke00000000006Ts~akiv&R975gt8f}a~0000000000"
    "uX-!bZ^MT`00ov)q-Bgi0000000000af+HYPsE2nYVo|HEnbX300000000009ZC>HF2#pH0Ea>7wpNTl0000000000auj0s4915*X+;wxKu?T70000000000"
    "FGOWf>&AybPplzU$w!Pp00000000007&Pxr$;XF4RXbdMQb3GA0000000000>OR%-rpSjtDsr-%+c=Cs0000000000V!!M=gvp0MY0wj~WHF3D0000000000"
    "MGy-rVakU<$JKAU?J0~v0000000000RE!0^J<Eqc<pH+4b|Q>G0000000000O^&hj8O(=3Xw=oQ{~C-y0000000000?{MVM^~{Gr{LH?ch!c!J0000000000"
    "`;K!_(ancIMQdPx5e$q#0000000000G($9Ytj>o(?k=iUngonM0000000000Sm^schtG#Vo0p3tBmIj&0000000000C)uB?VbF&__$2h@tM!XO0000000000"
    "VsQ1lJJE+gsaDgSH13N)0000000000$OzJJ71D=5QLGqCz2=KR00000000008T|^!?$U=qpQW?yMc<1+00000000007^ro@$J2*EH^P&J&ew}T0000000000"
    "fn?`ipwx#z!hxs~SI~<<00000000007>zZbc-4nM=VRfN-^hzV0000000000n|w8bP}YY)N$4#MXu*p>0000000000%#FMUDA$KTmyUgS@VSdX0000000000"
    "Ya5_1|JR2=bk@Jwc(RK?0000000000G=R1F)!2tXhCk#l0jrBZ0000000000?H1Bytl5V^cuCrNiKB}^0000000000R65IZg4%~b@^g{85uA%a0000000000"
    "P;-+bSlfp{FW&s_nUsq_0000000000G7u|xE!>Adj=FFdB8`hc0000000000!3pDc0^Nr|0-dTis)dU{0000000000FYh)f*WHIeLSt=BGJcCd0000000000"
    "s!yVltKNq|6YG0gx^{~|0000000000V!EKZf8U2dD-xMsLT`&e0000000000iN39;Q{aa{M7D=r%4dr}0000000000OY1~nC*g-c9<>TpQecZf0000000000"
    "+Rx$Y`r(H_Z|vko+E|M~0000000000X8Zq5&f<qa{5jb(Vo{4g00000000009sL@Rq2q@@bbG2C>Pd@00000000000D`c*>bmWIXn~oa-azcwh0000000000"
    "$iFqqN9Bh=E>Z8~`8tb000000000009iJ@G8s>*U-k|8kfisIh0000000000U6BsZ?dFF-a$Cx!2`r010000000000v?W*0z~_fRo?cglktB;i0000000000"
    "SnGk(ljw&)C5TdG86As20000000000a&ZddX6c7OyS08sp%sfj0000000000HAid)I_if&AH|+2DGrN30000000000&if)m4eN(M3zT&au?CAk0000000000"
    "aUQCl;OmD#KLEz~IRA=30000000000N<MuewCsmKa_3U!!1szk0000000000i0T*8hwX<zTqobyNbrh40000000000U9$DGTkeNI!<{YB(CCUl0000000000"
    "_#~3iFYkvyTqpt1SmBC50000000000h|y>+1n`GI;G`VZ;Ms~m0000000000JV*7>*zkuy67fRhXwr&60000000000M-{Wet?`FIsOmQR@XCrn0000000000"
    ")XE(PgYt(!UvIk`dBci8000000000075A{ESo4QK@?4Wf0lbPp0000000000Iw|O+F7$^$9Fc*1iL{D90000000000aPU401@(tOo9Rft5v__q0000000000"
    ">f12i+x3S)Cq%3anx%?B0000000000)p|K6v-XETfPQXhBA$vs0000000000VqDbeiuZ>=Ty|2@s+NjC0000000000!Vi5dWB7+aeF+;&GmnZu0000000000"
    "AtbN(JNbt|ofV7IyN8NE0000000000uqftH75axjZ+K;FM1YDw0000000000p{$)h@A`*8ybo;_&3KAH0000000000AU{#a$@_;uG`VEbRdI?y0000000000"
    "UvGT!r2L0Kn^F&<-f4<J0000000000i_DtDfc=L+r@1eKXJU##00000000009kZiBUH*qa3Kb=J@LGyM0000000000`w3iDI{$}2D;e2<c~go&0000000000"
    "orUG*7XXMrIg3D-0!)fP0000000000LF8Q5^#F)Kv!hSBi$sb*0000000000CFI@<)B%V<4V(Jo6g`SS0000000000h(m~IvI2-e$*DCIo;8X<0000000000"
    "tH5>9kOPQ7UkLR^CoYOW0000000000${EBeZv==yNAE0ouqKK?0000000000Ck>ETO$CTR1DDaKIUkBZ00000000000a^AwD+Y)_1vuBw!WW7_0000000000"
    "l)Dqc2nUEj(+V=~OAv}c0000000000Bl85t<p+pB;YZg0)Ch_|0000000000?7ikL!U%{!s3*7qUIB_g0000000000EZWuJoe79QtK!S>=J|<00000000000"
    "CI;`ncnXL>U$F(sZ}N#i0000000000764DMQVWPcMa)@>`09y30000000000JcbgMD-4K0&1<|af#Zol0000000000*H%_l0}Y5kf5;)U3EYW50000000000"
    "C@?qD*$s$5(;9vklGKSn0000000000Y|zp0t`3MmJDbH-8qJA70000000000;q^k5f)9v5J%a3OqQ!|o0000000000%3i0VR1k<jP6+2#D!z$80000000000"
    "WXT%<B@u`~>}9DFvbKpp0000000000?(@8k^AU(ZlG7)jIj@O80000000000rktlUz!Hc+w~1Evz^92o0000000000&Vl&SiW7)H-M}dpNT7*80000000000"
    "qhZWYQxu3md9juI&X|cn0000000000Wcojm85M{?36zebRgsB60000000000QVMW*-4%#H2%!iI+=+=m0000000000txtLCo)(Be^bq(iVuFc40000000000"
    "u84TcT^EQzLmW{I=z57j0000000000oO6o*85oE_ar+0LZF7l00000000000vd+tD(-??A{^k(p@@t7e0000000000Em&Fci5ZAMTNhy7cVvk`0000000000"
    "P|i<xJsOBW4#nw${9K7Z0000000000TJOu??HY(cjwqthfK`b=0000000000iFxV~n;VEgQ+j>J1Wt)S000000000094pA~L>!1f-tuo@h(?J(0000000000"
    "Q^J8;>l}zcq{?la3qOfK0000000000Z2dDljva_VC^~C(jyH)w0000000000hQf(-EgpzK+Vy4W5HN{A0000000000B-szl$R3D5yI-^Sk|>El0000000000"
    "{9NVTUmu7-`(#|B6d{Q~00000000005Jo*g@*jvmjhemhl^KaZ0000000000U5rYNe;|lJW0f=D77~d-0000000000>nA{22O)?+aOxU(mkNnM0000000000"
    "xe!r2i6MwUrY_LB7Xyhv0000000000%RaEp1R{t)_x7)=m-~o700000000009;%R}c_N5FRoE6z7xaig0000000000xJo5h=OTzev}a+Zmh6Z?0000000000"
    "ni!#fO(Tdv06Ofn73GLP0000000000#S1sGt0Ra&G2|0~l--Cx0000000000I~*E^0VIe(Gy60O6V`}80000000000|3Q6~P9%sx0TirOk<N%f0000000000"
    "5_OYwk|c;ghr`uq566f=0000000000cdd`R%p{0Fx1J6}jlYOM0000000000F5{#D{UnG$g*vnA3b=?s0000000000ISvNiA|;4G;42_wh_Hx20000000000"
    "oF?YyJtc@hzX!6d1*wQY0000000000RU~+}O(lpx4`ap7f}w~&0000000000YX9!wQ6-2##?rRQ|CxwD0000000000+OK{7NhOFt*6ZS-e3FPj0000000000"
    "s4Z8fH6@5ZD?_GQ`HF}@0000000000{}A_;6eWm26DLFZb%TgN0000000000*gyX}=Ol<gG7)5y@_UFt0000000000Aj?HMt|W*+G=`HvZ*+)20000000000"
    ";18lQY9xq2<|AAd>}-fY00000000008f)vu8zhK835ZB7X=R8&0000000000-8uP~!y|}5dQjt>=3R(D0000000000D%qk+Vk3w^^cuH(WLAhk0000000000"
    "5T?I<_#%ivPlZn8;!cP_0000000000kNqaph9ZbSP%Sj;VMmBS0000000000wLGxz4I+p@%CPdN;XjB#0000000000hQKm%jv<IZgsl`cVmOFE0000000000"
    "(+39P2qB0-bl;ny<1mOo0000000000jPnU3eISTHJCq9GWhsb20000000000Sgv2<>K}+ekW&!W=pl$e0000000000ZP*gWQ6GpvRZm!HY#N9_0000000000"
    "{t+*Hu^xy(VY!We@DhkX0000000000MEk$12_A?*j>B92bPI?;0000000000MDtDkSsjQ#S|Jn6`2&bR0000000000KC;Xwp&W=n3jFE}fBc6)0000000000"
    "{Bp4^+#85MZb@lX1@(tO0000000000P+C>v2OEe$j3MzwjqQg(00000000003XKsRAsUE4cbw&Z7v_gR0000000000_b)%MBpHZ6I?}12q~3=>0000000000"
    ">Ss4@4;Y9*-rAr`HP?qg0000000000yF6)m-WG^JX4|6|%Fl;D0000000000$jRffkrjwQb|5fJW5|a<0000000000jSO3WDinx7myWFj0l|kr0000000000"
    "Dw}^xsuGAm>-i5{qq&Db0000000000z6%0=5)p_%jy6keO0tJQ0000000000Wb~4OWDkfyqZm_D_Ns?K0000000000HhfpVp$&*Y>##a0r=y2J0000000000"
    "-mgI8$_t1<4-a@WU7UwN0000000000IVfbu-U*06c$wZJ8I^}X0000000000iDhj6+Xsk15Jd>G+l_}n00000000004d??O!3Bswy#|k;rG|$<0000000000"
    "2QbySj01>3WYY|(cYlXK0000000000x4nvEJpqV7@1GltQh0|z0000000000ToKQ&(f)@(L19$}H*tqR0000000000H>*rCNc@LDMGREcC2EI20000000000"
    "iVLoOpZSMC-<7epA7h6=0000000000k+PjA*7k=$^zsL2C0vI<0000000000VTE?Y>hp&{fmSA`H&ur~0000000000QcqrS-0+7$Ka9XzS5AjO0000000000"
    "3QnYetL=wCinjB~g-3@#00000000004<mbQSn7vB5N<R=#6X8Y0000000000>TeaV<K~Azh)eer6gr1M00000000009@eG8P2-0^py#lXb~A@S0000000000"
    "G51~}m*0m#2iw&P?JS2t0000000000j(Jx?z}tsFJ4xTZcqWHH0000000000<(O1-$=8QKFt8Z`8X$*20000000000R+snbveSn^2|yBs&>4q70000000000"
    "@S0B5e9wnKj$Q-$n-hmX0000000000^qK2vC(DOGij6i8d<};{0000000000)KS1$v&M%&{lDROa0iD#0000000000On*@_BEpA2VC%^9cman%0000000000"
    "#`+kSb-jl`fCY;^mHUQ30000000000w49JDuegUmFa*aF#r1|j0000000000hCMQq(6fg?KiwNN2=InL0000000000t+jV?+pdQ|eugMqUh0NG0000000000"
    "mEGED(W!?(w)Kli$K-}U0000000000w2jptw4{eXzaZtgLf?i!0000000000rmwmxg`bB&!ZrnI(AkDS0000000000_u}n#MVf~|c<o*JZ`6iB0000000000"
    "Y08*#^^}J|k7uO%AkT(C0000000000Q&sVZm5+x&Mw=b{;>m_U00000000004|6RgCW?nZ;C>E`wZw)%0000000000_!5J%r-X+<pa{G?n7@WV0000000000"
    ">qj}98h?jCFvwS$in@kC0000000000ASxC&fO&^NuH(Wii?xP80000000000<BD|5+;fLOBwB7Fm9U0D00000000001SqCHHExGMjNGIAqN|2L0000000000"
    "Tfh8oj%bHK<QuI#ucn4T0000000000YE0SJ>|%#N?8t-Pw4sJT00000000008bZ>;P+f;W`=!&evz&%N0000000000%VxA4xL1cj)34x1u9t>D0000000000"
    "aAO9@AX0}wvYc2rr;>(100000000002;okph)jn-*COd1p^b(>0000000000l86+T?M8<{hxk8|o`{A(0000000000{{r5EO+kl1`qK(8qJxG&0000000000"
    "QTZ`QraOl~d29ELu6~9<0000000000fE1U?^)-h;J$184#(9Q70000000000vU0FdIxvSoa9dn|>vV=d0000000000FxNEGb1H{G&^Y}}9&m<00000000000"
    "3J5Q^q9lhv*W)ZhUTcOx0000000000<Oix{#~z13HR^%$s%C~j0000000000SlO^f;uwcOnc0%>17e0i00000000002!Bq(^Ad+ZvOo67XkCUt0000000000"
    "k^7B6{0xUcE~-Ki*;s}^0000000000mr;5t{04_WyrvFXR8)pQ0000000000vV$Jw^8klH48!m=+E0c+0000000000gOr8a<NAg`%Zz_BYD<Pd0000000000"
    "t%kaE%=CsptC3OX14o8H0000000000%3R18uI`3FT3`L)rb3250000000000dl|YfiRgwvgo7EiQa*-20000000000V85CnUgCy8+KW8$20Dg70000000000"
    "6H#?mE!>7c3I_cu!!?FL0000000000JJD1&_tl0#$ba$hhB1af0000000000ZbC#Uyw8R}xrVUSQ7wi*0000000000QKwFqe94ADm8j5aBPoVJ0000000000"
    "f&zc&IKzfO1#{>@`y_@y0000000000+YH@=@w|pWiH?!%*dT^L0000000000Lh9qernZJaVms*#yc~u=0000000000F~4%XSg?jbAWe>aq!@-k0000000000"
    "n<EH|2daiZ_~h#xk`#tO0000000000X|Ezav!jMUBqVLkgb;>60000000000j`~3ATb+hL<tu$~d<=#_0000000000`JCaR0hfkAaUF-}cnF3-0000000000"
    "kuHOKq>zR{0vh%cdIW|*0000000000PDiANM2dz$)j~Dbe*lI+00000000007=Ods;e&=iBe|~<iu{E@0000000000;%#4ue0_#ME1uA&nD~W20000000000"
    "n5*`q6nBO|BohO8tMr9H0000000000D9{H%s&IxtN;$G*!tjMa0000000000hjS5gJ!*zP*)d5^-Ry-x0000000000nJtn{&SQo^2p%&T{OE;10000000000"
    "R+b3lTwR7h79ZQA9_58V0000000000rA6~e=~jk7JyxvHMB#-%0000000000dehL7b5Mprw&W3kZrz1I0000000000#n&Iw`$>jCzw36Eo7sgx0000000000"
    "ZR%bdf<uNtkc>w1%hiQI0000000000X`>8+20eyAW$5d5|Ivj&0000000000o|>l_i8h8nb|BIMHqM1W00000000001!%7I2{4901Xg5vZ_0&00000000000"
    "h@+n=i7AFaM5+zGtHy;u00000000006y^EI1|x<*bv`+->cWLU0000000000n#4Qjf*giG&NEUkEWd?700000000002HNA7{S}5ljjO{SaJz*-0000000000"
    "OucaKb`OR??Z)_Fwzq{q0000000000QU+oE?g)lJC8u21{<DQZ00000000004ox%LWdepkbZuH;Nw9@L0000000000Z7F7$+xvw;6FXhum8^w80000000000"
    "S~GY_Q1yjCJTcHX<fw%}0000000000$pYvB#O{SaE^!`7G^K?=0000000000qc1eoHt2;w9UQasg`tH&0000000000hq!yzsNsb`2WfX2*`0+z0000000000"
    "*D-|?8ry|HHCkb&Et-Wu0000000000_@Ee;iqwTbf|sj>gO-Ir0000000000PIl#I`ptzvj5M*P+LDDp0000000000JtCWuX~%^?^+e70GLMBo0000000000"
    "G8O3`*uaHAT-;h`jEjXp0000000000jK6*vM!JPSUk<h7=7)tq0000000000wUG;nv$KUjo$fYULWG4t00000000006#PDTAgzT!xsW@&o`8iw0000000000"
    "32(^ojHZP^P6!_L`+S8!0000000000|5{i4_@9M90BJe&S$Ty(0000000000TriuYWSNCPYR2fNxOIg<0000000000g<nJS&ys~eFEoe#7jlI_0000000000"
    "-&p_gI*f%t?8R^9cW#A10000000000)Qo<rrG<q+MvwhX*lLA90000000000$ZK#W5PyY0*rSUhIcJ4H00000000008)k%WdU%CEJr{LVnq!4P0000000000"
    "JnU(Y<Z*>S9Q_p1`(K4X0000000000jBe16PHTlg6lunXU0j7g0000000000bzNtAxMYPuz;5JZzgUGp0000000000Tq)jIB3^|+#v#gAAytJy0000000000"
    "p+e*|j8}y~y_M2VgHeS*0000000000uo)>B_E3dDOWs&G<xPb^0000000000?;czXU`mBR4iC5jM@of20000000000!on;Y%0q=fr8`TUs78fA0000000000"
    "kabE<G(CktuXk<p2}6ZI0000000000!KY-eo;HO*&DMGLYCnZQ0000000000x5+9<2{45~o@~pL%R7ZY0000000000+Dl9#bSZ^E!A!lmD>#Kf0000000000"
    "lAd@+-y(%T){3fDi!_Bm0000000000Kr&y!N*skidOGe{>M(^s0000000000QFpJIwiSgyQg7p~NG*jx0000000000Cf*p@A`gW?`aI=ErYVI$0000000000"
    "CD%%fj|hc85w}hc115z)00000000000(`o-`~ihP-{r#iULu7+0000000000p|~hvYWsvh>;hQ;x*ml<0000000000n6eLr*z|-z)#_p=6dQ#=0000000000"
    "R2AtXNA83`{#-L}Z5M?=0000000000O^hx*w&#RE&=?)r#S?`<0000000000_`+GxCE<iXwB2%M9T0^;0000000000#AM{$mD+?r4{L`QbPR<+0000000000"
    "E*(f>2GoQ=PJpfO$q0o&0000000000rkDO-cg=)A*4#Jn9tDL!0000000000q<?|c=*EOV5kPqua{+}w0000000000nt)2eTEK)rXKBQ0#{Gmq0000000000"
    "`bE<v&AEg?Js52082W@j0000000000K6{n_KeL2D01U&DYW9Rc0000000000*vkq}v#f+b6a*Y=yYhrU0000000000KI5ztC#Hlz<G;yr3-5$L0000000000"
    "<81Xzo1cV0)czi;TI+;B0000000000J``U151E8ORe)|4spo`10000000000!h~QOg_49o#sdqp_Tz*=0000000000;c0fS`iq1>kw`~{L*Rr!0000000000"
    "7H=80aD{|GBo20skKBYn0000000000(Lcj3=YE7h;uGr3+Sr6Z0000000000h==VMUU-B+Ga2?oCDnvL0000000000uJI9c)p3MCf8Ff%ZqbB50000000000"
    "yD6$aO>2ZeGaw=4x6On=0000000000Bik=~#bks)vY_P+0Lp|w0000000000SFs!>K3;@CW*&`oN5+Ie0000000000&{yg|w^xKfxL9f&j>3dM0000000000"
    "0eTY1Fj0g+4-e1-)V_p30000000000V8af+s!D`F)Bq7J8M}l)0000000000Uf)<DB}9ZkZS8-VUAKfl0000000000b9rj<pFM;?MwD+#p|gZQ0000000000"
    "6g|By8#jbN#k4X(<gbK40000000000vDM1emN0}sP!1e_Cai=&00000000009@`A*5-Nm1)_HyfXsCoh0000000000ro+5uk0XRYfDi!IsHB8I0000000000"
    "p@Uyd3>}0(pFT13=%9o^0000000000Ux-?pi57%F%U#P~D4m2r0000000000I6o6l2M~lnrN@UJXPJaR0000000000ffes&g$aZ}%GDn-rImz00000000000"
    "jd)Yx0|SIW)8R~=;*o?v0000000000wbiA(g8YL(WJSngAC81T0000000000OPWjf0rrDH6<SM$TZ)800000000000pfjqEf$xJrgL#h!mxhEu0000000000"
    "5f3MX0qKK4P5n*m(Sn3P0000000000?wKdkf#QQe4?jRQ41a_`0000000000jX@E60^EZ@XP6)8MSFxm0000000000L~}`cgVlpT@M5I|e|LmH0000000000"
    "W}G$81kZy&O8AP7w{wI*0000000000ODa2fhRB0J5weMh?{9=a0000000000LyEt12*QIv;OScRCTxU20000000000rYR`Eio1hARr$BqU1)?r0000000000"
    "%5~aX4Yh+n4mw&glw^cJ00000000000^B=AkgkJ3s0I56%3y>*0000000000q{*>*6R3kgz_n9b|6GJX00000000001gxR@mZ5_{@}nIuG+Bf|0000000000"
    "c<^w;8k>Va+$j=<XjOzj0000000000Qv`+6o|J<?8lmc9oKb{80000000000?o>l5BaVYWOk-fk&rO6t0000000000m`@SIr-y?;3Cb+F0!oBH0000000000"
    "tYK82EP;bS_AUuHHAaL$0000000000efV*_v3i3*s_U*tXF`NP0000000000U4hRSH*|wQ#aY+tm_CF+0000000000qiNEzylsO(;nF)K$~uHV0000000000"
    "r5@mjLT7_Ops<zp`Zk0>0000000000whdmH$Y6s&noOQ#D>H;Z0000000000FQl>JPFjONZvYQ*TQ7t_0000000000Ur#_|)l-8&xh0_<iz|dc0000000000"
    "pL}l_Tup;O8Ws+4x+jD{0000000000K$)Zi<3@u(Dk~3Y=_7<d0000000000of1VLYCwZPiN0M47$Af|000000000023EFh@i~J))G4!8MjV7d0000000000"
    "($HP%cr$}QrpCxpbQpv{0000000000SMZ>e04;++pzc=yq7;Nc0000000000<{{;3h$e$TS^F7c&k%$_0000000000*wT+{5Fmp<bVkT}{0xLZ0000000000"
    "fN%ztm>GjWizysUDG7u?0000000000G-$5nAQOW?JpRtxRRx4V0000000000N+<1GsSJZaBp#Y6f&qj;00000000005F04=F$RM`-QqSmuKj~R0000000000"
    "<gmoDy8nVe38r2R+WCV(00000000007FH;bL-~S0Lcz+Y1@?nL0000000000{dc&G%<_UjCXO@?G4g{z0000000000>x0^+RqTR5QD#*@T<(KF0000000000"
    "Ip_1--sOToVWZ|jhw6hs0000000000IuB|sXy1ZB^`4;>vgU(80000000000Kxv|u@z{burLsT1+~R{k0000000000r(yyZd(whH6F7Q42j7E00000000000"
    "x@hmL1k8d!*N6(4G24Sc0000000000+UU+!j>UpNmj0>GTi1g?0000000000Q&0vI7{7u*<nj^dh0}vT0000000000d<^^Fp}2xTWWW32ug`-(0000000000"
    "s)`c4E3$$>u)QF`*~^1K0000000000GB~oOwX1?aXGc1S1IU9w0000000000Y=S?VKc#{|B(?!MEyROB0000000000sMa5w$)18hhGQ_<R=|Tm0000000000"
    "K%%&#Q<#E4EUABJfV_i10000000000g;H;`-I0PouHdNZsknnc0000000000%v7n!XN!VBt7Bzi)3k#?0000000000YZlDm@q~gvz>99u{IG*S0000000000"
    "w*)=?dwzmIin4t@Car@&00000000002#sMG26%!%sxeHaPpN}I0000000000t{k5<ka2=QxMGtJd8LCu0000000000{vcaT8f$_;P@fEZqM?I80000000000"
    "QV08Cq-26X6+v<C%bkNj0000000000`8?!sE?$B_o}qea^_hb}0000000000OfP(RxL1NejT++b9+rbZ0000000000ol7WpLQ#T1dj(;HNRop<0000000000"
    "M*mG~%u0el2)wW<a*l&Q0000000000n72|>Rz!k8(3gthn~H-#0000000000<>8wx;5~vsZ}O+H#fF1G00000000003w@IKX*YsEx^cXj?}CFs0000000000"
    "9q>G}^Du%yfiZxb8GnO70000000000iJ^I2eJX-L=^|3PLwkcj0000000000f1ft@1|)(&n=*LwZFhq}0000000000FZf}AkR5_RK211Cm~(?b0000000000"
    "(YfjP85e>;dSBSM!f%5>0000000000kn+1=qY#2X^HrWj?Q4TT0000000000ps>cbEDC}^Q613k7ifb(0000000000I0E?nwF81cKBBX<LS%zL0000000000"
    "h4no%KmCC~V9}13ZD4~y0000000000yqJbp$o7FiUYDnrm|TNE00000000003;MrmQSgC4+~7mC!&rkr0000000000rzhBK+UbEnhnlJI?o@+70000000000"
    "zc0OAW8;B9{;(cO8c~Bl0000000000iXdG*?A(Ds@DS<2MNNZ10000000000H3+5)b=HAE0jEw_aY=(f0000000000`1FOy{m+3w+Fz9ookfE{0000000000"
    "|Mco}hRK0I9w-Ci$U%cZ0000000000eHOzF55s{#cI@5N^gV+>0000000000r)T_?m%M>Mi<xWbAUcCU0000000000toDQqAhv-(103=kO*Vr-0000000000"
    "!O==_sIP%Qg<!XEc`}1Q000000000076&}UFsXq+xKjb;r7nX&0000000000+oVPXxuStULgE#A(kg>M0000000000M!8rtL7ag=&iFbo|0aV#0000000000"
    "hEiZr$(4aX{)4s$EF*(J0000000000(0)r?QICN@dvMMGSs#Ny0000000000QcMU{*@%Ha?yATeh8u%G0000000000L&=0YVS<4`{U7UCvloLv0000000000"
    "(F3m!>3e}dPrQ4$;1h#D0000000000EedtfaCLz|ifq6-4iJMt0000000000j_SmK_-=tfR~`S~I}C$B0000000000BD0JgfM|h0UAuj#X$XTr0000000000"
    "9&);;2V#LhLo<Sumjr`A0000000000ur)j%k6VF2u5nVN#Q=jq00000000004GaK;7FB^jNMkhE^ZbH90000000000X9M)rolb#3w)AZ<BKd+q0000000000"
    "?I{KoB}jolpAto_Q1ya90000000000)r}-UtU-Z5uBrb{fANAq0000000000qDm%PGdqDmLP1FsuI++A00000000007zqk!x-@}6aDjaE-06Zq0000000000"
    "(W?DzK`wzn?i=3Z3+94A0000000000EEg|x$R~k7M!2%tI^u#r0000000000lQqnJPa%Oo2L3+FY2JcB0000000000WbFc!)*698yp+eQnA(Cs0000000000"
    "2POf$T@-;p^&^pS$JT;C0000000000-p2I%<PCv9K2zWk_R@kt0000000000N4E=FYzKirA<FrIB+r6C0000000000tlo3n^8kTBFEApvQp<us0000000000"
    "Z6sTXd-{Mt@+_XFfX9MB0000000000?r-mA1N4AE`De04uET;r0000000000kf>gJi|v3w&WNLl+rNT90000000000y0zBK6X$?H0XMd12)u$o0000000000"
    "&3=VpoZx^!9vUyzG`NC50000000000EbEv*CE9>MvDs3`V6=ij0000000000JmNlWuG4@(NF8P~jIe@00000000000TuY_!H_d=RZj7xCx2%Fd0000000000"
    "_6_>a!N!0<w9ai#;i!T@0000000000Y7uJ@Ou&FZqKoV23Z;TT00000000007Ck1r*13Q{$7*ZPGogY&0000000000XLwoDVY7fhvJ*1>Tb+VH0000000000"
    "wyH8<?W}-6>Mew0gPDRr0000000000Z@b@Td8U9s0%4~3sg;620000000000@R5%21)zXHg;d?K&yj*a0000000000o)v8XlbL`(0I+R@^o@c*0000000000"
    "-Yuz}Ad`SV02`cT7>a^G00000000005W;ZIu8e>{5IwC*JBETl0000000000n7Q$oJ%)flzP;rrU4nu@0000000000*ZWxm&3}MEm_&;8eSU&K0000000000"
    "H*nVzU3q{&D|z3goqB>m000000000082*>g?Q(!X0|P2Aymo>>0000000000*o?_Pe{6t2tck9M*>ZwF0000000000;scsz5oUluw6w9I^=^Vd0000000000"
    "lS(&Eq+ftQs4IAN5Nm=!0000000000Q%KMGH(7u{6_!T#DQAK}0000000000g5-~I%u;|rhMQp=LSuqJ0000000000h6bm&VN8HPjN=X3SYLub0000000000"
    "z?DMS_eFp}uoaLcZd-yt0000000000lX;NUjz54vfY~hcfmeb+000000000007}BPCOLpW9q=hTlv9F00000000000tTNVuzcPS8&lL*rq)&oC0000000000"
    "o+$o7SuB7+?ZvAIv`d0O0000000000w2OuO^Cf^lWe@dw!AF8X0000000000&AYd{j~{?QCjrbn%|n7f0000000000&KTf?EE#}6A*vQc*FS<l0000000000"
    "lmbd)$`XJ;Lm9h<-#dap00000000000@*-bX$*isdtIOU<u`&r0000000000_WKWh2?l^bwuPyc>NA2s0000000000u9gLsRsVlL>7c(&?Jt5r0000000000"
    "0Zz%9n(}`@{k_K+?kj>o0000000000rriM>;^lup<LjsF>?eXi0000000000WCcbiEZKiRi8)5U=_7(b0000000000`h+adc+7u5-L;5@<R5}S0000000000"
    "DjZBE$iII;(r;Bb-5Y{H0000000000x{5nG8nb^uRCbce(-(q30000000000WRQ^@Z>E1hQP?Fj$P<D<0000000000?d;^;#+iRWx}^1Ix(|Xt0000000000"
    "7hueeA&q}PeCp?3stbZZ0000000000r>WjQe}I2Lhq&Smm<NJD0000000000REe?y;B$XK$Q{gTg9Cy<0000000000?Y`dRLuY?LGbxH|Z2y5k0000000000"
    "E(-ejs#<?QvFQQ^Q~QBH0000000000)GC`%6;6LZGj1F(H}`=+0000000000sNh)qf<b>ksl~kC81sQZ0000000000W$Vcb^fZ4!0RPDf`0jx~0000000000"
    ")(VA<XefU`Dk&<b)arpi0000000000wf-QL;2VEH5>vyfuI7P20000000000z=kw3Tn~Rhsf&9HhT?%h0000000000#9~%;+X8<;-r+dETHb*`0000000000"
    "cv2m2UiN-Ko@C*>EZTuU0000000000rWGJC<>`Jv+9F%``_+L!000000000032gPqZ{2=CfGdn=$kBm70000000000Y85m}{?L9ve~nrKlFflY0000000000"
    "htYDxki>pK$5*VqSjmAv00000000009AHRDC%Aq<M44xg9L0e^0000000000^?*y}!K;2i>0okg-N1oB0000000000#`6-dV4r?Kp1rb7o4kQQ0000000000"
    "V|X>Q1CxG0TfPn^R=9yc0000000000ea=Y(tA>6+2y$%f4YYwk0000000000;kD+kR(gIwm*=sT!moiq0000000000O75b`25x>p^)1C3bgO|t0000000000"
    "csCVcyJ3Dn5>$+CBBy~s0000000000I8oGwbX9&p-)QZJ&7y%o0000000000KKph{GD&_wMtja&b)JDh0000000000$kc?t^gMn*BDo;r8JdAW0000000000"
    "aJ8cAyD)x0q-&uTyOn`J0000000000LJ7Obha`SLTH>AlSdoE10000000000B8b{eSQvgl1DwL6^NfK&0000000000-02N+Eew7@RTsPti->_h0000000000"
    "RpwzV2LOIR64N$FAB2HG0000000000T{Yu#<@0?&@lD@Yuz!I-0000000000){dS($me}Pu?AvEK6`;c0000000000lB%GFuiJe<4SZ`N$##K20000000000"
    "XF&Epo6mhfzAUKhP;!Al0000000000I|!^|i^F|DhnK{#)op=50000000000*}2Z;f46-=8ar)wS80Jj0000000000A`NT6c&mLtJC;sK)?|S|0000000000"
    "@wtuZb)S7eqbC(FQec5V00000000008hArkcawcU4Q_2Q%Ugj!0000000000c>kCKeTRKOH|PUdL05r50000000000;0K%HhI@TL*c64Xv{HdU0000000000"
    "Hc3|Oly7}Nv~ry~B~F1r0000000000M|;;7r(=CUe}MS~lSzR<0000000000^i?Zdy;pre`MgCL{zQR500000000006p>!A*Gqjs-li^nW<Y^J0000000000"
    "d*po!_CI|<=Dc7!%sYWV00000000004G5NG7&LuA(|w3pFE@ce0000000000m$S37J}7-aT4++^k1~Nk0000000000{hs6OXdHb&J^O0l?Ja>o0000000000"
    "6M`}dmk@nGHMM?ONhyIq0000000000tQ4&Q$pn2s{1iG@q9lPp0000000000unu6#{`h=AQkhBO_#c5l0000000000<|kNNIPH8u?!Z|CO&ftg0000000000"
    "I2fX<b>VzKk_s2Yo)&>X0000000000b(c!Aw$*$<^z<z%?h%1N0000000000d0k#+`^kJj)c#6AI}L$A0000000000AR_9%L%w`K?4}wKh6sT`0000000000"
    "I-UBmkFtC~_G?L^&I5r!0000000000u7;dB-==&(w9JG36#sxg0000000000N0!`4GMjur+DCRPS^9uL0000000000<%7rCh>v_gC|u1noArP|0000000000"
    "Wd~~G;)8rZTZlCx+wg!u0000000000m)Ou+KzMvWEK6Jb80>&R0000000000T(1Z?p=^9WStIMqROf&|0000000000hC?251z~(ZoR5*2j^cno0000000000"
    "^-LZBZB=|gu@3uf#@&EG0000000000euEX+*GYUpRv*++{Mdj%0000000000-0}(ZMLv8$YPrWvFw}rR0000000000;NW#ew=#S{#zZD+V$Xm;0000000000"
    "Wxa62D<^zF7h|-plFEQU0000000000ue#k`qZ@oc(ZMPs!Nq_;0000000000@d}^_9uRy$XbwWp?ZALQ0000000000ap0P^n*@A7QueaC7`%W$0000000000"
    "Xl=1t9Qk`d0w*@@LAQWF00000000002;|1+pY3}<>P>`hX|jMo0000000000ns&ZWCgOWQf*4jtkF9_}0000000000P*;11uhx4&J2Yf`v#5YT0000000000"
    "ZMfipJIi}Oj8115)}w$x0000000000Da_F?%D;O+=oA<t_nv@20000000000!o)F!TeN#X!!$3U7Mg%S0000000000b=2c@@2Gn~lUrq-G?jor0000000000"
    "b}h^Cg`Imq$To8xQILQ@00000000002?ERR9+G=N*>e#rY>R+E0000000000W`xFGx`ul|Jiz$Pg@%AY0000000000&gP&rSbKXwW69wQo`HZs0000000000"
    "c3VAT_-}ha!aRlTw0wX-0000000000oD_fEnq+%G%*UI0%6EW30000000000d_+l>KUsS~`4`^0-g1CI0000000000PaS`N=1qG*y4nP`@oj)W0000000000"
    "TG0cKk3xGug}umg18IOj0000000000&yu>dIyZYj&TZoR6J&ru0000000000@RNq*=PP?a37HC3BVT|(0000000000`36f7mmqsUs_glRF<O8?0000000000"
    "9v<s0NELfPBo=9(K2?A~0000000000q6<nb`w4qM?!pkBN>G460000000000z5z=Du>N{Ld7!C{RZD<C0000000000v@+43X7hSLK3wc|Uq*mH0000000000"
    "xX{1&9_V^NtfP@$XhDEL00000000001rGZE*xY(RIV4g~a6EuO0000000000-yCD8l+k)XTZ`sTcQ=4Q0000000000crV`MQO0^egN6raeKLSR0000000000"
    "7kEyn5W9LnDZ_1_f-QhR0000000000>nNAy(64$xz3$QdhA4nQ0000000000Jan6nlB9Y-xVG$yiX(tP0000000000LX19{RhfD~i^z6PjUIqN0000000000"
    "F{5Mk8IF2Ds?jJ=j~ReK0000000000QS0rN--3ETjDNeAkQ9JG0000000000)0V(+rgwTkpnvBqkq>}C0000000000{MWv9Zftr$Vt-{9kqUr70000000000"
    "1mVk=H(`1}KrHTFkOhE10000000000BIL@<0akiIt}{9dj{tx`0000000000niHP{%}RPeONInU8ux!d0000000000U1u*dnLm0!>bd!0743gO0000000000"
    "obr%PW;A+0%o!PQ595D80000000000$3pm1G%0#OH)r>C2-$x?0000000000g}RYJ10H%nqIe8;0MCCw0000000000bi9cb(-L|>ge1^%_r!ld0000000000"
    "EvpR9qX&9GM?t!7?Ye(J0000000000Q|U%&b^LiiW2HiD;;?@}0000000000e%WN@NAY<;M<?NJ)~0_z0000000000SKg998|HaHZJiBv$((;c0000000000"
    "a$5a(@7j4lK;A!xyOMuE0000000000c(Bln#Lsy^J;3jytcZU=00000000000Yf=@n#6fP&KZ@?oPK{m0000000000r*vaKak+UwX2GWkjC6lM0000000000"
    "5vY6GNUnK6e2t%1dTM__0000000000*FUaFA)|RffmHOmXJCIo0000000000o}MR*`j>e?>+BsiR91gL00000000000w)xm)QovRD=n1EKTLl>0000000000"
    "n>(O=uYq|$uj|fzDnfri00000000005<mDVi*|WH^ZrFo6gYoC0000000000`;$ncXKQ&tVLfp|{Vjh$0000000000|D3(#L|}P9bbZKK<s*MU0000000000"
    "vb=-sB2{@nmX*Vo%o%?`0000000000yj91$0ZMs5LtNVavJZbi0000000000w#u|k;XZjl?5Bj0m<4}80000000000J$`th!83V4{!Thlefxeu0000000000"
    "-ES|oq9}PlK@@^RVex)I0000000000*)-vVgdKT691iGdMCX1$0000000000w7~-kXA*fpPaby6Cf$BO0000000000k$B`iN(XsBntxq#2-AK*0000000000"
    "lDp6uF8p{v!felC>BxRS0000000000=j;`i67hIIl_343$-aI-0000000000t!@FY_vLs%(A5_}skDAT00000000007Ng)=-P(9SL&s1%h^l@-0000000000"
    "JnR6J!_Rm?sxR4{W}tpR0000000000RIIO4s>FCe*Gb}TM3#O)0000000000dqp~vk-2z4icp=kA&h=N0000000000*BS0hd9HXshkiSG{(*i#0000000000"
    "m5Ug6VWW6Jmk#d0+IN0H0000000000*8BzzN|<;+e>>)rwrzeu0000000000$j=4}GmUsa4M0`-kz;;90000000000jb>$J9D;a209S1IY*~Il0000000000"
    "NF#6-26uQs8M|DVM^Ao00000000000DQI*V@N0NLE2mfBAw_;b0000000000O_#rB+F*D<^Hq-5`#OF=0000000000;}W0u#8r4eJOVO>)i8cQ0000000000"
    "3_W1QuS$48$>wwLt|op!0000000000_jgdqnm>3zX+DJ@h#Y=E0000000000!zI1^g*13T*h7~bVG@2o0000000000-krEvaVdB}_Ynx|IS7700000000000"
    "d(CU!TpoBp`sUM(68?Qa0000000000{O4JUNE3KK)Y5|m>+^j;0000000000c3B2pGzfS=z^I^7!|8oM0000000000{*CliApLhh^E<P3o8Nsv0000000000"
    "q4B454DxqCn_on4bJcx70000000000eDI^V_~v&&`R3d|OUr#g0000000000mG)~P<=b~aH>CyWBEfw?00000000004{(F7(a?85lx;_7`nG*Q0000000000"
    "@yKmozQuPyJQFLy(X4$y0000000000R&A~`tGahUaA`U7sG@y90000000000QOU?En6Gz0UPJr-f0%th0000000000?MOgGg`{^tI&}HuR*ro@0000000000"
    "Nm!VFa+!BPKF_?RErfkQ0000000000aAQ>5UygS`o+r*h1bTfy0000000000d)D_^O@ntpkx;g|+HZY80000000000d>sb(I(T<LOdb9Wv1NTg0000000000"
    "growtC~bE@0AVgahg*F>0000000000sn>9m7Gifm>F&=>T~U2N0000000000`6ASh1Xp)JG@5HVGe><u0000000000k@Byq@=JF>A~uQz2|ay40000000000"
    "eX@t+;6Qgk-j%ek-ZFha0000000000)z02I&oy^IrHpeqv?zT*0000000000tFt(*y()J=s_jpsi5`7G000000000056u`otRHtkAB@uNUKD*m0000000000"
    "8;1hvniO|HK|6#QGzxt{0000000000*-d${i3xW=InM1S2>^XS0000000000X^jArc>Z=kP8gUL-t~My0000000000*0CFqXY+PIs_YBtvg>?60000000000"
    "F(*!^R_Jy>e;(4Fhv9rc0000000000m026uMcsBl4S{nwT-SU+00000000005VZy(HPUuKf#-O(FwJ~G0000000000xuuhYB*%6@7A0#L1;czm0000000000"
    "p_fJf6TEgn{S4D#*|>Z_0000000000-}ax81F?2MbaFY6t*(4P0000000000gD)mK^QCq`rYI++fuwvu0000000000pRHT_;+l3q&H^Q&RhoQ20000000000"
    "PTj-N(vNmPBh<TuDUf_X0000000000mn!1D!i07}(r<K7{e^r$0000000000n*ln%vUzqu9I4j?(R+MA0000000000WzJ63qHcCTFjH=#q;Y&e0000000000"
    "3jL%1l4Ev2LP5bPcxQY-0000000000p!vK#gIIPzjYe6UOI>_G0000000000H835Ib4_+YLN>qm9#eck0000000000<_t*lV?uU7o8x;w@<@C@0000000000"
    "y8FOsR5x}&#UU7T#XfvM0000000000&S(qrL@ahd0K^=Vm^6Gq0000000000HNdKiH6eCDe4P@YYbty|0000000000`)wjNCKh%;Y=MHBJ|KKR0000000000"
    "J>K>377KPj4MX*U5f*$v000000000027Rcz2LN_JjkWVx<O_U20000000000akjmc`1Ex^CaeY^w*h=W0000000000jrO>H>FISq4*Is$iT8Uz0000000000"
    "YpES}+TL|Qb8?M)T<v>60000000000COdC?%hPp0mgo`^FXMYa0000000000$rmw>yvTJxqc6~z0oi*%0000000000ajWL7t-W<X)f!F{)XsZA0000000000"
    "Bw`cVp0ag7TzwUPro?+d00000000001*kd_kf(J(b}Fykc)EK)00000000009qHdyft+<fQT3xQOR#%D0000000000gpr1%a*=gFCb4{V9j1Fg0000000000"
    "OAWI9V}^A=D>1^a@0)u-0000000000g~;V#ReN<nl1TmD!I67F0000000000OE0C(MsRgNmddsRl!tpj0000000000rLA*AIAwJ}WQunlW_^1=0000000000"
    "w6D~`DOz<vI$OvuICFbI0000000000hM`MH8&7pWOAs<R3Tb;l0000000000EALy@3`BK6%6vUI++KS?0000000000xde%8{y23&@N;u8u2g$K0000000000"
    "K~7c&@GW&f?8sapfJ%En0000000000-Zw|7;UaZF{Z4`lQb2n^0000000000o-N-{(-(C>Q`v0qBQ|?L0000000000lbfOa#0+&nBRi$d^(%Wo0000000000"
    ")$I|lwgGiOrAM@;$02(_0000000000aFDZPsP=R~3(_KcnHPIN0000000000f$s?(n(K5xkga)3YYlrq00000000006L*-}i{NxXVXZ+JJp+3{0000000000"
    "LX+Z|ebsb8zN7=<4*7aO00000000009k6R#Z_0E)*k(wn;O=@r0000000000vN^XNVZU@h-^Q43vgCR|0000000000ADl?zQnYkH5}185gxY#Q0000000000"
    "Y(t}{M5%N@o~?iNRnU4s0000000000xtmIEHlB1qzS<|aC&qd}00000000005)r~PD3f$Rr}F-K`MY{R0000000000h!2hU8HjX1i^z>W%dvVu0000000000"
    "Hv=!c3w?Azq%U0qou_(00000000000Cs-?f{c&_aAp7;nZk>8S0000000000cX|ar?`CvBLH^*9K$Chv0000000000GS)!*;9GP+Fbt<&5{Y_10000000000"
    "a@SzB(ol3jEBMeY<bHZU0000000000Nb2Zo#6@&KWR>Rtwsd+x0000000000!g^dDwmNh`4YzaHhiZC20000000000{PaG|r!RCsV=(@-SzvlV0000000000"
    "0>u+`nIv>TmDCTHD^_|y0000000000@m5|Kiy3r4;C3;8{Y!d40000000000*cPI-d=7L#cIxYG&p~=X0000000000!Ej<sZUb~cgaXK8p*MO!0000000000"
    "%+Mm-U-)xCO_EMwb1iy600000000001*2wUQSEa;`fLbiMI(AZ0000000000y$ZJELgI5kusL~k7a4j$00000000003wG&TG}m)LF^<fP=?;280000000000"
    "*o7R;CCqa`uS?{ty99bb0000000000CGBiA7r}Et+YO-Ajrw^&0000000000?q2wn2)1)TV{qRFVDNcB0000000000MvmO>`Koh31rLHiGUj<e0000000000"
    "BQ7^U>z{K#YlUZm1l)N*0000000000m+>Wv+>~=bQg3g}*3o%D0000000000nv2HF&53hBXlQvJsmFOh0000000000GOsoWzkYK-UJPY;d%bx;0000000000"
    "Y7dk=uyb=j@AeVeO|yAG0000000000JzxB3p=fhJ#lv4iAgOsk0000000000w^5Cdl3jB^q7E*!@}7A>0000000000)R|<kgHm%qDsx;c#gutK0000000000"
    "pz5E=bVqYQ8;#1Vmx_5n00000000008~0q?Wju30CHPk|YJhn_0000000000MRK|5Rxxux`%mP*Ja&0N0000000000Di71{M<#PXR4`9e4{Uir0000000000"
    "$AHJ~I2&_7-!ypd;bD0|0000000000C3>{zDG+l&Q~5rTv{!jR0000000000NUpcs8U=GeV;O@vhfR4v0000000000EYR-C3;J?Ezc3`|Swne10000000000"
    ";F5Q<{O)o<BHIC-EID~V0000000000Ua4=E?c;JlH?;d#|1Nnz0000000000v&Qsu-q><LyI1iT(j<960000000000+?^y$&&_f`T4BQ9q#AiZ0000000000"
    "<KY}1z`}As%xq4qcMy3%0000000000$wm9@u(xtR#uasbN(OmA0000000000jeOm;p{#O1^7j>19sGDe0000000000JUQWclA&@y4zCb1@bP#+0000000000"
    "&@Lo7gO+kY$JRs@!{>NF0000000000SrEkYbBl67+6t2VmECwi0000000000(~#b&WPox&@aEU(Xw!H=0000000000L1U3yRdsSezy|i&JIQ!J0000000000"
    "t2+7!Mrv|E0tU>_4!?Ln00000000005z02QHeYf;VSqWx;k0-_0000000000e#B%~CscAkoDm+$w5oVO0000000000?cAaG7fEtJT~%4khoE>s0000000000"
    "X~qkd2tIN^T<6NuT9$Y~0000000000^fSOO`7&}qO=D8uEsS_T0000000000j@aDA=_hhP-ek}20D^cx0000000000L1%wh+8lB~&4(cZ)OUD50000000000"
    "3zNm?$`Nuv!)GWNr)_vZ0000000000_72T+y9RPVe4xWKdSiG%00000000001&*`)tNU?4sLj$*O<8zA0000000000HN%#IobYi#_lh-hAWwKe0000000000"
    "lVOwvjpcDbBXrA`^F(++0000000000930ApecEwA+$40r#yWUF0000000000+ND(kZqIQ**f<aBnJ{=j0000000000%CS^_Uc_-g$4h({Z6<g?0000000000"
    "`10iTPPuVFV)>d!Kpc2L0000000000WgH}JKdx~=THQ)|6B2kp000000000058*H2FQailUz2OE<_CB{00000000000`w(QAeeDLD!1I`x&3!Q0000000000"
    "JP0Ji5RGv_Y2<w%jq`Uv0000000000$IwSH0fKQr*tRZQVCi>20000000000qyVp-@pf@QCCbj3Gv9YW0000000000(@J*;;%jk00lGce2Gw^!0000000000"
    "Ut+#-(qM5wB_dlI+RAr80000000000Mf#!H!c}oVL7oO(t-yCc0000000000ki|bhvr2J54ay&(fVOu)0000000000K6kaHqd###IYEZzQ>=GD0000000000"
    "Si&F*lr(Wbe4iODC!%*i0000000000;aQ(;geh@Ag@Czq`j>Y=0000000000)bLKwbRKa)2Y^Mr&5d_J0000000000MLJC~WfO5gy*Zr*p@Vlo0000000000"
    "D(SbARS0oFR7s#vba{6`0000000000kFc!tMg4F<g-R})M{jpP0000000000azjZ~Hu7*l`q(Ds8fAAt0000000000-&Cu$Cg*TKbdYH=?pk+10000000000"
    ")R*uT7u;|_pA6P~!BBTV0000000000RKGTM2hngqEa^+glty<z0000000000Xnvd4_r-8P*1@V1Xgqg700000000004TwB9=(=z~Ml#Y~J2H1b0000000000"
    "PLDU3*spLvJl3J94k&j(0000000000E=Q{Q$fR&UW=Z+;;T?BC0000000000t75WSxtVZ4cPF?#wG($h0000000000(3EYysg7_!BMiQVhzWN<0000000000"
    "n~?b-nuBma8<XnJTmN=I00000000007728Qig<898CV?_FZFgn0000000000K(YGkdTnq(&Gt210_%1_0000000000Av~*6YhrLf><Sd2)!=qO0000000000"
    "yj-KYTUT&EDlnJisMdBs000000000055@o_OiXY<Gy<O{e9U%00000000000CM)5NJV9_k#T*M~Pr`OU000000000006${;EH-dJjvKqBBDi)y0000000000"
    "q_m-F9V>7^MVgM{^{sY500000000007C*h&4Ipqpnlxn{$)k2a0000000000Rguz5{uFRPLWUkzoSAk&0000000000ZC?Ms?g?-}^eUQ+Z;y6B0000000000"
    "v9iK!zWr}NTfVozLWOof0000000000OKL1$pXYBtH{4e86?=9-0000000000o=JC9fYEP2Fd$bU>2P*H0000000000zP8>^VY_cY04A7Cyk>Sl0000000000"
    "u@|0DLZxp&TZ#F0k6d;@0000000000c+&7zBad%D@a0;aVp4WM0000000000CruJ#1bJ^jde@u4HAr?q0000000000wCz20<zjC@qxqZQ2tIZ|0000000000"
    "I~uQz#!PQOC+Ir;+cS1R0000000000yf~q(r#5duxjtAHuPJsw0000000000HLp+7h#+r31)0+;f**E30000000000z2o)%X$o&Z$C8XaRTXwX0000000000"
    ";p)pWOZ09)uaRU+C<}H#0000000000oGjRFEZ%NFXT(=k`v7)80000000000-p~544ajal^7(dK&h~Xc0000000000q5l&3?y+t_N{Q25q3m@)0000000000"
    "!sh=-(3@^RV#$77bmDbD0000000000Be0;Kv4w6xFcFGZN7!{h0000000000y@u5Ml5cK6v$qCM8qRe<0000000000Xq9?kbXjgd)Ds3n?Zb6I0000000000"
    "7zc97RYPt-m3)~qz`1om0000000000vj&<vH!W^J?MTufldp9^000000000062H-~7#D6p$j`wFW~FsN0000000000G+X#A`T%Y~DQJ1|IGc4q0000000000"
    ">;>(v+v;sV{b&l+3z2m|0000000000GwO~uz0_?$Ln(%}-iCER0000000000;xfI%p1y5B>i_+cv3zwv0000000000@CsK}fT(Ri??Z!agmQI20000000000"
    "Kb<@7VUle?NlYh2S7>!W0000000000t`tg(Lws#O=WqKNDqeL!0000000000HJo8BC1!0v$psPY{8M#60000000000s9s^z22gE4-DG*Y&q;Ma0000000000"
    "1&1hr=s9gbBjiPhqCa&&0000000000CqR}k$|G$+jM|4%bv1QB0000000000^t6fPs|{^H51&vPNGo+f0000000000ToB@(jrVLorHHZL8X<K+0000000000"
    "Kz)8(Z{ch}K<>Mu?G|-F0000000000kS0DJQOj&V-X?)#zzlUj0000000000CpQ4(GPP_#Zix~ilLB=>0000000000{@J6g6rXHB?QDVGW%zVJ0000000000"
    "^f?H7_K0jiPblA*IPP>n0000000000>wyeR*m7(@ixNFi3*>Y_0000000000)!dIDx?F5Pn1ZYa-r00O0000000000kacnRn?`IvYR^8ru+MZr0000000000"
    "6L*l)eK2f5|4#*PgT-_}0000000000I}FURUm9#cM_UmfR=adS0000000000>R$tsK?H0-I7oEQD6w=v00000000008J!h(BkpTJ&}~0``=)e20000000000"
    "pYGvZ1=wpq{UU@Z&YW~W0000000000bALxi=fP`0zP?k`pptYz0000000000Zh9p%$*XHX0+_#jbBJ_60000000000hYDLEtCed&#s%3WMt*ca0000000000"
    "pIXNejel!E0AI1p7<6<%0000000000mz7BbZ)s~lp!!*I>uGdA0000000000W9Gp4P*ZC_rZ40czF%}e0000000000;CBq~GCgZR0aanTkX3X*0000000000"
    "0d))K6envyt-<14V@q^E0000000000stBFn_7H18qRe^zH9>Sh0000000000yIr8$*!pTf(9AcV2sd;<0000000000C!h4!y5wp=G8ryE+bncI0000000000"
    "%lCNIoX%=Mya;aIts-<l0000000000o{|jIez<BtYw`$vfEaW@0000000000c$$vWV4`Y3FD(!oQx0@M0000000000O*-DwLX2ua0}z$CB?NRp0000000000"
    "1D)E`Bz9^*-tnka`1x}{0000000000cI{%;24HGHuqj>g%I|YP0000000000pH^1b=t*ioc@D#lo#k^t0000000000SOb*W$}?&}DEI3saNBc00000000000"
    "lpg!stQ~4Vx;p8`LeX<T0000000000In!$2jt6Q$Brwck6~}Wx0000000000Axy&IaPetCRxYOg=e%=30000000000OSItQQrl@jRE&t4y0UXX0000000000"
    "hBl?;G{tE^1fhgBji_@#0000000000)cx+}7O!bQZo{k6U!HS700000000001vQ=L_?T%xdrA*)GL&;b00000000006bFyz+Jb37ERDbm1&VV(0000000000"
    ";GU!AyliPeapKUZ*MD<B0000000000OeebJo>pl<L!nVds&#Wf0000000000P0O+2fIw+Lnjlo<d~0(+0000000000y0vNDVk&7sXSeBsPhoRF0000000000"
    "jS<1uL=<U2sk|d1B3E-j0000000000o(Y%GCjMwZP=^!3^h|R=0000000000=E%y&2<T`)S#HW=$3k;J0000000000LV7#B>d|OGx5!xpnmBVn0000000000"
    "lt{?1%)4knUhUGRY%X&^0000000000(_T=buBB)|N;5}FKO}QN0000000000+UK&7kdJ6UZ}G?O5gKzq0000000000nhwu?a(QS##eQa#<PUQ|0000000000"
    "`u&4xRAXpBLozfww*_-R0000000000*x;~HHce<i++srGiTiRu0000000000DdIFY7&mA@jAx>ZT=8;10000000000$t>>~`XFdPMVDnaFXwVV0000000000"
    "uIs+}+X`qv14*6Z0o`&y0000000000uStW}z4T{5w$oaX)Y5W50000000000#&H|4pWbIcT#m0mr^s?Z0000000000(~f3_fyie->Cde3dA@Q$0000000000"
    "w97+TW3p#JPw!)-Otf-90000000000U{jJPMVx0qi$msEAF6Ud0000000000wNNqiCWdD~kRwG8@}F`*0000000000r0)N_2ykaWS>}<)#FcVD0000000000"
    "5m?=S>RD$%+Tz`Tmy2>h0000000000+PA1b%tL2D25>q)YJqY<0000000000`Zz)Rtu1Fj+vK?XJa=+H0000000000MFxhlj~8b^N|<WB4{dTl0000000000"
    "wMtfHaRFyQPT^pH;$m_@0000000000A~BH@Q|o3x*cPotwODdM0000000000gya>yG}UH6=#O9xh)!}q0000000000x{t$Q7QbddZljFXT10X{0000000000"
    "rCoUe_^4(;U8q8&Ejn^Q0000000000K6w_L*^*{JxAl{A05Ebu0000000000Sq*wIyL@IqX5x`W(<O310000000000@I-^boMvV~Y{g9)rW<lV0000000000"
    "=t>bxeo$sWy!F`ecoA|y00000000007)k@tUpi($N6JLZO9yg50000000000er-5VK_q5C44R^;9{q7Z0000000000@86HfA`WIi`nv;)@$zv%0000000000"
    "XUyn41o&k@5{G$h#OQHA0000000000yImrx<>6&OKvES{m)>ze0000000000;l3LY$IE3vf^_ITYSeK+0000000000yc)D|sI_H4%tQn$J<4%F0000000000"
    "{q-ZviJxUasxV3$5WsOj0000000000ql+seYl&q*xKuz5<F#=>0000000000x`e2DOmk&G(!bOGwX1PJ0000000000F3EY;EnQ_m^40L|h@o*n0000000000"
    "(?EVY4@YG{3j2fDTbFS_0000000000defAu@GxaS29tTbE{$<O00000000006=|F*(i&wz=K;Q#0fTWs0000000000Y3Xymvjk;8nX5f$)Oc|~0000000000"
    "Wjy+6mF{Ff6!hdPr*3gT0000000000)=3UAci3b=Nk5?Ad1P@w0000000000rumK+S;AyMGVxo4Oj>b30000000000sxw<5JFH|tzik8+A5d{X0000000000"
    "w%<`w9hPK3;~0jJ@kMb!0000000000v*l5)|9@mamsZX6!#i<60000000000axFSJ;%Q_+$%@85mN9Wa0000000000+Men5#8YHIbs1i7XeV(%0000000000"
    "!XZEFr#)mqi)Mg<IvsI90000000000`<)pYizj42{M8wB3=?rc0000000000bo*C_ZV_Za#LE^%-Ux9(0000000000_aWIfQu|{-(5qweul{gA0000000000"
    "aQS;IHsxbL8QSc5f%I@d0000000000t7#u>8qZ@ukh-4PQtEI(0000000000lv<+s{<vd6F0o|~B;asB0000000000|Fh%R<Dz3g<smK^_0@1d0000000000"
    "zg@x&$&6z_rEui*#mjI&0000000000!ap6Hu6AQUY5{eimcej90000000000(m~UQlwe~(AUMn!XSZ-b0000000000+DpUKdP-wJ!+6$SH?43$0000000000"
    "utm&<VKievLnnZ32cvL60000000000Cl}43NFHNAlc%*q*O+iX0000000000E2yL?FbHEnu3cl_rjBqx0000000000hu|?H81iC3f30;sc7$+10000000000"
    "D5z790Ni3g1N+!sMS5^R0000000000*yiYT>BV9|C#rcq6mW1r0000000000hlJ49(ywAb<{m`b;$?6^00000000001!?KHyqRJ^GQA`-v0HFJ0000000000"
    "CVUbfr-NcZ`RE5ke^GEi0000000000)}<;9l5Ju@IR@GbO-FD*0000000000<joL=eOF>Y-*|_J8a;480000000000IozX=X+dH@=A`_$=Q40W0000000000"
    "skp)|Rx4saKJF!$v?y>u000000000021`t0Llt5`*SPN{fgW%`0000000000KzkoDG5=veuxLhNO%!lI0000000000Au$`FAL(I0v0)ZS847Sf0000000000"
    "l~f+H4%1;kwGWuA$^UOa0000000000Xw3&~{=8v8Fn*^MVC!!{0000000000goC!v?xtZt#nv;&_114d0000000000z+MUB-;iNIT;UbHio$O|0000000000"
    "?nd9H(RyJ((`6>}9<Fad0000000000`6Cx1#AIPWAuK*+vYBr{0000000000uicMVw@zU|BK$QEMTKub0000000000`dcVVt2kjm%bn)z)^Kk?0000000000"
    "u7<w=pdw*F|JW${XIyVU0000000000p+z)&l?-7(o^{nS_(yL*0000000000yntoFi}qkZpgYBqhcj<L0000000000&L1+=g5Y34;quTG79Vdw0000000000"
    "zY2M{ddgrxQ)AJ$qY7_80000000000UMU%JbF^SU(^Q{yF!pXh0000000000mSgb*ZJuC2TaOb+yy0#@0000000000I-%fnXNX`x$Jz!iN6l_P0000000000"
    "L9NBbVsc<W_#ILr(YS6v0000000000M5SE%UR+>6iYEplTBL430000000000cK68<Tt;9(<DL#K;*V}X0000000000{hfRYS}<Th_um;vX?t!!0000000000"
    ";F9CwSQ=nJqDIzn?`Cd50000000000MbK@TSOj1|*)H;|bW(0W0000000000bD$JivfN)lf%F^u_&shw0000000000m>EztwXk16cJl#Vd?{`~0000000000"
    "LI7;UxrARpu!mCD{uFLN0000000000fk;LJz*t{E2@SkxfB<el0000000000pXR*o$1Gn!df)650qku+0000000000xz4SF(EwjS;#!m1fY)t60000000000"
    "?ezA(+|*t`BimiR|H5rR0000000000hD~;z>8M^nGrao1eXeal0000000000jLZof_<UYK>RA}%`k8G&0000000000Qf7}g2vJ@@Ll9*tcZF?00000000000"
    ";SS~O86{po6FZNR@^EcH0000000000g1YE9E%{wQRW?N-ZCq_Y0000000000SY3D+Ld{)3>0LF~=0|Nn0000000000eCu(DSfO1&tPKvfUo&k$0000000000"
    "Hd(WUaCKcknW25Z*B)&^0000000000n3@_5ib-8Sh;=gcP6};60000000000?4yMnr5#;Ba^>1s#r14J0000000000PHEDu!SP%`H%%q%I^k?U0000000000"
    "+TPy4-^E-&wg|wmugq*f0000000000-1vx5|CwAs=5tx7Be-lp0000000000V5_cNAZ}bhqw1g3mZNMy0000000000w(2e4LPA_X<;X!d36E?*0000000000"
    "=Bnh?W)@sPe`@W~dU|X?0000000000Jm9cBjOtrJZ!1xz>t$>}0000000000*M4vlv%XtElV=;YTTyI40000000000(Ru+?+mc&A%zF9)%RFpA0000000000"
    "b;v>d1ZP`755b&{IVfyE0000000000)msk<Fgsg7JC?9orxR>I0000000000HYrGVTn}47L005w6#r{L0000000000xFpI$isM>91zqO4fa+^N0000000000"
    "c0zM&xwl$CXa;3c?A2>P0000000000zIGw&>5E!GVy+G`SHWvQ0000000000o3*mP8(>;M$gq)E!K`aQ0000000000S@Q(uPBdCTmT8>IDwu0P0000000000"
    "1SIxag9utcqcq)!lY?tO0000000000=sRQ^x!hSm<!u6$`)+GM0000000000A}=i@@UU4xKykY8Vp?lJ0000000000%keI9D1})-k(#re$wg~G0000000000"
    "HF!q?Vp&;0(nXw|FEMLC0000000000YPM9-oGn>E+|DBMlpJe70000000000v!FEG*a2BUrIGcV`3Gx20000000000FDu0#7S>om4(h_6UGr){0000000000"
    "{7K}PRH|4&`!hNGz}{*=0000000000VexN7lzv!1V9N%yBg$$(0000000000Y!yZy)lyhM6AaqOg|%ux0000000000ZK^~j7bjRi6jEV4=b&mp0000000000"
    "Zlp<8TKiW(Hu*{rNQ`Pg0000000000xc4%wp3hf6a9u@4sCQ~W0000000000YM9<6<fB(Wq}u-92V-hL0000000000o_Z?1DtK2wvdkslW=?8A0000000000"
    "4-qVJa!gl11cKE?#W`v~0000000000o~ZvcyC7FUP+Wc*BPD7;0000000000kX{=21oc)xmHQWFfDmdx0000000000duigSPsvt5Ul^h>-TP@k0000000000"
    "^#8p%o1IobIBOh+Ip=9W0000000000$5<1W=W$j*!mNgDmC|WI0000000000b*f0QG)7iHeU;$t@V#k30000000000$1B8lf*Mvp3En9aOsQ!=0000000000"
    "LKE!S(C$@0=RRWPrj%(w0000000000u$LysAH!8Zyl{za0f1>h0000000000rSI7;Z<kd-4lJLvTWe`R0000000000w^3{WziU-MW+|h?wN`0B0000000000"
    "uk=GJ4?tBwXd#uO4?<}`00000000005)a3oUldh9lgxBoX)S3$0000000000rPo_Zu;^4kiQ_@~!Wd~l00000000000)=QD0lic}(H{JQ90X}V0000000000"
    "zfDsIQjt_Z`!=HhcJF9F0000000000plE{kqGeP-pyW1u&)R4}0000000000E(~1V^Ep&NNZ_{qD8^_&0000000000E^kJtL=IFyjp8Dag0W~o0000000000"
    "D#)itmEu!C`J;L~+?;4Y0000000000<W1(P<+f8mC*8dXHi&3I0000000000_BLq2HH%X~p+@WhkaK820000000000^tqFIg<n%Z@|XoK>RxC-0000000000"
    "rwz}=)H72+yNn5!L`rBt0000000000m1uyHBM4JKe%kj%pEYPe0000000000gaHoFaokcs*gDWd`XFdP00000000003J>0Gzpzq3S7wr#RSalA0000000000"
    "zQ<Nu424oagB3ndvG->{0000000000Xky-nSXoj)`vB8$4dZ7(0000000000jTPJCqb*WEL*Y;aYR+ds0000000000^B1#Q?g3Ij{BE}s$GK-f0000000000"
    "C7{gVH`Y-=r(y=EB&BCS0000000000<>ZZIfT~eI=5#&Ef{<rG0000000000${bzO$9_>jJYHOb;d^I50000000000Y<{2}4pUJ;J=bE~Kxbz_0000000000"
    "l0d#PQzubCf|yX=pi*Z*0000000000!kHWbm-|paky<5^0X}Cy0000000000!N4JV+0Rfw1=ipIV<~4q0000000000DC{$h8l+G_Zo`5z#S~{i0000000000"
    "jZ(QMT6j=EQ`-tNC;(?b0000000000wJI1om`qSWRMvP1jO%7V0000000000DXX@#)F4nm^lZ$$@YZHP0000000000TNU;PAL>s)(pLppR>EdL0000000000"
    "nE#eEk&;h9bWbDZysc(H0000000000kAfV`|2j`VbqpqLB$;MF0000000000a0^qdYU55oVn+@4jD%)D0000000000YzGIo(Th$%!?@Ut_HSlD0000000000"
    "3*-dIG&D{?IvI#JVOwTE0000000000TMik&liW=}M<E*k%|>QG0000000000?eh<N?u1Q1kcpA+IWlHI0000000000%mKFAL@iA~kqYY$s2yfN0000000000"
    "gr(QKmeou^?OZ2k7YSxS0000000000KukdZ<b6y)Dmg|0i1cMZ0000000000He<hoEGA4q*H=g8`QK$g0000000000>Q^2naL!9Wl5r{@ZOdgq0000000000"
    "Y?ms=u69d6-ahZN;kIQ!0000000000OpmZa=pIWzT6mSPSE6M=0000000000wyb=59LGvPlGYp>&y8h30000000000$&MphOK(a*6#<?3M|ovH0000000000"
    "Fn=K;br(uNZ#QK*!enJY00000000007#D^zn!iavXM)kbJy2yp0000000000372o{x@Sp1ufsWRx;te+00000000004nO>?*AGcR;oS8rI45O50000000000"
    "wogje@V7`nvaem-wi0DP0000000000ED3nZ24F})!X3$DHvVKl0000000000nX9Ab83;%~un3T*w&`R*0000000000iFfOZDzHaDFH#DLH`Qc70000000000"
    "46SV%Iao(P+uM)txWHsU0000000000s*FhTMF2-YXNP#cI;><s0000000000k5-y<QK&{haW)*<yq9D^0000000000=t_b9TTn(ooXXNBKZ9gI0000000000"
    "KB^n@W%xxvnI-gy!ER(g0000000000p|ppwZ=gj$0x8(!Lt11&0000000000nyTZOc}PV-kG9qx#zka60000000000G|Wv5gz!W_)2~MxNHJtU0000000000"
    "{aNH_kC;S2hxUTL$sA-r00000000009+brooIpfCNNnKjN(f{?0000000000#JUe*s^~*Nweq)$%kyJE0000000000eGcKgx{yOah-+bdOW$Ka0000000000"
    "QqNa|%{W6qQ@Avy%F1Iv0000000000&^?kA<KRL-&$@~1NVa1@0000000000Aeyew`-nn7m{hnk#-U?C0000000000awN+a7%@UXQfC-oLXBfU0000000000"
    "O&tbGIN3o!uhaony?A3l0000000000z{GvWT!BGAMt;osHe_Q!0000000000S6fi<g(*Qm&_&QMuTNt@000000000093h+}w9!C7;tXZnB|Bq40000000000"
    "n^u3s=XpRtGPUC4nkHjF0000000000G@)rcAtFFP0?Mu@4iaNP0000000000oxLaK#gsolD0C7KfBj-W0000000000l0tCeW92?TEt4B$?C4@Z0000000000"
    "Q6iMwCN@4mGNh{CRMcWX0000000000+pc;;8HYVUZX}>Sx4&XQ0000000000aybizMc6z*3z6JB602fB0000000000UlT18wkbS7FK0lOWR_w;0000000000"
    "p^nE{b$dHN6eO$$t$|`d0000000000MEt?Xjm<hh+C%_K>TF^_0000000000MXm$S2_QN^yha@y8(3mM0000000000u=R7>^Kv;r+@|9=K0{(a0000000000"
    "fF$}{R>n9$a^f>wQZ8aZ0000000000{fm9VI~q7ZrFQN?RvBVI0000000000aed`_u5LF#o#6AONCaX)0000000000+e#VrxWhI;l$|6*DDPoF0000000000"
    "K}7(dWf?X=s48LD^VwlQ0000000000rM`jczHc=^8U>0Ts>ESH0000000000ObCTT&&D)B3aWq7Nv~l*0000000000fphU0p&m3qp_U@B(V1aD0000000000"
    "KvJ&UK6f)f97t1hK!jmH0000000000l=Y(evd}U>m^<><lx|@_0000000000d9SwO2P-l_QSRTF%~)YT0000000000{?O0^NQN;$X5IXB=|W*Z0000000000"
    "9<12Rec~`c6ox4!=PY4B0000000000ACk4Kvp+CEfn`Q!#TH>e0000000000!v%|B<2x@vd;wKjg8*Sb0000000000`s&6CU-~UTehA~?BI;m30000000000"
    "k&G4QySgku%!ddCr_*3S000000000019B~R+k7iPze(iG61`wR0000000000ej&w0pFk==ofwruX{KO60000000000Dj}28=mjZ2s=hgIt&d<p0000000000"
    "DLyW3nan3ZP+E#z;&@;{0000000000I*neJYm_EH(Y_ik3}RqF0000000000o3FYp$6O^qx5Up;F-u@T0000000000We$yR2_+;z#W$9PSTtZj0000000000"
    "coZO<kL)5q?BiO7h#g=+0000000000T`Ruwwz45W{FDk2$Od3Q0000000000puDlN+;$*9kS!#|JK$eH0000000000g7eo*192WerD3%>9<yIS0000000000"
    "0Y6-y=kXgrak1$b7=~X!0000000000)W&&*hIJW0&oVaS>{wqw0000000000)o4Le4FMNGy1~SNUn^fg0000000000<ibKiqnQ*y?+l0_GyPsb0000000000"
    "-h=F8@j((m7HwAAN6ubA0000000000!BwdS`0o!uA5NE`vz}f+0000000000k47g;i*gG<doZ^~l5So=0000000000SE_@bp#}#)S(GB6|2<wn0000000000"
    "uvhO5K&JvgE&4@(7Y$xO0000000000+(0%#*Pi@8Fyf86_1j%Q0000000000`Kg(T4fgXt3*`JDvZ`G`0000000000Nh~8V6|&|(QQ+MfMtNO80000000000"
    "a3{mc>XpVnSK($Aokv|j0000000000&Gw49j7QeLhzt{Pq7_|00000000000-(BJJzmw~~8;4zrLE~IN0000000000xJ#l0Ldx~OM|Q!!Ww2a80000000000"
    "a%jj&5{Ui3eE+#?`g~kK0000000000V)9wyEPMjM+egCg>PTEb0000000000xyo9&5C{gq)O1y)A{1Oe0000000000+ANPGIbsRGjH~YKjNV&70000000000"
    "W`m!kDuoQdZV?G16{uT40000000000I$Q`JDQyqH2ehx~uW(yH0000000000q^bS^)D98AGU~1McQ{)>0000000000Az~uD`ArhQET44Ck^fpi0000000000"
    "5F=jDt9BE>?>KeqA;wxj0000000000Gqxa6kc$++$Q>cqOpIDU0000000000<L1~<O^+48ue0e7H&R+a0000000000(EvQOorD&^er4gJ1Q=RC0000000000"
    "8B*ZNJ8&1k>e|S1&fQr+0000000000uw_gS^jH|cr(Z+Qp`%$q0000000000LpQXJt3?^W)sUTji)mRv0000000000-o39EMLin8H;9}-m@ipC0000000000"
    "_xmMGP(d5OuOFOO#r0S~0000000000587AL0!19a7zU3<=DS!x0000000000_Nza&G&~)^NKdLD(t%h&0000000000zvVk^hb11sZkbMbg-2LG0000000000"
    "jnsrG0|Fnw#If5v1r1n00000000000M#NGWu+ksE-es{@R?=5M0000000000n81M2KujROFf)CfeV1220000000000zZ1@G$*CZ~t_qCceq2{T0000000000"
    "5-xfkzX2h@=y5n&T_IOM0000000000Q<IEE?o%Pa$imk_9OhO)0000000000)lvYXB%mR{ymrgZzN%J00000000000gHJG%hUOu_XTEzoNN-j^0000000000"
    "%F&(H!6zcXZ>$~1yE9fm0000000000%#J-@kYpmj%3@Qo9Qaj00000000000n>z#_teqmjV*|fAZoE}M00000000006piKw&(k8nY_I~Lvw&4V0000000000"
    "V>sChvj-!<y^Fvj@I+NW0000000000V)D=&5;`NmgtwXmCka(R0000000000I&i`)rfVa>o>en~Tg_BJ0000000000%x0o3DV-z0H;63UkdahC0000000000"
    "EQ@6kRnH^9KyfHi%2iZA0000000000kEn~t-2x=Q?66%B3mQ~F00000000007Sn#^XEr3jJ7c(+Pu^2N0000000000>yw?O^kgK!$}Wwqh@w+K0000000000"
    "^-(7r$BHDte7|6Rs%BF_0000000000@x!lgY^fx`-rd4kxGGaX0000000000n*ITyPs1d@xFP;Bxb9Lw0000000000B?9Y@mDwb~A|e57t+G-;0000000000"
    ">!TDb=IkWEliTHLns!n^0000000000>ER6|>HQ?YZV*G2fjUw^0000000000K)PinMhzvv#DcF-W&Tk>0000000000^eU_2q#Gr``Sig8NWxJ-0000000000"
    "e-QKAtSKeHoUL6DEQV1)0000000000Pqsms);A@<Ahh%i6G~A)0000000000H?5T|Dn=#118`@Z{0&h+0000000000TL-E)pHwBlg65kS=+RI>0000000000"
    "#nw4iEn_9XEFE~&)Rj;`0000000000j4c2t%yA{aP#NMc!dXy20000000000(H*!;bATnlK;XZBuN_c80000000000PS7~F8jmHwCst2FIKxjs0000000000"
    "Fi5#q#GNI;Ix8ca4@*x#0000000000PXtnJeXJ$G(O@4@<<d?-0000000000^Xn4TQ@bU=?3CrUxmr#@0000000000Q|1e7Rmvs6Ic66hjp9u}0000000000"
    "v)W7#kl7``<Z@QrUusQ20000000000SFq)q5$Pqs7+FL?GV)A70000000000bx!8)=lCVSgtvmA1bR$B0000000000Se#zvAPFYGCM>f|*8xmG0000000000"
    "G=W}R${8lWtyOoJs)<WL0000000000PUQ74@+v04L(*{8ei2JR0000000000=BzY8raUIVK|n!SRGCUZ0000000000O2R5<^iC$gs6C2vD<Mii0000000000"
    "p9|=M>R~3pdTH%<1gS|t0000000000FmQZumUSk;w_yyq-!Mr)0000000000J=gaP28t%Yx5S8vy|_p~00000000002PkSaN1rCZ(=Y@)o<K-I0000000000"
    "-AUudXSF8421Fnug2+ce0000000000;UjaNb<8HfTAhB8X;4Q%0000000000UY#kKe&i;=%i6%%RM<v90000000000nVlAtko_jWvYLoaL|{fh0000000000"
    ";tNV_xfdtE5qoz+IOs({0000000000QqR-=1~VtX=^8HxF>pmd0000000000JkP@@hfgQKIT%s$F8D-10000000000;qocRM{Os-ToHd$GJr%t0000000000"
    "ejygOR*WaWsH9qcI|xKT0000000000Z9<sU!K^318*qiiN{~Z90000000000zdXSImd+=@y%M2&U>HL{0000000000%J-cT<nbrKi7HR?e4s)=0000000000"
    "%vwR~wiYPB+*L7upeRB>00000000002u-?X9X%+(2{Y+&%dkN}0000000000*R>d8C1ohU5nvb|{y0HE0000000000M0{+z+>R)~`g>s3IKe<b0000000000"
    "r-#K%QMxF=z$9bsdr3e*0000000000K)*Bgkmo4Cdhvs6kE1_80000000000PzuMBsTL`~k}N$0IkY}N0000000000W?^F|t4Aro5Z|q@^~60u0000000000"
    "$e8_cxOyqT>kZrQ$<;hS0000000000NTmSf{jVv&?Yd+UyyiPV0000000000>I|R;sOKrb^PIwD()T(*0000000000uT>9S;~^@*)@p1z6A3y%0000000000"
    "%>{yP++Hfc%G%lDgc><O0000000000>Y^*bzn&_<`=o-dEi5=d0000000000HM2k4x#23nn}6Qz6FxUU0000000000u)Og?_a`gBgdt_|Jy1440000000000"
    "q);N<jBzW#N<TzBuVOVo0000000000OT8}n1HLQ3c6qYTT6Hu)0000000000n*2@QvJ5Q1AC{?`Er>Hf0000000000-j+bX?_4av%zo986q_<Y0000000000"
    ">OOE-2(B!^#0G3^{H-xS0000000000=-OD%RsJl%n!VM&%fK)|0000000000kAFVVAWJR4`QGNg#HKGm00000000002|wC`UXv}rsKe@Oa?3410000000000"
    "@EO}(W7#dhqUZQt_3tY{0000000000@*JaN=N~S>GlDVVtrRLi000000000009BNkqG>L`SIM+K{5dE<0000000000T+R^xOt~(=|MsH2Kwu_70000000000"
    "-1v9vY7sBM!(Bwo*M}rP0000000000>XCv-v2riKla5e4`>-NF0000000000kT>0Q;M_03dbDbs(%T?F0000000000-wk6;+b1x<i_0FO3K1Sa0000000000"
    "vbR0*cx5oaV0m_Zb#59!0000000000HnJT)D5o&M37^%IG|(160000000000H5MX9$m}q{xob79mNpYW0000000000u>5HsI5siBD;ORt)k6<J0000000000"
    "2!0gbR)aCX>1ekl{v-%M0000000000nKild0Ms$S-fl8D9uWaR0000000000iL<rE1Sm4VEy7a@vk&w?0000000000c41{6EPpb<^3(HnctqGg0000000000"
    "e9mdcK-n_D%(PjW5o_PS0000000000_lRVM3^p^s@G0GsHnsJ?0000000000{vY=yS(P)uR}!Z^y}tp#0000000000Y$C&O@b)vnavKmAX`Kkb0000000000"
    "8y{*coLV%%1RcfRKXebk0000000000G^LFqCcrenR>ZJQaUv7I0000000000`$EqfSSK~WuVBC}`G*$30000000000lZVjY`;RriTpZsJ2<sWZ0000000000"
    "+5s~<+xa!Xz^z(*?n@oO0000000000t3a6WzhpMRn`J<HepMjA0000000000p<9LafY3I;Rdj?9GXx^Q0000000000c~kzYn>{zcHuIH_aj_%70000000000"
    "p(ji9`>Z#>@O4RC8C)g60000000000xwVL!93MEq{UGNWkOL>c00000000004~r)}?~^#dcKtUf-lr(Q0000000000EE)fAT?;wDQ54Bu6-g?<0000000000"
    "nb79YP=z_b1#2gnN9QZR0000000000)grd|vH?24fC}KThJr1?00000000009DziEZGbw!mim8#<Qgx)0000000000_FQQMY5_aI6(UM>Ian~j0000000000"
    ";3oVVl7u_JV)oSCp&>EA0000000000Wl+>%&j~!htNx~IwCORx0000000000x=P7!3z0m))4m1^dayFU00000000004TrMdFB(0-zPdW8`*kzG0000000000"
    "_I77WCZ#>V-FAsHLOL|S0000000000;flXZ*fBo9{&hK*R{b=<0000000000WHgd}FTg&)lRS~ZLBBP?0000000000t>m_M6;MCGeooh)41hMk0000000000"
    "@eBpkbKyV0s4r3<z&|&@0000000000W)G?t8ahD0dt^v0W&St70000000000O~pMBAcR1`%x?g<1iU!F0000000000{9q}ez|KIx2-d)<r+7KQ0000000000"
    "Q#H$=@EJkCO5AcAS2Q}n0000000000)Fy<*rDZ|DPK<x$AMZNA0000000000XB2~}*0DjrY2P8W5~(}D000000000062mL;dHzAbj6-{4Kw><=0000000000"
    "hxp+@hD$=gLMW}txf(sd0000000000I(i8`@|Z%vkNnuckJ3HB0000000000r40s(wB|y<a4#St(}_O70000000000c6d9?!!$#{ZZM?Alt4eg0000000000"
    "v40Jy7K1~;fNh3{<@P_o00000000002<hL!qtQda7%TK1YgRzO0000000000@_81%VIV}nzN~fOJF!5(00000000009rzSMM{h*Hb#vHv?F&J`0000000000"
    "kSIUzO1?zEIyS9Nh-N{+0000000000{LuJpV+}>X(yWBi6v08j0000000000)wZQxhg?O#ypHd0n;SyF0000000000m7l3-t*b@Ad6RyVBz8i;0000000000"
    ";tz;0%=$&Z2$iaPxX?nt0000000000-69rZ+DS&hc5BXKUMoYu0000000000dc|F<&X`8P<nzib9fm`|00000000006nkm9pXWxv$?Ryi``|;s0000000000"
    "`Nl&%L^emjpA>QA06av%0000000000y)5_PvV=##A1bVrGnYib0000000000LNAV1;?hUJLF?kEn({=z00000000006;cyP$skC;xfb35Jx)cy0000000000"
    "j2Mw;UT#Rh1Z!r1A*w~d0000000000`UMhTn7c^8m!?DoPX<Q70000000000@tvUHZ3s!gFo;2M%wk5s0000000000P-91z)l^Bqzs)73pua}I0000000000"
    "so4-R#-K^S0yesV(i}&?0000000000PJ{<vHtb2jY=6h+XnIG$0000000000nC3Oi95+h9eP|ZAYS>4>000000000041}g9aDhs|>aTCp-Zn_U0000000000"
    "n%`p|C(BB}s1uwT%#}#M0000000000$$WTWI2233a8hJ$IrT`u0000000000?cVO2np{i3x+;;cEmcXt0000000000l8MoLM5If=M_cQ@ueC|Q0000000000"
    "=y{6lEbB|apYLe$#TH7y0000000000d~O(mOfgKr=nnm#a(GI>000000000064E1;oOVpWA+Bjtw%SU-0000000000zwSGJ7Q0NqL$XVjkv&Vm0000000000"
    "$qw+9v;IuLmjFQW1ffg70000000000<9p9EZ9z@IgMH)h5Cu%Y0000000000Q-Eo+JcCWZP7c!>wrNbj0000000000EoMz>8^le(@RzIn^v+Dc0000000000"
    "i+2jL0t8OLX#WWv&NNNH0000000000tNt(J>q1Vz`6-JALYqy%0000000000M@3Cm(11?B4*bhPQ36iD0000000000q?Pewsl86X=vrx8{Ao_W0000000000"
    ");X+XaQRNag)0EsL()#b0000000000?|Tff9yCwD-BK+`C_7KU0000000000Gp3=Tt!+=hJGPB4s-#cA0000000000VCb2b7N$?YBN=E4Wd~5e0000000000"
    "5M(gTRozd(PXOI*!#q&H0000000000D~EVGwh2(cRGB?UZgNn-0000000000IIFE7E+kOEtuqXJW2jKT0000000000lT~+e6FyMDRvydepx#ix0000000000"
    "-8~fvXID_b&#3%wBo$G>0000000000pBu-$DR5B0VIm{->q}9<0000000000W~R((T!>J>VA{Y~^@35r0000000000&y{87|DRC6Mj%*xKDtrB0000000000"
    "L!r?O6}M2p6GLl3#qm+V0000000000Uw}}yoXt?cJCb~fhbdCP0000000000uj(PZm*Y^t7+nrFgJDv@0000000000zL$mf2Ki9H^75i6wwY4E0000000000"
    "CqKJE>kU!BN29z#9@0|40000000000SC|5HMIuqa7QNkNxe8Oj0000000000>sp&<7B*49*N=#Eg+o)o0000000000YCgsFUP)2F+G;;dfP7QH0000000000"
    "W%|GOA6!wua-D?Kr?peS0000000000aYoBVT5eIm7<bz0`R-G|00000000000NN(J4uDa>%0|O+bSPB70000000000tZWi7K9Nzt0$jmL6=GDt0000000000"
    "0k+M4>7h};5C&IQ+M86s0000000000K5W{55wcOhF6K~q!PQj200000000009V=l7xWQ4tBk~%|$PZP(0000000000+5l`3;m}dQPbbmo?nza^0000000000"
    "Y8ngOkl#_j;ZbURGK5vY0000000000zobsB%k5FX<UQRnlD<{I0000000000)67Q1mHSb^i?<-M4Ea^S0000000000eZK=_^9WMFii#_#pfgs$0000000000"
    "xT%%S=oM1H7K8M`Om0@d0000000000b+`;hdLmN5H*IpK52;qb0000000000o)~!ytS(Z(_Y|B2=;Kzv0000000000EcfbTf;v*bh(>6H)E-yB0000000000"
    "{eW_g0Y*~5p3TkC(^*%*00000000003F9BmEm2ayaCXC&<dj#y0000000000JxT**3tUpb1Pwre1<+T(0000000000isLMmo@P?OWnDTsH49k40000000000"
    "<lC&E>u^%Q#51XCaz<Fd0000000000D`1n!_Igsl+FOIbyMb810000000000SDo-t#Dr47)}g=U54~8x0000000000RlwAHSdLP_!_icQZTMKg0000000000"
    "74}bwx0h1Ds<$~x(=%DX0000000000U_bG?<)Bi);+7)dKW|yU0000000000>F!~l=BZM^UWi3XvZ`6Y0000000000Z%qMiys}cjE19)4D&|?h0000000000"
    "t1%2WX}eOuTjIrDry*Lv0000000000S8~4!@555S1AWPdDP3B?0000000000G3HnKPs~!lX@V1Pu$WrF0000000000<YN*Oj?_}XMho}TJJedh0000000000"
    "Fv2TntK3q+<uJ!}%MV+?0000000000&ez}ws^n6@R1DV>UrSrS0000000000e0y#MjO<dt*|vFY_J&))0000000000<%1D!Q}j~6Fi8SJk-}TR0000000000"
    "&Bq2)0{l|Iq1?kzFaBG=0000000000<sFN}ngdh7I*Jil&^cVd0000000000?p)i_9t%^z9BV!rbah<70000000000oLe-GkP=hCe3+e68nIl!0000000000"
    "rts1R@)=XWBrAf%!t7kY0000000000#q#q^MIlqbPdq;?Z75y90000000000v^dmoj3`sU8RaKR8Dw3+0000000000E19r@$uCpDls=E$#-Cll0000000000"
    "=IX_g{x(y<0%FLeb=zIQ0000000000&GGWEEk0Agk*6?XB^O@60000000000?M`<(SVdF7s-c?Y)ly!-00000000004xksieN0opLconRhmKyr0000000000"
    "p8vXYol{f5Z8*7KIm%wZ00000000004v0vlwpvrbX+O+y>;+%I0000000000!c(V^$YE2!@(Ff%pg~{20000000000E`7ds(r8n_N6lq~RefK;0000000000"
    "zaDj%({EG2fWAtg3%Osw0000000000<x;XX%XU-1tP<4P!t`Ij00000000005HwwqxP4Q=gn!teJs4oX0000000000q}*KWnuJrpDO^Z#8a80S0000000000"
    "7iXA-af?&H<vEe%_fufN0000000000)$R74JCaktM6-$n)^T9K0000000000Qj1@C_?T0`;bIKcw2xrG0000000000@L9Q4r=L^69OLa>l&xUE0000000000"
    "BXsL@N2XK2md*yIbjx7C0000000000TJ+o(*sN2)*S{{bRq0^B00000000000`+u%ShG{WXi+_nHwR(B0000000000)MB6Q$GKC$UzPto8YN-C0000000000"
    ";!I`zCBRd_A55vk{X${D0000000000m8k;fa>i4@Nc<EX;$UIG0000000000o=Lj$uFO-wTzYp#$9`eJ0000000000uKmtv+R{_N+?oYEt(;-N0000000000"
    "c$R~a_1IIuUNZFgle%HR0000000000p5z*}0N+!<I2Cw&d)Q&X0000000000+PK#n{Nz)>1Q^D@WA<Ud0000000000+2i~T=;~9zJ`WSXOcY|k0000000000"
    "NH)#W#PCzVYtNH;HZo$s0000000000-hhj>lJ--;AuWCAAW&k!0000000000Ka_g=QT$WDwfqAK3~pk;0000000000AVJ;j0s>UPzY<;7_lsh{0000000000"
    "GFbN1r3X~N!9%@I<f&r70000000000A*u-ZIt^67I%HXH(Z^!I0000000000r3Y7{!V*-##7Jf@zvW`U0000000000U_m4qJs4ELvc(vWtpj7g0000000000"
    "2q?3qsvT6ooUf&coFQYt0000000000OpHd53nNs(1KU9xj6Y+*0000000000;65QCU?^0;ejFk|d|hL}0000000000X%L2>s4Y~$Wk`GUY<pwC0000000000"
    "pV7Bg<uX*jOEZErUYTRS0000000000EK~W57C2PEw6SaaPq$;h0000000000!)1+#JUvvvA((7PLe*oy00000000007239xS3*?4D7(27HS=S@0000000000"
    "%YeA}XGc`P9`A=;C=q190000000000m@3u$ZA?_ao(K#d94}<R0000000000DyUECX;D<bCER*d5KUyj0000000000GRkTOTvk-TH$Dmd1Z!l#0000000000"
    "V_|8WL|atAXanpK`iNw}0000000000WIm4bBVbg(5dc__@1|tH0000000000>M<Nt`eanV$OwXT<-}yb0000000000q6}6B$Z1r-4)UJ5+u~%v0000000000"
    "JD+}1jc!!HW02!s(g0<^0000000000YP#5fN^?}eV7P=`$sT3E0000000000)-I>n|94ctSU$J1z&vHZ0000000000GtDfet$b9#=Ju~zxLRev0000000000"
    "EoQ7MRDo2$h_<6quy|#_0000000000u;XX7^MzEv)LO@rsFr2G0000000000_(cbJjEYpi=rIW%p|oYd0000000000^OY=X9*<PO<=w&WnbKvz0000000000"
    "h=q`OsgqQ|skq7rlkjE00000000000q{i!wEtpim865p+j1FeN0000000000F$?#Ss+?57CSo!WhAd{l00000000005d~{=AfZ&ih~z{5e@bS+0000000000"
    "F=hZ7kEK+=GyMuPdT3_A0000000000ci?Yv_Ni3B1O{QQbcSZY0000000000&n0qXR<2aQ-&h}TZ=`0x0000000000B<R20uCi3XxlbQ>YQko~0000000000"
    "S=kZD|F%@XM?^ZtW#DGO0000000000SgRloNxM|Qe2__6Vg6>o00000000003J|55ioaCAH;yJWT^wh?0000000000Te(+Q#KTm;U6-m;SvqIH0000000000"
    "G+FU6_QzDf;{~s{Raj@h0000000000ccCg0Ak0+2cry)SQg&y+00000000005k$P>LC{pd6?&dhPn2iC0000000000?DAfbThvs*oHIO!OtNRd0000000000"
    "?qACVZP--6>$$%GO3-J(000000000020gkfcHC6J_GSXVNA73900000000007ChwbcHmUNbd=q}MGR=b00000000001rGu@Z{$?KTAV5gL@H>&0000000000"
    "!)MI#Ug%W7hS<WCLP%)90000000000H~9MqM(k9;@1lQXL1t*c0000000000Msi7KB=A(gO;tL0K!j+(0000000000+J|lT`SVo3k4enWKcZ;B0000000000"
    "-0C2t#`jdfn%NF(Kfq|f0000000000IZYf`iu+W+PG8$eKi+7-0000000000)?gU^ME_L4n=}h-Km2IG0000000000nb*yL_5xMFJF|DuKN@Mk0000000000"
    "aAWn#o(5IGBQ^SYKsaf@0000000000Kek85JqlI8FcfTMK~`zN0000000000^nra~(+yR?N2;ciLUd`s0000000000c6d*LUlCQnS5bomM3QO10000000000"
    "tV?zj;uKZD8e`kHMX+hW0000000000eyyqmTNqWqf;ot-N6u-$0000000000*H5K9${SU{axP=#N$hFB0000000000pX45=FCSIF$eTr9ObTki0000000000"
    "!5z>piy~FPdfNyaPbg}@0000000000Bdu}1+$B}OJV%BYQbuaP0000000000u0*^*Bq>$E2052kRb*<w0000000000N|tD#VJua^v<em2Sb}Q60000000000"
    ";ZrZjlrL4lDi+z9T%c;e0000000000U@ys{y)sq6TD82KV7_X=0000000000soL%u+%;9e`azlCWZY`N0000000000r3YCX@Hkb#{}o7OY5How0000000000"
    "LTJE_`8!p>OpBr_ZWwF80000000000Xs+E~_&!y@*871lb2e+h0000000000`-^FL>_Jt)PQkx+cvNe^0000000000dm%l~)kIamBnuDyeR6BS0000000000"
    "r(ms(v`1CIZqzoygOF>$0000000000%}S*9h)Pw!f!CwNiLPtF0000000000)bJKeQ%zOCbfCflkj!hq0000000000tYiAm6HryaFuVVHmg;N30000000000"
    "MYKc2%2HLp*|Wzhod|5e0000000000iu31ccvV%vcov5fq$X^@0000000000YBJmf9avSs9B-;Ht3+(T0000000000*MSw+xLQ@f=}De}vSMt&0000000000"
    "x$=l0OkGvLzcSARx`1rJ0000000000|CK8w*I!k@!L_Nl!Jcfu0000000000n|$&kT4Ghe|6ZD#$-Hd90000000000b+^)C)nrw`fVvH=(b{al0000000000"
    "f>*&GNoQ5SY*cXQ+4yY00000000000sco~LwP{trq_{3m;udYd0000000000-U^;08*Ej;NprQ{>NIV@00000000003P5)4d2Us}YY=gj^HOcV0000000000"
    "A09D3(r{J46&%iS{BUi+00000000003bw;qBy&~3Y9cvu1&?jO0000000000wiqLVaCKF{POl!74y|p#00000000006IVp{ws%#)-cIY-7t3wH0000000000"
    "4}u*0_jy&oE{zL3A?a<v0000000000o2sDFGkjIRT=wg{DhF=B0000000000p5ev3YJOF~O&0lCG$n4p00000000004LE`PoPbrpBYB?@K0|K60000000000"
    "&%Vdg%7RtE=c-HMM`3Qj0000000000+J=sV^n_KwuQzJSQGag000000000007NN>e8i!TDmaAvTTb*ve0000000000bRKbfJc(7nfReD<WxH;`0000000000"
    "-w$^~TZ>h|k$AEHZ`p3Z0000000000NQTkvc8yiQ)alkWdiQR?0000000000o04_AkB?QrTXKzigcWbV0000000000$wgT1rIA&@L!Lp;jx%q-0000000000"
    "ys&P_xsz4Ears{_nNe@R0000000000W8|%i%av8Y2M|@7qi=7(0000000000t?>$a+m}_q5^jwSu8nWN0000000000i0OLO>6ul)pPzM#xT|l#0000000000"
    ";@tAz_M26}&@g%o!^v;J0000000000sN84U0-jaCh^+^c&F62x0000000000$ob9*4xm-Q_E1F*>;iDW0000000000Fhw1$7@}3c0?oHm@ey#q0000000000"
    "($7Z+BcxTp{E!iu_8@S;0000000000fL(K0Ev8k#a%PL&`!8_70000000000Ixh0YH>g#>W<0te0Y7lS0000000000h4EZXLaJ5336iH`22OCm0000000000"
    "Mrh#YOsrMF1!82Q3te!)0000000000UQ?-&SFTmS5T^m-5NvS30000000000vUB7`Vz5=fy<3nW7JG2O0000000000B#OTOY_e6r!+IfK8;Wqi0000000000"
    "n^$DIceGW&&;G@mAenH$0000000000`J4iAg0@w_iQ<meC8%(~0000000000ARERWjks07rXY<HE4OgK0000000000^pe}Ym%3HJyR*?uFvf7e0000000000"
    "Tg7ioqP$hWftGECHPvvy0000000000HFK}RtiDyipTlv#I^=M`0000000000XI;k}x4>1v!8PyqKl5<F0000000000)kNe~!NOI*o9Ff~MFMfa0000000000"
    "W9l+#%fwZ{#nYK*N)d6u0000000000^bX)r)y7r8^z`VSPatu?0000000000YjYS2-^f+K&_(stR4;MB0000000000tk?*U=*m^V1i{Y?T0U{W0000000000"
    "oB!S0@yu1gNnwdYUrlkq00000000008>z(U`p#9rFd;K~WL$B;00000000004wmV(1JPB$aQoe`X=`!70000000000T8DW!4AWJ>w(6SaZhCRR0000000000"
    "<2|8o6xCJ0v*~&sbcu1m0000000000i+YV!9M@IA0P~Jjd6;p)0000000000GyZDjBiU8JPXXwRey4H30000000000$BAYdE8A7TPY;*GgSK(N0000000000"
    "AKt9*G2K<bs28pLh{bWh0000000000DZFZ6INw#k4;4E#j?{6$0000000000!Q{+mKH*is7y?pmljCu~0000000000&lms!MB`P!e%S)3nDTMJ0000000000"
    "HW@J|N##|*<&|FIodI&d0000000000)~1ZrP3Kj>`!RMMqY!ey0000000000nSwtDQt4H|cc@xfs2_5`0000000000TWxl?RqIv2>hm0ytuAuF0000000000"
    "1t?zOSnXB74G+%KvORLZ0000000000dUGRyTkloCiS&LFxJ+`u0000000000nh^7jUGY`G2%kVwy<2j?0000000000PQdvHU-MPKLJ1_3!fJBB0000000000"
    "bwgz7U-ebM%Pl(9$9ZzV0000000000^IBgkVE0wPRtJ<A&4_Zq0000000000t;ZM2U-?zQkdQ)M(wB0;0000000000hDy4?Ui($RBg9st*QRp70000000000"
    "W2AEmUHw(S#+>2m+_iGR0000000000CYErbTK`qR4i)q>;>2>m0000000000w?!9(R{>VQ%59#0=hJe)0000000000&Q3S3Qv+7O+}h>J?Ba630000000000"
    "A4W(bPX$)M<n_iF^6_%O0000000000vai$ONe5QIRvtcR_W*Oi0000000000k`H6<LkU*E!m~%Y{10=$0000000000!0CTUJquRAs1@J~0v~h00000000000"
    "Ny4-AH4Rq4k)s1(2QG8K0000000000G#!ZXEe}?}3%#zj3_Wwe0000000000gMNlVB@tG@g~8nn5=?Wz0000000000LuxqY8xvN*ju4Gy7h7|{0000000000"
    "d1NQy5fxUzt_iun9BOmG0000000000EAiYj2NzbrWei>$B6)Mb0000000000W0d0G`xsWhMR<64Cx~;v0000000000DAK{^?;2LXzN`e-ESGb@0000000000"
    "e=Y<{;~ZANT?zj-GNyCD0000000000Z!I$m)*e>CsqJH!H??!X00000000000#n#4$RJk0C%G^CJj8Rr0000000000LnD2yxgu7;XUkq+Leq1=0000000000"
    "Fe0mwsw7sx(}V5AN8)q90000000000*3;|RnkH7j0Yca-P4RQU0000000000J-o<=iYQjVb7dQpQUG+o0000000000YG2Eoc`8=GuT8W4R}XZ+0000000000"
    "W<h-#Xe?I1LRy??T^@A60000000000Hs)snRxVb+o$AumVl8yQ0000000000;n5UULoimrPhL|(XgqYl0000000000bBGNWFfvxa-4q(JZA)~(0000000000"
    "@r~O?95hzI)@>spb6Rx30000000000U}gyC2R2r~s>o}ScxiON0000000000%9S1Y@;6q%=!EYCet2}i0000000000G@zt@-8ojk76PkygNJm$0000000000"
    "s0k-|$2(TQxSyKsh?aD~0000000000D^@P*usv45TfMqtj-_<K0000000000#|cE!nLk#*bGirHleBce0000000000fvE8{f<acmjrM(0nZtCz0000000000"
    "Vz$5&YC~4QF3}myp3-!{0000000000ZnrvaQAJk3*{Rn@q~UbH0000000000urKLGI7e2%AmFpZsql2b0000000000E~8bS9!XZfa?`3num5zw0000000000"
    "@RbR91WQ)HT;sUDv<`K^0000000000{fmy(=}cC@XK&g)x*c`E0000000000SqW~o&Q4aq64wj9zbtjY00000000005|S|;vrtyR^La@>#XEJt0000000000"
    "Dym`&mr_>1b+?bf%1U*>0000000000s?~~udQ?`xDeu}u&{=iB0000000000nV2(#T~=1Wlz=D6)o69V0000000000{&waOKv-75Im>ZQ+jn)q0000000000"
    "<@78qB3f3!sxG$F;D&X;0000000000PnM9^16)?XRSNN2=9P880000000000Mc<kd<y}_5$h{uq>!fwS0000000000&QsOj#a~vyiKshm@w0Wn0000000000"
    "^>@iYreRjVC@Mtv_QG|*0000000000zs~?ShGSO17NRtQ{Lyv50000000000G2aBlWo1^t-){*H0^xPQ0000000000RVrWUL}ymO3iH022k>>k0000000000"
    "F*TW}BWYH^8;emW4gYn(0000000000&VxJy0&7;lpqIV25)O920000000000F`Cl)-)vUE40!)V7#()N0000000000Wc=iiy>3>(>x&cC9V~Xh0000000000"
    "Y%cxAns8RY!!q_}BRh7$0000000000QM9C<cXC$18_c8lC`xv~000000000091>+2RCHFreYY5kE?IWK0000000000)62&=Fm_hJUQ|&YG-!6f0000000000"
    "eq~nv3wTz*Oae->Id^uz0000000000AY+<q=XqAZ$sqqmKZbU|0000000000#7G^2!h2T0S`Tg9M3r{H0000000000cA98Ioqblolr4~QN~Ctc0000000000"
    "HrxfVcYjvE;o4*gP_uTx00000000004U&G$QGr&#(01pbRl;__00000000001wdSvD}z?R>R=Q(ThVsF0000000000Bx)!V1%+0?z4+(RVBmJZ0000000000"
    "aFlF8-iB7dww+#UX76^u0000000000??SR8w}@82Wv_GvZ2oq@0000000000uK9wykBU~nQjI2{aSeCC0000000000usmoAXpC0C{+4h%cN}-X0000000000"
    "{g3MrK#o?x`zue^d@Far0000000000o_Chh7m!xKx2wK$f;xA=0000000000mUr?{?~zu(!F9V3h)H+A0000000000@DquP#*<dSpXouajaYZU0000000000"
    "w9w;(ot0L=)@DvjlV^9p0000000000=P`^vbeC4Z^i19An09x-0000000000lDiAKN|{!`Yy0Yno`rY70000000000y7BAyAe&ae#L;spq?C8S0000000000"
    "X;`-W_MBG0jxTh{siSwm0000000000q;vGf%$`=jMRe$Cud;W*0000000000cE>MSqM%m5d;iP|wZV750000000000;@9!CccNCnpfDb+y3lvP0000000000"
    "^C-mIOr%!8NN3nhz~6Vk0000000000tUfT$A*NQq?*jPm#qM{&00000000007@fb6_NP|B9b<Wt%l&u20000000000KFIqa%c)ktWan5l(F}ON0000000000"
    "Bt&k6pQ~2DGVhYu)*E=h0000000000(v+aNbFEgu7uq9y+$wm$0000000000PCgvNN3T}El?MnQ;yHN00000000000q{M2X8?jcvK-W9O=SX<K0000000000"
    ")vFa*@3L0Ff~dP_?N@lf0000000000>vyK;!n9Vv@7jY6^JaL!0000000000?yRO#mbO;F3eJ$S_;q-|0000000000<j$X%X}DIvR^i@P{)BkI0000000000"
    "*I^sHJi1oEXLM@)1C)5c0000000000Xl9<i54={u<MA(}38Q$x0000000000<1Qbe;=NYDn<p(v53+c`0000000000+qCOrwZB%tGlpUB6v24F0000000000"
    "GeJlPh{0CCZI;ZH8qj#a0000000000#vFNzTEkYr72$0>Am4bv0000000000a*)FCEX7v9+U`W-CGL2@00000000005#?w%|Hf9pg-B6~EB$!D0000000000"
    "gk!K@(a2W7*>;jKF${UY0000000000rBvN%qsmsmm;wmdHXC`s0000000000Qvozqb<9@4j&!krJSus>0000000000YO?qwN6uEjX;qFWLOFTB0000000000"
    "$gpqL7|>S0@-3dwM@V_V0000000000M@8Lg>d{uf?S|lZO;>rq0000000000%D9E%yVF*{9O+acQ)YR<0000000000EmW;Njn!7bOWpd(Sao^80000000000"
    "M_`zIUe{K@B<#_0UW9qT0000000000`X!5>FWFYWcAnWAWRrQo0000000000BJRni0NYl;0(}w1X`*?+0000000000p8=VY(cD(Rmk@bwZ?Sp60000000000"
    "NU1zyqTW`(+~E%xb-;PR0000000000|1J3#bKq9Mp?TH8de3>l0000000000px6?$L*iDzr(nrzfZln)00000000002J+1`6y#RGuOI6bhV6O40000000000"
    "62oqr<mFbthCe#Mi~M=O0000000000o|HN0wC7g9*hsc&kqdgj0000000000hp%!Qh3QtnbHrX1ml}G&0000000000tz2F;RqIy37CSS*oGE(10000000000"
    ">f}H+CGA$gg*7v4qBwfM0000000000ACX=j_3l={iacHxs7HFh0000000000CaU}G#PC+Y*VwbctyX%#0000000000+IU>0l=4=<F_SZGvt@d~0000000000"
    "8%D)MWb{_RTlCx*xpaEK0000000000$Xa^LGxk=%9)+*OzJq$e0000000000yZ^sD1Nc_JJkF|c#FBcz0000000000(S6#L()m`vXjsl1%AtC|0000000000"
    ">KzW)q5D?AZLAr|&#-#H0000000000;3%I1as5`n3{#AC)xUbc0000000000k>}AEK>t?22h}Aa+s=Bx0000000000-J<Fj4*^%eEW72-;N5z_0000000000"
    "oTDuP-U3&^B#z5_=InaF0000000000t}@@$tOQrUzSh<#?E8Aa0000000000@t8xCdj?m)x*Z4B@d|su0000000000Kp&qxN(fiL=j&U8_ZfS@0000000000"
    "dJZ<x7z$Uw?#a|J{V03D0000000000d#_AU=L=WBo<MQl0yulX0000000000A4?~_wGCImxY#g^2uFLs0000000000L@Ejugb!E1`x+BE4pw`>0000000000"
    "%9S};QW00cI*=FW6J>kA0000000000h}QXuAQM->7{NW38FYKV0000000000Vl|Ja?i5$RYc_>MAA@_q0000000000F#A2SycSo$?s)C;B$9i;0000000000"
    "(r`+uiWpbGW)XIwDxrJ80000000000Bj5OsSQ=NroroS#FtB^T00000000000gsJiCLCA5L1*UvHNShn0000000000O36$m^c`2gA2!XcJI;H+0000000000"
    "+17g1!5>$^_kGe_LEU@60000000000jR0J9k0DpUjmPy1N9=pR0000000000MUONBT_ab(tD82vO#6Gl0000000000*ULzMDkWFI`+}5eQVM*)0000000000"
    "DF58+_a;}sPX-SfSQ&i400000000005pp$1#VA+6XRf8mT_}9O0000000000ac!2Ak}6lg4+L0wV>f)j0000000000B05CTUo2O^@Q)=aX-0g&0000000000"
    "1tEj}EG}2T+}aJ<ZdH810000000000^lFzJ`Y%_&l5G%)bYy(M0000000000&S1qW$1zvH)6gtAdUJfh0000000000a7$4vlrvYrZH;2-e}a6#0000000000"
    "wsU_TVKrC42`sUhg^_%~0000000000eIk?sE;m=edNmkIi=ceK0000000000q@dH-`Z!m>doq{$kgt5e00000000002RDMD$2wQQ(5o%0mcD$z0000000000"
    "fiC1+lss3!MJn4{oXvc|0000000000_0K&EVLn&Di8h}MqTGDI0000000000H)PYSEkIYmWLbH=r|W#c0000000000EHvps`axH~of~Oxt@?bx0000000000"
    "uZY9S#zR-Y@da)lvk85`0000000000ovNlslSNm+Eun|dxEOuF0000000000(&E;*Uq@HK0d}>4z9)Ua0000000000D|(n7EJ;_uH4XPN#5R4v0000000000"
    "hPf|!_)1s6jPFb2$whs@0000000000!0F(~#7tMf%cZlG&s2TD0000000000xKr>Akxo~@ySOb!)nk3Y0000000000M?j)TT~JrR2(h{N+H!rs0000000000"
    "O!zu^DN<L!eiv7&;DLR>0000000000ro{fA^;1{C;&cUE=8%2B0000000000G43(F!Btnl{!zpY?4NzW0000000000$-NlSjaOH|fH#xB@veQq0000000000"
    "N__X*Sy@-WGOlrP_q~0<0000000000lyVZ@C0kd();?b&{mgy90000000000gXyi;@myEHF>+hg0^EJT0000000000_9+g@y<S(q3(;kV2<v^o0000000000"
    "#(a0OiC|a26Xbn44*Gq-0000000000(JC&KRbp4b6Evgi6A6C60000000000_k4D4A!Jv;(Mi>v85n-R00000000006TFl`?PXWM46YMUA18jm0000000000"
    "12N4Fxo20vjp<ziB{qJ*0000000000rX*z3glSj62!*q>Dn)+40000000000)8CzvP-|DfLkSdTFjRiP0000000000ZL7XZ9Bo&?1pj^+He-Ik0000000000"
    "P0$Me=WbWP3h(pFJ92)&0000000000RhP!Bvv60yA7XNTL4kh20000000000T}d`oe{xsA^x*k0N05HN0000000000MK7NCN_1DiRrHDDOrL(h0000000000"
    ">XAa17Is&_0wp4rQm%f$0000000000D7t4c;&)fTzWc#QSiOG00000000000-0)Gyt$A0#PyuQCUCe&K0000000000=6*$4d3#sDY`zw*W7~ef0000000000"
    "AFYPxM15Dl+`GzPY3hE!0000000000XT(`@5Pw&|VS<DcZ~1<}0000000000n+*f{+kjWV#+pOMbO?XI0000000000mC?q2r-E0&xs3;TdKZ7d0000000000"
    "G+F5YbA(sG1?0~wfF^&y0000000000R_(5UK89DoZzi1Hg*AV`0000000000+16YB35Zv~w%u!yi$s6G0000000000m|iq`)rnWYrjkBHkyC%b0000000000"
    "aLs)5po>?)>&+AQmSTUv0000000000K5ThyY>ijIQW)#0oN<4^0000000000<AZMHH;-4qqWZ&JqJV$E0000000000HeH}x1Cdw2ll`3!sE>cZ0000000000"
    "8bMRj&XQNa`l)!rt)73t0000000000XL!p=nv_?-e27zZv#o!?0000000000{!2~3WtLaK>>(~Fxx9bC0000000000xC9+HF_>4t2>%D$zRQ2W0000000000"
    "cI$qw{Fzt4lBnQ~#M*zr00000000006iK%x$eUNdRj0Q=%ISZ=0000000000ZDtCflbu(<0fdb6&-j190000000000WsJNUU!PaNvSTmP>H~nl0000000000"
    "&I1ycDxp`v9Q~H8Z3}?F0000000000jTKK7_M%t7q0;<?@DhN)0000000000e+syj!K7EfsHk9Ha~goa0000000000ek+j}jiy(?h(p3S_91}40000000000"
    "XTctxSg2RPrDq`$c`1Ov00000000008x^G{B&t`yV%dG``!ImO0000000000c5i61@2gk9Bwg3UemH=@0000000000P#LX3yRBEiKesKO0YHGj0000000000"
    "ja#?WhObw^9;|?OghznD00000000001%JzHQn6RS9{|u$2Ty>&0000000000m&`2(9<x`#qIVlAidKNY0000000000A&nxZ>9kkC4%cD>4PJo20000000000"
    "e+lkOwYFElyyB(Yk7a<s0000000000j*8UvfVfw{4PWH75^aFM0000000000Em<0+OuARVW)L2am2`l>0000000000HogN|7`#`&CA(5+7<_=h0000000000"
    "iUQRX<h@tGvm=5*n}dMB00000000000&mjQufJEoWDcwv9*cm$0000000000drUx`d%;)0o<Gp@ppt;V0000000000*5hJpN5faZ#Cz_{BbtD~0000000000"
    "@O^$j6UA4+IEMzNrlEkq0000000000qGQ(_-^N$KS|%8ODXD<K0000000000&E2s8tH@Wth8-hVtgwK<0000000000OEpsLcFI@4A2BB}FSvlf0000000000"
    "`?CDsLd;jdh$SQnvcG`90000000000x}Wvi4$fD=A-@^pG{=Cz0000000000WkiAC+Rs<OPr?Vhx6XjT0000000000*y-c#rqNfxYV+-sI@W-|0000000000"
    "@z`MmbJJJA*cQxfz1@Jo0000000000j#@t?Kh;;j`S77eK;?kI0000000000jJLo|3)fe`Ge&P8!|Z^-0000000000#yBd0*VtFU+}bnvMf8Bc0000000000"
    "8YjTMquN)%TXXNw$oqi60000000000Wy{6}aNJkG64aroO9O$x0000000000hHipcJl<EpV8Kd)&I*CR0000000000TUj@^2;f)1tpDj+Q4)c`0000000000"
    "yVNNl)!|pbOH+R{)ER-m0000000000iYcCyq2pJ;q<{(xS0RDG0000000000prF_kZsk|N8cd7i*(ia)0000000000-GUvVI_Fow3wi;)Trh#a0000000000"
    "92v1W2kBS9-Og;4-#3B40000000000KCIW?)9P2i=MK$nVn2bv00000000008->-7pzK$`kNzP><wk+P0000000000lI-qJZSGgVIAUuaXikB^0000000000"
    "gezzqI`CJ(K>n!q>Q#Zj0000000000$%=*c2J%<H4wu@_ZC!!D0000000000KvRp})ALur`SAay@MM9&0000000000$IZRep!HY4XwVpcb8LaY0000000000"
    "I74CCZTDBexcMYk_H%*20000000000bo{yOI{8<?QZXbid3%At0000000000RzeI92>VySj5`(w{DOhN0000000000yj`3?)%;h$&fNOmeu{y>0000000000"
    "f|ss=qW)LFb=lFj0g{2h0000000000ggA%NZva@p<Mo=3gqeZB0000000000rip4ZJpx$3f%a8q2cdz$0000000000yVIPc3ItfdqMrgjil~9W0000000000"
    "re9(v*acX?t+k*P4X}a00000000000M0WqUr3YBR1pP1VkGFxq0000000000abUPya|u|$20F0B62F1K00000000003un<0K?_*G8!;iAmBxX<0000000000"
    "?b43e4h>kqm!^Vq7|wyf0000000000_v|#W+zwd4<6hKCo7I8900000000001O5k`st{PfULF=A9^HY!0000000000?@mmacoJB^XsJi}pyYwT0000000000"
    "n<z%9Mif}UX~byIBkX~|0000000000;?p_I6c$*(w=;RCrt^Wo0000000000qM2&~;}=-Ky1RIODf@xI0000000000w89@uu^Cvv%^+u0tOA0-0000000000"
    "`Inlhe;Zi9RVPF&FA9Rd0000000000P+}k(P90dlxswkAvJry70000000000$8gf193NP~Zb`}6G#P@x0000000000B{V-e>mXRbtFCddw;+PR0000000000"
    "*^$oXxguD=^h*hcIw*p`0000000000OmFC=h$L9R&#91Iy)S~m0000000000-?8*|S0-4%uTTgyKsSQG0000000000`951&CMa0I9LaDB!#;w*0000000000"
    "|NDbX^(k1unVZVsMMi?a0000000000SkBf@#4A|9qEHsM$W4O40000000000ZJWDdlPy@l!f;fLOI3ov0000000000nn;?2VlP<0YlMYk&Rl}P0000000000"
    "OuO2&F)>)cBNVDRQDlO^0000000000=h)=(05e#?aHzr$)N6vk0000000000&4wU8&@@=U%*oH?R&#>D0000000000YO3{>pEg*)!vNB|*?NM&0000000000"
    "ASulGZa7%L$JNP^T!MnY0000000000ROkz7J~~*yVbQl|--&|20000000000Y|R|w4Ln%D5PY0HVv&Nt0000000000&An!4-91>qR9<)!<(PuN0000000000"
    ";hs_TtUp-5wg61&XP|<>00000000002UbOld_h>is_PfO>ZgLh0000000000rpT!)OhZ_}w5QyZZLfmB0000000000C*0=J8%0>aTD_WT@V0`$0000000000"
    ">~&;;>qc0>7`0MBbH0MW0000000000V0(%-yGU5TcNYZ~_Qit00000000000;?cnIib`0(>};~@c+G;q0000000000+Foz6T1;5L|2<p5`_zKK0000000000"
    "tq1;hDo$9yEE52ie%yk<0000000000!?Dys`cGKE{B)sg0px<f0000000000enMLh%28Or=<7d1gzJL90000000000N1yrMnNwK6aRA&G2lIl!0000000000"
    "fvp#_X;oOj96NUHiTZ-T0000000000ky0IqIagS~Wh4*73<86|0000000000<8yLe30YXc*hHt8j|qdo0000000000)b@Tg*;-h@<|IFE5)p&I0000000000"
    "&A_S*sa#mV7){qhl^BD-0000000000GPhmhcwSh*?nQ7J7$Aed0000000000Yrp@vNMKmN=$8ubnkR$60000000000;J)~b7-Cq!jmxIQ9WR5x0000000000"
    "@S3n*=wn#GQfWn+pf-cR00000000001UZi_xMf(tzxn2GBR+$`0000000000gMR$@hi6#8RRxSirbUCm0000000000&ro2%S7}(lkQ6l=DNTdG0000000000"
    "P?aKsCu>;1{IBBht5k!)0000000000Y%!}q_iR|e4PceVE?k4a0000000000hzer(#%@@^N3Bhqv15b400000000002nMgImT*|WE+7nYG;4#v0000000000"
    "Q3?7}W^!1-Kta(*w{nBP0000000000&H1YKHFQ|O1(}^4I(mb^0000000000+`K7~1$J1#@~?68yn%zj0000000000=QrLW)puCHjt)-AKZ%3D0000000000"
    "R8TOhq<L7tT0JwL!jOZ&0000000000i0g+bbbDC9+r%MtMVN!Y0000000000<nBzJLw#7lhAkRP$e)A20000000000)<5S36MtC1<^LQZOQ(at0000000000"
    "y~Dm@;(%DdcEl?8&8~yM0000000000{a-t?v4U8@xC2DZP_~1>00000000001^UGHfP`4UHb!Zq)4hYh0000000000Fn7Z&P=;8*U=*8qR>gzB0000000000"
    ">Aap)ABb4M!S>lt+028$0000000000m$1HV?ul5y*>xr+T-1ZW0000000000oHvViy^C1DC>nwM-rIw~0000000000UpH)VjEz{pHEinAVdI0q0000000000"
    "M3u=}TaQ@4a`a-U<m!XK0000000000vNj7dDv?;gYCG$HXYzx<00000000005CNI__>x$_oXLY$>iL7f0000000000!P(la$COyWiYF~CZ2^S90000000000"
    "ET%Y5mX=t+xZURi@Cby!0000000000v`ZS@WSCgMnsl_<au9^T0000000000!msC6Gn!byy>yeY^%sP|0000000000y0-|t0i0OCoo0oIcprqo0000000000"
    "1WDx!(4APo!P1If`zD0I00000000000ZHXWo}XC2o<XEGelCQ-00000000007oBivY@t}dyF1bi0XBrd0000000000uPX0#I-^*?nn@MqgFS@60000000000"
    "F4xFv2&GuSy7O(k21SIx0000000000_=jaj)}~m%qQcyhiA;pR0000000000a3U=Wq^MZH$39|f3{-@`00000000000gQ*WajICrvhe;wk6VPm0000000000"
    "531ZnKde~5;9#>F5@UqG0000000000{9%;446azf(-(m9lxl>)0000000000F_{(?+OJr^6P{(t7jlHa00000000005;e<Ir?FVT6MST%nt6o40000000000"
    "1{)f8bhB8%TsnPv9f5?v0000000000axiOrLA6-GZeg%epooOP0000000000yjmt?4!2mq%6a}OBano^0000000000NPn{}+qhW3^9yJLrk8}k0000000000"
    "e1m$}r@C0cW1#KZD4&GD0000000000zD(m|bi7!=p1+T@tEPm&0000000000bih^5K)zVOCh|^=F0O>Y00000000000K_>O4Zv8ydB7`Xv9*N20000000000"
    "&^-4;*}+)ABC{qxG`)nt0000000000KoIszrNdajlTtt!x5R|N0000000000x+9)3am84`Qzv=xIn0E>0000000000qg+??JjYnT<GInwywilh0000000000"
    "WT1(Y2+3H$%aTf@Kih=B0000000000ThyEm)XG@EemdlR!s3L$0000000000^xgPmpUhamgE^m9Me2mW0000000000kfH^aYR*`|TNr#X$nk{00000000000"
    "ng=VPHPBeViC=pQOZkMr0000000000bn?)J0Mb~%ke8k1%>aeK0000000000gh*dQ%hOoE=SbtfPzZ&<0000000000E$=<qmDO0l6#Pt@(+`Ef0000000000"
    "*vr&LVAoi{nK|2ZRu_f90000000000>BEbQD%n`T^NEg6*&c<!0000000000%e=~_^x9a!uXSfCTqcFU0000000000+MD^8zT8;AJ%wfk-z|l}0000000000"
    "fkuE{h~8MhB|(hfVKs%o0000000000CZfjqQQ%m>>M_~7<UEDI0000000000E0i>88{$~N2H;JXXGDd-0000000000UlggV<l|Vtb7$&u>Pv;d0000000000"
    "g}IE!t>sw2OscC+ZBvE70000000000(mv?EcIQ~YMGuWC@LGky0000000000(U8WMKj~P&6Z?q<b7F<S00000000005u7qt2<uqDX40MF^=XB`0000000000"
    "A5Av`(d<~j`1!-WcyWcm0000000000gOJ~yneJG?Zr1mj`*?-G0000000000%u+TlV(?hNgUmE{et?C*0000000000g@8u8D)Lys=RkH-0f>db0000000000"
    "ItCj@^Yd81P?)zbgpY;50000000000bkM%nyY*PWXgBl=2bYDw0000000000!pl5zg!fp$=z=HciJpbP0000000000ujHQ_P5D^Be&0;Q45o#^0000000000"
    "#|QJ+75iAg-?3kykFABk0000000000mmxu~-uzg=!CqT@619cE0000000000ryiA~r~X*L%}P61mAr+(0000000000hU@vdZ~$4ry4w0P7{rCZ0000000000"
    "z;t^0IRaV0J9Cu}o6Cj30000000000;HDvG0|Z&X0(l1O9n*!t0000000000ac=qr%>`M&#864bpxTAN00000000000Jsm_mIqnDEMHlnBjSa?0000000000"
    "8gOjxU<p~k@CY(|rs;*i0000000000MuQtdD+^h`!LrU)De;BC00000000007~Ut`^$c0SR>3MUtoVh%00000000006>Xfszz$i!VVOY-F93$X0000000000"
    "$sC0Dix64Bi=PhVu?L300000000000!`bPAR}xvkjyR3JG!KTr0000000000j*!@BBNSP{8p*hqw-$!L0000000000v@Kh%?-g0V;<b!$Iv$3=0000000000"
    "#hrXuycb!(onr(^y(Nag0000000000O3g=Xi5XeI^fntIKrM#A0000000000&8WxjRvTHsos`V@!Ze1!0000000000+#CnxBpq46N#PUFMLdSU0000000000"
    "0~Qff^B!5ivw`fU$U}y}0000000000&w%hi!XR0|g4$7iOG}2p0000000000%1N-Wks?{ZY*0T_&QgZJ0000000000e_dOsVI*0=BS^3)QCfz;0000000000"
    "dZ_#?GA3ETStg+U(_x0d0000000000N*A9)11MR*$WIE=R%wR70000000000c$Q7^)G1lO7Afkb*>Hxy0000000000kP;I+rz=^&`W8ERTzH1S0000000000"
    "Ab9#EdM#PNCw}-#-+zX{0000000000v9k`hO)pu%Ps1G^Vuyyn0000000000(y?0eAu(CNDodvC<c@~G00000000003t#Eq_A*((UwVzeXO@P*0000000000"
    ">Ziql%rsfRrM1qG>Yawb0000000000`gr=?qBdE;uwi9jZKa050000000000$-a{Lc{o|XGviDz@T`Ww0000000000;bocGQ94<`;GTf}a<qoP0000000000"
    "&RQ9RDm+=hWcmBb^}B|^00000000009Eu(L13p>6cUQWXc*BOk0000000000R^YHk-alEu#vQj~`^tvE0000000000>7emyxj|XLE12yre$s}(0000000000"
    "Wc?|El|xy;KH5F`0NRGY0000000000tOm5$az$CdeVRwYgW-n20000000000*v}=xQAb(8ZbG4n2I+>t0000000000_-hrCGD%s$hCAU-iSUNN0000000000"
    "Anq8P6-!ybNAH~x4ETn?0000000000U1`QM`%GED9mj3aj{k<h0000000000#_+SO<4#$?i!A4j5eJ9B0000000000YL|Cf&rn&w3W=abln#f$0000000000"
    "U2vz6zEWAh5=Jol78ZxV0000000000u#VZHu~b>WB+V?VnH`6~0000000000bOog3s8(6Pu!k&G93_Xq0000000000yXTy#q*z(NG2g}eo-BvJ0000000000"
    "nMhhirCM3QC_RsxAvA};00000000009XoOht6W*Y2MZrHq&tVe0000000000T+(_^wq9AlQ?d)lCPRn70000000000WFZP=$6#5&d7SH0sY-{y0000000000"
    "N;|fn-eOt6`7%}5D^iER00000000009y1kF{A5|cO5fg8u33k`0000000000^c%g>AZJ;?><S;rFky$l0000000000-Uvi;OleubSRbr3vuKCF0000000000"
    ">I^`Ie`{I50EQfwHE@T(0000000000EjveFx@}p&WT*!4w|9rY0000000000yQc3`{cl;o{_g-kI)8`20000000000p<>UrNOD=gO2(msyoQIs0000000000"
    "^MQQZoOD^hyOe*uK8}aL0000000000#21Yf`gU2s&%`k7z?Fx<0000000000B{!G2V0l@<|G!HQL!F1f0000000000Ci5^D(R*3I#P4P%#iWP80000000000"
    "-d^yBOMY3vpuHF|N34gy0000000000`>m=#(12OMUVIoY$+L&R00000000009_J+NUV~Y{L4D02OS^}_0000000000KZ%3{_k~%&Q>WPm&BBMk0000000000"
    "c+BPln21@x&+FddPRfVC0000000000<>+W+L5o?y_B(&F(9ws$0000000000pcW{g@{L)*_J~}0QrU;V0000000000!Hq#NtB_g14#XHY)ZmA}0000000000"
    "V}{a^YLi*Oc-~d)Rp^Jn0000000000qd53GFqT=sZ^^Zp*YAhG0000000000ncUB*`<Pk4F?QNNS@(y)0000000000U#fOh%$r%j@YQAC+Wv>Y0000000000"
    "5ZVd3p`Ka5>e=~tTn3210000000000#iOQ9d!bptS93TG-VKPr0000000000n?GuhSfp9Nbd$rJUloYJ0000000000tL1U{IHy^_e=9vG;T(v-0000000000"
    "4bIvC9I9EsrYuLWVkC&b0000000000-Y~)=0j*iUEQo_O<SU540000000000I7UU==dW47OD~kTWiyDt0000000000Ge&`-&$3y-JQEu_=Q@bM0000000000"
    "?r4+nx3yWoHzm5dXhMj<0000000000edd@#ptxDUbygEQ>Pd*e0000000000|MspCh`U+9G?>V{Yf*^60000000000i_-GfZoXN-soQ%!?O2Gw0000000000"
    "Ke2%iRKZ!m7!uFGZeWPO0000000000GRt@}H^f=Mu`Fap@Mnm?0000000000e^~Xp7spw^u^ipbac_vg0000000000-j)<2^~qVlOVit0^LB{90000000000"
    "mhhSc&&*lCPsrTrbbg4y0000000000Tk@xJrO#Qw{Km|F_JxSR0000000000*=KK^c+y$GLM?U{c#Vj_0000000000zZ@mCNYz=u6<MRP`ILyj0000000000"
    "w%q*b71&w8X{a7hdz^^C0000000000baN_f;M!ThHF>h}{G*7#0000000000q^#j<sNGq>aw#d6eyfPU0000000000?o(D7Zs1wK2Nz8}0JDg|0000000000"
    "1vR{+G2>al@So`Bfx3vm0000000000lp9Ar^W|B<9wNq)1Hy>F0000000000O4BM9v*=mCiTkfWg~^D(0000000000)+d`>aqC&YCyH?J2GNMX0000000000"
    "<mAgiF78>t;VaUih}ek00000000000AJ*{o>F`;=vp%0x3gC#q0000000000ImbPzr1M$8kszW7jOU2J0000000000;DhFiUiMkQaxbO54eyA+0000000000"
    "z1JSj7x`JhLCJD+kM@Yb0000000000e~ucU(EC}y_e<3&5&nq40000000000)N>6oiT+u@h836Kk_Cyt0000000000Zc{6?K>=F8=YxWm6b*^M0000000000"
    "_xPt)`U6_P4)tM9mK2G=00000000008!)#?vj$qg=*Is97#xYf0000000000hQy4$YzbPxYgZ<_nInn70000000000?txt5CJb7@kOZJ`8!L&x0000000000"
    "_%mAQ;SO5BMl3faoid5Q0000000000Sf;uEoe^5ViY&I+9y*D@0000000000x1_FETNGNrLSyHOph1bi0000000000#rkX`8W&o?YA%&KB1wtB0000000000"
    "Ep#Z5+8J8F_3Mf1qfm*!0000000000f@l1~oE%!fWc-|$C0L2T00000000000;i*`U>{n*r|;WBr(cP{0000000000bpN|FCL&tEcW`3tC})Yl0000000000"
    "qdVg~?Ic>jn<-3`s&0wE0000000000QYN>mwI^D@(Z#enEOv>&0000000000NQ%{Be=1tQ+3&aCt$m5W0000000000PBPp{N-bKzeo}FRFNKM~0000000000"
    "Ch2HI7BE`Ca+v%jv5bkp0000000000m3bLA<1$*ndTe#aGL(tH0000000000WUh+_u{B!2T6y<pw3~^*0000000000SyEk>e>hse)@hyxHlvBa0000000000"
    "J1{KHPCHt_rs##Hx2lQ20000000000%o3K59zI&YjiAXyIkJhs0000000000*Tgu$?m$|=PdtU<y19wK00000000009}%)dze8HUt7g=HJi&>;0000000000"
    "Y{H)bkVaa-XUftXzQ~Ed0000000000gQK6@U`blQITD<?KhTN50000000000Cx1`3F-%&(=;%vU!Pkkv0000000000CLm3D0#91NHu?SXLf?tN0000000000"
    "K~Xja(@|Q$;O)(i#O8^>0000000000KHaE7qf}bJu5s5aM(&Bg0000000000<BmcTb5~lxS1&fo$MuQ80000000000_AsGbLRwnDpV0+qN&Shy0000000000"
    "K#pPq5nWoqO9NsC%LIzR0000000000gK;G6-d|e46>nyxObm*^0000000000ilv16t72Nf%D*l}&J&8j00000000007xf?(cx76^9H7?aPaBHB0000000000"
    "@hX=fLugvS(Ykqu(ISe#0000000000-4TWV3~O4zthzHQQ!0wU00000000000Wlev)ooh9tf(K()G>;{0000000000U7V5coN!veVz@MORym5m0000000000"
    "uwm0XV{=--(!4Vn*g%TF0000000000$Y0d9D0W)FuKn1*SxAb&0000000000X?2R4?RZ+ip~B#1+fRzX0000000000WP$4-vU^&<Wm5+ZT~~_00000000000"
    "eq!x|c79sGq4x;2-d>8p0000000000g!hI0I)Pfi2=}2|U}lQI0000000000J|C;c|ASh<PMMAY;%$n+0000000000dAL#>#fDnIAtsBaV|9wa0000000000"
    "{h@L-i-}smIOS7H<$Q|30000000000l)3;-QjA)_Iv%C$W`v5s00000000003*qvd8jo7Q-55KK=!=TL0000000000DfB*H<dIsy(Hzn%YLkk<0000000000"
    "xzgIruasKA#&IUd>zazd0000000000hFiQMeV1CmbYJvgZK8_60000000000Ty5ExOqyE2fi3v>?x~8v00000000001Lr4a9-Ugiqa*H&aIuQO0000000000"
    "M5mp~@}FA3iHVXO^0<n?0000000000?2It{$)Z}o=BA;lbHIwg0000000000#CFy-q@`NGZo}6z^~Z|90000000000l|zNGfv8%*$R_c)cF&5y0000000000"
    "E7CHlVyjxfs~D;|_|}TR0000000000P65?cN3L4H$l$)UdESb^0000000000&#z(hFR@y{)W+*9`{jzj0000000000Z@kXC9JE@%bQ0~JeC>+B0000000000"
    "`~?!{4YyjrVGDl?{`88#0000000000L2Dtf1G-wkNcmlIfBcHT00000000002+Q(N{=8bi-XliS00fJ`00000000009t3;?{=Ztl*wXVYf(whl0000000000"
    "M)k4*1H)Rt*-ayY0~3qD0000000000P<@Ss4aQo)o7|Akgc^&$0000000000VrS(B9?4q30Q6%O1|o~V0000000000V+zcQH_TeV;Yp!PhbfD|0000000000"
    "8$C&$Ue8*<UJajW2r-Mm0000000000^TfE*m(p6m!kwmMi8zbE0000000000WM_80=ha%kXIcR;3P6j%0000000000-s#=#QrTL-rK0%3ibspU0000000000"
    "*%3?k-P~Hh*VWYz3r~x{0000000000$fOx;jo@0qRa4>)i&l%k00000000007HFm3W#n4GYkFt23SNuA0000000000Iw4BoYv@|Qc&e`TiDiqx0000000000"
    "s+D5~r0iP2(Lw>p2yKhN0000000000)QU~u67gEV%%Z3uhjfd;0000000000G{$2Az4cna%qi~l1bmCY0000000000Km^CJrutgIU}C;Hf`f~|0000000000"
    "j=r~8#{XKt%#DWz|B8#i0000000000G(9Y;{sddVr^W@Cdy<R50000000000d;Uh+FbZ41CEP$V_?e5q0000000000rGP3OJ`Y>KZk<3Fb)k#E0000000000"
    "|3Uet2NYYtxY-Pi@~Dfz0000000000*1*-PZWvp@HejvOaIlNO00000000002*e01UL0G%4Sp0F@VAS=0000000000^!n<R=pS3aFEd&>aKDSd0000000000"
    "SklfyDkEFK8mB(>@WzY40000000000V7La1KPFqiVgZ5?a?Xpu0000000000O!KrdL@8UqRY)gO^VN&M0000000000qfbI|Ml4&vyP@Hmblr=<0000000000"
    "|EaANM=x8ztsLj+_2i4d00000000002G$5vMlxH#FPx`5ckGM60000000000Vy9(fL^WH$TmzGn`16av0000000000yYbb1K{#8$KQ$TddHajN0000000000"
    "zKCnrJv&>#?G&m@`vQ!>00000000006F@kqIX+v!dWE&Cd<u-f0000000000Ve3sNH9=dz>}jhF{t=A80000000000QC_}_FhpCxS*81Me;JIx0000000000"
    "j%=F@EJs_w&>9xi03nRP0000000000!s+(lCrVquWvyyCf+&o@0000000000mZ{=DBu!huEkJ3V12Bxh0000000000u77Y~Ay8YuDYk(Cg*S}A0000000000"
    "wm-6YA5&Yvbn+u&20x6z0000000000TD`f#9#&hx9@xLZhenLR0000000000LClzPA6Z+#J4(eK2~Lc_00000000007C>RZAzWL)(<`KSidBrj0000000000"
    "d+a3RC0|>>`(Mn?3tf!B0000000000A*J;ZEMi;0!pW^5jbx0#0000000000u07?0G-X@BJ50E34s49T0000000000%!vhXKWJOPddge1k8_N`0000000000"
    "Bwc6?O>0}ghbIX45POWk0000000000U5g)+U2a>zcSbETl7fuD0000000000A?Q3fadBI~SPDC46N-$$00000000008b{JIh;&=PK31WUl#z_U0000000000"
    "^n3BVqIX-sJ`K>e6`72{0000000000QCh4D!FpT3S`s(XmY|Hl0000000000;I$M~<9%Deu1H4Y7pRQD0000000000&qcb>34vR{VODMIn6He$0000000000"
    "8}W+MG=y8gu%r&^8MlnU0000000000pTw}(WQSY8!m4=On!b#{0000000000kxRc{nu=S%C^`7T8^(;l0000000000DG^HO)s0)gg&^9ToXw2D0000000000"
    "p`hgP7m-`Qcrj~f9o3A$0000000000HH9MlV3b?HpFH>{p4^PU000000000097>mlv6oxGpc2m49^{O`0000000000ih9-%3Y%NN2!uI!pX-dk0000000000"
    "yrFX%Yo1%ceqo0QAoGmC0000000000>q-J6)uCI!pnC*}p!$r!0000000000QvnvLNTplA6r=t1AOelR0000000000Bv3*x#;9AsXsEwYp$Uz^0000000000"
    "oWdG1POMwNKwOuaArXzh0000000000^y4xz;jdf3J&)1Pp%{(80000000000U_TzOf3sV__z}DCAs~&w0000000000Bw%nTDz{s}8?~VPp(l;N0000000000"
    "Y|*?O;ksMEG)d3$Auo-<0000000000bRL%urM_Fh<(`w$p*D@c0000000000d(x76cEVf0(}p^mAwG@30000000000w0J||RmNMumeCMQp+$|r0000000000"
    "T%0_@L&{sg)lXyOAWe<H0000000000rN|MTK+aphAJTqfpj3^(0000000000%!pR;OwwDx87Z*HA6$*V00000000002xy)@Y1Ui7U?oHopJR={0000000000"
    "mhpT9mfBmumzRe^9&3%j0000000000t+0i{)81RaTu{eVopO!90000000000fiO(?BI8@YLLg8~9D0qw0000000000QT$^igy&np@Yil7n}LnM0000000000"
    "C;|e@_v>50Gkj>^8HtU+0000000000h3avefACwtU^~xtn2?RY0000000000_;<$MAN5<nYueb=7MP8|0000000000^f$pX-T7O<1)&`2l%I{j0000000000"
    "*S8}%y#8Cj)YdVa5~q#800000000005QVRe!2?{tg2S)ckFJft00000000003N0qk@d#YNzOF5h47QEH0000000000A3aH*Q4U<dAmBo%h`o)#0000000000"
    "!$R7b<r7@MU7v;y1jUWO0000000000S<9MUvKd^!88ncte$0))0000000000Ok_-qydPY^|K;{`_|uKR00000000001e>tI2qs*>y8KZ}aodf+0000000000"
    "<ot-8n=D+w?7ui2>EeyR0000000000Sr?bSc{5zVKX*ddU+Rs(0000000000;8vG!r#f7~i0sQ})$xtM00000000004mQ%VB|==lAD8LBN%@Vy0000000000"
    "QoFmR^GRI5-AEzXya0~C00000000009~Amy5K>&gX?O;#E(ngm0000000000^=&f)ds<w;ZJaDIpAU|}0000000000QT7d;GGko8m@6k?4i}EV0000000000"
    "tJ*J0Hf&tLm7Iw*d>)R#0000000000g-Aamg>_uO5@zS4=p~N80000000000PHo~q9)DcGu>RAhQ!S3b0000000000hIYTZ0Et|{A>HCQy)=%%0000000000"
    "tdV@IE0bKn4S1(dB|MJ60000000000J?y_Lo}FC4Cq}j8i$jjU0000000000tg|mtSEyXT5Wb)A@Jf!r0000000000a}{zpR<m5da#s3lR8o$>0000000000"
    "6hAHZnZ8`W{)JdiwpotA0000000000cU=dkA<A69a>GrN7GaLS0000000000!$??J?A2VrkH=aub!d*j0000000000u5Ky=`Qco^24<Hg(r=Ey0000000000"
    "h6ApWM($j|Ao}@mEq9K<0000000000kvkLi)B0S%7~gmfhklO000000000004NHm}pa)&RE^rF{-i4090000000000OJ+mysT5tnrwFAwHI0tI0000000000"
    "d%<pH@F88mxdT7NiIk4O0000000000=#V^SaxY!LrM%9X+?$TT0000000000*a+R>E<Ihqtd2^#Eu)UW0000000000j$MGpBuibu4ArO}eX5SY0000000000"
    "Si@EVQ(0ZW4C~*>%CU~X0000000000Vs4OwxMy9!+AHR@7P*eV0000000000{gPccRCZm!!1A%<V8D*R0000000000YGv*YC52tU`<gjWsmG4M0000000000"
    "@ELRUDU@Bn%ztzF@6L|E00000000005kFt+VWVBZsVc<WG}ey50000000000e}Yie&az#=ADbKBcioP_0000000000Fl0`JcEVl22{Jvzxa5w&0000000000"
    "|KA7|U(#K`N(PTj_v?<p0000000000(eW>yj^bUw%`SI5GxLtX0000000000W|ea~3-evTZNghpZ2FGC0000000000r^bT=+XG&}6xgCrq5+S<0000000000"
    "Yk?3G1QlMut%kDr)CiBj0000000000jKRr*j3!>d80GK_1Q3tF0000000000@q%A$dN^LdMGC}AFBgx%0000000000T)~0q(@I{z3=NWbRvwSQ0000000000"
    "ZQGg-pIu(SqCLu1c_oj)0000000000*x6DC;Ba2RhYmBSmn@IK0000000000H?c>NmxNxxbSjfLu``dr0000000000Asc)S#+Y8fcY;yV#X66`0000000000"
    "2hsB2Ypq_umIu{C)j^NI0000000000evGbaiNapM-KvCD-$;+Z00000000006bG^mAJ|^NSzX7|<4=#k0000000000P5S>UE$&{x6c7Z8;#QBq0000000000"
    "%O9I3v;<$kAGD!E+g*>q0000000000^Du^Dv>RW*d^!ir&18?j0000000000U+2R<E;C=iIh333xoeNW0000000000p8QrGB1>PuX0pY0pK_1D0000000000"
    "G%7-4kYiuK35M)1etD0;0000000000!YPPvd3;~MGyr$yRe+Dc0000000000th6ho-IZU!=n>3fCWnu}0000000000j)X5}y{})uG^|o{?~RYZ0000000000"
    "^-`7;8OmS4DZp{!u#}I$0000000000df8$U@#J5?&pwcQYMYP10000000000!syH3A^>2(E*TXH9HNiF0000000000So#~n3l3nwQ$S}~#i)<K0000000000"
    "ME)XPGaO*RNTuyGX0MOH0000000000N>SqFnJQqw8#xbX{<V+50000000000JWyMTKRIB)-0U*Gjl7S*0000000000$XAbiBuHSuim|@}6vL0e0000000000"
    "zr=s-NLOIMJ9r!ck;#w10000000000<nt{Kt!7}r@-eW41kjJb0000000000079D&QFdU!z|Y}WZq|>$0000000000T4bCVHHKio#1Ml@&fJf{0000000000"
    "l;EMqSe9VGxq=luB;$|30000000000-`HzMyQW~ktzZlxaOsc00000000000@z2l0Teo1q4LeTlvG0$-0000000000gGi%iImlqZJfgvr>GhAm0000000000"
    "QN_DmQ`um^qb5xY7yOUF0000000000`^fIlsOn(AP^@br9RZNQ0000000000Le>%NIQ(G1QDmc<DFl$f0000000000+Hd;V0uEuoh66$9Fb9yp0000000000"
    "h!}pw1t4L-CDJzqGYXKv0000000000=u&doKr&&#T^M`oFAb2u0000000000xS^~-vqWLQV;=sLClHXp0000000000vWEgDTvuVhY%H|<855Ae0000000000"
    "i-$?HI%;9Stcs8s2NsaP000000000010+`#PJLm(RDp)*?ii530000000000x8DGFm5^b;msp8n(HoG!0000000000YCRkQ5Ts$isaTAGt{#xU0000000000"
    "(7%4)ytiS%y~%<<h9Qu^0000000000pXvQPo6BLq3G`o<SR|0Z0000000000oIP(us^4M2z#kWYB`1);0000000000UOuQG>GNU05|x|w>?x4I0000000000"
    "s&0P#RtRFiGo0Y_t}Kwi0000000000&K{-&^BiKpGJ~dZYcG(%0000000000;ZwWqyfI?H;bk>jA~KM`0000000000VE5upwMJsVXhG70(=?F40000000000"
    "Y%7A$+*@M6lihy2em9W70000000000P7?kgG;w0UFakZ+BRY`40000000000L(aSEyM|)G4tEZs!aR_`0000000000hk+p3bev+q`2+RvT0fA$0000000000"
    "WloWoU$SDr!f^7F>_L#g0000000000{Z@0)e8^(JG=&IUcSMlD0000000000n7u19&E8_b9tk-b`$mw!0000000000a-<(+QT1ZLOT>krcS(@I0000000000"
    "3`<`X3Jhbw=kxNs>`Rcq0000000000$$L((_abA!|9@3QT27F_00000000000GkaL8aiXZ1C?|Sz)+CD0000000000PLS6gaZzKy;trhh9aE6N0000000000"
    "SVQNS{Agpqn8_E$b5)SQ0000000000$eVXpyntiC4xuw)!B>#L00000000008-c?tu$W`OJaIdl23nB700000000001=|Jd*RW&25*B+)LtK!+0000000000"
    "ACY-vGs$DXg=;=ZcV3Xd000000000042xN7!r^1UhfUFIqF|7~0000000000cd+2mg!*H^3z&b3#A1-Z0000000000V(>_6dJ|;8w4RiF-DHr!0000000000"
    "#yM2^qAX;<DOLCp?Pid`0000000000IR_CK{zYWK)#I50^JtL30000000000J=3&Si(h2GGyskP?`n|10000000000W*BuvN_k|!#b4>c;B1h<0000000000"
    "6H$kBI+J9;2DrfX$8M0o0000000000*!LiGTdickbiOE4q;QbH0000000000Ifr-DtjA=)kSI2ab#jov0000000000xOu=OF5+as)T{kYJ9Lo10000000000"
    ">q#;A-uz_1#r>01^>vWI0000000000Epq?cz87V{+tUqBqj!+N00000000007j0l+%`#=cm*ao#M0t?F0000000000D7A?g3QlFfcH;i%*?N$`0000000000"
    "`NXi1bZKS4`h}nHVSJFl0000000000<Q5BE425ODoP&0_+<lP10000000000ekINd(V%6(-H@2`Nq>;Q0000000000O=~YJ!n|d`JYyPFseq8c0000000000"
    "(PCsy-`HiqHn0$o`hk$Z00000000001J!{9DD`E)Zcdw5J%f<I0000000000?jMy*n-ONfpvE|mafFb-00000000006>SoJHZEqsri+@cmxYkP0000000000"
    "_fUbO_DW{JIkKTau!fMp00000000007GP<|*k@+I3r{cDyN8g#00000000003NKRh*@R}m(sm>PyNHm$0000000000D^kNW_n~IMM?PXpu8EMq0000000000"
    "3|>T&Ex%^K8oJ3vmWq(T0000000000GUr?xdfaBf1^+vLbBmC`0000000000AA*$B+4*L`ydj!~MvRca0000000000X~S~NNf&3p_cq1q4~>w(0000000000"
    "W<P%n$24caX;Yp$(2bD50000000000`nARCPEu#Uf*RWIhmMfI0000000000^5#aW;BRNZ!F~1lHjj|N0000000000Sm)1Ee2r(o$jJ8)-H(vK0000000000"
    "!uYlDA**M=<`}~4d61C60000000000hFngH(a2}OW{WjN4Uv$*0000000000I8^Q_i{@v*mOV4lmXVOa0000000000TOsS%P6TMc#7vw^7Lt&_0000000000"
    "g9VzP93p7IJJwNYi;|GQ00000000002trVb^gn38O!qY7^^%al0000000000S<QQf)?H}8tDA{|Rg;jw0000000000e2yF7!Fy=H%EfJYs*{kw0000000000"
    "KV`k1wV7zZuzX)E^^=gm0000000000Rdl?`u(xQy3o8EmHI$IR0000000000E8cTxvejt7hd1d-ZIqC}0000000000ZN)x<x$|hi&kn$Mo0O2i0000000000"
    "vP^g$#Sv-1kQpN9z?6`{0000000000nwEP+)Gukkc)_mu-IS2P0000000000%EhDm<xOe8FK`7B^OTUl0000000000wERm;`D<ywWRJy;0F{uy0000000000"
    "1YXZN4T@>N#E6d629=P&0000000000eZxf7AgF1;cjxsz29=P&00000000006g4YqGsbDaJ{o4!{*;iw0000000000*=m1rMC56}uEF_b@RX3i0000000000"
    "&)r36Qvzzh^#FUn+LVyM0000000000EC++iUm$A0Kl`(-y_Ar^00000000009bh{YX+CPexYKU^n3Ryf0000000000<zaKdZd_`>ldW}oY?P3|0000000000"
    "@~-t)aC&OMx3nMsH<XaT0000000000r3g)wZJ27ndo<pf`;(Bs0000000000NwE<|X0~d;VWYEkx08^-0000000000YTMvETGVR5q0FbYYm<<`0000000000"
    "oViy>N%CsIZF|Cm7?Y5|0000000000J$@57GZ1UQ{CW!qzLJo@0000000000=v%Sd7cOhSgO-Y9T#}H$0000000000ABD~W_e*QQHJL}m^O2Ci0000000000"
    "Z7c|V&}nPHN=`~mgprWI0000000000XN3D1q=#$3@_?Eq4v~<+0000000000Xun0Naiwd(WBfmNk&uwU00000000004iJR;IKyke){8=B4v>(*0000000000"
    "+^AFC`QU57eEO*xg^!TH0000000000R=iOmwEk<riW7dj^^TCh00000000008hsm<XB=$6HKN9EV2+T$0000000000fgMf@6gq6cv~i1G#f^}_0000000000"
    "BhZFWxmRq!I8D^9B#n^30000000000kl$Q2S9NT_XnCcFevFX70000000000o2D0H?UHQ3shW%U(u<J50000000000nDHqGd$4T4UKWeVA&Zc}0000000000"
    "Fm*;g|IKW``=%4MYKoA+0000000000P39jVdFyPz@&=W4t%;Dp0000000000#QA8+=Ll`Uy6}F*>4=cP0000000000dSuj4N+xZ<!5%?hABd2^0000000000"
    "9@`YrpF?fHeM?LIO^1-c0000000000l@i@!=U{EXU@;9^b%v0@0000000000w!hyZAbxGY-ir1wm4%SN0000000000P?ygqNt<oJX%crit%Q)k0000000000"
    "QBGfsVYqF;bb|gcy@Qaz0000000000Vwpi|Xw_}NbCvwi#Db8(0000000000Xe}w*T=H$e*ahL!!-0^%0000000000Jio~JJr8cc7mgeBxqy(s0000000000"
    "uLINE3M_8GqUam!rhkyY0000000000r+zt@!bonw<NI@Vihhv50000000000LJ=W`V`Xl@sY-iLWPOmp0000000000o^~Ks>w#{->-Q!)G<=Z20000000000"
    "q(vIBSDtRbDp7{>`FfDR0000000000JQeXjrn+vx>!YYVwRw=h0000000000Q+$W3)Yfjmt3937Wq6Rl0000000000vk}0Y;PP(3^J{Z{3wMyf0000000000"
    "QEYv@#}046Lwq{-rFD?N00000000005hJ3rhbnKt9lQYmHFS``0000000000jUKmJ9!77#-7TnMxN?xd0000000000lIAy!jA3uU_4dk#FmaH;0000000000"
    "=#My;)qHQjXyc+-o^O!A000000000001C#|{FiUQ0T&#?0&bAN0000000000K1A)Q1+#C!leHqTUTl!S00000000004lcWH^UZI-?c$;mvucpQ0000000000"
    "tF}CY$>?vu-@=H~0BMlG0000000000!e+ms#Q|`@I(fk0NN1410000000000`!L4|9uIK9%qcb}jAf9(0000000000c+MA0Zya#IG<3dl%VUth0000000000"
    "A(p`BuqklBcZewM17eWC0000000000BNdZp&o*$t57ch8E?|(r0000000000x!u3RwnK2h(aUPIL|%}<00000000007=m~7Qc!TfndJ4-K3tH%0000000000"
    "iR}xbm0WPZ(Xn0u7+R3P0000000000<9KGgjA(Gc0hX6W)mM<f00000000009I+4iPIPd<6lko<epQgb0000000000OKU(5^M7!_*5T*198-|M0000000000"
    "bbK+gkBe}?484>vw@{G40000000000rMVlMJC|_4gtPr(R!)$>0000000000?)Nu;4Wn?t1%GXI0!)y=0000000000NesUf9ItS|UD+;>!%2|90000000000"
    "!IYTta=UQA&w26iokx(s0000000000I;bo*@5gY!DTM=gg+-9S0000000000@%_^pWz=xM%a~~5YD18~0000000000OiRlX#^7+khN>nYNkNdn0000000000"
    "E*44y6YX%o*i~<YAU}}60000000000u+&EHO!;uY%!ux8@jQ^g0000000000<ovhYbOmw1Z81<kygHD;0000000000;RKY_ixF|a#yvn)fjE%B0000000000"
    "t76gFk{xlt=9_8|K{k-U0000000000R*YuziYalx);W97`!kTh0000000000^Y=+7bv1Foq6HrHu`!Uq00000000002Y@PPQbBRRNwAXFV=s`v0000000000"
    "h14+hB296?Y{!9_5iO9w0000000000V7!mH=2&sSFa{PtyDE^t0000000000N725Uon&#qz2832U?`Bl0000000000;;c1;M{se#J;&=#0w$2a0000000000"
    "*yw9l<a=?z+b;~kpCgdK0000000000*B;sbbBA%j!i<kAIU$h10000000000bLjMA_L6bHBT;3O&K{7!0000000000Vc#}uYo2kyB=CIwUmTFY0000000000"
    "L!B1=(y4L4@!Z93?HQ230000000000zq<<oEwypLwTdt7b{CMq0000000000hSZH-dBAbNn{26i{1lMD0000000000E`b#Nx65(B(e)x9eiD$t0000000000"
    "Wm^}-=+<$-fticm`wx)70000000000()<mZ3*vFW)XLJmbq$cf0000000000AOCM@AnkF$`uyj?>k5#-0000000000_WQIJC-`x|7o)oGT?mlC0000000000"
    "@QhT79|Lm0UobjM%mt9Z0000000000yGzEa2@Z0=_!m~(HUp5r0000000000{?UPA<rs3n4$b|fodA%)0000000000X1V^AvLkZ9m4!i80sN1^0000000000"
    "j+?YCaV~Pey4YOjzxI#70000000000Arc!AAvtou-gztsckz$F0000000000$+oE{!b5VvSx=9@ChU*E0000000000CnMNcR8DfheLrLx&*qQ70000000000"
    "@bDds*I07EpfHLdaN&=@0000000000wYl=BOJj1tAPL~!3EYps0000000000DHqO`ux)a{POn5<oYs%P0000000000^P(p`26%G7gbA0UC()0<0000000000"
    "w)S>|OoDR27XG!qtILnT0000000000BCIn}gN$;(SaX24D8`S#0000000000(OlJys+MxVs87|Gp1_a50000000000ZKBg4!k}`%Re6(X4ZDxP0000000000"
    "wIjmu$*OX|A6KC^b+wPc0000000000w$B;#!L)L~K(7-F)~}Di0000000000iU_+gs=ji-^$-30FRG8g0000000000Y}o*}g2-~fOu!i(gQSnZ0000000000"
    "s!c|aOVV<{J_ja%&z_IK0000000000k8tM61l)4K)hpgP6`7B~0000000000Y)w3HuH|yT5Ow}tRFjXu0000000000dheo-NbhpM{V78Ljg61M0000000000"
    "2x4_K)A(}0n~`WVzlV>&0000000000Wv~uwP6Bhl?eZ!P>w%BJ0000000000*X7ASx(jo_{FFwa5PXlo0000000000tPKLc78P^A#J+a{Fm{i?0000000000"
    "BQp~9V;*zBRJVFUNpO$A0000000000jA2#op(k^|t;Gv?TxyTN0000000000FQg{7&@gkr(J<$=X=IPU0000000000RR-Z(@Hun9$Un&sabAzW0000000000"
    "Pm|y107G-YkU4Ida#)YR0000000000S5|#$0!(wjHf54#Zc>lH0000000000z*-gU^i*@ewh8izWK5610000000000(yIb}*j;nLAZ|$-RYs4%0000000000"
    ")>NP-t!8t;Y}r&OKtPYc00000000009AQcJa&L3Mq>|9cB|4A50000000000;M-XEDR^_h&fs_t1~iYr0000000000a?mL`(tvZo?g|PJ;VqBA0000000000"
    "97-3(Yl(Bf3RpniwkMCk0000000000C{%o?_K|bIBoEk)har!^0000000000;`y`ha+-6%I(LmSQyY)K0000000000ibffy;G%QDUDq7o85NJf0000000000"
    "Zbb*%KdW=Vi*`b)+YXPw0000000000*#X&?kh62Z%A2~Rm<W%+00000000008HefV(z|oO7>+^XPy&y@0000000000Yv7aJ2E=o~g_M(d1pJP`0000000000"
    "AQ>}%EX;Gj5%z3*v-OU^0000000000emo-EL)3G?z4?j!UGI*-0000000000F6-9nOx$z8Nv16e1nG{z0000000000Y#t2iM&xtA=9Hy=rQ(jj0000000000"
    ")5Y<oFzj={&MAZrLfnqP0000000000URq@C2=sHn8g%(_+0~A~0000000000_*6iR&iiw~=Yln0YtN3r0000000000nJ!{bfCF^ET4iXE_sEXG0000000000"
    "HDIO$9Sd~8i7LXmeZr2w0000000000vSR6WqY`w$o#Aqm{kx9900000000003$F)s5*l>CxtXfecD0Vc0000000000+EoAZXd!gKX(=Q?>8_5z0000000000"
    "Scjcwrzmv5SzvJeRH%-?0000000000R_sgy&o6Yq8h|utx}lE10000000000;(!?W;5Kx?d^CmJ8Jmv40000000000BSH&--aT}{Ln(K*bCiz20000000000"
    "FfB08$V7C&GzuS5$c>J`0000000000DnF7Kp-XhY7lV-t7>AC)0000000000EOk7RXi{{*q(9?6V}Opp0000000000Ow2iPAX#+4vFj*Es(FsU0000000000"
    "twELQ$zOE905f-k?Q)L50000000000YtmUWW@dE2Si#~<Eo_d!0000000000qI;5n_H1;(bV7gtYGsbV0000000000X*x;=d~<ZbCSqjzqh5}{0000000000"
    ";0uiW_jz={Gt8rw+gFaj0000000000Bkf|9Yk+jXTr5*D5mAo70000000000P}|uC*oJh#YA%N+MM{pq0000000000h12_mK8<w19DY>NcSDZA0000000000"
    ")zZ^Ppp$gKKgZ9IsXUIr0000000000Wh+N1{+V>ZJXyu1+BJ^A0000000000+CxrDSD<vjI>NTv3NMbp0000000000Jt1*Js-|?nxcIx$IVg_60000000000"
    "I`Jx}^{jNjjA`jGXCaQj0000000000a^dz1IJ0!XoPkrAk{XV{0000000000lOIqAaJh8A!L1IgyAzJU0000000000MA@+}oWFFy)><zQ;tY<!0000000000"
    "FFcnQyTx?CyY<Or1qP1500000000001a^B@%gc1YN$c`YCI5}U0000000000UbNYR%h7beVyM1zLHLcp0000000000@FXoVy4Q5T=$S!tS@Dg)0000000000"
    "S-kgjkll2^On&4KZ0e1`0000000000;;;FD8{>4q;j;P$apR4^0000000000fe;Y$7U*=qEGG9!UEGbp0000000000W?@#&eC%|<Et+2#Fx8E~0000000000"
    ">fMA^e(-d_h^&3l@y(6E0000000000s3aA(PV{uZqfPuItHzDM0000000000Nz{NBA^3E_+HHoyWWSBT0000000000V~athDExH5nEl3CC%BEk0000000000"
    "p6Jktm;iOaL+xmq`>>6`0000000000%Kh~~nFV#gV8lRS=&6ms0000000000wWm`u0Sk4&9Jwrr=Aw<j0000000000yZp)ogAjGVD47?}=$nne0000000000"
    "In-@2`W1D+vw80J=#-7X0000000000&a~B(WgB(C2&enf=8cWO0000000000sV<&0#UORSjlzCu;)adD0000000000_R(S%876hWqHVXp+kcI~0000000000"
    "<u1QJVJmgOs7dAC(s+%)0000000000pVdyboiKI44ERUN$8n9o0000000000Q&z2i&NX$wFd83;x@wKU0000000000FhO>q^E!3FcTuV#t7DD80000000000"
    "UnWH%4M26kKR+p&nOu#)00000000007nZdI8%1@%?u6U;g;kBf0000000000dmv*`9ZGe;?!y2?a88ZD0000000000x>e~66i;=)p`F-)SVxV(0000000000"
    "2FBw&|5J6qXtScYK0uAY0000000000jsfoG-dJ_Oq^6$kA~}t~0000000000b(o_Nvt4z-z!(fF1u~7m0000000000@*1GOdt!CK9;+&G<tmN90000000000"
    "D;a<JIA?XhA^$nb!y}Er0000000000R3qHO>T7ktDz<q$pd5|B0000000000o5@lnk#KduopCA4dKHbp0000000000C!BsDEOm9j=hBFOQx1*50000000000"
    "CoB8CyLolMVtdzKDhG|g0000000000$_8WtKz?<=a|~5)|No4@0000000000J-<}*xPo=SgC{|^()f(P0000000000u;gf#CWm#v?QPpUr16Zv0000000000"
    "Ob0yih>LZ=AAw^Abn1-20000000000KDTKJ;E#2{c>)<4LgS3U0000000000xP*=5E0lG>UXwe14&02u0000000000<Nrd4Y?yVxG}Q_@+0=}{0000000000"
    "^0+1Xp`3NVSZT&kqRouJ00000000003qa97%%F9^Hu0^?X~vAe0000000000U?t&R>!fwSFb)B*FTaey00000000008~N~D|EP7qsA2yD^tX(_0000000000"
    "ZcCg$2(5L%294;=wy=!A0000000000hOB-J2C;R(q%r&xc&UuQ0000000000imWrq__TGv@bG!uH=>Nd0000000000t=m?6;J9_bN^GGm_L_{q0000000000"
    "8Onnwy}Wh67e%rSvy+U#00000000000SH#sj=*)mv>fb$Zj6k;0000000000j3sG`Rm63`l9Y~<D20r`0000000000_9A~u5y*AGf8^Ii;eCw20000000000"
    "&AT)t!pn8Q<zl>_n0Ac70000000000v`w@yXwP-P_qoHSPH&9B0000000000K<X5V1JiZDUg2zD189uE00000000004x?>Kl-6~?;m|hHwqT6F0000000000"
    "xrx<_8rpThFo2)|X<3ZG0000000000)~{n|mfdy0`{pS08B&bE00000000001Sc>02jO+V|NKs`$x4jC0000000000-T+TqZ{&5r=edzMctec90000000000"
    "_fFcS%;$B$bQN%wB|MD40000000000?jq>(AM16%R0nCx(KL*}0000000000TaJ%kY3_BvK!^L^d@YQ?0000000000+M?$IsquBd?Tn__CMJx)0000000000"
    "266}Z-t=|A3lQnL&mN4x0000000000at#Jq3;1=wSO`LlcNdJn0000000000zlQohF8g)BiES`a9uSPc0000000000f$#gqNB(udN;fkS#0ZSQ0000000000"
    "R#_nkSOIpxSY64{XaS7C0000000000+0}s^Uj%l*V%Kq&4El?}0000000000qINVPT?cl+E1j!lukwq)0000000000M@>*IQVMp!S*G<wQR|Dq0000000000"
    "Xlv|SJ`Hxjt?JG>^5cuZ0000000000p5h<JArN-J&|$VolH7~H0000000000f5ZWa{1SG+gaa#jG1ZH}0000000000rXs1k&J}jRdB>~J&&`X#0000000000"
    "unsMGm>71zWu>=LZN`hh0000000000HJYT`SsQl1{*=uH3criM0000000000&(<dW5*~KI@{MforMHW~00000000007uL%6!ytCR0*Dh8L9mO!0000000000"
    "sHvyrY$JBS-g-fN+o+4c00000000008DrGo4JLNLJzjMxb)k#E00000000002oz`orzm#7(kU(_4w{R=000000000056*dKI4gF*PSxshrjm=m0000000000"
    "#G)W4!Yy{da#$)BK8uUM0000000000!=rOtL@;*1?W$`R)q{(`0000000000rQ(SpzcO~fb;K=uYkZ5q0000000000261|jGBtL<z?|<!0d<SO0000000000"
    "fM8t1o;P;Dd@C@3mTil`0000000000Q_T6(13GrVo4{x+D`tzp0000000000TbEO+VLW!gV;E{pzg~;L0000000000>j;Czw?1~jVRld5Qdf(>0000000000"
    "6{%R41wnSeCpS9P<xh*i0000000000;{T+PN<((QK(|XScu0%D0000000000UX?f|hDCP3M^mNY2|<g%0000000000n(^btw?}rs$7g8hnmLQW0000000000"
    "qMUc--br@AP*Ju!D>93~0000000000gC4~5`b&1eZEUdAx+;sn0000000000N&lH+3r=>xx)pe=Nh6EE00000000000>{aw4^Vc%!l^F2*Bgt#0000000000"
    "yZs|C2U2#x5{32rWE6|Q0000000000ZtR|(@l$rdBAT;m@eGT=0000000000d}~o(&sBE79Z1y-eFcla0000000000;FM1fp;vamzaqu$2mXq{0000000000"
    "eu=G2XIXZ@8@}xqlJ<(g0000000000F=~IcB3pLANv^q-8}Ew10000000000;8)q_(_D7IW`~n;r00sj0000000000W^>PhdtP?Heck1?EZ~a30000000000"
    "quq5O8en$7wnQ6kwAhNk0000000000cDLYUuwiz<AyJ;8JJE{30000000000zab>*K4W&k*Ou8`!pVxj0000000000VLen&#$<ND@u@egNWzN10000000000"
    "IV*)SM`m`wgB~N7&bo@g0000000000C=AOn!e@5CtcV6FRI`e}00000000003IRInH)(djf_zMD*{h1c0000000000#46_esA_h=9FAXWUZje^0000000000"
    "G^*nS7HoFFlToTL<D81X0000000000IiN<Ne{FWa`C6C8XO)V;0000000000vn?ZV=5BVtaXyhZ?2U@R0000000000fpd)yOK^6;4v+$laE6M&0000000000"
    "hLV{bu5ot2>f=?^^nQxK0000000000vt+a@5Oa3GFybl{d3TDy0000000000YtwvHaddXT0{p*E{cnoE0000000000mcxH`(RFsfLnTm!foO`r0000000000"
    "pe?3pFn4yq7!3@w1!0Q70000000000_zBqwj(B#!mtrK>iCK!k00000000004^aU!>Unm+-PFnN4N{800000000000Q2PhHM0<9?4ihc=kxGid0000000000"
    "EdkbroqTq{M9hBo6+?=^00000000006I)bg^nG@~oUiZVnLCQW0000000000b^&ZWNq=_0H*B!G9W;u-0000000000zU&3yn}Bw}C9&~;p)88P0000000000"
    "Tonr~?16T`kuBscB_)c$0000000000zH?;$IfHh<laRQ>s2z&H0000000000SOEJjg@ks%ONpO9EEbBu0000000000l(V;)&4qTr)W*r5uMUd90000000000"
    ";e~@46Nh%dLilImG6#yl0000000000vTsL;Rfu-L#muS&w*QI10000000000Z)xnql!<o0Z1chgIrxdd0000000000aA&xu(291z<F^XuyYPv?0000000000"
    "uzb)m35<5Yg}P>=KIw_T00000000008<p;KK8<$3ha#0d!QqL(0000000000NWlAjaE^AsN?umBL)wYJ0000000000(GB5UpO1FHFWhPl#?pzv0000000000"
    "QjLcs%aC@!nhKLfNXv=90000000000V@}xN^O1JI>Q|3n%EO7k0000000000onXAB8IyLvkTM`zOuUJ}0000000000m(v`8Jd}38<p>5h&a{cZ0000000000"
    "?zZB9T$OgfR7nB%Ppyf-0000000000IvV%4d6ss-Jz~<D(WQyN00000000006b(Z)l$UnE`@uOQQ=W;y00000000006`!vNteAGd1K0PE)Ru|B0000000000"
    "*WV>V!I^fzu@BAURga0l0000000000^xlr~)0%d`Y1(-o*N2I~00000000000<6;}<ePTDjitUpSb&MZ0000000000o>txQ@tk(Re&1V6*?5V-0000000000"
    "V(ye;{hfBeto9K=T5*ZM0000000000<AURe2A_7oZ&{%o+i8iw0000000000yd$!74WM?wGIq-4TVjd80000000000hv4Zo5}|g$R=Z7^+**mi0000000000"
    ";mC8`6ry&(JD2M?T~mp`0000000000T<kb_6{B{*On79)-AjqU0000000000lUgDf6r^^*>CAXPUPOt&0000000000B_kQn5T$m&cLV9C-aLuG0000000000"
    "r&R{33#N9!R*H51UNwop0000000000vjNk&1gCbu^v~W#-z|y20000000000;WlUv{HJ!ntU~2}UnYsb0000000000%mlBu@u+sd=1@tf-yVs;0000000000"
    "4118p<*9bS0%^;`Ul)nM0000000000JgN0Z*s6BGVfic4-w%nv0000000000^q$8~$*Xq2b8suvUkHi70000000000&1>)JxU6=-lRS#b-vEig0000000000"
    "U>{Z~r>%CtCd>%7U-^i@0000000000O3CCLl&*Hbl8tAV-|>jR0000000000AbEt;f3J4HHScC`U+Rd!0000000000em)*xYOr>|dtb*s-{OeC0000000000"
    "{3x5xQ?Yiy#!N{DU)zYl0000000000GDzeyJhFDccO|XH-qVP{0000000000y{qDxBeQnE^;ffiUd)KV0000000000B_Dni3bb~>t3Vtz-o%K&0000000000"
    "rv-+u@3eNn<zal}UA>6F0000000000Il*Xe)U|fN8j10a-L;6o0000000000{Yx!ZxVCn{{GWw1U9O1000000000007NFW`o40nrK9EAy+@^@Y0000000000"
    "sJC~jeYkeOk3%wXT%U-*0000000000)Qk%pU%7U`uA^QE+?R;J0000000000zZspuKe~3nNt)4`Tabvr0000000000j#x9EAG>zI6%myv+lYw30000000000"
    "T|?Eh{=0U-yz#QJT7ihb0000000000Rn&<H-Mn_d0VXvz+Ifh;0000000000ngv*#y1jP5lb0p7S#pTL0000000000Nzq1gm%etuATW+G*=mTu0000000000"
    "j4@r1bH8@LV%d18SYwF50000000000gdn2!PQZ4+1%2-t*jtFe0000000000QJ5U5DZzHY#RU9_S5%0=00000000007?abg1HyK|Q(CC<)=Y@N0000000000"
    "{R+<h+`@LiVrTwWRz--w0000000000BKi`jwZnG6rxggf)jf#70000000000tXiq5jl_1q&Q8x2RW^vg0000000000yC#?hWyN;Dm;7CB)Gmm?0000000000"
    "bXexWJH~dvu?W7xQzwYP0000000000`01M75yy7G&RqEt(;tYy0000000000Z`HhC=f`%yqT(4}QW%K90000000000_Y_k|yvTOI+=D5n(GZBh0000000000"
    "x1wa5kjZwyJ1R2kPzi{@0000000000&UpA+WXg8HZNEJ$&;f|R0000000000VUZP$H_LXwBP~{EPx^<z0000000000l;*-Q3e0xE6<vj&&hm%A0000000000"
    "igE`{+{|{s?>N=dP3woi0000000000WiRUCt<848Z>moO&Eto_0000000000Nf&bSe$IBlMw|XHOx%aS0000000000SR|%|PtSJ1CH3oB%hZR!0000000000"
    "wNQ<iAJBHdxuGF{O3jDB0000000000hL$ce@6dL@#)_$+$;F4j0000000000>hF+LzR`BT_L+^oNWO=_00000000002UoiSjna0&|G3xK$F_&S0000000000"
    "{>8LEThn&Hk<pRwMz4p!0000000000^frVJDb#krUCqS;#;1qC00000000003#2e|_SAO3BbI#<MWBbk0000000000Xa~V<!_{`ckrl%p#F&S`0000000000"
    "DqB1Rkk)p<RyCj}Ly?ET0000000000c$uj<T-SEME8iY1!ik5#0000000000bG#1@DA;zuyZlHmL4t?C0000000000K)!W{^VoL4$ag3%z<P(k0000000000"
    "0GZ;azS(xb{5i)dKXZq`0000000000t~+&ciQ0C+wID1azH5iT0000000000`s9q^Q`>gH3lUBiJ!FT#0000000000p$`1>9o%-n!?_*_yj+LC0000000000"
    "-1(HG=iGL{@W_w%I#q|j0000000000>E|i$u-$gRoa~a~xlM<_00000000001saXddER!w($X2qI7WxS0000000000Y0;2DLf>}4m?;*nw?2o!0000000000"
    "P)`Rz3gC9Y1D|=0HaCaB0000000000{M5q9(cpH#8{su;wJ(Rj0000000000qNA19nBjK7>76`8G$@C_0000000000gZxouU*dMaeS?1)vml4S0000000000"
    "+91ktCF6F$+duH?F&T%z0000000000=26h~>f?669f5$jun~vA0000000000+R!%cujF>XLiHhnFA9gi0000000000{NB%2bmexyTuR1At^$X^0000000000"
    "g1v1kIOcZ1buM2CE&GPR0000000000tF7W^`{s7Ql~3NrtMi7y0000000000vhV;1zUOwp&095qD(r^90000000000(#M(yfarF>DFIeBspN*h0000000000"
    "NOXc?LFsnDvF$PDC*6j?0000000000Q8$Vd0_t|abkDkyrqzbP0000000000DpG_Q!s>RwcvRv#CC-Mx00000000003I;=Mf$Mg_&nlYUqsE570000000000"
    "GnAMMK<swFh5*!kBEN>f0000000000-#u6G{p@zYq=SJNp|^&>0000000000NJLQ=yX|(sH)52oAFzhN0000000000uIN~NckXt;L`wcYo~VYv0000000000"
    "M{GbFGw*i5<LR)>9HEB50000000000RuJ0n?eBKL81J1_o0*2d00000000005*GONr|@>b?+3Tn8Ip#;0000000000xxQ>EVexjrbK>n)n2LtL0000000000"
    "iAxHG8S-|(wk$Ny7K4Vs0000000000yZxdE(eie{zm$7Jm3xN30000000000j1~#1hx2y8n#i@X6Lf~a0000000000H`fknJ@j_KQ+CuGl5B>+0000000000"
    "`hbQ&^7MAV?@P&n5M_qI00000000005miDgruBBfiuZ-#j$MYp0000000000x0DnpS@w3oAt(Gh4OWK00000000000D1px}3-@-w$A}$~i%y2X0000000000"
    "p<6{iy!Up%hRWOE2}g#&0000000000Uu8&AZ1{G-WSS@|h(CtF0000000000o8PKr8~Jv?c&7Dl1~`Vm0000000000k_$g_$@zA`$DA~;gfND{0000000000"
    "fhyN_cKUX}Tr1b;0x5>T0000000000rF}(iBKvm0MgY$ofgy&#0000000000IYwMm&HHx1iOCI1{~3nB0000000000dd2M|cl>t1KQ)DOeG-Pi0000000000"
    "p*+mnAN_W~Xz?bN`wE7@0000000000?d-{P$NhG|7p&y6c>{*P0000000000mlwI~ZT@z^P!q7p_xpvw0000000000;r(ko6aRL=Ak})>b@YY60000000000"
    "0O7P^xBqs)nx8i1^X!Gd0000000000HUef^S^#&z#URz`api@;0000000000x}s4Z`~Y{rt_Dcx@7;yK0000000000*k^66oB?;htk{9zZPtar0000000000"
    "KIqp1JOX#XieZA%>&}J10000000000{;Xvm+5&gLKJ8P!X~%`Y0000000000>^}DbcLR68co6}l=f8!(0000000000(I#!J5(Ia^-(!@6Ww?dF0000000000"
    "d+nzltps<#8+sUB<FJLm0000000000y&H&DMg@1k=INI(VX1|{0000000000UOMhn-UWBS;Yc0+-l2uS0000000000GSa#kbq069yDYQCT$+Wz0000000000"
    "18mNu3I}(<A1h>s+LDF90000000000p^ypoo(Ffpve`OASc`?g0000000000*T_r_G6;9TEY3ms)q{n=0000000000cAa?Y!w7f4GM<B}Q+$QM0000000000"
    "OP6hpR0(&$bKeS7(R78t0000000000BR>rG;|X`bmnY)*PHlz20000000000%N;c&ate3AL<|&|%w>hZ00000000005xmG3{|a}&Gb5`pN?wJ)0000000000"
    "$D*guiVJtZ3Iejl$5w^F0000000000wpy}I6%2R4aXvFtMNfsm0000000000ux!bZo(y-u8g4n=!bgR`0000000000e__p;CJlGMrY6r|KtP4S0000000000"
    "^TD2Gtqpg;&Het~y*P!y0000000000*%lu7G7fjZIEm9&J28d80000000000|L7}Dw+?r}k6u^HxG9Ce0000000000HNmK8IuCcidJ~m7HX?<<0000000000"
    "L@0GfybpK4u05Bcvl@lK00000000000jF{5JP>!l)pBSCF%yNr0000000000Gsb>VybyQ5m`D?4t_y|00000000000s1AL6IuUokpH!m0D+GnW0000000000"
    "Gzm>>w-I;1mn<L?sr-b%0000000000p+3(YGZJ^eG-j7#CiR5C0000000000yf@^NtrB;@8<!HTqwR#i0000000000SAA^FCKGqS`&69rAm)U?0000000000"
    "IG9Mvo)dS#dNn^Yp5BDO0000000000HxXHi6%=>CMa~6p8`p%u00000000007;%0LixhXj47K{Dn9qd30000000000ut1qp{}gw?d4DL_7086Z0000000000"
    "%Ls~Gaus*LIm4L+lfZ<)0000000000G*U4r<P~?o^c7hy5V?fF0000000000y`^uqR2Fx@QVnrZjj@El0000000000FpMP!#1?nJ3P0>|3aW&_0000000000"
    "Uyd^xG8cEi!$8%HhoXeQ00000000007&UtDo)>q(DX2E11e=7w0000000000Aqm=s3K)05;tCtJfs=&50000000000QIj*^bQpKQoBoW#{)>db0000000000"
    "a$INo-57Vk2^p8mdxV6*0000000000Q0-ONMHzR%(C<Ug_<V%G0000000000z0TiutQmK}ofw|bb#;Wm0000000000g^A_X5gK>E7b&pJ@@<5`0000000000"
    "ccw@9bsBfT^|pk>Z)SwR0000000000T(0KY*cx}h*;gjG>|TVx00000000002`Zz0IvaPubfKT8X;*~60000000000NTOZinj3e(a4Gwe=1+vc0000000000"
    "<e&Bk`Wtt^aD_v9V@QO+0000000000va3$-R~&c1FiV47;6Q}H0000000000dQL``v>bQ9QxL^GT{(on00000000004H3Tm4;^>FhH4WS+A)N{0000000000"
    "Hhh*PY8`jLbMJWXRw{(R0000000000#~82;!yR|P!88EF(;|ex0000000000if`++8XkARDA2Q#P#c860000000000PYEd=aUOTTO;?^?%@c&c0000000000"
    ";%-Ss#vXUT+F!~hN(_X+00000000005-BY%8XtGSet$yi#sq}G0000000000ufVj`Y#(>P+)ELxL;Zum0000000000g}%T%z8`nMuv$TB!1aT`0000000000"
    "U#l@=4j^~HnL*tmJ??|R00000000005D@4{T_AVBNS5!|x#okw0000000000Vs*srsvvj3VeW#9Hs6E50000000000C1j^j_aJw`o8O^2v)6;b0000000000"
    "C}nYULLqm+p8r<nFVKU)0000000000GXVouj3IZx5#+3ntjL4F00000000008o@mF)FF4kr6R5~DZzul0000000000s4f3g93pqX{ZM7qrMZK^0000000000"
    "t<NfQVj_3I(FEOdBC><P0000000000_Aj7Kry_U2$I~AVpQ?kv0000000000PS-!@>LPc*ic>hJ8>5530000000000hED1}EhBfp#ThX?n45#Z0000000000"
    "Y#xAAZX<WVBD@sN6qJL&0000000000%+KK`t|NEARM72Qk&J`D0000000000ckA!C>?3!;0qn-;4TXci0000000000Hu{PODkOKn*rcj)ihYB?0000000000"
    "+=P=8W+ZpOf{>&426ltM0000000000GZgCFpd@#|r9`}XgKmSs00000000002{CVD+9Y?t`d0t?|7L^00000000000Fk}O#5+!%QBwcrSd|!jW0000000000"
    "ayEyjNhNo{((Eqt_*a9#0000000000oiN*Ee<gRot(!1wbx?!A0000000000g4^HTvn6-HUjK&R@koQf0000000000>c4dr=OuT*n5{fhZb5^<0000000000"
    "re=Kz876nY1Uy{B>N$hJ0000000000g;>|ONhWu|PC)Q5XEKAp0000000000Qyyd^c_w$j8&CtB<0^x|0000000000-9Q9OrzUs6AOWBRU?YRT0000000000"
    "p-hJ^)FyYp_>IwO+#7?y0000000000DpDQI|0Z|9|2?VBSQLZ60000000000dJUf?Dkpcqp{P4B)eM8c00000000003s&SdQzv)8?roiqQ3Zp*0000000000"
    "Rs+TVdM9_l3NGX9&HaMF0000000000((D+2p(l611BoF>O7?=l0000000000|LM20#wT~c_8x7a#_oc^0000000000RML5z>L+)=^^e;4Lg#|O0000000000"
    "M}0Op4JdcO6nuV8zu$tu0000000000SHi`GE+}`vbAxT7JJ^E20000000000`FQ_>PAGT4CPuvUx6p#X0000000000sd&3FZ76rZL1ceLG|7U%0000000000"
    ";h52liYRx$-hIoIu)%`B00000000008?hOerYLv74l}jlEV_cg0000000000&6sOU!6<jY?;}w!sj`B=0000000000btj*!+9-FxnDDcDC98tK0000000000"
    "iZqP2@+fz}A35^Gp`(Jp0000000000h?fd!2`P8Ln4KUF9-M-}0000000000=O^;Q9w~Rg7`{_hnv{aT0000000000A-x*IGAVb!zdE3u7L9_y0000000000"
    "t}a|?MJac{owFL`l7)i600000000003sjD=Rw;MD&dAp(4}OBc0000000000tptgkW+`{TUK;Ffi*|y*00000000005*Q>abt!kia!72m2XBJF0000000000"
    "v}HG0f+=^v9P@1VgJ*)k00000000002*sKvjwyG*ak>IK0APZ^0000000000iVY@`m??L_i9SMpd{~0O0000000000widCPp(%I3cOO8(_)vnt0000000000"
    "1Gp1EsVR5BTm29ObxDH20000000000>c}O2uPJxHMtI3a@j-&X0000000000=ODjUv?+JMQ80dqZ90O$0000000000bC<Wrw<&kPlf^^K=`w=A0000000000"
    "2n6TPxhZ$R9oQ}pW-Efg00000000009zG>*x+!<S5C~LK;v<5<0000000000E*(mtxhZ$Ref7tcUL1nJ0000000000wRkOPw<&kPe?Xtv+7yDo0000000000"
    "DJZAGvnhALDT(eKR}F%|00000000002Lq3+t|@oGjc>|g(*=US0000000000#^Agss3~{A(33}~PyT_x0000000000;g|?Apec92|CM^~%l3i50000000000"
    "(E>`!mMM3@HWqj_NbiBb00000000005%yT`iz#=&h_qCF#OHy)00000000008A=VHe<^ps2oWX5K;VJE0000000000W7a;~aVdAe*~|S8z1V@k0000000000"
    "XmZ7$VkvjP35D}oInjZ@0000000000qxE>~Q7L!8v)me|waJ0N0000000000jyEyDKPh*>=yHhkF~Whs0000000000q7TLyEGc)u$mdW)uDXH10000000000"
    "S5*)A7AbeYX4{C9DzkyW0000000000Dd6N;04aCC+U-Q;rmKO#0000000000k%t~W=qPu<Ilq}WBcy@A00000000002qToE&M0@ll<%93pPYff0000000000"
    "2Z|kuvnY4K4j*9T8<l~;000000000036zrUmMC|?y6yTrn2mwJ0000000000i5Sbzcqn(kuls<P6o!Go0000000000{O+DEStxhF0O<zwkbZ%{0000000000"
    ";_l@LI4F0(!&RYD4R?XS0000000000ukUwV6)1PW8RBHKiEn|x0000000000-^eU0@h5k{5-w&I255o60000000000>rvcr%O`ih#@V@hf?$Eb0000000000"
    "Phwszq$hX4ONDgf{#b#)0000000000fzgUzdnb3mwa}_fdr^VF0000000000{dPG9Q73o6C0ud3_ep`k0000000000Ix0dSB`0^lsmI(YbV7l^0000000000"
    "x59j^_a=A1U$X(6@H&CO0000000000=@-GN$R>BdQ@`{OZ8L$u0000000000O9AH=m?n3?sEf*o=_`T20000000000S60UIWhQsPak!TQW+Z{Y0000000000"
    "hVC{zGA4Jx#N}{-;v9j%0000000000P@xd={3Un5x3g*jUloDC0000000000GDa&C#wB;aUBa1$+YN!h0000000000rL<n2jU{)$+OZN2SO$T>0000000000"
    "A;f}`Qzdu6J0j|s)c%0L0000000000AWU!W7bSPVnzJ+~QTKqr000000000093=d(+9Y?t3<d|e&F_G~0000000000kRt)vn<RI@rnz-aO6Y*V0000000000"
    "^&wVeTO@bDh;EhV#^8X!0000000000!1mHq86<bW!o+%nL)n190000000000a(Pdt)gyPnZCpAfz|nxf0000000000fPte}kRx}%p4I5dJj#H;0000000000"
    "X6B}9Nh5c_YS5Z;xx#?J0000000000oPROf03&z6@a$wCHoJhp0000000000og^Enwjy`HL(5yxva^7|0000000000->f^+Y9e>Qc?*<<FRXyT0000000000"
    ";_qr$9U^zYs!bw4tfYXz00000000008pVJ&&LMZe=rJ4pD4l@700000000000t8iWeIa+iQ&^L~rImod00000000007KFe4Dj|2k{a~MlB94H-0000000000"
    "&K#KJ)*yGl{_9gxpN4?I0000000000qNTPUfgpFlXqm+#9Djho00000000005ccw}DIj;iTnP#In0J7{0000000000jpZbf&>wfe>)SHY6>xyS0000000000"
    "mP;hXbsu-YFR@p#l4yXy0000000000r2oiC86S7RJ1LBn4`G170000000000EZOQEyB>GIBIM+Lj9Gxd0000000000udiW1TON172!eKP2~vQ-0000000000"
    "p;`s7`5kw_{GwH1hDw0I0000000000f3dA=mK}G%7Uz~)14Dqo0000000000z>6|+F&%fnYwk{2fIEP|0000000000;J|Cb${csV5m}O5{WE~T0000000000"
    "Stv1EVjOqCC&zVXdMtpz0000000000qnTPe_#1b?z}O9U_auP80000000000zI~cvj2m~reS>R^bRB@e0000000000MA1{#9vgSSLYj-G@fCo;0000000000"
    "wxENVuNrs2Jr88aZVrIJ0000000000CQ~Q1J{otxGSlkr>jr?p0000000000mrpz{%o%sUo2@H0(*A$I00000000006#c&nSQ&S~iWRn*%<q4|0000000000"
    "q{6RI;uv?pn7?2M$KZd!0000000000RR;+MYZ!OHER7d}!O?%f0000000000Co;(}@)vi&xL?KwyuyFL0000000000EUN9+cNcfSrx;n5wzGf00000000000"
    "X4Z55`xbY=S(&>qv7~>%0000000000;~iP8einDYGQ-)`tCfGi0000000000sOOsL{}p$@uy>z@riOpO0000000000!J0O?e-(GYGMM;3p?8140000000000"
    "HJ&LV{uFn>AqLR|oM?Z*00000000004q0(3dlYxT<DwMQmRWzm0000000000R5A<L_7iu&+)uBqkxGBS00000000005s~yTaT9mIiSCS%j5~k80000000000"
    "Ox!9v>JoRrN?@6Phb(`<00000000005PPfqVG?)1goeO(f*pUr0000000000XVi0h*b#TYs6y^}eGY%X0000000000RZoGvO%Zp%3w92Mc>jLD0000000000"
    "=Tgzg!4P-AGy)BrbMSt^0000000000Bt~JQG!S>de#q6kZsC5w00000000008IUehrVn?(RH{_vY0`ec0000000000%9l3Y77us89f@@qWy5~J0000000000"
    "K*V-Rh7NbYI7TE@V6=X~0000000000hkWdv^bL2wD8o#fTcv)$0000000000rYDu&VGVb{N#WY+R+fIi0000000000rYvp}&J1_J0)o0jQip!P0000000000"
    "kMeXsIShBezlPVPO?ZC50000000000Zs4YCqYHPy-_RWqNojt-0000000000NL>M)3=4O_;X!$YL|T5p0000000000C(wb}bP9LCB)ioDKudnW0000000000"
    "6QCt0+zEHU4;~+hJ3M~C00000000007c}aNLJ4=k157~}H!Xg^0000000000JoC&As0eq!ekLfeG9G@w0000000000h(yb>3<!6?-j2UeE)Rad0000000000"
    "1&`u@a0hq5jArxsDFA-J0000000000y*Utm(*}3I^Nz2pCGma00000000000xT{gHH3oOUbLrn@A>w_&000000000008FGEmj!pgiC1ta9n*cl0000000000"
    "o`D6-_XKyql)6mw7{q<R0000000000n6q-hR|I#!`}1JN6t#W80000000000`$*{nw*z;;EajM?5T<>=0000000000&8TLX6$5v`gm2)443~Yt0000000000"
    "7&_>Oa{_n3fOI2p2#9^a0000000000<YQ~q&;fVAf7MN61bKbH0000000000JdN##Edh7H?{iCD0BU`}0000000000dsb(qi2!%N#~uh_{91j$0000000000"
    "ToQ#c<^OiTWTR?p_)C4j00000000004b;;yK>v2Y3ZPMa^gMmQ0000000000>VB@Jn*MgcR`Hsa@GX770000000000LiTTv^!;|g>T@Kw>>ho<0000000000"
    "bRBc_P5pMjaRkiX=ns9s0000000000%-0gsru=rme}8Wc<p6!a0000000000qzI}p|NC~p(ZTFO;qiRH0000000000Qj?PpSNnFr1E?W?-Qs+}0000000000"
    "DqAXWuKISsw+`&V+0%T$0000000000fkVkv1^RZthteqv*2H|k0000000000t}n~NTlsdtFVMJJ(zSfR00000000001_O(^vG{htOXPO2&Zd080000000000"
    ";GknX2l#fted|FC%a?q>0000000000j_j{ZT=#asYQpzz$B2Bu0000000000Y01j(u=aMqtVMR!!+Ctb0000000000#2ODJ1@?Bp08*Pmz-oNJ0000000000"
    "_rurkSoL<m0K`PKyjy(00000000000Rj7Ybtn_xkO?Xf(xlDY(0000000000HLGDy|MPahk4v1YwLN^m0000000000>#x}8QS)}dU>XK3vMzkU0000000000"
    "(3w9Wr1Eyad%<h8t{;5B0000000000HB9mP_3?JVb{(rgs}Ov^0000000000bgD>rNAY&R{R*PdrvZGx0000000000-<P19nDBPMu4XQ0r1E>f0000000000"
    "&G%Dh>F;*HD-A9Iq2qhN0000000000l&!tpIq!DBCEI?Voz#240000000000h@3b9itcv6J+1yhn#Fs-0000000000|EC|++3j|~7hsF%mbQDq0000000000"
    "PqWQfDeZQ^OV9$7lc#&Y0000000000&Y_)=c<gq-mD)l<keGYG0000000000&{pbz$Ln^$j$0}7jEQ@|0000000000thEj)7VCDv-7l1=iF$j$0000000000"
    "wxMN=Wa@UnBs5`RhHHDk0000000000M*c~yvgvlf0Yz>dgIs&S0000000000udWu30O@wXDnFUre@%P90000000000N0ufrPUv>PJpaF*d_H@?0000000000"
    "Wth~Mo9A}G-C@9Ec`tjw0000000000UYL8w=;n67uS0k%b|8De0000000000hk16tHRg7}P=UquauIvL0000000000G#?;?g5`F=b@zqBZvuP30000000000"
    "zHHMJ&*XN%yMmXIZ1Z}+0000000000ctz`W8{~Gt%Kz75Y2<pq0000000000xQxQ9XXAFjLQ5q!X4QJY0000000000(<QXEv*LEZ!9iUMW5#;G0000000000"
    "A@mxT|KWDP|G!=0U$=U|0000000000^x9EUOW}6Fm7)Z@T&Q}$0000000000q`kB0mf&{4EXFUDS($pk0000000000hJTx1;@@__X9vA-R*HJS0000000000"
    "?*nAEE#G#);6jQ`Q+s;A0000000000GcIWGc;0rvP*rs+P;7d@0000000000t>FR?#NBqlRU2FiO<j7x0000000000>ep-+58Zaa9BdxyNltpe0000000000"
    "CnsnJTHJQPw9&E3Mn8JM0000000000Y<SP#q}z7Dxjh-ILoj;400000000005D9vq@7i|2V$@fTKp}d-0000000000Y!*LBI@)%?@40YsJra7r0000000000"
    "*4_2lgxPk$pb}?QIs<yZ0000000000r<`?C&)9artpYweH}rYH0000000000IgpjI8Q6BfYOeSnH061~0000000000;q@%~W7l@T4mz3!G1hs&0000000000"
    "_zDv#uGV(I*vTdCEysDl0000000000*9;#&_|<m63WA>2D!6&T0000000000)Ho<SLe+M_*oFtdC#iYB0000000000NX!KwjMR3(oBU&{B$|1^0000000000"
    "i3E@8)zfysh6lHmA&Ysy0000000000^d6g=AJcZg-OB@l9(;Mg0000000000+gJfJY0`GU+-*&78*O>O0000000000o9~#gve9<Hy%CpP7+!h60000000000"
    "hafo={Lps5)7#@r6;FA<0000000000{V2IzMbLJ@SMWAH5<q#t0000000000PDb#Ij?Z?$gmaTF4>5Vb0000000000)U~yt*Uompo{9V*3?g~J0000000000"
    ">a^gSAkKEc-d1%K2@`q10000000000=YU9iY0Y-Pmj3_=1_XJ)0000000000B3HyevdnhC{->h;0`+*n0000000000@pLQc`pb5}Q*>SR|K)hV0000000000"
    "uoqQ#L(6u+(1<kc{MLBD0000000000vwRrpi^_Ju#0e|r`Nw#`0000000000RN+iP)X8?hYa2x1_PBV!0000000000>w?LA9LaXT2I7L-^Qm~i0000000000"
    "(4J+SWXN{F(t+yN@S1qQ0000000000T5|ZGtjBi11|NRa?TdK80000000000<2qS~^u~6;_}xI%>U?;>0000000000z%^e|JjQmw-QqRW=WTev0000000000"
    "N}%BYgvEBi_fcTg<X(8d0000000000+PAlo%fxoThEK!S;ZJzL0000000000#ksf#6U27FzyD#|-avT30000000000Y2&+JTElk0`-VT@+c9{+0000000000"
    "7jl@Ip~7~+Z3$oI*dlnq0000000000E2Aa2=)rctRjb48)f0HY0000000000`?ET@FTr-e^^kD((gb+G0000000000-{G#BcEEPPg5GEV(Diq~0000000000"
    "GHO|0y}x$AR=BSV&E|K&0000000000NQw~#1iyB`ph8F(%GY<m0000000000eSfZwO1^f$r=c|_$H;fU0000000000C5H$3kiB-mrulj`#JP9C0000000000"
    "nBMb5*SvPX)@~a{!K!z_0000000000FvCi69lUnHj3pXazMFTz0000000000MRMbSW4m_10780eyNq|h0000000000E-968sk(N+ap*pPxP5oP0000000000"
    "LnSvs@40rs6FqB|wQhI70000000000-J_uLG`V)bKV$-~vR`+=0000000000QsStDdboDLDuM^cuTXcu0000000000`W+|tzqfY46T5xktU-6c0000000000"
    "FqHC11-Ev<JTqwhsWNxK0000000000NP^;gO15^u+YZVarz3a30000000000oh8nXkF|EdMslV(q!f3+0000000000g}aP`)U<ZMuz>(tp#^uq0000000000"
    "Sb_Ca8MJo5USZpNp7wUY0000000000aJ1hBUbA+<jx{=;o9A}G0000000000Ba$MRqOx|tc_JXhnAmo}0000000000$5)jI=dpIcZA_5rmC1I%0000000000"
    "xIzzDE3tOKsMn1Wle%`m0000000000Ops@WZ?Ja2Xy_X}kgImU0000000000+<kSEv#)l*?V>$rjhuGC0000000000y+{y$_pWxpZY%baij8)_0000000000"
    "OD+#dJFa%XKc+~+hkkaz0000000000*=tGheyw)El{Yx@gl~4h0000000000#y*yJ!mM_{wgbo`f?#&Q0000000000W8}=+1*~?!)Gqs0e^GY80000000000"
    "44T9pNvn3iM%~nld_s1>00000000007M)Z*jH-6PJ~LRtc{6sv0000000000+$vN$&#896167Fnb|iMd0000000000w1Aov5~+5;(Pn5gbQN~M0000000000"
    "_S^o;RH$~q;EelkaRzq40000000000`gvGjm#22XgzH4DZufP-00000000008km03*`{{D^`=McYv^^r0000000000sdGpl9Hw@_a^nUqY1wta0000000000"
    "1xk%cUZr-xJ;jA@X3BNI0000000000gIFX`prm%di*U!YW4m?00000000000dEDeO;-hxJvI)=mV61h(0000000000LKCL(Bcpb}=d_<eUY&Ko0000000000"
    "H99|xWukV#Y?eHWTaI<W0000000000tB!{Pr=fPhfzqhfSbufE0000000000_)S(+=%9AMU5)1>R&aH|0000000000ceR3nDWG=1QtSJ2Q(<+$0000000000"
    "e*)x*YM*w%nh@#3P*Qck0000000000YI0F+tDbhibs@4APD6FT0000000000k3ym=?45SNBSKnhOEh)B0000000000L$IyHES+}1*p}qMNF{Z^0000000000"
    "<M>KAZJc(%?n6`>MizCz0000000000!dX#<t($hhm~E_dLkD%h0000000000JP??!?V5JL7C7+GK=^dP0000000000q*?Z`Et+<~pT{CEKIwG80000000000"
    "RdYj)ZJBn!h(Je=JKA)>0000000000shU<pt(bPe3CCdZILmav0000000000^$Z8o?3Z@HVSjQ}HoSDe0000000000k_Rk2ESGk`(T#t*Gp%&M0000000000"
    "*vFiHYnFDvj~0s~F`jh50000000000AB!-Ws+D%Y@;#W1E{}A;0000000000y^g+)>6CWBEGVk^D}Z#s00000000003=ke%D3o@<f%(H{DRFeb0000000000"
    "V@xIjXOnioGS=wVCSr8J00000000008-cEXrIL2QapD_FBvW+20000000000itCNn<B@j2m_lH_Aw+b*00000000003dx%WB9V5$+#0zwA2oEq0000000000"
    "@m8-5V32mee2*!u942(Y0000000000o6?``osV|Fy2P|68W(iH0000000000S$RaG+m3d?%HCb17YKB~0000000000iHLMD8IE?q1_CG}75Q_(0000000000"
    "fISD7R*iPRpHdd266$ln0000000000w6NOKlZ<x2cAG0F5ZiOW0000000000LP;6v(2I7!1-n_S4a{@E0000000000;*iJL4U2ZbLuz_63%zr|0000000000"
    ")@lx^N{V*CXozLJ2(EL$0000000000TDlfmhlzH;g7or62A^}l0000000000tzrrA!-#gkvW@!E1CVpT00000000004Mz`d0El+L3Nut-0fBSC0000000000"
    "x=K;kJco9`zr)w={&91_0000000000^icRNd4_ht;~rgp{9<#!0000000000_B<_kwS{)Tlq0+j`crej0000000000{P7sJ@q~83?y$_C_e68R0000000000"
    "OjG3RErfQ!2c~8*^)++A0000000000Aw;+wYJ+yb5#nLP@+Nb@0000000000xTn2Krh<0BA4HZ_@E3Ey0000000000Qeho@;(>O+QmfnN?Fe(g0000000000"
    "D^vin9)Whi!}bY!>-loP0000000000gkz8GT7Y)Ih6VWx>FRR80000000000m~!(pmw$G^(YAn{=G$_>0000000000s$^A)(|&fqw&{2-<jiuw0000000000"
    "_8i{u4}NyQQhm9;;k|Oe0000000000xqh~5Onr91!%!Md->!1N0000000000HEjp}h<tXy9z%27+@Es50000000000rpx)N#d~(Zl{1`a+K_U<0000000000"
    "OntU>0(*AAIst9<*MV}t0000000000VP$4vKYDh+D~r8<)pByc0000000000EN3!te0g@ji?C$~(_?bL0000000000<Yhysxp;QKWl%kj&{T530000000000"
    "%tXQo_jh){@<|*N&P8&-0000000000Am`VZG<SBuMSR+p%QkYr00000000009Wbp}a&~sWc*ai`$tQBa00000000002m@<OuyuC8q6bu!#u#$I0000000000"
    "7(+UD?sRs*^d7$u#0hf10000000000kkdolEOd6jg6n9B!TNE)0000000000t(UNVYjbwMYfLHszUy(o0000000000t#YYhs&aO~%plltyxeiX0000000000"
    "&_`E@>2Y?zwnt6jxy^CF0000000000Qu^ZcC~<bcU!Y)3x4v<}0000000000b!~O6XmEDG+X?Emw6Af%0000000000c7!*wsBd<_Mq|MmvY>Im0000000000"
    "lOj?F>27wwy^FqkuaR-U00000000002`lceDQ<SaRbsi=tb%dC00000000007=bjeYHfDFLB@$Os&jF`00000000000SHbAtZa6`l$HmEr(|)!0000000000"
    "0Hb5R?Q3?xY_ba0q*Zai0000000000QZ=5`FKc$d<@jnLqDFDR0000000000IHD#_a%y(K6-fYSpEq&90000000000?#v}FwP|+1E`p+|oG5X?0000000000"
    "x0u9t_-J;(LxU;inHh1w0000000000%;+mFJZN^naVu>dmkM#f0000000000aiD}$fM<5V*;z$Tl>2bN0000000000-=g>J#AbHDjO~egk?e550000000000"
    "SktZj31)V{yu}itj@@v;00000000008WqrnPi1z%e!;iGi_UPs0000000000V@rpOm1K6n@xM^rh`(^a0000000000aasHk++%jYC7`wOg|KkI0000000000"
    "gd$iPBV%^JP(i-`f}wE00000000000+KI!cYGQW4g@Qr{fRb>)0000000000uX1O=vSD_>-$EJ)eS>ho0000000000M!;-j`(Sp!fMFd1dUSBW0000000000"
    "*;V|PL|}HncfBL^c4ctD0000000000sun+9j$d}b`!xmOb5?M`0000000000_2pf<*j{$P9jckda7S>!0000000000`lf-eBVKmE`}nw`Z8&hi0000000000"
    "`8?A=Z(Vl4v;jMKYAJBQ0000000000EsJ|`yIgj_TBG+xXBu$80000000000)pVmo2wZl+A-%>6WD9V>0000000000GTn2%Ra<tzIa~mf!T4{$0000000000"
    "fTtDTqgr;r(W>|Ty4r8R0000000000#0IUb@>zDkGPWaFw7hS?0000000000!><%5LRogekK0_Kt)6ed0000000000{LKncl2~@Y;Viw`rhsq20000000000"
    "Hewqm<5zaT=hS2Ypki;p0000000000Fm2MQG*@=Of&OzCnM7~E0000000000w(q;xhE{gKd(;Xak|uA!0000000000i|S*u*;RJH)`3YCiwJMP0000000000"
    "a$Z(BEmd~FVKGqqgX(U;0000000000GfW?9fmC+D^9?`Ie9UgZ0000000000l(b4k)>C%Ca6&ehb*^r}0000000000RhVSTD^qsBnfX;sZjf%k0000000000"
    "K=}Chfl_wBdtSctW^!)800000000008r74~*im-C+?$VzUsP_u0000000000r!T=!Fj02EnB3_RST=6J0000000000tOyNxh){OGjZ5u`P#A8&0000000000"
    "?qCB+;ZJtJg%i5+NcwHS0000000000GC457I!|`MgkNw>LELS?0000000000KawBblumZQR$4HZIlgVc0000000000+iYE;?@e~V*)AH>F`#Y00000000000"
    "$jNmKOHFpb<OfawDuQjm0000000000%&GJnr%ZOhLlEp9B4lmA0000000000tPN(+159?n{c;p98b)ov0000000000F!og`VM}(v+*;=_5-4rJ0000000000"
    "8)Q%>ze;w%yz+!33JPt&0000000000F<9Zo9ZGh<dSwj?0_<$S0000000000I@6R2eMxq}-rb<#`OR#=0000000000`qlRE+(>r7?uK8h@vm&a0000000000"
    "HDQx^JxF%Ibc<<c>5*)}0000000000vUT>Oo=0}TOcvi2;d5-j0000000000H59CG07rJfKvLkg*i~%60000000000g*`yrVn%krSZa+*&^K(r0000000000"
    "Wb!`g#YJ|&ScwMF#~EzE0000000000n=>?)DMfa`A=+z7zx!*z0000000000?E$RfjYM|9k%!&3w%u#M0000000000A6_!f@<VpOZM=IHuD@%*0000000000"
    "_DEA$SVMNey`yGnrJ-xU0000000000I>VZPzCw1uOK85YoP%q?0000000000t<981B|>(<@ie39lVxkb00000000007irC@j6rt5k~B3Ji$`m~0000000000"
    "I+oC>^gwpN@R5`_f+=gj0000000000-c>3_T|jog6?oxIc?)a60000000000#HeO|#y@tz$PqkOaP4Zq0000000000xQ`K1Fh6#{>xslvXU}TD0000000000"
    "dkeR~nm%^GRtuX(Ua@Mx0000000000&>Z8*1wMAb(7@j;Rg-GK0000000000gTx_iaXogxXVtO?Om%9&0000000000QUm{~-8^={-E%b8LRV_Q0000000000"
    "2HFOIN<4PJ632v_IXP;;0000000000WJ$&Dw>x&g;`zf=FdJ&X0000000000F;d3$B|CP&88;FGC;e%_0000000000F!hy`lsb06xT%k(9p7od0000000000"
    "EHl8$13GrVjeJ}`6v1i00000000000<P6c5b2)ava@v^93ZrSj0000000000AcIu<<2ZJ}Fz#eg0flM60000000000rk2+9R5*6P(Bk0A_hxCp0000000000"
    "ov+e^#y57rebp*D?nr6C0000000000%V~C}I5&2{u0vO!<SJ>v0000000000?wQzLt2TDPbJ=(L+6-yH00000000002^0{i9yWHsxk8>x(e7x#0000000000"
    "9jr5jk~Ma~rz_l&$IxiN0000000000FE(281~qoTB|t;ay|QS)0000000000K!%#{do*^yGc4r;w3KMT0000000000Rd5S@@iTV7&YROMs&;6=0000000000"
    "aLB`!XES!d<2oQtpjc?Y0000000000lw#UG-ZFN;m9^YtmO5y_0000000000!xCn5RWf$K)3+LOj2vjd00000000001Fd~C%`tYsneFR!g8pZ~0000000000"
    "Pz5-CMKN~3;R0fCc;IKi0000000000w(qM_zA$#coTH9mZo+540000000000GC=UJH!ya<?t6ntWTa=n0000000000$jTd)u`hPO!Z%VVT83x90000000000"
    "d}BocEH8Gz6@v`^PiSYr0000000000P2Pj^rY?5C<q5>fMM`JD0000000000L=Wz2A})5o9e9e4J1l3w0000000000V#=Cko-KC3=Xr5SF%D<I0000000000"
    "sN<mJ8ZCCfDZ7gMCh%s!000000000087~*-mn?R`;^*h29MWdM0000000000x_}gF6)bkZ{;Qfr60~N(0000000000i>$A0lPh+>qO-K!2bN~Q0000000000"
    "l7|2A5i54UxvEEP{dZ=-0000000000(c&8OkScb-LOo^l@>yoU0000000000NR=3K5Gr=SI~N~w=sRY>00000000000CEp-kSTV+jpT;u-5qAY0000000000"
    "^-LD&5Gi)RU{S7D)Bk0_0000000000F8V{=ktlY+p9!bE$l+zc0000000000uT8H@6DW4TNV$R=zQbj}0000000000b{lUql_z$<R}52ev!!Lg0000000000"
    "i#M8+7bkYWxUMO`sE1|10000000000?sZ3NnkII@lS~)|o@r&k0000000000qN<I#9VT|b(2+1hlS^g50000000000uBn54p(S>}aVms)h%IHn0000000000"
    "4o5(?B_(#iaKa?0eGg^80000000000%2wK9sw8&6xdkN8a`9xq0000000000;jCzaFC=!ra-fs$XVYZB0000000000T^#}xwIg=Giavz~UA1Ju0000000000"
    "IefhqJ0o_$`HKn{QkP`F0000000000eVwt1!6J6Rt28wrM|otx0000000000C{^`qNFsK?$>bIyJX>VI0000000000L3f|L&mnfeI!M7BF+F6!0000000000"
    "#@EJrS0Q%5{|ZJBCLd(L0000000000ze<9o-yn9t7umM_8UbX$0000000000E~Mx?XdrgLX<jAb594FN000000000061WgO@gH`;9!I0V1jS>(0000000000"
    "aas|mdLMScAU{Hw`KDvQ0000000000PhSCi1Rr+5Zt@dp?ucW+0000000000teW??jvjWv1Pv-P<7#8T0000000000jo=Df86I}P&pMX&*Gyx;0000000000"
    "^aN65q#bs^^CxPv%r0ZV0000000000<lE!YFCBKkUx2r1!4PA>0000000000VL8{OyBv1F4)Bi|w(?@Y0000000000Yw<$dM;vy*{2~Rzs?=h@0000000000"
    "3fMh$)Ejod7sV)MpSEJa0000000000I>Fv=VH<Y9gn|qMl$c_`00000000001c5xu?izN$FbBe!h<akc0000000000YU^!=d>VGZ7ynu+eOzL|0000000000"
    "YSf9K3L196HWul$aXw<e00000000006RTSonHhG#b%267W*}m~0000000000B1z}!CmD9YTGLp#S^{Fg00000000006-I^>w-|Q7x8J8RPvl|10000000000"
    "{_T;GMi_R$_j;?PL&jmi0000000000>5OAu))#ibZ+f2;IH+O30000000000@NQIhW*2tAytBW4EQ(>k00000000009pJs#_7--)F8oL2AZ%g40000000000"
    "hb;MihZc6gHYs#P6;5Hm0000000000L7-n|7#4QGKM;VT2{2*60000000000T8B!DsTFp>p@ygX{Sjfn0000000000;w=;EIu&-n@nI%W^7CN80000000000"
    "^xqPH%oKLOig!4p=G9=p0000000000m+7KLUKDn~)SMae+P7f90000000000=aw8j@Dp~xK&J&h&zWGr0000000000?K9b4f)jSYA#6B@!+T)B0000000000"
    "!3;8}6ccv95y9=vw_RYs0000000000b!~tZrxJF+XM151tUqAD00000000008KNDdITCijk8saapdnzu0000000000zV>2G%n^3L^L2BTlmlSE0000000000"
    "cekk!UlDe|+QAa35zAk|0000000000SujiK@ep>vC=Qw=`i@_~0000000000bFHA$gb;SXeS7qs;!<D00000000000*LnlY7Z7&9vb32M%NAe20000000000"
    "nB>gPst<O+qGSP#v)W$30000000000$2L~jJr8!k_%)*eoSt650000000000cfI)J&<=LM<kR+cgkoO60000000000yTlFgV-9w}6zOy9Y$jg70000000000"
    "qL}XZ_6>HxU?7rWRq9>90000000000LWT7Bi4AtZdL+r&J+58A0000000000tJZk!91V8BjG&NGCURZC0000000000?by`PuMBp;ML%H84mMrD0000000000"
    "9oS)|LJW4mfUB=c_xW7F0000000000O(1z;)eCmO7LfMJ-@RPG0000000000j{S23XbX10w3n<-$bnqI0000000000@Vl#r`U-ZyhHXjKuti+J0000000000"
    "i>G<&j0$$ZLcBy`nF(CL0000000000Y793h9}0HBxGom+fz4aM0000000000qc%M~u?cp-#cwr<YLQ#O0000000000NOI*RLkV`k7b+ShQ&n5Q0000000000"
    "Y72?p)ChLK+Xy_zI~iNR00000000009ZrE`W(an`#<mi0B;8uT0000000000aE2$b_Xl>srrs4G4WU}V0000000000b^z7GhzEASTr0iT^<-MW0000000000"
    "LI~rR83%U2l}CV=-Y8nY0000000000=;^8{ss?tze%{t!$Lv|a0000000000cJYOEItF&Y#+o=Wu&`Oc0000000000{%Xy4$^~}7LMI&znRHpe0000000000"
    "mZI7$S_O8%(k}e)fjC*f0000000000P2>iV=>&GbAY4u0YW!Hh0000000000Hhim~cm#I9Tv~|QQ@~ij0000000000W40k!2LyJ&Fw`XAJcL-l0000000000"
    "<=(=xl>>Ibdvu!dB}iDn0000000000&_U?0BLjB8=;dG!4-8nq0000000000H7n~#u>y9$rqk~^_|I3s0000000000Cg&u3J_2^YW$3<f;geUu0000000000"
    "w88E|%K>)4`p!kS%2!vw0000000000r<J*!R{?gwraSoxv>R8z0000000000zJnr);{bNRTurBJo!?f#0000000000eVUmRZvb|{`7a{dhND)%0000000000"
    "$AvXU`u}ynYZ)h1aA#J)0000000000gYgV0h5vQH$v<=1Su0k+0000000000mx(sV5C3(*DA~hiL+@3<0000000000>G{Yin*MdbcN^CGEVEU>0000000000"
    "U+&snB>r{4Aw6f87I#&^0000000000-V%dquKjhu5p*;-06SH{0000000000O%vE!IQ@0NVD|dl>Hbu}0000000000k(=;1!u)l>CHi-S)52810000000000"
    "j#uLjO#F4gO|@t(y@pi40000000000FZ0Ic)%$h8L0*d1rb<-60000000000RvNO(U;A~y@G6>qkPcM90000000000>Ug`v>H2lRaZyzydeT$C0000000000"
    "(7O5FbNY3_+Xs2ZW0q6E0000000000@r#2a{`qyl8ic}QO<GgH0000000000Etqedi1~HEnEdYgHXc*J0000000000aEsJa6Zv((N2j}rAmUQM0000000000"
    "oDs=Wp7?dZIlAQ$3Z_!P0000000000n$}~cDfo53UTZ3k^J!AR0000000000OoqEvwfA+v9`Qy2-7QkU0000000000nat*ELHBjQX@U4=#_>_W0000000000"
    "YE}I5&GvP`RZ{%EueDLY0000000000qRbNYTK09o_?Zs~nR!vb0000000000C*DAO==F8LL5}Z0f;~~d0000000000-!+|*cJ+0@m}OaLYXMQf0000000000"
    "vVT581@(2n?^OzYRK-xh0000000000gl9XIl=OALSe3thJ&91j0000000000JejJMB=mK_?pZczCQVSl0000000000ov;;2wexkrupGZW4-rtn0000000000"
    "ZWBYIM)P&RzUmSH@TpJ00000000000n1v&c*z$G22Gzn|!7)$30000000000w92RbYVvi!s|6;WlDSU60000000000U#7s@{qc3cD2mj=V?j>90000000000"
    "H{4zqk@0oFo^NK+G|5fC0000000000(=T(LB=L2?FT``s1yN1F0000000000(q!&lxbStr|CE%z*4RwI0000000000%Ra$IOz?HUfbU?Yr(jIL0000000000"
    "T!Gv};O}+7*x+!Ac<4*O0000000000An-6mbnkV*DtBCONpMTR0000000000u}-fw2=8^kmA|`L8u&`U0000000000t6)e5obGkNlgB1b>wikX0000000000"
    "s_E>bFYa}~oSZdLya!6a0000000000Nv)0i!tHgy&`W@4jgLvd0000000000BX1l2Rqb`aP~)DJUKdHg0000000000)$)dx=<IdCMQ|zLFP})j0000000000"
    "^!vcGdhB(;8vwXV0Vhbn0000000000Ax^qO4D5BlQlR48(ym9q0000000000_pD<yo$GbL3SL2&q&7#u000000000058SPDE$el_Wk+m+cD_cy0000000000"
    "2A_%Vyy|tphiUttNJd7$0000000000c)915OX_vN<k5!p8qP()0000000000{qDF_+39t__l>59?Nvp<0000000000G6U`zX6bdn-Kz3hz}-Z^0000000000"
    "vblHD@#uBHyfy=LlVwD}00000000007}iE_eCTz+w?1&!X6-}30000000000031x)1?Y9aM9|TRI(0+90000000000lV`RDj^}m2^&xMM4*f#F0000000000"
    "qhrkO6z6ro+=}|>;)FuL0000000000jY-0pn&x%D#f#yTw+uqS0000000000(~<BEALezyPi}~jjFdsZ0000000000_q4B3qUCkKa(q+WVH`og0000000000"
    "d@ERMB;|F$%>Sl#H>5zo0000000000=WRk^rQ~(M{X*wk4J<&w0000000000y7b05CFFI$Qy=jb$7esl0000000000bT~fyqvLhJg<pOLbay_$0000000000"
    "ov{F{A>(zxfZ9HWA%{J{0000000000ve;%)p5k@DfFt{l&z3yE0000000000J0{8W8RB)o%g=iuex^IX0000000000x)Okyl;L&2pmK>fEw(zq0000000000"
    "uNLG+4&imc>7VS*+{HP-0000000000pQ=p*hv0R<lk08djMX^600000000003r7{z{@-=L-00PGJ>@sR0000000000c4U*nb>DTs2`%k>?({am0000000000"
    "W_uvg>)v(1SxbERp9D6*0000000000S_BmeVcvDXwbx7MQ4}@60000000000)_2Kp)!lW#3T7lj1SB-T0000000000T~3w^N!@k8n2F8_w=*-q0000000000"
    "asbrhyWDlaqSq2QYC|%>0000000000mImAWE!=g$W#hTZ9aAyD0000000000PT2!qpWAi7&_Im7(PA*b00000000008u6wL5ZiUYng^vI4COAs0000000000"
    "f5sdVfZBDyF@LIgbp$QI00000000000Mfm2@7Z<0_7b{*-y<u)0000000000AdKQNUfFfPU+H8rN<%5Y0000000000U2WrA%-D6nN-8{swPGj00000000000"
    "KkgXwIoNf;EBw%NA%Z2q0000000000N{8G2rq^}Am5H?XjiDpJ0000000000`mlqA6W4XX{`hwa`@kW<0000000000+T&@Bf7W%teB@Y5(ZC<T0000000000"
    "W0>X~>(zC@UOL8wt^XUq0000000000AV*R<R@HUDo<Lf+iar>?0000000000k!S_%z|?iXL`*l+XnhpG0000000000Idxb#E7WzsQmTK<N4yZg0000000000"
    "nB}7-mD6>==TxgKPxcAG0000000000H8McD{?c{8t2nnu5Oe~-0000000000k<*3zXVP`R&@<1VrQ`L#0000000000Eke!m(9w0kAQlN|wS3#Z0000000000"
    "k@jw*Ini~%NK4-EkV@b`0000000000IcdWbqR@4~_|E$YyeaoT0000000000s}dtk3ea`H+&c}&78e6R0000000000YHQ|Fa?f?Z{`DZ7PMHco0000000000"
    "`RipY+Rk;r;Whfs!VM8X0000000000*$n#MLC$r+m4PAi-Odz10000000000jYj)(sm*o3ZVqI}_>UMs0000000000oDstA5Y2VK#f+a=6Hy#M0000000000"
    "gXX4Ic+7Rc%-7i276~9g0000000000#ytYE-^+Et>=3;2qv;_)0000000000>W?f3N6U4<MoVb@Fv=r90000000000aaDd?uF7@50$)r3zN#fa0000000000"
    "+Xhg$70PwMpe5G<ON%E!0000000000uN?XyeaUsensj#v*lQ_30000000000ZV%O*<j8fvb|st|WlSqT0000000000FfA-HOvrV>c}%B1@+>Vt0000000000"
    "^C*$zv&VJ7TAWsRe+@4{000000000032KhB8^?9P2JgRP26r$(0000000000!#P`?gT{5h?H|+?O=mGc0000000000ZQ(AY>&11z33@8hlUFi80000000000"
    "QmR12Q^j?_)|G{r+D9`$0000000000zum|EyTo<C(F>?(AviQZ00000000001;zVMB*b;VB6U<kXec#60000000000ZK)HwjKg)n>}X9TuNXE!0000000000"
    "Nz_I=^}=<)T7Xp%_6RpX0000000000;f~DGUBY$1<5MpNJ^4340000000000h$7s7#=&*K3VfXdgy}dy0000000000h1je?FTr)d@{wW*%h@?V0000000000"
    "EBbX8n80<w!7M=+63RM20000000000#L#X40l;;@p5^;2S-Lww0000000000ou$tEX}@*A231u`psGAT00000000001H#M*)4p}UBNPyC=bAl00000000000"
    "M{l4eJic|n6uR!3E{i@u0000000000vG#yerM-2)1mLsHb$dTR0000000000l%b`O4!w22IuDV9-Zel#0000000000IJrgLcf57L?FN1$0<J(n0000000000"
    "?uM8);k$Lft|NlRB_=^Y00000000000aVqTO1pKy6tRA6NuEJK0000000000ziBfMw7PY`uKQFKZ5Bd60000000000b`k1~9lCYEzE9@Bkd8t?0000000000"
    "ag;C>hq-mYe)$t`v<5>!0000000000{yRRW@VIrrghjs}*nUGm0000000000Y*9)=TDWz<WPi!h`}RXX00000000002p){(!?$(7u=5j$A8$lJ0000000000"
    "9_u=gEw^>Rr|y40L+3<50000000000{Q-_am$r4lhrI6iW?w}>0000000000@T<J|0Je3&;;BKjiq}Oz0000000000L^A%gYPEI1OafYQu1`il0000000000"
    "O5TZe)U<WLS`ryB(#J+X0000000000OZk>LKD2efNhV<M^*={I0000000000mdsoLsIzszRf-U@8MsG40000000000d7%=?5wmr`7K;mYJupZ>0000000000"
    "LBa2md$M)F8+0-`VW>zz0000000000I&~{u<*{|Z`jKY-gdj;k0000000000vIhMoPqB5t@GyzQsF+DW0000000000_r9C&xUhA=k)3;s%n(XI0000000000"
    "ROddnBCvJ9ZX+L4@Q6x400000000008Oa@bjIVXT9}iU<6ah;>0000000000lC5t#_O5ln-K_rOH+f4y00000000004&H0_U#@k)>`48rTk%Uk0000000000"
    ")T2qQ$*pz3+AlJ8e`!oW0000000000I!HiXGp%*NJgt5|qv1?I0000000000h!lqiovd}hrVYdh$5~B400000000004N_IB2CQ|!Se@h1>d{R=0000000000"
    "6-weoaI1B|jHtev4oXfy0000000000?nPzS*{XHG*?Tr)GQv(k0000000000=8hF#L#lPa&tfbqRyt2W0000000000NNy_Jtf_Ut1ge4Zd9qJH0000000000"
    "V=dcF7O8c>+RlsIP8Co<0000000000gcV=2f2ei95XEP|U{O#&0000000000_;2kC>8EwT>|Mp7a*a?x00000000004Til|Qm1vmIzN?zgvn4q0000000000"
    "5TOQ>yQX!(LJ!DfmIYBj0000000000P6`0PB&K!1By>(is6kOc00000000006bs?pjiq(K{u3D`x_wbV0000000000uI%9G_M~;d7?l_V%(+oO0000000000"
    "HL-`|U!--wJ0?it-StsG0000000000Km5bp$fI?@VOQwE?=ez90000000000m2##6GNX0Cxz?hn0d7)20000000000>(gd`o1%5Va6f6B6RJ``0000000000"
    "@ZUUv1)_DpGb}BmCFW8<0000000000R#sp&aG`a;GdSSFHzHF&0000000000(Dn$)+n{y8bN+JyN?uby00000000005P@!KNT7AV`x)9~T$)or0000000000"
    "$F#^pwx4yt`y~YPZq`#k0000000000sZmdbC7*S`9KGYQff7_e0000000000WfwYImY#LMk$P&UlulGY0000000000u*Jag2%dGoVfl&Kr;1cS0000000000"
    "IUE~#eVui{jMOtuyT(*N0000000000v)X_g^qh6TL#J5_&jM9I0000000000)4(YQZJc$$I1KF;<33eD0000000000M~+Qt=$m!GmY&~s_j*-80000000000"
    "^3~O8W}9`uP91SD47XK40000000000IyH9q=9+cDvDbt_A@fy00000000000d@>H>X_|GwY6y^{HZN8{0000000000p_m5u?wNJKEM53dOlwv^0000000000"
    "mNLizcA0g+EQBLKVy9L>0000000000P0PE`{+M;Zdn<Qsc;i+;0000000000uPsV3iI{c3CIMN|jvrS*0000000000t0dg;6qt3uWV0@1rCV1(0000000000"
    "EPLAAp_g^Q>F-@4yq8x%0000000000BK)nwFPC+|@MKH^)6-W#0000000000cyhmQzLs^sgTP<@=?_>y00000000008mV`&O_p`Q+_3!*0Zdpx0000000000"
    "`D#B^-IaB~vEz#{7>HOv00000000000E2%hZIyMvIR|iAFT_|t00000000009lW6U{FHUTg2@krMgUnr0000000000>%BvhjFff2VENanT|8Mp0000000000"
    "NP=?M8<cgxJJOEEba+`n0000000000#coo_s*`oV?pzbvi?mrl00000000000JlPBIg@q3bpo#BqVQQj0000000000kdC6Y$C7oxjVq_-xhz^h0000000000"
    "4k^*ORg!hU1VKmO&}dpf00000000005xOmQ<B@g1$*b1U=cHOd0000000000HFX#6aFKPuPt={e{@_|b00000000005RUC6{*ZORh9l*j6&+hZ0000000000"
    "JU9;?i;#7|I0E%}ELmGX0000000000Q>WY97m#(pDcKcGLzP=V0000000000?GCbJq>pvLQ4x?DTG3lT0000000000p`!7)FpqV>8WXJHa1C5Q0000000000"
    "3YPiYypDCidPlONhe=#O0000000000!W=2oNRD;DIv49*o`qaM0000000000U~1o?)Qxq(AM2hIwZU9K0000000000e6o8<V2yRa8Y~vX%l%wH0000000000"
    "vM627>x^~4phB5%;yGPF0000000000mjL54cZ_wwotJPH`E^}D0000000000$Zc$W0*rOQ;gjUJ53*fA0000000000+M`&mjf-`_SD-ytChlE80000000000"
    "Xx0YG7>jklbS1>`JStv500000000003^~@*ql$IFD2%{}Q)XU30000000000TmL1`E{b))JUx3HYNB310000000000^ITQGxrueaccDD3fZkp}0000000000"
    "XobV4M2U64%OOoWml|I{0000000000R;<s9&xm!vrzZ8rtyW(^0000000000kzUtsT8MSP#sqy(#FAe?0000000000kq<{x<%e~^ONSxY+Rk4<0000000000"
    "d+VM#aEEokiuD7pcm-fU00000000000}dl1`-XMEszxdCgCk%-0000000000>XH`?hK6;(YUPnKk3nES0000000000?3KX!5QcTYXaG-hnqOc*0000000000"
    "!!ly)n}v12Eyi)PrF~#P0000000000E`>nfCWUptMr8~0u$y2&0000000000=%}C8v4nNNPYf+Gyt!aN0000000000tGUC`JcM<?r|IW#$Jby$0000000000"
    "HPi6W$Afji@GDKP()D0K0000000000LJ@1!QiFBCvjJ}H-4kIz0000000000jgNNR-GX(%keOR6=`mqI0000000000)M;1gXo7XX<RK?%^iN?x0000000000"
    "(YsOm^MQ51P|4w@|7~GF0000000000LZPV=e}Q$tUcQ{<3XEYu0000000000;WeW#3W0UNnJ;M~7OG)D0000000000Yn6ghm4J1?o=aF@A;@7s0000000000"
    "o!g6hAb@qi&p>;gEaqWA0000000000Gb976tABOC$vxxTHw0op0000000000=ZY-hHh*=%7jdB*Ln2~80000000000cPmUD!hUtYLkY)PPC#Nn0000000000"
    "o*|}UO@4L2>@;PXSzcm500000000008Hs+b*nM@taw3h|WPD;k0000000000q<t9zWPNqObO2BraGGL30000000000`x?7l?|gN@ef_>%d$?ji0000000000"
    "*nFqKdVF=j6iDlyhSp+00000000000`@SMB27GnE+z#g9k@R9f00000000009{q5mk$ZK(GJwD(of2a}0000000000{8H8-9eZ`aw#tiYs4!zd0000000000"
    "P~4WOsCsq4@f=yOvrc0`0000000000+_d&OG<tQwaiux;zHDPa0000000000R`@yAzj<}Q(ULwt%8Fw^0000000000e!`@COL=v`aT9rn)u>}Y0000000000"
    "3@>{t*LZcn@m2cL;KpM>0000000000!C3+3Vt94H-H6j3?BruW0000000000R$hCn?ss*-(%Kbl_X1=<0000000000hgdCqdUtieE_%|s0wH8T0000000000"
    "5oB9f26uJ9l$9C_4nJf-0000000000v7AXjl6G~#j9dR)8C_&R0000000000ARat49(HxWpZbxoBzt5)000000000090YeXs&#e1ZlU7^FqvdP0000000000"
    "U}`==Hg$EtSAGRuJGW#&0000000000tZdd=!gO`O{J#jcM%83M0000000000xh8{wPIPs^=9K;rQuAa$0000000000L9~dq+H-ZloGbWkT@htK0000000000"
    "0x>K1W^;AG!LkO;XfI_z0000000000ytuAL^Kx~-v{*YXbWLSI0000000000C1+Qpe{yxe5lp9%e`{qx0000000000{8woZ403hAWxHGViiu@F0000000000"
    "{H7?2m~nN$I@M`kmZxPv0000000000;&0g$C2@7YEy&`@p~YoD0000000000YR<o{v2b<3p~BZTt>a}t0000000000Pu?F^K5%uwG4olbxB+HB0000000000"
    "P`f=2%WrkSY>0Xo#2;or0000000000CV^enS8sK|*sw~C&pl>90000000000j7!6=<ZgAq7Mt4y+goNp0000000000nIdABac*_MctbLO=6Pm70000000000"
    "sorIc{%v)@p{(fp@t0;m0000000000l+KBai*0qlzG-24{k3L50000000000P$Pzr7;SaH3YPx)2-Idk0000000000)(sPtrEGP;e@r5J6!K<30000000000"
    "8o%h3GHi9gS*R}h9}s6i00000000007ihnZziV~Cy~KriD=ue10000000000#5qrUOlx(()4YlIHcV$g00000000007iCaa*=lvb)vKIyLTYC~0000000000"
    "1{-@GW@>f7y{sVZO^9be0000000000i;RcN@@aLz$l7;fS*B+|0000000000oa!xIe`$5V98m<;WW;Ac0000000000GG->s3u$%0-YO_WaN=h`0000000000"
    "K`<#5m}qstN`pnMdjM!a0000000000#Ch&BBxrTOTXnt=haPA^0000000000u5&IVuxE9^OC0rWl00ZY0000000000{Z66bJ7;ykM3YOzomyx>0000000000"
    "qdfm-$7XfFX!Ro#sd#8W0000000000m~8;FQf76)^j+Xtw3cW<0000000000)Dm63-DP#a;^*R*zqDvT0000000000PhHK0Xk~T4Z?28Z%F<{+0000000000"
    "0e%Su^JI0vz|2DT)$nLQ0000000000-lfSmePngO`dwrf;SOm)0000000000-v!1Q2V`}?S$$M2>?~<O0000000000|IW6ikYjbg+C_CW_eyC%0000000000"
    "Ffvry8Dn+8x6&vw0%>VL0000000000Y+zQfqGENx6hF}+4Tot!0000000000sfu+ZD`IuPFQ(7~7^P`I0000000000;o)gAvtf0>0wamnBExAw0000000000"
    "3zzHFI$?Fd%+Oq%E#YZE00000000008r!l8!eDj4tjeuXIR9xt0000000000;tYVyNMLoqrbFNMLLF*A0000000000mH(+J&tG-Gt8CDXO*?8p0000000000"
    "SyoVDRbO?$9p48ZSXpX70000000000g!|FH+g^3RabqU5Vs~ml0000000000WZ8{}VqSH?55mk=ZIx<30000000000O4{NC>RolfR>PYRc(ZCi0000000000"
    "iRdqXa$R-6$1gPNg3)R~0000000000bMDhV{9JXwmCrWtjqhqe0000000000Rn{$xhg@~QJ8MKKnGI_|0000000000go>Q66kK({BFp`sq$_Jc0000000000"
    "Q1yQCqg!>rs<#7Put{q`00000000003js8mHCuJSP5Jh0yk~1b00000000002ACT1$y#;5Mp$^($c1Y_0000000000kff18VOn*-4xK2Z)uU@b0000000000"
    "#w;ey`&o6sgD)uG;=yY`0000000000U{3~{nOSwfR##_;@ZW1d0000000000y+VwBI$3qV+bJrV{{3q}0000000000{2iA`-dJ_OJokbQ4IFGh0000000000"
    "Lih#FgjjXJq7Q`M8#-)20000000000u;{nlEm(EHD8oMoD_Cqm0000000000Xa<Z**;jSI0b<C1I(BS80000000000jA&uzhF5jKB|I=gN|bCs0000000000"
    "J|-7~Hdl4P#-WKzTC!|F0000000000qE?LZ=T>#V1motCYS3&z0000000000*$|)xn^twe{7+gFd+uyN00000000001Z**pPgZrn<~a7oiwtc*0000000000"
    "Mt>h<1y*&yvhB>LoGNWV0000000000!eL2Kyj69;n7;6@tw?P^0000000000m@1AcbX9f0wEJ_}zGiJe0000000000?C{LNELC;DDD`L_(1dM30000000000"
    ";LGIb<y3XRCi~%d;i7Fo0000000000l+v_upHy|gqh0ah^1y9C0000000000EF2szSyXkv(`>0$1mA5y0000000000%)j~X5>$1-*mxYy75!~M0000000000"
    "lhiDU%u{v1&e2>$CmU`+0000000000q;ZtXg;RCF>c@PrI5}=W0000000000Ag&LIK2vqTAghiKN>^?`0000000000ETMgq_EL4grOTvRTXk+g0000000000"
    ">0!ORtx|QsmwC~PY?E$40000000000bqB82V^Vd%ECnZ>eX(vp0000000000`KT_l7*ch>R=&50j?ZpD0000000000lU*Jp%u#i~PW6>YpY3iy0000000000"
    "W;|T`eNlD5F8b5fuM2NL0000000000lw8LOE>U&B7afExz$tG)0000000000KGi-X+)#DEJ+8P$&_{1T0000000000i@l*4h){LFm;Hkr;AL+>0000000000"
    "mMqV_F;I2DQ(O3`?}KkZ0000000000hB>C+*iUu9mlr7J{-JL`0000000000eES7deNT13e1av;4Zv?e0000000000n2F8n9#3_^KI{Tp9Nuq00000000000"
    "0Q=i`y-sz&$)wVTD*SIi0000000000(q<6NSWb1oDt^-?91U<l0000000000HsndV@J)5VgCWV$A{=l)0000000000NQ!3?g-vz9d1O;WDJyV50000000000"
    "EO1Dt6-{-(DsOvwFFJ5Q00000000002Cd6VqD*zbjcH18HA!$l0000000000^jFZ%Dol02zkA6gJ6Lc)00000000009*8vsuS<2n&&>gqKxc430000000000"
    "p;X$xE=zU5%!mBDMRssN0000000000qST$xsY-Rg(b2+}N`-Jh0000000000K_6cUAWC(>*3(@jPn2*#0000000000q)Yufkx6yH__fwyQ=@P|0000000000"
    "=x|p${YZ7dMW8lHSF&(G0000000000GNw${V@P$t*Sdz=TETEY0000000000r2@R+#7A|&sV&eFUeItr0000000000V4Z|6AV+n;)F2$~VBc^+0000000000"
    "iH(Jhbw+i-X}oi1WA1Q30000000000KkA*+#6@+$cqrh0W&LnK0000000000rpm0(4Mlap7>#N$XAE&b0000000000-bImAP(*dWMiLx(Xd7`r0000000000"
    "3v((oj6-$68sPm^X)19*0000000000Pia>`!9sPwrXCr@X*qE~0000000000%s=-^??H9I`1N+cX-IKE0000000000qz&mM7D08uERG;ZX;*PT0000000000"
    "_~{FyH9&R1I1e9KXl8Lh0000000000>gzs~O+R(OI-z^*W_58u0000000000pekX@T|RZdLTo<;WrT4+0000000000IK5FCW<7PlTbOg4Vv}(|0000000000"
    ")w=IeW;}Jkq)j2PU!rk90000000000nmQUsUORQb6tq!2T(NOL0000000000s#CDYO*(bJ$ny<BSio^W0000000000Bps2dG&yy^&NN@RQ_pcg0000000000"
    "EUufo5;%3hG@nkOPTp}q0000000000<sGlk=Qee~7F*@@NbPYz0000000000OiS~<vo&?Vtt{8JLi}++0000000000Aq3=fb~JUsI%b#VI}36^0000000000"
    "a7TCbFEe$(a%hQ*G#YY10000000000vE#r$;4yW;tDqLFEGcq80000000000S*O%fhcI=(5PYakBsg+F0000000000-(bcuB`<Zr^Jl1e8%J_L0000000000"
    "x8AAQxh-|To6&zW5>|3R0000000000TJMreMJ#o|OD33t2xW3W0000000000JYkiG$0~KeR?}K-{&R9b0000000000)dIzhKq+;==|wU6^MZ0f0000000000"
    "o1!&duqSoER_lZ>>5+0k0000000000;=2Lx7AAGT@dcR?-k@?o0000000000=>O&HaU^xXD_4vc(ywwr0000000000AApQbyCQYK!~tj4!@hDr0000000000"
    "_k_E->>zc(uF2favCVQo0000000000;#N7+{~mR~0&}5(n%r_g0000000000Nj}U!?i_W%$H8cCeCu*R0000000000pMyH2u^M&24SMDnR{C;40000000000"
    "Om9l1Ll||y<6qAHB?)st0000000000#Yw{FofUP!KsVH(>KAiB0000000000d0S`@x)OE3KE$hnq9$`d0000000000(Ch-lkq>pi<H`skOf_>r0000000000"
    "Kt@=n9t?HBJP)4$=0kHp0000000000Ft9FPS_pN(TvZ~UZc=kV00000000006mv|FJ_L2ZJYhzD<Y03^0000000000RU9S)#{YD{L1HJkMQ?LJ0000000000"
    "Z-xRt@%nVYs(f5tj(u}L0000000000Y?Q;J!u52(I~UH~xr=i^0000000000yb^kiJn(eD1qxK7!<lnH0000000000#LYHYVd`|iK`q7`r>Ap30000000000"
    ">X8a$Gvsu@6j}VUU$t{U0000000000hPWOLwcT{U-+AHd=)!YA0000000000`&ghO>DF|>r_#<oLeO(S00000000001AF5f)6aCk&dWX!aoux30000000000"
    "093GhcF1(Vf!7>9d+KvQ0000000000YACV7*1&YYB1s*lX7_VI0000000000s2se(^|*AvdRKJ?HUxA)0000000000K;9C;&#-jB#xGdw;Sh8{0000000000"
    "cN}BsU#N7zYk7{<U>$To0000000000hlW~Xp`UcX?c3l%sVQ_o0000000000ugj)akCt@6$o?0uu{Cr+0000000000ESvT2B#m^ybBX+ya6xoH0000000000"
    "G^F%VU4wMMb0b%M+)Q*p00000000006P8h_GkJ8tFLdzR>sNF@00000000000U){^qi%G-EhCXJm11;20000000000J@MPurek!#<<|lH&TMo*0000000000"
    "`?zEPI#_hT*f|Tik9KrG00000000001S9@cSWI-lbQ9c8%zt!10000000000uFg>H0YG%XOzjohZHRP00000000000I(Ht;I5Tv>VwkltSder;0000000000"
    "0GEaS{v~w4iEK<DZ<lmH0000000000B{)c-Rv2`^E8s)JmYsA!00000000003bZDzND6eom&2(H%c68Z0000000000Iw{?=>HBlQU!4$EFsF1t0000000000"
    "H)`KzPwjKS8k!R0?y7V^0000000000g-M}zkKS{@UFJ#mF0OPy0000000000%G=z`z0Y&N$bTYs60vkZ0000000000OXy*Q;lOjiL0rJajk9z>0000000000"
    "fZD??{IPSukR;oOfwgo%0000000000;~&UU52JIyc{_ti&bD+w00000000007eZow9F%jwiphZQRJU|M0000000000Cj*X&C53aqV$8S(4Yzba0000000000"
    "J#3T8EOv9ij~Z}&61H?e0000000000)fxz+G-q?bcFYorgtT-(0000000000a;$-LKUj0Xc=`-ug0gf#0000000000TrKADPDyjXBIQ0<GO%<&0000000000"
    "P%r@JVLEfbbwcax*sXLx00000000005+Vv(b}MthC6djn!mD&Z00000000007zRF3iX3ym8=9pbJgRg+00000000009m@^pn+<coAVqnKj;VA&0000000000"
    "LWeQCs{V4o!WeY(!>M#Y0000000000Y$&$Ox9@Vm)S;tLt*UfD0000000000)OABA#NcwkUK8?*8LV_b0000000000fhK3!(9v?hD#^sR*R6Cw0000000000"
    "H2b+);KFjiM7DF8%CB@l0000000000B!sNx_p@@qUq4cZ`>}LD0000000000=a#Sr9i?)>C(<8Lc(imt0000000000@d!9dRhDwVP28KtPPcSG0000000000"
    "#;>$IriXICf6IlpjJk9{0000000000<zSd<6?t;NXbqk7KfZK80000000000s%<fIsA_V+tIOOgb;5K&0000000000{ai?eWn6N=^B>ZSFvoO20000000000"
    "Pb!uXQ%-Wg1hs!ua?Er<0000000000Y0)f+cR+H$90{MnI?{AN00000000005l$Xm*)np#d79}Cir92O00000000002b_HVekF3iUgsEiVBd5=0000000000"
    "CbwDMZWwaF{cYwrzUFj50000000000EStw|uM2X(oU99tqwaJ-0000000000iK(F-NB?oaFmV2_5chOI0000000000U>9iTI`eVBd=B0V1ORnF0000000000"
    "wT$9Oi|28`M5OL}b_sPr000000000062U#?H{5Z+;J$IpUlVme0000000000?`x)CK+$o)zU!#Tw;gpr0000000000$3#h6q{eZ;Ke*LDb|`f~0000000000"
    "4FgkqW4v*|(Xl!Ek~4Kc0000000000PG0b0eX?=D%VieG1wVB_0000000000{|!#H@u+dYmtn1P#!7WS0000000000=;G!Wz@KrzmxK(-%vN<k0000000000"
    "meJ+H?v`=DKn{#56k~Nj0000000000!U;bem5y=13iI$%p>K6S0000000000@M@x4&4zKnyr)irdVF<20000000000k<`Djv43&EHleuqrHOSw0000000000"
    "twG?HS$T25W)U><D3^6W0000000000MA=c{z;kiH-@cq^{-kw40000000000wCiAD({6FV)^xyP8?$vl0000000000j~D{Ad}?vP-rKv>bis8%0000000000"
    "VE-w4sAh4%w4AH5|Ic+m0000000000vOmpELSu2j17NK1u-<h*00000000006nWs#HDGbTdd}K{eeHEX00000000005ldK}Ze4M}&yZ7HTKsiD0000000000"
    "I8`ZC*jjPGoKG`!JPUR}0000000000|HFjTVOVj%gwFw=8X9&$0000000000@<v=f3085y{M_1)^eA>f0000000000Va}s0(Nl52r&E%U%r|yG0000000000"
    "%3*Xlyisw$2b2-(q(*i?0000000000wFu|_%ujK^g8o72e^qus0000000000qskN)22OFnm-7tOUu1Sb0000000000*EW<FYD{s!{{C6?Mss#R0000000000"
    "^b$Hi`bu%YNRwk-HiC9Q00000000001gMXZwMlWnJq|M`Es=IW0000000000$YSKwnn-cL??%5jDxh{i0000000000{jZ19s7G<YWEB5&EU$Jz0000000000"
    "XZ*-k-bQi2olWV)GQM^|0000000000$h7w-I!1B8ugP=`Jk54M0000000000;YDK*yG3!ph+mjAN8ENm0000000000aU5IOUqx}iHRehnQ|op>0000000000"
    "6{6%@Cq;3<Rb^qgUix-G00000000000#(F|5=C*qQlY*iX$f~g0000000000Ch3v!Bt>z+c+1w?aTs?%0000000000<yBWcVnuPl_ap6Ccqey20000000000"
    "sFEHd&P8#+DC<NQd^UGL0000000000)Rys3Y({awRAC{Kd_{La0000000000&so}dKu2-F$Hv7(c~o~m0000000000{?g)0PDpXU(_3|_abtHt0000000000"
    "$N3a{nMrZLuwZjOWpZ~w0000000000lEr$rB};L@wG{(=Q-OCt0000000000!z~`N_DpfWC%b3fJCJuk0000000000>^*xB5KnQyk{-Oz9iMkV0000000000"
    "oh_x9bWw4@!}C~f_pNt80000000000MT&_=CRB02jL&DR#k_Yw0000000000i2=+GCs%R6cz!=|h0Aw900000000005J3>;cv^A52+cO6Gun4R0000000000"
    "fcE*o9bR$3(ubrh&FFVP0000000000de1DB7h-Y1Lwn!xPWN{}0000000000rFl$LXl8N1>T^EZwFP)U0000000000sG|J!5o>Y4Q$85b{u6jW0000000000"
    "r)h4q6mW6C=NIXKCn9)20000000000ivCLoV0Cf8L5e9sH86NU0000000000ELM8^*?MunN@Ja{F+O-e0000000000)OzB4Xn=9Rcnef}CQNuh0000000000"
    "$98Rd@r7}~GV5GT9a?xm0000000000MN|alT8nYO<Z8##A82?$0000000000lpT<(ijZ-@0P~`^H+FbH0000000000^xXNnW|eWk;OqXUZiIM10000000000"
    "r<^JS)tPa?^S$a%&XRaQ0000000000"
)

_SECTION3_RADIUS_TABLE_SHAPE = (2240, 3)
_DENSE_RACING_PATH_SHAPE = (5865, 3)
