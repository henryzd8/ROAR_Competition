"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
from WaypointLine import WaypointLine
from SectionStats import SectionStats
import atexit

# from scipy.interpolate import interp1d

useDebug = False
useDebugPrinting = False
debugData = {}
dbg_carLocations = []
dbg_wpsToFollow = []
dbg_str = []
dbg_str2 = []
dbg_steer = []


def _steer_sections():
    """Parse ROAR_STEER_SECTION="2:1.15,6:1.10" into {2: 1.15, 6: 1.10}.

    A per-section multiplier on the steering gain, for tuning against a live
    server without editing this file between runs.  Unset in competition.
    """
    out = {}
    for part in os.environ.get("ROAR_STEER_SECTION", "").split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            try:
                out[int(k.strip())] = float(v)
            except ValueError:
                pass
    return out


_STEER_SECTION = _steer_sections()

# Sections that do NOT snap their target onto the hand-built racing line and
# follow the raw waypoints instead.  Section 9 holds the hairpin - radius 20 m,
# 72 km/h, 83 ticks a lap, the slowest point on the track by a wide margin - so
# whether the line is used there is worth a measurement rather than an
# assumption.  Default "0,9" is exactly the shipped behaviour.
_NOSNAP = set()
for _p in os.environ.get("ROAR_NOSNAP", "0,9").split(","):
    if _p.strip().isdigit():
        _NOSNAP.add(int(_p))


def _int_sections(name):
    """Parse "9:20,2:30" into {9: 20, 2: 30}."""
    out = {}
    for part in os.environ.get(name, "").split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            try:
                out[int(k.strip())] = int(float(v))
            except ValueError:
                pass
    return out


# The path the car actually drives is the pure-pursuit target, and two things
# set it: how far ahead it aims, and how many waypoints it averages to get
# there.  Averaging more points cuts the corner harder, which OPENS its radius
# - the one way left to raise corner speed once grip is maxed, and it needs no
# edits to the line itself.
#
# Section 9 averages 0 points, so the hairpin - radius 20 m, 83 ticks a lap,
# the slowest corner on the track - is driven completely unsmoothed.  That is
# why it did not respond to grip or to the racing line: it is geometry-bound,
# not speed-bound.  AVG_SECTION sets the count outright rather than scaling it,
# so a 0 can be lifted off the floor.
LOOKAHEAD_SCALE = float(os.environ.get("ROAR_LOOKAHEAD", 1.0))
AVG_SECTION = _int_sections("ROAR_AVG_SECTION")


def _float_sections(name):
    """Parse "9:3.5,1:0.6" into {9: 3.5, 1: 0.6}."""
    out = {}
    for part in os.environ.get(name, "").split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            try:
                out[int(k.strip())] = float(v)
            except ValueError:
                pass
    return out


SHIFT_SECTION = _float_sections("ROAR_SHIFT_SECTION")

# Two hand-placed brake events sit outside the speed controller entirely: one
# at waypoint 800 (section 3, the longest section at 331 ticks) and one at
# waypoint 2381 (section 9, on the approach to the hairpin).  Both are literal
# constants that were never swept - the trigger speeds decide how much straight
# is given away before each corner.  2381 fires above 257 km/h and the car's
# limiter is 257.3, so it is armed by a margin of 0.3 km/h; setting it to 999
# disables it outright.
def _widen_primary_specs():
    """Parse ROAR_WIDEN_PRIMARY="2521:2557:0.5,..." -> [(start, end, m), ...].

    Sections 0 and 9 are the only ones that never consult WaypointLine, so the
    line-widening that bought sub-320 cannot reach them - and between them they
    are 524 of a 2093-tick lap, a quarter of the race, including the hairpin at
    89 km/h.  Reaching them means shifting the raw waypoints instead.

    Note this array also feeds the throttle controller's radius lookahead for
    EVERY section, so a shift here changes corner-speed targets as well as the
    driven path.  That is coherent - the radius really did change - but it is
    why the windows must stay inside sections 0 and 9.
    """
    out = []
    for part in os.environ.get("ROAR_WIDEN_PRIMARY", "").split(","):
        bits = part.split(":")
        if len(bits) != 3:
            continue
        try:
            out.append((int(bits[0]), int(bits[1]), float(bits[2])))
        except ValueError:
            pass
    return out


def widen_waypoints(wps, specs):
    """Shift windows of a waypoint list along the local left-normal.

    Same raised-cosine profile as WaypointLine.apply_widen - zero at both ends
    of the window, peaking in the middle - so the path stays continuous.
    """
    if not specs:
        return wps
    n = len(wps)
    locs = [np.array(w.location, dtype=float) for w in wps]
    out = [np.array(l) for l in locs]
    for start, end, amount in specs:
        span = end - start
        if span < 4:
            continue
        for k in range(span + 1):
            i = (start + k) % n
            tx, ty = (locs[(i + 1) % n][0] - locs[(i - 1) % n][0],
                      locs[(i + 1) % n][1] - locs[(i - 1) % n][1])
            norm = math.hypot(tx, ty)
            if norm < 1e-6:
                continue
            nx, ny = -ty / norm, tx / norm
            w = 0.5 * (1 - math.cos(2 * math.pi * k / span))
            out[i][0] += nx * amount * w
            out[i][1] += ny * amount * w
        print(f"WIDEN_PRIMARY {start}-{end} by {amount:+.2f} m")
    return [
        roar_py_interface.RoarPyWaypoint(out[i], wps[i].roll_pitch_yaw,
                                         wps[i].lane_width)
        for i in range(n)
    ]


WIDEN_PRIMARY = _widen_primary_specs()

# The standing start is the single largest untouched block of time on the
# card: lap 1 spends 437 ticks in section 0 against 322 on a flying lap, so
# 115 ticks - 5.75 s - go to getting off the line.  The controller asks for
# full throttle from rest at tick 1, which is the one moment a tyre model will
# punish it: spinning wheels put down less force than gripping ones, so a
# capped launch can out-accelerate a maximal one.  Capping throttle for the
# first LAUNCH_TICKS ticks tests that.  Defaults are a no-op.
LAUNCH_TICKS = int(float(os.environ.get("ROAR_LAUNCH_TICKS", 0)))
LAUNCH_THR = float(os.environ.get("ROAR_LAUNCH_THR", 1.0))
# `gear = max(1, int(speed/60))` is commented "gears do not appear to have an
# impact" - never actually measured. GEAR_DIV changes that divisor.
GEAR_DIV = float(os.environ.get("ROAR_GEAR_DIV", 60.0))

S3_BRAKE_SPD = float(os.environ.get("ROAR_S3_BRAKE_SPD", 162.0))
S3_SLOW_SPD = float(os.environ.get("ROAR_S3_SLOW_SPD", 160.0))
WP2381_SPD = float(os.environ.get("ROAR_WP2381_SPD", 257.0))


def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(waypoints)
        d = dist_to_waypoint(location, waypoints[ind])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind


def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)


_SOLUTION = None


@atexit.register
def printRaceSummary():
    """Final scoreboard, printed even if the run is interrupted."""
    s = _SOLUTION
    if s is None or not getattr(s, "num_ticks", 0):
        return
    total = s.num_ticks * 0.05
    print("RACE  " + "=" * 58)
    print(f"RACE  FINAL: {total:.2f} s   top speed {s.best_speed:.1f} km/h")
    for i, t in enumerate(s.lap_times, 1):
        print(f"RACE    lap {i}: {t:7.2f} s")
    if len(s.lap_times) < 3:
        # The final lap ends when the runner stops, so it never triggers the
        # section-0 crossing that banners the other laps - derive it instead.
        part = total - sum(s.lap_times)
        print(f"RACE    lap {len(s.lap_times)+1}: {part:7.2f} s (final lap)")
    print("RACE    reference 321.70 s  |  this code as given 321.25 s")
    print("RACE  " + "=" * 58)


@atexit.register
def saveDebugData():
    print("Saving...")
    fname = "\\debugData\\line.txt"
    # A fresh checkout has no debugData/, and this runs at exit - so without
    # the makedirs the run ENDS on a FileNotFoundError traceback even though
    # the race itself completed cleanly. Harmless to the result, alarming to
    # anyone reading the output.
    os.makedirs(f"{os.path.dirname(__file__)}\\debugData", exist_ok=True)
    with open(
        f"{os.path.dirname(__file__)}{fname}", "w+"
    ) as outfile:
        outfile.write("\n--- Debug steer\n")
        for line in dbg_steer:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Locatons\n")
        for line in dbg_carLocations:
            outfile.write(f"{line}\n")
        outfile.write("\n--- wpsToFollow\n")
        for line in dbg_wpsToFollow:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Debug str\n")
        for line in dbg_str2:
            outfile.write(f"{line}\n")
        outfile.write("\n--- More Debug str\n")
        for line in dbg_str:
            outfile.write(f"{line}\n")
    print(f"Saved. {fname}")

    if useDebug:
        print("Saving debug data")
        jsonData = json.dumps(debugData, indent=4)
        with open(
            f"{os.path.dirname(__file__)}\\debugData\\debugData.json", "w+"
        ) as outfile:
            outfile.write(jsonData)
        print("Debug Data Saved")


class RoarCompetitionSolution:
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
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()
        self.section_stats = None
        self.section_indeces = []
        self.num_ticks = 0
        self.current_section = 0
        self.lapNum = 1
        self.previous_waypoint_to_follow = None
        self.max_radius = 10000
        self.previous_location = None
        self.total_dist = 0
        self.waypoint_line = WaypointLine()
        self.previous_brake = False
        self.s3_mult = 1
        # --- live lap timer (display only; no effect on control) ------------
        # The runner prints a line every tick, so pipe its stdout through
        # `Select-String "RACE"` to watch just this.  0.05 s per tick is the
        # runner's fixed control step, and it is simulated time, so these
        # figures match the official result exactly.
        self.lap_times = []
        self.lap_start_tick = 0
        self.best_speed = 0.0
        self._last_report_tick = 0
        global _SOLUTION
        _SOLUTION = self

    async def initialize(self) -> None:
        # NOTE waypoints are changed through this line
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
            )[35:]
        )
        self.maneuverable_waypoints = widen_waypoints(
            self.maneuverable_waypoints, WIDEN_PRIMARY)
        self.section_stats = SectionStats(
            self.maneuverable_waypoints, self.location_sensor, self.velocity_sensor)

        sectionLocations = [
            [-278, 372], # Section 0 start location
            [64, 890], # Section 1 start location
            [511, 1037], # Section 2 start location
            [762, 908], # Section 3 start location
            [198, 307], # Section 4 start location
            [-11, 60], # Section 5 start location
            [-85, -339], # Section 6 start location
            [-210, -1060], # Section 7 start location 
            [-318, -991], # Section 8 start location
            [-352, -119], # Section 9 start location
        ]
        # for i in sectionLocations:
        #     self.section_indeces.append(
        #         findClosestIndex(i, self.maneuverable_waypoints)
        #     )
        self.section_indeces = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indeces}")
        print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        self.previous_location = vehicle_location


    async def step(self) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        self.num_ticks += 1
        self.section_stats.step()

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1
                    lap_s = (self.num_ticks - self.lap_start_tick) * 0.05
                    self.lap_times.append(lap_s)
                    self.lap_start_tick = self.num_ticks
                    print(f"RACE  >>> LAP {len(self.lap_times)} COMPLETE: "
                          f"{lap_s:7.2f} s   (total {self.num_ticks*0.05:7.2f} s)"
                          f"   top speed {self.best_speed:.1f} km/h")

        # --- live timer readout (display only) ---
        self.best_speed = max(self.best_speed, current_speed_kmh)
        if self.num_ticks - self._last_report_tick >= 20:      # once a second
            self._last_report_tick = self.num_ticks
            total = self.num_ticks * 0.05
            lap_t = (self.num_ticks - self.lap_start_tick) * 0.05
            bar = "#" * int(current_speed_kmh / 10)
            print(f"RACE  lap {self.lapNum}  {lap_t:6.2f}s | total {total:7.2f}s"
                  f" | sec {self.current_section} | {current_speed_kmh:6.1f} km/h "
                  f"{bar}")

        nextWaypointIndex = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh, vehicle_location)
        waypoint_to_follow_location = waypoint_to_follow.location
        snap_to_line_location = self.waypoint_line.get_next_waypoint_location(waypoint_to_follow.location)
        if self.current_section not in _NOSNAP:
            waypoint_to_follow_location = snap_to_line_location

        # Pure pursuit controller to steer the vehicle
        steer_control, steer_debug = self.lat_controller.run(
            vehicle_location, vehicle_rotation, waypoint_to_follow_location, self.current_waypoint_idx
        )

        # Custom controller to control the vehicle's speed
        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            nextWaypointIndex : nextWaypointIndex + 300
        ]
        num_points_before_lookahead = 9
        wp_len = len(self.maneuverable_waypoints)
        wp_ind_for_throttle = ((nextWaypointIndex + wp_len) - num_points_before_lookahead) % wp_len
        additional_waypoints = (self.maneuverable_waypoints * 2)[
            wp_ind_for_throttle : wp_ind_for_throttle + 300
        ]
        throttle, brake, gear, speed_data, throttle_debug_str = self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
            additional_waypoints,
        )

        steerMultiplier = round((current_speed_kmh + 0.001) / 120, 3)
        # Per-section steering-authority trim, applied at the very end so it
        # composes with the hand-tuned ladder below.  Sections 2 and 6 refuse
        # any extra corner speed - if that is because the car runs wide rather
        # than because it runs out of grip, more steer there should unlock it.
        # Unset in competition (see _STEER_SECTION), so a no-op there.
        
        if self.current_waypoint_idx in [800, 801]:
            self.s3_mult = 0.85
            if current_speed_kmh >= S3_BRAKE_SPD:
                self.s3_mult = 0.95
                if not self.previous_brake:
                    throttle = 0
                    brake = 1
                    self.previous_brake = True
            if current_speed_kmh < S3_SLOW_SPD:
                self.s3_mult = 0.75
            print(f"spd {current_speed_kmh} mult{self.s3_mult} sec={self.current_section}")
        if self.current_waypoint_idx in [802, 803, 804]:
            self.previous_brake = False

        if self.current_section == 2:
            steerMultiplier *= 1.2
        if self.current_section in [3]:
            if self.current_waypoint_idx < 813:
                steerMultiplier *= self.s3_mult
            elif self.current_waypoint_idx < 845:
                steerMultiplier *= 1.45
            else:
                steerMultiplier *= 1
                self.s3_mult = 1

        if self.current_section == 4:
            steerMultiplier = min(1.45, steerMultiplier * 1.65)
        if self.current_section == 5:
            steerMultiplier *= 1.1
        if self.current_section in [6]:
            steerMultiplier = np.clip(steerMultiplier * 3.2, 3.1, 7)
        if self.current_section == 7:
            steerMultiplier *= 1.75

        if self.current_section == 9:
            if self.current_waypoint_idx > 2580:
                steerMultiplier = max(steerMultiplier, 1.7)
            else:
                steerMultiplier = max(steerMultiplier, 1.5)

        if self.current_section in _STEER_SECTION:
            steerMultiplier *= _STEER_SECTION[self.current_section]

        steer_value = np.clip(steer_control * steerMultiplier, -1, 1)
        # sec3
        if  820 < self.current_waypoint_idx < 837:
            steer_value = np.clip(steer_control * steerMultiplier, -0.007, 1)
        if self.current_waypoint_idx in [2381, 2382] and current_speed_kmh > WP2381_SPD:
            if not self.previous_brake:
              throttle = 0
              brake = 1
              self.previous_brake = True
        if self.current_waypoint_idx in [2383, 2384, 2385]:
            self.previous_brake = False

        if LAUNCH_TICKS and self.num_ticks <= LAUNCH_TICKS:
            throttle = min(throttle, LAUNCH_THR)

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": steer_value,
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,  # Gears do not appear to have an impact on speed
        }
        
        if useDebug:
            dbg_carLocations.append(f"{vehicle_location[0]}, {vehicle_location[1]}")
            dbg_wpsToFollow.append(f"{waypoint_to_follow_location[0]}, {waypoint_to_follow_location[1]}")

            self.total_dist += np.linalg.norm(vehicle_location - self.previous_location)
            self.previous_location = vehicle_location
            s = f"{self.total_dist:.0f}, {current_speed_kmh:.0f}, {speed_data.recommended_speed_now:.0f}, {speed_data.name}, {brake*10:.2f}"
            dbg_str.append(s)
            wp_ind = (self.lapNum-1)*3000 + self.current_waypoint_idx
            s = f"{wp_ind:.0f}, {current_speed_kmh:.0f}, {speed_data.recommended_speed_now:.0f}, {speed_data.name}, {brake*10:.2f}"
            dbg_steer.append(s)

            wpl = waypoint_to_follow_location
            d = np.linalg.norm(waypoint_to_follow.location - vehicle_location)
            s = f"d {self.total_dist:.0f} t {self.num_ticks} ind {self.current_waypoint_idx} \
sp {current_speed_kmh:.2f} rec {speed_data.recommended_speed_now:.1f} dif {(current_speed_kmh - speed_data.recommended_speed_now):.1f} \
r={speed_data.r:.0f}: {throttle_debug_str}, \
t {control['throttle']:.3f} \
br {control['brake']:.3f} \
st: {control['steer']:.10f}, \
{steer_control:.6f}, {steerMultiplier:.6f} trgt wp:ind {nextWaypointIndex} {nextWaypointIndex - self.current_waypoint_idx} {d:.1f} \
loc: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) wp({wpl[0]:.1f}, {wpl[1]:.1f}) {steer_debug} section {self.current_section}"
            dbg_str2.append(s)


        if useDebug:
            debugData[self.num_ticks] = {}
            debugData[self.num_ticks]["loc"] = [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ]
            debugData[self.num_ticks]["throttle"] = round(float(control["throttle"]), 3)
            debugData[self.num_ticks]["brake"] = round(float(control["brake"]), 3)
            debugData[self.num_ticks]["steer"] = round(float(control["steer"]), 10)
            debugData[self.num_ticks]["speed"] = round(current_speed_kmh, 3)
            debugData[self.num_ticks]["lap"] = self.lapNum

#             if useDebugPrinting and self.num_ticks % 5 == 0:
#                 print(
#                     f"- Target waypoint: ({waypoint_to_follow.location[0]:.2f}, {waypoint_to_follow.location[1]:.2f}) index {nextWaypointIndex} \n\
# Current location: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) index {self.current_waypoint_idx} section {self.current_section} \n\
# Distance to target waypoint: {math.sqrt((waypoint_to_follow.location[0] - vehicle_location[0]) ** 2 + (waypoint_to_follow.location[1] - vehicle_location[1]) ** 2):.3f}\n"
#                 )

#                 print(
#                     f"--- Speed: {current_speed_kmh:.2f} kph \n\
# Throttle: {control['throttle']:.3f} \n\
# Brake: {control['brake']:.3f} \n\
# Steer: {control['steer']:.10f} \n"
#                 )

        await self.vehicle.apply_action(control)
        return control

    def get_lookahead_value(self, speed):
        """
        Returns the number of waypoints to look ahead based on the speed the car is currently going
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
                return max(1, int(round(num_points * LOOKAHEAD_SCALE)))
        return max(1, int(round(8 * LOOKAHEAD_SCALE)))

    def get_lookahead_index(self, speed):
        """
        Adds the lookahead waypoint to the current waypoint and normalizes it so that the value does not go out of bounds
        """
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed)
        #       + " cur_ind " + str(self.current_waypoint_idx)
        #       + " num_points " + str(num_waypoints)
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(
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
    def next_waypoint_smooth(self, current_speed: float, vehicle_location: float):
        """
        If the speed is higher than 70, 'smooth out' the path that the car will take
        """
        if self.current_section == 3:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 44, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [5, 7]:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [6]:
            kdd = 0.28
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def new_RoarPyWaypoint(self, location):
        return roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=12.0)


    def average_point(self, current_speed):
        """
        Returns a new averaged waypoint based on the location of a number of other waypoints
        """
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2

        # Section specific tuning
        if self.current_section == 0:
            num_points = round(lookahead_value * 1.5)
        if self.current_section == 3:
            next_waypoint_index = self.current_waypoint_idx + 22
            num_points = 35
        if self.current_section == 4:
            num_points = lookahead_value + 5
            next_waypoint_index = self.current_waypoint_idx + 24
        if self.current_section == 5:
            # num_points = round(lookahead_value * 1.1)
            num_points = lookahead_value
        if self.current_section == 6:
            num_points = lookahead_value
            # num_points = 5
            next_waypoint_index = self.current_waypoint_idx + 28
        if self.current_section == 7:
            # Jolt between sections 6 and 7 likely due to the differences in lookahead values and steering multipliers. 
            num_points = round(lookahead_value * 1.25)
        if self.current_section == 9:
            # (self.current_waypoint_idx + 8) % len(self.maneuverable_waypoints)
            num_points = 0

        if self.current_section in AVG_SECTION:
            num_points = AVG_SECTION[self.current_section]

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

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
            # How far the averaged target may move off the raw waypoint. This
            # caps the whole averaging lever: past 2 m, adding points changes
            # nothing. line_room.py found 1.4-1.8 m of proven width at the
            # corners that matter, so the cap sits right on the scale of the
            # room available and may well be the binding constraint.
            max_shift_distance = SHIFT_SECTION.get(self.current_section, 2.0)
            if self.current_section == 1 and 1 not in SHIFT_SECTION:
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
