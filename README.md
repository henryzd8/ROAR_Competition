# ROAR_Competition

ROAR Simulation Racing Series — [Summer 2026](https://roar.berkeley.edu/simulation-racing/)

Monza Map v1.1 : Best clean time 320.20s

## Provenance

The controller evolved through discussions and simulation trials with assistance
from AI agents, informed by the [ROAR past results](https://roar.berkeley.edu/past-results/),
and included tuning and contributions adapted from publicly reviewed repositories.

## Design overview

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

## ROAR Monza optimization results

- Test date: 2026-08-21
- Map: Monza v1.1
- CARLA client: 0.9.12
- Simulator: 0.9.12-dirty
- Scoring: official `evaluate_solution` elapsed simulation time for three laps

| Version | Three-lap time | Major collisions | Result | Main change |
| --- | ---: | ---: | --- | --- |
| Public starter baseline | 1658.90 s | Not instrumented | Finished | Official public starter submission, locally measured; proportional controller with 20 m/s target |
| v1 | 579.70 s | 0 | Finished | Cyclic curvature profile, braking propagation, dynamic lookahead |
| v2 | 513.75 s | 0 | Finished | 8.5 m/s^2 lateral limit, 48 m/s profile cap |
| v3 | 461.85 s | 0 | Finished | 11.5 m/s^2 lateral limit, 55 m/s profile cap |
| v4 | 426.50 s | 0 | Finished | 15.0 m/s^2 lateral limit, 60 m/s profile cap |
| v5 | DNF | 0 observed | Off track/stuck | Global 20.0 m/s^2 lateral limit, 20 m/s minimum |
| v6 | DNF | 0 observed | Off track/stuck | Interpolated global limit, 18.5 m/s minimum |
| v7 | 409.05 s | 0 | Finished | 17.5 m/s^2 lateral limit with 17 m/s minimum |
| v8 | DNF | 0 observed | Off track/stuck | 25.0 m/s^2 outside critical corners |
| v9 | 400.10 s | 0 | Finished | 20.0 m/s^2 outside critical corners; 17 m/s cap in critical corners |
| v10 | 398.05 s | 0 observed | Finished | Full 83 m/s straight envelope, one-tick gearbox launch, and validated 17.5 m/s^2 braking propagation |
| v11 | **327.70 s** | 0 observed | Finished | Optimized racing line with Menger-curvature, section-specific friction, steering, and speed control |
| v12 | **321.65 s** | 0 observed | Finished | Reactive three-point Menger-radius speed target, section-specific friction and heading-PID gains, a 0.80 hard-brake threshold, and two fixed low-brake stability zones on one optimized path |
| v13 | **320.35 s** | 0 observed | Finished | Dense racing-line tracker with speed-scheduled and distance-based lookahead, multi-radius braking preview, exact 10-section state, a 24-tick Section 3 braking horizon, and localized Section 5 `distance_gain=0.24` tuning |
| v14 | **320.20 s** | 0 observed | Finished | Raise the localized Section 5 `distance_gain` from 0.24 to 0.245 at waypoints 1320-1359, retaining all other v13 controller settings |

Best observed validated result: **320.20 seconds**. A second clean fresh-restart
run completed in **320.25 seconds**, reflecting normal simulation variation. The
320.20-second run is an **80.70% reduction** from the starter baseline and
approximately **5.18× faster**.
