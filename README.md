# ROAR_Competition

[ROAR Simulation Racing Series](https://roar.berkeley.edu/simulation-racing/) — Summer 2026

Monza Map v1.1 : Best clean time 320.20s

## Provenance

The controller evolved through discussions and simulation trials with assistance
from AI agents, informed with the [ROAR past results](https://roar.berkeley.edu/past-results/),
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
| v11 | 327.70 s | 0 observed | Finished | Optimized racing line with Menger-curvature, section-specific friction, steering, and speed control |
| v12 | 321.65 s | 0 observed | Finished | Reactive three-point Menger-radius speed target, section-specific friction and heading-PID gains, a 0.80 hard-brake threshold, and two fixed low-brake stability zones on one optimized path |
| v13 | 320.35 s | 0 observed | Finished | Dense racing-line tracker with speed-scheduled and distance-based lookahead, multi-radius braking preview, exact 10-section state, a 24-tick Section 3 braking horizon, and localized Section 5 `distance_gain=0.24` tuning |
| **v14** | **320.20 s** | 0 observed | Finished | Raise the localized Section 5 `distance_gain` from 0.24 to 0.245 at waypoints 1320-1359, retaining all other v13 controller settings |
| **v16** | **320.00 s** | 0 | Finished | Bayesian-optimized steering-scale multipliers on Curva Grande (+6.4%), Lesmo 1 (+13.7%), Lesmo 2 (−4.4%), Parabolica (+3.6%), and a 0.029 brake-release boost on single corners; 50-evaluation GP + Expected Improvement search with fresh CARLA restart per evaluation; `snap_target_to_path` IndexError fix |

Best observed validated result: **320.00 seconds** (v16). The v16 run was
validated across three independent fresh-CARLA restarts (320.00, 320.05, 320.10s),
all with zero collisions. The 320.00-second run is an **80.73% reduction** from the
starter baseline and approximately **5.18× faster**.

## v16 Telemetry Analysis: Anatomy of a 320-Second Lap

> Full interactive report with SVG charts: [`docs/v16_telemetry_report.html`](docs/v16_telemetry_report.html)
> (open in a browser — self-contained, light/dark themes, no dependencies)

Per-tick telemetry analysis of the v16 best-config 3-lap run (320.00s, 0 collisions)
on the Monza simulation circuit. Flying lap 2 data (2,091 ticks, 0.05s timestep).

### Summary metrics

| Metric | Value |
| --- | --- |
| 3-lap total | 320.00 s |
| Flying lap (lap 2) | 104.50 s |
| Top speed | 71.5 m/s (257 km/h) |
| Track length | 5,588 m |
| Average speed | 53.5 m/s |
| Collisions | 0 |
| Throttle usage | 93% WOT, 7% brake, 0% coast |

### The grip ceiling

The car is **scrub-limited, not grip-limited**. At every corner apex the car is at
full throttle (throttle = 1.0) yet achieves only a_lat ≈ 22 m/s² — well below the
tire grip limit of 33 m/s². This 37% unused grip headroom is the single largest
source of lost time.

| Metric | Value | % of tire limit |
| --- | ---: | ---: |
| Tire grip limit | 33.0 m/s² | 100% |
| Max a_lat achieved | 26.8 m/s² | 81% |
| P90 sustained a_lat | 20.7 m/s² | 63% |
| Unused grip headroom | — | 37% |

### Corner-by-corner analysis

Each corner's geometry, achieved speed, lateral acceleration, and grip utilization.
The "v@33" column shows the theoretical speed at the tire limit — the gap between
achieved v_min and v@33 is the time lost to scrub.

| Corner | Sec | R_min (m) | v_min (m/s) | v_max (m/s) | a_lat p90 | a_lat max | v@33 (m/s) | Thr mean | Brk mean | Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rettifilo | 0 | 74 | 27.1 | 64.3 | 11.7 | 13.4 | 78.3 | 1.00 | 0.000 | Chicane |
| Biassono | 1 | 51 | 34.9 | 68.4 | 22.3 | 25.5 | 41.0 | 0.86 | 0.146 | Straight+ |
| Curva Grande | 2 | 64 | 39.6 | 52.4 | 25.2 | 26.6 | 45.9 | 0.97 | 0.030 | Single |
| Roggia | 3 | 70 | 41.2 | 68.6 | 18.4 | 25.3 | 48.1 | 0.98 | 0.026 | Chicane |
| Lesmo 1 | 4 | 84 | 43.8 | 70.9 | 22.0 | 23.9 | 52.7 | 0.80 | 0.206 | Single |
| Lesmo 2 | 5 | 99 | 45.3 | 58.7 | 20.4 | 22.3 | 57.1 | 1.00 | 0.000 | Single |
| Ascari | 6 | 75 | 43.6 | 71.2 | 25.7 | 26.8 | 49.7 | 0.94 | 0.068 | Chicane |
| Vialone | 7 | 77 | 43.3 | 47.5 | 20.3 | 24.6 | 50.4 | 1.00 | 0.000 | Chicane |
| Straight | 8 | 203 | 47.6 | 71.3 | 8.0 | 11.2 | 82.0 | 1.00 | 0.000 | Straight |
| Parabolica | 9 | 25 | 24.7 | 71.5 | 24.6 | 26.2 | 28.7 | 0.79 | 0.211 | Single |

**The scrub signature:** at every single-corner apex (Curva Grande, Lesmo, Ascari,
Parabolica), throttle mean is 0.79–1.00 and brake mean is 0.00–0.03 — yet a_lat
plateaus at 20–27. The car is *wide open at the apex* and still can't reach the tire
limit. The `v@33` column shows how much faster each corner could theoretically be:
Parabolica could do 28.7 m/s (vs achieved 24.7), Curva Grande 45.9 (vs 39.6). That
4–6 m/s gap at every corner is the time the scrub limit costs.

### The throttle/brake trace: bang-bang with no trail-brake

The car is WOT (throttle = 1.0, brake = 0.0) 93% of the time. Braking occurs in
sharp 1–2 second pulses before corners, then immediately releases to full throttle —
even as the car continues to decelerate through the apex. This "WOT at apex"
behavior is the scrub: the car can't sustain the planned speed, so it scrubs off
5–10 m/s at full throttle. There is no trail-brake or lift-off coast phase — the car
is either full-throttle or full-brake, never in between.

### The steering command trace

Chicanes (Rettifilo, Roggia, Ascari, Vialone) produce sharp left-right direction
reversals where any controller perturbation crashes. Single corners (Curva Grande,
Lesmo, Parabolica) produce sustained steering deflections where the v16
steering-scale tuning has effect. The per-section multipliers (Curva Grande +6.4%,
Lesmo 1 +13.7%, Lesmo 2 −4.4%, Parabolica +3.6%) adjust authority on these sustained
deflections without touching the chicane transitions.

### Why 320 seconds is the ceiling

The grip-limit lap (all apexes at v@33) would take ~96 seconds. The achieved flying
lap is 104.5 seconds. The 8.5-second gap is almost entirely scrub — the car enters
each corner at v_target (above sustainable grip), releases to WOT, and scrubs 4–6 m/s
before the apex. No steering-gain or brake-release tuning can close this gap because
it doesn't change v_target. Breaking 320 seconds requires a controller that sustains
a_lat = 33 at the apex — which means model-predictive brake/throttle timing, not
reactive bang-bang.
