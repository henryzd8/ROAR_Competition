# ROAR_Competition

[ROAR Simulation Racing Series](https://roar.berkeley.edu/simulation-racing/) — Summer 2026

Monza Map v1.1 : Best clean time 320.20s for v14 on 2026-08-21

- Monza Map v1.1 : Best clean time 320.00s for v16 on 2026-09-13

- Monza Map v1.1 : Best clean time 319.55s for v17 on 2026-09-19

## Provenance

The controller evolved through discussions and simulation trials with assistance
from AI agents, informed with the [ROAR past results](https://roar.berkeley.edu/past-results/),
and included tuning and contributions adapted from publicly reviewed repositories.

## ROAR Monza optimization results

- Era: Summer 2026 v1–v14 rows tested 2026-08-21; v17 tested 2026-09-19
- Map: Monza v1.1
- CARLA client: 0.9.12
- Simulator: 0.9.12-dirty
- Scoring: official `evaluate_solution` elapsed simulation time for three laps
- Protocol for runtime numbers: fresh CARLA simulator restart per validated run

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
| **v16** | **320.00 s** | 0 | Finished | Bayesian-optimized steering-scale multipliers on Curva Grande (+6.4%), Lesmo 1 (+13.7%), Lesmo 2 (−4.4%), Parabolica (+3.6%), and a 0.029 brake-release boost on single corners |
| **v17** | **319.55 s** | 0 | Finished | Modularized onto the HKU platform: hand-built racing line with 13 baked late-apex widening windows (only apexes move; the L2 right-LEFT complex ±0.3 m), refined per-section grip ladder (μ 1:3.02, 2:3.45, 3:3.36, 4:3.24, 5:2.98, 7:2.85, 8:2.98; others per measured caps), BRAKE_K 815, and the Section-3 dual throttle model (μ 3.65) |

Best observed validated result: **319.55 seconds best print, 319.60 seconds typical** (v17),
measured across ~30 clean fresh-CARLA runs in the 319.55–319.80 s band, all zero-collision;
a four-generation benchmark run of this exact committed configuration printed **319.45 s**.
That is an **80.7% reduction** from the starter baseline and approximately **5.19× faster**.

## Design overview

The submission is a **six-module set** (`competition_code/`, loaded by the
unchanged competition runner):

1. `submission.py` — `RoarCompetitionSolution` orchestrates the modules once
   per simulator tick and is the only interface required by the competition
   runner; it also keeps the section state (ten calibrated ranges used for
   steering, friction, preview, and brake-recovery gain scheduling), waypoint
   progress tracking, and launch handling.
2. `WaypointLine.py` — the hand-built 5,865-point racing line with the
   13 late-apex widening windows baked in as raised-cosine left-normal
   shifts, plus the target-snapping used by most sections.
3. `LateralController.py` — pure-pursuit bicycle steering (4.7 m effective
   wheelbase, gain 1.5) toward the section-scheduled lookahead target.
4. `ThrottleController.py` — multi-preview curvature into grip-limited
   target speeds via the per-section μ ladder, `BRAKE_K`-projected backward
   through the braking-distance model, stateful bang-bang actuation with
   brake-hold latches and per-section throttle-recovery envelopes, and the
   Section 3 dual model with its own prediction window.
5. `SectionStats.py` — per-section live statistics used by the throttle
   model's prediction logic.
6. `SpeedData.py` — the speed-recommendation record shared between the
   throttle model's preview stages.

## v17 Technical Analysis

### The baked widening windows (the racing line)

`WaypointLine.py` carries the hand-built 5,865-point racing line (5,593 m)
with 13 late-apex widening windows — raised-cosine left-normal shifts,
positive = left:

```
1313:1393:+0.9, 1393:1473:-0.9     sec-2 entry/apex    (R 54.5 -> 63.6 m)
 883: 963:-0.8,  963:1043:+0.8     sec-1 entry/apex    (R 37.4 -> 41 m)
4036:4116:+0.5, 4116:4196:-0.5     sec-6 entry/apex    (R 65 -> 73 m)
1688:1768:+0.3, 1768:1848:-0.3     sec-3 entry/apex    (R 50 -> 55 m)
1790:1815:+0.15                    lesmo2_in compensator (wall-margin restore)
2790:2860:-0.3, 2860:2900:+0.3     sec-4 late-apex pair
2915:2985:+0.3, 2985:3055:-0.3     L2 right-LEFT complex
```

Only apex-side windows move — exit shifts were measured to fail. Magnitude
is per-corner and narrow: each step past these values has a measured crash
boundary. The line-grip coupling is exploited: each widened corner's μ was
re-raised to its new measured cap.

### The per-section grip ladder (ThrottleController)

μ = sqrt-model grip per section: `0: 3.05, 1: 3.02, 2: 3.45, 3: 3.36,
4: 3.24, 5: 2.98, 6: 3.30, 7: 2.85, 8: 2.98, 9: 2.10`. Every value is one
step from a measured crash boundary (S1 3.03 crash-loops, S2 3.50 crashes,
S4 3.29 departs at the L2 turn-in, the S3 new-model 3.7 crashes at the
lesmo_exit). `BRAKE_K 815` sizes the backward braking projection (down
from 825 — the L2 turn-in margin).

### Measured performance

| Metric | Value |
|---|---|
| Campaign band (~30 clean fresh-CARLA runs) | 319.55–319.65 s, best print **319.55 s** |
| Lap structure | 110.25 s standing / 104.35 s flying / 104.85 s flying |
| Top speed | 257 km/h (drag-limited) |
| Per-section traverses (flying lap) | S0 21.4 · S1 9.1 · S2 8.0 · S3 16.8 · S4 4.5 · S5 7.9 · S6 11.4 · S7 1.9 · S8 15.3 · S9 7.0 s |


## v16 Telemetry Analysis: Anatomy of a 320-Second Lap (historical — the v16-era predecessor)

> Full interactive report with SVG charts: [`docs/v16_telemetry_report.html`](docs/v16_telemetry_report.html)
> (open in a browser — self-contained, light/dark themes, no dependencies)
> This analysis was measured on the v16 stock-line configuration; v17 (this
> branch) superseded it by moving the line itself — see
> [`v17 Technical Analysis`](#v17-technical-analysis) above for what changed and why.

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

### Why 320 seconds is the ceiling hard to break

The grip-limit lap (all apexes at v@33) would take ~96 seconds. The achieved flying
lap is 104.5 seconds. The 8.5-second gap is almost entirely scrub — the car enters
each corner at v_target (above sustainable grip), releases to WOT, and scrubs 4–6 m/s
before the apex. No steering-gain or brake-release tuning can close this gap because
it doesn't change v_target. Breaking 320 seconds requires a controller that sustains
a_lat = 33 at the apex — which means model-predictive brake/throttle timing, not
reactive bang-bang.
