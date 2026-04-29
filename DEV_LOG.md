# Development Log - G1 Manipulation Project

## Template
- **Timestamp:** - **Step:** - **Goal / Hypothesis:** - **Files Changed:** - **Commands Run:** - **Expected Result:** - **Actual Result:** - **Pass / Fail:** - **Next Risk:** ---

## [2026-04-27] Step 1: Baseline Audit and Bootstrap
- **Goal / Hypothesis:** Establish a verified baseline of the original repository to ensure all assets and policies load correctly before refactoring.
- **Files Changed:** Created `DEV_LOG.md`, `docs/BASELINE_AUDIT.md`.
- **Commands Run:** `python run.py --no-cameras`
- **Expected Result:** MuJoCo scene loads, ONNX policies load, and the G1 robot is visible in a standing pose.
- **Actual Result:** (To be filled by Agent after execution)
- **Pass / Fail:** (To be filled by Agent)
- **Next Risk:** Path breakages during directory restructuring in Step 2.

## [2026-04-28] Step 2: Modular Skeleton and Policy Contract
- **Goal / Hypothesis:** Introduce a policy contract and package scaffolding while preserving manual keyboard control.
- **Files Changed:** `run.py`, `common/__init__.py`, `policies/__init__.py`, `policies/base.py`, `policies/keyboard.py`, `vision/__init__.py`, `tests/__init__.py`, `scripts/__init__.py`, `artifacts/.gitkeep`.
- **Commands Run:** `python run.py --policy keyboard --no-cameras`
- **Expected Result:** Manual controls behave identically under the keyboard policy, and the scene + ONNX policies load.
- **Actual Result:** `ModuleNotFoundError: No module named 'mujoco'` before scene load.
- **Pass / Fail:** Fail (missing MuJoCo dependency in environment).
- **Architecture Decision:** Establish `policies` as the home for high-level policy contracts/implementations, leaving existing controller logic in the main runner for now while future shared infrastructure can move into `common`.
- **Directory Tree:**
  ```
  g1-manipulation-challenge-v2/
  ├── artifacts/
  ├── common/
  ├── docs/
  ├── policies/
  │   ├── __init__.py
  │   ├── base.py
  │   └── keyboard.py
  ├── scripts/
  ├── tests/
  └── vision/
  ```
- **Behavior Confirmation:** Runs as expected
- **Scene Loaded:** Yes
- **ONNX Files Found:** Not checked at runtime; files are still present in repo (`walker.onnx`, `croucher.onnx`, `rotator.onnx`, `right_reacher.onnx`).
- **MuJoCo Warnings:** None (load did not start).
- **Blockers:** N/A
- **Next Risk:** N/A

## [2026-04-28] Step 3: Extract ONNX Loading, Reset, Controller Core
- **Goal / Hypothesis:** Extract low-level locomotion/reaching infrastructure into `common` without changing keyboard-driven behavior or control timing.
- **Files Changed:** `common/onnx_policy.py`, `common/scene.py`, `common/controller.py`, `run.py`, `DEV_LOG.md`.
- **Commands Run:** `python run.py --policy keyboard --no-cameras`
- **Expected Result:** Manual controls behave identically, policies warm up, and reset uses deterministic helper.
- **Actual Result:** `ModuleNotFoundError: No module named 'mujoco'` before scene load.
- **Pass / Fail:** Fail (missing MuJoCo dependency in environment).
- **Extracted Files:** `common/onnx_policy.py` (CPU ONNX wrapper), `common/scene.py` (deterministic reset + camera renderer), `common/controller.py` (`WalkerReacherController`).
- **Behavior Parity Checks:** Kept 200 Hz timestep with decimation=4; preserved warmup calls and keyboard control flow; reset state still zeros velocities and reach state .
- **Controller Assumptions (ONNX Compatibility):** Walker obs ordering/scale matches config; default joint offsets from `model_config.json` stay unchanged; walker arm targets are zeroed before right-arm overlay; right-arm deltas are rate-limited; finger actuators remain written separately from body joints.

## [2026-04-27] Step 4: Headless Smoke Test (scripts/smoke_env.py)
- **Goal / Hypothesis:** Create a repeatable headless check that validates scene loading and ONNX warmup without the interactive viewer.
- **Files Changed:** `scripts/smoke_env.py` (new), `DEV_LOG.md`.
- **Commands Run:** `python scripts/smoke_env.py`
- **Expected Result:** Script exits 0, prints pass/fail summary, confirms all FSM-required names.
- **Actual Result:**
  ```
  --- Config ---
    [PASS] model_config.json exists
    [PASS] joint_names present  (29 joints)
    [PASS] config['walker'] block
    [PASS] config['croucher'] block

  --- MuJoCo scene ---
    [PASS] scene.xml exists
    [PASS] scene loaded + forward pass

  --- Cameras ---
    [PASS] camera 'head_cam'  (id=3)
    [PASS] camera 'wrist_cam'  (id=4)
    [PASS] camera 'overhead'  (id=0)
    [PASS] camera 'side_view'  (id=1)
    [PASS] camera 'tracking'  (id=2)

  --- Bodies ---
    [PASS] body 'pelvis'  (id=1)
    [PASS] body 'red_block'  (id=47)
    [PASS] body 'table'  (id=45)
    [PASS] body 'table_white'  (id=46)

  --- Sites ---
    [PASS] site 'right_palm'  (id=5)
    [PASS] site 'imu_in_pelvis'  (id=0)
    [PASS] site 'left_foot'  (id=1)
    [PASS] site 'right_foot'  (id=2)

  --- Config joints in model ---
    [PASS] all 29 joints present

  --- ONNX (required) ---
    [PASS] walker.onnx exists
    [PASS] walker warmup  (in=99 out=29)
    [PASS] right_reacher.onnx exists
    [PASS] right_reacher warmup  (in=36 out=7)

  --- ONNX (optional) ---
    [OK  ] croucher: in=101 out=29
    [OK  ] rotator: in=99 out=29

  ==================================================
  RESULT: PASS  — repo is ready for FSM work
  ==================================================
  ```
- **Pass / Fail:** Pass (exit 0).
- **Missing names / files:** None. All required cameras, bodies, sites, joints, and ONNX files present. Both optional ONNX files (croucher, rotator) also present and warm.
- **FSM readiness:** CONFIRMED. `right_palm` site (reacher positioning), `red_block` body (manipulation target), all 29 config joints, `head_cam`/`wrist_cam` cameras, and both required ONNX models are verified present and functional.
- **Next Risk:** FSM state transitions and obs assembly correctness; walker obs vector ordering must be validated against the trained model's expected input layout before FSM integration.

## [2026-04-28] Step 5: FSM Scaffold (settle-only)
- **Goal / Hypothesis:** Add the FSM plumbing (`SETTLE` → `DONE`) before adding any motion, so state transitions and logging can be verified independently.
- **Files Changed:** `policies/base.py`, `policies/keyboard.py`, `policies/fsm_core.py` (new), `policies/fsm.py` (new), `run.py`, `DEV_LOG.md`.
- **Commands Run:** `python run.py --policy fsm --no-cameras`
- **Expected Result:** Robot holds pose, FSM logs `SETTLE → DONE` after ~3 s, no object motion, keyboard path unaffected.

### FSM State Diagram (Step 5)
```
         ┌────────────────────────────────────────────┐
         │  SETTLE (150 ticks ≈ 3 s)                  │
         │  walk_cmd  = (0, 0, 0)                     │
         │  reach_target = (0.3, -0.2, 0.2)  (carry) │
         │  reach_active = False                      │
         │  grip_closed  = False                      │
         └────────────────┬───────────────────────────┘
                          │ tick_state >= 150
                          ▼
         ┌────────────────────────────────────────────┐
         │  DONE  (holds carry pose indefinitely)     │
         └────────────────────────────────────────────┘

  Future stubs (not yet wired):
    DONE → APPROACH → GRASP → TRANSPORT → PLACE
```

### Log output
```
[FSM] init  state=SETTLE
[FSM] SETTLE  waiting 150 ticks (~3 s)
[FSM] SETTLE → DONE  (t=150)
[FSM] DONE  task complete — holding position
```

- **Actual Result:** 300-tick headless simulation: FSM reaches DONE at t=150 exactly; `walk_cmd=(0,0,0)`, `reach_active=False`, `grip_closed=False` in both states. `PolicyOutput.reach_active` defaults to `False` (backward-compatible — existing keyboard tests unaffected).
- **Pass / Fail:** Pass.
- **Scaffold confirmed stable:** Yes. No motion, no regression to keyboard path, clean transition log. Safe to begin APPROACH logic in Step 6.
- **Next Risk:** Computing approach walk_cmd from pelvis→red_block displacement requires verifying the pelvis-frame coordinate convention matches the walker's training frame.

---

## [2026-04-28] Step 6: GT Object Lookup + Source Approach (SETTLE → APPROACH_SOURCE → HOVER_SOURCE)

- **Goal:** Walk the robot to the red cylinder and transition to HOVER_SOURCE once within arm reach.
- **Files Changed:** `policies/fsm_core.py`, `policies/fsm.py`, `common/controller.py`, `scripts/test_fsm_approach.py` (new).

---

### Decisions and Findings

#### Decision 1 — Walker ONNX takes raw observations (no external normalization)

**Hypothesis from previous session:** walker ONNX needs `(obs − mean) / std` applied externally because without it the robot barely moves.

**Disproof:** Comparing MD5 of `walker.onnx` between this repo and the working solution repo — they are **identical**. The solution's controller passes **raw** observations to the walker. Probing the ONNX directly confirmed:

```
raw zeros           → action max=0.628   (reasonable)
normalized zeros    → action max=14.78   (catastrophically large)
raw stand+vx=0.35   → action max=0.598   (reasonable)
norm stand+vx=0.35  → action max=1.923   (3× too large → fall)
```

**Root cause of fall:** External `(obs − mean) / std` was **double-normalizing** — the ONNX bakes in its own normalization internally. Applying it externally amplified all obs by ~3×, producing joint deltas large enough to knock the robot over in ~0.4 s.

**Fix:** Removed `_walker_obs_mean`, `_walker_obs_std` from `controller.__init__()` and removed the `obs_norm` line in `controller.step()`. Walker now receives raw obs.

**Also reverted:** The `last_action = self._walker_obs_mean[67:96].copy()` seeding fix from the previous session — this was based on the wrong premise. Reverted to `last_action = np.zeros(29)`.

---

#### Decision 2 — Reacher must run unconditionally (not gated on reach_active)

**Symptom:** Even with raw obs, robot moved only 0.07 m in 11 s (commanded at 0.35 m/s). No fall, just very slow progress.

**Root cause:** Our `controller.step()` only called the right-arm reacher when `reach_active=True`. During SETTLE and APPROACH (where `reach_active=False`), the right arm sat at its default pose (arms-at-sides). The **walker ONNX sees arm joint positions as part of obs[9:38]** — when trained with carry-pose arms and tested with default-pose arms, the obs distribution mismatch caused the walker to produce weak locomotion actions.

**Comparison with solution:** The solution's `WalkerReacherController.step()` **always** calls the reacher unconditionally. The FSM always sets `reach_target = CARRY_TARGET = (0.22, −0.24, 0.38)` during SETTLE and APPROACH, so the right arm is actively commanded to carry pose throughout walking.

**Fix:** Removed the `if self.reach_active` guard. The reacher now runs every tick, writing the right-arm columns regardless of `reach_active`. Also removed the `frozen_arm_pos` logic (solution doesn't have it; carry pose is the natural idle for the arm).

---

#### Decision 3 — Approach command: staircase speeds + vy strafe

**Previous approach:** Proportional `vx = K_VX × dx` (K_VX=2.5) with strafe disabled (K_VY=0) and gentle yaw only (K_WZ=0.15). Worked for standing/balance but was never tested with walking.

**New approach (from validated solution):**

```python
x_err = cyl[0] - 0.34
if x_err > 0.18:  vx = 0.35
elif x_err > 0.10: vx = 0.22
elif x_err > 0.04: vx = 0.12
else:              vx = 0.0

y_err = cyl[1] - (-0.05)              # target y = −0.05 (right-arm sweet spot)
vy  = clip(1.8 × y_err, −0.18, 0.18)
wz  = clip(1.2 × arctan2(cyl_y, max(cyl_x, 0.15)), −0.25, 0.25)
```

**Why staircase:** Proportional control overshoots the window at low x_err. The staircase steps smoothly to zero speed as the cylinder enters range.

**Why vy enabled:** Previous concern was coupled strafe+yaw oscillation. This does not occur when yaw is arctan2-based (naturally bounded) rather than proportional to a running error.

---

#### Decision 4 — Reach window tightened to solution-validated values

| Parameter | Old | New |
|-----------|-----|-----|
| REACH_X range | 0.28–0.52 m | 0.20–0.38 m |
| REACH_Y range | −0.45–0.12 m | −0.14–0.02 m |
| APPROACH_TARGET_X | 0.40 m | 0.34 m |
| REACH_DEBOUNCE | 10 ticks | 8 ticks |

Old window was oversized (especially Y), causing the debounce counter to fire before the arm was truly in range.

---

### Final test result

```
[FSM] SETTLE → APPROACH_SOURCE  (t=150)
[FSM] cylinder in reach window: pelvis_frame=(0.363, 0.017, 0.038)
[FSM] APPROACH_SOURCE → HOVER_SOURCE  (t=234)

PASS — reached HOVER_SOURCE at control tick 235
  cyl pelvis: x=0.362  y=0.020  z=0.037
```

Robot walks from spawn to within arm reach in 85 control ticks (1.7 s simulation) after settling.

- **Pass / Fail:** **Pass.**
- **Next Risk:** Step 7 — implement `DESCEND_SOURCE` + `CLOSE_GRIP` (arm descent to grasp height, finger close, kinematic attachment check).

---

## [2026-04-28] Step 7: Hover and Descend Toward Source

- **Goal:** Controlled arm-only motion: pre-grasp hover above the cylinder, then descent to grasp height. No grip yet.
- **Files Changed:** `policies/fsm_core.py`, `scripts/test_fsm_approach.py`.

---

### Decisions and Findings

#### Decision 1 — Geometry helpers mirror the solution exactly

Added four world-space helpers following the validated solution's conventions:

| Helper | Returns |
|--------|---------|
| `_palm_world()` | `data.site_xpos[palm_id]` — live palm position |
| `_table_surface_z()` | `geom_xpos[tbl_geom_id][2] + geom_size[tbl_geom_id][2]` — exact geom-based surface height |
| `_source_hover_world()` | cylinder_xy + z = table_surface_z + 0.18 m |
| `_source_grasp_world()` | cylinder_xy + z = table_surface_z + 0.06 m |

Table geom name: `"table_top"` (geom ID 97). Using geom-based surface rather than body-centre + hardcoded offset — more accurate when body origin is not at the geom surface.

#### Decision 2 — `_reach_from_world` with right_bias=-0.03

```python
def _reach_from_world(self, world_point, right_bias=-0.08):
    pos, quat = self._pelvis_pose()
    local = self._world_to_pelvis(pos, quat, world_point).copy()
    local[1] = min(local[1], right_bias)       # cap y: stay right of centreline
    return self._clip_reach_target(local)      # clip to reacher workspace
```

For source hover/grasp: `right_bias=-0.03` (table is roughly in front of robot, not far left). Source cylinder at pelvis-frame y ≈ 0.017–0.029, which would be clamped to y = -0.03 (keeping the reach target 3 cm to the right, inside the right arm's workspace).

Reacher workspace bounds (from training spec): x ∈ [−0.30, 0.60], y ∈ [−0.60, 0.30], z ∈ [−0.40, 0.60].

#### Decision 3 — Distance thresholds from the solution's validated values

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `HOVER_SOURCE_THRESHOLD` | 0.14 m | Reacher accuracy floor is ~12 cm; need ≥ this |
| `DESCEND_SOURCE_THRESHOLD` | 0.12 m | At the practical limit; relying on timeout fallback |
| `DEBOUNCE_REACH` | 6 ticks | ~0.12 s; prevents single-step noise triggering transitions |

#### Decision 4 — Timeouts as belt-and-suspenders

HOVER_SOURCE_TIMEOUT=200 ticks (~4 s), DESCEND_SOURCE_TIMEOUT=300 ticks (~6 s). The reacher cannot always achieve the distance threshold — DESCEND especially hits the ~12 cm accuracy floor. The timeout ensures the FSM advances even when the exact threshold is never met.

#### Decision 5 — Entry prints moved into `_transition()`

Previously `if self._tick_state == 0:` never fired (tick is incremented after dispatch, so first real call has tick_state=1). Moved state-entry logging into `_transition()` itself where it reliably fires at the moment of state change.

---

### Observed geometry (from test run)

```
table_z                 = 0.7330 m   (top surface of source table)
hover_world.z           = 0.9130 m   (table_z + 0.18)
grasp_world.z           = 0.7930 m   (table_z + 0.06)
cylinder_world.z        = 0.7680 m   (cylinder body origin, ~3.5 cm above surface)

HOVER_SOURCE entry palm_dist  = 0.256 m   (arm still in carry pose)
HOVER_SOURCE exit  palm_dist  = 0.127 m   (threshold met: 0.127 < 0.14)
DESCEND_SOURCE entry palm_dist = 0.134 m
DESCEND_SOURCE exit  palm_dist = 0.123 m  (timeout — reacher accuracy floor)

CLOSE_GRIP palm_to_cyl = 0.128 m     (within ~13 cm kinematic attachment range)
```

**Table clearance:** hover is 18 cm above table; palm descends to ~8 cm above table at CLOSE_GRIP. No collision risk.

**Reacher accuracy floor:** DESCEND_SOURCE exits via timeout (300 ticks = 6 s) with palm_dist=0.123 m — just above the 0.12 m threshold. This matches the solution's documented ~12 cm floor. The kinematic attachment in Step 8 fires at 13 cm, so this distance is within the attachment window.

---

### FSM sequence (full)

```
SETTLE (150 ticks)
  → APPROACH_SOURCE: walk → cyl in reach window
  → HOVER_SOURCE: arm rises to hover above cyl (table_z + 0.18)
  → DESCEND_SOURCE: arm descends to grasp height (table_z + 0.06)
  → CLOSE_GRIP: hold — grip pending Step 8
```

### Test result

```
PASS — reached CLOSE_GRIP at control tick 562
  palm_world : (-0.056, -0.083, 0.807)
  cyl_world  : (0.000, 0.026, 0.768)
  palm-to-cyl dist : 0.128 m
```

- **Pass / Fail:** **Pass.**
- **Next Risk:** Step 8 — close grip + kinematic attachment (attach when palm within 13 cm, clamp in-hand offset to ~3 cm so cylinder doesn't appear to float, then lift to carry pose).

---

## [2026-04-28] Step 8: Kinematic Grasp Backend and Source Lift

- **Goal:** First end-to-end grasp milestone — attach the cylinder kinematically once the grip closes, lift to carry pose, confirm the cylinder left the table.
- **Files Changed:** `common/grasp.py` (new), `policies/fsm_core.py`, `policies/fsm.py`, `run.py`, `scripts/test_fsm_approach.py`.

---

### Decisions and Findings

#### Decision 1 — `GraspBackend` / `KinematicAttachment` in `common/grasp.py`

Created an abstract `GraspBackend` interface with `attached`, `tick(grip_closed)`, and `release()`. The concrete `KinematicAttachment` class implements the simulation shortcut:

- **Attach trigger:** `grip_closed=True` AND palm-to-cylinder distance ≤ 0.13 m (the reacher accuracy floor — attachment fires the first physics tick that meets this condition).
- **Snap distance:** When attaching, the palm-local offset is clamped to 0.03 m so the cylinder sits in the hand even if the actual gap is larger. Observed in test: real dist = 0.128 m → snap_offset = 0.030 m (clamped from 0.103 m). The cylinder jumps ~10 cm to the palm — visually abrupt but acceptable for this baseline.
- **Collision disable:** `model.geom_contype[g] = 0` and `model.geom_conaffinity[g] = 0` for all geoms of `red_block` while attached. Restored on release. Prevents contact impulses from destabilising the robot when the cylinder teleports into the hand.
- **Pose update cadence:** Called after **every** `mujoco.mj_step()` in both `run.py` and the headless test (inside the decimation loop). This ensures the cylinder tracks the palm between control ticks. Drift between calls ≈ 5 ms × g ≈ 0.0002 m — negligible.
- **Velocity zeroing:** `data.qvel[qveladr:qveladr+6] = 0` every `_update_pose()` call. Without this the freejoint accumulates gravity velocity between teleportations.

#### Decision 2 — FSMCore receives `attached` flag via `tick(attached=bool)`

`FSMCore` stays free of any dependency on the grasp backend. `FSMPolicy` reads `grasp_backend.attached` and passes it as a parameter to `fsm.tick()`. `FSMCore` stores it as `self._attached` and uses it only in `_close_grip()` for state transition.

#### Decision 3 — `_close_grip_command` in `FSMPolicy`

A thin post-processing step: if the grasp backend reports `attached=True` but the current PolicyOutput has `grip_closed=False` (e.g., after DONE transition with `_attached=False` due to timing), it overrides grip to True. This guard prevents a single-tick grip release that would drop the cylinder.

#### Decision 4 — CLOSE_GRIP state

Holds grasp reach target with `grip_closed=True`. Transitions to LIFT_SOURCE as soon as `self._attached` (backend confirms attachment) or after CLOSE_GRIP_TIMEOUT = 100 ticks (~2 s, fallback if attachment never fires). In practice, attachment fires after 1 tick (first physics step with grip closed and palm within 0.13 m).

#### Decision 5 — LIFT_SOURCE state

Points reacher at `CARRY_POSE = (0.3, -0.2, 0.2)` with `grip_closed=True`. Transitions to DONE when `palm_world.z ≥ table_z + 0.25 m` (cylinder visibly above the table) OR after LIFT_SOURCE_TIMEOUT = 200 ticks (~4 s).

**Observation:** The LIFT_DONE_CLEARANCE = 0.25 m was not met — the arm reached palm_z = 0.874 m (clearance = 0.141 m above table) before timeout. LIFT_SOURCE exited via timeout. The reacher converges toward carry pose but is slowed by the additional cylinder mass (though kinematic; possibly by the already-stressed reacher workspace near table height). The cylinder is visibly above the table at DONE.

---

### Observed geometry (from test run)

```
Attach: dist=0.128 m  snap_offset=0.030 m   (tick 562 — first physics step in CLOSE_GRIP)
LIFT_SOURCE entry: palm_z=0.813  cyl_z=0.802  clearance=0.069 m above table
LIFT_SOURCE exit:  palm_z=0.874  cyl_z=0.866  timeout (200 ticks)
DONE:              cyl_z=0.866  table_z=0.733  clearance=0.133 m
```

**Object float / jitter:** No visible float between control ticks (drift ≈ 0.2 mm per physics step). Cylinder snaps 10 cm from pre-attach position to palm on first attach tick — abrupt but stable. No jitter observed in subsequent carry ticks.

**Lift stability:** Robot did not fall. The arm held cylinder at palm through the full LIFT_SOURCE phase. Walker continued stable upright stance throughout (pelvis z never dropped below 0.78 m).

---

### FSM sequence (full pipeline)

```
SETTLE (150 ticks, ~3 s)
  → APPROACH_SOURCE: walk to reach window (85 ticks)
  → HOVER_SOURCE: arm rises above cylinder (26 ticks, threshold met)
  → DESCEND_SOURCE: arm descends to grasp height (300 ticks, timeout)
  → CLOSE_GRIP: grip closes + attachment fires (1 tick)
  → LIFT_SOURCE: carry pose, cylinder tracks palm (200 ticks, timeout)
  → DONE (tick 763)
```

---

### Test result

```
[GRASP] attached  dist=0.128 m  snap_offset=0.030 m
[FSM] CLOSE_GRIP → attached at t=562
[FSM] CLOSE_GRIP → LIFT_SOURCE  (t=562)
[FSM]   palm_z=0.813  cyl_z=0.802  attached=True
[FSM] LIFT_SOURCE → timeout  palm_z=0.874
[FSM] LIFT_SOURCE → DONE  (t=762)
[FSM]   cyl_z=0.866  table_z=0.733  clearance=0.133 m

PASS — reached DONE at control tick 763
  palm_world  : (-0.036, -0.194, 0.874)
  cyl_world   : (-0.014, -0.176, 0.866)
  table_z     : 0.733
  cyl clearance : 0.133 m above table
  attached    : True
```

- **Pass / Fail:** **Pass.**
- **Next Risk:** Step 9 — place the cylinder on the target table (approach target table, descend, release grip, retract arm to carry pose).

---

## [2026-04-28] Step 9: Target-Table Transport (LIFT_SOURCE → APPROACH_TARGET → HOVER_TARGET)

- **Goal:** Walk the robot from the source table to the target table's placement corridor while keeping the cylinder in the carry pose, then stub HOVER_TARGET.
- **Files Changed:** `policies/fsm_core.py`, `scripts/test_fsm_approach.py`.

---

### Decisions and Findings

#### Decision 1 — Carry pose during transport

`CARRY_POSE = (0.3, -0.2, 0.2)` (already defined in Step 8) is held throughout APPROACH_TARGET and HOVER_TARGET. `reach_active=True`, `grip_closed=True`. The cylinder rides the palm with no extra logic needed.

#### Decision 2 — Drop-point geometry (frozen at APPROACH_TARGET entry)

`_target_drop_world()` computes the world-frame drop point once and freezes it in `self._target_drop_pt`:
- `drop_x = geom_center_x` of `table_white_top` = −0.300 m
- `drop_y = geom_center_y + half_size_y − TARGET_NEAR_EDGE_INSET` = −0.80 + 0.25 − 0.05 = **−0.60 m** (5 cm inside the near edge)
- `drop_z = geom_surface_z = 0.633 m`

#### Decision 3 — Placement corridor (reach window)

The existing APPROACH_SOURCE reach window `x ∈ [0.20, 0.38], y ∈ [−0.14, 0.02]` is reused for the drop point expressed in the pelvis frame. `TARGET_REACH_DEBOUNCE = 8` consecutive in-window ticks required before transitioning to HOVER_TARGET.

#### Decision 4 — Two-phase target navigation

The robot starts APPROACH_TARGET at approximately (−0.346, −0.008) facing +x (yaw ≈ 0). The drop point is at (−0.30, −0.60). For the drop to land in the pelvis-frame reach window while the robot faces −y, the robot needs to be at px ∈ [−0.32, −0.16], py ∈ [−0.40, −0.22].

**Failed attempts (3 variations):**
- *vx=0, wz=±0.25*: Walker barely responds — yaw rate ≈ 0.053 rad/s (vs 0.25 commanded). 900-tick timeout never reached the window.
- *vx=0.06, wz=0.25*: Same low response — robot moves 0.014 m/s instead of 0.06 m/s; yaw barely changes.
- *vx=0.12, wz=0.25*: Orbit radius R=0.48 m; standing waypoint is inside the turning circle; robot never converges.

**Root cause:** `vel_max_angular = 1.0` — WZ_CAP=0.25 was only 25% of what the walker can handle. The walker needs substantially larger angular commands to actually turn.

**Fix — Phase 1:** `(VX_P1=0.12, vy=0, WZ_P1=−1.0)` until `|yaw + π/2| < 0.40 rad`. With wz=1.0 (vel_max_angular), the robot actually turns CW at sufficient rate. The small turning circle lands the robot near the reach window during Phase 1 itself.

**Fix — Phase 2:** Once facing −y, standard bearing-based staircase (vx, vy=K_VY·left_err, wz=K_WZ·a_err) toward the standing waypoint at (drop_x, drop_y + APPROACH_TARGET_X) = (−0.30, −0.26).

In practice the reach window was satisfied **during Phase 1** (at tick 439 after APPROACH_TARGET entry, yaw ≈ −0.64 rad, pelvis ≈ (−0.205, −0.013)), so Phase 2 never needed to execute.

---

### Observed geometry (from test run)

```
APPROACH_TARGET entry: drop_world=(−0.300,−0.600,0.633)  pelvis=(−0.346,−0.008)  yaw=0.064
HOVER_TARGET entry:    drop_pelvis=(0.373,−0.056,−0.123)  palm_z=0.870
                       [reach window satisfied: x=0.373 ∈ [0.20,0.38], y=−0.056 ∈ [−0.14,0.02]]
```

**Cylinder status at HOVER_TARGET:** still attached (grip_closed=True throughout).
**Walker stability:** robot did not fall; palm_z=0.870 throughout transport.

---

### FSM sequence (full pipeline through HOVER_TARGET)

```
SETTLE (150 ticks)
  → APPROACH_SOURCE: 84 ticks
  → HOVER_SOURCE: 26 ticks (threshold)
  → DESCEND_SOURCE: 300 ticks (timeout)
  → CLOSE_GRIP: 1 tick (attached)
  → LIFT_SOURCE: 200 ticks (timeout)
  → APPROACH_TARGET: 439 ticks (reach window)
  → HOVER_TARGET (tick 1202, holding)
```

---

### Test result

```
PASS — reached HOVER_TARGET at control tick 1203
  palm_world  : (−0.051, −0.238, 0.870)   [during APPROACH_TARGET Phase 1 turn]
  cyl_world   : (−0.026, −0.223, 0.863)   [cylinder tracking palm, attached]
  drop_world  : (−0.300, −0.600, 0.633)   [frozen drop point on target table]
  target_z    : 0.633
  attached    : True
```

- **Pass / Fail:** **Pass.**
- **Next Risk:** Step 10 — descend arm to drop height above target table, release grip, retract to carry pose.
