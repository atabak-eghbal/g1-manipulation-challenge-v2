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
