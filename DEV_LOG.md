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
- **Behavior Confirmation:** Not verified (MuJoCo unavailable); expected manual controls unchanged.
- **Scene Loaded:** No (MuJoCo import failed).
- **ONNX Files Found:** Not checked at runtime; files are still present in repo (`walker.onnx`, `croucher.onnx`, `rotator.onnx`, `right_reacher.onnx`).
- **MuJoCo Warnings:** None (load did not start).
- **Blockers:** Install MuJoCo in the environment to validate the manual controls and scene load.
- **Next Risk:** Manual controls cannot be revalidated until MuJoCo is installed.
