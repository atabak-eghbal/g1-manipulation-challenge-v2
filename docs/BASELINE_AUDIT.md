# Baseline Audit: g1-manipulation-challenge

## Repository Tree (Original)
- assets/ (Mesh files: .obj, .STL)
- run.py (Main simulation script)
- scene.xml (MuJoCo scene definition)
- g1.xml (Robot model)
- model_config.json (Joint/PD configurations)
- walker.onnx (Locomotion policy)
- croucher.onnx (Optional policy)
- rotator.onnx (Optional policy)
- right_reacher.onnx (Reaching policy)
- README.md

## Hardware & Asset Inventory
- **Required Core Files:** `run.py`, `scene.xml`, `g1.xml`, `model_config.json`, `walker.onnx`, `right_reacher.onnx`.
- **Defined Cameras:** `head_cam` (torso/forward), `wrist_cam` (right palm/outward).
- **ONNX Policies:** - **Core:** `walker.onnx` (99D obs), `right_reacher.onnx` (36D obs).
  - **Optional:** `croucher.onnx`, `rotator.onnx`.

## Current Controls and Limitations
- **Control Interface:** Keyboard-driven manual toggling between Walk and Reach modes.
- **Functional Limitations:** No autonomous sequencing; 100% manual intervention required for task completion; no vision processing integrated into control loop.
