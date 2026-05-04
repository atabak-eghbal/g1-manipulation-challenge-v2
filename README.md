# G1 Pick & Place Challenge — Autonomous Baseline

This repository contains the autonomous pick-and-place baseline for the Unitree G1 humanoid. It converts the original manual keyboard-controlled demo into a robust, 12-state Finite-State Machine (FSM) controller that coordinates locomotion and manipulation in MuJoCo.

**Technical Case Study & Demo:** [atabak-eghbal.github.io/lucky-robots-g1-pick-place-case-study/](https://atabak-eghbal.github.io/lucky-robots-g1-pick-place-case-study/index.html)

## Project Overview

The task is to autonomously pick up a red cylinder from a source table and place it on a target blue table. This submission implements a modular, debuggable baseline that separates the high-level sequencing from the underlying ONNX control policies.

### Key Features
- **Autonomous Sequencing:** 12-state FSM handling approach, hover, grasp, transport, and release.
- **Simulation Stability:** Implements a 15-tick collision delay during release to prevent NaN instabilities common in kinematic-to-physical transitions.
- **Sweet-Spot Navigation:** Optimized approach waypoints to keep the manipulation target in the right arm's natural workspace.
- **Always-on Reacher:** Continuous arm control to preserve the walker's trained observation distribution.

## Repository Structure

```
.
├── run.py              # Main entry point (supports --policy fsm)
├── scene.xml           # MuJoCo scene definition
├── g1.xml              # G1 robot model
├── model_config.json   # Joint and PD configurations
├── common/             # Controller, Grasp, and ONNX wrappers
├── policies/           # FSM core logic and policy adapters
├── scripts/            # Integration tests and video recording
├── DEV_LOG.md          # Detailed engineering decision log
└── demo_video.mp4      # Full autonomous task demonstration
```

## Setup & Running

### Prerequisites
```bash
pip install mujoco onnxruntime numpy opencv-python
```

### Running the Autonomous Demo
To run the full autonomous pick-and-place sequence:
```bash
python run.py --policy fsm
```

### Running Integration Tests
To verify the task completion headlessly:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/test_fsm_approach.py
```

## Controls (Keyboard Mode)
To run the original manual mode for comparison:
```bash
python run.py --policy keyboard
```

| Key | Mode | Action |
|-----|------|--------|
| `.` | Both | Toggle WALK / REACH mode |
| Arrows | Walk | Move fwd/back, strafe L/R |
| `;` / `'` | Walk | Turn left/right |
| Up/Down | Reach | Arm forward/backward |
| Left/Right| Reach | Arm left/right |
| `,` | Both | Toggle grip (open/close) |
| Space | Both | Reset robot |

## Documentation
For a deep dive into the architecture, lessons learned, and failure analysis, please visit the [Technical Case Study](https://atabak-eghbal.github.io/lucky-robots-g1-pick-place-case-study/index.html).
