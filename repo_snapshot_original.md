# Repo Snapshot: g1-manipulation-challenge

## Repo Tree

```text
g1-manipulation-challenge
├── assets
│   ├── head_link.obj
│   ├── left_ankle_pitch_link.obj
│   ├── left_ankle_roll_link.obj
│   ├── left_elbow_link.obj
│   ├── left_hand_index_0_link.STL
│   ├── left_hand_index_1_link.STL
│   ├── left_hand_middle_0_link.STL
│   ├── left_hand_middle_1_link.STL
│   ├── left_hand_palm_link.STL
│   ├── left_hand_thumb_0_link.STL
│   ├── left_hand_thumb_1_link.STL
│   ├── left_hand_thumb_2_link.STL
│   ├── left_hip_pitch_link.obj
│   ├── left_hip_roll_link.obj
│   ├── left_hip_yaw_link.obj
│   ├── left_knee_link.obj
│   ├── left_shoulder_pitch_link.obj
│   ├── left_shoulder_roll_link.obj
│   ├── left_shoulder_yaw_link.obj
│   ├── left_wrist_pitch_link.obj
│   ├── left_wrist_roll_link.obj
│   ├── left_wrist_yaw_link.obj
│   ├── logo_link.obj
│   ├── pelvis.obj
│   ├── pelvis_contour_link.obj
│   ├── right_ankle_pitch_link.obj
│   ├── right_ankle_roll_link.obj
│   ├── right_elbow_link.obj
│   ├── right_hand_index_0_link.STL
│   ├── right_hand_index_1_link.STL
│   ├── right_hand_middle_0_link.STL
│   ├── right_hand_middle_1_link.STL
│   ├── right_hand_palm_link.STL
│   ├── right_hand_thumb_0_link.STL
│   ├── right_hand_thumb_1_link.STL
│   ├── right_hand_thumb_2_link.STL
│   ├── right_hip_pitch_link.obj
│   ├── right_hip_roll_link.obj
│   ├── right_hip_yaw_link.obj
│   ├── right_knee_link.obj
│   ├── right_shoulder_pitch_link.obj
│   ├── right_shoulder_roll_link.obj
│   ├── right_shoulder_yaw_link.obj
│   ├── right_wrist_pitch_link.obj
│   ├── right_wrist_roll_link.obj
│   ├── right_wrist_yaw_link.obj
│   ├── torso_link_rev_1_0.obj
│   ├── waist_roll_link_rev_1_0.obj
│   └── waist_yaw_link_rev_1_0.obj
├── .gitignore
├── croucher.onnx
├── croucher.onnx.data
├── g1.xml
├── model_config.json
├── README.md
├── repo_snapshot.md
├── right_reacher.onnx
├── right_reacher.onnx.data
├── rotator.onnx
├── rotator.onnx.data
├── run.py
├── scene.xml
├── walker.onnx
└── walker.onnx.data
```

## File Contents

---

## FILE: `.gitignore`

```
__pycache__/
*.pyc
```

---

## FILE: `README.md`

```md
# Lucky Robots Pick & Place Challenge

## Overview

We're inviting you to tackle an open-ended robotics manipulation problem. This isn't a test with a correct answer — it's an opportunity for us to understand how you approach novel challenges, decompose complex problems, and iterate toward solutions.

**The task: Get the G1 humanoid robot to pick up the red cylinder from the brown table and place it on the blue table.**

That's it. Everything else is up to you. Solve with VLM/VLA, IK logic, some custom policy, doesn't matter. 

## What's Included

This repo contains a fully working MuJoCo simulation of a Unitree G1 humanoid robot standing near two tables — a brown table with a red cylinder on it, and a white target table nearby. The robot comes with pre-trained locomotion (walker) and right-arm reaching (reacher) policies that you can use, modify, extend, or replace entirely. There's also a simple "grab" functionality that you can use if you like.

Two cameras are supplied in the scene. One mimics the existing head camera and another is attached to the hand.

```
picknplace_takehome/
├── run.py              # Main simulation script with keyboard control
├── scene.xml           # MuJoCo scene (brown table, blue table, red cylinder, cameras)
├── g1.xml              # G1 robot model (29 DoF, head + wrist cameras)
├── model_config.json   # Joint configuration, action scales, PD gains
├── walker.onnx         # Pre-trained walking policy (99D obs → 29D action)
├── croucher.onnx       # Pre-trained crouching policy
├── rotator.onnx        # Pre-trained rotation policy
├── right_reacher.onnx  # Pre-trained right arm reaching policy (36D obs → 7D action)
├── assets/             # G1 robot mesh files (.obj)
└── README.md           # This file
```

### Prerequisites

```bash
pip install mujoco onnxruntime numpy opencv-python
```

### Running the Demo

```bash
python run.py
```

This opens a MuJoCo viewer with the G1 robot and two camera windows (head cam, wrist cam). Controls:

| Key | Walk Mode | Reach Mode |
|-----|-----------|------------|
| `.` | Switch to Reach | Switch to Walk |
| Up/Down | Walk forward/back | Reach forward/back |
| Left/Right | Strafe left/right | Reach left/right |
| `;` / `'` | Turn left/right | Reach up/down |
| `\` or `/` | Stop | Reset reach target |
| `,` | Toggle right hand grip (open/close) | Toggle right hand grip (open/close) |
| Space | Reset robot | Reset robot |

The walk/reach toggle and keyboard control are there so you can get a feel for the robot and the scene. Your job is to automate it.

### What the Policies Do

- **Walker** (`walker.onnx`): Takes velocity commands (vx, vy, yaw_rate) and produces stable bipedal locomotion. 99D observation → 29D joint targets.
- **Right Reacher** (`right_reacher.onnx`): Takes a target position/orientation in pelvis frame and drives the right arm toward it. 36D observation → 7D arm joint targets. The reacher overlays on top of the walker — legs keep walking while the arm reaches.

The policies use PD control through MuJoCo's built-in position actuators. See `run.py` for the full observation/action pipeline.

### The Robot

The Unitree G1 has 29 actuated degrees of freedom:
- **Legs (12 DoF)**: 6 per leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
- **Waist (3 DoF)**: yaw, roll, pitch
- **Arms (14 DoF)**: 7 per arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)

Each hand has a **Unitree Dex3-1** 3-finger dexterous hand with 7 actuated joints: thumb (3 DoF), index finger (2 DoF), and middle finger (2 DoF) — 14 finger DoF total across both hands. The finger actuators use position control and are available via `data.ctrl`. See the actuator list in `g1.xml`.

### The Scene

- **Brown table** (source): 80cm x 50cm surface at 71.3cm height, in front of the robot
- **Blue table** (target): 70cm x 50cm surface at 61.3cm height, to the robot's right
- **Red cylinder**: 4cm diameter, 7cm tall, on a freejoint (fully dynamic). Starts on the brown table near the edge closest to the robot
- **Cameras**: `head_cam` (on torso, looking forward from head height), `wrist_cam` (on right palm, looking outward)

## What We Care About

We want to see how you solve the problem.

Specifically:

- **Problem decomposition.** How do you break this into tractable sub-problems? What do you identify as the core challenges (approach, grasp strategy, lift coordination) versus incidental ones?
- **Decision-making under uncertainty.** Why did you choose the tools, methods, and approaches you chose? What alternatives did you consider and reject?
- **Iteration and learning.** What did you try that didn't work? What did that teach you? How did your understanding of the problem evolve?
- **Technical communication.** Can you explain your reasoning clearly to someone who wasn't in your head while you were working?

## Constraints (or Lack Thereof)

- **Simulation**: MuJoCo is provided and set up, but you can port to another simulator if you have a strong reason to.
- **Policies**: You can use the provided walker/reacher as building blocks, fine-tune them, train new ones from scratch, or bypass learned policies entirely with analytical controllers. Your call.
- **Approach**: Reinforcement learning, trajectory optimization, motion planning, hand-tuned controllers, vision-based policies using the cameras, some hybrid — all valid. We're interested in *why* you chose what you chose.
- **Grasp strategy**: The Inspire hands have 3 fingers each — how you coordinate them for a reliable grasp is up to you. Pre-programmed grip, learned grasp policy, force-based closure, something else entirely — your call.
- **Code assistance**: We don't care if you use coding models to assist you, but you need to understand what was done and be able to explain it.

## Deliverables

1. **Code** sufficient for us to understand and reproduce what you did. Cleanliness matters less than clarity — comments explaining your thinking are more valuable than pristine architecture.
2. **A short video** showing your best result and progress, even if the cylinder never makes it to the blue table. Partial progress is real progress.
3. **A brief write-up** (PDF, Google Doc, or Markdown) covering:
   - Your approach and why you chose it
   - What worked, what didn't, and what you learned
   - What you'd do next with more time

## Evaluation

| What | Why it matters |
|------|---------------|
| Problem analysis | Did you identify the right challenges? How did you decompose the problem? |
| Methodological reasoning | Can you justify your choices? |
| Adaptability | How did you respond when things didn't work? |
| Technical depth | Do you understand the tools you're using? |
| Communication | Can you explain complex ideas clearly? |

## Some Questions to Consider

You don't need to answer all of these, but they might help structure your thinking:

- How do you decompose this into phases (approach, grasp, transport, place)? Can they overlap?
- How do you coordinate locomotion and manipulation? Does the robot need to stop walking to reach?
- The provided reacher targets a point in pelvis frame — how do you decide *what* to target and *when*?
- How would you use the wrist camera? The head camera? Do you need them?
- How do you verify the cylinder is actually grasped before trying to move? How do you verify it's placed?
- Where does sim-to-real transfer factor into your thinking (even if you're not doing it)?

## Submission

Send us:
- Your write-up (PDF, Google Doc, or Markdown)
- A link to a repo or zip of your code
- Your video (or a link to it)

Email to: **harrison@luckyrobots.com**, cc: **nur@luckyrobots.com**

Questions before and during are welcome.
```

---

## FILE: `assets/head_link.obj`

_Skipped: file is too large (7767912 bytes)._ 

---

## FILE: `assets/left_ankle_pitch_link.obj`

_Skipped: file is too large (613450 bytes)._ 

---

## FILE: `assets/left_ankle_roll_link.obj`

_Skipped: file is too large (5558051 bytes)._ 

---

## FILE: `assets/left_elbow_link.obj`

_Skipped: file is too large (745851 bytes)._ 

---

## FILE: `assets/left_hand_index_0_link.STL`

_Skipped: file is too large (475984 bytes)._ 

---

## FILE: `assets/left_hand_index_1_link.STL`

_Skipped: file is too large (1521784 bytes)._ 

---

## FILE: `assets/left_hand_middle_0_link.STL`

_Skipped: file is too large (475984 bytes)._ 

---

## FILE: `assets/left_hand_middle_1_link.STL`

_Skipped: file is too large (1521784 bytes)._ 

---

## FILE: `assets/left_hand_palm_link.STL`

_Skipped: file is too large (2140184 bytes)._ 

---

## FILE: `assets/left_hand_thumb_0_link.STL`

_Skipped: non-text or binary file._

---

## FILE: `assets/left_hand_thumb_1_link.STL`

_Skipped: file is too large (475984 bytes)._ 

---

## FILE: `assets/left_hand_thumb_2_link.STL`

_Skipped: file is too large (1521784 bytes)._ 

---

## FILE: `assets/left_hip_pitch_link.obj`

_Skipped: file is too large (1518316 bytes)._ 

---

## FILE: `assets/left_hip_roll_link.obj`

_Skipped: file is too large (1624506 bytes)._ 

---

## FILE: `assets/left_hip_yaw_link.obj`

_Skipped: file is too large (2515575 bytes)._ 

---

## FILE: `assets/left_knee_link.obj`

_Skipped: file is too large (7267119 bytes)._ 

---

## FILE: `assets/left_shoulder_pitch_link.obj`

_Skipped: file is too large (1486668 bytes)._ 

---

## FILE: `assets/left_shoulder_roll_link.obj`

_Skipped: file is too large (3384092 bytes)._ 

---

## FILE: `assets/left_shoulder_yaw_link.obj`

_Skipped: file is too large (2121472 bytes)._ 

---

## FILE: `assets/left_wrist_pitch_link.obj`

_Skipped: file is too large (695437 bytes)._ 

---

## FILE: `assets/left_wrist_roll_link.obj`

_Skipped: file is too large (2946517 bytes)._ 

---

## FILE: `assets/left_wrist_yaw_link.obj`

_Skipped: file is too large (2632279 bytes)._ 

---

## FILE: `assets/logo_link.obj`

_Skipped: file is too large (2016446 bytes)._ 

---

## FILE: `assets/pelvis.obj`

_Skipped: file is too large (8662517 bytes)._ 

---

## FILE: `assets/pelvis_contour_link.obj`

_Skipped: file is too large (15251435 bytes)._ 

---

## FILE: `assets/right_ankle_pitch_link.obj`

_Skipped: file is too large (614089 bytes)._ 

---

## FILE: `assets/right_ankle_roll_link.obj`

_Skipped: file is too large (5562121 bytes)._ 

---

## FILE: `assets/right_elbow_link.obj`

_Skipped: file is too large (749121 bytes)._ 

---

## FILE: `assets/right_hand_index_0_link.STL`

_Skipped: file is too large (475984 bytes)._ 

---

## FILE: `assets/right_hand_index_1_link.STL`

_Skipped: file is too large (1521784 bytes)._ 

---

## FILE: `assets/right_hand_middle_0_link.STL`

_Skipped: file is too large (475984 bytes)._ 

---

## FILE: `assets/right_hand_middle_1_link.STL`

_Skipped: file is too large (1521784 bytes)._ 

---

## FILE: `assets/right_hand_palm_link.STL`

_Skipped: file is too large (2140184 bytes)._ 

---

## FILE: `assets/right_hand_thumb_0_link.STL`

_Skipped: non-text or binary file._

---

## FILE: `assets/right_hand_thumb_1_link.STL`

_Skipped: file is too large (475984 bytes)._ 

---

## FILE: `assets/right_hand_thumb_2_link.STL`

_Skipped: file is too large (1521784 bytes)._ 

---

## FILE: `assets/right_hip_pitch_link.obj`

_Skipped: file is too large (1527108 bytes)._ 

---

## FILE: `assets/right_hip_roll_link.obj`

_Skipped: file is too large (1628958 bytes)._ 

---

## FILE: `assets/right_hip_yaw_link.obj`

_Skipped: file is too large (2522611 bytes)._ 

---

## FILE: `assets/right_knee_link.obj`

_Skipped: file is too large (7248935 bytes)._ 

---

## FILE: `assets/right_shoulder_pitch_link.obj`

_Skipped: file is too large (1496372 bytes)._ 

---

## FILE: `assets/right_shoulder_roll_link.obj`

_Skipped: file is too large (3403337 bytes)._ 

---

## FILE: `assets/right_shoulder_yaw_link.obj`

_Skipped: file is too large (2118467 bytes)._ 

---

## FILE: `assets/right_wrist_pitch_link.obj`

_Skipped: file is too large (663484 bytes)._ 

---

## FILE: `assets/right_wrist_roll_link.obj`

_Skipped: file is too large (2961073 bytes)._ 

---

## FILE: `assets/right_wrist_yaw_link.obj`

_Skipped: file is too large (2873722 bytes)._ 

---

## FILE: `assets/torso_link_rev_1_0.obj`

_Skipped: file is too large (21824558 bytes)._ 

---

## FILE: `assets/waist_roll_link_rev_1_0.obj`

_Skipped: file is too large (731523 bytes)._ 

---

## FILE: `assets/waist_yaw_link_rev_1_0.obj`

_Skipped: file is too large (5358020 bytes)._ 

---

## FILE: `croucher.onnx`

_Skipped: non-text or binary file._

---

## FILE: `croucher.onnx.data`

_Skipped: file is too large (881448 bytes)._ 

---

## FILE: `g1.xml`

```xml
<mujoco model="g1_29dof_rev_1_0">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option integrator="implicitfast" timestep="0.005" impratio="1.0" cone="pyramidal" 
          jacobian="auto" solver="Newton" iterations="10" tolerance="1e-08" 
          ls_iterations="20" ls_tolerance="0.01" gravity="0 0 -9.81"/>

  <default>
    <default class="g1">
      <position inheritrange="1"/>
      <default class="visual">
        <geom group="2" type="mesh" density="0" material="silver" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1"/>
        <default class="foot_capsule">
          <geom type="capsule" size="0.01"/>
        </default>
      </default>
      <site group="5" rgba="1 0 0 1"/>
    </default>
  </default>

  <asset>
    <material name="silver" rgba="0.7 0.7 0.7 1"/>
    <material name="black" rgba="0.2 0.2 0.2 1"/>

    <mesh name="pelvis" file="pelvis.obj"/>
    <mesh name="pelvis_contour_link" file="pelvis_contour_link.obj"/>
    <mesh name="left_hip_pitch_link" file="left_hip_pitch_link.obj"/>
    <mesh name="left_hip_roll_link" file="left_hip_roll_link.obj"/>
    <mesh name="left_hip_yaw_link" file="left_hip_yaw_link.obj"/>
    <mesh name="left_knee_link" file="left_knee_link.obj"/>
    <mesh name="left_ankle_pitch_link" file="left_ankle_pitch_link.obj"/>
    <mesh name="left_ankle_roll_link" file="left_ankle_roll_link.obj"/>
    <mesh name="right_hip_pitch_link" file="right_hip_pitch_link.obj"/>
    <mesh name="right_hip_roll_link" file="right_hip_roll_link.obj"/>
    <mesh name="right_hip_yaw_link" file="right_hip_yaw_link.obj"/>
    <mesh name="right_knee_link" file="right_knee_link.obj"/>
    <mesh name="right_ankle_pitch_link" file="right_ankle_pitch_link.obj"/>
    <mesh name="right_ankle_roll_link" file="right_ankle_roll_link.obj"/>
    <mesh name="waist_yaw_link_rev_1_0" file="waist_yaw_link_rev_1_0.obj"/>
    <mesh name="waist_roll_link_rev_1_0" file="waist_roll_link_rev_1_0.obj"/>
    <mesh name="torso_link_rev_1_0" file="torso_link_rev_1_0.obj"/>
    <mesh name="logo_link" file="logo_link.obj"/>
    <mesh name="head_link" file="head_link.obj"/>
    <mesh name="left_shoulder_pitch_link" file="left_shoulder_pitch_link.obj"/>
    <mesh name="left_shoulder_roll_link" file="left_shoulder_roll_link.obj"/>
    <mesh name="left_shoulder_yaw_link" file="left_shoulder_yaw_link.obj"/>
    <mesh name="left_elbow_link" file="left_elbow_link.obj"/>
    <mesh name="left_wrist_roll_link" file="left_wrist_roll_link.obj"/>
    <mesh name="left_wrist_pitch_link" file="left_wrist_pitch_link.obj"/>
    <mesh name="left_wrist_yaw_link" file="left_wrist_yaw_link.obj"/>
    <mesh name="left_hand_palm_link" file="left_hand_palm_link.STL"/>
    <mesh name="left_hand_thumb_0_link" file="left_hand_thumb_0_link.STL"/>
    <mesh name="left_hand_thumb_1_link" file="left_hand_thumb_1_link.STL"/>
    <mesh name="left_hand_thumb_2_link" file="left_hand_thumb_2_link.STL"/>
    <mesh name="left_hand_middle_0_link" file="left_hand_middle_0_link.STL"/>
    <mesh name="left_hand_middle_1_link" file="left_hand_middle_1_link.STL"/>
    <mesh name="left_hand_index_0_link" file="left_hand_index_0_link.STL"/>
    <mesh name="left_hand_index_1_link" file="left_hand_index_1_link.STL"/>
    <mesh name="right_shoulder_pitch_link" file="right_shoulder_pitch_link.obj"/>
    <mesh name="right_shoulder_roll_link" file="right_shoulder_roll_link.obj"/>
    <mesh name="right_shoulder_yaw_link" file="right_shoulder_yaw_link.obj"/>
    <mesh name="right_elbow_link" file="right_elbow_link.obj"/>
    <mesh name="right_wrist_roll_link" file="right_wrist_roll_link.obj"/>
    <mesh name="right_wrist_pitch_link" file="right_wrist_pitch_link.obj"/>
    <mesh name="right_wrist_yaw_link" file="right_wrist_yaw_link.obj"/>
    <mesh name="right_hand_palm_link" file="right_hand_palm_link.STL"/>
    <mesh name="right_hand_thumb_0_link" file="right_hand_thumb_0_link.STL"/>
    <mesh name="right_hand_thumb_1_link" file="right_hand_thumb_1_link.STL"/>
    <mesh name="right_hand_thumb_2_link" file="right_hand_thumb_2_link.STL"/>
    <mesh name="right_hand_middle_0_link" file="right_hand_middle_0_link.STL"/>
    <mesh name="right_hand_middle_1_link" file="right_hand_middle_1_link.STL"/>
    <mesh name="right_hand_index_0_link" file="right_hand_index_0_link.STL"/>
    <mesh name="right_hand_index_1_link" file="right_hand_index_1_link.STL"/>
  </asset>


<worldbody>
    <body name="pelvis" pos="-0.6 0 0.79" childclass="g1">
      <light pos="0 0 2" mode="trackcom"/>
      <camera name="tracking" pos="1.734 -1.135 .35" xyaxes="0.552 0.834 -0.000 -0.170 0.112 0.979" mode="trackcom"/>
      <inertial pos="0 0 -0.07605" quat="1 0 -0.000399148 0" mass="3.813" diaginertia="0.010549 0.0093089 0.0079184"/>
      <freejoint name="floating_base_joint"/>
      <geom class="visual" material="black" mesh="pelvis"/>
      <geom class="visual" mesh="pelvis_contour_link"/>
      <geom name="pelvis_collision" class="collision" type="sphere" size="0.07" pos="0 0 -0.08"/>
      <site name="imu_in_pelvis" size="0.01" pos="0.04525 0 -0.08339"/>
      <body name="left_hip_pitch_link" pos="0 0.064452 -0.1027">
        <inertial pos="0.002741 0.047791 -0.02606" quat="0.954862 0.293964 0.0302556 0.030122" mass="1.35"
          diaginertia="0.00181517 0.00153422 0.00116212"/>
        <joint name="left_hip_pitch_joint" axis="0 1 0" range="-2.5307 2.8798" armature="0.01018" actuatorfrcrange="-88 88"/>
        <geom class="visual" material="black" mesh="left_hip_pitch_link"/>
        <body name="left_hip_roll_link" pos="0 0.052 -0.030465" quat="0.996179 0 -0.0873386 0">
          <inertial pos="0.029812 -0.001045 -0.087934" quat="0.977808 -1.97119e-05 0.205576 -0.0403793" mass="1.52"
            diaginertia="0.00254986 0.00241169 0.00148755"/>
          <joint name="left_hip_roll_joint" axis="1 0 0" range="-0.5236 2.9671" armature="0.02510" actuatorfrcrange="-139 139"/>
          <geom class="visual" mesh="left_hip_roll_link"/>
          <geom name="left_hip_collision" class="collision" size="0.06" fromto="0.02 0 0 0.02 0 -0.08"/>
          <body name="left_hip_yaw_link" pos="0.025001 0 -0.12412">
            <inertial pos="-0.057709 -0.010981 -0.15078" quat="0.600598 0.15832 0.223482 0.751181" mass="1.702"
              diaginertia="0.00776166 0.00717575 0.00160139"/>
            <joint name="left_hip_yaw_joint" axis="0 0 1" range="-2.7576 2.7576" armature="0.01018" actuatorfrcrange="-88 88"/>
            <geom class="visual" mesh="left_hip_yaw_link"/>
            <geom name="left_thigh_collision" class="collision" size="0.055" fromto="-0.0 0 -0.03 -0.06 0 -0.17"/>
            <body name="left_knee_link" pos="-0.078273 0.0021489 -0.17734" quat="0.996179 0 0.0873386 0">
              <inertial pos="0.005457 0.003964 -0.12074" quat="0.923418 -0.0327699 0.0158246 0.382067" mass="1.932"
                diaginertia="0.0113804 0.0112778 0.00146458"/>
              <joint name="left_knee_joint" axis="0 1 0" range="-0.087267 2.8798" armature="0.02510" actuatorfrcrange="-139 139"/>
              <geom class="visual" mesh="left_knee_link"/>
              <geom name="left_shin_collision" class="collision" size="0.045" fromto="0.01 0 0 0.01 0 -0.15"/>
              <geom name="left_linkage_brace_collision" class="collision" size="0.03" fromto="0.01 0 -0.2 0.01 0 -0.28"/>
              <body name="left_ankle_pitch_link" pos="0 -9.4445e-05 -0.30001">
                <inertial pos="-0.007269 0 0.011137" quat="0.603053 0.369225 0.369225 0.603053" mass="0.074"
                  diaginertia="1.89e-05 1.40805e-05 6.9195e-06"/>
                <joint name="left_ankle_pitch_joint" axis="0 1 0" range="-0.87267 0.5236" armature="0.00722" actuatorfrcrange="-50 50"/>
                <geom class="visual" mesh="left_ankle_pitch_link"/>
                <body name="left_ankle_roll_link" pos="0 0 -0.017558">
                  <site name="left_foot" rgba="1 0 0 1" pos="0.04 0 -0.037"/>
                  <inertial pos="0.026505 0 -0.016425" quat="-0.000481092 0.728482 -0.000618967 0.685065" mass="0.608"
                    diaginertia="0.00167218 0.0016161 0.000217621"/>
                  <joint name="left_ankle_roll_joint" axis="1 0 0" range="-0.2618 0.2618" armature="0.00722" actuatorfrcrange="-50 50"/>
                  <geom class="visual" material="black" mesh="left_ankle_roll_link"/>
                  <geom name="left_foot1_collision" class="foot_capsule" fromto="0.1 -0.026 -0.025 0.05 -0.027 -0.025"/>
                  <geom name="left_foot2_collision" class="foot_capsule"
                    fromto="-0.044 -0.018 -0.025 0.123 -0.018 -0.025"/>
                  <geom name="left_foot3_collision" class="foot_capsule" fromto="-0.052 -0.01 -0.025 0.13 -0.01 -0.025"/>
                  <geom name="left_foot4_collision" class="foot_capsule" fromto="-0.054 0 -0.025 0.132 0 -0.025"/>
                  <geom name="left_foot5_collision" class="foot_capsule" fromto="-0.052 0.01 -0.025 0.13 0.01 -0.025"/>
                  <geom name="left_foot6_collision" class="foot_capsule" fromto="-0.044 0.018 -0.025 0.123 0.018 -0.025"/>
                  <geom name="left_foot7_collision" class="foot_capsule" fromto="0.1 0.026 -0.025 0.05 0.026 -0.025"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_hip_pitch_link" pos="0 -0.064452 -0.1027">
        <inertial pos="0.002741 -0.047791 -0.02606" quat="0.954862 -0.293964 0.0302556 -0.030122" mass="1.35"
          diaginertia="0.00181517 0.00153422 0.00116212"/>
        <joint name="right_hip_pitch_joint" axis="0 1 0" range="-2.5307 2.8798" armature="0.01018" actuatorfrcrange="-88 88"/>
        <geom class="visual" material="black" mesh="right_hip_pitch_link"/>
        <body name="right_hip_roll_link" pos="0 -0.052 -0.030465" quat="0.996179 0 -0.0873386 0">
          <inertial pos="0.029812 0.001045 -0.087934" quat="0.977808 1.97119e-05 0.205576 0.0403793" mass="1.52"
            diaginertia="0.00254986 0.00241169 0.00148755"/>
          <joint name="right_hip_roll_joint" axis="1 0 0" range="-2.9671 0.5236" armature="0.02510" actuatorfrcrange="-139 139"/>
          <geom class="visual" mesh="right_hip_roll_link"/>
          <geom name="right_hip_collision" class="collision" size="0.06" fromto="0.02 0 0 0.02 0 -0.08"/>
          <body name="right_hip_yaw_link" pos="0.025001 0 -0.12412">
            <inertial pos="-0.057709 0.010981 -0.15078" quat="0.751181 0.223482 0.15832 0.600598" mass="1.702"
              diaginertia="0.00776166 0.00717575 0.00160139"/>
            <joint name="right_hip_yaw_joint" axis="0 0 1" range="-2.7576 2.7576" armature="0.01018" actuatorfrcrange="-88 88"/>
            <geom class="visual" mesh="right_hip_yaw_link"/>
            <geom name="right_thigh_collision" class="collision" size="0.055" fromto="-0.0 0 -0.03 -0.06 0 -0.17"/>
            <body name="right_knee_link" pos="-0.078273 -0.0021489 -0.17734" quat="0.996179 0 0.0873386 0">
              <inertial pos="0.005457 -0.003964 -0.12074" quat="0.923439 0.0345276 0.0116333 -0.382012" mass="1.932"
                diaginertia="0.011374 0.0112843 0.00146452"/>
              <joint name="right_knee_joint" axis="0 1 0" range="-0.087267 2.8798" armature="0.02510" actuatorfrcrange="-139 139"/>
              <geom class="visual" mesh="right_knee_link"/>
              <geom name="right_shin_collision" class="collision" size="0.045" fromto="0.01 0 0 0.01 0 -0.15"/>
              <geom name="right_linkage_brace_collision" class="collision" size="0.03" fromto="0.01 0 -0.2 0.01 0 -0.28"/>
              <body name="right_ankle_pitch_link" pos="0 9.4445e-05 -0.30001">
                <inertial pos="-0.007269 0 0.011137" quat="0.603053 0.369225 0.369225 0.603053" mass="0.074"
                  diaginertia="1.89e-05 1.40805e-05 6.9195e-06"/>
                <joint name="right_ankle_pitch_joint" axis="0 1 0" range="-0.87267 0.5236" armature="0.00722" actuatorfrcrange="-50 50"/>
                <geom class="visual" mesh="right_ankle_pitch_link"/>
                <body name="right_ankle_roll_link" pos="0 0 -0.017558">
                  <site name="right_foot" rgba="1 0 0 1" pos="0.04 0 -0.037"/>
                  <inertial pos="0.026505 0 -0.016425" quat="0.000481092 0.728482 0.000618967 0.685065" mass="0.608"
                    diaginertia="0.00167218 0.0016161 0.000217621"/>
                  <joint name="right_ankle_roll_joint" axis="1 0 0" range="-0.2618 0.2618" armature="0.00722" actuatorfrcrange="-50 50"/>
                  <geom class="visual" material="black" mesh="right_ankle_roll_link"/>
                  <geom name="right_foot1_collision" class="foot_capsule" fromto="0.1 -0.026 -0.025 0.05 -0.026 -0.025"/>
                  <geom name="right_foot2_collision" class="foot_capsule"
                    fromto="-0.044 -0.018 -0.025 0.123 -0.018 -0.025"/>
                  <geom name="right_foot3_collision" class="foot_capsule" fromto="-0.052 -0.01 -0.025 0.13 -0.01 -0.025"/>
                  <geom name="right_foot4_collision" class="foot_capsule" fromto="-0.054 0 -0.025 0.132 0 -0.025"/>
                  <geom name="right_foot5_collision" class="foot_capsule" fromto="-0.052 0.01 -0.025 0.13 0.01 -0.025"/>
                  <geom name="right_foot6_collision" class="foot_capsule"
                    fromto="-0.044 0.018 -0.025 0.123 0.018 -0.025"/>
                  <geom name="right_foot7_collision" class="foot_capsule" fromto="0.1 0.026 -0.025 0.05 0.026 -0.025"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="waist_yaw_link_rev_1_0">
        <inertial pos="0.003494 0.000233 0.018034" quat="0.289697 0.591001 -0.337795 0.672821" mass="0.214"
          diaginertia="0.000163531 0.000107714 0.000102205"/>
        <joint name="waist_yaw_joint" axis="0 0 1" range="-2.618 2.618" armature="0.01018" actuatorfrcrange="-88 88"/>
        <geom class="visual" mesh="waist_yaw_link_rev_1_0"/>
        <body name="waist_roll_link_rev_1_0" pos="-0.0039635 0 0.044">
          <inertial pos="0 2.3e-05 0" quat="0.5 0.5 -0.5 0.5" mass="0.086" diaginertia="8.245e-06 7.079e-06 6.339e-06"/>
          <joint name="waist_roll_joint" axis="1 0 0" range="-0.52 0.52" armature="0.00722" actuatorfrcrange="-50 50"/>
          <geom class="visual" mesh="waist_roll_link_rev_1_0"/>
          <body name="torso_link_rev_1_0">
            <inertial pos="0.00203158 0.000339683 0.184568" quat="0.999803 -6.03319e-05 0.0198256 0.00131986"
              mass="7.818" diaginertia="0.121847 0.109825 0.0273735"/>
            <joint name="waist_pitch_joint" axis="0 1 0" range="-0.52 0.52" armature="0.00722" actuatorfrcrange="-50 50"/>
            <geom class="visual" mesh="torso_link_rev_1_0"/>
            <geom class="visual" pos="0.0039635 0 -0.044" quat="1 0 0 0" material="black" mesh="logo_link"/>
            <geom class="visual" pos="0.0039635 0 -0.044" material="black" mesh="head_link"/>
            <geom name="torso_collision" class="collision" size="0.09" fromto="0.01 0 0.08 0.01 0 0.2"/>
            <geom name="head_collision" class="collision" type="sphere" size="0.06" pos="0 0 .43"/>
            <camera name="head_cam" pos="0.05 0 0.43" xyaxes="0 -1 0 0.34 0 0.94" fovy="60"/>
            <site name="imu_in_torso" size="0.01" pos="-0.03959 -0.00224 0.14792"/>
            <body name="left_shoulder_pitch_link" pos="0.0039563 0.10022 0.24778"
              quat="0.990264 0.139201 1.38722e-05 -9.86868e-05">
              <inertial pos="0 0.035892 -0.011628" quat="0.654152 0.0130458 -0.326267 0.68225" mass="0.718"
                diaginertia="0.000465864 0.000432842 0.000406394"/>
              <joint name="left_shoulder_pitch_joint" axis="0 1 0" range="-3.0892 2.6704" armature="0.00361" actuatorfrcrange="-25 25"/>
              <geom class="visual" mesh="left_shoulder_pitch_link"/>
              <body name="left_shoulder_roll_link" pos="0 0.038 -0.013831" quat="0.990268 -0.139172 0 0">
                <inertial pos="-0.000227 0.00727 -0.063243" quat="0.701256 -0.0196223 -0.00710317 0.712604" mass="0.643"
                  diaginertia="0.000691311 0.000618011 0.000388977"/>
                <joint name="left_shoulder_roll_joint" axis="1 0 0" range="-1.5882 2.2515" armature="0.00361" actuatorfrcrange="-25 25"/>
                <geom class="visual" mesh="left_shoulder_roll_link"/>
                <body name="left_shoulder_yaw_link" pos="0 0.00624 -0.1032">
                  <inertial pos="0.010773 -0.002949 -0.072009" quat="0.716879 -0.0964829 -0.0679942 0.687134"
                    mass="0.734" diaginertia="0.00106187 0.00103217 0.000400661"/>
                  <joint name="left_shoulder_yaw_joint" axis="0 0 1" range="-2.618 2.618" armature="0.00361" actuatorfrcrange="-25 25"/>
                  <geom class="visual" mesh="left_shoulder_yaw_link"/>
                  <geom name="left_shoulder_yaw_collision" class="collision" size="0.035" fromto="0 0 -0.08 0 0 0.05"/>
                  <body name="left_elbow_link" pos="0.015783 0 -0.080518">
                    <inertial pos="0.064956 0.004454 -0.010062" quat="0.541765 0.636132 0.388821 0.388129" mass="0.6"
                      diaginertia="0.000443035 0.000421612 0.000259353"/>
                    <joint name="left_elbow_joint" axis="0 1 0" range="-1.0472 2.0944" armature="0.00361" actuatorfrcrange="-25 25"/>
                    <geom class="visual" mesh="left_elbow_link"/>
                    <geom name="left_elbow_yaw_collision" class="collision" size="0.035"
                      fromto="-0.01 0 -0.01 0.08 0 -0.01"/>
                    <body name="left_wrist_roll_link" pos="0.1 0.00188791 -0.01">
                      <inertial pos="0.0171394 0.000537591 4.8864e-07" quat="0.575338 0.411667 -0.574906 0.411094"
                        mass="0.085445" diaginertia="5.48211e-05 4.96646e-05 3.57798e-05"/>
                      <joint name="left_wrist_roll_joint" axis="1 0 0" range="-1.97222 1.97222" armature="0.00361" actuatorfrcrange="-25 25"/>
                      <geom class="visual" mesh="left_wrist_roll_link"/>
                      <body name="left_wrist_pitch_link" pos="0.038 0 0">
                        <inertial pos="0.0229999 -0.00111685 -0.00111658" quat="0.249998 0.661363 0.293036 0.643608"
                          mass="0.48405" diaginertia="0.000430353 0.000429873 0.000164648"/>
                        <joint name="left_wrist_pitch_joint" axis="0 1 0" range="-1.61443 1.61443" armature="0.00425" actuatorfrcrange="-5 5"/>
                        <geom class="visual" mesh="left_wrist_pitch_link"/>
                        <geom name="left_wrist_collision" class="collision" size="0.035" fromto="-0.01 0 0 0.06 0 0"/>
                        <body name="left_wrist_yaw_link" pos="0.046 0 0">
                          <inertial pos="0.0708244 0.000191745 0.00161742" quat="0.510571 0.526295 0.468078 0.493188"
                            mass="0.254576" diaginertia="0.000646113 0.000559993 0.000147566"/>
                          <joint name="left_wrist_yaw_joint" axis="0 0 1" range="-1.61443 1.61443" armature="0.00425" actuatorfrcrange="-5 5"/>
                          <geom class="visual" mesh="left_wrist_yaw_link"/>
                          <geom pos="0.0415 0.003 0" quat="1 0 0 0" type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_palm_link"/>
                          <geom pos="0.0415 0.003 0" quat="1 0 0 0" type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_palm_link"/>
                          <site name="left_palm" pos="0.08 0 0" size="0.01"/>
                          <body name="left_hand_thumb_0_link" pos="0.067 0.003 0">
                            <inertial pos="-0.000884246 -0.00863407 0.000944293" quat="0.462991 0.643965 -0.460173 0.398986" mass="0.0862366" diaginertia="1.6546e-05 1.60058e-05 1.43741e-05"/>
                            <joint name="left_hand_thumb_0_joint" axis="0 1 0" range="-1.0472 1.0472" actuatorfrcrange="-2.45 2.45"/>
                            <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_thumb_0_link"/>
                            <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_thumb_0_link"/>
                            <body name="left_hand_thumb_1_link" pos="-0.0025 -0.0193 0">
                              <inertial pos="-0.000827888 -0.0354744 -0.0003809" quat="0.685598 0.705471 -0.15207 0.0956069" mass="0.0588507" diaginertia="1.28514e-05 1.22902e-05 5.9666e-06"/>
                              <joint name="left_hand_thumb_1_joint" axis="0 0 1" range="-0.724312 1.0472" actuatorfrcrange="-1.4 1.4"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_thumb_1_link"/>
                              <geom size="0.01 0.015 0.01" pos="-0.001 -0.032 0" type="box" rgba="0.7 0.7 0.7 1"/>
                              <body name="left_hand_thumb_2_link" pos="0 -0.0458 0">
                                <inertial pos="-0.00171735 -0.0262819 0.000107789" quat="0.703174 0.710977 -0.00017564 -0.00766553" mass="0.0203063" diaginertia="4.61314e-06 3.86645e-06 1.53495e-06"/>
                                <joint name="left_hand_thumb_2_joint" axis="0 0 1" range="0 1.74533" actuatorfrcrange="-1.4 1.4"/>
                                <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_thumb_2_link"/>
                                <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_thumb_2_link"/>
                              </body>
                            </body>
                          </body>
                          <body name="left_hand_middle_0_link" pos="0.1192 0.0046 -0.0285">
                            <inertial pos="0.0354744 0.000827888 0.0003809" quat="0.391313 0.552395 0.417187 0.606373" mass="0.0588507" diaginertia="1.28514e-05 1.22902e-05 5.9666e-06"/>
                            <joint name="left_hand_middle_0_joint" axis="0 0 1" range="-1.5708 0" actuatorfrcrange="-1.4 1.4"/>
                            <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_middle_0_link"/>
                            <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_middle_0_link"/>
                            <body name="left_hand_middle_1_link" pos="0.0458 0 0">
                              <inertial pos="0.0262819 0.00171735 -0.000107789" quat="0.502612 0.491799 0.502639 0.502861" mass="0.0203063" diaginertia="4.61314e-06 3.86645e-06 1.53495e-06"/>
                              <joint name="left_hand_middle_1_joint" axis="0 0 1" range="-1.74533 0" actuatorfrcrange="-1.4 1.4"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_middle_1_link"/>
                              <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_middle_1_link"/>
                            </body>
                          </body>
                          <body name="left_hand_index_0_link" pos="0.1192 0.0046 0.0285">
                            <inertial pos="0.0354744 0.000827888 0.0003809" quat="0.391313 0.552395 0.417187 0.606373" mass="0.0588507" diaginertia="1.28514e-05 1.22902e-05 5.9666e-06"/>
                            <joint name="left_hand_index_0_joint" axis="0 0 1" range="-1.5708 0" actuatorfrcrange="-1.4 1.4"/>
                            <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_index_0_link"/>
                            <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_index_0_link"/>
                            <body name="left_hand_index_1_link" pos="0.0458 0 0">
                              <inertial pos="0.0262819 0.00171735 -0.000107789" quat="0.502612 0.491799 0.502639 0.502861" mass="0.0203063" diaginertia="4.61314e-06 3.86645e-06 1.53495e-06"/>
                              <joint name="left_hand_index_1_joint" axis="0 0 1" range="-1.74533 0" actuatorfrcrange="-1.4 1.4"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="left_hand_index_1_link"/>
                              <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="left_hand_index_1_link"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
            <body name="right_shoulder_pitch_link" pos="0.0039563 -0.10021 0.24778"
              quat="0.990264 -0.139201 1.38722e-05 9.86868e-05">
              <inertial pos="0 -0.035892 -0.011628" quat="0.68225 -0.326267 0.0130458 0.654152" mass="0.718"
                diaginertia="0.000465864 0.000432842 0.000406394"/>
              <joint name="right_shoulder_pitch_joint" axis="0 1 0" range="-3.0892 2.6704" armature="0.00361" actuatorfrcrange="-25 25"/>
              <geom class="visual" mesh="right_shoulder_pitch_link"/>
              <body name="right_shoulder_roll_link" pos="0 -0.038 -0.013831" quat="0.990268 0.139172 0 0">
                <inertial pos="-0.000227 -0.00727 -0.063243" quat="0.712604 -0.00710317 -0.0196223 0.701256"
                  mass="0.643" diaginertia="0.000691311 0.000618011 0.000388977"/>
                <joint name="right_shoulder_roll_joint" axis="1 0 0" range="-2.2515 1.5882" armature="0.00361" actuatorfrcrange="-25 25"/>
                <geom class="visual" mesh="right_shoulder_roll_link"/>
                <body name="right_shoulder_yaw_link" pos="0 -0.00624 -0.1032">
                  <inertial pos="0.010773 0.002949 -0.072009" quat="0.687134 -0.0679942 -0.0964829 0.716879"
                    mass="0.734" diaginertia="0.00106187 0.00103217 0.000400661"/>
                  <joint name="right_shoulder_yaw_joint" axis="0 0 1" range="-2.618 2.618" armature="0.00361" actuatorfrcrange="-25 25"/>
                  <geom class="visual" mesh="right_shoulder_yaw_link"/>
                  <geom name="right_shoulder_yaw_collision" class="collision" size="0.035" fromto="0 0 -0.08 0 0 0.05"/>
                  <body name="right_elbow_link" pos="0.015783 0 -0.080518">
                    <inertial pos="0.064956 -0.004454 -0.010062" quat="0.388129 0.388821 0.636132 0.541765" mass="0.6"
                      diaginertia="0.000443035 0.000421612 0.000259353"/>
                    <joint name="right_elbow_joint" axis="0 1 0" range="-1.0472 2.0944" armature="0.00361" actuatorfrcrange="-25 25"/>
                    <geom class="visual" mesh="right_elbow_link"/>
                    <geom name="right_elbow_yaw_collision" class="collision" size="0.035"
                      fromto="-0.01 0 -0.01 0.08 0 -0.01"/>
                    <body name="right_wrist_roll_link" pos="0.1 -0.00188791 -0.01">
                      <inertial pos="0.0171394 -0.000537591 4.8864e-07" quat="0.411667 0.575338 -0.411094 0.574906"
                        mass="0.085445" diaginertia="5.48211e-05 4.96646e-05 3.57798e-05"/>
                      <joint name="right_wrist_roll_joint" axis="1 0 0" range="-1.97222 1.97222" armature="0.00361" actuatorfrcrange="-25 25"/>
                      <geom class="visual" mesh="right_wrist_roll_link"/>
                      <body name="right_wrist_pitch_link" pos="0.038 0 0">
                        <inertial pos="0.0229999 0.00111685 -0.00111658" quat="0.643608 0.293036 0.661363 0.249998"
                          mass="0.48405" diaginertia="0.000430353 0.000429873 0.000164648"/>
                        <joint name="right_wrist_pitch_joint" axis="0 1 0" range="-1.61443 1.61443" armature="0.00425" actuatorfrcrange="-5 5"/>
                        <geom class="visual" mesh="right_wrist_pitch_link"/>
                        <geom name="right_wrist_collision" class="collision" size="0.035" fromto="-0.01 0 0 0.06 0 0"/>
                        <body name="right_wrist_yaw_link" pos="0.046 0 0">
                          <inertial pos="0.0708244 -0.000191745 0.00161742" quat="0.493188 0.468078 0.526295 0.510571"
                            mass="0.254576" diaginertia="0.000646113 0.000559993 0.000147566"/>
                          <joint name="right_wrist_yaw_joint" axis="0 0 1" range="-1.61443 1.61443" armature="0.00425" actuatorfrcrange="-5 5"/>
                          <geom class="visual" mesh="right_wrist_yaw_link"/>
                          <geom pos="0.0415 -0.003 0" quat="1 0 0 0" type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_palm_link"/>
                          <geom pos="0.0415 -0.003 0" quat="1 0 0 0" type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_palm_link"/>
                          <site name="right_palm" pos="0.08 0 0" size="0.01"/>
                          <camera name="wrist_cam" pos="0.08 0.04 0" xyaxes="0 0 1 0 1 0" fovy="75"/>
                          <body name="right_hand_thumb_0_link" pos="0.067 -0.003 0">
                            <inertial pos="-0.000884246 0.00863407 0.000944293" quat="0.643965 0.462991 -0.398986 0.460173" mass="0.0862366" diaginertia="1.6546e-05 1.60058e-05 1.43741e-05"/>
                            <joint name="right_hand_thumb_0_joint" axis="0 1 0" range="-1.0472 1.0472" actuatorfrcrange="-2.45 2.45"/>
                            <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_thumb_0_link"/>
                            <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_thumb_0_link"/>
                            <body name="right_hand_thumb_1_link" pos="-0.0025 0.0193 0">
                              <inertial pos="-0.000827888 0.0354744 -0.0003809" quat="0.705471 0.685598 -0.0956069 0.15207" mass="0.0588507" diaginertia="1.28514e-05 1.22902e-05 5.9666e-06"/>
                              <joint name="right_hand_thumb_1_joint" axis="0 0 1" range="-1.0472 0.724312" actuatorfrcrange="-1.4 1.4"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_thumb_1_link"/>
                              <geom size="0.01 0.015 0.01" pos="-0.001 0.032 0" type="box" rgba="0.7 0.7 0.7 1"/>
                              <body name="right_hand_thumb_2_link" pos="0 0.0458 0">
                                <inertial pos="-0.00171735 0.0262819 0.000107789" quat="0.710977 0.703174 0.00766553 0.00017564" mass="0.0203063" diaginertia="4.61314e-06 3.86645e-06 1.53495e-06"/>
                                <joint name="right_hand_thumb_2_joint" axis="0 0 1" range="-1.74533 0" actuatorfrcrange="-1.4 1.4"/>
                                <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_thumb_2_link"/>
                                <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_thumb_2_link"/>
                              </body>
                            </body>
                          </body>
                          <body name="right_hand_middle_0_link" pos="0.1192 -0.0046 -0.0285">
                            <inertial pos="0.0354744 -0.000827888 0.0003809" quat="0.606373 0.417187 0.552395 0.391313" mass="0.0588507" diaginertia="1.28514e-05 1.22902e-05 5.9666e-06"/>
                            <joint name="right_hand_middle_0_joint" axis="0 0 1" range="0 1.5708" actuatorfrcrange="-1.4 1.4"/>
                            <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_middle_0_link"/>
                            <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_middle_0_link"/>
                            <body name="right_hand_middle_1_link" pos="0.0458 0 0">
                              <inertial pos="0.0262819 -0.00171735 -0.000107789" quat="0.502861 0.502639 0.491799 0.502612" mass="0.0203063" diaginertia="4.61314e-06 3.86645e-06 1.53495e-06"/>
                              <joint name="right_hand_middle_1_joint" axis="0 0 1" range="0 1.74533" actuatorfrcrange="-1.4 1.4"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_middle_1_link"/>
                              <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_middle_1_link"/>
                            </body>
                          </body>
                          <body name="right_hand_index_0_link" pos="0.1192 -0.0046 0.0285">
                            <inertial pos="0.0354744 -0.000827888 0.0003809" quat="0.606373 0.417187 0.552395 0.391313" mass="0.0588507" diaginertia="1.28514e-05 1.22902e-05 5.9666e-06"/>
                            <joint name="right_hand_index_0_joint" axis="0 0 1" range="0 1.5708" actuatorfrcrange="-1.4 1.4"/>
                            <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_index_0_link"/>
                            <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_index_0_link"/>
                            <body name="right_hand_index_1_link" pos="0.0458 0 0">
                              <inertial pos="0.0262819 -0.00171735 -0.000107789" quat="0.502861 0.502639 0.491799 0.502612" mass="0.0203063" diaginertia="4.61314e-06 3.86645e-06 1.53495e-06"/>
                              <joint name="right_hand_index_1_joint" axis="0 0 1" range="0 1.74533" actuatorfrcrange="-1.4 1.4"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.7 0.7 0.7 1" mesh="right_hand_index_1_link"/>
                              <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="right_hand_index_1_link"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

<actuator>
    <position class="g1" name="left_hip_pitch_joint" joint="left_hip_pitch_joint" kp="40.179" kv="2.558"/>
    <position class="g1" name="left_hip_roll_joint" joint="left_hip_roll_joint" kp="99.098" kv="6.309"/>
    <position class="g1" name="left_hip_yaw_joint" joint="left_hip_yaw_joint" kp="40.179" kv="2.558"/>
    <position class="g1" name="left_knee_joint" joint="left_knee_joint" kp="99.098" kv="6.309"/>
    <position class="g1" name="left_ankle_pitch_joint" joint="left_ankle_pitch_joint" kp="28.501" kv="1.814"/>
    <position class="g1" name="left_ankle_roll_joint" joint="left_ankle_roll_joint" kp="28.501" kv="1.814"/>

    <position class="g1" name="right_hip_pitch_joint" joint="right_hip_pitch_joint" kp="40.179" kv="2.558"/>
    <position class="g1" name="right_hip_roll_joint" joint="right_hip_roll_joint" kp="99.098" kv="6.309"/>
    <position class="g1" name="right_hip_yaw_joint" joint="right_hip_yaw_joint" kp="40.179" kv="2.558"/>
    <position class="g1" name="right_knee_joint" joint="right_knee_joint" kp="99.098" kv="6.309"/>
    <position class="g1" name="right_ankle_pitch_joint" joint="right_ankle_pitch_joint" kp="28.501" kv="1.814"/>
    <position class="g1" name="right_ankle_roll_joint" joint="right_ankle_roll_joint" kp="28.501" kv="1.814"/>

    <position class="g1" name="waist_yaw_joint" joint="waist_yaw_joint" kp="40.179" kv="2.558"/>
    <position class="g1" name="waist_roll_joint" joint="waist_roll_joint" kp="28.501" kv="1.814"/>
    <position class="g1" name="waist_pitch_joint" joint="waist_pitch_joint" kp="28.501" kv="1.814"/>

    <position class="g1" name="left_shoulder_pitch_joint" joint="left_shoulder_pitch_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="left_shoulder_roll_joint" joint="left_shoulder_roll_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="left_shoulder_yaw_joint" joint="left_shoulder_yaw_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="left_elbow_joint" joint="left_elbow_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="left_wrist_roll_joint" joint="left_wrist_roll_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="left_wrist_pitch_joint" joint="left_wrist_pitch_joint" kp="16.778" kv="1.068"/>
    <position class="g1" name="left_wrist_yaw_joint" joint="left_wrist_yaw_joint" kp="16.778" kv="1.068"/>

    <position class="g1" name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="right_shoulder_roll_joint" joint="right_shoulder_roll_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="right_shoulder_yaw_joint" joint="right_shoulder_yaw_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="right_elbow_joint" joint="right_elbow_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="right_wrist_roll_joint" joint="right_wrist_roll_joint" kp="14.251" kv="0.907"/>
    <position class="g1" name="right_wrist_pitch_joint" joint="right_wrist_pitch_joint" kp="16.778" kv="1.068"/>
    <position class="g1" name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint" kp="16.778" kv="1.068"/>

    <!-- Inspire hand actuators (position control) -->
    <position name="left_hand_thumb_0_joint" joint="left_hand_thumb_0_joint" kp="2.0" kv="0.1"/>
    <position name="left_hand_thumb_1_joint" joint="left_hand_thumb_1_joint" kp="1.5" kv="0.1"/>
    <position name="left_hand_thumb_2_joint" joint="left_hand_thumb_2_joint" kp="1.5" kv="0.1"/>
    <position name="left_hand_middle_0_joint" joint="left_hand_middle_0_joint" kp="1.5" kv="0.1"/>
    <position name="left_hand_middle_1_joint" joint="left_hand_middle_1_joint" kp="1.5" kv="0.1"/>
    <position name="left_hand_index_0_joint" joint="left_hand_index_0_joint" kp="1.5" kv="0.1"/>
    <position name="left_hand_index_1_joint" joint="left_hand_index_1_joint" kp="1.5" kv="0.1"/>
    <position name="right_hand_thumb_0_joint" joint="right_hand_thumb_0_joint" kp="2.0" kv="0.1"/>
    <position name="right_hand_thumb_1_joint" joint="right_hand_thumb_1_joint" kp="1.5" kv="0.1"/>
    <position name="right_hand_thumb_2_joint" joint="right_hand_thumb_2_joint" kp="1.5" kv="0.1"/>
    <position name="right_hand_middle_0_joint" joint="right_hand_middle_0_joint" kp="1.5" kv="0.1"/>
    <position name="right_hand_middle_1_joint" joint="right_hand_middle_1_joint" kp="1.5" kv="0.1"/>
    <position name="right_hand_index_0_joint" joint="right_hand_index_0_joint" kp="1.5" kv="0.1"/>
    <position name="right_hand_index_1_joint" joint="right_hand_index_1_joint" kp="1.5" kv="0.1"/>
  </actuator>

  <contact>
    <exclude body1="left_elbow_link" body2="left_wrist_pitch_link"/>
    <exclude body1="right_elbow_link" body2="right_wrist_pitch_link"/>
    <exclude body1="pelvis" body2="right_hip_roll_link"/>
    <exclude body1="pelvis" body2="left_hip_roll_link"/>
  </contact>

  <sensor>
    <gyro site="imu_in_torso" name="imu-torso-angular-velocity" cutoff="34.9" noise="0.0005"/>
    <accelerometer site="imu_in_torso" name="imu-torso-linear-acceleration" cutoff="157" noise="0.01"/>
    <gyro site="imu_in_pelvis" name="imu-pelvis-angular-velocity" cutoff="34.9" noise="0.0005"/>
    <accelerometer site="imu_in_pelvis" name="imu-pelvis-linear-acceleration" cutoff="157" noise="0.01"/>
  </sensor>

  
</mujoco>
```

---

## FILE: `model_config.json`

```json
{
  "joint_names": [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
  ],
  "default_joint_pos": {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0
  },
  "action_scales": {
    "left_hip_pitch_joint": 0.547546,
    "left_hip_roll_joint": 0.350661,
    "left_hip_yaw_joint": 0.547546,
    "left_knee_joint": 0.350661,
    "left_ankle_pitch_joint": 0.438577,
    "left_ankle_roll_joint": 0.438577,
    "right_hip_pitch_joint": 0.547546,
    "right_hip_roll_joint": 0.350661,
    "right_hip_yaw_joint": 0.547546,
    "right_knee_joint": 0.350661,
    "right_ankle_pitch_joint": 0.438577,
    "right_ankle_roll_joint": 0.438577,
    "waist_yaw_joint": 0.547546,
    "waist_roll_joint": 0.438577,
    "waist_pitch_joint": 0.438577,
    "left_shoulder_pitch_joint": 0.438577,
    "left_shoulder_roll_joint": 0.438577,
    "left_shoulder_yaw_joint": 0.438577,
    "left_elbow_joint": 0.438577,
    "left_wrist_roll_joint": 0.438577,
    "left_wrist_pitch_joint": 0.074501,
    "left_wrist_yaw_joint": 0.074501,
    "right_shoulder_pitch_joint": 0.438577,
    "right_shoulder_roll_joint": 0.438577,
    "right_shoulder_yaw_joint": 0.438577,
    "right_elbow_joint": 0.438577,
    "right_wrist_roll_joint": 0.438577,
    "right_wrist_pitch_joint": 0.074501,
    "right_wrist_yaw_joint": 0.074501
  },
  "walker": {
    "input_dim": 99,
    "output_dim": 29,
    "obs_mean": [
      0.2764410078525543,
      -0.0011231002863496542,
      -0.00887491274625063,
      -0.012028997763991356,
      0.03522094711661339,
      0.008403312414884567,
      -0.003294438822194934,
      0.0002468313614372164,
      -0.9934708476066589,
      -0.0907304584980011,
      0.006743466015905142,
      -0.001323652919381857,
      0.22801245748996735,
      -0.0420287624001503,
      0.0052517312578856945,
      -0.09373214840888977,
      -0.0022980200592428446,
      0.007062251679599285,
      0.22945237159729004,
      -0.03992355987429619,
      -0.005189856979995966,
      -0.003225596621632576,
      0.0012408958282321692,
      0.000897581921890378,
      -0.03501684591174126,
      0.053861577063798904,
      0.01695811189711094,
      -0.025254687294363976,
      -0.00033013586653396487,
      -0.022842688485980034,
      0.04303140565752983,
      -0.03617796301841736,
      -0.05281267687678337,
      -0.016581477597355843,
      -0.023698199540376663,
      0.0004230269987601787,
      -0.018825259059667587,
      -0.045088429003953934,
      -0.05162270367145538,
      -0.0017306592781096697,
      -0.00020749638497363776,
      0.011742794886231422,
      0.0056776381097733974,
      -0.003033487591892481,
      -0.03596210852265358,
      0.007169663440436125,
      -0.012108002789318562,
      0.006323100067675114,
      0.00047465605894103646,
      0.000647611974272877,
      -0.013602513819932938,
      0.01769345998764038,
      -0.01986178196966648,
      0.03318953141570091,
      0.0037734888028353453,
      0.0095991101115942,
      0.004500563256442547,
      0.0021920751314610243,
      -0.004432667978107929,
      0.006716209463775158,
      0.027406007051467896,
      0.006401436403393745,
      -0.0032767809461802244,
      0.00846377294510603,
      0.006889165844768286,
      -0.004579206462949514,
      -0.006883460562676191,
      -0.1486780345439911,
      0.3012807071208954,
      -0.2587428092956543,
      0.32165414094924927,
      0.3147725462913513,
      0.08125457167625427,
      -0.15473435819149017,
      -0.28669875860214233,
      0.2629929482936859,
      0.32942909002304077,
      0.3306274712085724,
      -0.07872061431407928,
      -0.00882444903254509,
      0.0009497009450569749,
      -0.13094140589237213,
      -0.2002839893102646,
      0.33255425095558167,
      0.09981878846883774,
      -0.265374094247818,
      -0.0005433406331576407,
      -0.5369775891304016,
      0.6211817860603333,
      -0.2017064094543457,
      -0.3248439431190491,
      -0.09776587784290314,
      -0.260663777589798,
      0.002399812452495098,
      -0.4840352535247803,
      -0.648914098739624,
      0.28178897500038147,
      0.000009319964192,
      0.00041731251985765994
    ],
    "obs_std": [
      1.119999885559082,
      0.5925261378288269,
      0.32594776153564453,
      0.7118846774101257,
      0.8981359004974365,
      0.8713164329528809,
      0.07316633313894272,
      0.05465417355298996,
      0.040836624801158905,
      0.22480855882167816,
      0.09196151793003082,
      0.1053195372223854,
      0.2902180552482605,
      0.19904658198356628,
      0.09467905759811401,
      0.21679627895355225,
      0.09269601851701736,
      0.10471580922603607,
      0.2920810580253601,
      0.19660590589046478,
      0.09433563798666,
      0.15374724566936493,
      0.06007464975118637,
      0.06440386921167374,
      0.22716952860355377,
      0.1453830599784851,
      0.1440180242061615,
      0.15696296095848083,
      0.20508013665676117,
      0.12280701100826263,
      0.12018921971321106,
      0.2249361276626587,
      0.1436595618724823,
      0.143561452627182,
      0.1574581116437912,
      0.20672756433486938,
      0.1239062175154686,
      0.12004795670509338,
      2.6440131664276123,
      1.5611387491226196,
      2.0153350830078125,
      4.0273003578186035,
      4.3755598068237305,
      2.9597818851470947,
      2.600646734237671,
      1.5530928373336792,
      2.016266345977783,
      4.008848667144775,
      4.380704879760742,
      2.9649221897125244,
      2.1029741764068604,
      1.3836265802383423,
      1.389782428741455,
      2.610670328140259,
      2.1601574420928955,
      3.2699809074401855,
      3.278838634490967,
      4.32094144821167,
      1.6473523378372192,
      1.5537536144256592,
      2.6113548278808594,
      2.1469244956970215,
      3.2662229537963867,
      3.2944107055664062,
      4.339025497436523,
      1.6455715894699097,
      1.5547151565551758,
      0.8396009802818298,
      0.5547389984130859,
      0.49920323491096497,
      1.3047682046890259,
      1.309758186340332,
      0.5676404237747192,
      0.8267436027526855,
      0.5543197393417358,
      0.49321919679641724,
      1.3026329278945923,
      1.3167572021484375,
      0.5665327310562134,
      0.472351998090744,
      0.6063718199729919,
      0.42982882261276245,
      0.7897917628288269,
      0.6228054165840149,
      0.5509909987449646,
      0.6418794989585876,
      0.7377936840057373,
      1.5482919216156006,
      1.5455572605133057,
      0.7844030261039734,
      0.6124319434165955,
      0.5508608818054199,
      0.6462455987930298,
      0.7424951791763306,
      1.563859462738037,
      1.5450600385665894,
      1.1348178386688232,
      0.5475941300392151,
      0.3739437758922577
    ]
  },
  "croucher": {
    "input_dim": 101,
    "output_dim": 29,
    "obs_mean": [
      -0.014261700212955475,
      -0.0015256619080901146,
      -0.07104996591806412,
      0.08402465283870697,
      0.19617024064064026,
      0.04232420399785042,
      0.009945405647158623,
      0.005697592161595821,
      -0.9673172235488892,
      -0.7404491305351257,
      0.016528654843568802,
      -0.059806033968925476,
      1.1599997282028198,
      -0.42899245023727417,
      -0.04512963071465492,
      -0.7139285802841187,
      -0.18219523131847382,
      0.10681703686714172,
      1.1616051197052002,
      -0.44435152411460876,
      0.11815741658210754,
      0.004003847949206829,
      0.018390238285064697,
      0.017038686200976372,
      -0.5881016254425049,
      0.224533811211586,
      0.04629017040133476,
      -0.16457831859588623,
      0.3489656448364258,
      0.21692031621932983,
      -0.09602002054452896,
      -0.20428518950939178,
      -0.5440213084220886,
      0.10759054869413376,
      0.42052602767944336,
      -0.5271346569061279,
      -0.21750664710998535,
      -0.0706288143992424,
      -0.3952099084854126,
      0.01875457540154457,
      0.13469567894935608,
      0.10036010295152664,
      0.02767588011920452,
      0.006179089192301035,
      -0.39470037817955017,
      -0.038679271936416626,
      0.27906712889671326,
      0.21088196337223053,
      -0.03757520765066147,
      0.011537430807948112,
      0.000071282433055,
      -0.06105758622288704,
      -0.08467136323451996,
      -0.19200098514556885,
      0.08638951927423477,
      0.06381367146968842,
      0.014261274598538876,
      0.013615953736007214,
      0.037713173776865005,
      -0.021199651062488556,
      -0.062187258154153824,
      -0.17009097337722778,
      -0.08419857174158096,
      0.0005883211852051318,
      -0.11533480882644653,
      -0.03991479426622391,
      -0.014236228540539742,
      -1.2952044010162354,
      0.08484380692243576,
      -0.1824222058057785,
      2.4944701194763184,
      -1.3858081102371216,
      -0.07759882509708405,
      -1.2816790342330933,
      -0.5512250065803528,
      0.17243026196956635,
      2.5755951404571533,
      -1.511191725730896,
      0.25833985209465027,
      0.005510265938937664,
      -0.11727026104927063,
      -0.3265622854232788,
      -1.8216381072998047,
      0.8877902030944824,
      0.22886404395103455,
      -0.5532782077789307,
      0.8074663281440735,
      2.8620448112487793,
      -1.2452894449234009,
      -0.5691277384757996,
      -1.8547887802124023,
      0.1585928201675415,
      0.9338672161102295,
      -1.2163255214691162,
      -2.9625375270843506,
      -0.9975863695144653,
      0.0,
      0.0,
      0.0,
      0.5423733592033386,
      0.5588040351867676
    ],
    "obs_std": [
      0.3474659323692322,
      0.32971280813217163,
      0.46664831042289734,
      1.0423789024353027,
      1.6083815097808838,
      1.0413382053375244,
      0.2074316442012787,
      0.12730160355567932,
      0.08611535280942917,
      0.4785121977329254,
      0.11616149544715881,
      0.19557489454746246,
      0.5106216669082642,
      0.1908293217420578,
      0.1222715675830841,
      0.5035606026649475,
      0.2355995774269104,
      0.21196699142456055,
      0.5028953552246094,
      0.1780749410390854,
      0.09761162102222443,
      0.07040099054574966,
      0.09590069949626923,
      0.1278066486120224,
      0.5945335030555725,
      0.3940373659133911,
      0.5054469704627991,
      0.5720513463020325,
      0.6200873851776123,
      0.24052157998085022,
      0.1555127054452896,
      0.4658084213733673,
      0.46817442774772644,
      0.5978217720985413,
      0.4964078664779663,
      0.7145026922225952,
      0.2642146646976471,
      0.14052928984165192,
      2.506350040435791,
      1.6340935230255127,
      2.1194469928741455,
      1.6562771797180176,
      1.7600340843200684,
      2.239684820175171,
      2.5764975547790527,
      1.6906346082687378,
      2.434779167175293,
      1.839568018913269,
      1.7275861501693726,
      2.268329381942749,
      1.9070736169815063,
      1.5923981666564941,
      1.806822419166565,
      2.906200647354126,
      2.7093393802642822,
      4.559217929840088,
      4.170853614807129,
      5.402627944946289,
      1.6825909614562988,
      1.5556590557098389,
      3.1233408451080322,
      2.527413845062256,
      4.870856285095215,
      4.206680774688721,
      5.403183460235596,
      1.6756086349487305,
      1.5618946552276611,
      0.9751729965209961,
      0.5106522440910339,
      0.648868978023529,
      1.4018257856369019,
      1.2515277862548828,
      0.48892202973365784,
      0.9916493892669678,
      0.76868736743927,
      0.657737135887146,
      1.3760528564453125,
      1.2443814277648926,
      0.47128432989120483,
      0.4551670253276825,
      0.6994064450263977,
      0.7649814486503601,
      1.6757733821868896,
      1.0768580436706543,
      1.2577259540557861,
      1.4693176746368408,
      1.6065880060195923,
      3.205024242401123,
      2.023322105407715,
      1.3912105560302734,
      1.2911109924316406,
      1.5184346437454224,
      1.3640635013580322,
      1.7877761125564575,
      3.415318250656128,
      1.834232211112976,
      0.0,
      0.0,
      0.0,
      0.09675803780555725,
      0.1217656061053276
    ]
  }
}
```

---

## FILE: `right_reacher.onnx`

_Skipped: non-text or binary file._

---

## FILE: `right_reacher.onnx.data`

_Skipped: non-text or binary file._

---

## FILE: `rotator.onnx`

_Skipped: file is too large (878886 bytes)._ 

---

## FILE: `rotator.onnx.data`

_Skipped: file is too large (877336 bytes)._ 

---

## FILE: `run.py`

```python
#!/usr/bin/env python3
"""G1 Table Red Block — standalone MuJoCo scene with walker + reacher policies.

Converted from the LuckyEngine G1-Table-Red-Block.hscene. Runs the G1 robot
with trained Walker/Croucher/Rotator/Reacher ONNX policies in a scene with
a table, red cylindrical block, and multiple cameras (head, wrist, overhead).

Controls (press keys in the GLFW viewer window):
  Arrow Keys   : Walk forward/back, strafe left/right
  ; / '        : Turn left / right
  ,            : Toggle crouch mode
  [ / ]        : Height down / up
  \\           : Stop (zero velocity)
  /            : Toggle arm freeze
  .            : Toggle reach mode (right arm)
  Numpad 8/2   : Reach target forward/backward
  Numpad 4/6   : Reach target left/right
  Numpad 7/1   : Reach target up/down
  Numpad 5     : Reset reach target (auto mode)
  U/J, Y/H, 9/0 : Reach orientation roll/pitch/yaw
  R            : Reset reach orientation
  Space        : Reset robot + zero velocity
  C            : Cycle camera view in main window
  1            : Toggle head camera window
  2            : Toggle wrist camera window

Prerequisites:
  pip install mujoco onnxruntime numpy opencv-python

Usage:
  python run.py
  python run.py --no-cameras    # Disable camera windows (faster)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# ONNX Policy
# --------------------------------------------------------------------------- #
class ONNXPolicy:
  """ONNX policy wrapper for CPU inference."""

  def __init__(self, model_path: str):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      model_path, sess_options, providers=["CPUExecutionProvider"]
    )
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
      obs = obs.reshape(1, -1)
    obs = obs.astype(np.float32)
    return self.session.run([self.output_name], {self.input_name: obs})[0][0]


# --------------------------------------------------------------------------- #
# G1 Controller (walker + croucher + rotator + reacher)
# --------------------------------------------------------------------------- #
class G1Controller:
  """Full G1 controller with locomotion mode switching and arm reaching."""

  # GLFW key codes
  KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
  KEY_SEMICOLON, KEY_APOSTROPHE = 59, 39
  KEY_COMMA, KEY_PERIOD, KEY_SLASH, KEY_BACKSLASH = 44, 46, 47, 92
  KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET = 91, 93
  KEY_KP_8, KEY_KP_2, KEY_KP_6, KEY_KP_4, KEY_KP_7, KEY_KP_1, KEY_KP_5 = (
    328, 322, 326, 324, 327, 321, 325
  )
  KEY_U, KEY_J, KEY_Y, KEY_H, KEY_9, KEY_0, KEY_R = 85, 74, 89, 72, 57, 48, 82
  KEY_COMMA_GRIP = 44  # , = Grip toggle

  WALKER_HEIGHT = 0.80

  def __init__(self, model, data, walker, croucher, rotator, config,
               right_reacher=None):
    self.model = model
    self.data = data
    self.walker_policy = walker
    self.croucher_policy = croucher
    self.rotator_policy = rotator
    self.right_reacher_policy = right_reacher
    self.config = config

    # --- Input mode: WALK or REACH ---
    # . toggles between them. Same keys (arrows, ;/') do different things.
    self.input_mode: Literal["walk", "reach"] = "walk"

    # Walk state
    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0
    self.vel_step_linear = 0.2
    self.vel_step_angular = 0.2
    self.vel_max_linear = 2.0
    self.vel_max_angular = 1.0

    # Reach state
    self.reach_active = False
    self.reach_target = np.array([0.3, -0.2, 0.2], dtype=np.float32)
    self.reach_orientation = np.zeros(3, dtype=np.float32)
    self.reach_step = 0.05
    self.last_arm_action = np.zeros(7, dtype=np.float32)
    self.last_arm_target = None
    self.arm_max_delta = 0.012
    # Frozen arm position — holds the last reach position when switching to walk
    self.frozen_arm_pos = None  # None = use defaults, array = hold position

    self.last_action = np.zeros(29, dtype=np.float32)

    # Right hand grip state
    self.grip_closed = False

    self._build_joint_mappings()
    self._build_reacher_mappings()
    self._compute_pd_gains()
    self._cache_actuator_ids()
    self._cache_finger_actuators()

    print("\n=== G1 Table Red Block Controller ===")
    print("  .         : Toggle WALK / REACH mode")
    print("  --- WALK mode ---")
    print("  Arrows    : Walk forward/back, strafe left/right")
    print("  ; / '     : Turn left / right")
    print("  \\         : Stop")
    print("  --- REACH mode ---")
    print("  Up/Down   : Reach forward / backward")
    print("  Left/Right: Reach left / right")
    print("  ; / '     : Reach up / down")
    print("  \\         : Reset reach to default")
    print("  --- Always ---")
    print("  ,         : Toggle grip (close/open right hand)")
    print("  Space     : Reset robot")
    print("=" * 40)

  def _build_joint_mappings(self):
    self.joint_names = self.config["joint_names"]
    self.num_joints = len(self.joint_names)
    self.joint_qpos_indices = {n: 7 + i for i, n in enumerate(self.joint_names)}
    self.joint_qvel_indices = {n: 6 + i for i, n in enumerate(self.joint_names)}

    self.default_joint_pos = np.zeros(self.num_joints, dtype=np.float32)
    for name, value in self.config["default_joint_pos"].items():
      if name in self.joint_names:
        self.default_joint_pos[self.joint_names.index(name)] = value

    self.action_scales = np.array(
      [self.config["action_scales"][n] for n in self.joint_names], dtype=np.float32
    )

    arm_patterns = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    self.arm_indices = []
    for i, name in enumerate(self.joint_names):
      if any(p in name for p in arm_patterns):
        self.arm_indices.append(i)

  def _build_reacher_mappings(self):
    rc = self.config.get("right_reacher", {})
    self.right_arm_joint_names = rc.get("arm_joint_names", [
      "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint", "right_elbow_joint",
      "right_wrist_roll_joint", "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ])
    self.right_arm_indices = [
      self.joint_names.index(n) for n in self.right_arm_joint_names
      if n in self.joint_names
    ]
    arm_scales = rc.get("arm_action_scales", {})
    self.arm_action_scales = np.array([
      arm_scales.get(n, self.action_scales[self.joint_names.index(n)])
      for n in self.right_arm_joint_names
    ], dtype=np.float32)
    arm_defaults = rc.get("arm_default_pos", {})
    self.arm_default_pos = np.array([
      arm_defaults.get(n, self.default_joint_pos[self.joint_names.index(n)])
      for n in self.right_arm_joint_names
    ], dtype=np.float32)
    self.right_palm_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
    )

  def _compute_pd_gains(self):
    S5020, D5020, E5020 = 14.2506, 0.9072, 25.0
    S7520_14, D7520_14, E7520_14 = 40.1792, 2.5579, 88.0
    S7520_22, D7520_22, E7520_22 = 99.0984, 6.3088, 139.0
    S4010, D4010, E4010 = 16.7783, 1.0681, 5.0

    self.kp = np.zeros(self.num_joints, dtype=np.float32)
    self.kd = np.zeros(self.num_joints, dtype=np.float32)
    self.effort_limit = np.zeros(self.num_joints, dtype=np.float32)

    for i, name in enumerate(self.joint_names):
      if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020
      elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
        self.kp[i], self.kd[i], self.effort_limit[i] = S7520_14, D7520_14, E7520_14
      elif "hip_roll" in name or "knee" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S7520_22, D7520_22, E7520_22
      elif "wrist_pitch" in name or "wrist_yaw" in name:
        self.kp[i], self.kd[i], self.effort_limit[i] = S4010, D4010, E4010
      elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020 * 2, D5020 * 2, E5020 * 2
      else:
        self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020

  # --- Keyboard ---
  def key_callback(self, key: int) -> None:
    # Grip toggle (works in any mode)
    if key == self.KEY_COMMA_GRIP:
      self.grip_closed = not self.grip_closed
      print(f"[GRIP] Right hand: {'CLOSED' if self.grip_closed else 'OPEN'}")
      return

    # Toggle input mode
    if key == self.KEY_PERIOD:
      if self.right_reacher_policy is None:
        print("[WARN] No right reacher policy loaded")
        return
      if self.input_mode == "walk":
        self.input_mode = "reach"
        self.reach_active = True
        # Init reach target to a sensible default in front of pelvis
        self.reach_target[:] = [0.3, -0.2, 0.2]
        self.reach_orientation[:] = 0.0
        self.last_arm_target = self._get_arm_joint_positions() + self.arm_default_pos
        print("[MODE] >>> REACH — arrows move hand, ;/' = up/down, \\ = reset target")
      else:
        self.input_mode = "walk"
        self.reach_active = False
        # Freeze arm where it is — read current right arm joint positions
        if self.last_arm_target is not None:
          self.frozen_arm_pos = self.last_arm_target.copy()
        self.last_arm_target = None
        print("[MODE] >>> WALK — arm holds position, arrows move robot")
      return

    # Route keys based on mode
    if self.input_mode == "walk":
      self._handle_walk_key(key)
    else:
      self._handle_reach_key(key)

  def _handle_walk_key(self, key: int) -> None:
    if key == self.KEY_UP:
      self.lin_vel_x = np.clip(self.lin_vel_x + self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_DOWN:
      self.lin_vel_x = np.clip(self.lin_vel_x - self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_LEFT:
      self.lin_vel_y = np.clip(self.lin_vel_y + self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_RIGHT:
      self.lin_vel_y = np.clip(self.lin_vel_y - self.vel_step_linear, -self.vel_max_linear, self.vel_max_linear)
    elif key == self.KEY_SEMICOLON:
      self.ang_vel_z = np.clip(self.ang_vel_z + self.vel_step_angular, -self.vel_max_angular, self.vel_max_angular)
    elif key == self.KEY_APOSTROPHE:
      self.ang_vel_z = np.clip(self.ang_vel_z - self.vel_step_angular, -self.vel_max_angular, self.vel_max_angular)
    elif key == self.KEY_BACKSLASH or key == self.KEY_SLASH:
      self.lin_vel_x = self.lin_vel_y = self.ang_vel_z = 0.0
      print("[WALK] STOPPED")
      return
    else:
      return
    print(f"[WALK] vel: x={self.lin_vel_x:.1f} y={self.lin_vel_y:.1f} yaw={self.ang_vel_z:.1f}")

  def _handle_reach_key(self, key: int) -> None:
    if key == self.KEY_UP:
      self.reach_target[0] = np.clip(self.reach_target[0] + self.reach_step, -0.3, 0.6)
    elif key == self.KEY_DOWN:
      self.reach_target[0] = np.clip(self.reach_target[0] - self.reach_step, -0.3, 0.6)
    elif key == self.KEY_LEFT:
      self.reach_target[1] = np.clip(self.reach_target[1] + self.reach_step, -0.6, 0.3)
    elif key == self.KEY_RIGHT:
      self.reach_target[1] = np.clip(self.reach_target[1] - self.reach_step, -0.6, 0.3)
    elif key == self.KEY_SEMICOLON:
      self.reach_target[2] = np.clip(self.reach_target[2] + self.reach_step, -0.4, 0.6)
    elif key == self.KEY_APOSTROPHE:
      self.reach_target[2] = np.clip(self.reach_target[2] - self.reach_step, -0.4, 0.6)
    elif key == self.KEY_BACKSLASH or key == self.KEY_SLASH:
      self.reach_target[:] = [0.3, -0.2, 0.2]
      self.reach_orientation[:] = 0.0
      print("[REACH] Target reset to default")
      return
    else:
      return
    print(f"[REACH] target: fwd={self.reach_target[0]:.2f} side={self.reach_target[1]:.2f} up={self.reach_target[2]:.2f}")

  # --- State helpers ---
  def _get_base_pose(self):
    return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

  @staticmethod
  def _quat_apply_inverse(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

  def _get_base_velocities(self):
    lin_vel_world = self.data.qvel[:3].copy()
    ang_vel_body = self.data.qvel[3:6].copy()
    _, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, lin_vel_world), ang_vel_body

  def _get_projected_gravity(self):
    _, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0]))

  def _get_joint_positions(self):
    pos = np.zeros(self.num_joints, dtype=np.float32)
    for i, n in enumerate(self.joint_names):
      pos[i] = self.data.qpos[self.joint_qpos_indices[n]] - self.default_joint_pos[i]
    return pos

  def _get_joint_velocities(self):
    vel = np.zeros(self.num_joints, dtype=np.float32)
    for i, n in enumerate(self.joint_names):
      vel[i] = self.data.qvel[self.joint_qvel_indices[n]]
    return vel

  def _get_arm_joint_positions(self):
    pos = np.zeros(len(self.right_arm_indices), dtype=np.float32)
    for i, idx in enumerate(self.right_arm_indices):
      n = self.joint_names[idx]
      pos[i] = self.data.qpos[self.joint_qpos_indices[n]] - self.arm_default_pos[i]
    return pos

  def _get_arm_joint_velocities(self):
    vel = np.zeros(len(self.right_arm_indices), dtype=np.float32)
    for i, idx in enumerate(self.right_arm_indices):
      vel[i] = self.data.qvel[self.joint_qvel_indices[self.joint_names[idx]]]
    return vel

  def _get_palm_pos_in_pelvis(self):
    palm_world = self.data.site_xpos[self.right_palm_site_id].copy()
    pos, quat = self._get_base_pose()
    return self._quat_apply_inverse(quat, palm_world - pos)

  def _get_palm_orientation_in_pelvis(self):
    mat = self.data.site_xmat[self.right_palm_site_id].reshape(3, 3)
    palm_q = np.zeros(4)
    mujoco.mju_mat2Quat(palm_q, mat.flatten())
    _, pelvis_q = self._get_base_pose()
    pinv = np.array([pelvis_q[0], -pelvis_q[1], -pelvis_q[2], -pelvis_q[3]])
    w1, x1, y1, z1 = pinv
    w2, x2, y2, z2 = palm_q
    rel = np.array([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    w, x, y, z = rel
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)

  # --- Step ---
  def step(self) -> np.ndarray:
    # Build walker observation (always runs — keeps legs stable)
    lin_vel, ang_vel = self._get_base_velocities()
    proj_gravity = self._get_projected_gravity()
    joint_pos = self._get_joint_positions()
    joint_vel = self._get_joint_velocities()

    cmd = np.array([self.lin_vel_x, self.lin_vel_y, self.ang_vel_z], dtype=np.float32)

    obs = np.concatenate([
      lin_vel, ang_vel, proj_gravity, joint_pos, joint_vel, self.last_action, cmd,
    ]).astype(np.float32)

    # Walker policy (handles legs, waist, standing, walking, turning)
    action = self.walker_policy(obs)
    target_pos = self.default_joint_pos + action * self.action_scales

    # Arms: left arm always at default, right arm holds last reach position
    for idx in self.arm_indices:
      target_pos[idx] = self.default_joint_pos[idx]

    # Right arm: if we have a frozen position from a previous reach, hold it
    if not self.reach_active and self.frozen_arm_pos is not None:
      for i, full_idx in enumerate(self.right_arm_indices):
        target_pos[full_idx] = self.frozen_arm_pos[i]

    # Right arm reacher overlay (when in reach mode)
    if self.reach_active and self.right_reacher_policy is not None:
      reacher_obs = np.concatenate([
        self.reach_target,
        self.reach_orientation,
        self._get_palm_pos_in_pelvis(),
        self._get_palm_orientation_in_pelvis(),
        self._get_arm_joint_positions(),
        self._get_arm_joint_velocities(),
        self.last_arm_action,
        proj_gravity.astype(np.float32),
      ]).astype(np.float32)

      arm_action = self.right_reacher_policy(reacher_obs)
      arm_target = self.arm_default_pos + arm_action * self.arm_action_scales

      if self.last_arm_target is not None:
        delta = np.clip(arm_target - self.last_arm_target, -self.arm_max_delta, self.arm_max_delta)
        arm_target = self.last_arm_target + delta
      self.last_arm_target = arm_target.copy()

      for i, full_idx in enumerate(self.right_arm_indices):
        target_pos[full_idx] = arm_target[i]
      self.last_arm_action = arm_action.copy()

    self.last_action = action.copy()
    return target_pos

  def _cache_actuator_ids(self):
    """Cache actuator IDs once at init instead of looking up every step."""
    self.actuator_ids = []
    for name in self.joint_names:
      self.actuator_ids.append(
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      )

  def _cache_finger_actuators(self):
    """Cache right hand finger actuator IDs and their closed targets."""
    # (actuator_id, closed_position) — targets at joint limits for a power grasp
    self.right_finger_actuators = []
    finger_closed = {
      "right_hand_thumb_0_joint":  0.8,     # curl thumb inward
      "right_hand_thumb_1_joint": -0.9,     # flex thumb
      "right_hand_thumb_2_joint": -1.5,     # curl thumb tip
      "right_hand_index_0_joint":  1.4,     # curl index
      "right_hand_index_1_joint":  1.5,     # curl index tip
      "right_hand_middle_0_joint": 1.4,     # curl middle
      "right_hand_middle_1_joint": 1.5,     # curl middle tip
    }
    for name, closed_val in finger_closed.items():
      aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      if aid >= 0:
        self.right_finger_actuators.append((aid, closed_val))

  def apply_pd_control(self, target_pos):
    for i, act_id in enumerate(self.actuator_ids):
      if act_id >= 0:
        self.data.ctrl[act_id] = target_pos[i]
    # Apply grip
    for act_id, closed_val in self.right_finger_actuators:
      self.data.ctrl[act_id] = closed_val if self.grip_closed else 0.0


# --------------------------------------------------------------------------- #
# Camera renderer (uses mujoco.Renderer — reliable offscreen rendering)
# --------------------------------------------------------------------------- #
class CameraRenderer:
  """Offscreen renderer for robot-mounted cameras using mujoco.Renderer."""

  def __init__(self, model, data, width=320, height=240):
    self.model = model
    self.data = data
    self.renderer = mujoco.Renderer(model, height, width)

  def render(self, camera_name: str) -> np.ndarray:
    """Render from a named camera, return RGB array (H, W, 3)."""
    self.renderer.update_scene(self.data, camera=camera_name)
    return self.renderer.render().copy()


# --------------------------------------------------------------------------- #
# Armature setup
# --------------------------------------------------------------------------- #
def set_armature(model, joint_names):
  ARM_5020 = 0.00360972
  ARM_7520_14 = 0.01017752
  ARM_7520_22 = 0.02510192
  ARM_4010 = 0.00425000
  ARM_2x5020 = 0.00721945

  for i, name in enumerate(joint_names):
    dof = 6 + i
    if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
      model.dof_armature[dof] = ARM_5020
    elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
      model.dof_armature[dof] = ARM_7520_14
    elif "hip_roll" in name or "knee" in name:
      model.dof_armature[dof] = ARM_7520_22
    elif "wrist_pitch" in name or "wrist_yaw" in name:
      model.dof_armature[dof] = ARM_4010
    elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
      model.dof_armature[dof] = ARM_2x5020
    else:
      model.dof_armature[dof] = ARM_5020


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
  parser = argparse.ArgumentParser(description="G1 Table Red Block — MuJoCo standalone")
  parser.add_argument("--no-cameras", action="store_true", help="Disable camera windows")
  parser.add_argument("--cam-fps", type=int, default=10, help="Camera render FPS (default: 10)")
  args = parser.parse_args()

  # Load config
  config_path = SCRIPT_DIR / "model_config.json"
  with open(config_path) as f:
    config = json.load(f)
  joint_names = config["joint_names"]

  # Load scene
  xml_path = SCRIPT_DIR / "scene.xml"
  print(f"Loading scene: {xml_path}")
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  model.opt.timestep = 0.005  # 200 Hz — must match training
  set_armature(model, joint_names)

  data = mujoco.MjData(model)

  # Init robot pose — spawn behind the table, facing it
  data.qpos[0] = -0.6  # x: back from table
  data.qpos[2] = 0.76
  data.qpos[3:7] = [1, 0, 0, 0]
  for name, value in config["default_joint_pos"].items():
    if name in joint_names:
      data.qpos[7 + joint_names.index(name)] = value
  mujoco.mj_forward(model, data)

  # Load policies
  print("Loading ONNX policies...")
  walker = ONNXPolicy(str(SCRIPT_DIR / "walker.onnx"))
  croucher = ONNXPolicy(str(SCRIPT_DIR / "croucher.onnx"))
  rotator = ONNXPolicy(str(SCRIPT_DIR / "rotator.onnx"))

  right_reacher = None
  rr_path = SCRIPT_DIR / "right_reacher.onnx"
  if rr_path.exists():
    right_reacher = ONNXPolicy(str(rr_path))
    print("  Right reacher loaded.")

  # Create controller
  ctrl = G1Controller(model, data, walker, croucher, rotator, config,
                      right_reacher=right_reacher)

  # Warm up ONNX models (first call triggers JIT compilation)
  print("Warming up policies...")
  _dummy99 = np.zeros((1, 99), dtype=np.float32)
  _dummy101 = np.zeros((1, 101), dtype=np.float32)
  _dummy36 = np.zeros((1, 36), dtype=np.float32)
  walker(_dummy99)
  croucher(_dummy101)
  rotator(_dummy99)
  if right_reacher:
    right_reacher(_dummy36)
  print("  Policies warm.")

  # Camera renderer (offscreen, for head/wrist cam windows)
  cam_renderer = None
  cv2 = None
  show_head_cam = not args.no_cameras
  show_wrist_cam = not args.no_cameras
  if not args.no_cameras:
    try:
      import cv2 as _cv2
      cv2 = _cv2
      cam_renderer = CameraRenderer(model, data, 320, 240)
      # Warm up renderer (first call compiles shaders)
      cam_renderer.render("head_cam")
      cam_renderer.render("wrist_cam")
      print("  Camera renderer ready (head_cam, wrist_cam).")
    except ImportError:
      print("  [WARN] opencv-python not installed — camera windows disabled.")
      print("  Install with: pip install opencv-python")
      show_head_cam = show_wrist_cam = False
    except Exception as e:
      print(f"  [WARN] Camera renderer init failed: {e}")
      show_head_cam = show_wrist_cam = False

  # Print controls
  print(f"\n{'='*50}")
  print("G1 TABLE RED BLOCK — MuJoCo Standalone")
  print(f"{'='*50}")
  print("  .          Toggle WALK / REACH mode")
  print("  --- WALK mode ---")
  print("  Arrows     Walk fwd/back, strafe L/R")
  print("  ; / '      Turn left / right")
  print("  \\          Stop")
  print("  --- REACH mode ---")
  print("  Up/Down    Reach forward / backward")
  print("  Left/Right Reach left / right")
  print("  ; / '      Reach up / down")
  print("  \\          Reset reach target")
  print("  --- Always ---")
  print("  Space      Reset robot")
  print(f"{'='*50}\n")

  # Mutable state for key callback
  state = {"reset": False}

  def on_key(keycode: int) -> None:
    if keycode == 32:  # Space
      state["reset"] = True
    else:
      ctrl.key_callback(keycode)

  # ------------------------------------------------------------------- #
  # Simulation loop using launch_passive (MuJoCo's built-in viewer)
  # ------------------------------------------------------------------- #
  from mujoco import viewer

  decimation = 4
  control_step = 0
  target_pos = ctrl.default_joint_pos.copy()
  sim_time = 0.0
  last_cam_render = 0.0
  cam_interval = 1.0 / args.cam_fps

  print("Launching MuJoCo viewer...")

  with viewer.launch_passive(model, data, key_callback=on_key) as v:
    # Reset clock AFTER viewer opens — prevents catchup lag burst on startup
    t0 = time.time()
    while v.is_running():
      # Handle spacebar reset
      if state["reset"]:
        mujoco.mj_resetData(model, data)
        data.qpos[0] = -0.6
        data.qpos[2] = 0.76
        data.qpos[3:7] = [1, 0, 0, 0]
        for name, value in config["default_joint_pos"].items():
          if name in joint_names:
            data.qpos[7 + joint_names.index(name)] = value
        mujoco.mj_forward(model, data)
        ctrl.last_action[:] = 0
        ctrl.last_arm_action[:] = 0
        ctrl.lin_vel_x = ctrl.lin_vel_y = ctrl.ang_vel_z = 0.0
        ctrl.reach_active = False
        ctrl.last_arm_target = None
        ctrl.frozen_arm_pos = None
        ctrl.grip_closed = False
        ctrl.input_mode = "walk"
        target_pos = ctrl.default_joint_pos.copy()
        state["reset"] = False
        print("[RESET] Robot reset → WALK mode")

      # Step physics in real time (cap catchup to avoid jitter snowball)
      wall = time.time() - t0
      max_catchup = 0.05  # Never try to catch up more than 50ms per frame
      if wall - sim_time > max_catchup:
        sim_time = wall - max_catchup
      while sim_time < wall:
        if control_step % decimation == 0:
          target_pos = ctrl.step()
        ctrl.apply_pd_control(target_pos)
        mujoco.mj_step(model, data)
        control_step += 1
        sim_time += model.opt.timestep

      # Sync viewer
      v.sync()

      # Render camera views at lower FPS
      if cam_renderer and cv2 and (show_head_cam or show_wrist_cam):
        now = time.time()
        if now - last_cam_render >= cam_interval:
          last_cam_render = now
          if show_head_cam:
            img = cam_renderer.render("head_cam")
            cv2.imshow("Head Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
          if show_wrist_cam:
            img = cam_renderer.render("wrist_cam")
            cv2.imshow("Wrist Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
          cv2.waitKey(1)

  # Cleanup
  if cv2:
    try:
      cv2.destroyAllWindows()
    except Exception:
      pass
  print("Done.")


if __name__ == "__main__":
  main()
```

---

## FILE: `scene.xml`

```xml
<mujoco model="g1_table_red_block">
  <!--
    G1 Table Red Block scene — converted from LuckyEngine .hscene
    Contains: G1 robot, ground plane, table, red cylindrical block (freejoint),
              head camera, wrist camera, overhead camera, tracking camera.
  -->

  <include file="g1.xml"/>

  <statistic center="0.2 0 0.5" extent="2"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <!-- Skybox -->
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <!-- Light grey floor -->
    <texture type="2d" name="groundplane" builtin="flat"
             rgb1="0.75 0.75 0.75" rgb2="0.7 0.7 0.7"
             mark="random" markrgb="0.72 0.72 0.72"
             width="512" height="512"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="8 8" reflectance="0.1"/>
    <!-- Table materials -->
    <material name="table_wood" rgba="0.55 0.35 0.2 1" specular="0.3" shininess="0.5"/>
    <material name="table_blue" rgba="0.1 0.12 0.3 1" specular="0.3" shininess="0.5"/>
    <!-- Red block material -->
    <material name="red_plastic" rgba="0.85 0.1 0.1 1" specular="0.5" shininess="0.8"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="10 10 0.01" material="groundplane"
          contype="1" conaffinity="1"/>

    <!-- Ambient light -->
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"
           castshadow="true"/>
    <light pos="-2 2 3" dir="0.5 -0.5 -1" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>

    <!-- Table (static) -->
    <!-- Hazel: pos=[0.351, 0.713, 0], scale=[0.8, 0.04, 0.5] -->
    <!-- MuJoCo Z-up: pos=[0.351, 0, 0.713], half-size=[0.4, 0.25, 0.02] -->
    <body name="table" pos="0.351 0 0.713">
      <geom name="table_top" type="box" size="0.4 0.25 0.02" material="table_wood"
            contype="1" conaffinity="1" friction="1 0.1 0.01" mass="10"/>
      <!-- Table legs -->
      <geom name="table_leg_1" type="cylinder" size="0.025 0.345" pos="0.35 0.2 -0.365"
            rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
      <geom name="table_leg_2" type="cylinder" size="0.025 0.345" pos="-0.35 0.2 -0.365"
            rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
      <geom name="table_leg_3" type="cylinder" size="0.025 0.345" pos="0.35 -0.2 -0.365"
            rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
      <geom name="table_leg_4" type="cylinder" size="0.025 0.345" pos="-0.35 -0.2 -0.365"
            rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
    </body>

    <!-- White table (target / drop-off) — to the right of the robot -->
    <body name="table_white" pos="-0.3 -0.8 0.613">
      <geom name="table_white_top" type="box" size="0.35 0.25 0.02" material="table_blue"
            contype="1" conaffinity="1" friction="1 0.1 0.01" mass="10"/>
      <geom name="table_white_leg_1" type="cylinder" size="0.025 0.295" pos="0.3 0.2 -0.315"
            rgba="0.08 0.09 0.22 1" contype="1" conaffinity="1" mass="1"/>
      <geom name="table_white_leg_2" type="cylinder" size="0.025 0.295" pos="-0.3 0.2 -0.315"
            rgba="0.08 0.09 0.22 1" contype="1" conaffinity="1" mass="1"/>
      <geom name="table_white_leg_3" type="cylinder" size="0.025 0.295" pos="0.3 -0.2 -0.315"
            rgba="0.08 0.09 0.22 1" contype="1" conaffinity="1" mass="1"/>
      <geom name="table_white_leg_4" type="cylinder" size="0.025 0.295" pos="-0.3 -0.2 -0.315"
            rgba="0.08 0.09 0.22 1" contype="1" conaffinity="1" mass="1"/>
    </body>

    <!-- Red block / cylinder (dynamic, with freejoint) -->
    <!-- Hazel: pos=[0.235, 0.85, -0.026], scale=[0.03, 0.07, 0.03] -->
    <!-- MuJoCo Z-up: pos=[0.235, 0.026, 0.85] -->
    <body name="red_block" pos="0.0 0.026 0.85">
      <freejoint name="red_block_joint"/>
      <geom name="red_cylinder" type="cylinder" size="0.02 0.035"
            material="red_plastic" density="100"
            contype="1" conaffinity="1" friction="3 0.1 0.01"/>
      <!-- Thin box caps to reinforce top/bottom collision without tipping -->
      <geom name="red_cap_top" type="box" size="0.018 0.018 0.002" pos="0 0 0.035"
            rgba="0.85 0.1 0.1 1" density="10"
            contype="1" conaffinity="1" friction="3 0.1 0.01"/>
      <geom name="red_cap_bot" type="box" size="0.018 0.018 0.002" pos="0 0 -0.035"
            rgba="0.85 0.1 0.1 1" density="10"
            contype="1" conaffinity="1" friction="3 0.1 0.01"/>
    </body>

    <!-- Fixed cameras -->
    <camera name="overhead" pos="0.3 0 2.5" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <camera name="side_view" pos="1.5 -1.5 1.2" xyaxes="0.707 0.707 0 -0.2 0.2 0.96" fovy="50"/>
  </worldbody>
</mujoco>
```

---

## FILE: `walker.onnx`

_Skipped: non-text or binary file._

---

## FILE: `walker.onnx.data`

_Skipped: file is too large (877336 bytes)._ 

