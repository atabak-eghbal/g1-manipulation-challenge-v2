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
from pathlib import Path

import mujoco
import numpy as np

from common.controller import WalkerReacherController
from common.onnx_policy import ONNXPolicy
from common.scene import CameraRenderer, reset_robot
from policies.keyboard import KeyboardPolicy

SCRIPT_DIR = Path(__file__).resolve().parent


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
  parser.add_argument(
    "--policy",
    choices=["keyboard"],
    default="keyboard",
    help="Control policy to use (default: keyboard)",
  )
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
  reset_robot(model, data, config, joint_names, reset_data=False)

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
  ctrl = WalkerReacherController(model, data, walker, config,
                                 right_reacher=right_reacher)
  policy = KeyboardPolicy(ctrl)

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
      policy.handle_key(keycode)

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
        reset_robot(model, data, config, joint_names)
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
