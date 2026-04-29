#!/usr/bin/env python3
"""Headless 700-tick integration test: SETTLE → APPROACH_SOURCE → HOVER_SOURCE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.controller import WalkerReacherController
from common.onnx_policy import ONNXPolicy
from common.scene import reset_robot
from policies.fsm import FSMPolicy


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


def main():
    config_path = ROOT / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)
    joint_names = config["joint_names"]

    xml_path = ROOT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)
    reset_robot(model, data, config, joint_names, reset_data=False)

    walker       = ONNXPolicy(str(ROOT / "walker.onnx"))
    croucher     = ONNXPolicy(str(ROOT / "croucher.onnx"))
    rotator      = ONNXPolicy(str(ROOT / "rotator.onnx"))
    right_reacher = None
    rr_path = ROOT / "right_reacher.onnx"
    if rr_path.exists():
        right_reacher = ONNXPolicy(str(rr_path))

    ctrl = WalkerReacherController(
        model, data, walker, croucher, rotator, config,
        right_reacher=right_reacher,
    )
    policy = FSMPolicy(ctrl)

    # Warm up ONNX
    walker(np.zeros((1, 99), dtype=np.float32))
    croucher(np.zeros((1, 101), dtype=np.float32))
    rotator(np.zeros((1, 99), dtype=np.float32))
    if right_reacher:
        right_reacher(np.zeros((1, 36), dtype=np.float32))

    from policies.fsm_core import FSMState

    decimation = 4
    MAX_CONTROL_TICKS = 700
    target_pos = ctrl.default_joint_pos.copy()

    for control_step in range(MAX_CONTROL_TICKS):
        out = policy.step()
        ctrl.lin_vel_x, ctrl.lin_vel_y, ctrl.ang_vel_z = out.walk_cmd
        ctrl.reach_target[:] = out.reach_target
        ctrl.reach_active    = out.reach_active
        ctrl.grip_closed     = out.grip_closed
        target_pos = ctrl.step()

        for _ in range(decimation):
            ctrl.apply_pd_control(target_pos)
            mujoco.mj_step(model, data)

        if policy._fsm.state == FSMState.HOVER_SOURCE:
            cyl = policy._fsm._cylinder_in_pelvis()
            print(f"\nPASS — reached HOVER_SOURCE at control tick {control_step + 1}")
            print(f"  cyl pelvis: x={cyl[0]:.3f}  y={cyl[1]:.3f}  z={cyl[2]:.3f}")
            sys.exit(0)

        pz = float(data.qpos[2])
        if pz < 0.40:
            print(f"\nFAIL — robot fell (pelvis z={pz:.3f}) at tick {control_step + 1}")
            sys.exit(1)

    # Final report
    from policies.fsm_core import FSMCore
    cyl = policy._fsm._cylinder_in_pelvis()
    print(f"\n=== FINAL ===")
    print(f"  state:        {policy._fsm.state.name}")
    print(f"  total ticks:  {MAX_CONTROL_TICKS}")
    print(f"  cyl pelvis:   x={cyl[0]:.3f}  y={cyl[1]:.3f}  z={cyl[2]:.3f}")
    from policies.fsm_core import REACH_X_MIN, REACH_X_MAX, REACH_Y_MIN, REACH_Y_MAX
    in_win = REACH_X_MIN <= cyl[0] <= REACH_X_MAX and REACH_Y_MIN <= cyl[1] <= REACH_Y_MAX
    print(f"  in_window:    {in_win}")
    print()
    print("FAIL — did not reach HOVER_SOURCE")
    sys.exit(1)


if __name__ == "__main__":
    main()
