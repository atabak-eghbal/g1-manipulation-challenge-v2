#!/usr/bin/env python3
"""Headless integration test: full pick-and-place pipeline through DONE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.controller import WalkerReacherController
from common.grasp import KinematicAttachment
from common.onnx_policy import ONNXPolicy
from common.scene import reset_robot
from policies.fsm import FSMPolicy
from policies.fsm_core import FSMState


def set_armature(model, joint_names):
    ARM_5020    = 0.00360972
    ARM_7520_14 = 0.01017752
    ARM_7520_22 = 0.02510192
    ARM_4010    = 0.00425000
    ARM_2x5020  = 0.00721945
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

    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data  = mujoco.MjData(model)
    reset_robot(model, data, config, joint_names, reset_data=False)

    walker        = ONNXPolicy(str(ROOT / "walker.onnx"))
    croucher      = ONNXPolicy(str(ROOT / "croucher.onnx"))
    rotator       = ONNXPolicy(str(ROOT / "rotator.onnx"))
    right_reacher = ONNXPolicy(str(ROOT / "right_reacher.onnx"))

    ctrl = WalkerReacherController(model, data, walker, croucher, rotator,
                                   config, right_reacher=right_reacher)

    rb_body_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    grasp_backend = KinematicAttachment(model, data, ctrl.right_palm_site_id, rb_body_id)
    policy        = FSMPolicy(ctrl, grasp_backend=grasp_backend)

    # Warm up ONNX
    walker(np.zeros((1, 99), dtype=np.float32))
    croucher(np.zeros((1, 101), dtype=np.float32))
    rotator(np.zeros((1, 99), dtype=np.float32))
    right_reacher(np.zeros((1, 36), dtype=np.float32))

    decimation     = 4
    MAX_CTRL_TICKS = 3000   # ~60 s: full pipeline through DONE

    target_pos = ctrl.default_joint_pos.copy()

    for tick in range(MAX_CTRL_TICKS):
        out = policy.step()
        target_pos = ctrl.step()
        for _ in range(decimation):
            ctrl.apply_pd_control(target_pos)
            mujoco.mj_step(model, data)
            grasp_backend.tick(ctrl.grip_closed)

        state = policy._fsm.state

        if state == FSMState.DONE:
            fsm   = policy._fsm
            cyl   = fsm._cylinder_world()
            palm  = fsm._palm_world()
            drop  = fsm._target_drop_pt
            tgt_z = fsm._target_surface_z()
            on_tbl = fsm._cylinder_on_target_table()
            verdict = "PASS" if on_tbl else "FAIL"
            print(f"\n{verdict} — reached DONE at control tick {tick + 1}")
            print(f"  palm_world      : ({palm[0]:.3f}, {palm[1]:.3f}, {palm[2]:.3f})")
            print(f"  cyl_world       : ({cyl[0]:.3f}, {cyl[1]:.3f}, {cyl[2]:.3f})")
            print(f"  drop_world      : ({drop[0]:.3f}, {drop[1]:.3f}, {drop[2]:.3f})")
            print(f"  target_z        : {tgt_z:.3f}")
            print(f"  cyl_clearance   : {cyl[2]-tgt_z:.3f} m")
            print(f"  on_target_table : {on_tbl}")
            print(f"  attached        : {grasp_backend.attached}")
            sys.exit(0 if on_tbl else 1)

        pz = float(data.qpos[2])
        if pz < 0.40:
            print(f"\nFAIL — robot fell (pelvis z={pz:.3f}) at tick {tick + 1}")
            sys.exit(1)

    # Timeout
    fsm   = policy._fsm
    cyl_w = fsm._cylinder_world()
    drop  = fsm._target_drop_pt
    drop_str = f"({drop[0]:.3f},{drop[1]:.3f},{drop[2]:.3f})" if drop is not None else "None"
    print(f"\n=== TIMEOUT after {MAX_CTRL_TICKS} ticks ===")
    print(f"  state           : {fsm.state.name}")
    print(f"  cyl_world       : ({cyl_w[0]:.3f}, {cyl_w[1]:.3f}, {cyl_w[2]:.3f})")
    print(f"  drop_world      : {drop_str}")
    print(f"  on_target_table : {fsm._cylinder_on_target_table()}")
    print(f"  attached        : {grasp_backend.attached}")
    print("\nFAIL — did not reach DONE")
    sys.exit(1)


if __name__ == "__main__":
    main()
