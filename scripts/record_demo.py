#!/usr/bin/env python3
"""Records a video of the G1 pick-and-place demo using the FSM policy."""

import json
import time
from pathlib import Path
import mujoco
import numpy as np
import cv2

from common.controller import WalkerReacherController
from common.grasp import KinematicAttachment
from common.onnx_policy import ONNXPolicy
from common.scene import CameraRenderer, reset_robot
from policies.fsm import FSMPolicy

SCRIPT_DIR = Path(__file__).resolve().parent.parent

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
    # Load config
    config_path = SCRIPT_DIR / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)
    joint_names = config["joint_names"]

    # Load scene
    xml_path = SCRIPT_DIR / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Reset
    reset_robot(model, data, config, joint_names, reset_data=True)

    # Load policies
    walker = ONNXPolicy(str(SCRIPT_DIR / "walker.onnx"))
    croucher = ONNXPolicy(str(SCRIPT_DIR / "croucher.onnx"))
    rotator = ONNXPolicy(str(SCRIPT_DIR / "rotator.onnx"))
    right_reacher = ONNXPolicy(str(SCRIPT_DIR / "right_reacher.onnx"))

    # Create controller
    ctrl = WalkerReacherController(model, data, walker, croucher, rotator, config,
                                   right_reacher=right_reacher)

    rb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    grasp_backend = KinematicAttachment(model, data, ctrl.right_palm_site_id, rb_body_id)
    policy = FSMPolicy(ctrl, grasp_backend=grasp_backend)

    # Setup VideoWriter
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = SCRIPT_DIR / "demo_video.mp4"
    video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    renderer = mujoco.Renderer(model, height, width)

    print(f"Recording demo to {out_path}...")

    decimation = 4
    sim_steps = 0
    max_steps = 15000 # Enough to finish the task
    
    last_render_time = 0
    render_interval = 1.0 / fps

    while sim_steps < max_steps:
        if sim_steps % decimation == 0:
            p_out = policy.step()
            ctrl.lin_vel_x, ctrl.lin_vel_y, ctrl.ang_vel_z = p_out.walk_cmd
            ctrl.reach_target[:] = p_out.reach_target
            ctrl.reach_active = p_out.reach_active
            ctrl.grip_closed = p_out.grip_closed
            target_pos = ctrl.step()
        
        ctrl.apply_pd_control(target_pos)
        mujoco.mj_step(model, data)
        grasp_backend.tick(ctrl.grip_closed)

        # Render at 30 FPS
        current_sim_time = data.time
        if current_sim_time - last_render_time >= render_interval:
            renderer.update_scene(data, camera="side_view")
            frame = renderer.render()
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            last_render_time = current_sim_time
        
        sim_steps += 1
        
        # Stop if FSM reaches DONE
        if hasattr(policy, '_fsm') and policy._fsm.state.name == "DONE":
            # Record a few more seconds of the DONE state
            for _ in range(int(2.0 / model.opt.timestep)):
                if sim_steps % decimation == 0:
                    ctrl.step()
                ctrl.apply_pd_control(target_pos)
                mujoco.mj_step(model, data)
                grasp_backend.tick(ctrl.grip_closed)
                
                if data.time - last_render_time >= render_interval:
                    renderer.update_scene(data, camera="side_view")
                    frame = renderer.render()
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    last_render_time = data.time
                sim_steps += 1
            break

    video_writer.release()
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    main()
