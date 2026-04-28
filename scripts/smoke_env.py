#!/usr/bin/env python3
"""Headless smoke test: validate scene loading and ONNX warmup.

Exits 0 on full pass, 1 on any required check failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_CAMERAS = ["head_cam", "wrist_cam", "overhead", "side_view", "tracking"]
REQUIRED_BODIES  = ["pelvis", "red_block", "table", "table_white"]
REQUIRED_SITES   = ["right_palm", "imu_in_pelvis", "left_foot", "right_foot"]

REQUIRED_ONNX = {
    "walker":        ("walker.onnx",        99),
    "right_reacher": ("right_reacher.onnx", 36),
}
OPTIONAL_ONNX = {
    "croucher": ("croucher.onnx",  101),
    "rotator":  ("rotator.onnx",    99),
}


# --------------------------------------------------------------------------- #

def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# --------------------------------------------------------------------------- #

def main() -> int:
    failures: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Config
    # ------------------------------------------------------------------ #
    section("Config")
    config_path = ROOT / "model_config.json"
    config_ok = _check("model_config.json exists", config_path.exists())
    if not config_ok:
        failures.append("model_config.json missing")
        config = {}
    else:
        with open(config_path) as f:
            config = json.load(f)
        joint_names = config.get("joint_names", [])
        _check("joint_names present", bool(joint_names), f"{len(joint_names)} joints")
        for key in ("walker", "croucher"):
            present = key in config
            _check(f"config['{key}'] block", present)
            if not present:
                failures.append(f"config missing '{key}' block")

    # ------------------------------------------------------------------ #
    # 2. MuJoCo scene
    # ------------------------------------------------------------------ #
    section("MuJoCo scene")
    try:
        import mujoco
    except ImportError as exc:
        print(f"  [FAIL] mujoco import failed: {exc}")
        failures.append("mujoco not importable")
        _print_summary(failures)
        return 1

    xml_path = ROOT / "scene.xml"
    xml_ok = _check("scene.xml exists", xml_path.exists())
    if not xml_ok:
        failures.append("scene.xml missing")
        _print_summary(failures)
        return 1

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data  = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _check("scene loaded + forward pass", True)
    except Exception as exc:
        _check("scene loaded", False, str(exc))
        failures.append(f"scene load failed: {exc}")
        _print_summary(failures)
        return 1

    # ------------------------------------------------------------------ #
    # 3. Cameras
    # ------------------------------------------------------------------ #
    section("Cameras")
    for cam in REQUIRED_CAMERAS:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        ok  = cid >= 0
        _check(f"camera '{cam}'", ok, f"id={cid}" if ok else "NOT FOUND")
        if not ok:
            failures.append(f"camera '{cam}' missing")

    # ------------------------------------------------------------------ #
    # 4. Bodies
    # ------------------------------------------------------------------ #
    section("Bodies")
    for body in REQUIRED_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        ok  = bid >= 0
        _check(f"body '{body}'", ok, f"id={bid}" if ok else "NOT FOUND")
        if not ok:
            failures.append(f"body '{body}' missing")

    # ------------------------------------------------------------------ #
    # 5. Sites
    # ------------------------------------------------------------------ #
    section("Sites")
    for site in REQUIRED_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        ok  = sid >= 0
        _check(f"site '{site}'", ok, f"id={sid}" if ok else "NOT FOUND")
        if not ok:
            failures.append(f"site '{site}' missing")

    # ------------------------------------------------------------------ #
    # 6. Joints from config
    # ------------------------------------------------------------------ #
    section("Config joints in model")
    joint_names = config.get("joint_names", [])
    missing_joints = []
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            missing_joints.append(jname)
    if missing_joints:
        _check(f"all {len(joint_names)} joints present", False,
               f"missing: {missing_joints}")
        failures.append(f"missing joints: {missing_joints}")
    else:
        _check(f"all {len(joint_names)} joints present", True)

    # ------------------------------------------------------------------ #
    # 7. ONNX — required
    # ------------------------------------------------------------------ #
    section("ONNX (required)")
    try:
        import numpy as np
        import onnxruntime as ort
        ort_ok = True
    except ImportError as exc:
        print(f"  [FAIL] onnxruntime import failed: {exc}")
        failures.append("onnxruntime not importable")
        _print_summary(failures)
        return 1

    for label, (fname, input_dim) in REQUIRED_ONNX.items():
        path = ROOT / fname
        exists = path.exists()
        _check(f"{fname} exists", exists)
        if not exists:
            failures.append(f"{fname} missing")
            continue
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                str(path), sess_opts, providers=["CPUExecutionProvider"]
            )
            in_name  = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            dummy = np.zeros((1, input_dim), dtype=np.float32)
            out   = sess.run([out_name], {in_name: dummy})[0]
            _check(
                f"{label} warmup",
                True,
                f"in={input_dim} out={out.shape[1]}",
            )
        except Exception as exc:
            _check(f"{label} warmup", False, str(exc))
            failures.append(f"{label} warmup failed: {exc}")

    # ------------------------------------------------------------------ #
    # 8. ONNX — optional (report only, never fail)
    # ------------------------------------------------------------------ #
    section("ONNX (optional)")
    for label, (fname, input_dim) in OPTIONAL_ONNX.items():
        path = ROOT / fname
        exists = path.exists()
        if not exists:
            print(f"  [INFO] {fname} not present (optional)")
            continue
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                str(path), sess_opts, providers=["CPUExecutionProvider"]
            )
            in_name  = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            dummy = np.zeros((1, input_dim), dtype=np.float32)
            out   = sess.run([out_name], {in_name: dummy})[0]
            print(f"  [OK  ] {label}: in={input_dim} out={out.shape[1]}")
        except Exception as exc:
            print(f"  [WARN] {label} load/warmup failed: {exc}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    _print_summary(failures)
    return 1 if failures else 0


def _print_summary(failures: list[str]) -> None:
    print("\n" + "=" * 50)
    if failures:
        print(f"RESULT: FAIL  ({len(failures)} issue(s))")
        for f in failures:
            print(f"  - {f}")
    else:
        print("RESULT: PASS  — repo is ready for FSM work")
    print("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
