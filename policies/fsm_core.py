"""Pure FSM state machine — no controller dependency."""

from __future__ import annotations

from enum import Enum, auto

import mujoco
import numpy as np

from .base import PolicyOutput

# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #

# Safe carry-pose: right arm held clear of the legs while walking.
CARRY_POSE: tuple[float, float, float] = (0.3, -0.2, 0.2)

# Ticks in SETTLE before beginning autonomous task (~3 s at 50 Hz).
SETTLE_TICKS = 150

# ---- Approach: staircase forward speeds ----
# Target x=0.34 m (within arm reach). Steps slow the robot down as it closes
# in so it doesn't overshoot the reachability window.
APPROACH_TARGET_X = 0.34   # m forward — cylinder centre-of-gravity at reach

# ---- Reachability window for APPROACH → HOVER transition ----
# Validated window from reference solution. y target is -0.05 (right-of-centre
# for the right arm). Window is intentionally narrow so we only transition when
# genuinely in reach.
REACH_X_MIN, REACH_X_MAX = 0.20, 0.38
REACH_Y_MIN, REACH_Y_MAX = -0.14, 0.02

# Consecutive in-window ticks before transition fires (~0.16 s at 50 Hz).
REACH_DEBOUNCE = 8

# ---- Staircase forward speeds (m/s) — see _approach_walk_cmd ----
VX_FAST   = 0.35   # x_err > 0.18 m
VX_MED    = 0.22   # x_err > 0.10 m
VX_SLOW   = 0.12   # x_err > 0.04 m

# ---- Lateral (vy) proportional gain toward y = -0.05 ----
K_VY      = 1.8
VY_CAP    = 0.18

# ---- Yaw: arctan2-based, pointing toward cylinder ----
K_WZ      = 1.2
WZ_CAP    = 0.25


# --------------------------------------------------------------------------- #
# State enumeration
# --------------------------------------------------------------------------- #

class FSMState(Enum):
    SETTLE          = auto()
    APPROACH_SOURCE = auto()
    HOVER_SOURCE    = auto()
    GRASP           = auto()   # stub
    TRANSPORT       = auto()   # stub
    PLACE           = auto()   # stub
    DONE            = auto()


# --------------------------------------------------------------------------- #
# Core machine
# --------------------------------------------------------------------------- #

class FSMCore:
    """Tick-driven state machine that emits a high-level PolicyOutput each step.

    Requires access to ``model`` and ``data`` for GT geometry queries; these
    are passed at construction time and never modified here.
    """

    def __init__(self, model, data) -> None:
        self._model = model
        self._data  = data

        # Cache MuJoCo body IDs used every tick.
        self._rb_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        self._tbl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")

        self.state        = FSMState.SETTLE
        self._tick_total  = 0
        self._tick_state  = 0
        self._reach_count = 0   # debounce counter for APPROACH → HOVER

        print(f"[FSM] init  state={self.state.name}  "
              f"red_block_id={self._rb_id}  table_id={self._tbl_id}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tick(self) -> PolicyOutput:
        """Advance the FSM one control step and return the current command."""
        out = self._dispatch()
        self._tick_total += 1
        self._tick_state += 1
        return out

    # ------------------------------------------------------------------ #
    # Dispatch and transition helper
    # ------------------------------------------------------------------ #

    def _dispatch(self) -> PolicyOutput:
        if self.state == FSMState.SETTLE:
            return self._settle()
        if self.state == FSMState.APPROACH_SOURCE:
            return self._approach_source()
        if self.state == FSMState.HOVER_SOURCE:
            return self._hover_source()
        if self.state == FSMState.DONE:
            return self._done()
        return self._settle()   # unreachable guard

    def _transition(self, new: FSMState) -> None:
        print(f"[FSM] {self.state.name} → {new.name}  (t={self._tick_total})")
        self.state       = new
        self._tick_state = 0

    # ------------------------------------------------------------------ #
    # State handlers
    # ------------------------------------------------------------------ #

    def _settle(self) -> PolicyOutput:
        if self._tick_state == 0:
            print(f"[FSM] SETTLE  holding {SETTLE_TICKS} ticks "
                  f"(~{SETTLE_TICKS / 50:.0f} s) before approach")
        if self._tick_state >= SETTLE_TICKS:
            self._transition(FSMState.APPROACH_SOURCE)
        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=False,
            grip_closed=False,
        )

    def _approach_source(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] APPROACH_SOURCE  walking toward red cylinder")

        cyl = self._cylinder_in_pelvis()

        if self._in_reach_window(cyl):
            self._reach_count += 1
            walk_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        else:
            self._reach_count = 0
            walk_cmd = self._approach_walk_cmd(cyl)

        if self._reach_count >= REACH_DEBOUNCE:
            cyl_str = f"({cyl[0]:.3f}, {cyl[1]:.3f}, {cyl[2]:.3f})"
            print(f"[FSM] cylinder in reach window: pelvis_frame={cyl_str}")
            self._transition(FSMState.HOVER_SOURCE)

        return PolicyOutput(
            walk_cmd=walk_cmd,
            reach_target=CARRY_POSE,
            reach_active=False,
            grip_closed=False,
        )

    def _hover_source(self) -> PolicyOutput:
        if self._tick_state == 0:
            cyl   = self._cylinder_in_pelvis()
            tbl_z = self._table_surface_z()
            print(
                f"[FSM] HOVER_SOURCE  "
                f"cyl_pelvis=({cyl[0]:.3f},{cyl[1]:.3f},{cyl[2]:.3f})"
                f"  table_surface_z={tbl_z:.4f}"
                f"  (arm descent in next step)"
            )
        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=False,
            grip_closed=False,
        )

    def _done(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] DONE  task complete — holding position")
        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=False,
            grip_closed=False,
        )

    # ------------------------------------------------------------------ #
    # GT geometry helpers
    # ------------------------------------------------------------------ #

    def _pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion) of the pelvis in world frame."""
        return self._data.qpos[:3].copy(), self._data.qpos[3:7].copy()

    @staticmethod
    def _world_to_pelvis(
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        vec_world: np.ndarray,
    ) -> np.ndarray:
        """Express a world-frame point in the pelvis body frame.

        Implements the passive rotation  q^{-1} * (v - p)  via the
        Rodrigues-style formula used throughout the controller.
        """
        v   = vec_world - pelvis_pos
        w   = pelvis_quat[0]
        xyz = pelvis_quat[1:4]
        t   = np.cross(xyz, v) * 2.0
        return v - w * t + np.cross(xyz, t)

    def _cylinder_world(self) -> np.ndarray:
        """World position of the red-block body origin (live from MuJoCo)."""
        return self._data.xpos[self._rb_id].copy()

    def _cylinder_in_pelvis(self) -> np.ndarray:
        """Cylinder position expressed in the pelvis body frame."""
        pos, quat = self._pelvis_pose()
        return self._world_to_pelvis(pos, quat, self._cylinder_world())

    def _table_surface_z(self) -> float:
        """World Z of the source table's top surface (live body position + half-height)."""
        body_z = float(self._data.xpos[self._tbl_id][2])
        # table_top geom is a box with half-size Z = 0.02 m.
        return body_z + 0.02

    # ------------------------------------------------------------------ #
    # Approach commander
    # ------------------------------------------------------------------ #

    def _approach_walk_cmd(self, cyl: np.ndarray) -> tuple[float, float, float]:
        """Staircase walk command: forward speed based on x-distance, plus vy/yaw.

        vx  — stepped (fast/med/slow/stop) as x_err closes.
        vy  — proportional toward y = -0.05 (right-arm sweet spot).
        wz  — arctan2-based yaw to face cylinder, bounded.
        """
        x_err = cyl[0] - APPROACH_TARGET_X
        if x_err > 0.18:
            vx = VX_FAST
        elif x_err > 0.10:
            vx = VX_MED
        elif x_err > 0.04:
            vx = VX_SLOW
        else:
            vx = 0.0

        y_err = cyl[1] - (-0.05)
        vy = float(np.clip(K_VY * y_err, -VY_CAP, VY_CAP))

        wz = float(np.clip(
            K_WZ * np.arctan2(cyl[1], max(cyl[0], 0.15)),
            -WZ_CAP, WZ_CAP,
        ))
        return (vx, vy, wz)

    def _in_reach_window(self, cyl: np.ndarray) -> bool:
        """True when the cylinder is inside the 2-D reachability box (x, y only)."""
        return (REACH_X_MIN < cyl[0] < REACH_X_MAX and
                REACH_Y_MIN < cyl[1] < REACH_Y_MAX)

    # ------------------------------------------------------------------ #
    # Stubs for future phases
    # ------------------------------------------------------------------ #

    def _grasp_reach_target(self) -> tuple[float, float, float]:
        """Right-arm reach target in pelvis frame for grasp pose (Step 7)."""
        raise NotImplementedError("_grasp_reach_target: implement in Step 7")

    def _target_pos(self) -> np.ndarray:
        """World position of the drop zone / white table (Step 8)."""
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "table_white")
        return self._data.xpos[bid].copy()
