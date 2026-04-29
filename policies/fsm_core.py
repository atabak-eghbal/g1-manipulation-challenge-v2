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
APPROACH_TARGET_X = 0.34   # m forward — cylinder at reach when this close

REACH_X_MIN, REACH_X_MAX = 0.20, 0.38   # x reachability window
REACH_Y_MIN, REACH_Y_MAX = -0.14, 0.02  # y reachability window

REACH_DEBOUNCE = 8   # consecutive in-window ticks before APPROACH → HOVER

VX_FAST, VX_MED, VX_SLOW = 0.35, 0.22, 0.12   # staircase vx (m/s)
K_VY,  VY_CAP  = 1.8, 0.18   # vy: proportional toward y = -0.05
K_WZ,  WZ_CAP  = 1.2, 0.25   # wz: arctan2-based yaw

# ---- Hover and grasp heights above table surface ----
HOVER_SOURCE_HEIGHT = 0.18   # m: pre-grasp hover above table top
GRASP_HEIGHT        = 0.06   # m: cylinder mid-body height above table top

# ---- Palm-to-target distance thresholds ----
# The reacher has an ~12 cm accuracy floor; thresholds must stay ≥ this.
HOVER_SOURCE_THRESHOLD   = 0.14   # m
DESCEND_SOURCE_THRESHOLD = 0.12   # m

# ---- Per-state timeouts (control ticks at 50 Hz) ----
HOVER_SOURCE_TIMEOUT   = 200   # ~4 s fallback if threshold never met
DESCEND_SOURCE_TIMEOUT = 300   # ~6 s fallback
CLOSE_GRIP_TIMEOUT     = 100   # ~2 s: advance to LIFT even if not yet attached
LIFT_SOURCE_TIMEOUT    = 200   # ~4 s: declare done if arm never clears table

# ---- General reach-state debounce ----
DEBOUNCE_REACH = 6   # consecutive ticks palm must be within threshold

# ---- Lift success criterion ----
LIFT_DONE_CLEARANCE = 0.25  # m above table top — cylinder visibly off the surface

# ---- Reacher workspace bounds in pelvis frame ----
# From the training spec; targets outside are clamped before being sent.
_REACH_LOW  = np.array([-0.30, -0.60, -0.40], dtype=np.float32)
_REACH_HIGH = np.array([ 0.60,  0.30,  0.60], dtype=np.float32)


# --------------------------------------------------------------------------- #
# State enumeration
# --------------------------------------------------------------------------- #

class FSMState(Enum):
    SETTLE          = auto()
    APPROACH_SOURCE = auto()
    HOVER_SOURCE    = auto()
    DESCEND_SOURCE  = auto()
    CLOSE_GRIP      = auto()   # close fingers; wait for backend to confirm attach
    LIFT_SOURCE     = auto()   # raise arm to carry pose; confirm cylinder left table
    DONE            = auto()


# --------------------------------------------------------------------------- #
# Core machine
# --------------------------------------------------------------------------- #

class FSMCore:
    """Tick-driven state machine that emits a high-level PolicyOutput each step.

    Holds references to MuJoCo model/data for GT geometry; never modifies them.
    """

    def __init__(self, model, data) -> None:
        self._model = model
        self._data  = data

        # ---- MuJoCo ID cache ----
        self._rb_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        self._tbl_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self._palm_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        # Geom-based table surface is more accurate than body-centre + offset.
        self._tbl_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

        self.state        = FSMState.SETTLE
        self._tick_total  = 0
        self._tick_state  = 0
        self._reach_count = 0   # general-purpose debounce counter
        self._attached    = False  # updated each tick by FSMPolicy from grasp backend

        print(
            f"[FSM] init  state={self.state.name}"
            f"  rb={self._rb_id}  tbl={self._tbl_id}"
            f"  palm={self._palm_id}  tbl_geom={self._tbl_geom_id}"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tick(self, attached: bool = False) -> PolicyOutput:
        """Advance the FSM by one control tick.

        attached: True when the grasp backend reports the cylinder is welded to
                  the palm this tick.  Passed in by FSMPolicy so FSMCore stays
                  free of any dependency on the backend.
        """
        self._attached = attached
        out = self._dispatch()
        self._tick_total += 1
        self._tick_state += 1
        return out

    # ------------------------------------------------------------------ #
    # Dispatch + transition
    # ------------------------------------------------------------------ #

    def _dispatch(self) -> PolicyOutput:
        if self.state == FSMState.SETTLE:          return self._settle()
        if self.state == FSMState.APPROACH_SOURCE: return self._approach_source()
        if self.state == FSMState.HOVER_SOURCE:    return self._hover_source()
        if self.state == FSMState.DESCEND_SOURCE:  return self._descend_source()
        if self.state == FSMState.CLOSE_GRIP:      return self._close_grip()
        if self.state == FSMState.LIFT_SOURCE:     return self._lift_source()
        return self._done()

    def _transition(self, new: FSMState) -> None:
        print(f"[FSM] {self.state.name} → {new.name}  (t={self._tick_total})")
        # Log world geometry at the moment of entry so the log is self-contained.
        if new == FSMState.HOVER_SOURCE:
            hover = self._source_hover_world()
            tbl_z = self._table_surface_z()
            dist  = float(np.linalg.norm(self._palm_world() - hover))
            print(f"[FSM]   hover_world=({hover[0]:.3f},{hover[1]:.3f},{hover[2]:.3f})"
                  f"  table_z={tbl_z:.4f}  entry_palm_dist={dist:.3f}")
        elif new == FSMState.DESCEND_SOURCE:
            grasp = self._source_grasp_world()
            dist  = float(np.linalg.norm(self._palm_world() - grasp))
            print(f"[FSM]   grasp_world=({grasp[0]:.3f},{grasp[1]:.3f},{grasp[2]:.3f})"
                  f"  entry_palm_dist={dist:.3f}")
        elif new == FSMState.CLOSE_GRIP:
            dist = float(np.linalg.norm(self._palm_world() - self._cylinder_world()))
            print(f"[FSM]   palm_to_cyl={dist:.3f} m")
        elif new == FSMState.LIFT_SOURCE:
            palm = self._palm_world()
            cyl  = self._cylinder_world()
            print(f"[FSM]   palm_z={palm[2]:.3f}  cyl_z={cyl[2]:.3f}  attached={self._attached}")
        elif new == FSMState.DONE:
            cyl  = self._cylinder_world()
            tbl_z = self._table_surface_z()
            print(f"[FSM]   cyl_z={cyl[2]:.3f}  table_z={tbl_z:.3f}"
                  f"  clearance={cyl[2] - tbl_z:.3f} m")
        self.state        = new
        self._tick_state  = 0
        self._reach_count = 0   # reset debounce for the new state

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
            print(f"[FSM] cylinder in reach window: "
                  f"pelvis_frame=({cyl[0]:.3f},{cyl[1]:.3f},{cyl[2]:.3f})")
            self._transition(FSMState.HOVER_SOURCE)
        return PolicyOutput(
            walk_cmd=walk_cmd,
            reach_target=CARRY_POSE,
            reach_active=False,
            grip_closed=False,
        )

    def _hover_source(self) -> PolicyOutput:
        hover = self._source_hover_world()
        reach = self._reach_from_world(hover, right_bias=-0.03)
        palm  = self._palm_world()
        dist  = float(np.linalg.norm(palm - hover))

        if dist < HOVER_SOURCE_THRESHOLD:
            self._reach_count += 1
        else:
            self._reach_count = 0

        if self._reach_count >= DEBOUNCE_REACH:
            print(f"[FSM] HOVER_SOURCE → threshold met  palm_dist={dist:.3f}")
            self._transition(FSMState.DESCEND_SOURCE)
        elif self._tick_state >= HOVER_SOURCE_TIMEOUT:
            print(f"[FSM] HOVER_SOURCE → timeout  palm_dist={dist:.3f}")
            self._transition(FSMState.DESCEND_SOURCE)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=reach,
            reach_active=True,
            grip_closed=False,
        )

    def _descend_source(self) -> PolicyOutput:
        grasp = self._source_grasp_world()
        reach = self._reach_from_world(grasp, right_bias=-0.03)
        palm  = self._palm_world()
        dist  = float(np.linalg.norm(palm - grasp))

        if dist < DESCEND_SOURCE_THRESHOLD:
            self._reach_count += 1
        else:
            self._reach_count = 0

        if self._reach_count >= DEBOUNCE_REACH:
            print(f"[FSM] DESCEND_SOURCE → threshold met  palm_dist={dist:.3f}")
            self._transition(FSMState.CLOSE_GRIP)
        elif self._tick_state >= DESCEND_SOURCE_TIMEOUT:
            print(f"[FSM] DESCEND_SOURCE → timeout  palm_dist={dist:.3f}")
            self._transition(FSMState.CLOSE_GRIP)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=reach,
            reach_active=True,
            grip_closed=False,
        )

    def _close_grip(self) -> PolicyOutput:
        grasp = self._source_grasp_world()
        reach = self._reach_from_world(grasp, right_bias=-0.03)

        if self._attached:
            print(f"[FSM] CLOSE_GRIP → attached at t={self._tick_total}")
            self._transition(FSMState.LIFT_SOURCE)
        elif self._tick_state >= CLOSE_GRIP_TIMEOUT:
            print(f"[FSM] CLOSE_GRIP → timeout (not attached)  t={self._tick_total}")
            self._transition(FSMState.LIFT_SOURCE)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=reach,
            reach_active=True,
            grip_closed=True,
        )

    def _lift_source(self) -> PolicyOutput:
        palm  = self._palm_world()
        tbl_z = self._table_surface_z()

        if palm[2] >= tbl_z + LIFT_DONE_CLEARANCE:
            print(f"[FSM] LIFT_SOURCE → done"
                  f"  palm_z={palm[2]:.3f}  clearance={palm[2] - tbl_z:.3f}")
            self._transition(FSMState.DONE)
        elif self._tick_state >= LIFT_SOURCE_TIMEOUT:
            print(f"[FSM] LIFT_SOURCE → timeout  palm_z={palm[2]:.3f}")
            self._transition(FSMState.DONE)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=True,
            grip_closed=True,
        )

    def _done(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] DONE  task complete — holding carry pose")
        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=True,
            grip_closed=self._attached,
        )

    # ------------------------------------------------------------------ #
    # GT geometry helpers
    # ------------------------------------------------------------------ #

    def _pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._data.qpos[:3].copy(), self._data.qpos[3:7].copy()

    @staticmethod
    def _world_to_pelvis(
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        vec_world: np.ndarray,
    ) -> np.ndarray:
        """Rotate world-frame point into pelvis frame: q⁻¹(v − p)."""
        v   = vec_world - pelvis_pos
        w   = pelvis_quat[0]
        xyz = pelvis_quat[1:4]
        t   = np.cross(xyz, v) * 2.0
        return v - w * t + np.cross(xyz, t)

    def _cylinder_world(self) -> np.ndarray:
        return self._data.xpos[self._rb_id].copy()

    def _cylinder_in_pelvis(self) -> np.ndarray:
        pos, quat = self._pelvis_pose()
        return self._world_to_pelvis(pos, quat, self._cylinder_world())

    def _palm_world(self) -> np.ndarray:
        return self._data.site_xpos[self._palm_id].copy()

    def _table_surface_z(self) -> float:
        """World Z of the source table's top surface.

        Uses geom_xpos + geom half-height when the geom ID is available
        (more accurate than body centre + hardcoded offset).
        Falls back to body approach if geom lookup failed.
        """
        if self._tbl_geom_id >= 0:
            return float(
                self._data.geom_xpos[self._tbl_geom_id][2]
                + self._model.geom_size[self._tbl_geom_id][2]
            )
        return float(self._data.xpos[self._tbl_id][2]) + 0.02

    def _source_hover_world(self) -> np.ndarray:
        """World point above the cylinder for pre-grasp hover."""
        p = self._cylinder_world().copy()
        p[2] = self._table_surface_z() + HOVER_SOURCE_HEIGHT
        return p

    def _source_grasp_world(self) -> np.ndarray:
        """World point at cylinder mid-body height for grasping."""
        p = self._cylinder_world().copy()
        p[2] = self._table_surface_z() + GRASP_HEIGHT
        return p

    @staticmethod
    def _clip_reach_target(reach: np.ndarray) -> np.ndarray:
        """Clip a pelvis-frame reach target to the reacher's workspace."""
        return np.clip(reach, _REACH_LOW, _REACH_HIGH).astype(np.float32)

    def _reach_from_world(
        self, world_point: np.ndarray, right_bias: float = -0.08
    ) -> np.ndarray:
        """Convert world point → clipped pelvis-frame reach target.

        right_bias clamps y so the target is at least this far to the
        robot's right (y ≤ right_bias in pelvis frame), keeping the reach
        target inside the right arm's natural workspace.
        """
        pos, quat = self._pelvis_pose()
        local = self._world_to_pelvis(pos, quat, world_point).copy().astype(np.float32)
        local[1] = min(float(local[1]), right_bias)
        return self._clip_reach_target(local)

    # ------------------------------------------------------------------ #
    # Approach commander
    # ------------------------------------------------------------------ #

    def _approach_walk_cmd(self, cyl: np.ndarray) -> tuple[float, float, float]:
        """Staircase vx + proportional vy/wz toward cylinder."""
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
        return (REACH_X_MIN < cyl[0] < REACH_X_MAX and
                REACH_Y_MIN < cyl[1] < REACH_Y_MAX)
