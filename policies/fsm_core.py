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

# ---- Target table approach ----
# Drop point is 5 cm inside the near edge of the target table (y-direction).
# Near edge = table_white geom_center_y + half_size_y (less-negative y = robot side).
TARGET_NEAR_EDGE_INSET  = 0.05   # m inward from near edge
TARGET_REACH_DEBOUNCE   = 8      # consecutive in-window ticks → HOVER_TARGET
TARGET_APPROACH_TIMEOUT = 900    # ~18 s fallback (large turn required from source side)

# ---- Target table placement ----
HOVER_TARGET_HEIGHT    = 0.18   # m above target surface for pre-place hover
PLACE_HEIGHT           = 0.06   # m above target surface for release
HOVER_TARGET_THRESHOLD = 0.14   # m palm-to-hover-point (same floor as source)
LOWER_TARGET_THRESHOLD = 0.14   # m palm-to-place-point
HOVER_TARGET_TIMEOUT   = 200    # ~4 s
LOWER_TARGET_TIMEOUT   = 300    # ~6 s
OPEN_GRIP_TIMEOUT      = 100    # ~2 s: wait for kinematic release
RETRACT_TIMEOUT        = 200    # ~4 s: arm clear of target table

# Table-membership margins for _cylinder_on_target_table().
ON_TABLE_XY_MARGIN = 0.05   # m: allow up to 5 cm outside geom footprint
ON_TABLE_Z_MAX     = 0.20   # m above surface: cap for height sanity check

# Phase 1 of target approach: turn CW to face -y before driving toward standing waypoint.
# Use VX_SLOW so the walker actually responds to the command (vx=0.06 is below effective
# minimum, giving near-zero turn rate).  WZ_P1 = 1.0 (vel_max_angular) creates a tight
# turning circle: R = VX_P1 / WZ_P1 ≈ 0.12 m — small enough to land in the reach window.
VX_P1 = 0.12   # m/s: minimum effective forward speed for the walker
WZ_P1 = 1.0    # rad/s: full angular rate so the robot actually turns
PHASE1_ALIGN_TOL = 0.40   # rad: exit Phase 1 when |yaw − (−π/2)| < this

# World-frame proximity to the standing waypoint required before HOVER_TARGET.
# Replaces the pelvis-frame reach-window check, which fires too early mid-turn
# (the drop point can project into the window at wrong orientations).
TARGET_APPROACH_DIST_THRESH = 0.20   # m: pelvis must be within this of the waypoint

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
    APPROACH_TARGET = auto()   # walk/turn toward target-table placement corridor
    HOVER_TARGET    = auto()   # move arm above drop point, stop walking
    LOWER_TARGET    = auto()   # descend arm to release height
    OPEN_GRIP       = auto()   # open fingers; wait for kinematic detach
    RETRACT         = auto()   # raise arm to carry pose
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
        self._rb_id            = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        self._tbl_id           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self._palm_id          = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        self._tbl_geom_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self._tbl_white_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_white")
        self._tbl_white_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_white_top")

        self.state             = FSMState.SETTLE
        self._tick_total       = 0
        self._tick_state       = 0
        self._reach_count      = 0     # general-purpose debounce counter
        self._attached         = False  # updated each tick by FSMPolicy from grasp backend
        self._target_drop_pt: np.ndarray | None = None  # frozen when APPROACH_TARGET begins

        print(
            f"[FSM] init  state={self.state.name}"
            f"  rb={self._rb_id}  tbl={self._tbl_id}"
            f"  palm={self._palm_id}  tbl_geom={self._tbl_geom_id}"
            f"  tbl_white={self._tbl_white_id}  tbl_white_geom={self._tbl_white_geom_id}"
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
        if self.state == FSMState.APPROACH_TARGET: return self._approach_target()
        if self.state == FSMState.HOVER_TARGET:    return self._hover_target()
        if self.state == FSMState.LOWER_TARGET:    return self._lower_target()
        if self.state == FSMState.OPEN_GRIP:       return self._open_grip()
        if self.state == FSMState.RETRACT:         return self._retract()
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
        elif new == FSMState.APPROACH_TARGET:
            # Freeze the drop point in world frame at the moment of state entry.
            self._target_drop_pt = self._target_drop_world()
            tgt_z = self._target_surface_z()
            p = self._target_drop_pt
            dist = float(np.linalg.norm(self._palm_world() - p))
            ppos = self._data.qpos[:3]
            yaw  = self._pelvis_yaw()
            print(f"[FSM]   drop_world=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                  f"  target_z={tgt_z:.4f}  palm_dist={dist:.3f}"
                  f"  pelvis=({ppos[0]:.3f},{ppos[1]:.3f})  yaw={yaw:.3f}")
        elif new == FSMState.HOVER_TARGET:
            p    = self._target_drop_in_pelvis()
            palm = self._palm_world()
            print(f"[FSM]   drop_pelvis=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                  f"  palm_z={palm[2]:.3f}")
        elif new == FSMState.LOWER_TARGET:
            hover = self._target_hover_world()
            palm  = self._palm_world()
            print(f"[FSM]   palm_dist_to_hover={np.linalg.norm(palm-hover):.3f}")
        elif new == FSMState.OPEN_GRIP:
            palm  = self._palm_world()
            cyl   = self._cylinder_world()
            tgt_z = self._target_surface_z()
            print(f"[FSM]   palm_z={palm[2]:.3f}  cyl_z={cyl[2]:.3f}"
                  f"  height_above_target={cyl[2]-tgt_z:.3f}")
        elif new == FSMState.RETRACT:
            cyl      = self._cylinder_world()
            tgt_z    = self._target_surface_z()
            on_table = self._cylinder_on_target_table()
            print(f"[FSM]   cyl_z={cyl[2]:.3f}  on_target_table={on_table}")
        elif new == FSMState.DONE:
            cyl      = self._cylinder_world()
            tgt_z    = self._target_surface_z()
            on_table = self._cylinder_on_target_table()
            print(f"[FSM]   cyl_z={cyl[2]:.3f}  target_z={tgt_z:.3f}"
                  f"  clearance={cyl[2]-tgt_z:.3f}  on_target_table={on_table}")
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
            print(f"[FSM] LIFT_SOURCE → approach target"
                  f"  palm_z={palm[2]:.3f}  clearance={palm[2] - tbl_z:.3f}")
            self._transition(FSMState.APPROACH_TARGET)
        elif self._tick_state >= LIFT_SOURCE_TIMEOUT:
            print(f"[FSM] LIFT_SOURCE → timeout → approach target  palm_z={palm[2]:.3f}")
            self._transition(FSMState.APPROACH_TARGET)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=True,
            grip_closed=True,
        )

    def _approach_target(self) -> PolicyOutput:
        if self._tick_state == 1:
            print("[FSM] APPROACH_TARGET  walking toward target table")
        drop = self._target_drop_in_pelvis()
        if self._near_target_waypoint():
            self._reach_count += 1
            walk_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        else:
            self._reach_count = 0
            walk_cmd = self._target_approach_walk_cmd(drop)
        if self._reach_count >= TARGET_REACH_DEBOUNCE:
            ppos = self._data.qpos[:3]
            yaw  = self._pelvis_yaw()
            print(f"[FSM] near target waypoint: "
                  f"pelvis=({ppos[0]:.3f},{ppos[1]:.3f})  yaw={yaw:.3f}  "
                  f"drop_pelvis=({drop[0]:.3f},{drop[1]:.3f},{drop[2]:.3f})")
            self._transition(FSMState.HOVER_TARGET)
        elif self._tick_state >= TARGET_APPROACH_TIMEOUT:
            ppos = self._data.qpos[:3]
            print(f"[FSM] APPROACH_TARGET → timeout  "
                  f"drop_pelvis=({drop[0]:.3f},{drop[1]:.3f})  "
                  f"pelvis=({ppos[0]:.3f},{ppos[1]:.3f})")
            self._transition(FSMState.HOVER_TARGET)
        return PolicyOutput(
            walk_cmd=walk_cmd,
            reach_target=CARRY_POSE,
            reach_active=True,
            grip_closed=True,
        )

    def _hover_target(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] HOVER_TARGET  moving arm above drop point")
        hover = self._target_hover_world()
        reach = self._reach_from_world(hover, right_bias=-0.03)
        palm  = self._palm_world()
        dist  = float(np.linalg.norm(palm - hover))

        if dist < HOVER_TARGET_THRESHOLD:
            self._reach_count += 1
        else:
            self._reach_count = 0

        if self._reach_count >= DEBOUNCE_REACH:
            print(f"[FSM] HOVER_TARGET → threshold met  palm_dist={dist:.3f}")
            self._transition(FSMState.LOWER_TARGET)
        elif self._tick_state >= HOVER_TARGET_TIMEOUT:
            print(f"[FSM] HOVER_TARGET → timeout  palm_dist={dist:.3f}")
            self._transition(FSMState.LOWER_TARGET)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=reach,
            reach_active=True,
            grip_closed=True,
        )

    def _lower_target(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] LOWER_TARGET  descending arm to release height")
        place = self._target_place_world()
        reach = self._reach_from_world(place, right_bias=-0.03)
        palm  = self._palm_world()
        dist  = float(np.linalg.norm(palm - place))

        if dist < LOWER_TARGET_THRESHOLD:
            self._reach_count += 1
        else:
            self._reach_count = 0

        if self._reach_count >= DEBOUNCE_REACH:
            print(f"[FSM] LOWER_TARGET → threshold met  palm_dist={dist:.3f}")
            self._transition(FSMState.OPEN_GRIP)
        elif self._tick_state >= LOWER_TARGET_TIMEOUT:
            print(f"[FSM] LOWER_TARGET → timeout  palm_dist={dist:.3f}")
            self._transition(FSMState.OPEN_GRIP)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=reach,
            reach_active=True,
            grip_closed=True,
        )

    def _open_grip(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] OPEN_GRIP  releasing cylinder")
        place = self._target_place_world()
        reach = self._reach_from_world(place, right_bias=-0.03)

        if not self._attached:
            print(f"[FSM] OPEN_GRIP → released  t={self._tick_total}")
            self._transition(FSMState.RETRACT)
        elif self._tick_state >= OPEN_GRIP_TIMEOUT:
            print(f"[FSM] OPEN_GRIP → timeout  attached={self._attached}  t={self._tick_total}")
            self._transition(FSMState.RETRACT)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=reach,
            reach_active=True,
            grip_closed=False,
        )

    def _retract(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] RETRACT  raising arm to carry pose")
        palm  = self._palm_world()
        tgt_z = self._target_surface_z()

        if palm[2] >= tgt_z + LIFT_DONE_CLEARANCE:
            print(f"[FSM] RETRACT → arm clear  palm_z={palm[2]:.3f}")
            self._transition(FSMState.DONE)
        elif self._tick_state >= RETRACT_TIMEOUT:
            print(f"[FSM] RETRACT → timeout  palm_z={palm[2]:.3f}")
            self._transition(FSMState.DONE)

        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=True,
            grip_closed=False,
        )

    def _done(self) -> PolicyOutput:
        if self._tick_state == 0:
            print("[FSM] DONE  task complete — holding carry pose")
        return PolicyOutput(
            walk_cmd=(0.0, 0.0, 0.0),
            reach_target=CARRY_POSE,
            reach_active=True,
            grip_closed=False,
        )

    # ------------------------------------------------------------------ #
    # GT geometry helpers
    # ------------------------------------------------------------------ #

    def _pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._data.qpos[:3].copy(), self._data.qpos[3:7].copy()

    def _pelvis_yaw(self) -> float:
        """Extract yaw (Z-rotation) from the pelvis quaternion."""
        qw, qx, qy, qz = (float(self._data.qpos[3]), float(self._data.qpos[4]),
                           float(self._data.qpos[5]), float(self._data.qpos[6]))
        return np.arctan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

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

    def _target_surface_z(self) -> float:
        """World Z of the target table's top surface."""
        if self._tbl_white_geom_id >= 0:
            return float(
                self._data.geom_xpos[self._tbl_white_geom_id][2]
                + self._model.geom_size[self._tbl_white_geom_id][2]
            )
        return float(self._data.xpos[self._tbl_white_id][2]) + 0.02

    def _target_drop_world(self) -> np.ndarray:
        """World point 5 cm inside the near edge of the target table, on the surface.

        Near edge: the edge with the less-negative y value (closest to the robot's
        starting position).  Inset of TARGET_NEAR_EDGE_INSET moves into the table.
        """
        if self._tbl_white_geom_id >= 0:
            gx  = float(self._data.geom_xpos[self._tbl_white_geom_id][0])
            gy  = float(self._data.geom_xpos[self._tbl_white_geom_id][1])
            near_edge_y = gy + float(self._model.geom_size[self._tbl_white_geom_id][1])
            drop_y = near_edge_y - TARGET_NEAR_EDGE_INSET
            drop_z = self._target_surface_z()
            return np.array([gx, drop_y, drop_z], dtype=np.float64)
        c = self._data.xpos[self._tbl_white_id].copy()
        return np.array([c[0], c[1] + 0.20, c[2] + 0.02], dtype=np.float64)

    def _target_drop_in_pelvis(self) -> np.ndarray:
        """Frozen drop point expressed in the current pelvis frame."""
        pos, quat = self._pelvis_pose()
        return self._world_to_pelvis(pos, quat, self._target_drop_pt)

    def _target_hover_world(self) -> np.ndarray:
        """World point HOVER_TARGET_HEIGHT above the frozen drop point."""
        p = self._target_drop_pt.copy()
        p[2] = self._target_surface_z() + HOVER_TARGET_HEIGHT
        return p

    def _target_place_world(self) -> np.ndarray:
        """World point PLACE_HEIGHT above the frozen drop point (release height)."""
        p = self._target_drop_pt.copy()
        p[2] = self._target_surface_z() + PLACE_HEIGHT
        return p

    def _cylinder_on_target_table(self) -> bool:
        """True when the cylinder is resting on the target table.

        Checks:
          1. Cylinder z is within (surface − 0.01, surface + ON_TABLE_Z_MAX).
          2. Cylinder XY is within the table geom footprint + ON_TABLE_XY_MARGIN.
        """
        cyl   = self._cylinder_world()
        tgt_z = self._target_surface_z()
        if not (tgt_z - 0.01 <= cyl[2] <= tgt_z + ON_TABLE_Z_MAX):
            return False
        if self._tbl_white_geom_id >= 0:
            gx = float(self._data.geom_xpos[self._tbl_white_geom_id][0])
            gy = float(self._data.geom_xpos[self._tbl_white_geom_id][1])
            hx = float(self._model.geom_size[self._tbl_white_geom_id][0])
            hy = float(self._model.geom_size[self._tbl_white_geom_id][1])
            return (abs(cyl[0] - gx) <= hx + ON_TABLE_XY_MARGIN and
                    abs(cyl[1] - gy) <= hy + ON_TABLE_XY_MARGIN)
        if self._tbl_white_id >= 0:
            c = self._data.xpos[self._tbl_white_id]
            return (abs(cyl[0] - c[0]) <= 0.40 and abs(cyl[1] - c[1]) <= 0.30)
        return False

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

    def _target_approach_walk_cmd(self, drop_pelvis: np.ndarray) -> tuple[float, float, float]:
        """Two-phase world-frame approach to the target-table placement corridor.

        Phase 1 — turn CW to face -y: uses vx=VX_P1 (not 0) so the walker stays
          stable.  R = VX_P1/WZ_P1 = 0.24 m; at this radius the target table
          remains outside the turning circle, so Phase 2 can reach it head-on.

        Phase 2 — drive to standing waypoint: once |yaw + π/2| < PHASE1_ALIGN_TOL,
          decompose world-frame positional error into forward/lateral components
          and apply staircase vx + proportional vy + bearing wz.
        """
        pelvis_pos = self._data.qpos[:3]
        yaw = self._pelvis_yaw()

        # ---- Phase 1: CW turn until facing -y --------------------------------
        if abs(yaw + np.pi / 2) > PHASE1_ALIGN_TOL:
            return (VX_P1, 0.0, -WZ_P1)

        # ---- Phase 2: drive toward standing waypoint -------------------------
        # Standing waypoint: APPROACH_TARGET_X m toward robot from the drop point.
        drop_w  = self._target_drop_pt
        stand_x = float(drop_w[0])
        stand_y = float(drop_w[1]) + APPROACH_TARGET_X   # e.g. -0.60 + 0.34 = -0.26

        ex = stand_x - float(pelvis_pos[0])
        ey = stand_y - float(pelvis_pos[1])
        dist = float(np.sqrt(ex * ex + ey * ey))

        bearing  = float(np.arctan2(ey, ex))
        a_err    = (bearing - yaw + np.pi) % (2.0 * np.pi) - np.pi

        cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
        left_err = -ex * sin_y + ey * cos_y

        if dist > 0.35:   vx = VX_FAST
        elif dist > 0.18: vx = VX_MED
        else:             vx = VX_SLOW
        vy = float(np.clip(K_VY * left_err, -VY_CAP, VY_CAP))
        wz = float(np.clip(K_WZ * a_err,   -WZ_CAP, WZ_CAP))
        return (vx, vy, wz)

    def _near_target_waypoint(self) -> bool:
        """True when robot is close to the standing waypoint AND facing roughly −y.

        The pelvis-frame reach-window check (`_in_reach_window`) is NOT used for
        the target approach because the drop point can satisfy the window even when
        the robot is mid-turn and far from the table.  This world-frame check
        requires both conditions simultaneously.
        """
        yaw = self._pelvis_yaw()
        if abs(yaw + np.pi / 2) > PHASE1_ALIGN_TOL:
            return False
        pelvis = self._data.qpos[:3]
        drop_w = self._target_drop_pt
        stand_x = float(drop_w[0])
        stand_y = float(drop_w[1]) + APPROACH_TARGET_X
        ex = stand_x - float(pelvis[0])
        ey = stand_y - float(pelvis[1])
        return float(np.sqrt(ex * ex + ey * ey)) < TARGET_APPROACH_DIST_THRESH

    def _in_reach_window(self, cyl: np.ndarray) -> bool:
        return (REACH_X_MIN < cyl[0] < REACH_X_MAX and
                REACH_Y_MIN < cyl[1] < REACH_Y_MAX)
