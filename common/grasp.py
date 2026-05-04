"""Grasp backend interface and kinematic-attachment implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GraspBackend(ABC):
    """Abstract grasp backend — called every physics tick from the main loop."""

    @property
    @abstractmethod
    def attached(self) -> bool:
        """True while the object is kinematically attached."""

    @abstractmethod
    def tick(self, grip_closed: bool) -> bool:
        """Update attachment state. Returns True if currently attached.

        Must be called after every mujoco.mj_step() so the cylinder pose is
        corrected before the next integration step.
        """

    @abstractmethod
    def release(self) -> None:
        """Force-detach the object and restore its physics."""


class KinematicAttachment(GraspBackend):
    """Teleport-weld the cylinder to the palm while the grip is closed.

    Simulation shortcut: bypasses contact forces. The cylinder is placed at a
    fixed palm-local offset (snapped to SNAP_DIST if the hand closed far away)
    and its freejoint velocity is zeroed every tick so it does not drift between
    teleportations.

    Collisions are disabled while attached to prevent geom-overlap impulses from
    destabilising the robot.
    """

    ATTACH_DIST:   float = 0.13  # m: auto-attach when palm is within this distance
    SNAP_DIST:     float = 0.03  # m: clamp palm-local offset so object sits in hand
    RELEASE_TICKS: int   = 15    # physics ticks (~0.075s) to wait before restoring geoms

    def __init__(
        self,
        model,
        data,
        palm_site_id: int,
        obj_body_id: int,
    ) -> None:
        self._model    = model
        self._data     = data
        self._palm_id  = palm_site_id
        self._obj_id   = obj_body_id

        # Freejoint addressing (first joint of the body; must be a freejoint).
        jnt_id = int(model.body_jntadr[obj_body_id])
        self._qposadr = int(model.jnt_qposadr[jnt_id])
        self._qveladr = int(model.jnt_dofadr[jnt_id])

        # Geoms belonging to this body — used to toggle collisions.
        self._geom_ids = [
            i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) == obj_body_id
        ]
        self._orig_contype     = {g: int(model.geom_contype[g])     for g in self._geom_ids}
        self._orig_conaffinity = {g: int(model.geom_conaffinity[g]) for g in self._geom_ids}

        self._is_attached    = False
        self._release_timer  = 0
        self._local_offset   = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def attached(self) -> bool:
        return self._is_attached

    def tick(self, grip_closed: bool) -> bool:
        if not self._is_attached:
            if self._release_timer > 0:
                self._release_timer -= 1
                if self._release_timer == 0:
                    self._restore_geoms()
            if grip_closed:
                palm = self._data.site_xpos[self._palm_id].copy()
                obj  = self._data.xpos[self._obj_id].copy()
                if float(np.linalg.norm(palm - obj)) <= self.ATTACH_DIST:
                    self._attach(palm, obj)
        else:
            if not grip_closed:
                self._release()
            else:
                self._update_pose()
        return self._is_attached

    def release(self) -> None:
        if self._is_attached:
            self._release()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _attach(self, palm_pos: np.ndarray, obj_pos: np.ndarray) -> None:
        palm_rot = self._data.site_xmat[self._palm_id].reshape(3, 3).copy()
        local = palm_rot.T @ (obj_pos - palm_pos)
        d = float(np.linalg.norm(local))
        if d > self.SNAP_DIST:
            local = local * (self.SNAP_DIST / d)
        self._local_offset = local

        for g in self._geom_ids:
            self._model.geom_contype[g]     = 0
            self._model.geom_conaffinity[g] = 0

        self._is_attached = True
        self._release_timer = 0
        snap = float(np.linalg.norm(self._local_offset))
        print(f"[GRASP] attached  dist={float(np.linalg.norm(palm_pos - obj_pos)):.3f} m"
              f"  snap_offset={snap:.3f} m")
        self._update_pose()

    def _update_pose(self) -> None:
        palm_pos = self._data.site_xpos[self._palm_id].copy()
        palm_rot = self._data.site_xmat[self._palm_id].reshape(3, 3).copy()
        new_pos  = palm_pos + palm_rot @ self._local_offset
        self._data.qpos[self._qposadr    :self._qposadr + 3] = new_pos
        self._data.qpos[self._qposadr + 3:self._qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self._data.qvel[self._qveladr    :self._qveladr + 6] = 0.0

    def _release(self) -> None:
        self._data.qvel[self._qveladr:self._qveladr + 6] = 0.0
        self._is_attached = False
        self._release_timer = self.RELEASE_TICKS
        print(f"[GRASP] released — waiting {self.RELEASE_TICKS} ticks to restore geoms")

    def _restore_geoms(self) -> None:
        for g in self._geom_ids:
            self._model.geom_contype[g]     = self._orig_contype[g]
            self._model.geom_conaffinity[g] = self._orig_conaffinity[g]
        print("[GRASP] geoms restored")
