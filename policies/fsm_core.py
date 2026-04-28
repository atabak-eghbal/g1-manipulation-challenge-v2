"""Pure FSM state machine — no controller dependency."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from .base import PolicyOutput

# Safe carry-pose for the right arm (pelvis frame, metres).
# Keeps the hand clear of the legs while standing.
CARRY_POSE: tuple[float, float, float] = (0.3, -0.2, 0.2)

# Ticks spent in SETTLE before moving to DONE.
# FSM runs at the controller rate (~50 Hz), so 150 ticks ≈ 3 s.
SETTLE_TICKS = 150


class FSMState(Enum):
    SETTLE    = auto()
    DONE      = auto()
    # ---- stubs for future phases (not yet wired) ----
    APPROACH  = auto()
    GRASP     = auto()
    TRANSPORT = auto()
    PLACE     = auto()


class FSMCore:
    """Tick-driven state machine that emits high-level PolicyOutput each step."""

    def __init__(self) -> None:
        self.state = FSMState.SETTLE
        self._tick_total = 0    # lifetime ticks
        self._tick_state = 0    # ticks in current state
        print(f"[FSM] init  state={self.state.name}")

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
    # Dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(self) -> PolicyOutput:
        if self.state == FSMState.SETTLE:
            return self._settle()
        if self.state == FSMState.DONE:
            return self._done()
        # Unreachable until future phases are wired.
        return self._settle()

    def _transition(self, new_state: FSMState) -> None:
        print(
            f"[FSM] {self.state.name} → {new_state.name}"
            f"  (t={self._tick_total})"
        )
        self.state = new_state
        self._tick_state = 0

    # ------------------------------------------------------------------ #
    # State handlers
    # ------------------------------------------------------------------ #

    def _settle(self) -> PolicyOutput:
        if self._tick_state == 0:
            print(f"[FSM] SETTLE  waiting {SETTLE_TICKS} ticks (~{SETTLE_TICKS/50:.0f} s)")
        if self._tick_state >= SETTLE_TICKS:
            self._transition(FSMState.DONE)
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
    # Stubs — to be implemented in later steps
    # ------------------------------------------------------------------ #

    def _source_pos(self, model, data) -> np.ndarray:
        """World position of the red block (needed for APPROACH / GRASP)."""
        raise NotImplementedError("_source_pos: implement in Step 6")

    def _target_pos(self, model, data) -> np.ndarray:
        """World position of the drop zone (needed for TRANSPORT / PLACE)."""
        raise NotImplementedError("_target_pos: implement in Step 6")

    def _approach_cmd(self) -> tuple[float, float, float]:
        """Walk command (vx, vy, wz) to move toward the source object."""
        raise NotImplementedError("_approach_cmd: implement in Step 6")

    def _grasp_reach_target(self) -> tuple[float, float, float]:
        """Right-arm reach target in pelvis frame for the grasp pose."""
        raise NotImplementedError("_grasp_reach_target: implement in Step 6")
