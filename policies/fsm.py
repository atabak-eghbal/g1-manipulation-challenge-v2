"""FSM-driven policy: wraps FSMCore and applies its output to the controller."""

from __future__ import annotations

from .base import BasePolicy, PolicyOutput
from .fsm_core import FSMCore


class FSMPolicy(BasePolicy):
    """Autonomous policy driven by FSMCore.

    Calls fsm.tick() each step, then writes the resulting command into
    the controller so that ctrl.step() picks it up immediately.
    """

    def __init__(self, controller) -> None:
        self._ctrl = controller
        self._fsm  = FSMCore(controller.model, controller.data)

    def step(self) -> PolicyOutput:
        out = self._fsm.tick()
        # Push FSM output into controller state before ctrl.step() runs.
        self._ctrl.lin_vel_x, self._ctrl.lin_vel_y, self._ctrl.ang_vel_z = out.walk_cmd
        self._ctrl.reach_target[:] = out.reach_target
        self._ctrl.reach_active    = out.reach_active
        self._ctrl.grip_closed     = out.grip_closed
        return out
