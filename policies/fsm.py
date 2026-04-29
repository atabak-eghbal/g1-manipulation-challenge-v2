"""FSM-driven policy: wraps FSMCore and applies its output to the controller."""

from __future__ import annotations

from .base import BasePolicy, PolicyOutput
from .fsm_core import FSMCore, FSMState

# States where grip_closed=False is intentional — do not override.
_RELEASE_STATES = frozenset({FSMState.OPEN_GRIP, FSMState.RETRACT, FSMState.DONE})


class FSMPolicy(BasePolicy):
    """Autonomous policy driven by FSMCore.

    Calls fsm.tick() each step, then writes the resulting command into
    the controller so that ctrl.step() picks it up immediately.

    grasp_backend (optional): a GraspBackend whose .attached state is forwarded
    to FSMCore each tick and whose grip_closed override is applied to the output.
    """

    def __init__(self, controller, grasp_backend=None) -> None:
        self._ctrl  = controller
        self._grasp = grasp_backend
        self._fsm   = FSMCore(controller.model, controller.data)

    def step(self) -> PolicyOutput:
        attached = self._grasp.attached if self._grasp is not None else False
        out = self._fsm.tick(attached=attached)
        out = self._close_grip_command(out)
        # Push FSM output into controller state before ctrl.step() runs.
        self._ctrl.lin_vel_x, self._ctrl.lin_vel_y, self._ctrl.ang_vel_z = out.walk_cmd
        self._ctrl.reach_target[:] = out.reach_target
        self._ctrl.reach_active    = out.reach_active
        self._ctrl.grip_closed     = out.grip_closed
        return out

    def _close_grip_command(self, out: PolicyOutput) -> PolicyOutput:
        """Keep grip closed while carrying — but not during intentional release.

        Prevents a single-tick grip opening caused by FSM state-transition timing
        from accidentally dropping the cylinder mid-carry.  The guard is bypassed
        in OPEN_GRIP / RETRACT / DONE so the FSM can actually release the object.
        """
        if (self._grasp is not None
                and self._grasp.attached
                and not out.grip_closed
                and self._fsm.state not in _RELEASE_STATES):
            return PolicyOutput(
                walk_cmd=out.walk_cmd,
                reach_target=out.reach_target,
                reach_active=out.reach_active,
                grip_closed=True,
            )
        return out
