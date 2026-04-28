"""Keyboard-driven policy wrapper for manual control."""

from __future__ import annotations

from .base import BasePolicy, PolicyOutput


class KeyboardPolicy(BasePolicy):
  """Policy wrapper that delegates keyboard input to the controller."""

  def __init__(self, controller):
    self._controller = controller

  def handle_key(self, keycode: int) -> None:
    """Forward keyboard events to the controller."""
    self._controller.key_callback(keycode)

  def step(self) -> PolicyOutput:
    """Expose the controller's current high-level command state."""
    walk_cmd = (
      self._controller.lin_vel_x,
      self._controller.lin_vel_y,
      self._controller.ang_vel_z,
    )
    reach_target = self._controller.reach_target
    return PolicyOutput(
      walk_cmd=walk_cmd,
      reach_target=reach_target,
      grip_closed=self._controller.grip_closed,
    )
