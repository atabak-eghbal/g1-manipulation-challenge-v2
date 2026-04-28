"""Policy interfaces and data contracts for high-level control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

WalkCommand = tuple[float, float, float]
ReachTarget = tuple[float, float, float]


@dataclass(frozen=True)
class PolicyOutput:
  """High-level command output from a policy step.

  walk_cmd: (lin_vel_x, lin_vel_y, ang_vel_z)
  reach_target: (x, y, z) target coordinates in pelvis frame
  """

  walk_cmd: WalkCommand
  reach_target: ReachTarget
  grip_closed: bool


class BasePolicy(ABC):
  """Abstract interface for policies that emit high-level commands."""

  def handle_key(self, keycode: int) -> None:
    """Optional keyboard hook for interactive policies."""
    return None

  @abstractmethod
  def step(self) -> PolicyOutput:
    """Return the latest policy output."""
