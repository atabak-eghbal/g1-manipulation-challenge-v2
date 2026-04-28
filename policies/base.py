"""Policy interfaces and data contracts for high-level control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PolicyOutput:
  """High-level command output from a policy step."""

  walk_cmd: Sequence[float]
  reach_target: Sequence[float]
  grip_closed: bool


class BasePolicy(ABC):
  """Abstract interface for policies that emit high-level commands."""

  @abstractmethod
  def step(self) -> PolicyOutput:
    """Return the latest policy output."""
