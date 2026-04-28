"""Scene helpers for deterministic reset and camera rendering."""

from __future__ import annotations

from typing import Iterable

import mujoco
import numpy as np


class CameraRenderer:
  """Offscreen renderer for robot-mounted cameras using mujoco.Renderer."""

  def __init__(self, model, data, width: int = 320, height: int = 240):
    self.model = model
    self.data = data
    self.renderer = mujoco.Renderer(model, height, width)

  def render(self, camera_name: str) -> np.ndarray:
    """Render from a named camera, return RGB array (H, W, 3)."""
    self.renderer.update_scene(self.data, camera=camera_name)
    return self.renderer.render().copy()


def reset_robot(
  model,
  data,
  config: dict,
  joint_names: Iterable[str],
  *,
  base_pos: tuple[float, float, float] = (-0.6, 0.0, 0.76),
  base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
  reset_data: bool = True,
) -> None:
  """Reset the robot to a deterministic pose and forward the model."""
  if reset_data:
    mujoco.mj_resetData(model, data)
  data.qpos[0:3] = base_pos
  data.qpos[3:7] = base_quat
  for name, value in config["default_joint_pos"].items():
    if name in joint_names:
      data.qpos[7 + joint_names.index(name)] = value
  mujoco.mj_forward(model, data)
