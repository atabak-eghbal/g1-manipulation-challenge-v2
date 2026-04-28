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
  """Reset the robot to a deterministic pose and forward the model.

  joint_names should be the full ordered joint list from the config/model.
  """
  if reset_data:
    mujoco.mj_resetData(model, data)
  joint_names_list = list(joint_names)
  joint_index = {name: idx for idx, name in enumerate(joint_names_list)}
  data.qpos[0:3] = base_pos
  data.qpos[3:7] = base_quat
  for name, value in config["default_joint_pos"].items():
    idx = joint_index.get(name)
    if idx is not None:
      data.qpos[7 + idx] = value
  mujoco.mj_forward(model, data)
