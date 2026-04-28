"""ONNX policy wrapper for CPU inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXPolicy:
  """Thin CPU-only ONNX inference wrapper.

  Expects a 1D or (1, N) float32 observation and returns the first output row.
  """

  def __init__(self, model_path: str | Path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      str(model_path), sess_options, providers=["CPUExecutionProvider"]
    )
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
      obs = obs.reshape(1, -1)
    obs = obs.astype(np.float32)
    return self.session.run([self.output_name], {self.input_name: obs})[0][0]
