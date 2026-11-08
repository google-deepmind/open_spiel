# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metrics and logging helpers.

A small, dependency-free metric writer for the example scripts. Metrics are
logged with absl logging and, if a log directory is given, appended to a
`metrics.jsonl` file in it (one JSON object per call).
"""

import contextlib
import json
import os
from typing import Any, Iterator, Mapping, Optional

from absl import logging

_METRICS_FILE = "metrics.jsonl"


class MetricWriter:
  """Writes hyper-parameters and scalar metrics to the log and a JSONL file."""

  def __init__(self, logdir: Optional[str] = None):
    """Creates the writer.

    Args:
      logdir: Directory for `metrics.jsonl`. If None, metrics are only logged.
    """
    self._file = None
    if logdir:
      os.makedirs(logdir, exist_ok=True)
      self._file = open(os.path.join(logdir, _METRICS_FILE), "a")

  def _write(self, record: Mapping[str, Any]) -> None:
    if self._file is not None:
      self._file.write(json.dumps(record, default=str) + "\n")

  def write_hparams(self, hparams: Mapping[str, Any]) -> None:
    """Writes the hyper-parameters of a run."""
    logging.info("[Hyperparameters] %s", dict(hparams))
    self._write({"hparams": dict(hparams)})

  def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
    """Writes scalar metrics for `step`."""
    logging.info(
        "[%d] %s", step, ", ".join(f"{k}={v}" for k, v in scalars.items()))
    self._write({"step": int(step), **{k: _to_json(v)
                                       for k, v in scalars.items()}})

  def flush(self) -> None:
    """Flushes buffered output."""
    if self._file is not None:
      self._file.flush()

  def close(self) -> None:
    """Flushes and closes the writer."""
    if self._file is not None:
      self._file.close()
      self._file = None


def _to_json(value: Any) -> Any:
  """Converts numpy and JAX scalars to Python numbers."""
  item = getattr(value, "item", None)
  return item() if callable(item) else value


def create_default_writer(logdir: Optional[str] = None,
                          just_logging: bool = False) -> MetricWriter:
  """Creates the default metrics writer.

  Sample usage:

    writer = metrics.create_default_writer("/some/path")
    writer.write_hparams({"learning_rate": 0.001, "batch_size": 64})
    ...
    writer.write_scalars(step, {"loss": loss})  # e.g. in a training loop.
    ...
    writer.flush()

  Args:
    logdir: Directory where `metrics.jsonl` is written. If None, metrics are
      only logged.
    just_logging: If true, metrics are only logged, even if `logdir` is set.

  Returns:
    A MetricWriter.
  """
  return MetricWriter(None if just_logging else logdir)


@contextlib.contextmanager
def ensure_flushes(writer: MetricWriter) -> Iterator[MetricWriter]:
  """Context manager that flushes `writer` on exit."""
  try:
    yield writer
  finally:
    writer.flush()
