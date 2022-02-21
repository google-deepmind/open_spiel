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
"""Metrics and logging helpers."""

from typing import Optional

# pylint: disable=g-import-not-at-top disable=unused-import
try:
  from clu import metric_writers
  from clu.metric_writers import ensure_flushes
  from clu.metric_writers import write_values
  from clu.values import *  # pylint: disable=wildcard-import
except ImportError as e:
  raise ImportError(
      str(e) +
      "\nCLU not found. Please install CLU: python3 -m pip install clu") from e
# pylint: enable=g-import-not-at-top enable=unused-import


def create_default_writer(logdir: Optional[str] = None,
                          just_logging: bool = False,
                          **kwargs) -> metric_writers.MetricWriter:
  """Create the default metrics writer.

  See metric_writers.LoggingWriter interface for the API to write the metrics
  and other metadata, e.g. hyper-parameters. Sample usage is as follows:

  writer = metrics.create_default_writer('/some/path')
  writer.write_hparams({"learning_rate": 0.001, "batch_size": 64})
  ...
  # e.g. in training loop.
  writer.write_scalars(step, {"loss": loss})
  ...
  writer.flush()

  Args:
    logdir: Path of the directory to store the metric logs as TF summary files.
      If None, files will not be created.
    just_logging: If true, metrics will be outputted only to INFO log.
    **kwargs: kwargs passed to the CLU default writer.

  Returns:
    a metric_writers.MetricWriter.
  """
  return metric_writers.create_default_writer(
      logdir=logdir, just_logging=just_logging, **kwargs)
