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
"""Tests for metrics."""

import glob
import os

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.utils import metrics


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters((True,), (False,))
  def test_create(self, just_logging: bool):
    logdir = self.create_tempdir()
    # Create the writer.
    writer = metrics.create_default_writer(
        logdir.full_path, just_logging=just_logging)
    self.assertIsInstance(writer, metrics.metric_writers.MultiWriter)

    # Write some metrics.
    writer.write_hparams({"param1": 1.0, "param2": 2.0})
    for step in range(5):
      writer.write_scalars(step, {"value": step * step})

    metrics.write_values(writer, 5, {
        "scalar": 1.23,
        "text": metrics.Text(value="foo")
    })
    # Flush the writer.
    writer.flush()

    # Check that the summary file exists if not just logging.
    self.assertLen(
        glob.glob(os.path.join(logdir.full_path, "events.out.tfevents.*")),
        0 if just_logging else 1)


if __name__ == "__main__":
  absltest.main()
