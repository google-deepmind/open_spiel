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
import json
import os

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.utils import metrics


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters((True,), (False,))
  def test_create(self, just_logging: bool):
    logdir = self.create_tempdir().full_path
    with metrics.ensure_flushes(
        metrics.create_default_writer(logdir, just_logging=just_logging)
    ) as writer:
      writer.write_hparams({"param1": 1.0, "param2": 2.0})
      for step in range(5):
        writer.write_scalars(step, {"value": step * step})
    writer.close()

    path = os.path.join(logdir, "metrics.jsonl")
    if just_logging:
      self.assertFalse(os.path.exists(path))
      return
    with open(path) as f:
      records = [json.loads(line) for line in f]
    self.assertEqual(records[0], {"hparams": {"param1": 1.0, "param2": 2.0}})
    self.assertEqual(records[1:],
                     [{"step": s, "value": s * s} for s in range(5)])

  def test_no_logdir(self):
    writer = metrics.create_default_writer()
    writer.write_scalars(0, {"value": 1.0})
    writer.flush()


if __name__ == "__main__":
  absltest.main()
