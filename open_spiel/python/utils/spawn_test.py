# Copyright 2019 DeepMind Technologies Limited
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

# Lint as: python3
"""Tests for open_spiel.python.utils.spawn."""

import random
import time

from absl.testing import absltest

from open_spiel.python.utils import spawn


class SpawnTest(absltest.TestCase):

  def test_spawn_works(self):
    max_sleep_time = 0.01  # 10ms

    def worker_fn(worker_id, queue):
      queue.put(worker_id)  # Show it's up and running.
      random.seed(time.time() + worker_id)
      while True:
        value = queue.get()
        if value is None:
          break
        time.sleep(max_sleep_time * random.random())
        queue.put((worker_id, value))

    num_workers = 5
    workers = [spawn.Process(worker_fn, kwargs={"worker_id": i})
               for i in range(num_workers)]

    # Make sure they're warmed up.
    for worker_id, worker in enumerate(workers):
      self.assertEqual(worker_id, worker.queue.get())

    num_work_units = 40
    expected_output = []
    for worker_id, worker in enumerate(workers):
      for i in range(num_work_units):
        worker.queue.put(i)
        expected_output.append((worker_id, i))
      worker.queue.put(None)

    start_time = time.time()

    output = []
    i = 0
    while len(output) < len(expected_output):
      for worker in workers:
        try:
          output.append(worker.queue.get_nowait())
        except spawn.Empty:
          pass

      time.sleep(0.001)
      i += 1
      self.assertLess(time.time() - start_time,
                      20 * max_sleep_time * num_work_units,
                      msg=f"Don't wait forever. Loop {i}, found {len(output)}")

    time_taken = time.time() - start_time
    print("Finished in {:.3f}s, {:.2f}x the max".format(
        time_taken, time_taken / (max_sleep_time * num_work_units)))

    for worker in workers:
      worker.join()

    # All messages arrived
    self.assertLen(output, len(expected_output))
    self.assertCountEqual(output, expected_output)

    # The messages arrived out of order, showing parallelism.
    self.assertNotEqual(output, expected_output)


if __name__ == "__main__":
  with spawn.main_handler():
    absltest.main()
