# Copyright 2026 DeepMind Technologies Limited
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

"""Functions the core tests name by "module:attr" (index_fns).

They live in a module that is not a test file: a test file may be imported
under a name pytest derives from its path, and a recorded "module:attr" must
name a module that imports under one name wherever it is resolved.
"""

from open_spiel.python.scale.envs import kuhn


def oob_index(info):
  """An index_fn that is out of range for a 12-state table (planted bug).

  Args:
    info: a batched kuhn InfoState.

  Returns:
    the kuhn index plus NUM_INFO_STATES.
  """
  return kuhn.info_index(info) + kuhn.NUM_INFO_STATES


OOB_INDEX = f"{__name__}:oob_index"
