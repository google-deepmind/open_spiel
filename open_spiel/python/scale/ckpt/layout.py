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

"""File names of a run directory's layout.

Shared by the modules that read and write it. Imports nothing, so any module
can use it without a cycle.
"""

#: <ckpt>/train_state.json: the run's committed step and league commits.
TRAIN_STATE = "train_state.json"
#: <run_dir>/store: the run's ArtifactStore of member artifacts.
STORE_DIR = "store"
