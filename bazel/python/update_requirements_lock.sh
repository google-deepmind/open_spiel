#!/usr/bin/env bash

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

# Regenerates requirements_lock.txt from requirements.in: one universal
# (Linux + macOS), hash-pinned lock file for the Python version of the Bazel
# build. Needs `uv` (https://docs.astral.sh/uv/), e.g. `pip install uv`.

set -euo pipefail

readonly PYTHON_VERSION="3.12"
cd "$(dirname "$0")"

uv pip compile \
  --universal \
  --python-version "${PYTHON_VERSION}" \
  --generate-hashes \
  --no-header \
  --output-file requirements_lock.txt \
  requirements.in
