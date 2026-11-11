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

"""The error every contract checker raises."""


class ContractError(AssertionError):
  """A component broke a framework contract."""


def require(cond, msg: str) -> None:
  """Raises ContractError(msg) unless `cond`.

  Args:
    cond: The condition.
    msg: The failure message.

  Raises:
    ContractError: `cond` is false.
  """
  if not bool(cond):
    raise ContractError(msg)
