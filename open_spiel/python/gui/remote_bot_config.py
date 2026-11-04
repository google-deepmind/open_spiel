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


"""The `config` a Remote bot seat's bot is sent with every request.

It is the seat's `config` setting, a JSON object, with its `checkpoint`
setting, when not empty, as `config.checkpoint`: the checkpoint a bot that
plays one reads (REMOTE_BOTS.md).
"""

from collections.abc import Mapping
import json

from open_spiel.python.gui import contract

# The key of the bot's `config` that carries the seat's `checkpoint`
# setting.
CHECKPOINT_KEY = "checkpoint"


def parse(text: str) -> dict[str, contract.JsonValue]:
  """Parses the `config` setting: a JSON object.

  Raises:
    ValueError: If `text` is not a JSON object.
  """
  try:
    value = json.loads(text)
  except json.JSONDecodeError as e:
    raise ValueError(f"The remote bot's config is not JSON: {e.msg}.") from e
  if not isinstance(value, dict):
    raise ValueError("The remote bot's config is not a JSON object.")
  return value


def seat_bot_config(
    config: Mapping[str, contract.ParameterValue],
) -> dict[str, contract.JsonValue]:
  """Returns the `config` a seat's bot is sent: its `config` and `checkpoint`.

  Raises:
    ValueError: If `config` is not a JSON object, or gives a `checkpoint`
      other than the `checkpoint` setting.
  """
  value = parse(str(config["config"]))
  checkpoint = str(config.get("checkpoint", "")).strip()
  if checkpoint:
    given = value.get(CHECKPOINT_KEY)
    if given is not None and given != checkpoint:
      raise ValueError(
          "The remote bot's config gives a checkpoint other than its "
          "checkpoint setting; give it in one place."
      )
    value[CHECKPOINT_KEY] = checkpoint
  return value
