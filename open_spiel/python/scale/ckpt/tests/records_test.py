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

"""ckpt.records: stored field names round-trip through encode and decode."""

import sys

import pytest

from open_spiel.python.scale.ckpt import records


def test_encode_then_decode_round_trips(monkeypatch):
  """Encode then decode round trips."""
  monkeypatch.setattr(records, "_NAMES", {})
  records.register_field_name(
      "run_id", "job", to_stored=str.upper, from_stored=str.lower
  )
  stored = records.encode({"run_id": "a7", "step": 3, "other": None})
  assert stored == {"job": "A7", "step": 3, "other": None}
  assert records.decode(stored) == {"run_id": "a7", "step": 3, "other": None}
  assert records.decode({"job": None}) == {"run_id": None}


def test_one_converter_without_its_inverse_is_refused(monkeypatch):
  """One converter without its inverse is refused."""
  monkeypatch.setattr(records, "_NAMES", {})
  with pytest.raises(ValueError, match="from_stored"):
    records.register_field_name("run_id", "job", to_stored=str.upper)


def test_two_fields_under_one_stored_name_are_refused(monkeypatch):
  """Two fields under one stored name are refused."""
  monkeypatch.setattr(records, "_NAMES", {})
  records.register_field_name("run_id", "job")
  with pytest.raises(ValueError, match="already field 'run_id'"):
    records.register_field_name("attempt_id", "job")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
