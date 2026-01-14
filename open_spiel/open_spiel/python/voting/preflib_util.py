# Copyright 2023 DeepMind Technologies Limited
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

"""Helpers to work with PrefLib data."""

import pyspiel
from open_spiel.python.voting import base


def parse_preflib_data(string_data: str) -> base.PreferenceProfile:
  """Parses the contents of a PrefLib data file.

  Currently only supports SOC and SOI. See https://www.preflib.org/format.

  Args:
    string_data: the name of the file to parse.

  Returns:
    A preference profile.
  """
  lines = string_data.split("\n")
  alternatives = []
  num_alternatives = None
  num_votes = None
  profile = base.PreferenceProfile()
  for raw_line in lines:
    line = raw_line.strip()
    if not line: continue
    if line.startswith("#"):
      parts = line.split(" ")
      if line.startswith("# DATA TYPE: "):
        assert(parts[3] == "soc" or parts[3] == "soi")
      elif line.startswith("# NUMBER ALTERNATIVES:"):
        num_alternatives = int(parts[3])
        alternatives = [None] * num_alternatives
      elif line.startswith("# NUMBER VOTERS:"):
        num_votes = int(parts[3])
      elif line.startswith("# ALTERNATIVE NAME "):
        num = int(parts[3].split(":")[0])
        index_of_colon = line.index(":")
        assert 1 <= num <= num_alternatives
        alternatives[num-1] = line[index_of_colon+2:]
    else:
      if profile.num_alternatives() == 0:
        profile = base.PreferenceProfile(alternatives=alternatives)
      index_of_colon = line.index(":")
      weight = int(line[:index_of_colon])
      vote_parts = line[index_of_colon+2:].split(",")
      vote = [alternatives[int(part) - 1] for part in vote_parts]
      if weight > 0:
        profile.add_vote(vote, weight)
  assert num_votes == profile.num_votes()
  return profile


def parse_preflib_datafile(filename: str) -> base.PreferenceProfile:
  """Parses a Preflib data file.

  Currently only supports SOC and SOI. See https://www.preflib.org/format.

  Args:
    filename: the name of the file to parse.

  Returns:
    A preference profile.
  """
  contents = pyspiel.read_contents_from_file(filename, "r")
  return parse_preflib_data(contents)
