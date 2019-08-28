# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adds useful functions for working with dictionaries representing policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import numpy as np
from typing import Dict, List, Tuple, Text


def get_best_response_actions_as_string(
    best_response_actions: Dict[bytes, int]) -> Text:
  """Turns a dict<bytes, int> into a bytestring compatible with C++.

  i.e. the bytestring can be copy-pasted as the brace initialization for a
  {std::unordered_,std::,absl::flat_hash_}map<std::string, int>.

  Args:
    best_response_actions: A dict mapping bytes to ints.

  Returns:
    A bytestring that can be copy-pasted to brace-initialize a C++
    std::map<std::string, T>.
  """
  best_response_keys = sorted(best_response_actions.keys())
  best_response_strings = [
      "%s: %i" % (k, best_response_actions[k]) for k in best_response_keys
  ]
  return "{%s}" % (", ".join(best_response_strings))


def tabular_policy_to_cpp_map(
    policy: Dict[bytes, List[Tuple[int, np.float64]]]) -> Text:
  """Turns a policy into a C++ compatible bytestring for brace-initializing.

  Args:
    policy: A dict representing a tabular policy. The keys are infostate
      bytestrings.

  Returns:
    A bytestring that can be copy-pasted to brace-initialize a C++
    std::map<std::string, open_spiel::ActionsAndProbs>.
  """
  cpp_entries = []
  policy_keys = sorted(policy.keys())
  for key in policy_keys:
    tuple_strs = ["{%i, %s}" % (p[0], p[1].astype(str)) for p in policy[key]]
    value = "{" + ", ".join(tuple_strs) + "}"
    cpp_entries.append('{"%s", %s}' % (key, value))
  return "{%s}" % (",\n".join(cpp_entries))
