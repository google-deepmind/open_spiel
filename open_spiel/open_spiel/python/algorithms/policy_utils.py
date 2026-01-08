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

"""Adds useful functions for working with dictionaries representing policies."""

from open_spiel.python.algorithms import get_all_states


def policy_to_dict(player_policy,
                   game,
                   all_states=None,
                   state_to_information_state=None):
  """Converts a Policy instance into a tabular policy represented as a dict.

  This is compatible with the C++ TabularExploitability code (i.e.
  pyspiel.exploitability, pyspiel.TabularBestResponse, etc.).

  While you do not have to pass the all_states and state_to_information_state
  arguments, creating them outside of this funciton will speed your code up
  dramatically.

  Args:
    player_policy: The policy you want to convert to a dict.
    game: The game the policy is for.
    all_states: The result of calling get_all_states.get_all_states. Can be
      cached for improved performance.
    state_to_information_state: A dict mapping str(state) to
      state.information_state for every state in the game. Can be cached for
      improved performance.

  Returns:
    A dictionary version of player_policy that can be passed to the C++
    TabularBestResponse, Exploitability, and BestResponse functions/classes.
  """
  if all_states is None:
    all_states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False)
    state_to_information_state = {
        state: all_states[state].information_state_string()
        for state in all_states
    }
  tabular_policy = dict()
  for state in all_states:
    information_state = state_to_information_state[state]
    tabular_policy[information_state] = list(
        player_policy.action_probabilities(all_states[state]).items())
  return tabular_policy


def get_best_response_actions_as_string(best_response_actions):
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


def tabular_policy_to_cpp_map(policy):
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
