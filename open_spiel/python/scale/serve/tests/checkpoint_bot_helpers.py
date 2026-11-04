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

"""Helpers of the checkpoint_bot tests.

In-memory bots, self-play and the privacy check.
"""

import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve.tests import helpers


def memory_loader(policy, params, rule=types.SAMPLE):
  """A PolicyServer loader of one in-memory artifact at every (path, step).

  Args:
    policy: the policy.
    params: its params.
    rule: the artifact's decision rule.

  Returns:
    loader(path, step) -> the artifact.
  """
  return lambda path, step: helpers.artifact(
      policy, params, rule=rule, step=step, path=path
  )


def memory_bot(policy, params, *, deterministic=True, seed=0, temperature=1.0):
  """A checkpoint bot of one in-memory artifact (memory_loader).

  Args:
    policy: the policy.
    params: its params.
    deterministic: greedy (True) or sampled decisions.
    seed: the bot's seed.
    temperature: the sampling temperature.

  Returns:
    The CheckpointBot.
  """
  srv = server_lib.PolicyServer(loader=memory_loader(policy, params))
  return checkpoint_bot_lib.checkpoint_bot(
      "mem://test",
      step=1,
      deterministic=deterministic,
      temperature=temperature,
      seed=seed,
      server=srv,
  )


def _sample_chance(state, rng):
  """A chance outcome of `state`, sampled.

  Args:
    state: A pyspiel chance state.
    rng: A numpy Generator.

  Returns:
    The action.
  """
  actions, probs = zip(*state.chance_outcomes())
  probs = np.asarray(probs, np.float64)
  return int(rng.choice(actions, p=probs / probs.sum()))


def self_play(bot, game, deals, choose=None):
  """The action sequences of games the bot plays against itself.

  One per deal (deal d samples chance with seed d). `choose(state, seat)`
  defaults to the bot's choose_action; every chosen action is checked legal.

  Args:
    bot: the CheckpointBot.
    game: the pyspiel game.
    deals: the deal seeds.
    choose: the decision function, or None.

  Returns:
    One action history per deal.
  """
  choose = choose or bot.choose_action
  out = []
  for d in deals:
    rng = np.random.default_rng(d)
    s = game.new_initial_state()
    while not s.is_terminal():
      if s.is_chance_node():
        s.apply_action(_sample_chance(s, rng))
        continue
      seat = s.current_player()
      a = choose(s, seat)
      assert a in s.legal_actions(), (a, s.legal_actions(), s.history())
      s.apply_action(a)
    out.append(list(s.history()))
  return out


def _view(state, player):
  """`player`'s view of `state`: its information state string, else its
  observation string.

  Args:
    state: A pyspiel state.
    player: The player.

  Returns:
    The string.
  """
  if state.get_game().get_type().provides_information_state_string:
    return state.information_state_string(player)
  return state.observation_string(player)


def _replay(game, history, alt_index=None, alt_outcome=None):
  """The state after `history` with chance outcome alt_outcome at alt_index.

  None when that history is not legal.

  Args:
    game: the pyspiel game.
    history: the actions.
    alt_index: None, or the history index replaced.
    alt_outcome: the chance outcome played there.

  Returns:
    The state, or None.
  """
  s = game.new_initial_state()
  for i, a in enumerate(history):
    if i == alt_index:
      a = alt_outcome
    if s.is_chance_node():
      if a not in dict(s.chance_outcomes()):
        return None
    elif a not in s.legal_actions():
      return None
    s.apply_action(a)
  return s


def check_privacy(
    bot, game, *, num_playouts, seed, max_alternatives=None, max_decisions=None
):
  """Checks that the bot's decisions ignore other seats' private chance.

  The other seats' private chance outcomes, on random playouts of `game`. At
  each decision of each playout, every history that replaces one chance
  outcome by an outcome unused in the history is replayed in pyspiel. When
  the seat to move has the same information state there (pyspiel) and
  another seat does not, the bot's action and distribution must be the same.
  max_alternatives caps the alternatives tried per decision, max_decisions the
  decisions per playout.

  Args:
    bot: the CheckpointBot.
    game: the pyspiel game.
    num_playouts: the playouts.
    seed: the playouts' seed.
    max_alternatives: the alternatives tried per decision, or None.
    max_decisions: the decisions per playout, or None.

  Returns:
    (number of such pairs checked, whether the seat's distribution changed
    when its own information state did).
  """
  rng = np.random.default_rng(seed)
  pairs, own_changes = 0, False
  for _ in range(num_playouts):
    s = game.new_initial_state()
    decisions = 0
    while not s.is_terminal():
      if s.is_chance_node():
        s.apply_action(_sample_chance(s, rng))
        continue
      if max_decisions is not None and decisions >= max_decisions:
        break
      decisions += 1
      seat = s.current_player()
      hist = list(s.history())
      action = bot.choose_action(s, seat)
      probs = dict(bot.last_probs)
      view = _view(s, seat)
      others = [_view(s, p) for p in range(game.num_players()) if p != seat]
      chance_idx, used = [], set()
      replay = game.new_initial_state()
      for i, a in enumerate(hist):
        if replay.is_chance_node():
          chance_idx.append((i, [o for o, _ in replay.chance_outcomes()]))
          used.add(a)
        replay.apply_action(a)
      candidates = []
      for i, outcomes in chance_idx:
        for o in outcomes:
          if o not in used:
            candidates.append((i, o))
      order = rng.permutation(len(candidates))
      tried = 0
      for k in order:
        if max_alternatives is not None and tried >= max_alternatives:
          break
        i, o = candidates[k]
        alt = _replay(game, hist, i, o)
        if alt is None or alt.is_terminal() or alt.current_player() != seat:
          continue
        tried += 1
        if _view(alt, seat) == view:
          if [
              _view(alt, p) for p in range(game.num_players()) if p != seat
          ] == others:
            continue  # a change nobody sees: not a privacy pair
          alt_action = bot.choose_action(alt, seat)
          assert alt_action == action and bot.last_probs == probs, (
              f"seat {seat}'s decision depends on another seat's private "
              f"outcome: {hist} -> {action} {probs}; {list(alt.history())} -> "
              f"{alt_action} {bot.last_probs}"
          )
          pairs += 1
        elif not own_changes:
          bot.choose_action(alt, seat)
          own_changes = bot.last_probs != probs
      s.apply_action(
          int(rng.choice(s.legal_actions())) if rng.random() < 0.5 else action
      )
  return pairs, own_changes
