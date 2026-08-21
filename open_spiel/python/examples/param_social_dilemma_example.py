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

"""Example: N-player parameterized social dilemma with Axelrod-style bots.

Plays one episode of python_param_social_dilemma using a rotation of simple
strategies inspired by Axelrod's 2-player Iterated Prisoner's Dilemma
tournament (Always Cooperate, Always Defect, Tit-for-Tat, Grim Trigger), and
prints the resulting action sequence and returns. Intended as a minimal
demonstration of using the game for MARL benchmarking; see
https://github.com/google-deepmind/open_spiel/issues/1431.

Axelrod's strategies were defined for exactly 2 players and reference "the"
opponent's last move; there is no single agreed-upon way to extend them to N
players. Tit-for-Tat and Grim Trigger below use one reasonable
generalization each (see their docstrings for the specific rule used) --
not a claim that this is *the* canonical N-player version.

Usage:
  python param_social_dilemma_example.py --players=4 \
      --termination_probability=0.1
"""

import random

from absl import app
from absl import flags

from open_spiel.python.games import param_social_dilemma
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("players", 4, "Number of players.")
flags.DEFINE_float("termination_probability", 0.125,
                    "Probability the episode ends after each round.")
flags.DEFINE_enum("payoff_kind", "linear", ["linear", "public_goods"],
                   "Payoff model: 'linear' (fraction-based generalization "
                   "of the 2x2 PD matrix) or 'public_goods' (classic "
                   "public-goods game, where the free-rider advantage "
                   "grows with the number of players).")
flags.DEFINE_float("reward_noise_std", 0.0,
                    "Std. dev. of the (discretized) reward noise. 0 to "
                    "disable.")
flags.DEFINE_bool("dynamic_payoffs", False,
                   "Whether payoffs can switch between regimes. Only "
                   "valid with --payoff_kind=linear.")
flags.DEFINE_string("payoff_regimes", "10 5 1 0 20 5 1 0",
                     "Whitespace-separated (temptation, reward, punishment, "
                     "sucker) groups of 4 floats, used when dynamic_payoffs "
                     "is set. Only used with --payoff_kind=linear.")
flags.DEFINE_float("payoff_change_prob", 0.1,
                    "Probability of switching payoff regime each round, "
                    "when dynamic_payoffs is set.")
flags.DEFINE_float("pgg_endowment", 10.0,
                    "Per-round endowment in public_goods mode.")
flags.DEFINE_float("pgg_multiplier", 1.5,
                    "Public-goods pool multiplier (must satisfy "
                    "1 < pgg_multiplier < players). Only used with "
                    "--payoff_kind=public_goods.")
flags.DEFINE_integer("seed", 0, "Random seed.")

Action = param_social_dilemma.Action


def _own_action_history(state, player):
  return [
      pa.action for pa in state.full_history() if pa.player == player
  ]


def always_cooperate(state, player):
  del state, player
  return Action.COOPERATE


def always_defect(state, player):
  del state, player
  return Action.DEFECT


def tit_for_tat(state, player):
  """N-player Tit-for-Tat: cooperates in round 1; afterwards, defects iff a
  majority of co-players defected last round (majority-rule generalization
  of "mirror the opponent's last move" -- with N-1 > 1 co-players there is
  no single move to mirror, so this aggregates them by majority vote)."""
  num_players = state.get_game().num_players()
  others = [p for p in range(num_players) if p != player]
  other_histories = [_own_action_history(state, p) for p in others]
  if not other_histories[0]:
    return Action.COOPERATE
  num_defectors = sum(1 for h in other_histories if h[-1] == Action.DEFECT)
  return Action.DEFECT if 2 * num_defectors > len(others) else Action.COOPERATE


def grim_trigger(state, player):
  """N-player Grim Trigger: cooperates until *any* co-player has *ever*
  defected, then defects forever against everyone. Since actions here are
  not addressed to a specific opponent, this can't retaliate selectively
  against just the defector(s) -- one defection poisons the well for the
  whole group, which is a harsher generalization than the 2-player original."""
  num_players = state.get_game().num_players()
  others = [p for p in range(num_players) if p != player]
  ever_defected = any(
      Action.DEFECT in _own_action_history(state, p) for p in others)
  return Action.DEFECT if ever_defected else Action.COOPERATE


BOTS = [tit_for_tat, grim_trigger, always_cooperate, always_defect]


def main(_):
  rng = random.Random(FLAGS.seed)
  game = pyspiel.load_game(
      "python_param_social_dilemma", {
          "players": FLAGS.players,
          "termination_probability": FLAGS.termination_probability,
          "payoff_kind": FLAGS.payoff_kind,
          "reward_noise_std": FLAGS.reward_noise_std,
          "dynamic_payoffs": FLAGS.dynamic_payoffs,
          "payoff_regimes": FLAGS.payoff_regimes,
          "payoff_change_prob": FLAGS.payoff_change_prob,
          "pgg_endowment": FLAGS.pgg_endowment,
          "pgg_multiplier": FLAGS.pgg_multiplier,
      })
  player_bots = [BOTS[p % len(BOTS)] for p in range(game.num_players())]
  print("Players -> strategies: " +
        ", ".join(f"p{p}={bot.__name__}"
                   for p, bot in enumerate(player_bots)))

  state = game.new_initial_state()
  round_num = 0
  while not state.is_terminal():
    if state.current_player() == pyspiel.PlayerId.CHANCE:
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(rng.choices(outcomes, weights=probs)[0])
      continue
    actions = [bot(state, p) for p, bot in enumerate(player_bots)]
    round_num += 1
    print(f"Round {round_num}: " + ", ".join(
        f"p{p}={Action(a).name}" for p, a in enumerate(actions)))
    state.apply_actions(actions)
    print(f"  Rewards: {list(state.rewards())}")

  print("\nFinal returns:")
  for p, bot in enumerate(player_bots):
    print(f"  p{p} ({bot.__name__}): {state.returns()[p]:.2f}")


if __name__ == "__main__":
  app.run(main)
