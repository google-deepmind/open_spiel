# Bots and RL agents

OpenSpiel has two interfaces for computer players:

* A **bot** chooses actions from a [`State`](concepts.html#the-tree-representation).
  Use bots for playing or evaluating games directly, including search-based or
  external-engine players. The interface is available in C++ as
  `open_spiel::Bot` and in Python as `pyspiel.Bot`.
* A **reinforcement-learning (RL) agent** consumes a
  `rl_environment.TimeStep`. Use RL agents in Python training and evaluation
  loops where observations, rewards, discounts, and episode boundaries are
  needed.

Bots and RL agents receive different inputs. To use a bot in an RL environment,
wrap it as an RL agent and create the environment with
`include_full_state=True`; see
[`MCTSAgent`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/mcts_agent.py)
for an example.

## Bots

The complete bot contract is defined in
[`spiel_bots.h`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel_bots.h).
Method names use `CamelCase` in C++ and `snake_case` in Python.

* `Step` / `step` (**required**): choose an action for the supplied state. The
  bot may update its internal state as though the action will be played.
* `StepVerbose` / `step_verbose`: optionally return an action and diagnostic
  text. The default calls `Step` and returns an empty string.
* `InformAction` / `inform_action`: tell a stateful bot about an action chosen
  by another player or by chance. Call it before applying the action to the
  state.
* `InformActions` / `inform_actions`: tell a stateful bot about the joint action
  at a simultaneous-move state.
* `Restart` / `restart`: reset the bot before a new trajectory.
* `RestartAt` / `restart_at`: reset the bot at an arbitrary state. Override this
  when the bot supports starting from non-root states.
* `ProvidesPolicy` / `provides_policy`: return `true` only when `GetPolicy` and
  `StepWithPolicy` are implemented.
* `GetPolicy` / `get_policy`: return action-probability pairs without selecting
  an action.
* `StepWithPolicy` / `step_with_policy`: return both action-probability pairs
  and the selected action.
* `ProvidesForceAction` / `provides_force_action`: return `true` only when
  `ForceAction` is implemented.
* `ForceAction` / `force_action`: make a stateful bot accept a specified action
  instead of choosing one.
* `IsClonable` / `is_clonable`: return `true` only when `Clone` is implemented.
* `Clone` / `clone`: make an independent copy. Randomness in the copy must be
  reseeded so samples are not correlated with the original.

### Creating a Python bot

Subclass `pyspiel.Bot`, initialize the base class, and implement `step`. Store a
player ID when the bot can act in simultaneous-move games or is dedicated to a
specific player.

```python
import pyspiel


class FirstLegalBot(pyspiel.Bot):

  def __init__(self, player_id):
    pyspiel.Bot.__init__(self)
    self._player_id = player_id

  def step(self, state):
    return state.legal_actions(self._player_id)[0]
```

Stateless bots can rely on the default no-op `restart` and `inform_action`
implementations. Stateful bots should override the synchronization methods they
need. During a turn, call `step` only on the acting bot, notify the other bots
with `inform_action`, and then apply the action. This ordering matters because
`inform_action` receives the state from before the action was applied.

The built-in evaluator handles restarts, turn-based action notifications,
chance nodes, and action collection for simultaneous nodes:

```python
game = pyspiel.load_game("tic_tac_toe")
bots = [
    FirstLegalBot(0),
    pyspiel.make_uniform_random_bot(1, 7),
]
returns = pyspiel.evaluate_bots(game.new_initial_state(), bots, seed=42)
```

Stateful simultaneous-move bots that depend on `inform_actions` need a custom
loop that invokes that callback before applying the joint action.

For a complete game loop and several bot types, see the Python
[`mcts.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/mcts.py)
example. For smaller examples, see the Python
[`UniformRandomBot`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/bots/uniform_random.py),
the Python
[`bot_test.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/tests/bot_test.py)
examples that mix Python and C++ bots, and the C++
[`evaluate_bots_test.cc`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/evaluate_bots_test.cc)
usage example.

### Creating and registering a C++ bot

Derive from `open_spiel::Bot` and override `Step` plus any optional methods the
bot supports. If users should be able to construct it by a short name, also
implement `BotFactory::CanPlayGame` and `BotFactory::Create`, then register the
factory with `REGISTER_SPIEL_BOT`. The uniform-random bot and its factory in
[`spiel_bots.cc`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel_bots.cc)
provide a compact example.

Registered bots can be discovered with `RegisteredBots` or
`BotsThatCanPlayGame` and constructed with `LoadBot`. The Python equivalents
are `pyspiel.registered_bots()`, `pyspiel.bots_that_can_play_game(...)`, and
`pyspiel.load_bot(...)`.

## RL agents

RL agents implement
[`rl_agent.AbstractAgent`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_agent.py).
The interface has two required methods:

* `__init__(player_id, observation_spec=None, name="agent", **kwargs)`:
  initialize the agent. Concrete agents commonly require the environment's
  observation or action specification as well.
* `step(time_step, is_evaluation=False)`: update the agent and, at decision
  steps, return `rl_agent.StepOutput(action, probs)`. Evaluation mode should
  avoid training-only side effects such as adding transitions to replay buffers
  or decaying exploration.

`StepOutput.action` is an integer action ID. `StepOutput.probs` is normally a
vector indexed by action ID, with illegal actions assigned probability zero.
At the final time step there is no action to choose; learning agents can use
that call to observe the final reward before returning `None`.

The `TimeStep` passed to an agent contains:

* `observations["info_state"]`: one observation or information-state tensor per
  player;
* `observations["legal_actions"]`: one list of legal action IDs per player;
* `observations["current_player"]`: the acting player, the simultaneous-play
  sentinel, or the terminal sentinel;
* `rewards`, `discounts`, and `step_type`; and
* `observations["serialized_state"]`, containing serialized data when the
  environment was created with `include_full_state=True` and an empty list
  otherwise.

`Environment.reset()` returns a `FIRST` time step whose rewards and discounts
are `None`. The environment advances through chance and mean-field nodes before
returning the next agent decision; a terminal `LAST` time step has zero
discounts.

### Creating and using an RL agent

Here is a deterministic agent and a turn-based environment loop:

```python
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment


class FirstLegalAgent(rl_agent.AbstractAgent):

  def __init__(self, player_id, num_actions, name="first_legal"):
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    if time_step.last():
      return None
    legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = legal_actions[0]
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0
    return rl_agent.StepOutput(action=action, probs=probs)


env = rl_environment.Environment("tic_tac_toe")
num_actions = env.action_spec()["num_actions"]
agents = [FirstLegalAgent(player, num_actions) for player in range(2)]

time_step = env.reset()
while not time_step.last():
  player = time_step.observations["current_player"]
  output = agents[player].step(time_step)
  time_step = env.step([output.action])

# Let learning agents consume the terminal reward.
for agent in agents:
  agent.step(time_step)
```

For a simultaneous-move game, call every agent at each decision step and pass
one action per player to `env.step`. The
[`rl_main_loop.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/rl_main_loop.py)
example demonstrates both cases. See
[`random_agent.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/random_agent.py)
for a minimal implementation and
[`tic_tac_toe_qlearner.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/tic_tac_toe_qlearner.py)
for training, evaluation mode, and terminal-step handling.

To expose trained agents through the OpenSpiel `Policy` interface, use
`RLAgentPolicy` or `JointRLAgentPolicy` from
[`rl_agent_policy.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_agent_policy.py).
