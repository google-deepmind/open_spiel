import itertools
import os
import typing
import warnings
from typing import List, Tuple

import distrax
import haiku
import haiku as hk
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from absl import app
from absl import flags
import wandb


from open_spiel.python.environments.iterated_matrix_game import IteratedPrisonersDilemma, IteratedMatchingPennies
from open_spiel.python.jax.opponent_shaping import OpponentShapingAgent
from open_spiel.python.rl_environment import Environment, TimeStep

warnings.simplefilter('ignore', FutureWarning)

"""
Example that trains two agents using LOLA (Foerster et al., 2017) and LOLA-DiCE (Foerster et al., 2018)
on iterated matrix games. Hyperparameters are taken from the paper and https://github.com/alexis-jacq/LOLA_DiCE.
"""
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", 'dice_1step_pytorchparams', "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("game", "ipd", "Name of the game.")
flags.DEFINE_integer("epochs", 200, "Number of training iterations.")
flags.DEFINE_integer("batch_size", 128, "Number of episodes in a batch.")
flags.DEFINE_integer("critic_mini_batches", 1, "Number of minibatches for critic.")
flags.DEFINE_integer("game_iterations", 150, "Number of iterated plays.")
flags.DEFINE_float("policy_lr", 0.2, "Policy learning rate.")
flags.DEFINE_float("opp_policy_lr", 0.3, "Policy learning rate.")
flags.DEFINE_float("critic_lr", 0.1, "Critic learning rate.")
flags.DEFINE_string("correction_type", 'dice', "Either 'opponent_shaping', 'dice' or None.")
flags.DEFINE_integer("n_lookaheads", 2, "Number of lookaheads for LOLA correction.")
flags.DEFINE_float("correction_max_grad_norm", None, "Maximum gradient norm of LOLA correction.")
flags.DEFINE_float("discount", 0.96, "Discount factor.")
flags.DEFINE_integer("policy_update_interval", 1, "Number of critic updates per before policy is updated.")
flags.DEFINE_integer("eval_batch_size", 1024, "Random seed.")
flags.DEFINE_bool("use_jit", False, "If true, JAX jit compilation will be enabled.")
flags.DEFINE_bool("use_opponent_modelling", True, "If false, ground truth opponent weights are used.")
flags.DEFINE_integer("opp_policy_mini_batches", 8, "Number of minibatches for opponent policy.")
flags.DEFINE_float("opponent_model_learning_rate", 0.3, "Learning rate for opponent model.")
flags.DEFINE_bool("debug", False, "If true, debug mode is enabled.")

def get_action_probs(agent: OpponentShapingAgent, game: str) -> List[typing.Dict[str, typing.Any]]:
    actions = ['C', 'D'] if game == 'ipd' else ['H', 'T']
    states = ['s0'] + [''.join(s) for s in itertools.product(actions, repeat=2)]
    params = agent.train_state.policy_params[agent.player_id]
    action_probs = []
    for i, s in enumerate(states):
        state = np.eye(len(states))[i]
        prob = agent.policy_network.apply(params, state).prob(0)
        action = actions[0]
        action_probs.append(dict(prob=prob.item(), name=f'P({action}|{s})'))
    return action_probs
def log_epoch_data(epoch: int, agents: List[OpponentShapingAgent], eval_batch):
    logs = {}
    for agent in agents:
        avg_step_reward = np.mean([ts.rewards[agent.player_id] for ts in eval_batch])
        probs = get_action_probs(agent, game=FLAGS.game)
        for info in probs:
            logs[f'agent_{agent.player_id}/{info["name"]}'] = info['prob']
        probs = ', '.join([f'{info["name"]}: {info["prob"]:.2f}' for info in probs])
        metrics = agent.metrics()
        logs.update({
            f'agent_{agent.player_id}/avg_step_reward': avg_step_reward,
            **{f'agent_{agent.player_id}/{k}': v.item() for k, v in metrics.items()}
        })
        print(f'[epoch {epoch}] Agent {agent.player_id}: {avg_step_reward:.2f} | {probs}')
    wandb.log(logs)


def collect_batch(env: Environment, agents: List[OpponentShapingAgent], eval: bool):
    episode = []
    time_step = env.reset()
    episode.append(time_step)
    while not time_step.last():
        actions = []
        for agent in agents:
            action, _ = agent.step(time_step, is_evaluation=eval)
            if action is not None:
                action = action.squeeze()
            actions.append(action)
        time_step = env.step(np.stack(actions, axis=1))
        time_step.observations["actions"] = actions
        episode.append(time_step)

    for agent in agents:
        agent.step(time_step, is_evaluation=eval)
    return episode


def make_agent(key: jax.random.PRNGKey, player_id: int, env: Environment,
               networks: Tuple[hk.Transformed, hk.Transformed]):
    policy_network, critic_network = networks
    return OpponentShapingAgent(
        player_id=player_id,
        opponent_ids=[1 - player_id],
        seed=key,
        info_state_size=env.observation_spec()["info_state"][player_id],
        num_actions=env.action_spec()["num_actions"][player_id],
        policy=policy_network,
        critic=critic_network,
        batch_size=FLAGS.batch_size,
        num_critic_mini_batches=FLAGS.critic_mini_batches,
        pi_learning_rate=FLAGS.policy_lr,
        opp_policy_learning_rate=FLAGS.opp_policy_lr,
        num_opponent_updates=FLAGS.opp_policy_mini_batches,
        critic_learning_rate=FLAGS.critic_lr,
        opponent_model_learning_rate=FLAGS.opponent_model_learning_rate,
        policy_update_interval=FLAGS.policy_update_interval,
        discount=FLAGS.discount,
        critic_discount=0, # Predict only the immediate reward (only for iterated matrix games)
        correction_type=FLAGS.correction_type,
        clip_grad_norm=FLAGS.correction_max_grad_norm,
        use_jit=FLAGS.use_jit,
        n_lookaheads=FLAGS.n_lookaheads,
        env=env
    )


def make_agent_networks(num_states: int, num_actions: int) -> Tuple[hk.Transformed, hk.Transformed]:
    def policy(obs):
        theta = hk.get_parameter('theta', init=haiku.initializers.Constant(0), shape=(num_states, num_actions))
        logits = jnp.select(obs, theta)
        logits = jnp.nan_to_num(logits)
        return distrax.Categorical(logits=logits)

    def value_fn(obs):
        w = hk.get_parameter("w", [num_states], init=jnp.zeros)
        return w[jnp.argmax(obs, axis=-1)].reshape(*obs.shape[:-1], 1)

    return hk.without_apply_rng(hk.transform(policy)), hk.without_apply_rng(hk.transform(value_fn))

def make_env(game: str, iterations: int, batch_size: int):
    if game == 'ipd':
        return IteratedPrisonersDilemma(iterations=iterations, batch_size=batch_size)
    elif game == 'imp':
        return IteratedMatchingPennies(iterations=iterations, batch_size=batch_size)

def setup_agents(env: Environment, rng: hk.PRNGSequence) -> List[OpponentShapingAgent]:
    agents = []
    num_actions = env.action_spec()["num_actions"]
    info_state_shape = env.observation_spec()["info_state"]
    for player_id in range(env.num_players):
        networks = make_agent_networks(num_states=info_state_shape[player_id][0], num_actions=num_actions[player_id])
        agent = make_agent(key=next(rng), player_id=player_id, env=env, networks=networks)
        agents.append(agent)
    return agents

def update_weights(agents: List[OpponentShapingAgent]):
    for agent in agents:
        for opp in filter(lambda a: a.player_id != agent.player_id, agents):
            agent.update_params(state=opp.train_state, player_id=opp.player_id)


def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = f'{FLAGS.game}_{FLAGS.seed}'
    wandb.login(key=os.environ.get('WANDB_API_KEY', None))
    wandb.init(
        project='open-spiel-opponent-modelling',
        group=FLAGS.exp_name,
        config={
            'game': FLAGS.game,
            'seed': FLAGS.seed,
            'epochs': FLAGS.epochs,
            'batch_size': FLAGS.batch_size,
            'critic_mini_batches': FLAGS.critic_mini_batches,
            'game_iterations': FLAGS.game_iterations,
            'policy_lr': FLAGS.policy_lr,
            'opp_policy_lr': FLAGS.opp_policy_lr,
            'critic_lr': FLAGS.critic_lr,
            'correction_type': FLAGS.correction_type,
            'n_lookaheads': FLAGS.n_lookaheads,
            'correction_max_grad_norm': FLAGS.correction_max_grad_norm,
            'discount': FLAGS.discount,
            'policy_update_interval': FLAGS.policy_update_interval,
            'use_opponent_modelling': FLAGS.use_opponent_modelling,
            'opp_policy_mini_batches': FLAGS.opp_policy_mini_batches,
            'opponent_model_learning_rate': FLAGS.opponent_model_learning_rate
        },
        mode='disabled' if FLAGS.debug else 'online'
    )

    rng = hk.PRNGSequence(key_or_seed=FLAGS.seed)
    env = make_env(iterations=FLAGS.game_iterations, batch_size=FLAGS.batch_size, game=FLAGS.game)
    agents = setup_agents(env=env, rng=rng)

    if not FLAGS.use_opponent_modelling:
        update_weights(agents)

    batch = collect_batch(env=env, agents=agents, eval=True)
    log_epoch_data(epoch=0, agents=agents, eval_batch=batch)
    for epoch in range(1, FLAGS.epochs+1):
        batch = collect_batch(env=env, agents=agents, eval=False)
        if not FLAGS.use_opponent_modelling:
            update_weights(agents)
        log_epoch_data(epoch=epoch, agents=agents, eval_batch=batch)
        print('#' * 100)

    wandb.finish()

if __name__ == "__main__":
    app.run(main)
