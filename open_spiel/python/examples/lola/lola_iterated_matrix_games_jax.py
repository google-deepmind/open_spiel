import logging
import random
import typing
import warnings
from typing import List, Tuple

import aim
from aim import Run
import distrax
import haiku
import haiku as hk
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pyspiel
from absl import app
from absl import flags
from dm_env import Environment

from open_spiel.python import rl_environment
from open_spiel.python.jax.lola import LolaPolicyGradientAgent

warnings.simplefilter('ignore', FutureWarning)

"""
Example that trains two agents using LOLA (Foerster et al., 2018) on iterated matrix games. Hyperparameters are taken from
the paper. 
"""
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", random.choice([42]), "Random seed.")
flags.DEFINE_string("game", "matrix_pd", "Name of the game.")
flags.DEFINE_integer("epochs", 1000, "Number of training iterations.")
flags.DEFINE_integer("batch_size", 128, "Number of episodes in a batch.")
flags.DEFINE_integer("game_iterations", 150, "Number of iterated plays.")
flags.DEFINE_float("policy_lr", 0.1, "Policy learning rate.")
flags.DEFINE_float("critic_lr", 0.3, "Critic learning rate.")
flags.DEFINE_string("correction_type", 'dice', "Either 'lola', 'dice' or None.")
flags.DEFINE_float("correction_max_grad_norm", None, "Maximum gradient norm of LOLA correction.")
flags.DEFINE_float("discount", 0.96, "Discount factor.")
flags.DEFINE_integer("policy_update_interval", 1, "Number of critic updates per before policy is updated.")
flags.DEFINE_integer("eval_batch_size", 30, "Random seed.")
flags.DEFINE_bool("use_jit", False, "If true, JAX jit compilation will be enabled.")
flags.DEFINE_bool("use_opponent_modelling", False, "If false, ground truth opponent weights are used.")
flags.DEFINE_bool("include_remaining_iterations", False, "If true, the percentage of the remaining iterations are included in the observations.")
def log_epoch_data(run: Run, epoch: int, agent: LolaPolicyGradientAgent, env: Environment, eval_batch, policy_network):
    def get_action_probs(policy_params: hk.Params) -> List[str]:
        states = ['s0', 'CC', 'CD', 'DC', 'DD']
        prob_strings = []
        for i, s in enumerate(states):
            state = np.eye(len(states))[i]
            prob = policy_network.apply(policy_params, state).prob(0)
            prob_strings.append(f'P(C|{s})={prob:.3f}')
            run.track(prob.item(), name=f'P(C|{s})', context={'agent': agent.player_id})
        return prob_strings

    avg_step_reward = np.mean([[time_step.rewards[agent.player_id] for time_step in episode] for episode in eval_batch])
    stats = dict(avg_step_reward=avg_step_reward)
    episode_stats = ','.join(f'{k}={v:.2f}' for k, v in stats.items())
    action_probs = get_action_probs(policy_params=agent.train_state.policy_params[agent.player_id])
    probs = ', '.join(action_probs)
    run.track(avg_step_reward, name='avg_step_reward', context={'agent': agent.player_id})
    print(f'[epoch {epoch}] Agent {agent.player_id}: {episode_stats} | {probs}')


def collect_batch(env: Environment, agents: List[LolaPolicyGradientAgent], n_episodes: int, eval: bool):
    def postprocess(timestep: rl_environment.TimeStep, actions: typing.List) -> rl_environment.TimeStep:
        observations = timestep.observations.copy()

        if timestep.first():
            observations["current_player"] = pyspiel.PlayerId.SIMULTANEOUS
        observations["actions"] = []

        values = []
        for agent in sorted(agents, key=lambda a: a.player_id):
            v_fn = agent.get_value_fn()
            values.append(jax.vmap(v_fn)(observations["info_state"][agent.player_id]))

        observations["values"] = jnp.stack(values, axis=0)
        observations["actions"] = actions
        return timestep._replace(observations=observations)

    episodes = []
    for _ in range(n_episodes):
        time_step = env.reset()
        t = 0
        time_step = postprocess(time_step, actions=None)
        episode = []
        while not time_step.last():
            agents_output, action_list = [], []
            for agent in agents:
                output = agent.step(time_step, is_evaluation=eval)
                agents_output.append(output)
                action_list.append(output.action)
            actions = np.stack(action_list, axis=1)
            time_step = env.step(actions)
            t += 1
            time_step = postprocess(timestep=time_step, actions=action_list)
            episode.append(time_step)

        for agent in agents:
            agent.step(time_step, is_evaluation=eval)
        episodes.append(episode)

    return episodes


def make_agent(key: jax.random.PRNGKey, player_id: int, env: Environment,
               networks: Tuple[hk.Transformed, hk.Transformed]):
    policy_network, critic_network = networks
    return LolaPolicyGradientAgent(
        player_id=player_id,
        opponent_ids=[1 - player_id],
        seed=key,
        info_state_size=env.observation_spec()["info_state"][player_id],
        num_actions=env.action_spec()["num_actions"][player_id],
        policy=policy_network,
        critic=critic_network,
        batch_size=FLAGS.batch_size,
        pi_learning_rate=FLAGS.policy_lr,
        critic_learning_rate=FLAGS.critic_lr,
        policy_update_interval=FLAGS.policy_update_interval,
        discount=FLAGS.discount,
        correction_type=FLAGS.correction_type,
        clip_grad_norm=FLAGS.correction_max_grad_norm,
        use_jit=FLAGS.use_jit,
        env=env
    )


def make_agent_networks(num_actions: int) -> Tuple[hk.Transformed, hk.Transformed]:
    def policy(obs):
        # w_init=haiku.initializers.Constant(1), b_init=haiku.initializers.Constant(0)
        theta = hk.get_parameter('theta', init=haiku.initializers.Constant(0), shape=(5,2))
        logits = jnp.select(obs, theta)
        return distrax.Categorical(logits=logits)

    def value_fn(obs):
        w = hk.get_parameter("w", [5], init=jnp.zeros)
        return w[jnp.argmax(obs, axis=-1)].reshape(*obs.shape[:-1], 1)

    return hk.without_apply_rng(hk.transform(policy)), hk.without_apply_rng(hk.transform(value_fn))

def make_env(iterations: int, batch_size: int, jitted: bool = False):
    if jitted:
        from open_spiel.python.environments.iterated_matrix_game_jax import IteratedPrisonersDilemma
    else:
        from open_spiel.python.environments.iterated_matrix_game import IteratedPrisonersDilemma
    return IteratedPrisonersDilemma(iterations=iterations, batch_size=batch_size)

def update_weights(agent: LolaPolicyGradientAgent, opponent: LolaPolicyGradientAgent):
    agent.update_params(state=opponent.train_state, player_id=opponent.player_id)
    opponent.update_params(state=agent.train_state, player_id=agent.player_id)


def main(_):
    run = Run(experiment='lola')
    run["hparams"] = {
        "seed": FLAGS.seed,
        "batch_size": FLAGS.batch_size,
        "discount": FLAGS.discount,
        "policy_lr": FLAGS.policy_lr,
        "critic_lr": FLAGS.critic_lr,
        "policy_update_interval": FLAGS.policy_update_interval,
        "correction_type": FLAGS.correction_type,
        "correction_max_grad_norm": FLAGS.correction_max_grad_norm,
        "use_jit": FLAGS.use_jit
    }

    rng = hk.PRNGSequence(key_or_seed=FLAGS.seed)
    for experiment in range(1):
        env = make_env(iterations=FLAGS.game_iterations, batch_size=FLAGS.batch_size, jitted=False)
        agents = []
        for player_id in range(env.num_players):
            networks = make_agent_networks(num_actions=env.action_spec()["num_actions"][player_id])
            policy_network, critic_network = networks
            agent = make_agent(key=next(rng), player_id=player_id, env=env, networks=networks)
            agents.append(agent)

        update_weights(agents[0], agents[1])
        batch = collect_batch(env=env, agents=agents, n_episodes=1, eval=True)
        for agent in agents:
            log_epoch_data(epoch=0, run=run, agent=agent, env=env, eval_batch=batch, policy_network=policy_network)
        for epoch in range(1, FLAGS.epochs+1):
            batch = collect_batch(env=env, agents=agents, n_episodes=1, eval=False)
            for agent in agents:
                for k, v in agent._metrics[-1].items():
                    #run.track(v, name=k, context={"agent": agent.player_id})
                    pass

            update_weights(agents[0], agents[1])

            for agent in agents:
                log_epoch_data(epoch=epoch, agent=agent, run=run, env=env, eval_batch=batch, policy_network=policy_network)
        print('#' * 100)


if __name__ == "__main__":
    app.run(main)
