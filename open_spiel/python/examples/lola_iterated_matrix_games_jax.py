import logging
import random
import warnings
from typing import List, Tuple

import distrax
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
from open_spiel.python.rl_agent import AbstractAgent

warnings.simplefilter('ignore', FutureWarning)

"""
Example that trains two agents using LOLA (Foerster et al., 2018) on iterated matrix games. Hyperparameters are taken from
the paper. 
"""
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", random.randint(0, 10000000), "Random seed.")
flags.DEFINE_string("game", "matrix_pd", "Name of the game.")
flags.DEFINE_integer("epochs", 1000, "Number of training iterations.")
flags.DEFINE_integer("batch_size", 64, "Number of episodes in a batch.")
flags.DEFINE_integer("game_iterations", 150, "Number of iterated plays.")
flags.DEFINE_float("policy_lr", 0.0005, "Policy learning rate.")
flags.DEFINE_float("critic_lr", 1.0, "Critic learning rate.")
flags.DEFINE_float("lola_weight", 1.0, "Weighting factor for the LOLA correction term. Zero resembles standard PG.")
flags.DEFINE_float("correction_max_grad_norm", None, "Maximum gradient norm of LOLA correction.")
flags.DEFINE_float("discount", 0.96, "Discount factor.")
flags.DEFINE_integer("policy_update_interval", 2, "Number of critic updates per before policy is updated.")
flags.DEFINE_integer("eval_batch_size", 30, "Random seed.")
flags.DEFINE_bool("use_jit", False, "If true, JAX jit compilation will be enabled.")


def log_epoch_data(epoch: int, agent: LolaPolicyGradientAgent, env: Environment, eval_batch, policy_network):
    def get_action_probs(policy_params: hk.Params, num_actions: int) -> List[str]:
        states = jnp.concatenate([jnp.zeros((1, num_actions * 2)), jnp.eye(num_actions * 2)], axis=0)
        logits = policy_network.apply(policy_params, states).logits
        probs = jax.nn.softmax(logits, axis=1)
        prob_strings = []
        for i, name in enumerate(['s0', 'CC', 'CD', 'DC', 'DD']):
            prob_strings.append(f'P(C|{name})={probs[i][0]:.3f}')
        return prob_strings

    avg_step_reward = np.mean([[time_step.rewards[agent.player_id] for time_step in episode] for episode in eval_batch])
    stats = dict(avg_step_reward=avg_step_reward)
    num_actions = env.action_spec()['num_actions']
    episode_stats = ','.join(f'{k}={v:.2f}' for k, v in stats.items())
    action_probs = get_action_probs(policy_params=agent.train_state.policy_params[agent.player_id],
                                    num_actions=num_actions)
    probs = ', '.join(action_probs)
    print(f'[epoch {epoch}] Agent {agent.player_id}: {episode_stats} | {probs}')


def append_action(env: rl_environment.Environment, timestep: rl_environment.TimeStep) -> rl_environment.TimeStep:
    observations = timestep.observations.copy()
    info_states = timestep.observations["info_state"]
    if timestep.first():
        observations["current_player"] = pyspiel.PlayerId.SIMULTANEOUS
    observations["actions"] = []
    for i, info_state in enumerate(info_states):
        observations["actions"].append(np.argmax(info_state[i * env.num_players:(i + 1) * env.num_players]))
    observations["legal_actions"] = [np.arange(env.num_actions_per_step) for _ in range(env.num_players)]
    return timestep._replace(observations=observations)


def collect_batch(env: Environment, agents: List[AbstractAgent], n_episodes: int, eval: bool):
    episodes = []
    for _ in range(n_episodes):
        time_step = env.reset()
        episode = []
        while not time_step.last():
            agents_output, action_list = [], []
            for agent in agents:
                output = agent.step(time_step, is_evaluation=eval)
                agents_output.append(output)
                action_list.append(output.action)
            time_step = env.step(action_list)
            time_step = append_action(env=env, timestep=time_step)
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
        info_state_size=env.observation_spec()["info_state"],
        num_actions=env.action_spec()["num_actions"],
        policy=policy_network,
        critic=critic_network,
        batch_size=FLAGS.batch_size,
        pi_learning_rate=FLAGS.policy_lr,
        critic_learning_rate=FLAGS.critic_lr,
        policy_update_interval=FLAGS.policy_update_interval,
        discount=FLAGS.discount,
        correction_weight=FLAGS.lola_weight,
        clip_grad_norm=FLAGS.correction_max_grad_norm,
        use_jit=FLAGS.use_jit
    )


def make_agent_networks(num_actions: int) -> Tuple[hk.Transformed, hk.Transformed]:
    def policy(obs):
        logits = hk.nets.MLP(output_sizes=[num_actions], with_bias=True)(obs)
        return distrax.Categorical(logits=logits)

    def value_fn(obs):
        values = hk.nets.MLP(output_sizes=[1], with_bias=True)(obs)
        return values

    return hk.without_apply_rng(hk.transform(policy)), hk.without_apply_rng(hk.transform(value_fn))


def make_iterated_matrix_game(game: str, config: dict) -> rl_environment.Environment:
    logging.info("Creating game %s", FLAGS.game)
    matrix_game = pyspiel.load_matrix_game(game)
    game = pyspiel.create_repeated_game(matrix_game, config)
    env = rl_environment.Environment(game)
    logging.info("Env specs: %s", env.observation_spec())
    logging.info("Action specs: %s", env.action_spec())
    return env


def update_weights(agent: LolaPolicyGradientAgent, opponent: LolaPolicyGradientAgent):
    agent.update_params(state=opponent.train_state, player_id=opponent.player_id)
    opponent.update_params(state=agent.train_state, player_id=agent.player_id)


def main(_):
    print(FLAGS.seed)
    env_config = {"num_repetitions": FLAGS.game_iterations, "batch_size": FLAGS.batch_size}
    rng = hk.PRNGSequence(key_or_seed=FLAGS.seed)
    for experiment in range(10):
        env = make_iterated_matrix_game(FLAGS.game, env_config)
        networks = make_agent_networks(num_actions=env.action_spec()["num_actions"])
        policy_network, critic_network = networks

        agents = [make_agent(key=next(rng), player_id=i, env=env, networks=networks) for i in range(env.num_players)]
        update_weights(agents[0], agents[1])

        for epoch in range(FLAGS.epochs):
            batch = collect_batch(env=env, agents=agents, n_episodes=FLAGS.batch_size, eval=False)
            update_weights(agents[0], agents[1])

            for agent in agents:
                log_epoch_data(epoch=epoch, agent=agent, env=env, eval_batch=batch, policy_network=policy_network)

        print('#' * 100)

if __name__ == "__main__":
    app.run(main)
