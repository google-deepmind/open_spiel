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
from aim import Run

from open_spiel.python.environments.iterated_matrix_game import IteratedPrisonersDilemma
from open_spiel.python.jax.lola import LolaPolicyGradientAgent
from open_spiel.python.rl_environment import Environment, TimeStep

warnings.simplefilter('ignore', FutureWarning)

"""
Example that trains two agents using LOLA (Foerster et al., 2017) and LOLA-DiCE (Foerster et al., 2018)
on iterated matrix games. Hyperparameters are taken from the paper and https://github.com/alexis-jacq/LOLA_DiCE.
"""
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("game", "matrix_pd", "Name of the game.")
flags.DEFINE_integer("epochs", 200, "Number of training iterations.")
flags.DEFINE_integer("batch_size", 1024, "Number of episodes in a batch.")
flags.DEFINE_integer("critic_mini_batches", 1, "Number of minibatches for critic.")
flags.DEFINE_integer("game_iterations", 150, "Number of iterated plays.")
flags.DEFINE_float("policy_lr", 0.1, "Policy learning rate.")
flags.DEFINE_float("opp_policy_lr", 0.1, "Policy learning rate.")
flags.DEFINE_float("critic_lr", 0.3, "Critic learning rate.")
flags.DEFINE_string("correction_type", 'dice', "Either 'lola', 'dice' or None.")
flags.DEFINE_integer("n_lookaheads", 1, "Number of lookaheads for LOLA correction.")
flags.DEFINE_float("correction_max_grad_norm", None, "Maximum gradient norm of LOLA correction.")
flags.DEFINE_float("discount", 0.96, "Discount factor.")
flags.DEFINE_integer("policy_update_interval", 1, "Number of critic updates per before policy is updated.")
flags.DEFINE_integer("eval_batch_size", 30, "Random seed.")
flags.DEFINE_bool("use_jit", False, "If true, JAX jit compilation will be enabled.")
flags.DEFINE_bool("use_opponent_modelling", True, "If false, ground truth opponent weights are used.")
flags.DEFINE_integer("opp_policy_mini_batches", 8, "Number of minibatches for opponent policy.")

def log_epoch_data(run: Run, epoch: int, agents: List[LolaPolicyGradientAgent], eval_batch):
    def get_action_probs(agent: LolaPolicyGradientAgent) -> List[str]:
        states = ['s0', 'CC', 'CD', 'DC', 'DD']
        prob_strings = []
        params = agent.train_state.policy_params[agent.player_id]
        for i, s in enumerate(states):
            state = np.eye(len(states))[i]
            prob = agent.policy_network.apply(params, state).prob(0)
            prob_strings.append(f'P(C|{s})={prob:.3f}')
            run.track(prob.item(), name=f'P(C|{s})', context={'agent': agent.player_id})
        return prob_strings

    for agent in agents:
        avg_step_reward = np.mean([ts.rewards[agent.player_id] for ts in eval_batch])
        probs = get_action_probs(agent)
        probs = ', '.join(probs)
        run.track(avg_step_reward, name='avg_step_reward', context={'agent': agent.player_id})
        print(f'[epoch {epoch}] Agent {agent.player_id}: {avg_step_reward:.2f} | {probs}')


def collect_batch(env: Environment, agents: List[LolaPolicyGradientAgent], eval: bool):
    def get_values(time_step: TimeStep, agent: LolaPolicyGradientAgent) -> jnp.ndarray:
        v_fn = agent.get_value_fn()
        return jax.vmap(v_fn)(time_step.observations["info_state"][agent.player_id])

    episode = []
    time_step = env.reset()
    episode.append(time_step)
    while not time_step.last():
        values = np.stack([get_values(time_step, agent) for agent in agents], axis=0)
        time_step.observations["values"] = values
        actions = [agent.step(time_step, is_evaluation=eval).action for agent in agents]
        time_step = env.step(np.stack(actions, axis=1))
        time_step.observations["actions"] = np.stack(actions, axis=0)
        episode.append(time_step)

    for agent in agents:
        agent.step(time_step, is_evaluation=eval)
    return episode


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
        num_critic_mini_batches=FLAGS.critic_mini_batches,
        pi_learning_rate=FLAGS.policy_lr,
        opp_policy_learning_rate=FLAGS.opp_policy_lr,
        num_opponent_updates=FLAGS.opp_policy_mini_batches,
        critic_learning_rate=FLAGS.critic_lr,
        policy_update_interval=FLAGS.policy_update_interval,
        discount=FLAGS.discount,
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
        return distrax.Categorical(logits=logits)

    def value_fn(obs):
        w = hk.get_parameter("w", [num_states], init=jnp.zeros)
        return w[jnp.argmax(obs, axis=-1)].reshape(*obs.shape[:-1], 1)

    return hk.without_apply_rng(hk.transform(policy)), hk.without_apply_rng(hk.transform(value_fn))

def make_env(iterations: int, batch_size: int):
    return IteratedPrisonersDilemma(iterations=iterations, batch_size=batch_size)

def setup_agents(env: Environment, rng: hk.PRNGSequence) -> List[LolaPolicyGradientAgent]:
    agents = []
    num_actions = env.action_spec()["num_actions"]
    info_state_shape = env.observation_spec()["info_state"]
    for player_id in range(env.num_players):
        networks = make_agent_networks(num_states=info_state_shape[player_id][0], num_actions=num_actions[player_id])
        agent = make_agent(key=next(rng), player_id=player_id, env=env, networks=networks)
        agents.append(agent)
    return agents

def update_weights(agents: List[LolaPolicyGradientAgent]):
    for agent in agents:
        for opp in filter(lambda a: a.player_id != agent.player_id, agents):
            agent.update_params(state=opp.train_state, player_id=opp.player_id)


def main(_):
    run = Run(experiment='opponent_shaping')
    run["hparams"] = {
        "seed": FLAGS.seed,
        "batch_size": FLAGS.batch_size,
        "game_iterations": FLAGS.game_iterations,
        "with_opp_modelling": FLAGS.use_opponent_modelling,
        "discount": FLAGS.discount,
        "policy_lr": FLAGS.policy_lr,
        "opp_policy_lr": FLAGS.opp_policy_lr,
        "critic_lr": FLAGS.critic_lr,
        "policy_update_interval": FLAGS.policy_update_interval,
        "correction_type": FLAGS.correction_type,
        "correction_max_grad_norm": FLAGS.correction_max_grad_norm,
        "n_lookaheads": FLAGS.n_lookaheads,
        "use_jit": FLAGS.use_jit
    }

    rng = hk.PRNGSequence(key_or_seed=FLAGS.seed)
    env = make_env(iterations=FLAGS.game_iterations, batch_size=FLAGS.batch_size)
    agents = setup_agents(env=env, rng=rng)

    if not FLAGS.use_opponent_modelling:
        update_weights(agents)

    batch = collect_batch(env=env, agents=agents, eval=True)
    log_epoch_data(epoch=0, agents=agents, run=run, eval_batch=batch)
    for epoch in range(1, FLAGS.epochs+1):
        batch = collect_batch(env=env, agents=agents, eval=False)
        if not FLAGS.use_opponent_modelling:
            update_weights(agents)
        log_epoch_data(epoch=epoch, agents=agents, run=run, eval_batch=batch)
        print('#' * 100)

if __name__ == "__main__":
    app.run(main)
