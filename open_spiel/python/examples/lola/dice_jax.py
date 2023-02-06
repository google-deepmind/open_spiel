# coding: utf-8
import random
import time
from functools import partial
from typing import Optional, Union, List, Tuple, NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import distrax
from copy import deepcopy
import flax.linen as nn
from tqdm import tqdm

from open_spiel.python.environments import iterated_matrix_game_jax, iterated_matrix_game


class Hp():
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.lr_v = 0.1
        self.gamma = 0.96
        self.n_update = 200
        self.len_rollout = 150
        self.batch_size = 128
        self.use_baseline = True
        self.seed = 42


hp = Hp()
env = iterated_matrix_game.IteratedPrisonersDilemmaEnv(iterations=hp.len_rollout, batch_size=hp.batch_size, include_remaining_iterations=False)
#env_step, env_reset = iterated_matrix_game_jax.make_env_fns(env=env, max_iters=hp.len_rollout, batch_size=hp.batch_size,
  #                                                          payoffs=env._payoff_matrix)


def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))


class Memory():
    def __init__(self):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []
        self.states = []

    def add(self, s, lp, other_lp, v, r):
        self.states.append(s)
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

@jax.jit
def dice_objective(self_logprobs, other_logprobs, values, rewards):
    self_logprobs = jnp.stack(self_logprobs, axis=1)
    other_logprobs = jnp.stack(other_logprobs, axis=1)
    values = jnp.stack(values, axis=1)
    rewards = jnp.stack(rewards, axis=1)

    # apply discount:
    cum_discount = jnp.cumprod(hp.gamma * jnp.ones_like(rewards), axis=1) / hp.gamma
    discounted_rewards = rewards * cum_discount
    discounted_values = values * cum_discount

    # stochastics nodes involved in rewards dependencies:
    dependencies = jnp.cumsum(self_logprobs + other_logprobs, axis=1)

    # logprob of each stochastic nodes:
    stochastic_nodes = self_logprobs + other_logprobs

    # dice objective:
    dice_objective = jnp.mean(jnp.sum(magic_box(dependencies) * discounted_rewards, axis=1))

    if hp.use_baseline:
        # variance_reduction:
        baseline_term = jnp.mean(jnp.sum((1 - magic_box(stochastic_nodes)) * discounted_values, axis=1))
        dice_objective = dice_objective + baseline_term

    return -dice_objective  # want to minimize -objective

@jax.jit
def act(key, batch_states, theta, values):
    batch_states = jnp.array(batch_states, dtype=int)
    logits = jax.vmap(lambda s: jnp.select(s, theta))(batch_states)
    v = jax.vmap(lambda s: jnp.select(s, values))(batch_states)
    m = distrax.Categorical(logits=logits)
    actions = m.sample(seed=key)
    log_probs_actions = m.log_prob(actions)
    return actions.astype(int), log_probs_actions, v

def inner_objective(theta, other_theta, values, other_values, key):
    step = env.reset()
    states, self_lp, other_lp, vs, rs = [], [], [], [], []
    for t in range(hp.len_rollout):
        s1, s2 = step.observations['info_state'][0], step.observations['info_state'][1]
        key, k1, k2 = jax.random.split(key, num=3)
        a1, lp1, v1 = act(k1, s1, theta, values)
        a2, lp2, v2 = act(k2, s2, other_theta, other_values)
        action = jax.lax.stop_gradient(jnp.stack([a1, a2], axis=1))
        step = env.step(action)
        r1, r2 = step.rewards[0], step.rewards[1]
        states.append(s2)
        self_lp.append(lp2)
        other_lp.append(lp1)
        vs.append(v2)
        rs.append(r2)


    return dice_objective(self_lp, other_lp, vs, rs)


def step(key, theta1, theta2, values1, values2):
    # just to evaluate progress:
    step = env.reset()
    score1 = 0
    score2 = 0
    for t in range(hp.len_rollout):
        key, k1, k2 = jax.random.split(key, num=3)
        s1, s2 = step.observations['info_state'][0], step.observations['info_state'][1]
        a1, lp1, v1 = act(k1, s1, theta1, values1)
        a2, lp2, v2 = act(k2, s2, theta2, values2)
        step = env.step(np.array(jnp.stack([a1, a2], axis=1)))
        # cumulate scores
        score1 += np.mean(step.rewards[0]) / float(hp.len_rollout)
        score2 += np.mean(step.rewards[1]) / float(hp.len_rollout)
    return (score1, score2)


class Agent():
    def __init__(self, key):
        # init theta and its optimizer
        self.key = key
        self.theta = jnp.zeros((5, 2))
        self.theta_optimizer = optax.adam(learning_rate=hp.lr_out)
        self.theta_opt_state = self.theta_optimizer.init(self.theta)
        # init values and its optimizer
        self.values = jnp.zeros(5)
        self.value_optimizer = optax.adam(learning_rate=hp.lr_v)
        self.value_opt_state = self.value_optimizer.init(self.values)

    def theta_update(self, objective, other_theta, other_values, key):
        grads, memory = jax.grad(objective, has_aux=True)(self.theta, other_theta, self.values, other_values, key)
        updates, opt_state = self.theta_optimizer.update(grads, self.theta_opt_state)
        self.theta = optax.apply_updates(self.theta, updates)
        self.theta_opt_state = opt_state
        return memory

    def value_update(self, states, rewards):
        def loss(params):
            s = jnp.stack(states, axis=1)
            rew = jnp.stack(rewards, axis=1)
            values = jax.vmap(jax.vmap(lambda s: jnp.select(s, params)))(s)
            return jnp.mean((rew - values) ** 2)

        grads = jax.grad(loss)(self.values)
        updates, opt_state = self.value_optimizer.update(grads, self.value_opt_state)
        self.values = optax.apply_updates(self.values, updates)
        self.value_opt_state = opt_state


    def out_lookahead(self, other_theta, other_values, n_lookaheads):
        def inner(theta, other_theta, values, other_values, key):
            other_theta = other_theta.copy()
            for k in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                key, k_in = jax.random.split(key)
                other_grad = jax.grad(inner_objective, argnums=1)(theta, other_theta, values, other_values, k_in)
                # update other's theta
                other_theta = other_theta - hp.lr_in * other_grad

            key, k_out = jax.random.split(key)

            step = env.reset()
            states, lp1s, lp2s, vs, rs = [], [], [], [], []
            for t in range(hp.len_rollout):
                s1, s2 = step.observations['info_state'][0], step.observations['info_state'][1]
                key, k1, k2 = jax.random.split(key, num=3)
                a1, lp1, v1 = act(k1, s1, theta, values)
                a2, lp2, v2 = act(k2, s2, other_theta, other_values)
                step = env.step(jnp.stack([a1, a2], axis=1))
                r1, r2 = step.rewards[0], step.rewards[1]
                states.append(s1)
                lp1s.append(lp1)
                lp2s.append(lp2)
                vs.append(v1)
                rs.append(r1)
            return dice_objective(lp1s, lp2s, vs, rs), dict(states=states, lp1s=lp1s, lp2s=lp2s, values=vs, rewards=rs)


        key, k_out = jax.random.split(self.key)
        start = time.time()
        grads, memory = jax.grad(inner, has_aux=True)(self.theta, other_theta, self.values, other_values, k_out)
        end = time.time()
        #print("out lookahead took", end - start, "seconds")
        updates, opt_state = self.theta_optimizer.update(grads, self.theta_opt_state)
        self.theta = optax.apply_updates(self.theta, updates)
        self.theta_opt_state = opt_state
        self.value_update(memory['states'], memory['rewards'])


def play(key, agent1, agent2, n_lookaheads):
    joint_scores = []

    print("start iterations with", n_lookaheads, "lookaheads:")
    for update in tqdm(range(hp.n_update)):
        start = time.time()
        # copy other's parameters:
        theta1_ = jnp.array(agent1.theta)
        values1_ = jnp.array(agent1.values)
        theta2_ = jnp.array(agent2.theta)
        values2_ = jnp.array(agent2.values)

        agent1.out_lookahead(theta2_, values2_, n_lookaheads)
        agent2.out_lookahead(theta1_, values1_, n_lookaheads)

        # evaluate progress:
        key, sample_key = jax.random.split(key)
        score = step(sample_key, agent1.theta, agent2.theta, agent1.values, agent2.values)
        joint_scores.append(0.5 * (score[0] + score[1]))

        # print
        states = jnp.eye(5, dtype=int)
        if update % 10 == 0:
            p1 = [distrax.Categorical(logits=agent1.theta[i]).prob(0).item() for i in range(5)]
            p2 = [distrax.Categorical(logits=agent2.theta[i]).prob(0).item() for i in range(5)]
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]),
                  'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),
                  ' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
        end = time.time()
        #print("loop time:", end - start, "seconds")


    return joint_scores


# plot progress:
if __name__ == "__main__":

    colors = ['b', 'c', 'm', 'r']
    for i in range(0, 4):
        key, play_key, agent1_key, agent2_key = jax.random.split(jax.random.PRNGKey(hp.seed), num=4)
        scores = play(play_key, Agent(agent1_key), Agent(agent2_key), i)
        plt.plot(scores, colors[i], label=str(i) + " lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()
