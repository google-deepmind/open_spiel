import logging
import typing
from copy import deepcopy
from functools import partial

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from jax import grad, vmap

from open_spiel.python import rl_agent
from open_spiel.python.rl_environment import TimeStep


@chex.dataclass
class TransitionBatch:
    info_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    terminal: np.ndarray
    legal_actions_mask: np.ndarray


class TrainState(typing.NamedTuple):
    policy_params: typing.List[hk.Params]
    critic_params: typing.List[hk.Params]
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState


UpdateFn = typing.Callable[[TrainState, TransitionBatch], typing.Tuple[TrainState, typing.Dict]]


def get_critic_update_fn(agent_id: int, critic_network: hk.Transformed, optimizer: optax.TransformUpdateFn) -> UpdateFn:
    """
    Returns the update function for the critic parameters.
    Args:
        agent_id: The id of the agent that will be updated.
        critic_network: A transformed haiku function.
        optimizer: Optimizer update function

    Returns:
        An update function that takes the current train state together with a transition batch and returns the new
        train state and a dictionary of metrics.
    """

    def loss_fn(params, batch: TransitionBatch):
        discounted_returns = vmap(partial(rlax.discounted_returns, stop_target_gradients=True))
        info_states, rewards = batch.info_state[agent_id], batch.reward[agent_id]
        discounts = batch.discount
        values = jnp.squeeze(critic_network.apply(params, info_states))
        target = discounted_returns(r_t=rewards, discount_t=discounts, v_t=jax.lax.stop_gradient(values))
        td_error = values - target
        return 0.5 * jnp.mean(td_error ** 2)

    def update(train_state: TrainState, batch: TransitionBatch):
        params = train_state.critic_params[agent_id]
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer(grads, train_state.critic_opt_state)
        critic_params = optax.apply_updates(params, updates)
        new_params = deepcopy(train_state.critic_params)
        new_params[agent_id] = critic_params
        new_state = train_state \
            ._replace(critic_params=new_params) \
            ._replace(critic_opt_state=opt_state)
        return new_state, dict(loss=loss)

    return update


def get_policy_update_fn(agent_id: int, policy_network: hk.Transformed, critic_network: hk.Transformed,
                         optimizer: optax.TransformUpdateFn, pi_lr: float, lola_weight: float) -> UpdateFn:
    def compute_lola_correction(train_state: TrainState, batch: TransitionBatch):
        """
        Computes the correction term according to Foerster et al. (2018).
        Args:
            train_state: the agent's train state.
            batch: a transition batch

        Returns:
            The correction term in the same format as the policy parameters.
        """
        # Read and store data
        params, unravel_policy_params = jax.flatten_util.ravel_pytree(train_state.policy_params[agent_id])
        opp_params, unravel_opp_policy_params = jax.flatten_util.ravel_pytree(train_state.policy_params[1 - agent_id])
        a_t, opp_a_t = batch.action[agent_id], batch.action[1 - agent_id]
        obs1, obs2 = batch.info_state[agent_id], batch.info_state[1 - agent_id]
        r_t, opp_r_t = batch.reward[agent_id], batch.reward[1 - agent_id]
        v_t = critic_network.apply(train_state.critic_params[agent_id], obs1).squeeze()
        opp_v_t = critic_network.apply(train_state.critic_params[1 - agent_id], obs2).squeeze()
        # Compute discounted sum of rewards
        compute_return = vmap(rlax.discounted_returns)
        G_t = compute_return(r_t=r_t, discount_t=batch.discount, v_t=jnp.zeros_like(r_t)) - v_t
        opp_G_t = compute_return(r_t=opp_r_t, discount_t=batch.discount, v_t=jnp.zeros_like(opp_r_t)) - opp_v_t

        # Standardize returns
        G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)
        opp_G_t = (opp_G_t - opp_G_t.mean()) / (opp_G_t.std() + 1e-8)

        def log_pi(params, o_t, a_t):
            return policy_network.apply(unravel_policy_params(params), o_t).log_prob(a_t)

        # Compute gradient of agent loss w.r.t opponent parameters
        G_grad_opp_params = grad(lambda param: (G_t * log_pi(param, obs2, opp_a_t)).mean())(opp_params)

        # Compute second order correction term according to (A.1) in https://arxiv.org/abs/1709.04326
        traj_log_prob = lambda params, o_t, a_t: log_pi(params, o_t, a_t).sum(-1)
        grad_log_pi = vmap(grad(traj_log_prob), in_axes=(None, 0, 0))(params, obs1, a_t)
        opp_grad_log_pi = vmap(grad(traj_log_prob), in_axes=(None, 0, 0))(opp_params, obs2, opp_a_t)
        jacobian = vmap(lambda R, a, b: R[0] * jnp.outer(a, b))(opp_G_t, grad_log_pi, opp_grad_log_pi)
        second_order_term = jacobian.mean(0)

        # scale by learning rate
        update = pi_lr * (G_grad_opp_params @ second_order_term)
        return unravel_policy_params(update)

    def policy_update(train_state: TrainState, batch: TransitionBatch):
        """
        Computes the vanilla policy gradient update.
        Args:
            train_state: the agent's train state.
            batch: a transition batch

        Returns:
            A tuple (loss, gradients).
        """
        def loss(params):
            r_t = batch.reward[agent_id]
            a_t = batch.action[agent_id]
            o_t = batch.info_state[agent_id]
            v_t = jnp.squeeze(critic_network.apply(train_state.critic_params[agent_id], o_t))
            logits = policy_network.apply(params, o_t).logits
            returns = vmap(partial(rlax.discounted_returns))
            R_t = returns(r_t=r_t, discount_t=batch.discount, v_t=v_t)
            loss = vmap(rlax.policy_gradient_loss)(logits, a_t, R_t, v_t)
            return loss.mean()

        value, grads = jax.value_and_grad(loss)(train_state.policy_params[agent_id])
        return value, grads

    def update(train_state: TrainState, batch: TransitionBatch) -> typing.Tuple[TrainState, typing.Dict]:
        """
        Updates the policy parameters in train_state. If lola_weight > 0, the correction term according to
        Foerster et al. will be applied.
        Args:
             train_state: the agent's train state.
            batch: a transition batch

        Returns:
            A tuple (new_train_state, metrics)
        """
        loss, policy_grads = policy_update(train_state, batch)
        if lola_weight > 0:
            gradient_correction = compute_lola_correction(train_state, batch)
            policy_grads = jax.tree_util.tree_map(lambda g, c: g - lola_weight * c, policy_grads, gradient_correction)

        updates, opt_state = optimizer(policy_grads, train_state.policy_opt_state)
        policy_params = optax.apply_updates(train_state.policy_params[agent_id], updates)
        new_policy_params = deepcopy(train_state.policy_params)
        new_policy_params[agent_id] = policy_params
        train_state = train_state._replace(policy_params=new_policy_params)._replace(policy_opt_state=opt_state)
        return train_state, dict(loss=loss)

    return update


class LolaPolicyGradientAgent(rl_agent.AbstractAgent):

    def __init__(self,
                 player_id: int,
                 opponent_ids: typing.List[int],
                 info_state_size: chex.Shape,
                 num_actions: int,
                 policy: hk.Transformed,
                 critic: hk.Transformed,
                 batch_size: int = 16,
                 critic_learning_rate: typing.Union[float, optax.Schedule] = 0.01,
                 pi_learning_rate: typing.Union[float, optax.Schedule] = 0.001,
                 lola_weight: float = 1.0,
                 clip_grad_norm: float = 0.5,
                 policy_update_interval: int = 8,
                 discount: float = 0.99,
                 seed: jax.random.PRNGKey = 42,
                 use_jit: bool = False):

        self.player_id = player_id
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._policy_update_interval = policy_update_interval
        self._discount = discount
        self._prev_time_step = None
        self._prev_action = None
        self._data = []
        self._metrics = []
        self._opponent_ids = opponent_ids
        self._rng = hk.PRNGSequence(seed)

        # Step counters
        self._step_counter = 0
        self._episode_counter = 0
        self._num_learn_steps = 0

        self._pi_network = policy
        self._critic_network = critic
        self._critic_opt = optax.sgd(learning_rate=critic_learning_rate)
        self._policy_opt = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm) if clip_grad_norm else optax.identity(),
            optax.sgd(learning_rate=pi_learning_rate)
        )
        self._train_state = self._init_train_state(info_state_size=info_state_size)
        self._current_policy = self.get_policy(return_probs=True)

        policy_update_fn = get_policy_update_fn(
            agent_id=player_id,
            policy_network=policy,
            critic_network=critic,
            pi_lr=pi_learning_rate,
            lola_weight=lola_weight,
            optimizer=self._policy_opt.update
        )
        critic_update_fn = get_critic_update_fn(
            agent_id=player_id,
            critic_network=critic,
            optimizer=self._critic_opt.update
        )
        if use_jit:
            self._policy_update_fn = jax.jit(policy_update_fn)
            self._critic_update_fn = jax.jit(critic_update_fn)
        else:
            self._policy_update_fn = policy_update_fn
            self._critic_update_fn = critic_update_fn

    @property
    def train_state(self):
        return deepcopy(self._train_state)

    @property
    def metrics(self):
        if len(self._metrics) > 0:
            return jax.tree_util.tree_map(lambda *xs: np.mean(np.array(xs)), *self._metrics)
        else:
            return {}

    def update_params(self, state: TrainState, player_id: int) -> None:
        """
        Updates the parameters of the other agents.
        Args:
            state: the train state of the other agent.
            player_id: id of the other agent

        Returns:

        """
        self._train_state.policy_params[player_id] = state.policy_params[player_id]
        self._train_state.critic_params[player_id] = state.critic_params[player_id]

    def get_policy(self, return_probs=True) -> typing.Callable:
        """
        Returns a function that takes a random key, an observation and optionally an action mask. The function produces
        actions which are sampled from the current policy. Additionally, if return_probs is true, it also returns the
        action probabilities.
        Args:
            return_probs: if true, the policy returns a tuple (action, action_probs).

        Returns: A function that maps observations to actions

        """
        def _policy(key: jax.random.PRNGKey, obs: jnp.ndarray, action_mask=None):
            """
            Takes a random key, the current observation and optionally an action mask.
            Args:
                key: a random key for sampling
                obs: numpy array of observations
                action_mask: optional numpy array to mask out illegal actions

            Returns: Either the sampled actions or, if return_probs is true, a tuple (actions, action_probs).

            """
            params = self._train_state.policy_params[self.player_id]
            logits = self._pi_network.apply(params, obs).logits
            probs = jax.nn.softmax(logits, axis=-1)
            if action_mask is None:
                action_mask = jnp.ones_like(probs)
            probs = probs * action_mask
            probs = probs / probs.sum()
            action_dist = distrax.Categorical(probs=probs)
            actions = action_dist.sample(seed=key)
            if return_probs:
                return actions, action_dist.prob(actions)
            else:
                return actions

        return jax.jit(_policy)

    def step(self, time_step: TimeStep, is_evaluation=False):
        """
        Produces an action and possible triggers a parameter update. LOLA agents depend on having access to previous
        actions made by the opponent. Assumes that the field "observations" of time_step contains a field "actions" and
        its first axis is indexed by the player id.
        Similar, the fields "rewards" and "legal_actions" are assumed to be of shape (num_players,).

        Args:
            time_step: a TimeStep instance which has a field "actions" in the observations dict.
            is_evaluation: if true, the agent will not update.

        Returns: a tuple containing the action that was taken and its probability under the current policy

        """
        do_step = time_step.is_simultaneous_move() or self.player_id == time_step.current_player()
        action, probs = None, []
        if not time_step.last() and do_step:
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]
            action_mask = np.zeros(self._num_actions)
            action_mask[legal_actions] = 1
            action, probs = self._current_policy(
                key=next(self._rng),
                obs=jnp.asarray(info_state),
                action_mask=action_mask
            )

        if not is_evaluation:
            self._store_time_step(time_step=time_step, action=action)
            if time_step.last() and self._should_update():
                self._train_step()

        return rl_agent.StepOutput(action=action, probs=probs)

    def _init_train_state(self, info_state_size: chex.Shape):
        init_inputs = jnp.ones(info_state_size)
        number_of_agents = len(self._opponent_ids) + 1
        policy_params = [self._pi_network.init(next(self._rng), init_inputs) for _ in range(number_of_agents)]
        critic_params = [self._critic_network.init(next(self._rng), init_inputs) for _ in range(number_of_agents)]
        policy_opt_state = self._policy_opt.init(policy_params[self.player_id])
        critic_opt_state = self._critic_opt.init(critic_params[self.player_id])
        return TrainState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state
        )

    def _store_time_step(self, time_step: TimeStep, action: np.ndarray):
        """
        Converts the timestep and the action into a transition and steps the counters.
        Args:
            time_step: the current time step.
            action: the action that was taken before observing time_step

        Returns: None

        """
        self._step_counter += 1
        if self._prev_time_step:
            transition = self._make_transition(time_step)
            self._data.append(transition)
        if time_step.last():
            self._prev_time_step = None
            self._prev_action = None
            self._episode_counter += 1
        else:
            self._prev_time_step = time_step
            self._prev_action = action

    def _train_step(self):
        """
        Updates the critic and the policy parameters. After the update, the data buffer is cleared.
        Returns:
        """
        logging.info(f"Updating agent {self.player_id}.")
        batch = self._construct_episode_batches(self._data)
        update_metrics = self._update_agent(batch)
        self._metrics.append(update_metrics)
        self._data.clear()

    def _should_update(self) -> bool:
        """
        Indicates whether to update or not.
        Returns: True, if the number of episodes in the buffer is equal to the batch size. False otherwise.
        """
        return self._episode_counter % self._batch_size == 0 and self._episode_counter > 0

    def _update_agent(self, batch: TransitionBatch) -> typing.Dict:
        """
        Updates the critic and policy parameters of the agent.
        Args:
            batch: A batch of training episodes.

        Returns:
            A dictionary that contains relevant training metrics.
        """
        metrics = {}
        self._num_learn_steps += 1
        critic_update_metrics = self._update_critic(batch)
        metrics.update((f'critic/{k}', v) for k, v in critic_update_metrics.items())
        if self._num_learn_steps % self._policy_update_interval == 0:
            policy_update_metrics = self._update_policy(batch)
            metrics.update((f'policy/{k}', v) for k, v in policy_update_metrics.items())
        return metrics

    def _construct_episode_batches(self, transitions: typing.List[TransitionBatch]) -> TransitionBatch:
        """
        Constructs a list of transitions into a single transition batch instance.
        The fields "info_state", "rewards", "legal_action_mask" and "actions" of the produced transition batch have
        shape (num_agents, batch_size, sequence_length, *shape).
        The fields "discount" and "terminal" have shape (batch_size, sequence_length).

        Args:
            transitions: a list of single step transitions

        Returns:
            A transition batch instance with items of according shape.
        """
        episode, batches = [], []
        max_episode_length = 0
        for transition in transitions:
            episode.append(transition)
            if transition.terminal:
                max_episode_length = max(max_episode_length, len(episode))
                batch = jax.tree_map(lambda *xs: jnp.stack(xs), *episode)
                batches.append(batch)
                episode.clear()
        padded = jax.tree_util.tree_map(lambda x: jnp.pad(x, pad_width=max_episode_length - len(x)), batches)
        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *padded)
        batch = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 2, 0) if len(x.shape) > 2 else x, batch)
        return batch

    def _update_policy(self, batch: TransitionBatch):
        self._train_state, metrics = self._policy_update_fn(self._train_state, batch)
        self._current_policy = self.get_policy(return_probs=True)
        return metrics

    def _update_critic(self, batch: TransitionBatch):
        self._train_state, metrics = self._critic_update_fn(self._train_state, batch)
        return metrics

    def _make_transition(self, time_step: TimeStep):
        assert self._prev_time_step is not None
        legal_actions = self._prev_time_step.observations["legal_actions"][self.player_id]
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1
        actions = np.array(time_step.observations["actions"])
        rewards = np.array(time_step.rewards)
        obs = np.array(self._prev_time_step.observations["info_state"])
        transition = TransitionBatch(
            info_state=obs,
            action=actions,
            reward=rewards,
            discount=self._discount * (1 - time_step.last()),
            terminal=time_step.last(),
            legal_actions_mask=legal_actions_mask
        )
        return transition
