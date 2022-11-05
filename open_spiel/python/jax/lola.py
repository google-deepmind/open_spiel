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
    values: np.ndarray = None


class TrainState(typing.NamedTuple):
    policy_params: typing.Dict[typing.Any, hk.Params]
    policy_opt_states: typing.Dict[typing.Any, optax.OptState]
    critic_opt_state: optax.OptState
    critic_params: hk.Params


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
        td_learning = vmap(partial(rlax.td_learning, stop_target_gradients=True))
        info_states, rewards = batch.info_state[agent_id], batch.reward[agent_id]
        discounts = batch.discount
        values = critic_network.apply(params, info_states)
        v_tm1 = values[:, :-1].reshape(-1)
        v_t = values[:, 1:].reshape(-1)
        r_t = rewards[:, 1:].reshape(-1)
        d_t = discounts[:, 1:].reshape(-1)
        td_error = td_learning(v_tm1=v_tm1, r_t=r_t, discount_t=d_t, v_t=v_t)
        return td_error.mean()

    def update(train_state: TrainState, batch: TransitionBatch):
        loss, grads = jax.value_and_grad(loss_fn)(train_state.critic_params, batch)
        updates, opt_state = optimizer(grads, train_state.critic_opt_state)
        critic_params = optax.apply_updates(train_state.critic_params, updates)
        new_state = train_state \
            ._replace(critic_params=critic_params) \
            ._replace(critic_opt_state=opt_state)
        return new_state, dict(loss=loss)

    return update


def get_policy_update_fn(agent_id: int, policy_network: hk.Transformed, critic_network: hk.Transformed,
                         optimizer: optax.TransformUpdateFn, pi_lr: float, correction_weight: float) -> UpdateFn:

    def dice_correction(train_state: TrainState, batch: TransitionBatch):

        def magic_box(x):
            return jnp.exp(x - jax.lax.stop_gradient(x))

        agent, opp = agent_id, 1-agent_id
        flat_param_dict = dict([(agent_id, jax.flatten_util.ravel_pytree(params)) for agent_id, params in train_state.policy_params.items()])
        params = dict((k, flat_param_dict[k][0]) for k in flat_param_dict)
        unravel_fns = dict((k, flat_param_dict[k][1]) for k in flat_param_dict)
        batch = jax.tree_util.tree_map(jnp.array, batch)
        a_t, o_t, r_t, values = batch.action, batch.info_state, batch.reward, batch.values

        # Compute advantages
        v_tp1, v_t = values[:, :, 1:], values[:, :, :-1]
        o_t, a_t = o_t[:, :, :-1], a_t[:, :, :-1]
        r_t = r_t[:, :, :-1]
        discounts = jnp.stack([batch.discount] * len(a_t), axis=0)[:, :, 1:]  # assume same discounts for all agents
        compute_return = vmap(vmap(partial(rlax.n_step_bootstrapped_returns, n=1, lambda_t=0.0)))
        G_t = compute_return(r_t=r_t, discount_t=discounts, v_t=v_tp1)
        adv_t = G_t - v_t

        # Standardize returns
        #adv_t = vmap(lambda x: (x - x.mean()) / (x.std() + 1e-8))(adv_t)

        def objective(params, opp_params, adv_t):
            agent_unravel = flat_param_dict[agent][1]
            opp_unravel = flat_param_dict[opp][1]
            logp = policy_network.apply(agent_unravel(params), o_t[agent]).log_prob(a_t[agent])
            opp_logp = policy_network.apply(opp_unravel(opp_params), o_t[opp]).log_prob(a_t[opp])
            cumlogp_t = logp.cumsum(-1)
            oppcumlogp_t = opp_logp.cumsum(-1)
            joint_cumlogp_t = magic_box(cumlogp_t + oppcumlogp_t)
            return (adv_t * joint_cumlogp_t).sum(-1).mean()

        # Define agent losses
        L0 = partial(objective, adv_t=adv_t[agent])
        L1 = partial(objective, adv_t=adv_t[opp])


        # Compute gradient of agent loss w.r.t opponent parameters
        pg_update = grad(L0, argnums=0)(params[agent], params[opp])
        L0_grad_opp_params = grad(L0, argnums=1)(params[agent], params[opp])

        # Compute jacobian of the opponent update step
        opp_update_fn = lambda params, opp_params: pi_lr * grad(L1, argnums=1)(params, opp_params)
        L1_grad_opp_params_grad_params = jax.jacobian(opp_update_fn, argnums=0)(params[agent], params[opp])

        # compute correction
        correction = pg_update + L0_grad_opp_params @ L1_grad_opp_params_grad_params
        return unravel_fns[agent](correction)

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
            values = jnp.squeeze(critic_network.apply(train_state.critic_params, o_t))
            v_t, v_tp1 = values[:, :-1], values[:, 1:]
            logits = policy_network.apply(params, o_t).logits
            compute_return = vmap(partial(rlax.n_step_bootstrapped_returns, n=1, lambda_t=0.0))
            G_t = compute_return(r_t=r_t[:, :-1], discount_t=batch.discount[:, :-1], v_t=v_tp1)
            adv_t = G_t - v_t
            loss = vmap(rlax.policy_gradient_loss)(logits[:, :-1], a_t[:, :-1], adv_t, jnp.ones_like(adv_t))
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
        if correction_weight > 0:
            gradient_correction = dice_correction(train_state, batch)
            policy_grads = jax.tree_util.tree_map(lambda g, c: -correction_weight * c, policy_grads, gradient_correction)

        updates, opt_state = optimizer(policy_grads, train_state.policy_opt_states[agent_id])
        policy_params = optax.apply_updates(train_state.policy_params[agent_id], updates)
        new_policy_params = deepcopy(train_state.policy_params)
        new_opt_states = deepcopy(train_state.policy_opt_states)
        new_policy_params[agent_id] = policy_params
        new_opt_states[agent_id] = opt_state
        train_state = train_state.\
            _replace(policy_params=new_policy_params).\
            _replace(policy_opt_states=new_opt_states)
        return train_state, dict(loss=loss)

    return update

def get_opponent_update_fn(agent_id: int, policy_network: hk.Transformed, optimizer: optax.TransformUpdateFn) -> UpdateFn:

    def loss_fn(params, batch: TransitionBatch):
        actions = batch.action[agent_id]
        log_prob = policy_network.apply(params, batch.info_state[agent_id]).log_prob(actions)
        return -log_prob.sum(axis=-1).mean()

    def update(train_state: TrainState, batch: TransitionBatch) -> typing.Tuple[TrainState, typing.Dict]:
        loss, policy_grads = jax.value_and_grad(loss_fn)(train_state.policy_params[agent_id], batch)
        updates, opt_state = optimizer(policy_grads, train_state.policy_opt_states[agent_id])
        policy_params = optax.apply_updates(train_state.policy_params[agent_id], updates)
        new_policy_params = deepcopy(train_state.policy_params)
        new_opt_states = deepcopy(train_state.policy_opt_states)
        new_policy_params[agent_id] = policy_params
        new_opt_states[agent_id] = opt_state
        train_state = train_state. \
            _replace(policy_params=new_policy_params). \
            _replace(policy_opt_states=new_opt_states)
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
                 opponent_model_learning_rate: typing.Union[float, optax.Schedule] = 0.001,
                 correction_weight: float = 1.0,
                 clip_grad_norm: float = 0.5,
                 policy_update_interval: int = 8,
                 discount: float = 0.99,
                 seed: jax.random.PRNGKey = 42,
                 fit_opponent_model = True,
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
        self._fit_opponent_model = fit_opponent_model
        self._opponent_ids = opponent_ids
        self._rng = hk.PRNGSequence(seed)

        # Step counters
        self._step_counter = 0
        self._episode_counter = 0
        self._num_learn_steps = 0

        self._pi_network = policy
        self._critic_network = critic
        self._critic_opt = optax.sgd(learning_rate=critic_learning_rate)
        self._opponent_opt = optax.adam(opponent_model_learning_rate)
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
            correction_weight=correction_weight,
            optimizer=self._policy_opt.update
        )

        critic_update_fn = get_critic_update_fn(
            agent_id=player_id,
            critic_network=critic,
            optimizer=self._critic_opt.update
        )

        self._policy_update_fns = {}

        if use_jit:
            self._policy_update_fns[player_id] = jax.jit(policy_update_fn)
            self._critic_update_fn = jax.jit(critic_update_fn)
        else:
            self._policy_update_fns[player_id] = policy_update_fn
            self._critic_update_fn = critic_update_fn

        for opponent in opponent_ids:
            opp_update_fn = get_opponent_update_fn(agent_id=opponent, policy_network=policy, optimizer=self._opponent_opt.update)
            if use_jit:
                self._policy_update_fns[opponent] = jax.jit(opp_update_fn)
            else:
                self._policy_update_fns[opponent] = opp_update_fn
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

    def get_value_fn(self) -> typing.Callable:
        def value_fn(obs: jnp.ndarray):
            obs = jnp.array(obs)
            return self._critic_network.apply(self.train_state.critic_params, obs).squeeze(-1)
        return jax.jit(value_fn)

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
        agent_ids = self._opponent_ids + [self.player_id]
        policy_params, policy_opt_states = {}, {}
        for agent_id in agent_ids:
            policy_params[agent_id] = self._pi_network.init(next(self._rng), init_inputs)
            if agent_id == self.player_id:
                policy_opt_state = self._policy_opt.init(policy_params[agent_id])
            else:
                policy_opt_state = self._opponent_opt.init(policy_params[agent_id])
            policy_opt_states[agent_id] = policy_opt_state

        critic_params = self._critic_network.init(next(self._rng), init_inputs)
        critic_opt_state = self._critic_opt.init(critic_params)
        return TrainState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_states=policy_opt_states,
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
        opponent_update_metrics = self._update_opponents(batch)
        critic_update_metrics = self._update_critic(batch)
        metrics.update((f'critic/{k}', v) for k, v in critic_update_metrics.items())
        metrics.update((f'opponents/{k}', v) for k, v in opponent_update_metrics.items())
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
        self._train_state, metrics = self._policy_update_fns[self.player_id](self._train_state, batch)
        self._current_policy = self.get_policy(return_probs=True)
        return metrics

    def _update_critic(self, batch: TransitionBatch):
        self._train_state, metrics = self._critic_update_fn(self._train_state, batch)
        return metrics

    def _update_opponents(self, batch: TransitionBatch):
        update_metrics = {}
        for opponent in self._opponent_ids:
            self._train_state, metrics = self._policy_update_fns[opponent](self._train_state, batch)
            update_metrics.update({f'agent_{opponent}/{k}': v for k, v in metrics.items()})
        return update_metrics

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
            legal_actions_mask=legal_actions_mask,
            values=self._prev_time_step.observations["values"]
        )
        return transition
