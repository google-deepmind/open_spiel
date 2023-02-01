import logging
import typing
from copy import deepcopy
from functools import partial

import chex
import distrax
import haiku
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from jax import grad, vmap
from open_spiel.python import rl_agent
from open_spiel.python.environments.iterated_matrix_game import IteratedPrisonersDilemmaEnv, IteratedMatrixGame
from open_spiel.python.rl_environment import TimeStep


@chex.dataclass
class TransitionBatch:
    info_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray = None
    terminal: np.ndarray = None
    legal_actions_mask: np.ndarray = None
    values: np.ndarray = None

class TrainState(typing.NamedTuple):
    policy_params: typing.Dict[typing.Any, hk.Params]
    policy_opt_states: typing.Dict[typing.Any, optax.OptState]
    critic_opt_state: optax.OptState
    critic_params: typing.Dict[typing.Any, hk.Params]


UpdateFn = typing.Callable[[TrainState, TransitionBatch], typing.Tuple[TrainState, typing.Dict]]


def get_critic_update_fn(agent_id: int, critic_network: hk.Transformed, optimizer: optax.TransformUpdateFn, num_minibatches: int = 8) -> UpdateFn:
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
        discounts = jnp.stack([batch.discount[agent_id]] * rewards.shape[0], axis=0)
        values = critic_network.apply(params, info_states).squeeze()
        v_tm1 = values[:, :-1].reshape(-1)
        v_t = values[:, 1:].reshape(-1)
        r_t = rewards[:, 1:].reshape(-1)
        d_t = discounts[:, 1:].reshape(-1)
        td_error = jax.lax.stop_gradient(r_t + d_t * v_t) - v_tm1
        #return jnp.square(td_error).mean()
        return jnp.mean((values - rewards) ** 2)

    def update(train_state: TrainState, batch: TransitionBatch):
        losses = []
        critic_params = train_state.critic_params[agent_id]
        opt_state = train_state.critic_opt_state[agent_id]
        for i in range(num_minibatches):
            start, end = i * (batch.reward.shape[1] // num_minibatches), (i + 1) * (batch.reward.shape[1] // num_minibatches)#
            mini_batch = jax.tree_util.tree_map(lambda x: x[:, start:end] if len(x.shape) > 2 else x, batch)
            loss, grads = jax.value_and_grad(loss_fn)(critic_params, mini_batch)
            updates, opt_state = optimizer(grads, opt_state)
            critic_params = optax.apply_updates(critic_params, updates)
            losses.append(loss)
        new_params = deepcopy(train_state.critic_params)
        new_opt_states = deepcopy(train_state.critic_opt_state)
        new_params[agent_id] = critic_params
        new_opt_states[agent_id] = opt_state
        state = train_state \
                ._replace(critic_params=new_params) \
                ._replace(critic_opt_state=new_opt_states)
        return state, dict(loss=jnp.mean(jnp.array(losses)).item())

    return update


def get_policy_update_fn(agent_id: int, rng: hk.PRNGSequence, policy_network: hk.Transformed, critic_network: hk.Transformed,
                         optimizer: optax.TransformUpdateFn, pi_lr: float, correction_type='lola') -> UpdateFn:

    def flat_params(params):
        flat_param_dict = dict([(agent_id, jax.flatten_util.ravel_pytree(p)) for agent_id, p in params.items()])
        params = dict((k, flat_param_dict[k][0]) for k in flat_param_dict)
        unravel_fns = dict((k, flat_param_dict[k][1]) for k in flat_param_dict)
        return params, unravel_fns
    def lola_correction(train_state: TrainState, batch: TransitionBatch) -> haiku.Params:
        a_t, o_t, r_t, values = batch.action, batch.info_state, batch.reward, batch.values
        params, unravel_fns = flat_params(train_state.policy_params)

        compute_returns = partial(rlax.lambda_returns, discount_t=batch.discount, lambda_=1.0)
        G_t = vmap(vmap(compute_returns))(r_t=r_t, v_t=values)
        b_t = G_t.mean(axis=1, keepdims=True)
        G_t = G_t - b_t

        log_pi = lambda params, i, a_t, o_t: policy_network.apply(unravel_fns[i](params), o_t).log_prob(a_t)
        grad_log_pi = vmap(vmap(grad(log_pi, argnums=0), in_axes=(None, None, 0, 0)), in_axes=(None, None, 0, 0))
        id, opp_id = agent_id, 1 - agent_id

        grad_log_pi_1 = grad_log_pi(params[id], id, a_t[id], o_t[id])
        grad_log_pi_2 = grad_log_pi(params[opp_id], opp_id, a_t[opp_id], o_t[opp_id])
        cross_term = vmap(jnp.outer)(grad_log_pi_1.sum(1), grad_log_pi_2.sum(1))
        cross_term = vmap(jnp.multiply)(G_t[opp_id, :, 0], cross_term).mean(0)
        G_theta_2 = vmap(vmap(jnp.multiply))(grad_log_pi_2, G_t[id]).sum(axis=1).mean(0)
        G_theta_1 = vmap(vmap(jnp.multiply))(grad_log_pi_1, G_t[id]).sum(axis=1).mean(0)
        gradients = -(G_theta_1 + pi_lr * G_theta_2 @ cross_term)
        return unravel_fns[id](gradients)


    def dice_correction(train_state: TrainState, batch: TransitionBatch):

        def magic_box(x):
            return jnp.exp(x - jax.lax.stop_gradient(x))

        agent, opp = agent_id, 1-agent_id
        params, unravel_fns = flat_params(train_state.policy_params)
        batch = jax.tree_util.tree_map(jnp.array, batch)
        a_t, o_t, r_t, values = batch.action, batch.info_state, batch.reward, batch.values
        compute_return = vmap(partial(rlax.lambda_returns, lambda_=1.0, discount_t=batch.discount))

        def objective(params, opp_params, id, opp_id):
            logp = policy_network.apply(unravel_fns[id](params), o_t[id]).log_prob(a_t[id])
            opp_logp = policy_network.apply(unravel_fns[opp_id](opp_params), o_t[opp_id]).log_prob(a_t[opp_id])
            dependencies = jnp.cumsum(logp + opp_logp, axis=-1)
            G_t = compute_return(r_t=r_t[id], v_t=values[id])
            # G_t = r_t + gamma * r_tp1 + gamma^2 * r_tp2 + ...

            cum_discount = jnp.cumprod(0.96 * jnp.ones_like(r_t[id]), axis=0) / 0.96
            G_t = r_t[id] * cum_discount
            b_t = G_t.mean(axis=0, keepdims=True)

            dice_obj = jnp.mean(jnp.sum(magic_box(dependencies) * G_t, axis=-1))
            baseline = jnp.mean(jnp.sum((1-magic_box(logp + opp_logp)) * b_t, axis=-1))
            dice_obj = jnp.mean(jnp.sum(magic_box(dependencies) * (G_t - b_t), axis=-1)) #dice_obj + baseline
            return dice_obj

        # Define agent losses
        L0 = partial(objective, id=agent, opp_id=opp)
        L1 = partial(objective, id=opp, opp_id=agent)

        # Compute opponent gradient
        def obj(params, opp_params):
            opp_update = grad(lambda p: objective(p, params, id=opp, opp_id=agent))(opp_params)
            return -L0(params, opp_params + pi_lr * opp_update)


        def dice_objective(params, opp_params, batch: TransitionBatch, agent_id, opp_id, gamma):
            theta = unravel_fns[agent_id](params)
            opp_theta = unravel_fns[opp_id](opp_params)
            self_logprobs = policy_network.apply(theta, batch.info_state[agent_id]).log_prob(batch.action[agent_id])
            other_logprobs = policy_network.apply(opp_theta, batch.info_state[opp_id]).log_prob(batch.action[opp_id])

            r_t = batch.reward[agent_id]
            v_t = batch.values[agent_id]
            discount = gamma * jnp.ones_like(r_t)# / gamma
            discounted_rewards = discount.cumprod(axis=-1) / gamma * r_t
            dependencies = jnp.cumsum(self_logprobs + other_logprobs, axis=1)
            stochastic_nodes = self_logprobs + other_logprobs
            dice_obj = jnp.mean(jnp.sum(magic_box(dependencies) * discounted_rewards, axis=-1))

            use_baseline = True
            if use_baseline:
                discounted_values = discount.cumprod(axis=-1) / gamma * v_t
                baseline = jnp.mean(jnp.sum((1-magic_box(stochastic_nodes)) *  discounted_values, axis=-1))
                dice_obj = dice_obj + baseline

            return dice_obj


        def out_lookahead(params, opp_params, id, opp_id, batch, rng):
            opp_update = grad(dice_objective)(opp_params, params, batch, opp_id, id, gamma=0.99)
            opp_pi_lr = pi_lr
            opp_new_params = opp_params + opp_pi_lr * opp_update

            env = IteratedPrisonersDilemmaEnv(batch_size=o_t.shape[1], iterations=o_t.shape[2],
                                              include_remaining_iterations=False)
            timestep = env.reset()
            rewards, actions, states, values = [], [], [], []
            info_state = timestep.observations['info_state']
            thetas = dict((i, unravel_fns[i](p)) for i, p in zip([id, opp_id], [params, opp_new_params]))
            while not timestep.last():
                action = jnp.stack([
                    policy_network.apply(theta, info_state[i]).sample(seed=next(rng))
                    for i, theta in thetas.items()
                ], axis=1)
                action = jax.lax.stop_gradient(action)
                timestep = env.step(action)
                rewards.append(timestep.rewards)
                actions.append(action)
                states.append(info_state)
                values.append([critic_network.apply(train_state.critic_params[i], info_state[i]) for i in sorted(thetas.keys())])
                info_state = timestep.observations['info_state']

            batch = TransitionBatch(
                info_state=jnp.array(states).transpose(1, 2, 0, 3),
                action=jnp.array(actions).transpose(2, 1, 0),
                reward=jnp.array(rewards).transpose(1, 2, 0),
                values=jnp.array(values).squeeze().transpose(1, 2, 0)
            )
            return dice_objective(params, opp_new_params, batch, agent_id, opp_id, gamma=0.99)

        param_update = -pi_lr * grad(out_lookahead)(params[agent], params[opp], agent, opp, batch, rng)

        # param_update = grad(obj, argnums=0)(params[agent], params[opp])
        # b = jax.jit(lookahead, static_argnums=(4))(params=unravel_fns[agent](params[agent]), opp_params=unravel_fns[opp](params[opp]), id=agent, opp_id=opp, rng=rng)
        return unravel_fns[agent](param_update)

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
            d_t = batch.discount[agent_id]
            values = jnp.squeeze(critic_network.apply(train_state.critic_params[agent_id], o_t))
            v_t, v_tp1 = values[:, :-1], values[:, 1:]
            logits = policy_network.apply(params, o_t).logits
            compute_return = vmap(partial(rlax.n_step_bootstrapped_returns, n=1, lambda_t=0.0))
            compute_return = vmap(partial(rlax.lambda_returns))
            discounts = jnp.stack([batch.discount[agent_id]] * r_t.shape[0], axis=0)
            G_t = compute_return(r_t=r_t[:, :-1], discount_t=discounts[:, :-1], v_t=jnp.zeros_like(v_tp1))
            adv_t = G_t #- v_t
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
        if correction_type is not None:
            if correction_type == 'lola':
                gradient_correction = lola_correction(train_state, batch)
            elif correction_type == 'dice':
                gradient_correction = dice_correction(train_state, batch)
            else:
                raise ValueError('Unknown correction type: {}'.format(correction_type))
            policy_grads = gradient_correction #jax.tree_util.tree_map(lambda _, c: correction_weight * c, policy_grads, gradient_correction)

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
                 clip_grad_norm: float = 0.5,
                 policy_update_interval: int = 8,
                 discount: float = 0.99,
                 seed: jax.random.PRNGKey = 42,
                 fit_opponent_model = True,
                 correction_type = 'lola',
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
            rng=self._rng,
            policy_network=policy,
            critic_network=critic,
            pi_lr=pi_learning_rate,
            optimizer=self._policy_opt.update,
            correction_type=correction_type
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
        self._train_state.policy_params[player_id] = deepcopy(state.policy_params[player_id])
        self._train_state.critic_params[player_id] = deepcopy(state.critic_params[player_id])
       # self._train_state.policy_opt_states[player_id] = deepcopy(state.policy_opt_states[player_id])
        #self._train_state.critic_opt_state[player_id] = deepcopy(state.critic_opt_state[player_id])

    def get_value_fn(self) -> typing.Callable:
        def value_fn(obs: jnp.ndarray):
            obs = jnp.array(obs)
            return self._critic_network.apply(self.train_state.critic_params[self.player_id], obs).squeeze(-1)
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
        Produces an action and possibly triggers a parameter update. LOLA agents depend on having access to previous
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
        critic_params, critic_opt_states = {}, {}
        for agent_id in agent_ids:
            policy_params[agent_id] = self._pi_network.init(next(self._rng), init_inputs)
            if agent_id == self.player_id:
                policy_opt_state = self._policy_opt.init(policy_params[agent_id])
            else:
                policy_opt_state = self._opponent_opt.init(policy_params[agent_id])
            policy_opt_states[agent_id] = policy_opt_state
            critic_params[agent_id] = self._critic_network.init(next(self._rng), init_inputs)
            critic_opt_states[agent_id] = self._critic_opt.init(critic_params[agent_id])


        return TrainState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_states=policy_opt_states,
            critic_opt_state=critic_opt_states
        )

    def _store_time_step(self, time_step: TimeStep, action: np.ndarray):
        """
        Converts the timestep and the action into a transition and steps the counters.
        Args:
            time_step: the current time step.
            action: the action that was taken before observing time_step

        Returns: None

        """
        self._step_counter += time_step.observations["batch_size"]
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
        return self._step_counter >= self._batch_size * self._episode_counter and self._episode_counter > 0

    def _update_agent(self, batch: TransitionBatch) -> typing.Dict:
        """
        Updates the critic and policy parameters of the agent.
        Args:
            batch: A batch of training episodes. Dimensions (N=player, B=batch_size, T=timesteps, S=state_dim):
                   action: (N, B, T),
                   discount: (B, T),
                   info_state: (N, B, T, *S),
                   legal_actions_mask: (N, B, T),
                   reward: (N, B, T),
                   terminal: (B, T),
                   values: (N, B, T)

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
                batch = batch.replace(
                    info_state=batch.info_state.transpose(1,2,0,3),
                    action=batch.action.transpose(1,2,0),
                    legal_actions_mask=batch.legal_actions_mask.T,
                    reward=batch.reward.transpose(1,2,0),
                    values=batch.values.squeeze().transpose(1,2,0),
                    discount=batch.discount.transpose(1,0),
                )
                batches.append(batch)
                episode.clear()
        return batches[0]

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
        legal_actions_mask = np.zeros((self._batch_size, self._num_actions))
        legal_actions_mask[..., legal_actions] = 1
        actions = np.array(time_step.observations["actions"])
        rewards = np.array(time_step.rewards)
        obs = np.array(self._prev_time_step.observations["info_state"])
        transition = TransitionBatch(
            info_state=obs,
            action=actions,
            reward=rewards,
            discount=np.array([self._discount * (1 - time_step.last())] * len(self._train_state.policy_params)),
            terminal=time_step.last(),
            legal_actions_mask=legal_actions_mask,
            values=self._prev_time_step.observations["values"]
        )
        return transition
