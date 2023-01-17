import jax
import jax.numpy as jnp
import haiku as hk
import chex
import optax

import numpy as np
from functools import partial
from copy import deepcopy
from typing import NamedTuple

import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.rl_environment import StepType

from network import PlayerNetwork, CriticNetwork, RetrospectiveReplayBuffer, NetworkOuptut, JaxFriendlyBuffer

# Exploitability class
class MeanPolicyEvaluation(policy.Policy):
    def __init__(self, network, params):
        game = pyspiel.load_game('leduc_poker')
        all_players = list(range(game.num_players()))

        super().__init__(game, all_players)
        self.network = network # actor network
        self.params = params

    def action_probabilities(self, state):
        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)
        legal_actions_mask = state.legal_actions_mask(current_player)
        info_state_vector = jnp.array( # actor net only takes single player info_state
            state.information_state_tensor(current_player), dtype=jnp.float32)

        # mean policy head is what converges to nash 
        policy = self.network(self.params, None, info_state_vector).pi_bar
        policy = np.where(legal_actions_mask, policy, 10e-20)
        policy = jax.nn.softmax(policy)

        return {action: policy[action] for action in range(len(policy))}

def get_armac_network(num_players, num_actions, layers=[256, 128]):
    def armac_network():
        actor = PlayerNetwork(layers, num_actions, name='actor')
        critic = CriticNetwork(256, num_actions, name='critic') #layers[-1]
        def init(history):
            actor_head = {player: actor(history[player]) for player in range(num_players)}
            critic_head = critic(history)
            return NetworkOuptut(avg_regret=jnp.stack([actor_head[player].w_bar for player in range(num_players)]),
                                mean_policy=jnp.stack([actor_head[player].pi_bar for player in range(num_players)]),
                                q_values=jnp.stack([critic_head[player] for player in range(num_players)]))
        return init, (init, actor, critic)
    return armac_network

class ARMAC(rl_agent.AbstractAgent):
    def __init__(self,
                 env,
                 epsilon_config,
                 layers = [256, 128],
                 num_epochs = 100,
                 num_episodes = 5000,
                 min_steps_per_epoch = 5000,
                 learning_steps = 100,
                 learning_rate = 0.001,
                 batch_size = 64,
                 trajectory_length = 32,
                 update_target_params_every = 1000,
                 network_buffer_max_len = 1024,
                 gamma= 0.99):

        # Environment
        self._env = env
        self._num_players = env.num_players
        self._player_iter = range(self._num_players)
        self._num_actions = env.action_spec()['num_actions']
        self._rngkey = jax.random.PRNGKey(42)

        # Network
        self._learning_rate = learning_rate
        self._network_layers = layers
        self._update_target_params_every = update_target_params_every
        self._gamma = gamma

        # Initialize 'main()' method variables
        self._num_epochs = num_epochs
        self._num_episodes = num_episodes
        self._min_steps_per_epoch = min_steps_per_epoch
        self._current_player = 1
        self._learning_steps = learning_steps
        self._batch_size = batch_size
        self._trajectory_len = trajectory_length
        self._network_buffer_max_len = network_buffer_max_len
        self._prev_history = None
        self._prev_action = None
        self._epsilon_schedule = optax.polynomial_schedule(**epsilon_config)

        # Jit network update and forward pass
        self._update_step = jax.jit(self._update_step)
        self._matched_regrets = jax.jit(self._matched_regrets)

        # Losses
        self._adv_loss = optax.l2_loss
        self._mean_policy_loss = optax.softmax_cross_entropy

        # Experiment tracking
        self._nash_convs = []
        self._losses = {'actor':[], 'critic':[]}
        self._total_train_steps = 0

        self.game = pyspiel.load_game('leduc_poker')

    def _next_rng_key(self):
        """Get the next rng subkey from class rngkey."""
        self._rngkey, subkey = jax.random.split(self._rngkey)
        return subkey

    def _setup_learner(self):
        history = self._env.reset().observations['info_state']
        network = hk.multi_transform(get_armac_network(self._num_players, self._num_actions, self._network_layers))

        self._init_fn = network.init
        self._network, self._actor_net, self._critic_net = network.apply
        self._opt_init, self._opt_update = optax.adam(self._learning_rate)
        
        self._current_params = self._init_fn(self._next_rng_key(), np.array(history).astype(np.float32))
        self._opt_state = self._opt_init(self._current_params)
        self._target_params = self._current_params 

    def _matched_regrets(self, params, player, legal_actions_mask, history):
        # This function maps to Eq. 1 in the paper
        network_output = self._network(params, None, jnp.array(history).astype(jnp.float32))
        player_output = jax.tree_map(lambda x: x[player], network_output) 
        advs = player_output.avg_regret * legal_actions_mask
        advantages = jax.nn.relu(advs)
        summed_regrets = jnp.sum(advantages)
        matched_regrets = jax.lax.cond(
            summed_regrets > 0,
            lambda _: advantages / summed_regrets,
            lambda _: legal_actions_mask / jnp.sum(legal_actions_mask), # uniform 
            None)
        return matched_regrets, player_output.q_values

    def _get_adv_derived_policy(self, params, player: int, legal_action_mask, history):
        adv_derived_policy, q_values = self._matched_regrets(params, player, legal_action_mask, history)
        adv_derived_policy = np.array(adv_derived_policy) * legal_action_mask
        adv_derived_policy /= adv_derived_policy.sum() 
        return adv_derived_policy, q_values
    
    def _get_regrets(self, params, player, legal_action_mask, history):
        policy_j, q_values_j = self._get_adv_derived_policy(params, player, legal_action_mask, history)
        policy_ev = np.sum(q_values_j * policy_j)
        regrets = ((q_values_j - policy_ev) * legal_action_mask) if player == self._current_player else None # if i == tau(s)
        return regrets, policy_j

    def _epsilon_greedy(self, policy, epsilon, legal_actions, opponent_action):
        probs = np.zeros(self._num_actions)
        # choose random action
        if np.random.rand() < epsilon and not opponent_action:
            action = np.random.choice(legal_actions) 
            probs[legal_actions] = 1.0 / len(legal_actions)
        # choose greedy action: select action on policy
        else:
            # action = np.random.choice(range(self._num_actions), p=policy)
            # probs = policy
            action = np.argmax(policy)
            probs = policy
        return action, probs

    def _sample_action_from_advatange(self, params, player: int, legal_actions, legal_action_mask, history):
        policy, _ = self._get_adv_derived_policy(params, player, legal_action_mask, history)
        epsilon = self._epsilon_schedule(self._total_train_steps)
        opponent_action = self._current_player != player
        action, policy = self._epsilon_greedy(policy, epsilon, legal_actions, opponent_action)
        return action, policy

    def _rollout_episode(self):
        time_step = self._env.reset()
        while not time_step.last():
            agent_output = self.step(time_step)
            time_step = self._env.step([agent_output.action])

            if time_step.last():
                self._replay_buffer['rewards'][-1] = time_step.rewards 
                break

    def main(self):
        self._setup_learner()
        self._network_buffer = RetrospectiveReplayBuffer(self._network_buffer_max_len, self._current_params)

        for epoch in range(self._num_epochs):
            self._reset_replay_buffer()
            self._behavior_params = self._network_buffer.get_behavior_params()
            for episode in range(self._num_episodes):
                self._current_player = (self._current_player + 1) % self._num_players
                self._opponent_params = self._network_buffer.get_opponent_params()
                self._rollout_episode()

                # what was the criteria used in original implementation
                if len(self._replay_buffer['i']) > self._min_steps_per_epoch:
                    print('broke', len(self._replay_buffer['i']))
                    break

            self._prepare_buffer() # need to find out why this takes so long
            for learning_step in range(self._learning_steps):
                actor_loss, critic_loss = self._learn_step()
                if self._total_train_steps % 25 == 0:
                    print(f'epoch: {epoch} | step: {learning_step} | actor: {actor_loss} | critic: {critic_loss} | update steps:{self._total_train_steps}')

            self._network_buffer.add(deepcopy(self._current_params))

            eval_policy = MeanPolicyEvaluation(self._actor_net, self._current_params)
            conv = exploitability.nash_conv(self.game, policy.python_policy_to_pyspiel_policy(policy.tabular_policy_from_callable(self.game, eval_policy.action_probabilities)))
            try:
                print(f'{epoch}: {conv} | min: {np.min(self._nash_convs)} | median: {np.median(self._nash_convs)}')
            except:
                pass
            # logging 
            self._nash_convs.append(conv)
            self._losses['actor'].append(actor_loss)
            self._losses['actor'].append(critic_loss)

            del eval_policy, self._replay_buffer

    def _get_legal_actions_mask(self, legal_actions):
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        return legal_actions_mask

    def step(self, time_step):
        """ Processes a single time step in a trajectory that is added to
        replay buffer for training network. """
        acting_player = time_step.observations['current_player']
        history = time_step.observations['info_state']
        info_state = history[acting_player]
        legal_actions = time_step.observations['legal_actions'][acting_player]
        legal_actions_mask = self._get_legal_actions_mask(legal_actions)
        discount = 0 if time_step.step_type == StepType.FIRST else self._gamma # asseert this is correct

        # always using j params to get regrets and policy, regrets is only used if current_player == acting_player
        regrets, policy_j = self._get_regrets(self._opponent_params, acting_player, legal_actions_mask, history)
        acting_params = self._behavior_params if acting_player == self._current_player else self._opponent_params
        action, probs = self._sample_action_from_advatange(acting_params, acting_player, legal_actions, legal_actions_mask, history)
        agent_output = rl_agent.StepOutput(action=action, probs=probs)
        
        # SARSE requirees q_tm1 and a_tm1 
        if not self._prev_history is None or not self._prev_action is None:
            self._add_transition(history, deepcopy(self._prev_history), info_state, deepcopy(self._prev_action), legal_actions_mask, acting_player, regrets, policy_j, discount, time_step.rewards)
        self._prev_history = history
        self._prev_action = action

        return agent_output

    def _add_transition(self, history, prev_history, info_state, prev_action, legal_action_mask, acting_player, regrets, policy_j, discount, rewards):
        regrets = [None for _ in range(self._num_actions)] if regrets is None else list(regrets)
        policy_j = list(policy_j)
        rewards = [0. for _ in range(self._num_players)] if rewards is None else rewards
        i = deepcopy(self._current_player)
        
        self._replay_buffer['i'].append(i)
        self._replay_buffer['history'].append(history)
        self._replay_buffer['prev_history'].append(prev_history)
        self._replay_buffer['info_state'].append(info_state)
        self._replay_buffer['prev_action'].append(prev_action)
        self._replay_buffer['legal_actions_mask'].append(legal_action_mask)
        self._replay_buffer['acting_player'].append(acting_player)
        self._replay_buffer['regret'].append(regrets)
        self._replay_buffer['policy_j'].append(policy_j)
        self._replay_buffer['discount'].append(discount)
        self._replay_buffer['rewards'].append(rewards)

    def _reset_replay_buffer(self): 
        self._replay_buffer = {'i': [], # current player index
                     'history': [], # concatenation of all player info states
                     'prev_history': [], # h_{t-1}
                     'info_state': [], # current player info state
                     'prev_action': [],
                     'legal_actions_mask': [],
                     'acting_player': [],
                     'regret': [],
                     'policy_j': [],
                     'discount': [],
                     'rewards': []
                     }

    def _prepare_buffer(self):
        # Converts replay_buffer of type dict to a chex.dataclass to simplfy use of Jax primitives
        def set_type(k, v):
            keep_types = ['prev_action', 'acting_player'] # these are ints used for indexing into network ouputs
            return jnp.array(v).astype(jnp.float32) if k not in keep_types else jnp.array(v) 

        # possibly switch to flax.serialization.to_state_dict then set types
        replay_buffer = {key: set_type(key, value) for key, value in self._replay_buffer.items()}
        self._replay_buffer = JaxFriendlyBuffer(**replay_buffer)

    def _sample_from_replay_buffer(self, batched=True):
        """ Samples a batch of trajectories uniformly random from replay buffer """
        if not batched:
            return sample(self._replay_buffer, 0)
        
        def sample(buffer, index):
            sample = jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, index, self._trajectory_len), buffer)
            return sample

        max_index = len(self._replay_buffer['i']) - self._trajectory_len
        index = jax.random.randint(self._next_rng_key(), (self._batch_size,), 0, max_index)
        batch = jax.vmap(sample, in_axes=(None, 0))(self._replay_buffer, index)
        return batch

    def _learn_step(self):
        batch = self._sample_from_replay_buffer() #(64, 32, attr.shape)
        self._current_params, self._opt_state, actor_loss, critic_loss = self._update_step(self._current_params, self._target_params, self._opt_state, batch)
        self._total_train_steps += 1

        if self._total_train_steps % self._update_target_params_every == 0:
            self._target_params = jax.tree_map(lambda x: x.copy(), self._current_params)
            print('target params updated')
        return actor_loss, critic_loss

    def _update_step(self, params, target_params, opt_state, data):

        def actor_loss_fn(params, data):
            
            def single_unit_loss(params, data):
                
                def advantage_loss(preds, labels):
                    loss = self._adv_loss(preds.w_bar, labels.regret)
                    return loss.mean()
                
                def policy_loss(logits, data): 
                    preds = jnp.where(data.legal_actions_mask, logits.pi_bar, -10e20)
                    labels = jax.nn.one_hot(jnp.argmax(data.policy_j), len(data.policy_j))
                    loss = self._mean_policy_loss(preds, labels)
                    return loss.mean()

                preds = self._actor_net(params, None, data.info_state) 
                loss = jax.lax.cond(data.i == data.acting_player, advantage_loss, policy_loss, preds, data)
                return loss
            
            batched_loss = jax.vmap(jax.vmap(single_unit_loss, in_axes=(None, 0)), in_axes=(None, 0))
            return batched_loss(params, data).mean()

        def critic_loss_fn(params, target_params, data):
            
            def single_unit_loss(params, target_params, data):
                
                def expected_sarsa(q_tm1, a_tm1, r_t, discount_t, q_t, probs_a_t):
                    target_tm1 = r_t + discount_t * jnp.dot(q_t, probs_a_t)
                    return jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]

                # confirming that we want to use the regret matching policy in the SARSE update
                q_tm1 = self._critic_net(params, None, data.prev_history)[data.acting_player]
                probs_a_t, q_t = self._matched_regrets(target_params, data.acting_player, data.legal_actions_mask, data.history)
                error = expected_sarsa(q_tm1, data.prev_action, data.rewards[data.acting_player], data.discount, q_t, probs_a_t)
                return error**2

            batched_loss = jax.vmap(jax.vmap(single_unit_loss, in_axes=(None, None, 0)), in_axes=(None, None, 0))
            return batched_loss(params, target_params, data).mean()

        def network_grads(params, target_params, data):
            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(params, data)
            critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(params, target_params, data)
            actor_grads, _ = hk.data_structures.partition(lambda module_name, n, v: 'actor' in module_name, actor_grads)
            critic_grads, _ = hk.data_structures.partition(lambda module_name, n, v: 'critic' in module_name, critic_grads)
            grads = hk.data_structures.merge(actor_grads, critic_grads)
            return actor_loss, critic_loss, grads

        actor_loss, critic_loss, grads = network_grads(params, target_params, data)
        updates, new_opt_state = self._opt_update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        updated_params = optax.incremental_update(new_params, params, 0.9)
        return updated_params, new_opt_state, actor_loss, critic_loss


if __name__ == '__main__':
    layers = [256, 128]
    num_epochs = 1000
    num_episodes = 5000
    learning_steps = 200
    learning_rate = 0.000005 # 5e-6
    batch_size = 64
    trajectory_length = 32
    update_target_params_every = 1000
    epsilon_config = {'init_value': 0.5,
                      'end_value': 0.01,
                      'transition_steps': 10_000,
                      'power': 1.} # update steps 

    env = rl_environment.Environment("leduc_poker")

    armac = ARMAC(env=env,
                  epsilon_config=epsilon_config,
                  layers=layers,
                  num_epochs=num_epochs,
                  num_episodes=num_episodes,
                  learning_steps=learning_steps,
                  learning_rate=learning_rate,
                  batch_size=batch_size,
                  trajectory_length=trajectory_length,
                  update_target_params_every=update_target_params_every)

armac.main()