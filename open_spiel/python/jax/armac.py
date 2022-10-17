import jax
import jax.numpy as jnp
import haiku as hk
import chex
import optax

import numpy as np
from functools import partial
from copy import deepcopy

import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.rl_environment import StepType

# Network Architectures

from typing import NamedTuple

class DoubleReLU(hk.Module):
    """ Double relu according to https://arxiv.org/pdf/1603.05201.pdf """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        x = jnp.concatenate([x, -x])
        return jax.nn.relu(x)

class InfoStateRepresentation(hk.Module):
    """ Composable block for mapping observations
     to information state representation. As described
     in paragraph 1 of Appendix F. """

    def __init__(self, linear_size):
        super().__init__()
        self.linear_size = linear_size
        self.double_relu = DoubleReLU()

    def __call__(self, x):
        x = hk.Linear(self.linear_size)(x)
        x = self.double_relu(x)
        # LSTM goes here instead of linear 
        x = hk.Linear(self.linear_size)(x)
        info_rep = jax.nn.relu(x)
        return info_rep

class ArchitectureB(hk.Module):
    """ Architecture B(x), as defined
     in paragraph 2 of Appendix F. 
     
     Question: there is not an activation layer descriped between
      LSTM in InfoStateRepresentation and h1 in B(x) in Appendix F.
      Was this intentional? I have used regular relu on output
      of InfoStateRepresentation for now."""

    def __init__(self, linear_size):
        super().__init__()
        self.linear_size = linear_size
        self.double_relu = DoubleReLU()

    def __call__(self, x):
        h1 = hk.Linear(self.linear_size)(x)
        h2 = self.double_relu(h1)
        h3 = h1 + hk.Linear(self.linear_size)(h2)
        return self.double_relu(h3)

class CriticNetwork(hk.Module):
    """ Global critic network maps players' info states to Q-values 
    for each player """
    def __init__(self, size, num_actions, name):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.n_A = hk.Linear(size)
        self.n_B = hk.Linear(size)
        self.b_block = ArchitectureB(size)

    def __call__(self, history):
        # are activations correct here?
        s_0, s_1 = history
        a_0 = jax.nn.relu(self.n_A(s_0) + self.n_B(s_1))
        a_1 = jax.nn.relu(self.n_B(s_0) + self.n_A(s_1))
        h1 = jnp.concatenate([a_0, a_1])
        h2 = self.b_block(h1)

        q_0 = hk.Linear(self.num_actions)(h2)
        q_1 = hk.Linear(self.num_actions)(h2)

        return jnp.stack([q_0, q_1])

class PlayerNetwork(hk.Module):
    """ Composable block for mapping single info state to average regret 
    head (w_bar) and mean policy head (pi_bar). See paragraph 3 of Appendix F. """
    def __init__(self, layers, num_actions, name):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.info_state_rep_block = InfoStateRepresentation(layers[0])
        self.b_block = ArchitectureB(layers[1])

    def __call__(self, info_state):
        info_rep = self.info_state_rep_block(info_state)
        b_out = self.b_block(info_rep) 

        w_bar = hk.Linear(self.num_actions)(b_out)
        pi_bar = hk.Linear(self.num_actions)(b_out)

        return ActorOutput(w_bar=w_bar,
                           pi_bar=pi_bar)
@chex.dataclass
class ActorOutput:
    w_bar : chex.Array
    pi_bar : chex.Array

@chex.dataclass
class NetworkOuptut:
    avg_regret : chex.Array
    mean_policy : chex.Array
    q_values : chex.Array
    
# Buffers and such
class NetworkBuffer:
    """ Stores network parameters every epoch in buffer """
    def __init__(self, max_len):  
        self.buffer = []
        self.max_len = max_len

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

    def add(self, params):
        """ Only keep the most 1028 recent networks """
        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0)
        self.buffer.append(params)


@chex.dataclass
class JaxFriendlyBuffer:
    i: chex.Array
    j: chex.Array
    history: chex.Array
    prev_history: chex.Array
    info_state: chex.Array
    action: chex.Array
    legal_actions_mask: chex.Array
    acting_player: chex.Array
    regret: chex.Array
    policy_j: chex.Array
    discount: chex.Array
    rewards: chex.Array

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
        
        policy = self.network(self.params, None, info_state_vector).pi_bar
        policy = np.where(legal_actions_mask, policy, 10e-20)
        policy = jax.nn.softmax(policy)

        return {action: policy[action] for action in range(len(policy))}


class ARMAC(rl_agent.AbstractAgent):
    def __init__(self,
                 env,
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
                 critic_loss = 'mse',
                 mean_policy_loss = 'cross_entropy',
                 gamma= 0.99):
        self._rngkey = jax.random.PRNGKey(42)
        
        # Environment
        self._env = env
        self._dummy_time_step = env.reset()
        self._dummy_history = self._dummy_time_step.observations['info_state'] # note open speil's use of 'info_state' == use of 'history' in ARMAC
        self._num_players = env.num_players
        self._player_iter = range(self._num_players)
        self._num_actions = env.action_spec()['num_actions']
        self._info_state_shape = env.observation_spec()['info_state']
        
        # Network
        self._learning_rate = learning_rate
        self._network_layers = layers
        self._network_input = jnp.array(self._dummy_history)
        self._network_input_shape = np.array(self._network_input).shape
        self._update_target_params_every = update_target_params_every
        self._gamma = gamma

        # Initialize 'main()' method variables
        self._num_epochs = num_epochs
        self._num_episodes = num_episodes
        self._min_steps_per_epoch = min_steps_per_epoch
        self._player = 1
        self._learning_steps = learning_steps
        self._batch_size = batch_size
        self._trajectory_len = trajectory_length
        self._prev_history = self._dummy_history

        # Inititialize Buffers
        self._network_buffer = NetworkBuffer(network_buffer_max_len)
        self._replay_buffer = self._reset_replay_buffer()
        self._reset_network_and_optimizer()
        
        # Jit network update and forward pass
        self._jitted_update_step = self._get_jitted_update_step()
        self._jitted_matched_regrets = self._get_jitted_matched_regrets()

        # Losses (critic, estimated advantage, mean policy)
        self._critic_loss = optax.huber_loss if critic_loss == 'huber' else optax.l2_loss 
        self._adv_loss = optax.l2_loss
        if mean_policy_loss == 'mse':
            self._mean_policy_loss = optax.l2_loss
        elif mean_policy_loss == 'cross_entropy':
            self._mean_policy_loss = optax.softmax_cross_entropy
        else:
            raise ValueError(f'{mean_policy_loss} note supported.\
             Please select from [mse, cross_entropy]')
            
        # Experiment tracking
        self._nash_convs = []
        self._losses = {'actor':[], 'critic':[]}
        self._total_train_steps = 0

        self.game = pyspiel.load_game('leduc_poker')
    
    def _next_rng_key(self):
        """Get the next rng subkey from class rngkey."""
        self._rngkey, subkey = jax.random.split(self._rngkey)
        return subkey

    def _reset_network_and_optimizer(self):
        def armac_network():
            actor = PlayerNetwork([256, 128], 3, name='actor')
            critic = CriticNetwork(128, 3, name='critic')
            def init(history):
                actor_head = {player: actor(history[player]) for player in self._player_iter}
                critic_head = critic(history)
                return NetworkOuptut(avg_regret=jnp.stack([actor_head[player].w_bar for player in self._player_iter]),
                                    mean_policy=jnp.stack([actor_head[player].pi_bar for player in self._player_iter]),
                                    q_values=jnp.stack([critic_head[player] for player in self._player_iter]))
            return init, (init, actor, critic)
        
        # initialize network
        network = hk.multi_transform(armac_network)
        self._current_params = network.init(self._next_rng_key(), self._network_input)
        self._target_params = network.init(self._next_rng_key(), self._network_input)
        self._network, self._actor_net, self._critic_net = network.apply
        
        # initialize optimizer
        self._opt_init, self._opt_update = optax.adam(self._learning_rate)
        self._opt_state = self._opt_init(self._current_params)

    def _get_jitted_matched_regrets(self):
        
        @jax.jit
        def matched_regrets(params, player, legal_actions_mask, history):
            network_output = self._network(params, None, jnp.array(history).astype(jnp.float32))
            player_output = jax.tree_map(lambda x: x[player], network_output) 
            advs = player_output.avg_regret * legal_actions_mask
            advantages = jax.nn.relu(advs)
            summed_regrets = jnp.sum(advantages)
            matched_regrets = jax.lax.cond(
                summed_regrets > 0, lambda _: advantages / summed_regrets,
                lambda _: legal_actions_mask / jnp.sum(legal_actions_mask), None)
            return advantages, matched_regrets, player_output.q_values
        return matched_regrets

    def _get_j_params(self):
        """ Samples previous network at iteration j that is used by opponent(s) """
        network_buffer_length = len(self._network_buffer)
        if network_buffer_length == 0:
            self._j = 0
            return self._current_params
        self._j = np.random.randint(0, network_buffer_length) # epoch is zero indexed i.e. (j != T)
        return self._network_buffer[self._j]

    def _get_exploratory_params(self):
        """ Selects current epoch with 50% probability or uniformly random
            selects previous epoch params. """
        network_buffer_length = len(self._network_buffer)
        if network_buffer_length == 0:
            return self._current_params
        if np.random.rand() >= .50:
            return self._current_params
        else:
            idx = np.random.randint(0, network_buffer_length)
            return self._network_buffer[idx]
            
    def _get_adv_derived_policy(self, params, player: int, legal_action_mask, history):
        advantages, adv_derived_policy, q_values = self._jitted_matched_regrets(params, player, legal_action_mask, history)
        adv_derived_policy = np.array(adv_derived_policy) * legal_action_mask
        adv_derived_policy /= adv_derived_policy.sum() 
        return adv_derived_policy, q_values
    
    def _epsilon_greedy(self, policy, epsilon, legal_actions):
        """ There are 4 possible mutations made to the params selected by 
        '_get_exploratory_params() method.'
            i) uniform random policy
           ii) policies derived by q_{\theta}^t(h, a) - h_{\theta}^t(h)
               for the current epoch plus levels of exploration
          iii) policies defined by mean regret \bar{W} plus levels
               of exploration
           iv) average policy (acotor network output)"""
        
        probs = np.zeros(self._num_actions)
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
            probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            # should this be traditional epislon greedy where we take argmax?
            # action = np.random.choice(range(self._num_actions), p=policy) 
            # probs = policy
            action = np.argmax(policy) # np.random.choice(range(self._num_actions), p=policy)
            probs = policy
        return action, probs

    def _sample_action_from_advatange(self, params, player: int, legal_actions, legal_action_mask, history):
        policy, _ = self._get_adv_derived_policy(params, player, legal_action_mask, history)
        action, policy = self._epsilon_greedy(policy, 0.05, legal_actions)
        return action, policy

    def _get_regrets(self, params, player, legal_action_mask, history):
        policy_j, q_values_j = self._get_adv_derived_policy(params, player, legal_action_mask, history)
        policy_ev = np.sum(q_values_j * policy_j)
        regrets = ((q_values_j - policy_ev) * legal_action_mask) if player == self._player else None # if i == tau(s)
        return regrets, policy_j
    
    def _rollout_episode(self):
        time_step = self._env.reset()
        while True:
            current_player = time_step.observations["current_player"]
            agent_output = self.step(time_step)
            time_step = self._env.step([agent_output.action])
            
            if time_step.last():
                self._replay_buffer['rewards'][-1] = time_step.rewards 
                break
    
    def main(self):
        for epoch in range(self._num_epochs):
            self._reset_network_and_optimizer()
            self._reset_replay_buffer()
            for episode in range(self._num_episodes):
                self._player = (self._player + 1) % self._num_players
                self._j_params = self._get_j_params()
                self._exploratory_params = self._get_exploratory_params()
                self._rollout_episode()
                
                # what was the criteria used in original implementation
                if len(self._replay_buffer['i']) > self._min_steps_per_epoch:
                    print('broke', len(self._replay_buffer['i']))
                    break
            
            self._prepare_buffer()
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
        replay buffer for training network.
        
        Args:
            time_step: an instance of rl_environment.TimeStep.
        
        Returns:
            A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        current_player = time_step.observations['current_player']
        history = time_step.observations['info_state']
        info_state = history[current_player]
        legal_actions = time_step.observations['legal_actions'][current_player]
        legal_actions_mask = self._get_legal_actions_mask(legal_actions)
        discount = 0 if time_step.step_type == StepType.FIRST else self._gamma # we don't want to bootstrap across episode boundries

        regrets, policy_j = self._get_regrets(self._j_params, current_player, legal_actions_mask, history) # always using j params
        exploratory_params = self._exploratory_params if current_player == self._player else self._j_params
        action, probs = self._sample_action_from_advatange(exploratory_params, current_player, legal_actions, legal_actions_mask, history)
        agent_output = rl_agent.StepOutput(action=action, probs=probs)
        
        self._add_transition(history, deepcopy(self._prev_history), info_state, action, legal_actions_mask, current_player, regrets, policy_j, discount, time_step.rewards)
        self._prev_history = history
        
        return agent_output

    def _add_transition(self, history, prev_history, info_state, action, legal_action_mask, acting_player, regrets, policy_j, discount, rewards):
        self._replay_buffer['i'].append(deepcopy(self._player))
        self._replay_buffer['j'].append(deepcopy(self._j))
        self._replay_buffer['history'].append(history)
        self._replay_buffer['prev_history'].append(prev_history)
        self._replay_buffer['info_state'].append(info_state)
        self._replay_buffer['action'].append(action)
        self._replay_buffer['legal_actions_mask'].append(legal_action_mask)
        self._replay_buffer['acting_player'].append(acting_player)
        self._replay_buffer['regret'].append(list(regrets)) if regrets is not None \
         else self._replay_buffer['regret'].append(jnp.array([None for _ in range(self._num_actions)]))
        self._replay_buffer['policy_j'].append(list(policy_j))
        self._replay_buffer['discount'].append(discount)
        self._replay_buffer['rewards'].append(rewards if rewards is not None else [0. for _ in range(self._num_players)])
        
    def _reset_replay_buffer(self): 
        self._replay_buffer = {'i': [],
                     'j': [],
                     'history': [],
                     'prev_history': [],
                     'info_state': [],
                     'action': [],
                     'legal_actions_mask': [],
                     'acting_player': [],
                     'regret': [],
                     'policy_j': [],
                     'discount': [],
                     'rewards': []
                     }
    
    def _prepare_buffer(self):
        """ Converts replay_buffer of type dict to a chex.dataclass to simplfy
            use of Jax primitives """
        def set_type(k, v):
            keep_types = ['action', 'acting_player']
            return jnp.array(v).astype(jnp.float32) if k not in keep_types else jnp.array(v) 
        
        self._len_of_replay_buffer = len(self._replay_buffer['i'])
        replay_buffer = {key: set_type(key, value) for key, value in self._replay_buffer.items()}
        self._replay_buffer = JaxFriendlyBuffer(**replay_buffer)

    def _sample_from_replay_buffer(self, batched=True):
        """ Samples a batch of trajectories uniformly random from replay buffer """
        def sample(buffer, index):
            sample = jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, index, self._trajectory_len), buffer)
            return sample
        
        if not batched:
            return sample(self._replay_buffer, 0)

        index = jax.random.randint(self._next_rng_key(), (self._batch_size,), 0, self._len_of_replay_buffer - self._trajectory_len) 
        batch = jax.vmap(sample, in_axes=(None, 0))(self._replay_buffer, index)
        return batch

    def _learn_step(self):
        batch = self._sample_from_replay_buffer() #(64, 32, attr.shape)
        self._current_params, self._opt_state, actor_loss, critic_loss =  self._jitted_update_step(self._current_params, self._target_params, self._opt_state, batch)
        self._total_train_steps += 1

        if self._total_train_steps % self._update_target_params_every == 0:
            self._target_params = jax.tree_map(lambda x: x.copy(), self._current_params)
            print('target params updated')
        return actor_loss, critic_loss

    def _get_jitted_update_step(self):
        def actor_loss_fn(params, data):  
            def advantage_loss(preds, labels):
                loss = self._adv_loss(preds.w_bar, labels.regret)
                return loss.mean()
            def policy_loss(logits, data): 
                preds = jnp.where(data.legal_actions_mask, logits.pi_bar, -10e20)
                labels = jax.nn.one_hot(jnp.argmax(data.policy_j), len(data.policy_j))
                loss = self._mean_policy_loss(preds, labels) 
                return loss.mean()

            preds = self._actor_net(params, None, data.info_state) 
            loss = jax.lax.cond(data.i == data.acting_player, advantage_loss, policy_loss, preds, data) # train W head when i == tau(s) else train policy head
            return loss

        def critic_loss_fn(params, target_params, data):
            """ Expected SARSA temporal difference error
                See "Reinforcement Learning" by Sutton and Barto Section 6.6 for more details.
                http://incompleteideas.net/book/RLbook2020.pdf
            """
            q_tm1 = self._critic_net(params, None, data.prev_history)[data.acting_player]
            _, pi_t, q_t = self._jitted_matched_regrets(target_params, data.acting_player, data.legal_actions_mask, data.history)
            # q_tm1_target = q_tm1[data.action] + data.rewards[data.acting_player] + data.discount * jnp.sum(pi_t * (q_t - q_tm1[data.action]))
            q_tm1_target = data.rewards[data.acting_player] + data.discount * jnp.dot(pi_t, q_t)
            q_tm1_target = jax.lax.stop_gradient(q_tm1_target)
            loss = self._critic_loss(q_tm1_target, q_tm1[data.action])
            return loss 

        def network_grads(params, target_params, data):
            actor_loss, actor_grads = jax.vmap(
                jax.vmap(jax.value_and_grad(actor_loss_fn), in_axes=(None, 0)),
                 in_axes=(None, 0))(params, data) 
            critic_loss, critic_grads = jax.vmap(
                jax.vmap(jax.value_and_grad(critic_loss_fn), in_axes=(None, None, 0)),
                         in_axes=(None, None, 0))(params, target_params, data) 
            # split gradients
            actor_grads, _ = hk.data_structures.partition(lambda module_name, n, v: 'actor' in module_name, actor_grads)
            critic_grads, _ = hk.data_structures.partition(lambda module_name, n, v: 'critic' in module_name, critic_grads)
            # mean across batch and time dim
            actor_grads = jax.tree_map(lambda x: jnp.mean(x, axis=(0,1)), actor_grads)
            critic_grads = jax.tree_map(lambda x: jnp.mean(x, axis=(0,1)), critic_grads)
            # join grads
            grads = hk.data_structures.merge(actor_grads, critic_grads)
            return actor_loss.mean(), critic_loss.mean(), grads
        
        @jax.jit
        def update_step(params, target_params, opt_state, data):        
            actor_loss, critic_loss, grads = network_grads(params, target_params, data)
            updates, new_opt_state = self._opt_update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, actor_loss, critic_loss
        return update_step 


if __name__ == '__main__':
    layers = [256, 128]
    num_epochs = 1000
    num_episodes = 5000
    learning_steps = 200
    learning_rate = 0.00001
    batch_size = 64
    trajectory_length = 32
    update_target_params_every = 1000

    env = rl_environment.Environment("leduc_poker")

    armac = ARMAC(env=env,
                layers=layers,
                num_epochs=num_epochs,
                num_episodes=num_episodes,
                learning_steps=learning_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                trajectory_length=trajectory_length,
                update_target_params_every=update_target_params_every)

    armac.main()







