import jax
import jax.numpy as jnp
import haiku as hk
import chex
import numpy as np

@chex.dataclass
class ActorOutput:
    w_bar : chex.Array
    pi_bar : chex.Array

@chex.dataclass
class NetworkOuptut:
    avg_regret : chex.Array
    mean_policy : chex.Array
    q_values : chex.Array
    
def double_relu(x):
    """ Double relu according to https://arxiv.org/pdf/1603.05201.pdf """
    x = jnp.concatenate([x, -x])
    return jax.nn.relu(x)

class InfoStateRepresentation(hk.Module):
    """ Composable block for mapping observations
     to information state representation. As described
     in paragraph 1 of Appendix F. """

    def __init__(self, linear_size):
        super().__init__('info_state_rep')
        self.linear_size = linear_size

    def __call__(self, x):
        x = hk.Linear(self.linear_size)(x)
        x = double_relu(x)
        # LSTM goes here instead of linear 
        x = hk.Linear(self.linear_size)(x)
        info_rep = jax.nn.relu(x)
        return info_rep

class ArchitectureB(hk.Module):
    """ Architecture B(x), as defined
     in paragraph 2 of Appendix F. 
     
     Question: there is not an activation layer described between
        LSTM in InfoStateRepresentation and h1 in B(x) in Appendix F.
        Was this intentional? I have used regular relu on output
        of InfoStateRepresentation for now."""

    def __init__(self, linear_size):
        super().__init__('arch_b')
        self.linear_size = linear_size

    def __call__(self, x):
        h1 = hk.Linear(self.linear_size)(x)
        h2 = double_relu(h1)
        h3 = h1 + hk.Linear(self.linear_size)(h2)
        return double_relu(h3)

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
        a_1 = jax.nn.relu(self.n_A(s_1) + self.n_B(s_0))
        h1 = jnp.concatenate([a_0, a_1])
        h2 = self.b_block(h1)

        q_0 = hk.Linear(self.num_actions)(h2)
        q_1 = hk.Linear(self.num_actions)(h2)

        return jnp.stack([q_0, q_1])

class PlayerNetwork(hk.Module):
    """ 
    Maps single info state to average regret head (w_bar) and mean policy head (pi_bar). 
    See paragraph 3 of Appendix F.
    """
    def __init__(self, layers, num_actions, name):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.info_state_representation = InfoStateRepresentation(layers[0])
        self.b_block = ArchitectureB(layers[1])

    def __call__(self, info_state):
        info_rep = self.info_state_representation(info_state)
        b_out = self.b_block(info_rep) 

        w_bar = hk.Linear(self.num_actions, name='w_bar')(b_out)
        pi_bar = hk.Linear(self.num_actions, name='pi_bar')(b_out)

        return ActorOutput(w_bar=w_bar,
                           pi_bar=pi_bar)

class RetrospectiveReplayBuffer:
    """ Stores network parameters every epoch in buffer """
    def __init__(self, max_len, init_params):  
        self.buffer = []
        self.init_params = init_params
        self.latest_params = None
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
        self.latest_params = params

    def is_empty(self):
        return len(self.buffer) == 0

    def get_opponent_params(self):
        if self.is_empty():
            return self.init_params
        return self.buffer[np.random.randint(0, len(self.buffer))]

    def get_behavior_params(self):
        """ Selects current epoch with 50% probability or
            selects previous epoch params at uniform random. """
        if self.is_empty():
            return self.init_params
        elif len(self.buffer) == 1:
            return self.buffer[0]
        elif np.random.rand() >= .50:
            return self.latest_params
        else:
            idx = np.random.randint(0, len(self.buffer)-1)
            return self.buffer[idx]

@chex.dataclass
class JaxFriendlyBuffer:
    i: chex.Array
    history: chex.Array
    prev_history: chex.Array
    info_state: chex.Array
    prev_action: chex.Array
    legal_actions_mask: chex.Array
    acting_player: chex.Array
    regret: chex.Array
    policy_j: chex.Array
    discount: chex.Array
    rewards: chex.Array