r"""Python implementation of the magnetic mirror descent (MMD) algorithm over the
sequence-from with dilated entropy.

See https://arxiv.org/abs/2206.05825.

One iteration of MMD consists of:
1) Compute gradients of dilated entropy
   and payoffs for current sequence form policies.
2) Compute behavioural form policy starting from the bottom
    of the tree and updating gradients of parent nodes along the way.
3) Convert behavioural form policy to equivalent sequence form policy.

The last sequence form policy converges linearly (exponentially fast)
to a \alpha-reduced normal-form  QRE.
"""

import pyspiel
import numpy as np
from scipy.stats import entropy
from open_spiel.python.algorithms.sequence_form_utils import construct_vars, sequence_to_policy, policy_to_sequence
from open_spiel.python.algorithms.sequence_form_utils import uniform_random_seq, _get_action_from_key
from open_spiel.python.algorithms.sequence_form_utils import is_root, _EMPTY_INFOSET_ACTION_KEYS, _EMPTY_INFOSET_KEYS
from open_spiel.python import policy
import copy

def neg_entropy(probs):
    return -entropy(probs)

def softmax(x):
    unnormalized = np.exp(x - np.max(x))
    return unnormalized/np.sum(unnormalized)


def divergence(x, y, val_x, val_y, grad_y):
    """
        Compute Bregman divergence between x and y, B_psi(x;y).

    Args:
        x: Numpy array.
        y: Numpy array.
        val_x: Value of psi evaluated at x.
        val_y: Value of psi evaluated at y.
        grad_y: Gradient of psi evaluated at y.

    Returns:
        Scalar.
    """
    return val_x - val_y - np.dot(grad_y,x-y)

def dilated_dgf_divergence(mmd_1, mmd_2):
    """
        Bregman divergence between two MMDDilatedEnt objects.

        The value is equivalent to a sum of two Bregman divergences
        over the sequence form, one for each player.

    Args:
        mmd_1: MMDDilatedEnt Object
        mmd_2: MMDDilatedEnt Object

    Returns:
        Scalar.
    """

    dgf_values = [mmd_1.dgf_eval(), mmd_2.dgf_eval()]
    dgf_grads = mmd_2.dgf_grads()
    div = 0
    for player in range(2):
        div += divergence(mmd_1.sequences[player], mmd_2.sequences[player], dgf_values[0][player],
                                      dgf_values[1][player], dgf_grads[player])
    return div


class MMDDilatedEnt(object):

    r"""
    Implements Magnetic Mirror Descent (MMD) with Dilated Entropy
    using the sequence form.

    The policies converge to a \alpha-reduced normal form QRE of a
    two-player zero-sum extensive-form game. If \alpha is set
    to zero then the method is equivalent to mirror descent ascent
    over the sequence form with dilated entropy and the policies
    will converge on average to a nash equilibrium.

    The main iteration loop is implemented in `update_sequences`:

    ```python
      game = pyspiel.load_game("game_name")
      mmd = MMDDilatedEnt(game, alpha=0.1)
      for i in range(num_iterations):
        mmd.update_sequences()
    ```
    The gap in the regularized game (i.e. 2x exploitability) converges
    to zero and can be computed:

    ```python
        gap = mmd.get_gap()
    ```
    The policy (i.e. behavioural form policy) can be retrieved:
    ```python
        policies = mmd.get_policies()
    ```

    The average sequences and policies can be retrieved:

    ```python
        avg_sequences = mmd.get_avg_sequences()
        avg_policies = mmd.get_avg_policies()
    ```

    """

    empy_state_action_keys = _EMPTY_INFOSET_ACTION_KEYS[:]
    empty_infoset_keys = _EMPTY_INFOSET_KEYS[:]

    def __init__(self, game, alpha, stepsize=None):
        """

        Args:
            game: a zeros-um spiel game with two players.
            alpha: weight of dilated entropy regularization. If alpha > 0
                MMD will converge to an alpha-QRE. If alpha = 0 mmd will converge
                to Nash on average.
            stepsize: MMD stepsize. Will be set automatically if None.
        """
        assert game.num_players() == 2
        assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
        assert (
                game.get_type().chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC
                or game.get_type().chance_mode ==
                pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
        assert alpha >= 0

        self.game =game
        self.alpha = float(alpha)


        self.infosets, self.infoset_actions_to_seq, \
        self.infoset_action_maps, self.infoset_parent_map, \
        self.payoff_mat, self.infoset_actions_children = construct_vars(game)

        if stepsize is not None:
            self.stepsize = stepsize
        else:
            self.stepsize = self.alpha/(np.max(np.abs(self.payoff_mat))**2)

        self.sequences = uniform_random_seq(game, self.infoset_actions_to_seq)
        self.avg_sequences = copy.deepcopy(self.sequences)
        self.iteration_count = 1

    def get_parent_seq(self, player, infostate):
        """ Looks up the parent sequence value for a given infostate.

        Args:
            player: player number, either 0 or 1.
            infostate: infostate id string.

        Returns:
            Scalar.
        """
        parent_isa_key = self.infoset_parent_map[player][infostate]
        seq_id = self.infoset_actions_to_seq[player][parent_isa_key]
        parent_seq = self.sequences[player][seq_id]
        return parent_seq

    def get_infostate_seq(self, player, infostate):
        """ Gets vector of sequence form values corresponding to a
            given infostate.

        Args:
            player: player number, either 0 or 1.
            infostate: infostate id string.

        Returns:
            Numpy array.
        """
        seq_idx = [self.infoset_actions_to_seq[player][isa_key]
                        for isa_key in self.infoset_action_maps[player][infostate]]
        seqs = np.array([self.sequences[player][idx] for idx in seq_idx])
        return seqs


    def dgf_eval(self):
        """ Computes the value of dilated entropy for current sequences.

        Returns:
            List of values, one for each player.
        """
        dgf_value = [0., 0.]

        for player in range(2):
            for infostate in self.infosets[player]:

                if is_root(infostate):
                    continue

                parent_seq = self.get_parent_seq(player, infostate)
                if parent_seq > 0:
                    children_seq = self.get_infostate_seq(player, infostate)
                    dgf_value[player] +=  parent_seq * neg_entropy(children_seq/parent_seq)

        return dgf_value

    def dgf_grads(self):
        """ Computes gradients of dilated entropy for each player and
            current sequences.

        Returns:
            A list of numpy arrays.
        """
        grads = [np.zeros(len(self.sequences[0])), np.zeros(len(self.sequences[1]))]
        for player in range(2):
            for infostate in self.infosets[player]:

                # infostates contain empty sequence for root variable
                if is_root(infostate):
                    continue

                parent_seq = self.get_parent_seq(player, infostate)
                if parent_seq > 0:

                    for isa_key in self.infoset_action_maps[player][infostate]:
                        # compute infostate term
                        seq_idx = self.infoset_actions_to_seq[player][isa_key]
                        seq = self.sequences[player][seq_idx]
                        grads[player][seq_idx] += np.log(seq/parent_seq) + 1

                        # compute terms from children if there are any
                        num_children = len(self.infoset_actions_children[player].get(isa_key, []))
                        grads[player][seq_idx] -= num_children
        return grads

    def update_sequences(self):
        """ Performs one step of MMD.
        """
        self.iteration_count += 1
        psi_grads = self.dgf_grads()
        grads = [(self.stepsize * self.payoff_mat @ self.sequences[1] - psi_grads[0])/((1+self.stepsize*self.alpha)),
                 (-self.stepsize * self.payoff_mat.T @ self.sequences[0] - psi_grads[1])/(1+self.stepsize*self.alpha)
                 ]

        _new_policy = policy.TabularPolicy(self.game)
        for player in range(2):
            self._update_state_sequences(self.empty_infoset_keys[player], grads[player], player, _new_policy)

        self.sequences = policy_to_sequence(self.game, _new_policy, self.infoset_actions_to_seq)
        self.update_avg_sequences()

    def _update_state_sequences(self, infostate, g, player, policy):

        isa_keys = self.infoset_action_maps[player][infostate]
        seq_idx = [self.infoset_actions_to_seq[player][isa_key] for isa_key in isa_keys]

        for isa_key, isa_idx in zip(isa_keys, seq_idx):

            # update children first if there are any
            children = self.infoset_actions_children[player].get(isa_key, [])
            for child in children:
                self._update_state_sequences(child, g, player, policy)
                # update gradient
                child_isa_keys = self.infoset_action_maps[player][child]
                child_seq_idx = [self.infoset_actions_to_seq[player][child_isa_key] for child_isa_key in child_isa_keys]
                g_child = np.array([g[idx] for idx in child_seq_idx])

                actions_child = [_get_action_from_key(child_isa_key) for child_isa_key in child_isa_keys]
                policy_child = policy.policy_for_key(child)[:]
                policy_child = np.array([policy_child[a] for a in actions_child])
                g[isa_idx] += np.dot(g_child, policy_child)
                g[isa_idx] += neg_entropy(policy_child)

        # no update needed for empty sequence
        if is_root(infostate):
            return

        state_policy = policy.policy_for_key(infostate)
        g_infostate = np.array([g[idx] for idx in seq_idx])
        actions = [_get_action_from_key(isa_key) for isa_key in isa_keys]
        new_state_policy = softmax(-g_infostate)
        for action, pr in zip(actions, new_state_policy):
            state_policy[action] = pr

    def get_gap(self):
        """
        Computes saddle point gap of the regularized game.
        The gap measures convergence to the alpha-QRE.

        Returns:
            Scalar.
        """
        assert self.alpha > 0, "gap cannot be computed for alpha = 0"
        grads = [
            (self.payoff_mat @ self.sequences[1]) / (self.alpha),
            (-self.payoff_mat.T @ self.sequences[0]) / (self.alpha)
            ]
        dgf_values = self.dgf_eval()

        br_policy = policy.TabularPolicy(self.game)
        for player in range(2):
            self._update_state_sequences(self.empty_infoset_keys[player], grads[player], player, br_policy)

        br_sequences = policy_to_sequence(self.game, br_policy, self.infoset_actions_to_seq)
        curr_sequences = copy.deepcopy(self.sequences)
        self.sequences = br_sequences
        br_dgf_values = self.dgf_eval()
        self.sequences = curr_sequences

        # gap of sequences (x,y)
        # d(x) + max_y' x.T A y'-d(y') + d(y) - min_x' d(x') + x'.T Ay

        gap = 0
        gap += curr_sequences[0].T @ self.payoff_mat @ br_sequences[1]
        gap += self.alpha * (dgf_values[1] -br_dgf_values[1])
        gap += self.alpha * (dgf_values[0] -br_dgf_values[0])
        gap += -br_sequences[0].T @ self.payoff_mat @ curr_sequences[1]
        return gap


    def update_avg_sequences(self):
        for player in range(2):
            self.avg_sequences[player] = self.avg_sequences[player]*(self.iteration_count-1) + self.sequences[player]
            self.avg_sequences[player] = self.avg_sequences[player]/self.iteration_count

    def current_sequences(self):
        """
        Returns: the current sequences for each player as list of
        numpy arrays.
        """
        return self.sequences

    def get_avg_sequences(self):
        """
        Returns: the average sequences for each player as list of
        numpy arrays.
        """
        return self.avg_sequences

    def get_policies(self):
        """
        Convert current sequences to equivalent behavioural form policies.

        Returns: Spiel TabularPolicy Object.
        """
        return sequence_to_policy(self.sequences, self.game,
                                  self.infoset_actions_to_seq, self.infoset_action_maps)

    def get_avg_policies(self):
        """
        Convert average sequences to equivalent behavioural form policies.

        Returns: Spiel TabularPolicy Object.
        """
        return sequence_to_policy(self.avg_sequences, self.game,
                                  self.infoset_actions_to_seq, self.infoset_action_maps)
