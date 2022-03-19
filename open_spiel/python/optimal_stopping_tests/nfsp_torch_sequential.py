"""NFSP agents trained on the optimal stopping game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from absl import app
import pyspiel

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch import nfsp
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_player_type import OptimalStoppingGamePlayerType

import matplotlib.pyplot as plt
import pandas as pd


def get_attacker_stage_policy(attacker_agent, num_states: int, num_actions: int, l: int, b: np.ndarray):
    """
    Extracts the attacker's stage policy from the NFSP agent

    :param attacker_agent: the NFSP attacker agent
    :param num_states: the number of states to consider for the stage policy
    :param num_actions: the number of actions to consider for the stage policy
    :param l: the number of stops left of the defender
    :param b: the current belief state
    """
    pi_2_stage = np.zeros((num_states+1, num_actions)).tolist()
    pi_2_stage[-1] = [0.5]*num_actions
    for s in range(num_states):
        o = {
            'info_state': [[l, b, b],
                           [l, b, s]],
            'legal_actions': [[],[0, 1]],
            'current_player': OptimalStoppingGamePlayerType.ATTACKER,
            "serialized_state": []
        }
        t_o= rl_environment.TimeStep(
            observations= o, rewards=None, discounts=None, step_type=None)
        pi_2_stage[s] = attacker_agent.step(t_o, is_evaluation=True).probs.tolist()
    return pi_2_stage

class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

def round_vec(vecs):
    return list(map(lambda vec: list(map(lambda x: round(x, 2), vec)), vecs))

def main(unused_argv):
    params = OptimalStoppingGameConfig.default_params()
    params["use_beliefs"] = True
    params["T_max"] = 5


    params["R_SLA"] = 0
    params["R_ST"] = 10
    params["R_COST"] = -1
    params["R_INT"] = -10

    game = pyspiel.load_game("python_optimal_stopping_game_sequential", params)
    num_players = game.config.num_players

    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [64, 64, 64], 'memory_rl': 600000,
                          'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}

    # network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [32, 32, 32], 'memory_rl': 60000,
    #                       'memory_sl': 1000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.009}

    hidden_layers_sizes = network_parameters['hidden_layers_sizes']
    batch_size = network_parameters['batch_size']
    rl_learning_rate = network_parameters['rl_learning_rate']
    sl_learning_rate = network_parameters['sl_learning_rate']
    memory_rl = network_parameters['memory_rl']
    memory_sl = network_parameters['memory_sl']

    expl_array  = []
    approx_expl_array = []
    ep_array = []

    eval_every = 10000
    #hidden_layers_sizes = [64, 64, 64]
    num_train_episodes = int(3e6)
    num_train_episodes = int(1000000000)
    kwargs = {
        "replay_buffer_capacity": memory_rl,
        "epsilon_decay_duration": num_train_episodes,
        "epsilon_start": 0.15,
        "epsilon_end": 0.001,
    }

    agents = [
        nfsp.NFSP(player_id=idx,
                  state_representation_size = info_state_size,
                  num_actions = num_actions,
                  hidden_layers_sizes = hidden_layers_sizes,
                  reservoir_buffer_capacity = memory_sl,
                  anticipatory_param = 0.1,
                  batch_size = batch_size,
                  rl_learning_rate = rl_learning_rate,
                  sl_learning_rate = sl_learning_rate,
                  min_buffer_size_to_learn = 2000,
                  learn_every = 64,
                  stopping_game = True,
                  optimizer_str="adam",
                  **kwargs) for idx in range(num_players)
    ]

    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    for ep in range(num_train_episodes):
        if (ep + 1) % eval_every == 0 and ep+1 > 9000:

            # print("calculating approx expl..")
            # approxexpl = OptimalStoppingGameUtil.approx_exploitability(agents, env)
            # print("approx eplx = " + str(approxexpl[-1]))

            losses = [agent.loss for agent in agents]
            print("Calculating exact exploitability.. (Don't do this for large games!)")
            try:
                expl = exploitability.exploitability(env.game, expl_policies_avg)
            except Exception as e:
                expl = 0
                print(e)
                print("Some exception when calcluation exploitability")
            print(f"Episode:{ep+1}, AVG Exploitability:{expl}, losses: {losses}")

            expl_array.append(expl)
            # approx_expl_array.append(approxexpl[-1])
            ep_array.append(ep)


        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            time_step.observations["info_state"] = round_vec(time_step.observations["info_state"])

            # print(f"time_step.rewards:{time_step.rewards}, player:{player_id}")
            action_output = agents[player_id].step(time_step)
            s = env.get_state

            # Update pi_2 if attacker
            if player_id == 1:
                s.update_pi_2(agents[player_id].pi_2_stage)

            action = [action_output.action]
            time_step = env.step(action)


        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    # abs_error = np.abs(np.subtract(approx_expl_array,expl_array))
    # plt.plot(approx_expl_array, label = "Approx exploitability")
    # plt.plot(expl_array, label = "Exploitability")
    # plt.plot(abs_error, label = "Absolute error exploit vs approx exploit")
    # plt.plot()
    # plt.legend(loc="upper left")
    # plt.savefig('approx_exploit vs exploit.png')


if __name__ == "__main__":
    app.run(main)