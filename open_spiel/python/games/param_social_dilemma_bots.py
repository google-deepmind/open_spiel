# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Axelrod-style bots for parameterized social dilemma games."""

import pyspiel
from open_spiel.python.games import param_social_dilemma


class AlwaysCooperateBot(pyspiel.Bot):
    def __init__(self, player_id):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        pass

    def step(self, state):
        return param_social_dilemma.Action.COOPERATE


class AlwaysDefectBot(pyspiel.Bot):
    def __init__(self, player_id):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        pass

    def step(self, state):
        return param_social_dilemma.Action.DEFECT


class TitForTatBot(pyspiel.Bot):
    def __init__(self, player_id, num_players):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._num_players = num_players

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        pass

    def step(self, state):
        history = state.full_history()
        if not history:
            return param_social_dilemma.Action.COOPERATE

        opponent_actions = []
        for h in history:
            if h.player != pyspiel.PlayerId.SIMULTANEOUS and h.player != self._player_id:
                opponent_actions.append(h.action)

        if not opponent_actions:
            return param_social_dilemma.Action.COOPERATE

        last_action = opponent_actions[-1]
        return last_action


class GrimTriggerBot(pyspiel.Bot):
    def __init__(self, player_id, num_players):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._num_players = num_players
        self._defected = False

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        self._defected = False

    def step(self, state):
        if self._defected:
            return param_social_dilemma.Action.DEFECT

        history = state.full_history()
        for h in history:
            if h.player != pyspiel.PlayerId.SIMULTANEOUS and h.player != self._player_id:
                if h.action == param_social_dilemma.Action.DEFECT:
                    self._defected = True
                    return param_social_dilemma.Action.DEFECT

        return param_social_dilemma.Action.COOPERATE


class PavlovBot(pyspiel.Bot):
    def __init__(self, player_id, num_players):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._num_players = num_players
        self._last_action = param_social_dilemma.Action.COOPERATE

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        self._last_action = param_social_dilemma.Action.COOPERATE

    def step(self, state):
        history = state.full_history()
        if not history:
            self._last_action = param_social_dilemma.Action.COOPERATE
            return self._last_action

        opponent_actions = []
        my_actions = []
        for h in history:
            if h.player == self._player_id:
                my_actions.append(h.action)
            elif h.player != pyspiel.PlayerId.SIMULTANEOUS:
                opponent_actions.append(h.action)

        if not opponent_actions or not my_actions:
            self._last_action = param_social_dilemma.Action.COOPERATE
            return self._last_action

        last_opponent_action = opponent_actions[-1]
        last_my_action = my_actions[-1]

        if last_my_action == last_opponent_action:
            self._last_action = param_social_dilemma.Action.COOPERATE
        else:
            self._last_action = param_social_dilemma.Action.DEFECT

        return self._last_action


class TitForTwoTatsBot(pyspiel.Bot):
    def __init__(self, player_id, num_players):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._num_players = num_players

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        pass

    def step(self, state):
        history = state.full_history()
        if not history:
            return param_social_dilemma.Action.COOPERATE

        opponent_actions = []
        for h in history:
            if h.player != pyspiel.PlayerId.SIMULTANEOUS and h.player != self._player_id:
                opponent_actions.append(h.action)

        if len(opponent_actions) < 2:
            return param_social_dilemma.Action.COOPERATE

        if (opponent_actions[-1] == param_social_dilemma.Action.DEFECT and
            opponent_actions[-2] == param_social_dilemma.Action.DEFECT):
            return param_social_dilemma.Action.DEFECT

        return param_social_dilemma.Action.COOPERATE


class GradualBot(pyspiel.Bot):
    def __init__(self, player_id, num_players):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._num_players = num_players
        self._defection_count = 0
        self._punish_remaining = 0
        self._cooperate_remaining = 0

    def player_id(self):
        return self._player_id

    def restart_at(self, state):
        self._defection_count = 0
        self._punish_remaining = 0
        self._cooperate_remaining = 0

    def step(self, state):
        if self._punish_remaining > 0:
            self._punish_remaining -= 1
            return param_social_dilemma.Action.DEFECT

        if self._cooperate_remaining > 0:
            self._cooperate_remaining -= 1
            return param_social_dilemma.Action.COOPERATE

        history = state.full_history()
        for h in history:
            if h.player != pyspiel.PlayerId.SIMULTANEOUS and h.player != self._player_id:
                if h.action == param_social_dilemma.Action.DEFECT:
                    opponent_actions = []
                    for entry in history:
                        if entry.player != pyspiel.PlayerId.SIMULTANEOUS and entry.player != self._player_id:
                            opponent_actions.append(entry.action)

                    if opponent_actions and opponent_actions[-1] == param_social_dilemma.Action.DEFECT:
                        previous_count = self._defection_count
                        for action in opponent_actions:
                            if action == param_social_dilemma.Action.DEFECT:
                                pass

                        self._defection_count = sum(1 for a in opponent_actions if a == param_social_dilemma.Action.DEFECT)
                        if self._defection_count > previous_count:
                            self._punish_remaining = self._defection_count - 1
                            self._cooperate_remaining = 2
                            return param_social_dilemma.Action.DEFECT

        return param_social_dilemma.Action.COOPERATE
