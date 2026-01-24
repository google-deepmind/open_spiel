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

"""Tests for param_social_dilemma_bots.py."""

from absl.testing import absltest
from open_spiel.python.games import param_social_dilemma
from open_spiel.python.games import param_social_dilemma_bots
import pyspiel


class ParamSocialDilemmaBotsTest(absltest.TestCase):

    def test_always_cooperate_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.AlwaysCooperateBot(player_id=0)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)
        
        state.apply_actions([action, param_social_dilemma.Action.DEFECT])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)

    def test_always_defect_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.AlwaysDefectBot(player_id=0)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.DEFECT)
        
        state.apply_actions([action, param_social_dilemma.Action.COOPERATE])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.DEFECT)

    def test_tit_for_tat_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.TitForTatBot(player_id=0, num_players=2)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)
        
        state.apply_actions([param_social_dilemma.Action.COOPERATE,
                           param_social_dilemma.Action.DEFECT])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.DEFECT)

    def test_grim_trigger_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.GrimTriggerBot(player_id=0, num_players=2)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)
        
        state.apply_actions([param_social_dilemma.Action.COOPERATE,
                           param_social_dilemma.Action.COOPERATE])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)
        
        state.apply_actions([param_social_dilemma.Action.COOPERATE,
                           param_social_dilemma.Action.DEFECT])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.DEFECT)
        
        state.apply_actions([param_social_dilemma.Action.DEFECT,
                           param_social_dilemma.Action.COOPERATE])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.DEFECT)

    def test_pavlov_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.PavlovBot(player_id=0, num_players=2)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)

    def test_tit_for_two_tats_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.TitForTwoTatsBot(player_id=0, num_players=2)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)
        
        state.apply_actions([param_social_dilemma.Action.COOPERATE,
                           param_social_dilemma.Action.DEFECT])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)
        
        state.apply_actions([param_social_dilemma.Action.COOPERATE,
                           param_social_dilemma.Action.DEFECT])
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.DEFECT)

    def test_gradual_bot(self):
        game = pyspiel.load_game("python_param_social_dilemma", {"num_players": 2})
        state = game.new_initial_state()
        bot = param_social_dilemma_bots.GradualBot(player_id=0, num_players=2)
        
        action = bot.step(state)
        self.assertEqual(action, param_social_dilemma.Action.COOPERATE)

    def test_bots_in_game(self):
        game = pyspiel.load_game("python_param_social_dilemma", {
            "num_players": 2,
            "max_game_length": 5
        })
        
        bot1 = param_social_dilemma_bots.TitForTatBot(player_id=0, num_players=2)
        bot2 = param_social_dilemma_bots.AlwaysCooperateBot(player_id=1)
        
        state = game.new_initial_state()
        while not state.is_terminal():
            actions = [bot1.step(state), bot2.step(state)]
            state.apply_actions(actions)
        
        returns = state.returns()
        self.assertEqual(len(returns), 2)


if __name__ == "__main__":
    absltest.main()
