# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for Python gobblet."""

import typing

from absl.testing import absltest

import pyspiel
import gobblet


class GobbletTest(absltest.TestCase):
    """Tests for gobblet game."""

    def test_win_current_player(self):
        """Test that current player wins when both players draw a line."""
        sas = [
                _returns_action([0, 0], gobblet.Action(reserves=1, dst=[0, 0])),
                _returns_action([0, 0], gobblet.Action(reserves=2, dst=[0, 0])),
                _returns_action([0, 0], gobblet.Action(reserves=1, dst=[0, 1])),
                _returns_action([0, 0], gobblet.Action(reserves=1, dst=[1, 1])),
                _returns_action([0, 0], gobblet.Action(reserves=2, dst=[0, 2])),
                _returns_action([0, 0], gobblet.Action(reserves=2, dst=[1, 2])),
                _returns_action([0, 0], gobblet.Action(reserves=0, dst=[1, 0])),
                _returns_action([0, 0], gobblet.Action(src=[0, 0], dst=[1, 0])),
                _returns_action([-1, 1], None)
                ]
        game = pyspiel.load_game("gobblet")
        state = game.new_initial_state()
        for sa in sas:
            self.assertEqual(state.returns(), sa.returns)
            if sa.action:
                state.apply_action(sa.action.idx())
            else:
                self.assertTrue(state.is_terminal())

    def test_suicide_current_player(self):
        """Test that current player loses when opponent draws a line."""
        sas = [
                _returns_action([0, 0], gobblet.Action(reserves=1, dst=[0, 0])),
                _returns_action([0, 0], gobblet.Action(reserves=2, dst=[0, 0])),
                _returns_action([0, 0], gobblet.Action(reserves=1, dst=[0, 1])),
                _returns_action([0, 0], gobblet.Action(reserves=1, dst=[1, 1])),
                _returns_action([0, 0], gobblet.Action(reserves=2, dst=[0, 2])),
                _returns_action([0, 0], gobblet.Action(reserves=2, dst=[2, 1])),
                _returns_action([0, 0], gobblet.Action(reserves=0, dst=[1, 0])),
                _returns_action([0, 0], gobblet.Action(src=[0, 0], dst=[1, 0])),
                _returns_action([1, -1], None)
                ]
        game = pyspiel.load_game("gobblet")
        state = game.new_initial_state()
        for sa in sas:
            self.assertEqual(state.returns(), sa.returns)
            if sa.action:
                state.apply_action(sa.action.idx())
            else:
                self.assertTrue(state.is_terminal())

    def test_state(self):
        """Checks that information states and legal actions are correct."""
        sas = [
                StateAction([0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             ],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             ],
                            [0, 0],
                            gobblet.Action(reserves=1, dst=[1, 1])),
                StateAction([1,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             ],
                            [1, 1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             ],
                            [0, 0],
                            gobblet.Action(reserves=0, dst=[2, 2])),
                StateAction([0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0,
                             ],
                            [1, 1, 1, 1, 0, 1, 1, 1, 0,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             ],
                            [0, 0],
                            gobblet.Action(reserves=-1, dst=[-1, -1])),
                ]
        game = pyspiel.load_game("gobblet")
        state = game.new_initial_state()
        for sa in sas:
            self.assertEqual(state.observation_tensor(), sa.state)
            self.assertEqual(state.legal_actions_mask(), sa.mask)
            self.assertEqual(state.returns(), sa.returns)
            state.apply_action(sa.action.idx())

    def test_state_egocentric(self):
        """Checks that egocentric information states are correct."""
        sas = [
                StateAction([0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             ],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             ],
                            [0, 0],
                            gobblet.Action(reserves=1, dst=[1, 1])),
                StateAction([0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             ],
                            [1, 1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             ],
                            [0, 0],
                            gobblet.Action(reserves=0, dst=[2, 2])),
                StateAction([0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0,
                             ],
                            [1, 1, 1, 1, 0, 1, 1, 1, 0,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1, 0, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             ],
                            [0, 0],
                            gobblet.Action(reserves=-1, dst=[-1, -1])),
                ]
        game = pyspiel.load_game("gobblet", {"egocentric_obs_tensor": True})
        state = game.new_initial_state()
        for sa in sas:
            self.assertEqual(state.observation_tensor(), sa.state)
            self.assertEqual(state.legal_actions_mask(), sa.mask)
            self.assertEqual(state.returns(), sa.returns)
            state.apply_action(sa.action.idx())


class StateAction(typing.NamedTuple):
    """StateAction holds a game's state and actions."""
    state: list
    mask: list
    returns: list
    action: gobblet.Action


def _returns_action(returns, action):
    return StateAction(None, None, returns, action)


if __name__ == "__main__":
    absltest.main()
