from absl.testing import absltest
import numpy as np

import pyspiel


class GamesRepeatedPokerTest(absltest.TestCase):

  def test_bindings(self):
    acpc_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,"
        "numBoardCards=0 3 1 1,stack=20000 20000)"
    )
    game_string = (
        f"repeated_poker(universal_poker_game_string={acpc_game_string},"
        "max_num_hands=3,reset_stacks=True,rotate_dealer=True)"
    )
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()

    self.assertEqual(state.dealer(), 1)
    self.assertEqual(state.small_blind(), 50)
    self.assertEqual(state.big_blind(), 100)
    self.assertEqual(state.stacks(), [20000, 20000])
    self.assertEqual(state.player_to_seat(0), 0)
    self.assertEqual(state.player_to_seat(1), 1)
    self.assertEqual(state.seat_to_player(0), 0)
    self.assertEqual(state.seat_to_player(1), 1)
    self.assertEqual(state.dealer_seat(), 1)
    self.assertEqual(state.small_blind_seat(), 1)
    self.assertEqual(state.big_blind_seat(), 0)
    self.assertEqual(state.acpc_hand_histories(), [])

    rng = np.random.RandomState(0)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = rng.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        state.apply_action(rng.choice(state.legal_actions()))
    self.assertLen(state.acpc_hand_histories(), 3)
    for history in state.acpc_hand_histories():
      self.assertIsInstance(history, str)


if __name__ == "__main__":
  absltest.main()
