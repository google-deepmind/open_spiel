// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>

#include "open_spiel/games/oh_hell.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace oh_hell {
namespace {

void GameConfigSimTest() {
    // only test with 2, 7, and 12 cards per suit and 3, 5, or 7 players
    // to reduce test output size for CI
  for (int players = kMinNumPlayers; players <= kMaxNumPlayers; players += 2) {
    for (int suits = kMinNumSuits; suits <= kMaxNumSuits; ++suits) {
      for (int cps = kMinNumCardsPerSuit; cps <= kMaxNumCardsPerSuit;
           cps += 5) {
        if (suits * cps - 1 >= players) {
          open_spiel::GameParameters params;
          params["players"] = GameParameter(players);
          params["num_suits"] = GameParameter(suits);
          params["num_cards_per_suit"] = GameParameter(cps);
          // test with a randomly selected number of tricks
          testing::RandomSimTest(*LoadGame("oh_hell", params), 1);
          // test with a fixed number of tricks
          params["num_tricks_fixed"] = GameParameter(1);
          testing::RandomSimTest(*LoadGame("oh_hell", params), 1);
        }
      }
    }
  }
}

void BasicGameTests() {
  testing::LoadGameTest("oh_hell");
  testing::ChanceOutcomesTest(*LoadGame("oh_hell"));
  testing::RandomSimTest(*LoadGame("oh_hell"), 3);
  testing::ResampleInfostateTest(*LoadGame("oh_hell"), /*num_sims=*/10);
}

std::string InformationStateTensorToString(Player player,
                                           const DeckProperties& deck_props,
                                           int num_players,
                                           int max_num_tricks,
                                           const std::vector<float>& tensor) {
  int num_tricks;
  Player dealer;
  int trump;
  std::vector<int> hand(deck_props.NumCards());
  std::vector<int> bids(num_players);
  std::vector<int> tricks_won(num_players);
  std::vector<Trick> tricks(max_num_tricks);

  auto ptr = tensor.begin();
  // num tricks chance action
  for (int i = 0; i < max_num_tricks; ++i) {
    if (ptr[i] == 1) {
      num_tricks = i + 1;
      break;
    }
  }
  ptr += max_num_tricks;
  // dealer selection
  for (int i = 0; i < num_players; ++i) {
    if (ptr[i] == 1) {
      dealer = i;
      break;
    }
  }
  ptr += num_players;
  // set trump
  for (int i = 0; i < deck_props.NumCards(); ++i) {
    if (ptr[i] == 1) {
      trump = i;
      break;
    }
  }
  ptr += deck_props.NumCards();
  // bypass dealt hand
  ptr += deck_props.NumCards();
  // Current hand
  for (int i = 0; i < deck_props.NumCards(); ++i) {
    if (ptr[i] == 1) hand[i] = 1;
  }
  ptr += deck_props.NumCards();
  // bids
  for (Player p = 0; p < num_players; ++p) {
    for (int i = 0; i <= max_num_tricks + 1; ++i) {
      if (ptr[i] == 1) {
        // account for no bid yet
        bids[p] = i - 1;
        break;
      }
    }
    ptr += max_num_tricks + 2;
  }
  // Points
  for (int i = 0; i < num_players; ++i) {
    int player_score = 0;
    for (int j = 0; j < max_num_tricks; ++j) {
      if (ptr[j] == 1) ++player_score;
    }
    tricks_won[i] = player_score;
    ptr += max_num_tricks;
  }
  // Trick history
  Player leader;
  int num_cards_played = 0;
  for (int trick = 0; trick < max_num_tricks; ++trick) {
    leader = kInvalidPlayer;
    for (int i = 0; i < num_players * deck_props.NumCards(); ++i) {
      if (ptr[i] == 1) {
        leader = i / deck_props.NumCards();
        int card = i % deck_props.NumCards();
        tricks[trick] = Trick(leader, deck_props.CardSuit(trump), card,
                              deck_props);
        ++num_cards_played;
        break;
      }
    }
    if (leader != kInvalidPlayer) {
      ptr += (leader + 1) * deck_props.NumCards();
      for (int i = 0; i < num_players - 1; ++i) {
        for (int j = 0; j < deck_props.NumCards(); ++j) {
          if (ptr[j] == 1) {
            tricks[trick].Play((leader + i + 1) % num_players, j);
            ++num_cards_played;
          }
        }
        ptr += deck_props.NumCards();
      }
      ptr += (num_players - std::max(leader, 0) - 1) * deck_props.NumCards();
    } else {
      ptr += (2 * num_players - 1) * deck_props.NumCards();
      break;
    }
  }

  // Now build InformationStateString.
  std::string rv = absl::StrFormat("Num Total Tricks: %d\n", num_tricks);
  absl::StrAppendFormat(&rv, "Dealer: %d\n", dealer);
  // guaranteed to be in kPlay or kBid phase, so all chance nodes have already
  // occured
  absl::StrAppendFormat(&rv, "Num Cards Dealt: %d\n",
                        num_tricks * num_players + 1);
  absl::StrAppendFormat(&rv, "Trump: %s\n", deck_props.CardString(trump));
  absl::StrAppendFormat(&rv, "Player: %d\n", player);
  for (int suit = 0; suit < deck_props.NumSuits(); ++suit) {
    absl::StrAppendFormat(&rv, "    %c: ", kSuitChar[suit]);
    for (int rank = deck_props.NumCardsPerSuit() - 1; rank >= 0; --rank) {
      if (hand[deck_props.Card(Suit(suit), rank)]) {
        absl::StrAppend(&rv, absl::string_view(&kRankChar[rank], 1));
      }
    }
    absl::StrAppend(&rv, "\n");
  }

  if (num_cards_played > 0) {
    absl::StrAppend(&rv, "\nTricks:\n");
    // wraps around to show which player started trick
    for (Player p = 0; p < 2 * num_players - 1; ++p) {
      absl::StrAppendFormat(&rv, "%d  ", p % num_players);
    }
    for (int i = 0; i <= (num_cards_played - 1) / num_players; ++i) {
      absl::StrAppend(&rv, "\n", std::string(3 * tricks[i].Leader(), ' '));
      for (auto card : tricks[i].Cards()) {
        absl::StrAppend(&rv, deck_props.CardString(card), " ");
      }
    }
  }

  absl::StrAppend(&rv, "\n\nBids:        ");
  for (Player p = 0; p < num_players; ++p) {
    absl::StrAppendFormat(&rv, "%d ", bids[p]);
  }
  absl::StrAppend(&rv, "\nTricks Won:    ");
  for (Player p = 0; p < num_players; ++p) {
    absl::StrAppendFormat(&rv, "%d ", tricks_won[p]);
  }
  absl::StrAppend(&rv, "\n");

  return rv;
}

// Build InformationStateString from InformationStateTensor and check that it
// is equal to state->InformationStateString(player).
void InformationStateTensorTest(int num_games = 10) {
  std::mt19937 rng(time(0));
  int num_players = kMinNumPlayers;
  int num_suits = kMaxNumSuits;
  int num_cards_per_suit = kMaxNumCardsPerSuit;
  open_spiel::GameParameters params;
  params["players"] = GameParameter(num_players);
  params["num_suits"] = GameParameter(num_suits);
  params["num_cards_per_suit"] = GameParameter(num_cards_per_suit);
  DeckProperties deck_props = DeckProperties(num_suits, num_cards_per_suit);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("oh_hell", params);
  std::shared_ptr<const OhHellGame> oh_hell_game =
      std::dynamic_pointer_cast<const OhHellGame>(game);
  int max_num_tricks = oh_hell_game->MaxNumTricks();
  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        std::vector<std::pair<open_spiel::Action, double>> outcomes =
            state->ChanceOutcomes();
        open_spiel::Action action =
            open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
      } else {
        auto player = state->CurrentPlayer();
        auto infostate = state->InformationStateTensor(player);

        std::string infostate_string = state->InformationStateString(player);
        std::string rebuilt_infostate_string =
            InformationStateTensorToString(player, deck_props, num_players,
                                           max_num_tricks, infostate);
        SPIEL_CHECK_EQ(infostate_string, rebuilt_infostate_string);

        std::vector<open_spiel::Action> actions = state->LegalActions();
        std::uniform_int_distribution<> dis(0, actions.size() - 1);
        auto action = actions[dis(rng)];
        state->ApplyAction(action);
      }
    }
  }
}

}  // namespace
}  // namespace oh_hell
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::oh_hell::BasicGameTests();
  open_spiel::oh_hell::GameConfigSimTest();
  open_spiel::oh_hell::InformationStateTensorTest();
}
