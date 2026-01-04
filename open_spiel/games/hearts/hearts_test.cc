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

#include "open_spiel/games/hearts/hearts.h"

#include <algorithm>
#include <array>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/json/include/nlohmann/json_fwd.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace hearts {
namespace {

std::map<std::string, int> BuildCardIntMap() {
  std::map<std::string, int> rv;
  for (int i = 0; i < kNumCards; ++i) rv[CardString(i)] = i;
  return rv;
}
std::map<std::string, int> card_int = BuildCardIntMap();

void BasicGameTests() {
  testing::LoadGameTest("hearts");
  testing::ChanceOutcomesTest(*LoadGame("hearts"));
  testing::RandomSimTest(*LoadGame("hearts"), 10);
  testing::ResampleInfostateTest(*LoadGame("hearts"), /*num_sims=*/10);

  auto observer = LoadGame("hearts")
                      ->MakeObserver(kInfoStateObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("hearts"), observer);
}

void ShootTheMoonTest() {
  GameParameters params;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("hearts", params);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  std::vector<Action> actions;
  actions = {static_cast<int>(PassDir::kNoPass),
             card_int["AC"],
             card_int["AD"],
             card_int["AH"],
             card_int["AS"],
             card_int["KC"],
             card_int["KD"],
             card_int["KH"],
             card_int["KS"],
             card_int["QC"],
             card_int["QD"],
             card_int["QH"],
             card_int["QS"],
             card_int["JC"],
             card_int["JD"],
             card_int["JH"],
             card_int["JS"],
             card_int["TC"],
             card_int["TD"],
             card_int["TH"],
             card_int["TS"],
             card_int["9C"],
             card_int["9D"],
             card_int["9H"],
             card_int["9S"],
             card_int["8C"],
             card_int["8D"],
             card_int["8H"],
             card_int["8S"],
             card_int["7C"],
             card_int["7D"],
             card_int["7H"],
             card_int["7S"],
             card_int["6C"],
             card_int["6D"],
             card_int["6H"],
             card_int["6S"],
             card_int["5C"],
             card_int["5D"],
             card_int["5H"],
             card_int["5S"],
             card_int["4C"],
             card_int["4D"],
             card_int["4H"],
             card_int["4S"],
             card_int["3C"],
             card_int["3D"],
             card_int["3H"],
             card_int["3S"],
             card_int["2C"],
             card_int["2D"],
             card_int["2H"],
             card_int["2S"]};
  for (auto action : actions) state->ApplyAction(action);
  state->ApplyAction(card_int["2C"]);
  state->ApplyAction(card_int["AD"]);
  // Check that we can play a heart even though it's the first trick because
  // we only have hearts.
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumCards / kNumPlayers);
  state->ApplyAction(card_int["AH"]);
  state->ApplyAction(card_int["AS"]);
  actions = {card_int["AC"], card_int["2D"], card_int["2H"], card_int["2S"],
             card_int["KC"], card_int["KD"], card_int["KH"], card_int["KS"],
             card_int["QC"], card_int["QD"], card_int["QH"], card_int["QS"],
             card_int["JC"], card_int["JD"], card_int["JH"], card_int["JS"],
             card_int["TC"], card_int["TD"], card_int["TH"], card_int["TS"],
             card_int["9C"], card_int["9D"], card_int["9H"], card_int["9S"],
             card_int["8C"], card_int["8D"], card_int["8H"], card_int["8S"],
             card_int["7C"], card_int["7D"], card_int["7H"], card_int["7S"],
             card_int["6C"], card_int["6D"], card_int["6H"], card_int["6S"],
             card_int["5C"], card_int["5D"], card_int["5H"], card_int["5S"],
             card_int["4C"], card_int["4D"], card_int["4H"], card_int["4S"],
             card_int["3C"], card_int["3D"], card_int["3H"], card_int["3S"]};
  for (auto action : actions) state->ApplyAction(action);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), kTotalPositivePoints);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 0);
  SPIEL_CHECK_EQ(state->PlayerReturn(2), 0);
  SPIEL_CHECK_EQ(state->PlayerReturn(3), 0);
}

std::string InformationStateTensorToString(Player player,
                                           const std::vector<float>& tensor) {
  PassDir pass_dir;
  std::array<absl::optional<Player>, kNumCards> dealt_hand;
  std::array<absl::optional<Player>, kNumCards> current_hand;
  std::vector<int> passed_cards;
  std::vector<int> received_cards;
  std::vector<int> points;
  std::array<Trick, kNumTricks> tricks;

  auto ptr = tensor.begin();
  // Pass dir
  for (int i = 0; i < kNumPlayers; ++i) {
    if (ptr[i] == 1) {
      pass_dir = static_cast<PassDir>(i);
      break;
    }
  }
  ptr += kNumPlayers;
  // Dealt hand
  for (int i = 0; i < kNumCards; ++i) {
    if (ptr[i] == 1) dealt_hand[i] = player;
  }
  ptr += kNumCards;
  // Passed cards
  for (int i = 0; i < kNumCards; ++i) {
    if (ptr[i] == 1) passed_cards.push_back(i);
  }
  ptr += kNumCards;
  // Received cards
  for (int i = 0; i < kNumCards; ++i) {
    if (ptr[i] == 1) received_cards.push_back(i);
  }
  ptr += kNumCards;
  // Current hand
  for (int i = 0; i < kNumCards; ++i) {
    if (ptr[i] == 1) current_hand[i] = player;
  }
  ptr += kNumCards;
  // Points
  for (int i = 0; i < kNumPlayers; ++i) {
    int player_score = kPointsForJD;
    for (int j = 0; j < kMaxScore; ++j) {
      if (ptr[j] == 1) ++player_score;
    }
    points.push_back(player_score);
    ptr += kMaxScore;
  }
  // Trick history
  Player leader;
  int num_cards_played = 0;
  for (int trick = 0; trick < kNumTricks; ++trick) {
    leader = kInvalidPlayer;
    for (int i = 0; i < kNumPlayers * kNumCards; ++i) {
      if (ptr[i] == 1) {
        leader = i / kNumCards;
        // jd_bonus is not relevant for our purposes, set to false.
        tricks[trick] = Trick(leader, i % kNumCards, false);
        ++num_cards_played;
        break;
      }
    }
    if (leader != kInvalidPlayer) {
      ptr += (leader + 1) * kNumCards;
      for (int i = 0; i < kNumPlayers - 1; ++i) {
        for (int j = 0; j < kNumCards; ++j) {
          if (ptr[j] == 1) {
            tricks[trick].Play((leader + i + 1) % kNumPlayers, j);
            ++num_cards_played;
          }
        }
        ptr += kNumCards;
      }
      ptr += (kNumPlayers - std::max(leader, 0) - 1) * kNumCards;
    } else {
      ptr += kTrickTensorSize;
      break;
    }
  }
  // Now build InformationStateString.
  std::string rv = "Pass Direction: ";
  absl::StrAppend(&rv, pass_dir_str[static_cast<int>(pass_dir)], "\n\n");
  absl::StrAppend(&rv, "Hand: \n");
  std::array<std::string, kNumSuits> cards;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    cards[suit].push_back(kSuitChar[suit]);
    cards[suit].push_back(' ');
    bool is_void = true;
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      if (player == current_hand[Card(Suit(suit), rank)]) {
        cards[suit].push_back(kRankChar[rank]);
        is_void = false;
      }
    }
    if (is_void) absl::StrAppend(&cards[suit], "none");
  }
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");

  if (!passed_cards.empty()) {
    absl::StrAppend(&rv, "\nPassed Cards: ");
    for (int card : passed_cards) {
      absl::StrAppend(&rv, CardString(card), " ");
    }
    absl::StrAppend(&rv, "\n");
  }
  if (!received_cards.empty()) {
    absl::StrAppend(&rv, "\nReceived Cards: ");
    for (int card : received_cards) {
      absl::StrAppend(&rv, CardString(card), " ");
    }
    absl::StrAppend(&rv, "\n");
  }
  if (num_cards_played > 0) {
    absl::StrAppend(&rv, "\nTricks:");
    absl::StrAppend(&rv, "\nN  E  S  W  N  E  S");
    for (int i = 0; i <= (num_cards_played - 1) / kNumPlayers; ++i) {
      absl::StrAppend(&rv, "\n", std::string(3 * tricks[i].Leader(), ' '));
      for (auto card : tricks[i].Cards()) {
        absl::StrAppend(&rv, CardString(card), " ");
      }
    }
    absl::StrAppend(&rv, "\n\nPoints:");
    for (int i = 0; i < kNumPlayers; ++i) {
      absl::StrAppend(&rv, "\n", DirString(i), ": ", points[i]);
    }
  }
  return rv;
}

// Build InformationStateString from InformationStateTensor and check that it
// is equal to state->InformationStateString(player).
void InformationStateTensorTest(int num_games = 100) {
  std::mt19937 rng(time(0));
  GameParameters params;
  params["jd_bonus"] = GameParameter(true);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("hearts", params);
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
            InformationStateTensorToString(player, infostate);
        SPIEL_CHECK_EQ(infostate_string, rebuilt_infostate_string);

        std::vector<open_spiel::Action> actions = state->LegalActions();
        std::uniform_int_distribution<> dis(0, actions.size() - 1);
        auto action = actions[dis(rng)];
        state->ApplyAction(action);
      }
    }
  }
}

void TestStructs() {
  auto game = LoadGame("hearts");
  auto state = game->NewInitialState();
  // Apply some actions to get a non-initial state.
  std::vector<Action> initial_actions = {
      1,                                                   // Pass left
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,  // Deal p0
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  // Deal p1
      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  // Deal p2
      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  // Deal p3
  };
  for (auto action : initial_actions) state->ApplyAction(action);

  // p0 passes 3 cards: 0, 1, 2
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);
  // p1 passes 3 cards: 13, 14, 15
  state->ApplyAction(13);
  state->ApplyAction(14);
  state->ApplyAction(15);
  // p2 passes 3 cards: 26, 27, 28
  state->ApplyAction(26);
  state->ApplyAction(27);
  state->ApplyAction(28);
  // p3 passes 3 cards: 39, 40, 41
  state->ApplyAction(39);
  state->ApplyAction(40);
  state->ApplyAction(41);

  // p1 has 2C (card 0), so p1 leads.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  HeartsState* hearts_state = static_cast<HeartsState*>(state.get());
  auto state_struct = hearts_state->ToStruct();

  SPIEL_CHECK_EQ(state_struct->ToJson(), hearts_state->ToJson());

  HeartsStateStruct* hearts_struct =
      static_cast<HeartsStateStruct*>(state_struct.get());
  SPIEL_CHECK_EQ(hearts_struct->phase, "Play");
  SPIEL_CHECK_EQ(hearts_struct->current_player, "Player_1");
  SPIEL_CHECK_EQ(hearts_struct->pass_direction, "Left");
  SPIEL_CHECK_EQ(hearts_struct->points[0], 0);
  SPIEL_CHECK_EQ(hearts_struct->hands[0].size(), 13);
  SPIEL_CHECK_EQ(hearts_struct->passed_cards[0].size(), 3);
  SPIEL_CHECK_EQ(hearts_struct->received_cards[0].size(), 3);
  SPIEL_CHECK_EQ(hearts_struct->tricks.size(), 0);
  SPIEL_CHECK_EQ(hearts_struct->hearts_broken, false);

  std::string state_json = hearts_state->ToJson();
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 HeartsStateStruct(state_json).ToJson());

  auto obs_struct = hearts_state->ToObservationStruct(0);

  // Check fields in the observation struct for player 0.
  HeartsObservationStruct* hearts_obs_struct =
      static_cast<HeartsObservationStruct*>(obs_struct.get());
  SPIEL_CHECK_EQ(hearts_obs_struct->phase, hearts_struct->phase);
  SPIEL_CHECK_EQ(hearts_obs_struct->current_player,
                 hearts_struct->current_player);
  SPIEL_CHECK_EQ(hearts_obs_struct->hands[0], hearts_struct->hands[0]);
  SPIEL_CHECK_EQ(hearts_obs_struct->passed_cards[0],
                 hearts_struct->passed_cards[0]);
  SPIEL_CHECK_EQ(hearts_obs_struct->received_cards[0],
                 hearts_struct->received_cards[0]);
  SPIEL_CHECK_EQ(hearts_obs_struct->points[0], hearts_struct->points[0]);
  SPIEL_CHECK_EQ(hearts_obs_struct->hands[1].size(), 13);
  SPIEL_CHECK_EQ(hearts_obs_struct->hands[1][0], "XX");
  SPIEL_CHECK_EQ(hearts_obs_struct->passed_cards[1].size(), 3);
  SPIEL_CHECK_EQ(hearts_obs_struct->passed_cards[1][0], "XX");
  SPIEL_CHECK_EQ(hearts_obs_struct->received_cards[1].size(), 3);
  SPIEL_CHECK_EQ(hearts_obs_struct->received_cards[1][0], "XX");
  SPIEL_CHECK_EQ(hearts_obs_struct->observing_player, 0);

  // Test json string -> obs_struct
  std::string obs_json = obs_struct->ToJson();
  SPIEL_CHECK_EQ(nlohmann::json::parse(obs_json).dump(),
                 HeartsObservationStruct(obs_json).ToJson());
}

}  // namespace
}  // namespace hearts
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::hearts::BasicGameTests();
  open_spiel::hearts::ShootTheMoonTest();
  open_spiel::hearts::InformationStateTensorTest();
  open_spiel::hearts::TestStructs();
}
