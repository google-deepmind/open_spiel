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
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "nlohmann/json_fwd.hpp"
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

// Builds a pass-phase state: pass direction, the deal of card c to seat
// c % 4 (so North holds 0, 4, ..., 48 and East 1, 5, ..., 49), then `passes`.
std::unique_ptr<State> PassPhaseState(const Game& game, PassDir pass_dir,
                                      const std::vector<Action>& passes) {
  std::unique_ptr<State> state = game.NewInitialState();
  state->ApplyAction(static_cast<int>(pass_dir));
  for (Action card = 0; card < kNumCards; ++card) {
    state->ApplyAction(card);
  }
  for (Action card : passes) {
    state->ApplyAction(card);
  }
  return state;
}

// The kNumCardsInPass pass actions starting at history index first_pass.
std::vector<Action> ResampledPass(const State& state, int first_pass) {
  const std::vector<Action> history = state.History();
  return std::vector<Action>(history.begin() + first_pass,
                             history.begin() + first_pass + kNumCardsInPass);
}

// Two pass-phase states that differ only in the pass made to `viewer`, which
// `viewer` has not yet received. That pass starts at history index
// `first_pass` and is `true_pass` in `with_true_pass`.
struct PassTwins {
  std::unique_ptr<State> with_true_pass;
  std::unique_ptr<State> with_other_pass;
  Player viewer;
  int first_pass;
  std::vector<Action> true_pass;
};

// Resamples for the viewer must be consistent with its information state and
// must not depend on the hidden pass.
void CheckPassHiddenFromViewer(const PassTwins& twins) {
  const State& truth = *twins.with_true_pass;
  const State& other = *twins.with_other_pass;
  const Player viewer = twins.viewer;
  SPIEL_CHECK_EQ(truth.InformationStateString(viewer),
                 other.InformationStateString(viewer));
  SPIEL_CHECK_EQ(truth.InformationStateTensor(viewer),
                 other.InformationStateTensor(viewer));
  SPIEL_CHECK_EQ(truth.ToObservationStruct(viewer)->ToJson(),
                 other.ToObservationStruct(viewer)->ToJson());
  constexpr int kNumSamples = 200;
  int true_pass_reproduced = 0;
  for (int seed = 0; seed < kNumSamples; ++seed) {
    std::mt19937 rng_truth(seed);
    std::mt19937 rng_other(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::unique_ptr<State> from_truth =
        truth.ResampleFromInfostate(viewer, [&]() { return dist(rng_truth); });
    std::unique_ptr<State> from_other =
        other.ResampleFromInfostate(viewer, [&]() { return dist(rng_other); });
    // Consistent with the viewer's information state.
    SPIEL_CHECK_EQ(from_truth->InformationStateString(viewer),
                   truth.InformationStateString(viewer));
    SPIEL_CHECK_EQ(from_truth->InformationStateTensor(viewer),
                   truth.InformationStateTensor(viewer));
    SPIEL_CHECK_EQ(from_truth->ToObservationStruct(viewer)->ToJson(),
                   truth.ToObservationStruct(viewer)->ToJson());
    // Independent of what the viewer cannot see.
    SPIEL_CHECK_EQ(from_truth->History(), from_other->History());
    if (ResampledPass(*from_truth, twins.first_pass) == twins.true_pass) {
      ++true_pass_reproduced;
    }
  }
  // A resampler that copies the true pass reproduces it every time; a private
  // one deals the passer 13 of the 39 cards the viewer cannot see.
  SPIEL_CHECK_LE(true_pass_reproduced, 2);
}

// The cards passed to a player are hidden from them until every player has
// passed (they arrive with the start of the play phase). During the pass
// phase, ResampleFromInfostate must therefore not copy them from the true
// state: two states that the viewer cannot tell apart must resample alike.
void ResamplePassPhasePrivacyTest() {
  std::shared_ptr<const Game> game = LoadGame("hearts");
  // History layout: pass direction, 52 deal actions, then the passes in seat
  // order, North first.
  constexpr int kNorthPass = 1 + kNumCards;
  constexpr int kEastPass = kNorthPass + kNumCardsInPass;

  // Case 1: the viewer has not passed yet. Pass left, so North passes to
  // East; East is to move.
  const std::vector<Action> north_true = {4, 8, 12};
  const std::vector<Action> north_other = {16, 20, 24};
  PassTwins before_viewer_passes{
      PassPhaseState(*game, PassDir::kLeft, north_true),
      PassPhaseState(*game, PassDir::kLeft, north_other), kEast, kNorthPass,
      north_true};
  SPIEL_CHECK_EQ(before_viewer_passes.with_true_pass->CurrentPlayer(), kEast);
  CheckPassHiddenFromViewer(before_viewer_passes);

  // Case 2: the viewer has already passed and the cards it will receive are
  // still hidden. Pass right, so North passes to West and East passes to
  // North; North and East have passed and South is to move.
  const std::vector<Action> north_east_true = {4, 8, 12, 1, 5, 9};
  const std::vector<Action> north_east_other = {4, 8, 12, 13, 17, 21};
  PassTwins after_viewer_passes{
      PassPhaseState(*game, PassDir::kRight, north_east_true),
      PassPhaseState(*game, PassDir::kRight, north_east_other), kNorth,
      kEastPass,
      std::vector<Action>(north_east_true.begin() + kNumCardsInPass,
                          north_east_true.end())};
  SPIEL_CHECK_EQ(after_viewer_passes.with_true_pass->CurrentPlayer(), kSouth);
  CheckPassHiddenFromViewer(after_viewer_passes);

  // Once all players have passed, East has received North's cards in case 1
  // and knows them, so every resample must keep them.
  std::unique_ptr<State> play = before_viewer_passes.with_true_pass->Clone();
  for (Action card : {1, 5, 9, 2, 6, 10, 3, 7, 11}) {
    play->ApplyAction(card);
  }
  SPIEL_CHECK_FALSE(play->IsChanceNode());
  constexpr int kNumPlaySamples = 20;
  for (int seed = 0; seed < kNumPlaySamples; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::unique_ptr<State> resampled =
        play->ResampleFromInfostate(kEast, [&]() { return dist(rng); });
    SPIEL_CHECK_EQ(ResampledPass(*resampled, kNorthPass), north_true);
    SPIEL_CHECK_EQ(resampled->InformationStateString(kEast),
                   play->InformationStateString(kEast));
  }
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
  std::array<std::optional<Player>, kNumCards> dealt_hand;
  std::array<std::optional<Player>, kNumCards> current_hand;
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

// Returns the cards in the received-cards section of `player`'s information
// state tensor.
std::vector<std::string> ReceivedCardsInTensor(const State& state,
                                               Player player) {
  std::vector<float> tensor(kInformationStateTensorSize);
  state.InformationStateTensor(player, absl::MakeSpan(tensor));
  // Pass direction, dealt hand and passed cards come first.
  const int offset = kNumPlayers + 2 * kNumCards;
  std::vector<std::string> cards;
  for (int card = 0; card < kNumCards; ++card) {
    if (tensor[offset + card] == 1) cards.push_back(CardString(card));
  }
  std::sort(cards.begin(), cards.end());
  return cards;
}

// Checks that each player's observation struct shows the cards passed to that
// player exactly when the information state does: only once every player has
// passed. Other players' received cards are masked or, before then, empty.
void CheckReceivedCardsMatchInformationState(const State& state) {
  // Passing is over once the pass direction, the deal and every pass are in
  // the history.
  const std::vector<Action> history = state.History();
  const bool passing_over =
      !history.empty() && history[0] != static_cast<Action>(PassDir::kNoPass) &&
      static_cast<int>(history.size()) >=
          1 + kNumCards + kNumPlayers * kNumCardsInPass;
  for (Player viewer = 0; viewer < kNumPlayers; ++viewer) {
    auto obs = state.ToObservationStruct(viewer);
    const auto* hearts_obs =
        static_cast<const HeartsObservationStruct*>(obs.get());
    const bool shown = absl::StrContains(state.InformationStateString(viewer),
                                         "Received Cards");
    SPIEL_CHECK_EQ(shown, passing_over);
    for (Player p = 0; p < kNumPlayers; ++p) {
      const std::vector<std::string>& received = hearts_obs->received_cards[p];
      if (!shown) {
        SPIEL_CHECK_TRUE(received.empty());
      } else if (p != viewer) {
        SPIEL_CHECK_EQ(received,
                       std::vector<std::string>(kNumCardsInPass, "XX"));
      } else {
        SPIEL_CHECK_EQ(received.size(), kNumCardsInPass);
        if (!state.IsTerminal()) {
          SPIEL_CHECK_EQ(received, ReceivedCardsInTensor(state, viewer));
        }
      }
    }
  }
}

// Deals hearts to North, spades to East, clubs to South and diamonds to West.
std::unique_ptr<State> DealSuitsState(const Game& game, PassDir pass_dir) {
  std::unique_ptr<State> state = game.NewInitialState();
  state->ApplyAction(static_cast<Action>(pass_dir));
  // Cards are dealt round-robin from North. Card k ^ 2 has suit (k % 4) ^ 2.
  for (int k = 0; k < kNumCards; ++k) state->ApplyAction(k ^ 2);
  return state;
}

void ReceivedCardsHiddenDuringPassTest() {
  std::shared_ptr<const Game> game = LoadGame("hearts");

  // North passes TH, 2H and 3H to the right, to West. West has not passed.
  std::unique_ptr<State> state = DealSuitsState(*game, PassDir::kRight);
  const std::vector<std::string> north_pass = {"TH", "2H", "3H"};
  for (const std::string& card : north_pass) {
    state->ApplyAction(card_int[card]);
    CheckReceivedCardsMatchInformationState(*state);
    for (Player viewer = 0; viewer < kNumPlayers; ++viewer) {
      auto obs = state->ToObservationStruct(viewer);
      for (const auto& received :
           static_cast<HeartsObservationStruct*>(obs.get())->received_cards) {
        SPIEL_CHECK_TRUE(received.empty());
      }
    }
  }

  // A twin in which North passes other hearts. No other player can tell the
  // two apart until passing ends: their observation structs are identical
  // while East, South and then West choose their passes.
  std::unique_ptr<State> twin = DealSuitsState(*game, PassDir::kRight);
  for (const std::string& card : {"AH", "KH", "QH"}) {
    twin->ApplyAction(card_int[card]);
  }
  for (int i = 0; i < 3 * kNumCardsInPass; ++i) {
    for (Player viewer = 1; viewer < kNumPlayers; ++viewer) {
      SPIEL_CHECK_EQ(state->ToObservationStruct(viewer)->ToJson(),
                     twin->ToObservationStruct(viewer)->ToJson());
    }
    if (i == 3 * kNumCardsInPass - 1) break;
    const Action action = state->LegalActions()[0];
    state->ApplyAction(action);
    twin->ApplyAction(action);
    CheckReceivedCardsMatchInformationState(*state);
  }

  // West completes the pass: every player now sees the cards it received.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kWest);
  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_TRUE(absl::StrContains(state->InformationStateString(kWest),
                                     "Received Cards"));
  CheckReceivedCardsMatchInformationState(*state);
  auto west_obs = state->ToObservationStruct(kWest);
  SPIEL_CHECK_EQ(
      static_cast<HeartsObservationStruct*>(west_obs.get())->received_cards[3],
      (std::vector<std::string>{"2H", "3H", "TH"}));

  // Random playouts over every pass direction.
  std::mt19937 rng(1234);
  for (int game_num = 0; game_num < 200; ++game_num) {
    std::unique_ptr<State> playout = game->NewInitialState();
    while (!playout->IsTerminal()) {
      CheckReceivedCardsMatchInformationState(*playout);
      std::vector<Action> actions = playout->LegalActions();
      std::uniform_int_distribution<> dis(0, actions.size() - 1);
      playout->ApplyAction(actions[dis(rng)]);
    }
    CheckReceivedCardsMatchInformationState(*playout);
  }
}

// Replays `history` with cards `a` and `b` exchanged in every action after the
// pass direction.
std::unique_ptr<State> ReplayWithSwap(const Game& game,
                                      const std::vector<Action>& history,
                                      Action a, Action b) {
  std::unique_ptr<State> state = game.NewInitialState();
  for (int i = 0; i < history.size(); ++i) {
    Action action = history[i];
    if (i > 0 && action == a) {
      action = b;
    } else if (i > 0 && action == b) {
      action = a;
    }
    state->ApplyAction(action);
  }
  return state;
}

// Checks that `viewer` sees the same thing in both states, and that its
// observation struct masks every other player's hand, passed cards and
// received cards.
void CheckSameView(const State& state, const State& twin, Player viewer) {
  SPIEL_CHECK_EQ(state.InformationStateString(viewer),
                 twin.InformationStateString(viewer));
  SPIEL_CHECK_EQ(state.InformationStateTensor(viewer),
                 twin.InformationStateTensor(viewer));
  SPIEL_CHECK_EQ(state.ToObservationStruct(viewer)->ToJson(),
                 twin.ToObservationStruct(viewer)->ToJson());
  auto obs = state.ToObservationStruct(viewer);
  const auto* hearts_obs =
      static_cast<const HeartsObservationStruct*>(obs.get());
  for (Player p = 0; p < kNumPlayers; ++p) {
    if (p == viewer) continue;
    for (const auto* cards :
         {&hearts_obs->hands[p], &hearts_obs->passed_cards[p],
          &hearts_obs->received_cards[p]}) {
      for (const std::string& card : *cards) SPIEL_CHECK_EQ(card, "XX");
    }
  }
}

// A player's views (information state string and tensor, observation struct)
// never reveal another player's hidden cards: exchanging two unplayed cards of
// one suit between two other players, neither card passed, changes nothing the
// viewer sees, at every point of the hand. The two of clubs and the queen of
// spades are left in place because the rules name them.
void ObservationHidesOtherHandsTest() {
  std::shared_ptr<const Game> game = LoadGame("hearts");
  constexpr int kMaxSwapsPerViewer = 3;
  std::mt19937 rng(4321);
  for (int game_num = 0; game_num < 50; ++game_num) {
    std::unique_ptr<State> state = game->NewInitialState();
    int num_swaps = 0;
    while (true) {
      const std::vector<Action> history = state->History();
      if (history.size() > kNumCards) {
        // Who was dealt each card: the deal is round-robin from North.
        std::vector<Player> dealt_to(kNumCards);
        for (int i = 0; i < kNumCards; ++i) {
          dealt_to[history[1 + i]] = i % kNumPlayers;
        }
        auto full = state->ToStruct();
        const auto* hearts_full =
            static_cast<const HeartsStateStruct*>(full.get());
        // Cards still held by the player they were dealt to.
        std::vector<std::vector<Action>> unmoved(kNumPlayers);
        for (Player p = 0; p < kNumPlayers; ++p) {
          for (const std::string& card_str : hearts_full->hands[p]) {
            const Action card = card_int[card_str];
            if (dealt_to[card] == p && card != Card(Suit::kClubs, 0) &&
                card != Card(Suit::kSpades, 10)) {
              unmoved[p].push_back(card);
            }
          }
        }
        for (Player viewer = 0; viewer < kNumPlayers; ++viewer) {
          int swaps = 0;
          for (Player p = 0; p < kNumPlayers && swaps < kMaxSwapsPerViewer;
               ++p) {
            for (Player q = p + 1;
                 q < kNumPlayers && swaps < kMaxSwapsPerViewer; ++q) {
              if (p == viewer || q == viewer) continue;
              for (Action a : unmoved[p]) {
                for (Action b : unmoved[q]) {
                  if (swaps == kMaxSwapsPerViewer) break;
                  if (CardSuit(a) != CardSuit(b)) continue;
                  std::unique_ptr<State> twin =
                      ReplayWithSwap(*game, history, a, b);
                  CheckSameView(*state, *twin, viewer);
                  ++swaps;
                }
              }
            }
          }
          num_swaps += swaps;
        }
      }
      if (state->IsTerminal()) break;
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
    SPIEL_CHECK_GT(num_swaps, 0);
  }
}

}  // namespace
}  // namespace hearts
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::hearts::BasicGameTests();
  open_spiel::hearts::ResamplePassPhasePrivacyTest();
  open_spiel::hearts::ShootTheMoonTest();
  open_spiel::hearts::InformationStateTensorTest();
  open_spiel::hearts::TestStructs();
  open_spiel::hearts::ReceivedCardsHiddenDuringPassTest();
  open_spiel::hearts::ObservationHidesOtherHandsTest();
}
