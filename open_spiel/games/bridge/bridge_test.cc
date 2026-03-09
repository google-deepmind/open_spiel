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

#include "open_spiel/games/bridge/bridge.h"

#include <random>

#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/games/bridge/bridge_scoring.h"
#include "open_spiel/games/bridge/bridge_uncontested_bidding.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace bridge {
namespace {

void ScoringTests() {
  SPIEL_CHECK_EQ(Score({4, kHearts, kUndoubled}, 11, true), 650);
  SPIEL_CHECK_EQ(Score({4, kDiamonds, kUndoubled}, 10, true), 130);
  SPIEL_CHECK_EQ(Score({3, kNoTrump, kUndoubled}, 6, false), -150);
  SPIEL_CHECK_EQ(Score({3, kNoTrump, kDoubled}, 6, false), -500);
  SPIEL_CHECK_EQ(Score({2, kSpades, kDoubled}, 8, true), 670);
}

void BasicGameTests() {
  testing::LoadGameTest("bridge_uncontested_bidding(num_redeals=1)");
  testing::RandomSimTest(*LoadGame("bridge_uncontested_bidding(num_redeals=1)"),
                         3);
  testing::LoadGameTest("bridge");
  testing::RandomSimTest(*LoadGame("bridge"), 3);
  testing::RandomSimTest(*LoadGame("bridge(use_double_dummy_result=false)"), 3);
  testing::ResampleInfostateTest(*LoadGame("bridge"), 10);
}

void DeserializeStateTest() {
  auto game = LoadGame("bridge_uncontested_bidding(num_redeals=1)");
  auto state = game->DeserializeState("AKQJ.543.QJ8.T92 97532.A2.9.QJ853");
  SPIEL_CHECK_EQ(state->ToString(), "AKQJ.543.QJ8.T92 97532.A2.9.QJ853 ");
}

void SerializeDoubleDummyResults() {
  auto game = LoadGame("bridge");
  auto state = game->NewInitialState();
  for (auto action : {33, 25, 3,  44, 47, 28, 23, 46, 1,  43, 30, 26, 29, 48,
                      24, 42, 13, 21, 17, 8,  5,  34, 6,  7,  37, 49, 11, 38,
                      51, 32, 20, 9,  0,  14, 35, 22, 10, 50, 15, 45, 39, 16,
                      12, 18, 27, 31, 41, 40, 4,  36, 19, 2,  52, 59, 52, 61}) {
    state->ApplyAction(action);
  }
  auto str = state->Serialize();
  str = absl::StrReplaceAll(str, {{"\n", ","}});
  SPIEL_CHECK_EQ(str,
                 "33,25,3,44,47,28,23,46,1,43,30,26,29,48,"
                 "24,42,13,21,17,8,5,34,6,7,37,49,11,38,51,"
                 "32,20,9,0,14,35,22,10,50,15,45,39,16,12,"
                 "18,27,31,41,40,4,36,19,2,52,59,52,61,"
                 "Double Dummy Results,"
                 "0,12,0,12,7,5,7,5,0,12,0,12,8,5,8,5,0,7,0,7,");
}

void DeserializeDoubleDummyResults() {
  auto game = LoadGame("bridge");
  // These results intentionally incorrect to check that the
  // implementation is using them rather than wastefully recomputing them.
  std::string serialized =
      "33,25,3,44,47,28,23,46,1,43,30,26,29,48,"
      "24,42,13,21,17,8,5,34,6,7,37,49,11,38,51,"
      "32,20,9,0,14,35,22,10,50,15,45,39,16,12,"
      "18,27,31,41,40,4,36,19,2,52,59,52,61,"
      "Double Dummy Results,"
      "12,12,0,12,7,5,7,5,9,12,0,12,6,5,8,5,3,7,0,7,";
  serialized = absl::StrReplaceAll(serialized, {{",", "\n"}});
  auto new_state = game->DeserializeState(serialized);
  SPIEL_CHECK_EQ(serialized, new_state->Serialize());
}

void ResamplePlayPhaseTests() {
  testing::ResampleInfostateTest(
      *LoadGame("bridge(use_double_dummy_result=false)"), 10);
}

void ResamplePreservesPlayerCards() {
  std::mt19937 rng(42);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  for (int sim = 0; sim < 10; ++sim) {
    auto state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      if (state->CurrentPlayer() >= 0) {
        for (int p = 0; p < kNumPlayers; ++p) {
          auto resampled = state->ResampleFromInfostate(p, sampler);
          auto* orig = dynamic_cast<const BridgeState*>(state.get());
          auto* resamp = dynamic_cast<const BridgeState*>(resampled.get());
          SPIEL_CHECK_EQ(orig->CurrentPhase(), resamp->CurrentPhase());
          for (int card = 0; card < kNumCards; ++card) {
            if (orig->PrivateObservationTensor(p)[card] == 1.0) {
              SPIEL_CHECK_EQ(resamp->PrivateObservationTensor(p)[card], 1.0);
            }
          }
        }
      }
      auto actions = state->LegalActions();
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
  }
}

void ResamplePreservesDummyCards() {
  std::mt19937 rng(123);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  for (int sim = 0; sim < 10; ++sim) {
    auto state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      auto* bridge_state = dynamic_cast<const BridgeState*>(state.get());
      if (bridge_state->CurrentPhase() == 2 && state->CurrentPlayer() >= 0) {
        for (int p = 0; p < kNumPlayers; ++p) {
          auto resampled = state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         resampled->InformationStateString(p));
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         resampled->InformationStateTensor(p));
        }
      }
      auto actions = state->LegalActions();
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
  }
}

void ResampleCanPlayToCompletion() {
  std::mt19937 rng(77);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  for (int sim = 0; sim < 10; ++sim) {
    auto state = game->NewInitialState();
    while (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(SampleAction(outcomes, rng).first);
    }
    auto actions = state->LegalActions();
    std::uniform_int_distribution<int> dist(0, actions.size() - 1);
    state->ApplyAction(actions[dist(rng)]);

    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      int p = state->CurrentPlayer();
      auto resampled = state->ResampleFromInfostate(p, sampler);
      while (!resampled->IsTerminal()) {
        auto res_actions = resampled->LegalActions();
        SPIEL_CHECK_FALSE(res_actions.empty());
        std::uniform_int_distribution<int> res_dist(0, res_actions.size() - 1);
        resampled->ApplyAction(res_actions[res_dist(rng)]);
      }
      SPIEL_CHECK_TRUE(resampled->IsTerminal());
      auto returns = resampled->Returns();
      double sum = 0;
      for (double r : returns) sum += r;
      SPIEL_CHECK_TRUE(Near(sum, 0.0, 1e-9));
      auto orig_actions = state->LegalActions();
      std::uniform_int_distribution<int> orig_dist(0, orig_actions.size() - 1);
      state->ApplyAction(orig_actions[orig_dist(rng)]);
    }
  }
}

void ResampleProducesDifferentHands() {
  std::mt19937 rng(99);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  int different_count = 0;
  int total_count = 0;
  for (int sim = 0; sim < 5; ++sim) {
    auto state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      auto* bridge_state = dynamic_cast<const BridgeState*>(state.get());
      if (bridge_state->CurrentPhase() == 1 ||
          bridge_state->CurrentPhase() == 2) {
        int p = state->CurrentPlayer();
        auto resampled = state->ResampleFromInfostate(p, sampler);
        if (state->ToString() != resampled->ToString()) {
          ++different_count;
        }
        ++total_count;
      }
      auto actions = state->LegalActions();
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
  }
  SPIEL_CHECK_GT(different_count, 0);
}

void ResampleAtOpeningLead() {
  std::mt19937 rng(200);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  for (int sim = 0; sim < 20; ++sim) {
    auto state = game->NewInitialState();
    while (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(SampleAction(outcomes, rng).first);
    }
    bool reached_opening_lead = false;
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      auto* bridge_state = dynamic_cast<const BridgeState*>(state.get());
      if (bridge_state->CurrentPhase() == 2 && !reached_opening_lead) {
        reached_opening_lead = true;
        for (int p = 0; p < kNumPlayers; ++p) {
          auto resampled = state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         resampled->InformationStateString(p));
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         resampled->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());
        }
      }
      auto actions = state->LegalActions();
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
  }
}

void ResampleMidTrick() {
  std::mt19937 rng(300);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  for (int sim = 0; sim < 10; ++sim) {
    auto state = game->NewInitialState();
    bool tested_mid_trick = false;
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      auto* bridge_state = dynamic_cast<const BridgeState*>(state.get());
      if (bridge_state->CurrentPhase() == 2 && !tested_mid_trick) {
        auto actions = state->LegalActions();
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        state->ApplyAction(actions[dist(rng)]);
        if (state->IsTerminal()) break;
        bridge_state = dynamic_cast<const BridgeState*>(state.get());
        if (bridge_state->CurrentPhase() == 2) {
          tested_mid_trick = true;
          for (int p = 0; p < kNumPlayers; ++p) {
            auto resampled = state->ResampleFromInfostate(p, sampler);
            SPIEL_CHECK_EQ(state->InformationStateString(p),
                           resampled->InformationStateString(p));
            SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                           resampled->InformationStateTensor(p));
            SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());
          }
        }
        continue;
      }
      auto actions = state->LegalActions();
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
  }
}

void ResampleFromAllPlayerPerspectives() {
  std::mt19937 rng(400);
  UniformProbabilitySampler sampler;
  auto game = LoadGame("bridge(use_double_dummy_result=false)");
  for (int sim = 0; sim < 10; ++sim) {
    auto state = game->NewInitialState();
    int play_steps = 0;
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      auto* bridge_state = dynamic_cast<const BridgeState*>(state.get());
      if (bridge_state->CurrentPhase() == 2) {
        ++play_steps;
        if (play_steps % 4 == 0) {
          for (int p = 0; p < kNumPlayers; ++p) {
            auto resampled = state->ResampleFromInfostate(p, sampler);
            auto* resamp = dynamic_cast<const BridgeState*>(resampled.get());
            SPIEL_CHECK_EQ(bridge_state->CurrentPhase(),
                           resamp->CurrentPhase());
            SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());
            SPIEL_CHECK_EQ(state->InformationStateString(p),
                           resampled->InformationStateString(p));
          }
        }
      }
      auto actions = state->LegalActions();
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
  }
}

}  // namespace
}  // namespace bridge
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::bridge::DeserializeStateTest();
  open_spiel::bridge::ScoringTests();
  open_spiel::bridge::BasicGameTests();
  open_spiel::bridge::SerializeDoubleDummyResults();
  open_spiel::bridge::DeserializeDoubleDummyResults();
  open_spiel::bridge::ResamplePlayPhaseTests();
  open_spiel::bridge::ResamplePreservesPlayerCards();
  open_spiel::bridge::ResamplePreservesDummyCards();
  open_spiel::bridge::ResampleCanPlayToCompletion();
  open_spiel::bridge::ResampleProducesDifferentHands();
  open_spiel::bridge::ResampleAtOpeningLead();
  open_spiel::bridge::ResampleMidTrick();
  open_spiel::bridge::ResampleFromAllPlayerPerspectives();
}
