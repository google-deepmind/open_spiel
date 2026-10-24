// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/catch_the_hidden_evader/catch_the_hidden_evader.h"

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace catch_the_hidden_evader {
namespace {

// The tiny map: stations 1 to 10, start stations 1, 4, 7 and 9.
std::shared_ptr<const Game> TinyGame(const std::string& parameters = "",
                                     int num_pursuers = 2) {
  return LoadGame(absl::StrCat("catch_the_hidden_evader(map=tiny,num_pursuers=",
                               num_pursuers, parameters, ")"));
}

void Play(State* state, const std::string& action) {
  state->ApplyAction(state->StringToAction(action));
}

// Evader at `evader`, pursuers at `pursuer0` and `pursuer1`.
std::unique_ptr<State> SetUp(const Game& game, int evader, int pursuer0,
                             int pursuer1) {
  std::unique_ptr<State> state = game.NewInitialState();
  Play(state.get(), absl::StrCat("Evader starts at ", evader));
  Play(state.get(), absl::StrCat("Pursuer 0 starts at ", pursuer0));
  Play(state.get(), absl::StrCat("Pursuer 1 starts at ", pursuer1));
  return state;
}

const CatchTheHiddenEvaderState& AsState(const State& state) {
  return down_cast<const CatchTheHiddenEvaderState&>(state);
}

std::vector<int> Sorted(const std::set<int>& values) {
  return {values.begin(), values.end()};
}

void BasicGameTests() {
  testing::LoadGameTest("catch_the_hidden_evader");
  testing::ChanceOutcomesTest(*LoadGame("catch_the_hidden_evader"));
  testing::RandomSimTest(*LoadGame("catch_the_hidden_evader"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("catch_the_hidden_evader"), 3);
  testing::ResampleInfostateTest(*LoadGame("catch_the_hidden_evader"), 5);
  testing::RandomSimTest(*TinyGame(), 50);
  testing::RandomSimTestWithUndo(*TinyGame(), 10);
  testing::ResampleInfostateTest(*TinyGame(), 20);
  testing::RandomSimTest(*TinyGame(",max_evader_moves=6,reveal_moves=[2;6],"
                                   "evader_tickets=[2;1;1;1;1]"),
                         50);
  auto observer = LoadGame("catch_the_hidden_evader")
                      ->MakeObserver(kInfoStateObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("catch_the_hidden_evader"),
                                       observer);
}

void ParseMapTest() {
  Map map = ParseMap(
      "# comment\n"
      "stations 3\n"
      "starts 1 3\n"
      "1 2 taxi\n"
      "2 1 bus  # the same pair again, another transport\n"
      "2 3 ferry\n");
  SPIEL_CHECK_EQ(map.num_stations, 3);
  SPIEL_CHECK_EQ(map.starts, (std::vector<int>{0, 2}));
  SPIEL_CHECK_EQ(map.links[0].size(), size_t{1});
  SPIEL_CHECK_EQ(map.links[0][0].first, 1);
  SPIEL_CHECK_EQ(map.links[0][0].second, kTaxiLink | kBusLink);
  SPIEL_CHECK_EQ(map.links[1].size(), size_t{2});
  SPIEL_CHECK_EQ(map.links[1][1].second, kFerryLink);
}

void CaptureByPursuerTest() {
  std::shared_ptr<const Game> game = TinyGame();
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 7);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kEvader);
  Play(state.get(), "Evader: 1->2 by taxi");
  Play(state.get(), "Pursuer 0: 4->5 by taxi");
  const int bus_before = AsState(*state).EvaderTickets(kBus);
  Play(state.get(), "Pursuer 1: 7->2 by bus");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{-1.0, 1.0}));
  // Tickets paid by pursuers go to the evader.
  SPIEL_CHECK_EQ(AsState(*state).EvaderTickets(kBus), bus_before + 1);
}

void CaptureByEvaderTest() {
  std::shared_ptr<const Game> game = TinyGame();
  std::unique_ptr<State> state = SetUp(*game, 4, 9, 1);
  Play(state.get(), "Evader: 4->9 by bus");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{-1.0, 1.0}));
}

void DoubleMoveTest() {
  std::shared_ptr<const Game> game = TinyGame();
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 7);
  Play(state.get(), "Evader: 1->6 by taxi (double)");
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kEvader);
  SPIEL_CHECK_TRUE(AsState(*state).DoublePending());
  SPIEL_CHECK_EQ(AsState(*state).EvaderTickets(kDouble), 1);
  // One double move per turn: no double variants for the second move.
  for (Action action : state->LegalActions()) SPIEL_CHECK_EQ(action % 2, 0);
  Play(state.get(), "Evader: 6->10 by taxi");
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kPursuers);
  SPIEL_CHECK_TRUE(absl::StrContains(state->InformationStateString(kPursuers),
                                     "E1:taxi+double P0:skip P1:skip"));
}

void LastMoveTest() {
  std::shared_ptr<const Game> game =
      TinyGame(",max_evader_moves=2,reveal_moves=[2]");
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 7);
  Play(state.get(), "Evader: 1->5 by taxi");
  Play(state.get(), "Pursuer 0: 4->3 by taxi");
  Play(state.get(), "Pursuer 1: 7->8 by taxi");
  Play(state.get(), "Evader: 5->10 by bus");
  Play(state.get(), "Pursuer 0: 3->2 by taxi");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  Play(state.get(), "Pursuer 1: 8->7 by taxi");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));

  // A double move as the last move: the pursuers skip their last turn.
  game = TinyGame(",max_evader_moves=1,reveal_moves=[1]");
  state = SetUp(*game, 1, 4, 7);
  Play(state.get(), "Evader: 1->5 by taxi (double)");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));
}

void ImmobilePursuersTest() {
  // Only underground tickets, and the tiny map has no underground: the
  // pursuers cannot move, and the evader wins after its first move.
  std::shared_ptr<const Game> game = TinyGame(",pursuer_tickets=[0;0;1]");
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 7);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  Play(state.get(), "Evader: 1->2 by taxi");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));

  // Checked after every move: the first move of a double ends the game too,
  // before the evader, left without tickets, would lose its second move.
  game = TinyGame(",pursuer_tickets=[0;0;1],evader_tickets=[1;0;0;0;1]");
  state = SetUp(*game, 1, 4, 7);
  Play(state.get(), "Evader: 1->2 by taxi (double)");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -1.0}));
}

void BlockedPursuersPassTest() {
  // Bus tickets only: the pursuers at 4 and 9 block each other's bus link,
  // so both pass, but they still hold usable tickets and the game goes on.
  std::shared_ptr<const Game> game = TinyGame(",pursuer_tickets=[0;1;0]");
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 9);
  Play(state.get(), "Evader: 1->2 by taxi");
  SPIEL_CHECK_EQ(
      state->LegalActions(),
      std::vector<Action>{
          down_cast<const CatchTheHiddenEvaderGame&>(*game).PassAction()});
  Play(state.get(), "Pursuer 0: pass");
  Play(state.get(), "Pursuer 1: pass");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kEvader);
}

void NoTicketEvaderTest() {
  std::shared_ptr<const Game> game = TinyGame(",evader_tickets=[0;0;0;0;0]");
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 7);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{-1.0, 1.0}));
}

void RevealTest() {
  std::shared_ptr<const Game> game = TinyGame(",reveal_moves=[1]");
  std::unique_ptr<State> state = SetUp(*game, 1, 4, 7);
  Play(state.get(), "Evader: 1->2 by taxi");
  SPIEL_CHECK_TRUE(
      absl::StrContains(state->InformationStateString(kPursuers), "E1:taxi@2"));
  SPIEL_CHECK_TRUE(
      absl::StrContains(state->ObservationString(kPursuers), "Last shown: 2"));
}

// Pursuers cannot tell apart histories that differ only in hidden evader
// stations; every player tells apart histories that differ in its own moves.
void InformationStateTest() {
  std::shared_ptr<const Game> game = TinyGame();
  std::unique_ptr<State> a = SetUp(*game, 1, 4, 7);
  std::unique_ptr<State> b = a->Clone();
  Play(a.get(), "Evader: 1->2 by taxi");
  Play(b.get(), "Evader: 1->5 by taxi");
  SPIEL_CHECK_EQ(a->InformationStateString(kPursuers),
                 b->InformationStateString(kPursuers));
  SPIEL_CHECK_EQ(a->InformationStateTensor(kPursuers),
                 b->InformationStateTensor(kPursuers));
  SPIEL_CHECK_EQ(a->ObservationTensor(kPursuers),
                 b->ObservationTensor(kPursuers));
  SPIEL_CHECK_NE(a->InformationStateString(kEvader),
                 b->InformationStateString(kEvader));
  SPIEL_CHECK_NE(a->InformationStateTensor(kEvader),
                 b->InformationStateTensor(kEvader));

  // Pursuer 0 goes 1->2 and back with taxi then bus, or bus then taxi: the
  // same stations and ticket counts, but different own histories.
  std::unique_ptr<State> c = SetUp(*game, 9, 1, 7);
  std::unique_ptr<State> d = c->Clone();
  for (State* state : {c.get(), d.get()}) {
    Play(state, "Evader: 9->10 by taxi");
    Play(state, state == c.get() ? "Pursuer 0: 1->2 by taxi"
                                 : "Pursuer 0: 1->2 by bus");
    Play(state, "Pursuer 1: 7->6 by taxi");
    Play(state, "Evader: 10->9 by taxi");
    Play(state, state == c.get() ? "Pursuer 0: 2->1 by bus"
                                 : "Pursuer 0: 2->1 by taxi");
    Play(state, "Pursuer 1: 6->7 by taxi");
  }
  SPIEL_CHECK_EQ(c->ObservationString(kPursuers),
                 d->ObservationString(kPursuers));
  SPIEL_CHECK_NE(c->InformationStateString(kPursuers),
                 d->InformationStateString(kPursuers));
  SPIEL_CHECK_NE(c->InformationStateTensor(kPursuers),
                 d->InformationStateTensor(kPursuers));
}

// The end of the game is public: the pursuers tell an open game from a
// capture, and a win from a loss, even where they cannot see the evader.
void TerminalInformationTest() {
  for (const std::string& parameters : {"", ",pursuer_tickets=[0;0;1]"}) {
    std::shared_ptr<const Game> game = TinyGame(parameters);
    std::unique_ptr<State> a = SetUp(*game, 1, 4, 7);
    std::unique_ptr<State> b = SetUp(*game, 9, 4, 7);
    Play(a.get(), "Evader: 1->2 by bus");  // Open, or immobile pursuers.
    Play(b.get(), "Evader: 9->4 by bus");  // Onto a pursuer: a capture.
    SPIEL_CHECK_TRUE(b->IsTerminal());
    SPIEL_CHECK_NE(a->Returns(), b->Returns());
    SPIEL_CHECK_NE(a->InformationStateString(kPursuers),
                   b->InformationStateString(kPursuers));
    SPIEL_CHECK_NE(a->InformationStateTensor(kPursuers),
                   b->InformationStateTensor(kPursuers));
    SPIEL_CHECK_NE(a->ObservationString(kPursuers),
                   b->ObservationString(kPursuers));
    SPIEL_CHECK_NE(a->ObservationTensor(kPursuers),
                   b->ObservationTensor(kPursuers));
  }
}

// Perfect recall, as integration_tests/api_test.py checks it, on the full tree
// of a small configuration: every information state of a player comes after
// the same sequence of that player's own information states and actions, and
// has a single tensor.
void PerfectRecallTest() {
  std::shared_ptr<const Game> game = TinyGame(
      ",max_evader_moves=2,reveal_moves=[2],evader_tickets=[2;1;0;1;1],"
      "pursuer_tickets=[2;1;0]");
  using OwnHistory = std::vector<std::pair<std::string, Action>>;
  std::map<std::pair<Player, std::string>,
           std::pair<OwnHistory, std::vector<float>>>
      seen;
  std::vector<OwnHistory> own(kNumPlayers);
  std::function<void(const State&)> visit = [&](const State& state) {
    if (state.IsTerminal()) return;
    const Player player = state.CurrentPlayer();
    std::string infostate;
    if (!state.IsChanceNode()) {
      infostate = state.InformationStateString(player);
      auto [it, inserted] =
          seen.insert({{player, infostate},
                       {own[player], state.InformationStateTensor(player)}});
      if (!inserted) {
        SPIEL_CHECK_TRUE(it->second.first == own[player]);
        SPIEL_CHECK_EQ(it->second.second, state.InformationStateTensor(player));
      }
    }
    for (Action action : state.LegalActions()) {
      if (!state.IsChanceNode()) own[player].push_back({infostate, action});
      visit(*state.Child(action));
      if (!state.IsChanceNode()) own[player].pop_back();
    }
  };
  visit(*game->NewInitialState());
  SPIEL_CHECK_EQ(seen.size(), size_t{11252});
}

// Plays random games under several configurations and resamples the pursuers'
// information state everywhere, terminal states included: the sampled state
// must look the same to the pursuers and end the same way.
void ResampleEverywhereTest() {
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  auto rng_fn = [&]() { return uniform(rng); };
  const std::vector<std::pair<int, std::string>> configurations = {
      {2, ""},
      {2, ",max_evader_moves=4,reveal_moves=[2;4]"},
      {2, ",max_evader_moves=6,reveal_moves=[3;6],evader_tickets=[1;1;0;0;1]"},
      {2, ",pursuer_tickets=[2;1;1],evader_tickets=[2;1;1;1;1]"},
      {2, ",pursuer_tickets=[0;0;1],evader_tickets=[1;0;0;0;1]"},
      {2, ",evader_tickets=[0;0;0;0;0]"},
      {1, ",pursuer_tickets=[0;1;0],evader_tickets=[0;0;0;1;0]"},
      {1, ",pursuer_tickets=[1;1;0],evader_tickets=[1;1;0;1;1]"}};
  for (const auto& [num_pursuers, parameters] : configurations) {
    std::shared_ptr<const Game> game = TinyGame(parameters, num_pursuers);
    for (int sim = 0; sim < 200; ++sim) {
      std::unique_ptr<State> state = game->NewInitialState();
      while (true) {
        std::unique_ptr<State> sample =
            state->ResampleFromInfostate(kPursuers, rng_fn);
        SPIEL_CHECK_EQ(sample->IsTerminal(), state->IsTerminal());
        SPIEL_CHECK_EQ(sample->Returns(), state->Returns());
        SPIEL_CHECK_EQ(sample->InformationStateString(kPursuers),
                       state->InformationStateString(kPursuers));
        SPIEL_CHECK_EQ(sample->InformationStateTensor(kPursuers),
                       state->InformationStateTensor(kPursuers));
        if (state->IsTerminal()) break;
        std::vector<Action> actions = state->LegalActions();
        std::uniform_int_distribution<int> pick(0, actions.size() - 1);
        state->ApplyAction(actions[pick(rng)]);
      }
    }
  }

  // A capture with the pursuer's last ticket. Without the capture the pursuer
  // would be left unable to move, and the evader would win at once instead of
  // losing later for lack of tickets: so every sample is the capture.
  std::shared_ptr<const Game> game =
      TinyGame(",pursuer_tickets=[0;1;0],evader_tickets=[0;0;0;1;0]", 1);
  std::unique_ptr<State> state = game->NewInitialState();
  Play(state.get(), "Evader starts at 1");
  Play(state.get(), "Pursuer 0 starts at 7");
  Play(state.get(), "Evader: 1->2 by secret");
  Play(state.get(), "Pursuer 0: 7->2 by bus");
  for (double r : {0.0, 0.5, 0.99}) {
    std::unique_ptr<State> sample =
        state->ResampleFromInfostate(kPursuers, [r]() { return r; });
    SPIEL_CHECK_EQ(sample->Returns(), (std::vector<double>{-1.0, 1.0}));
    SPIEL_CHECK_EQ(AsState(*sample).EvaderStation(), 1);  // Station 2.
  }
}

// Resampling covers every start the pursuers cannot rule out during the setup,
// and every final station of a very long game, whose path counts would
// overflow a double without rescaling.
void ResampleCoverageTest() {
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  auto rng_fn = [&]() { return uniform(rng); };
  std::shared_ptr<const Game> game = TinyGame();
  std::unique_ptr<State> state = game->NewInitialState();
  Play(state.get(), "Evader starts at 1");
  std::set<int> starts;
  for (int i = 0; i < 200; ++i) {
    starts.insert(AsState(*state->ResampleFromInfostate(kPursuers, rng_fn))
                      .EvaderStation());
  }
  SPIEL_CHECK_EQ(Sorted(starts),
                 (std::vector<int>{0, 3, 6, 8}));  // 1, 4, 7, 9.
  Play(state.get(), "Pursuer 0 starts at 4");
  starts.clear();
  for (int i = 0; i < 200; ++i) {
    starts.insert(AsState(*state->ResampleFromInfostate(kPursuers, rng_fn))
                      .EvaderStation());
  }
  SPIEL_CHECK_EQ(Sorted(starts), (std::vector<int>{0, 6, 8}));  // 1, 7 and 9.

  // Pursuers at 4 and 9 hold only bus tickets and block each other's bus
  // link, so they pass; the evader goes 1->2->1... with secret tickets.
  game = TinyGame(
      ",max_evader_moves=1100,reveal_moves=[],pursuer_tickets=[0;1;0],"
      "evader_tickets=[0;0;0;2000;0]");
  state = SetUp(*game, 1, 4, 9);
  for (int move = 0; move < 1000; ++move) {
    Play(state.get(),
         move % 2 == 0 ? "Evader: 1->2 by secret" : "Evader: 2->1 by secret");
    Play(state.get(), "Pursuer 0: pass");
    Play(state.get(), "Pursuer 1: pass");
  }
  std::set<int> finals;
  for (int i = 0; i < 50; ++i) {
    finals.insert(AsState(*state->ResampleFromInfostate(kPursuers, rng_fn))
                      .EvaderStation());
  }
  // The links left free form a bipartite graph: after an even number of moves
  // from 1 or 7 the evader is at 1, 3, 7 or 10, each with probability 0.2 to
  // 0.3. Unscaled counts overflowed, and the draw always gave 10.
  SPIEL_CHECK_EQ(Sorted(finals), (std::vector<int>{0, 2, 6, 9}));
}

// Tensors have a constant size and finite values within [0, 1], for both
// players and in every phase, terminal states included.
void TensorTest() {
  std::mt19937 rng(7);
  for (const std::shared_ptr<const Game>& game :
       {LoadGame("catch_the_hidden_evader"), TinyGame()}) {
    for (int sim = 0; sim < 20; ++sim) {
      std::unique_ptr<State> state = game->NewInitialState();
      while (true) {
        for (Player player = 0; player < kNumPlayers; ++player) {
          for (const std::vector<float>& tensor :
               {state->ObservationTensor(player),
                state->InformationStateTensor(player)}) {
            for (float value : tensor) {
              SPIEL_CHECK_TRUE(std::isfinite(value));
              SPIEL_CHECK_GE(value, 0.0f);
              SPIEL_CHECK_LE(value, 1.0f);
            }
          }
          SPIEL_CHECK_EQ(state->ObservationTensor(player).size(),
                         static_cast<size_t>(game->ObservationTensorSize()));
          SPIEL_CHECK_EQ(
              state->InformationStateTensor(player).size(),
              static_cast<size_t>(game->InformationStateTensorSize()));
        }
        if (state->IsTerminal()) break;
        std::vector<Action> actions = state->LegalActions();
        std::uniform_int_distribution<int> pick(0, actions.size() - 1);
        state->ApplyAction(actions[pick(rng)]);
      }
    }
  }
}

void StructTest() {
  std::mt19937 rng(11);
  std::shared_ptr<const Game> game = TinyGame();
  for (int sim = 0; sim < 20; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      const std::string json = state->ToJson();
      SPIEL_CHECK_EQ(nlohmann::json::parse(json).dump(),
                     CatchTheHiddenEvaderStateStruct(json).ToJson());
      for (Player player = 0; player < kNumPlayers; ++player) {
        auto obs = state->ToObservationStruct(player);
        const auto& contents =
            static_cast<const CatchTheHiddenEvaderObservationStruct&>(*obs);
        SPIEL_CHECK_EQ(contents.observing_player, player);
        if (player == kPursuers) SPIEL_CHECK_EQ(contents.evader_station, -1);
        const std::string obs_json = obs->ToJson();
        SPIEL_CHECK_EQ(
            nlohmann::json::parse(obs_json).dump(),
            CatchTheHiddenEvaderObservationStruct(obs_json).ToJson());
      }
      for (Action action : state->LegalActions()) {
        auto action_struct = state->ActionToStruct(action);
        SPIEL_CHECK_EQ(state->StructToActions(*action_struct),
                       std::vector<Action>{action});
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> pick(0, actions.size() - 1);
      state->ApplyAction(actions[pick(rng)]);
    }
  }
}

}  // namespace
}  // namespace catch_the_hidden_evader
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::catch_the_hidden_evader::BasicGameTests();
  open_spiel::catch_the_hidden_evader::ParseMapTest();
  open_spiel::catch_the_hidden_evader::CaptureByPursuerTest();
  open_spiel::catch_the_hidden_evader::CaptureByEvaderTest();
  open_spiel::catch_the_hidden_evader::DoubleMoveTest();
  open_spiel::catch_the_hidden_evader::LastMoveTest();
  open_spiel::catch_the_hidden_evader::ImmobilePursuersTest();
  open_spiel::catch_the_hidden_evader::BlockedPursuersPassTest();
  open_spiel::catch_the_hidden_evader::NoTicketEvaderTest();
  open_spiel::catch_the_hidden_evader::RevealTest();
  open_spiel::catch_the_hidden_evader::InformationStateTest();
  open_spiel::catch_the_hidden_evader::TerminalInformationTest();
  open_spiel::catch_the_hidden_evader::PerfectRecallTest();
  open_spiel::catch_the_hidden_evader::ResampleEverywhereTest();
  open_spiel::catch_the_hidden_evader::ResampleCoverageTest();
  open_spiel::catch_the_hidden_evader::TensorTest();
  open_spiel::catch_the_hidden_evader::StructTest();
}
