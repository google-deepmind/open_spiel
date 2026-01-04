// Copyright 2021 DeepMind Technologies Limited
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

#include "go_open_spiel.h"  // NOLINT

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"

using open_spiel::Game;
using open_spiel::State;

/* We need this because games are shared pointers and we need to return
 raw pointers to objects that contain them.*/
namespace {
struct GamePointerHolder {
  std::shared_ptr<const Game> ptr;
};
}  // namespace

extern "C" {

void Test() { std::cout << "Testing, testing, 1 2 3!" << std::endl; }

/* Game functions. */
void* LoadGame(const char* name) {
  return reinterpret_cast<void*>(
      new GamePointerHolder{open_spiel::LoadGame(name)});
}

void DeleteGame(void* game_ptr) {
  GamePointerHolder* game = reinterpret_cast<GamePointerHolder*>(game_ptr);
  delete game;
}

char* GameShortName(const void* game_ptr) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  std::string short_name = game->GetType().short_name;
  return strdup(short_name.c_str());
}

char* GameLongName(const void* game_ptr) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  std::string long_name = game->GetType().long_name;
  return strdup(long_name.c_str());
}

void* GameNewInitialState(const void* game_ptr) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  std::unique_ptr<State> state = game->NewInitialState();
  void* state_ptr = reinterpret_cast<void*>(state.release());
  return state_ptr;
}

int GameNumPlayers(const void* game_ptr) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  return game->NumPlayers();
}

int GameMaxGameLength(const void* game_ptr) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  return game->MaxGameLength();
}

int GameNumDistinctActions(const void* game_ptr) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  return game->NumDistinctActions();
}

void DeleteState(void* state_ptr) {
  State* state = reinterpret_cast<State*>(state_ptr);
  delete state;
}

void* StateClone(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::unique_ptr<State> state_copy = state->Clone();
  return reinterpret_cast<void*>(state_copy.release());
}

char* StateToString(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string state_str = state->ToString();
  return strdup(state_str.c_str());
}

int StateNumLegalActions(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->LegalActions().size();
}

int StateNumDistinctActions(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->NumDistinctActions();
}

void StateFillLegalActions(const void* state_ptr, void* array_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  int* legal_actions_array = reinterpret_cast<int*>(array_ptr);
  absl::c_copy(state->LegalActions(), legal_actions_array);
}

void StateFillLegalActionsMask(const void* state_ptr, void* array_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  int* legal_actions_mask_array = reinterpret_cast<int*>(array_ptr);
  std::vector<int> legal_actions_mask = state->LegalActionsMask();
  absl::c_copy(state->LegalActionsMask(), legal_actions_mask_array);
}

int StateSizeObservation(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->GetGame()->ObservationTensorSize();
}

void StateFillObservation(const void* state_ptr, void* array_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  double* observation_array = reinterpret_cast<double*>(array_ptr);
  absl::c_copy(state->ObservationTensor(), observation_array);
}

int StateSizeChanceOutcomes(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->ChanceOutcomes().size();
}

void StateFillChanceOutcomes(const void* state_ptr, void* action_ptr,
                             void* proba_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  int* action_array = reinterpret_cast<int*>(action_ptr);
  double* proba_array = reinterpret_cast<double*>(proba_ptr);
  std::vector<std::pair<open_spiel::Action, double>> outcomes =
      state->ChanceOutcomes();
  std::pair<open_spiel::Action, double> outcome;
  for (int i = 0; i < outcomes.size(); ++i) {
    outcome = outcomes[i];
    action_array[i] = outcome.first;
    proba_array[i] = outcome.second;
  }
}

void StateFillObservationPlayer(const void* state_ptr, void* array_ptr,
                                int player) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  double* observation_array = reinterpret_cast<double*>(array_ptr);
  std::vector<float> observation = state->ObservationTensor(player);
  for (int i = 0; i < observation.size(); ++i) {
    observation_array[i] = observation[i];
  }
}

int StateSizeInformationState(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->GetGame()->InformationStateTensorSize();
}

void StateFillInformationState(const void* state_ptr, void* array_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  double* information_state_array = reinterpret_cast<double*>(array_ptr);
  std::vector<float> information_state = state->InformationStateTensor();
  for (int i = 0; i < information_state.size(); ++i) {
    information_state_array[i] = information_state[i];
  }
}

void StateFillInformationStatePlayer(const void* state_ptr, void* array_ptr,
                                     int player) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  double* information_state_array = reinterpret_cast<double*>(array_ptr);
  std::vector<float> information_state = state->InformationStateTensor(player);
  absl::c_copy(information_state, information_state_array);
}

int StateIsTerminal(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->IsTerminal() ? 1 : 0;
}

int StateIsChanceNode(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->IsChanceNode() ? 1 : 0;
}

int StateCurrentPlayer(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->CurrentPlayer();
}

char* StateActionToString(const void* state_ptr, int player, int action) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string action_str = state->ActionToString(player, action);
  return strdup(action_str.c_str());
}

void StateApplyAction(void* state_ptr, int action) {
  State* state = reinterpret_cast<State*>(state_ptr);
  state->ApplyAction(action);
}

double StatePlayerReturn(const void* state_ptr, int player) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->PlayerReturn(player);
}

} /* extern "C" */
