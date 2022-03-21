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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"

using ::open_spiel::Action;
using ::open_spiel::ActionsAndProbs;
using ::open_spiel::Game;
using ::open_spiel::GameParameter;
using ::open_spiel::GameParameters;
using ::open_spiel::State;

// A number of functions in this file returns pointers to dynamically-allocated
// memory. These are temporary memory buffers used to store data that must be
// freed on the Rust API (rust_open_spiel.rs).

/* We need this because games are shared pointers and we need to return
 raw pointers to objects that contain them.*/
namespace {

struct GamePointerHolder {
  std::shared_ptr<const Game> ptr;
};

template <class T>
T* AllocBuf(const std::vector<T>& vec, int* size) {
  *size = vec.size();
  size_t num_bytes = *size * sizeof(T);
  T* buf = static_cast<T*>(malloc(num_bytes));
  memcpy(buf, vec.data(), num_bytes);
  return buf;
}

char* AllocAndCopyString(const std::string& str) {
  char* buf = static_cast<char*>(malloc(str.length() * sizeof(char)));
  strncpy(buf, str.data(), str.length());
  return buf;
}

}  // namespace

extern "C" {

/* GameParameters functions. */
void* NewGameParameters() {
  return reinterpret_cast<void*>(new GameParameters());
}

void DeleteGameParameters(void* params_ptr) {
  GameParameters* params = reinterpret_cast<GameParameters*>(params_ptr);
  delete params;
}

void GameParametersSetInt(void* params_ptr, const char* key, int value) {
  GameParameters* params = reinterpret_cast<GameParameters*>(params_ptr);
  params->insert_or_assign(std::string(key), GameParameter(value));
}

void GameParametersSetDouble(void* params_ptr, const char* key, double value) {
  GameParameters* params = reinterpret_cast<GameParameters*>(params_ptr);
  params->insert_or_assign(std::string(key), GameParameter(value));
}

void GameParametersSetString(void* params_ptr, const char* key,
                             const char* value) {
  GameParameters* params = reinterpret_cast<GameParameters*>(params_ptr);
  params->insert_or_assign(std::string(key), GameParameter(std::string(value)));
}

/* Game functions. */
void* LoadGame(const char* name) {
  return reinterpret_cast<void*>(
      new GamePointerHolder{open_spiel::LoadGame(name)});
}

void* LoadGameFromParameters(const void* params_ptr) {
  const GameParameters* params =
      reinterpret_cast<const GameParameters*>(params_ptr);
  return reinterpret_cast<void*>(
      new GamePointerHolder{open_spiel::LoadGame(*params)});
}

void DeleteGame(void* game_ptr) {
  GamePointerHolder* game = reinterpret_cast<GamePointerHolder*>(game_ptr);
  delete game;
}

char* GameShortName(const void* game_ptr, unsigned long* length) {  // NOLINT
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  std::string short_name = game->GetType().short_name;
  *length = short_name.length();
  return AllocAndCopyString(short_name);
}

char* GameLongName(const void* game_ptr, unsigned long* length) {  // NOLINT
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  std::string long_name = game->GetType().long_name;
  *length = long_name.length();
  return AllocAndCopyString(long_name);
}

void* GameNewInitialState(const void* game_ptr) {
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  std::unique_ptr<State> state = game->NewInitialState();
  void* state_ptr = reinterpret_cast<void*>(state.release());
  return state_ptr;
}

int GameNumPlayers(const void* game_ptr) {
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  return game->NumPlayers();
}

int GameMaxGameLength(const void* game_ptr) {
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  return game->MaxGameLength();
}

int GameNumDistinctActions(const void* game_ptr) {
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  return game->NumDistinctActions();
}

int* GameObservationTensorShape(const void* game_ptr, int* size) {
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  std::vector<int> shape = game->ObservationTensorShape();
  return AllocBuf(shape, size);
}

int* GameInformationStateTensorShape(const void* game_ptr, int* size) {
  const Game* game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr.get();
  std::vector<int> shape = game->InformationStateTensorShape();
  return AllocBuf(shape, size);
}

/* State functions. */
void DeleteState(void* state_ptr) {
  State* state = reinterpret_cast<State*>(state_ptr);
  delete state;
}

void* StateClone(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::unique_ptr<State> state_copy = state->Clone();
  return reinterpret_cast<void*>(state_copy.release());
}

char* StateToString(const void* state_ptr, unsigned long* length) {  // NOLINT
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string state_str = state->ToString();
  *length = state_str.length();
  return AllocAndCopyString(state_str);
}

long* StateLegalActions(const void* state_ptr,  // NOLINT
                        int* num_legal_actions) {
  assert(sizeof(long) == sizeof(Action));  // NOLINT
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::vector<Action> legal_actions = state->LegalActions();
  return AllocBuf(legal_actions, num_legal_actions);
}

int StateCurrentPlayer(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->CurrentPlayer();
}

char* StateActionToString(const void* state_ptr, int player, int action,
                          unsigned long* length) {  // NOLINT
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string action_str = state->ActionToString(player, action);
  *length = action_str.length();
  return AllocAndCopyString(action_str);
}

int StateIsTerminal(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->IsTerminal() ? 1 : 0;
}

int StateIsChanceNode(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->IsChanceNode() ? 1 : 0;
}

void StateApplyAction(void* state_ptr, long action) {  // NOLINT
  State* state = reinterpret_cast<State*>(state_ptr);
  state->ApplyAction(action);
}

double StatePlayerReturn(const void* state_ptr, int player) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->PlayerReturn(player);
}

int StateNumPlayers(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->NumPlayers();
}

void StateReturns(const void* state_ptr, double* returns_buf) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::vector<double> returns = state->Returns();
  memcpy(returns_buf, returns.data(), returns.size() * sizeof(double));
}

double* StateChanceOutcomeProbs(const void* state_ptr, int* size) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  ActionsAndProbs chance_outcomes = state->ChanceOutcomes();
  *size = chance_outcomes.size();
  size_t num_bytes = *size * sizeof(double);
  double* buf = static_cast<double*>(malloc(num_bytes));
  for (int i = 0; i < chance_outcomes.size(); ++i) {
    buf[i] = chance_outcomes[i].second;
  }
  return buf;
}

char* StateObservationString(const void* state_ptr,
                             unsigned long* length) {  // NOLINT
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string obs_str = state->ObservationString();
  *length = obs_str.length();
  return AllocAndCopyString(obs_str);
}

char* StateInformationStateString(const void* state_ptr,
                                  unsigned long* length) {  // NOLINT
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string infostate_str = state->InformationStateString();
  *length = infostate_str.length();
  return AllocAndCopyString(infostate_str);
}

int StateInformationStateTensorSize(const void* state_ptr) {
  const Game* parent_game =
      reinterpret_cast<const State*>(state_ptr)->GetGame().get();
  return parent_game->InformationStateTensorSize();
}

int StateObservationTensorSize(const void* state_ptr) {
  const Game* parent_game =
      reinterpret_cast<const State*>(state_ptr)->GetGame().get();
  return parent_game->ObservationTensorSize();
}

void StateObservationTensor(const void* state_ptr, float* obs_buf, int length) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  open_spiel::Player cur_player = state->CurrentPlayer();
  // Currently turn-based games are assumed. See README.md for how to remove
  // this restriction.
  SPIEL_CHECK_GE(cur_player, 0);
  state->ObservationTensor(cur_player, absl::MakeSpan(obs_buf, length));
}

void StateInformationStateTensor(const void* state_ptr, float* infostate_buf,
                                 int length) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  open_spiel::Player cur_player = state->CurrentPlayer();
  // Currently turn-based games are assumed. See README.md for how to remove
  // this restriction.
  SPIEL_CHECK_GE(cur_player, 0);
  state->InformationStateTensor(cur_player,
                                absl::MakeSpan(infostate_buf, length));
}

} /* extern "C" */
