
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
using open_spiel::Action;
using open_spiel::ActionsAndProbs;

/* We need this because games are shared pointers and we need to return
 raw pointers to objects that contain them.*/
namespace {
struct GamePointerHolder {
  std::shared_ptr<const Game> ptr;
};

template <class T>
T* AllocBuf(const std::vector<T>& vec, int* size) {
  *size = vec.size();
  size_t num_bytes = *size  * sizeof(T);
  T* buf = static_cast<T*>(malloc(num_bytes));
  memcpy(buf, vec.data(), num_bytes);
  return buf;
}

}  // namespace

extern "C" {

void test() {
  std::cout << "This is a test!" << std::endl;
}

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

int* GameObservationTensorShape(const void* game_ptr, int* size) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  std::vector<int> shape = game->ObservationTensorShape();
  return AllocBuf(shape, size);
}

int* GameInformationStateTensorShape(const void* game_ptr, int* size) {
  std::shared_ptr<const Game> game =
      reinterpret_cast<const GamePointerHolder*>(game_ptr)->ptr;
  std::vector<int> shape = game->InformationStateTensorShape();
  return AllocBuf(shape, size);
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

long* StateLegalActions(const void* state_ptr, int* num_legal_actions) {
  assert(sizeof(long) == sizeof(Action));
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::vector<Action> legal_actions = state->LegalActions();
  return AllocBuf(legal_actions, num_legal_actions);
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

int StateIsTerminal(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->IsTerminal() ? 1 : 0;
}

int StateIsChanceNode(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  return state->IsChanceNode() ? 1 : 0;
}

void StateApplyAction(void* state_ptr, long action) {
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

double* StateReturns(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::vector<double> returns = state->Returns();
  int size = 0;
  return AllocBuf(returns, &size);
}

double* StateChanceOutcomeProbs(const void* state_ptr, int* size) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  ActionsAndProbs chance_outcomes = state->ChanceOutcomes(); 
  *size = chance_outcomes.size();
  size_t num_bytes = *size  * sizeof(double);
  double* buf = static_cast<double*>(malloc(num_bytes));
  for (int i = 0; i < chance_outcomes.size(); ++i) {
    buf[i] = chance_outcomes[i].second;
  }
  return buf;
}

char* StateObservationString(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string state_str = state->ObservationString();
  return strdup(state_str.c_str());
}

char* StateInformationStateString(const void* state_ptr) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::string state_str = state->InformationStateString();
  return strdup(state_str.c_str());
}

float* StateObservationTensor(const void* state_ptr, int *size) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::vector<float> tensor = state->ObservationTensor();
  return AllocBuf(tensor, size);
}

float* StateInformationStateTensor(const void* state_ptr, int *size) {
  const State* state = reinterpret_cast<const State*>(state_ptr);
  std::vector<float> tensor = state->InformationStateTensor();
  return AllocBuf(tensor, size);
}

}  /* extern "C" */

