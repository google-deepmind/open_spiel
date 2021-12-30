
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

/* We need this because games are shared pointers and we need to return
 raw pointers to objects that contain them.*/
namespace {
struct GamePointerHolder {
  std::shared_ptr<const Game> ptr;
};
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
  *num_legal_actions = legal_actions.size();
  size_t size = *num_legal_actions * sizeof(long);
  long* buf = static_cast<long*>(malloc(size));
  memcpy(buf, legal_actions.data(), size);
  return buf;
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
  size_t size = returns.size() * sizeof(double);
  double* buf = static_cast<double*>(malloc(size));
  memcpy(buf, returns.data(), size);
  return buf;
}

}  /* extern "C" */

