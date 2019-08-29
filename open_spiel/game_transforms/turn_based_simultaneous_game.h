// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_OPEN_SPIEL_GAME_TRANSFORMS_TURN_BASED_SIMULTANEOUS_GAME_H_
#define THIRD_PARTY_OPEN_SPIEL_GAME_TRANSFORMS_TURN_BASED_SIMULTANEOUS_GAME_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// This wrapper turns any n-player simultaneous move game into an equivalent
// turn-based game where simultaneous move nodes are encoded as n turns.
//
// TODO(lanctot): implement UndoAction for these games.

namespace open_spiel {

class TurnBasedSimultaneousState : public State {
 public:
  TurnBasedSimultaneousState(int num_distinct_actions, int num_players,
                             std::unique_ptr<State> state);
  TurnBasedSimultaneousState(const TurnBasedSimultaneousState& other);

  int CurrentPlayer() const override;
  std::string ActionToString(int player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(int player) const override;
  void InformationStateAsNormalizedVector(
      int player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Access to the wrapped state, used for debugging and in the tests.
  const State* SimultaneousGameState() const { return state_.get(); }
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void DetermineWhoseTurn();
  void RolloutModeIncrementCurrentPlayer();

  std::unique_ptr<State> state_;

  // A vector of actions that is used primarily to store the intermediate
  // actions taken by the players when extending the simultaneous move nodes
  // to be turn-based.
  std::vector<Action> action_vector_;

  // The current player (which will never be kSimultaneousPlayerId).
  int current_player_;

  // Are we currently rolling out a simultaneous move node?
  bool rollout_mode_;
};

class TurnBasedSimultaneousGame : public Game {
 public:
  explicit TurnBasedSimultaneousGame(std::unique_ptr<Game> game);
  TurnBasedSimultaneousGame(const TurnBasedSimultaneousGame& other)
      : Game(other), game_(other.game_->Clone()) {}

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new TurnBasedSimultaneousState(
        NumDistinctActions(), NumPlayers(), game_->NewInitialState()));
  }

  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }
  int MaxChanceOutcomes() const override { return game_->MaxChanceOutcomes(); }
  int NumPlayers() const override { return game_->NumPlayers(); }
  double MinUtility() const override { return game_->MinUtility(); }
  double MaxUtility() const override { return game_->MaxUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }
  std::vector<int> InformationStateNormalizedVectorShape() const override {
    // We flatten the representation of the underlying game and add one-hot
    // indications of the to-play player and the observing player.
    return {2 * NumPlayers() + game_->InformationStateNormalizedVectorSize()};
  }
  int MaxGameLength() const override {
    return game_->MaxGameLength() * NumPlayers();
  }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new TurnBasedSimultaneousGame(*this));
  }

 private:
  std::unique_ptr<Game> game_;
};

// Equivalent loader functions that return back the transformed game.
// Important: takes ownership of the game that is passed in.
std::unique_ptr<Game> ConvertToTurnBased(const Game& game);

// These are equivalent to LoadGame but return a converted game. They are simple
// wrappers provided for the Python API.
std::unique_ptr<Game> LoadGameAsTurnBased(const std::string& name);
std::unique_ptr<Game> LoadGameAsTurnBased(const std::string& name,
                                          const GameParameters& params);

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAME_TRANSFORMS_TURN_BASED_SIMULTANEOUS_GAME_H_
