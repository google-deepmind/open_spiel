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

// ?? - for question i have - or to check later
// xx - to remove before submission

#ifndef OPEN_SPIEL_GAMES_PHANTOM_HEX_H_
#define OPEN_SPIEL_GAMES_PHANTOM_HEX_H_  // What r these for ??

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector> // Make sure to check what r we importingg each for ??

#include "open_spiel/games/hex.h"
#include "open_spiel/spiel.h" // base functions etc. for the game xx

// Game description insert ??

namespace open_spiel {
  namespace phantom_hex {
  
    // kDefaultObsType decide if we will reveal any info. to other player xx
    inline constexpr const char* kDefaultObsType = "reveal-nothing"; 
    // do we need inline & constexpr here ? I feel like it's not 
    // necessary as they dont add any value here ??

    // longest sequence xx
    // For hex longest sequence needs to be determined using board size ??
    inline const int kNumOfCells = hex::kDefaultBoardSize * hex::kDefaultBoardSize;
    // EDIT Hex, to have x, y as board size not x, x ??
    inline constexpr int kLongestSequence = 2 * kNumOfCells - 1;
    inline constexpr int kBitsPerAction = 10; // Not sure why this is 10 ??

    // Add here if anything else is needed to be revealed ??
    enum class ObservationType {
      kRevealNothing,
      kRevealNumTurns, // how many turns have passed ??
    };

    class PhantomHexState: public State { // why do we need the public here ??
      public: 
        // Constructor created. shared_ptr was needed to instanciate the game, 
        // not really sure why ?? 
        PhantomHexState(std::shared_ptr<const Game> game, ObservationType obs_type);

        // Basically we are just pulling the hex board and redefiningg all the methods
        // same as that on this class xx
        // returning the current player, we will use this for updating the actions
        // etc xx
        Player CurrentPlayer() const override {return state_.CurrentPlayer(); }

        // im not sure why do we have action_id ??
        std::string ActionToString(Player player, Action action_id) const override {
          return state_.ActionToString(player, action_id);
        }
        std::string ToString() const override {return state_.ToString();}
        bool IsTerminal() const override {return state_.IsTerminal();}
        std::vector<double> Returns() const override {return state_.Returns();}
        std::string ObservationString(Player player) const override;
        // Span<T> is something like a vector, Im not sure if I got the difference
        // Try to check what it is ??
        void ObservationTensor(Player player,
                               absl::Span<float> values) const override;
        
        // Phantom games funcs.
        std::string InformationStateString(Player player) const override;
        void InformationStateTensor (Player player,
                                absl::Span<float> values) const override;

        std::unique_ptr<State> Clone() const override;
        void UndoAction(Player player, Action move) override;
        std::vector<Action> LegalActions() const override;

      protected:
        void DoApplyAction(Action move) override;

      private:
        std::string ViewToString(Player player) const;
        std::string ActionSequenceToString(Player player) const;

        hex::HexState state_; // game state - from reg hex board xx
        ObservationType obs_type;  // observation types stored var xx

        // make sure u change this to _history on base class after all
        std::vector<std::pair<int, Action>> action_sequence_;
        std::array<hex::CellState, kNumOfCells> black_view_;
        std::array<hex::CellState, kNumOfCells> white_view_;
    };

    class PhantomHexGame: public Game {
      public:
        explicit PhantomHexGame(const GameParameters& params);
        std::unique_ptr<State> NewInitialState() const override {
          return std::unique_ptr<State>(
            new PhantomHexState(shared_from_this(), obs_type_)
          );
        }
        int NumDistinctActions() const override {
          return game_ -> NumDistinctActions();
        }
        int NumPlayers() const override {return game_->NumPlayers();}
        double MinUtility() const override {return game_->MinUtility();}
        double UtilitySum() const override {return game_->UtilitySum();}
        double MaxUtility() const override {return game_->MaxUtility();}

        // These will depend on the obstype
        std::vector<int> InformationStateTensorShape() const override;
        std::vector<int> ObservationTensorShape() const override;
        int MaxGameLength() const override {return kLongestSequence;}
        
      private:
        std::shared_ptr<hex::HexGame> game_;
        ObservationType obs_type_;
    };

    // Whats the point on having inline here, calling switch will
    // force quit inline ??

    // Im  not sure whhat this line is for ??
    inline std::ostream& operator << (std::ostream& stream,
                                      const ObservationType& obs_type) {
      switch (obs_type) {
        case ObservationType::kRevealNothing:
          return stream << "Reveal Nothing";
        case ObservationType::kRevealNumTurns:
          return stream << "Reveal Num Turns";
        default:
          SpielFatalError("Unknown observation type");
      }
    }
  }
}

#endif