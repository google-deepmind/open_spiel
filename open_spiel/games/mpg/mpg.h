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

#ifndef OPEN_SPIEL_GAMES_MPG_H_
#define OPEN_SPIEL_GAMES_MPG_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stack>

#include "open_spiel/spiel.h"

// Simple game of Noughts and Crosses:
// https://en.wikipedia.org/wiki/Tic-tac-toe
//
// Parameters: none

namespace open_spiel::mpg
{

    // Constants.
    inline constexpr int kNumPlayers = 2;
    inline constexpr int kNumRows = 3;
    inline constexpr int kNumCols = 3;
    inline constexpr int kNumCells = kNumRows * kNumCols;
    inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.
    using WeightType=float;
    using NodeType=std::uint16_t;
    struct WeightedOutgoingEdge
    {
        NodeType to;
        WeightType weight;
        WeightedOutgoingEdge(NodeType to, WeightType weight): to(to), weight(weight){}
    };

    struct  AdjacencyMatrixType: public std::vector<std::vector<bool>>
    {
        using std::vector<std::vector<bool>>::vector;
    };

    struct WeightedGraphType: public std::vector<std::map<NodeType,WeightType>>
    {
        using std::vector<std::map<NodeType,WeightType>>::vector;
        [[nodiscard]] WeightedGraphType dual() const;
        WeightedGraphType operator~() const;
        static WeightedGraphType from_string(const std::string& str);
        [[nodiscard]] AdjacencyMatrixType adjacency_matrix() const;
    };
    struct  GraphType :public std::vector<std::vector<NodeType>>
    {
        using std::vector<std::vector<NodeType>>::vector;
    };

    std::ostream& operator<< (std::ostream& os, const WeightedGraphType& graph);
    std::ostream& operator<< (std::ostream& os, const GraphType& graph);
    std::ostream& operator<< (std::ostream& os, const AdjacencyMatrixType& graph);


    enum ObservationAxis : size_t
    {
        kAdjacencyMatrix = 0,
        kWeightsMatrix
    };

    enum PlayerIdentifier : size_t
    {
        kPlayer1 = 0,
        kPlayer2,
        kMaxPlayer = kPlayer1,
        kMinPlayer = kPlayer2
    };

    struct Environment
    {
        WeightedGraphType graph;
        NodeType starting_state{};
        Environment()= default;
        Environment(WeightedGraphType graph, NodeType starting_state);
    };

    // State of an in-play game.
    class MPGEnvironmentState : public State
    {
     public:
        MPGEnvironmentState(std::shared_ptr<const Game> game,std::shared_ptr<Environment> environment);
        explicit MPGEnvironmentState(const std::shared_ptr<const Game>& game);

        MPGEnvironmentState(const MPGEnvironmentState&) = default;
        MPGEnvironmentState& operator=(const MPGEnvironmentState&) = default;

        [[nodiscard]] Player CurrentPlayer() const override
        {
            return IsTerminal() ? kTerminalPlayerId : current_player_;
        }
        std::string ActionToString(Player player, Action action_id) const override;
        std::string ToString() const override;
        bool IsTerminal() const override;
        [[nodiscard]] std::vector<double> Returns() const override;
        [[nodiscard]] std::string InformationStateString(Player player) const override;
        [[nodiscard]] std::string ObservationString(Player player) const override;
        void ObservationTensor(Player player,
                             absl::Span<float> values) const override;
        [[nodiscard]] std::unique_ptr<State> Clone() const override;
        void UndoAction(Player player, Action move) override;
        std::vector<Action> LegalActions() const override;
        NodeType StateAt(NodeType cell) const { return cell; }
        Player outcome() const { return outcome_; }

        [[nodiscard]] virtual int MaxNumMoves() const;

        // Only used by Ultimate Tic-Tac-Toe.
        void SetCurrentPlayer(Player player) { current_player_ = player; }
        NodeType GetCurrentState() const;
        WeightType GetMeanPayoff() const;
     protected:

      void DoApplyAction(Action move) override;
      NodeType current_state = 0;
      WeightType mean_payoff = 0;
      std::vector<NodeType> state_history;
      std::shared_ptr<Environment> environment;

     private:
      Player current_player_ = 0;         // Player zero goes first
      Player outcome_ = kInvalidPlayer;
      int num_moves_ = 0;
      friend class MPGMetaGame;
    };


    // Game object.
    class MPGMetaGame : public Game
    {
     public:
      explicit MPGMetaGame(const GameParameters&params, std::unique_ptr<class EnvironmentFactory> environment_factory = nullptr);
      int NumDistinctActions() const override;
      std::unique_ptr<State> NewInitialState() const override;
      int NumPlayers() const override { return kNumPlayers; }
      double MinUtility() const override { return -1; }
      absl::optional<double> UtilitySum() const override { return 0; }
      double MaxUtility() const override { return 1; }
      std::vector<int> ObservationTensorShape() const override {
          throw std::runtime_error("Not implemented");
      }

      Game::TensorShapeSpecs ObservationTensorShapeSpecs() const override;

      std::vector<std::vector<int>> ObservationTensorsShapeList() const override
      {
          return {{MaxGraphSize(),MaxGraphSize(),2},{1}};
      }

        std::unique_ptr<State> NewInitialEnvironmentState() const override;

      int MaxGameLength() const override;
      int MaxGraphSize() const;
      std::string ActionToString(Player player, Action action_id) const override;
      std::shared_ptr<Environment> GetLastEnvironment() const;
    protected:
        std::unique_ptr<EnvironmentFactory> environment_factory;
        mutable std::shared_ptr<Environment> last_environment;
        mutable absl::Mutex environment_mutex;
        mutable std::map<int, std::shared_ptr<Environment>> environment_cache;
        int cache_max_size=100;
        mutable int current_id=0;
    };

    NodeType PlayerToState(Player player);
    std::string StateToString(NodeType state);


}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MPG_H_
