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

#include "mpg.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"
#include "mpg_generator.h"

namespace open_spiel::mpg {
    std::unique_ptr<MetaFactory> metaFactory = std::make_unique<ExampleFactory>();

    std::vector<NodeType> choose(NodeType n, int k, std::mt19937_64 &rng, bool distinct)
    {
        if(!distinct)
        {
            std::uniform_int_distribution<NodeType> dist(0,n-1);
            std::vector<NodeType> result(n);
            for(int i=0;i<k;i++)
            {
                auto j=dist(rng);
                result[i]=j;
            }
            return result;
        }
        if(k>n)
            throw std::invalid_argument("k must be less than or equal to n for distinct=true");
        else if(k<choose_parameters::threshold*n)
        {
            std::vector<NodeType> result;
            std::unordered_set<NodeType> v_set;
            while(v_set.size()<k)
            {
                std::uniform_int_distribution<NodeType> dist(0,n-1);
                auto j=dist(rng);
                v_set.insert(j);
            }
            for(auto i:v_set)
                result.push_back(i);
            return result;
        }
        else
        {
            std::vector<NodeType> result(n);
            for(int i=0;i<n;i++)
                result[i]=i;
            return choose(result,k,rng,distinct);
        }
    }

    namespace
    {
        // Facts about the game.
        const GameType kGameType{
            /*short_name=*/"mpg",
            /*long_name=*/"Mean Payoffs Game",
            GameType::Dynamics::kSequential,
            GameType::ChanceMode::kDeterministic,
            GameType::Information::kPerfectInformation,
            GameType::Utility::kZeroSum,
            GameType::RewardModel::kTerminal,
            /*max_num_players=*/2,
            /*min_num_players=*/2,
            /*provides_information_state_string=*/true,
            /*provides_information_state_tensor=*/false,
            /*provides_observation_string=*/true,
            /*provides_observation_tensor=*/true,
            /*parameter_specification=*/{{"max_moves",GameParameter(GameParameter::Type::kInt,true)}}  // no parameters
        };



        std::shared_ptr<const Game> Factory(const GameParameters& params)
        {
            return metaFactory->CreateGame(params);
        }



        REGISTER_SPIEL_GAME(kGameType, Factory);

        RegisterSingleTensorObserver single_tensor(kGameType.short_name);

    }  // namespace



std::string StateToString(NodeType state) {
    return absl::StrCat("State(", state, ")");
}


void MPGState::DoApplyAction(Action move) {
  SPIEL_CHECK_TRUE(graph[current_state].count(move));
  float K= static_cast<float>(num_moves_)/static_cast<float>(num_moves_+1);
  mean_payoff = K * mean_payoff +graph[current_state].at(move) / static_cast<float>(num_moves_ + 1);
  current_state = move;
  current_player_ = ! current_player_;
  num_moves_ += 1;
}

std::vector<Action> MPGState::LegalActions() const {
  if (IsTerminal()) return {};
    std::vector<Action> moves;
    moves.reserve(graph[current_state].size());
    for(auto [v, w]: graph[current_state])
        moves.push_back(v);
  return moves;
}

std::string MPGState::ActionToString(Player player,
                                           Action action_id) const {
  return game_->ActionToString(player, action_id);
}

MPGState::MPGState(std::shared_ptr<const Game> game) : State(game), graph(dynamic_cast<const MPGGame*>(game.get())->GetGraph()),
    adjacency_matrix(dynamic_cast<const MPGGame*>(game.get())->GetAdjacencyMatrix()),
    current_state(dynamic_cast<const MPGGame*>(game.get())->GetStartingState()), current_player_(0),
    num_moves_(0), state_history({0})
{
}

std::string MPGState::ToString() const {

    std::ostringstream stream;
    stream << "Graph: \n{";
    for(int i = 0; i < graph.size(); i++)
  {
      stream << i << ": ";
      for(auto [v, w]: graph[i])
          stream << "(" << v << ", " << w << ") ";
      stream << "\n";
  }
    stream << "}\n";
    stream << "Current state: " << current_state << "\n";
    stream << "Current player: " << current_player_ << "\n";

  return stream.str();
}

bool MPGState::IsTerminal() const {
  return num_moves_ >= MaxNumMoves();
}

    int MPGState::MaxNumMoves() const
    {
        return game_->GetParameters()["max_moves"].int_value();
    }

    double Player1Return(double mean_payoff)
    {
        if(mean_payoff > 0)
            return 1;
        else if(mean_payoff < 0)
            return -1;
        else
            return 0;
    }

    std::vector<double> MPGState::Returns() const
    {
    if (!IsTerminal())
        return {0.0, 0.0};
    else
    {
        auto S= Player1Return(mean_payoff);
        return {S, -S};
    }

}

std::string MPGState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string MPGState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void MPGState::ObservationTensor(Player player,
                                       absl::Span<float> values) const{
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Extract `environment` as a rank 3 tensor.
  auto environmentSubSpan= values.subspan(0, values.size() - 1);
  TensorView<3> view(environmentSubSpan, {graph.size(),graph.size(),2}, true);
    for(int u = 0; u < graph.size(); u++) for(auto [v, w]: graph[u])
    {
        view[{u, v, 0}] = v;
        view[{u, v, 1}] = w;
    }
    // Add the current state.
    values[values.size() - 1] = current_state;
}

void MPGState::UndoAction(Player player, Action move) {
SPIEL_CHECK_GE(move, 0);
SPIEL_CHECK_LT(move, num_distinct_actions_);
    float K= static_cast<float>(num_moves_)/static_cast<float>(num_moves_+1);
    state_history.pop();
    current_state = state_history.top();
    mean_payoff = mean_payoff / K - graph[current_state].at(move) / static_cast<float>(num_moves_ + 1);
    current_player_ = player;
    outcome_ = kInvalidPlayer;
    num_moves_ -= 1;
    history_.pop_back();
    --move_number_;
}

std::unique_ptr<State> MPGState::Clone() const {
  return std::unique_ptr<State>(new MPGState(*this));
}

std::string MPGGame::ActionToString(Player player,
                                          Action action_id) const {
  return absl::StrCat(action_id);
}

MPGGame::MPGGame(const GameParameters& params)
    : Game(kGameType, params) {}

    MPGGame::MPGGame(const GameParameters &_params, WeightedGraphType _graph, NodeType starting_state): Game(kGameType, _params), graph(std::move(_graph)),
                                                                                                      adjacency_matrix(graph.size(), std::vector<bool>(graph.size(), false)), starting_state(starting_state)
    {
        for(int u = 0; u < graph.size(); u++) for(auto [v, w]: graph[u])
            adjacency_matrix[u][v]= true;
    }


    WeightedGraphType WeightedGraphType::dual() const
    {
        WeightedGraphType dual(begin(),end());
        for(auto & adjList :dual)
            for(auto &[_,weight]:adjList)
                weight=-weight;
        return dual;
    }

    WeightedGraphType WeightedGraphType::operator~() const {
        return dual();
    }

    WeightedGraphType WeightedGraphType::from_string(const std::string &str) {
        std::stringstream  stream(str);
        WeightedGraphType graph;
        int graph_size=0;
        while(stream)
        {
            int u, v;
            float w;
            stream >> u >> v >> w;
            graph_size=std::max({graph_size,u+1,v+1});
            if(graph_size> graph.size())
            {
                //To guarantee linear time complexity
                graph.reserve(std::max<size_t>(2*graph_size,graph.size()));
                graph.resize(graph_size);
            }
            graph[u].emplace(v, w);
        }
        return graph;
    }
}  // namespace open_spiel
