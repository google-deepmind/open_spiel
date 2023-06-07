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
#include <iomanip>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"
#include "mpg_generator.h"
#include "string_compress.h"

namespace open_spiel::mpg {
    //std::unique_ptr<MetaFactory> metaFactory = std::make_unique<ExampleFactory>();
    std::unique_ptr<MetaFactory> metaFactory = std::make_unique<ParserMetaFactory>();

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
            /*parameter_specification=*/{{"max_moves",GameParameter(GameParameter::Type::kInt,true)},
                                         {"generator",GameParameter(GameParameter::Type::kString,true)},
                                         {"max_size",GameParameter(GameParameter::Type::kInt,true)},
                                         {"generator_params", GameParameter(GameParameter::Type::kString,false)},
                                         {"specs_file",GameParameter(GameParameter::Type::kString,false)},
                                         {"representation",GameParameter(GameParameter::Type::kString)}}  // no parameters
        };



        std::shared_ptr<const Game> Factory(const GameParameters& params)
        {
            auto game=metaFactory->CreateGame(params);
            game->NewInitialEnvironmentState();
            return game;
        }



        REGISTER_SPIEL_GAME(kGameType, Factory);

        RegisterSingleTensorObserver single_tensor(kGameType.short_name);

        WeightType  MeanPayoff(const std::vector<WeightType> & M)
        {
            if(M.empty())
                return 0;
            WeightType sum = 0;
            for(auto w: M)
                sum += w;
            return sum / static_cast<WeightType>(M.size());
        }

        WeightType WeightFromPerspective(WeightType weight,Player player)
        {
            if(player == mpg::PlayerIdentifier::kMaxPlayer)
                return weight;
            else
                return -weight;
        }

    }  // namespace



    std::string StateToString(NodeType state) {
        return absl::StrCat("State(", state, ")");
    }

    std::ostream& operator<<(std::ostream &os, const WeightedGraphType &graph)
    {
        for(int u=0;u<graph.size();++u)
        {
            os << u << ": ";
            for(auto [v,w]: graph[u])
                os << v << "(" << w << ") ";
            os << '\n';
        }
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const GraphType &graph)
    {
        for(int u=0;u<graph.size();++u)
        {
            os << u << ": ";
            for(auto v: graph[u])
                os << v << " ";
            os << '\n';
        }
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const AdjacencyMatrixType &graph) {
        for(int u=0;u<graph.size();++u)
        {
            for (int v = 0; v < graph.size(); ++v)
                os << graph[u][v] << ' ';
            os << '\n';
        }
        return os;
    }


    void MPGEnvironmentState::DoApplyAction(Action move)
    {
      SPIEL_CHECK_TRUE(environment->graph[current_state].count(move));
      float K= static_cast<float>(num_moves_)/static_cast<float>(num_moves_+1);
      mean_payoff = K * mean_payoff +environment->graph[current_state].at(move) / static_cast<float>(num_moves_ + 1);
      current_state = move;
      current_player_ = ! current_player_;
      state_history.push_back(current_state);
      num_moves_ += 1;
        }

    std::vector<Action> MPGEnvironmentState::LegalActions() const {
      if (IsTerminal()) return {};
        std::vector<Action> moves;
        moves.reserve(environment->graph[current_state].size());
        for(auto [v, w]: environment->graph[current_state])
            moves.push_back(v);
      return moves;
    }

    std::string MPGEnvironmentState::ActionToString(Player player,
                                                    Action action_id) const {
      return game_->ActionToString(player, action_id);
    }

    MPGEnvironmentState::MPGEnvironmentState(const std::shared_ptr<const Game>& game) : MPGEnvironmentState(game,dynamic_cast<const MPGMetaGame *>(game.get())->GetLastEnvironment())
    {
    }

    std::string MPGEnvironmentState::ToString() const
    {
        std::ostringstream stream;
        if(!game_->GetParameters().count("representation") || game_->GetParameters().at("representation").string_value()=="normal")
        {
            stream << "@@@\n";
            stream << "@Graph: \n{";
            for(int i = 0; i < environment->graph.size(); i++)
            {
                stream << i << ": ";
                for(auto [v, w]: environment->graph[i])
                    stream << "(" << v << ", " << w << ") ";
                stream << "\n";
            }
            stream << "}\n";
            stream << "@Current state: " << current_state << "\n";
            stream << "@Current player: " << current_player_ << "\n";
            stream << "@Number of moves: " << num_moves_ << "\n";
            stream << "@@@\n";
            return stream.str();
        }
        else if (game_->GetParameters().at("representation").string_value() == "compressed")
        {
            for(int i = 0; i < environment->graph.size(); i++)
            {
                stream << i << ":";
                for(auto [v, w]: environment->graph[i])
                    stream  << v << ":" << std::setprecision(3) << w << ":";
                stream << "\n";
            }
            stream << current_state << ':' << current_player_;
            return base64_encode(bzip_compress(stream.str()));
        }
        else if (game_->GetParameters().at("representation").string_value() == "hash")
        {
            for(int i = 0; i < environment->graph.size(); i++)
            {
                stream << i << ":";
                for(auto [v, w]: environment->graph[i])
                    stream  << v << ":" << std::setprecision(3) << w << ":";
                stream << "\n";
            }
            stream << current_state << ':' << current_player_;
            return std::to_string(std::hash<std::string>{}(stream.str()));
        }
        else if(game_->GetParameters().at("representation").string_value() == "minimal")
        {
            stream << current_state << ':' << current_player_;
            return stream.str();
        }
        else
        {
            std::cerr << "Unknown representation: " << game_->GetParameters().at("representation").string_value() << "\n";
            exit(1);
        }
    }

    bool MPGEnvironmentState::IsTerminal() const {
      return num_moves_ >= MaxNumMoves();
    }

    int MPGEnvironmentState::MaxNumMoves() const
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

    std::vector<double> MPGEnvironmentState::Returns() const
    {
        if (!IsTerminal())
            return {0.0, 0.0};
        else
        {

            //SPIEL_CHECK_FLOAT_NEAR(MeanPayoff(M), mean_payoff, 1e-3);
            auto S= Player1Return(mean_payoff);
            return {S, -S};
        }
    }

    std::string MPGEnvironmentState::InformationStateString(Player player) const {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);
      return HistoryString();
    }

    std::string MPGEnvironmentState::ObservationString(Player player) const {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);
      return ToString();
    }


    void MPGEnvironmentState::ObservationTensor(Player player,
                                                absl::Span<float> values) const{
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);

      // Extract `environment` as a rank 3 tensor.
      auto environmentSubSpan= values.subspan(0, values.size() - 1);
      TensorView<3> view(environmentSubSpan, {environment->graph.size(),environment->graph.size(),2}, true);
        for(int u = 0; u < environment->graph.size(); u++) for(auto [v, w]: environment->graph[u])
        {
            view[{u, v, ObservationAxis::kAdjacencyMatrix}] = v;
            view[{u, v, ObservationAxis::kWeightsMatrix}] = WeightFromPerspective(w,player);
        }
        // Add the current state.
        values[values.size() - 1] = current_state;
    }

    void MPGEnvironmentState::UndoAction(Player player, Action move) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, num_distinct_actions_);
        float K= static_cast<float>(num_moves_)/static_cast<float>(num_moves_+1);
        state_history.pop_back();
        current_state = state_history.back();
        mean_payoff = mean_payoff / K - environment->graph[current_state].at(move) / static_cast<float>(num_moves_ + 1);
        current_player_ = player;
        outcome_ = kInvalidPlayer;
        num_moves_ -= 1;
        history_.pop_back();
        --move_number_;
    }

    std::unique_ptr<State> MPGEnvironmentState::Clone() const
    {
      return std::unique_ptr<State>(new MPGEnvironmentState(*this));
    }

    MPGEnvironmentState::MPGEnvironmentState(std::shared_ptr<const Game> game, std::shared_ptr<Environment> environment):State(std::move(game)),
                                                                                                            environment(std::move(environment)),
                                                                                                              current_player_(0),
                                                                                                              num_moves_(0),
                                                                                                              mean_payoff(0)
     {
        current_state=this->environment->starting_state;
        state_history={current_state};
     }

    WeightType MPGEnvironmentState::GetMeanPayoff() const {
        return mean_payoff;
    }

    NodeType MPGEnvironmentState::GetCurrentState() const {
        return current_state;
    }

    std::string MPGMetaGame::ActionToString(Player player,
                                            Action action_id) const
    {
      return absl::StrCat(action_id);
    }

    MPGMetaGame::MPGMetaGame(const GameParameters& params, std::unique_ptr<EnvironmentFactory> environment_factory)
        : Game(kGameType, params) , environment_factory(std::move(environment_factory))
        {}


    std::unique_ptr<State> MPGMetaGame::NewInitialEnvironmentState() const
    {
        auto state= std::make_unique<MPGEnvironmentState>(shared_from_this(),environment_factory->NewEnvironment(*this));
        absl::MutexLock lock(&environment_mutex);
        last_environment = state->environment;
        return state;
    }

    std::unique_ptr<State> MPGMetaGame::NewInitialState() const
    {
        absl::MutexLock lock(&environment_mutex);
        return std::make_unique<MPGEnvironmentState>(shared_from_this(),last_environment);
    }

    Game::TensorShapeSpecs MPGMetaGame::ObservationTensorShapeSpecs() const {
        return Game::TensorShapeSpecs::kNestedList;
    }

    int MPGMetaGame::MaxGameLength() const
    {
        return game_parameters_.at("max_moves").int_value();
    }

    int MPGMetaGame::MaxGraphSize() const {
        return game_parameters_.at("max_size").int_value();
    }

    int MPGMetaGame::NumDistinctActions() const {
        return MaxGraphSize();
    }

    std::shared_ptr<Environment> MPGMetaGame::GetLastEnvironment() const {
        absl::MutexLock lock(&environment_mutex);
        return last_environment;
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

    WeightedGraphType WeightedGraphType::from_string(const std::string &str)
    {
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

    AdjacencyMatrixType WeightedGraphType::adjacency_matrix() const
    {
        AdjacencyMatrixType adjMatrix(size(), std::vector<bool>(size(), false));
        for(int u = 0; u < size(); u++) for(auto [v, w]: (*this)[u])
            adjMatrix[u][v] = true;
        return adjMatrix;
    }

    Environment::Environment(WeightedGraphType graph, NodeType starting_state) : graph(std::move(graph)), starting_state(starting_state)
    {
    }

}  // namespace open_spiel
