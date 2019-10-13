#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::universal_poker {
    const GameType kGameType{
            /*short_name=*/"universal_poker",
            /*long_name=*/"Universal Poker",
                           GameType::Dynamics::kSequential,
                           GameType::ChanceMode::kExplicitStochastic,
                           GameType::Information::kImperfectInformation,
                           GameType::Utility::kZeroSum,
                           GameType::RewardModel::kTerminal,
            /*max_num_players=*/10,
            /*min_num_players=*/2,
            /*provides_information_state=*/true,
            /*provides_information_state_as_normalized_vector=*/true,
            /*provides_observation=*/true,
            /*provides_observation_as_normalized_vector=*/true,
            /*parameter_specification=*/
                           {{"players", GameParameter(kDefaultPlayers)}}};

    std::shared_ptr<const Game> Factory(const GameParameters& params) {
        return std::shared_ptr<const Game>(new UniversalPokerGame(params));
    }

    REGISTER_SPIEL_GAME(kGameType, Factory);

    // namespace universal_poker
    UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game) : State(game) {

    }

    std::string UniversalPokerState::ToString() const {
        return std::__cxx11::string();
    }

    bool UniversalPokerState::IsTerminal() const {
        return false;
    }

    std::string UniversalPokerState::InformationState(Player player) const {
        return State::InformationState(player);
    }

    std::string UniversalPokerState::ActionToString(Player player, Action move) const {
        return std::__cxx11::string();
    }

    Player UniversalPokerState::CurrentPlayer() const {
        return 0;
    }

    std::vector<double> UniversalPokerState::Returns() const {
        return std::vector<double>();
    }

    void UniversalPokerState::InformationStateAsNormalizedVector(Player player, std::vector<double> *values) const {
        State::InformationStateAsNormalizedVector(player, values);
    }

    std::string UniversalPokerState::Observation(Player player) const {
        return State::Observation(player);
    }

    void UniversalPokerState::ObservationAsNormalizedVector(Player player, std::vector<double> *values) const {
        State::ObservationAsNormalizedVector(player, values);
    }

    std::unique_ptr<State> UniversalPokerState::Clone() const {
        return std::unique_ptr<State>();
    }

    std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes() const {
        return State::ChanceOutcomes();
    }


    UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
        :Game(kGameType, params)
    {

    }

    std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
        return std::unique_ptr<State>();
    }

    double UniversalPokerGame::MaxUtility() const {
        return 0;
    }

    double UniversalPokerGame::MinUtility() const {
        return 0;
    }

    std::vector<int> UniversalPokerGame::InformationStateNormalizedVectorShape() const {
        return Game::InformationStateNormalizedVectorShape();
    }

    std::vector<int> UniversalPokerGame::ObservationNormalizedVectorShape() const {
        return Game::ObservationNormalizedVectorShape();
    }

    int UniversalPokerGame::MaxGameLength() const {
        return 0;
    }

    int UniversalPokerGame::MaxChanceOutcomes() const {
        return Game::MaxChanceOutcomes();
    }

    int UniversalPokerGame::NumPlayers() const {
        return 0;
    }
}
