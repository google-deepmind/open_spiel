#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <utility>
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

#include "open_spiel/games/universal_poker/PokerGame/PokerGame.h"

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
                           {{"gameDesc", GameParameter(gameDesc)}}};

    std::shared_ptr<const Game> Factory(const GameParameters &params) {
        return std::shared_ptr<const Game>(new UniversalPokerGame(params));
    }

    REGISTER_SPIEL_GAME(kGameType, Factory);

    // namespace universal_poker
    UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game)
            : State(game),
            pokerGame_((UniversalPokerGame*)game.get()),
            internalState_( pokerGame_->getPokerGame().getGame(), nullptr, PokerGameState::GameAction()) {
    }

    std::string UniversalPokerState::ToString() const {
        return internalState_.getName();
    }

    bool UniversalPokerState::IsTerminal() const {
        return internalState_.getType() == BettingNode::TERMINAL_FOLD_NODE ||
        internalState_.getType() == BettingNode::TERMINAL_SHOWDOWN_NODE;
    }

    std::string UniversalPokerState::InformationState(Player player) const {
        return "";
    }

    std::string UniversalPokerState::ActionToString(Player player, Action move) const {
        return std::__cxx11::string();
    }

    Player UniversalPokerState::CurrentPlayer() const {
        if( IsTerminal() ){
            return Player(kTerminalPlayerId);
        }

        return Player(internalState_.getPlayer() == PLAYER_DEALER ? kChancePlayerId : internalState_.getPlayer());
    }

    std::vector<double> UniversalPokerState::Returns() const {
        if (!IsTerminal()) {
            return std::vector<double>(num_players_, 0.0);
        }

        std::vector<double> returns(num_players_);
        for (auto player = Player{0}; player < num_players_; ++player) {
            // Money vs money at start.
            returns[player] = internalState_.getTotalReward(player);
        }

        return returns;
    }

    void UniversalPokerState::InformationStateAsNormalizedVector(Player player, std::vector<double> *values) const {



    }

    std::string UniversalPokerState::Observation(Player player) const {
        return "";
    }

    void UniversalPokerState::ObservationAsNormalizedVector(Player player, std::vector<double> *values) const {
    }

    std::unique_ptr<State> UniversalPokerState::Clone() const {
        return std::unique_ptr<State>(new UniversalPokerState(*this));
    }

    std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes() const {
        SPIEL_CHECK_TRUE(IsChanceNode());
        std::vector<std::pair<Action, double>> outcomes;
        auto internalActions = internalState_.getActionsAllowed();

        const double p = 1.0 / (double)internalActions.size();
        for (uint64_t card = 0; card < internalActions.size(); ++card) {
            outcomes.push_back({card, p});
        }
        return outcomes;
    }

    std::vector<Action> UniversalPokerState::LegalActions() const {
        std::vector<Action> actions;

        for( uint64_t idx = 0; idx < internalState_.getActionsAllowed().size(); idx++){
            actions.push_back(idx);
        }

        return actions;
    }

    void UniversalPokerState::DoApplyAction(Action action_id) {
        internalState_ = pokerGame_->getPokerGame().updateState(internalState_, action_id);

    }


    /**
     * Universal Poker Game Constructor
     * @param params
     */
    UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
            : Game(kGameType, params),
              gameDesc_(ParameterValue<std::string>("gameDesc")),
              pokerGame_(PokerGame::createFromGamedef(gameDesc_)) {
        maxGameLength_ = pokerGame_.getGameLength();
        maxBoardCardCombinations_ = numBoardCardCombinations_(0);

        for (int r = 1; r < numRounds_(); r++) {
            int combos = numBoardCardCombinations_(r);
            if (combos > maxBoardCardCombinations_) {
                maxBoardCardCombinations_ = combos;
            }
        }
    }

    std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
        return std::unique_ptr<State>(new UniversalPokerState(shared_from_this()));
    }

    std::vector<int> UniversalPokerGame::InformationStateNormalizedVectorShape() const {
        // One-hot encoding for player number (who is to play).
        // 2 slots of cards (total_cards_ bits each): private card, public card
        // Followed by maximum game length * 2 bits each (call / raise)
        return {0}; //{(numPlayers_()) + ((numBoardCards_(numRounds_() - 1) + numHoleCards_(0))) + (MaxGameLength() * 2)};
    }

    std::vector<int> UniversalPokerGame::ObservationNormalizedVectorShape() const {
        // One-hot encoding for player number (who is to play).
        // 2 slots of cards (total_cards_ bits each): private card, public card
        // Followed by the contribution of each player to the pot
        return {0}; //{(numPlayers_()) + (numBoardCards_(numRounds_() - 1) + numHoleCards_(0)) + (numPlayers_())};
    }

    double UniversalPokerGame::MaxUtility() const {
        // In poker, the utility is defined as the money a player has at the end of
        // the game minus then money the player had before starting the game.
        // The most a player can win *per opponent* is the most each player can put
        // into the pot, which is the raise amounts on each round times the maximum
        // number raises, plus the original chip they put in to play.
        return 100.0;
    }

    double UniversalPokerGame::MinUtility() const {
        // In poker, the utility is defined as the money a player has at the end of
        // the game minus then money the player had before starting the game.
        // The most any single player can lose is the maximum number of raises per
        // round times the amounts of each of the raises, plus the original chip they
        // put in to play.
        return -100.0;
    }


    int UniversalPokerGame::numBoardCardCombinations_(const int r) const {
        assert(r < numRounds_());
        int deckSize = deckSize_();

        for (int i = 0; i < r; i++) {
            deckSize -= numBoardCards_(r);
        }

        return choose_(deckSize, numBoardCards_(r));
    }

    int UniversalPokerGame::choose_(int n, int k) {
        if (k == 0) {
            return 1;
        }
        return (n * choose_(n - 1, k - 1)) / k;
    }

    int UniversalPokerGame::MaxChanceOutcomes() const {
        return maxBoardCardCombinations_;
    }

    int UniversalPokerGame::NumPlayers() const {
        return numPlayers_();
    }

    const PokerGame &UniversalPokerGame::getPokerGame() const {
        return pokerGame_;
    }
}
