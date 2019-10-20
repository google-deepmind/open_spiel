#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <open_spiel/games/universal_poker/logic/game_tree.h>

#include "open_spiel/spiel.h"

namespace open_spiel {
    namespace universal_poker {

        inline const std::string gameDesc(
                "GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = 3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

        class UniversalPokerGame;

        enum ActionType {
            kFold = 0, kCall = 1, kRaise = 2
        };

        class UniversalPokerState : public State {
        public:
            explicit UniversalPokerState(std::shared_ptr<const Game> game);

            Player CurrentPlayer() const override;

            std::string ActionToString(Player player, Action move) const override;

            std::string ToString() const override;

            bool IsTerminal() const override;

            std::vector<double> Returns() const override;

            std::string InformationState(Player player) const override;

            std::string Observation(Player player) const override;

            void InformationStateAsNormalizedVector(
                    Player player, std::vector<double> *values) const override;

            void ObservationAsNormalizedVector(
                    Player player, std::vector<double> *values) const override;

            std::unique_ptr<State> Clone() const override;

            // The probability of taking each possible action in a particular info state.
            std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

            std::vector<Action> LegalActions() const override;

        protected:
            void DoApplyAction(Action action_id) override;

        private:
            logic::GameTree::GameNode gameNode_;

        };

        class UniversalPokerGame : public Game {
        public:
            explicit UniversalPokerGame(const GameParameters &params);

            int NumDistinctActions() const override;

            std::unique_ptr<State> NewInitialState() const override;

            int NumPlayers() const override;

            double MinUtility() const override;

            double MaxUtility() const override;

            int MaxChanceOutcomes() const override;

            double UtilitySum() const override;

            std::shared_ptr<const Game> Clone() const override;

            std::vector<int> InformationStateNormalizedVectorShape() const override;

            std::vector<int> ObservationNormalizedVectorShape() const override;

            int MaxGameLength() const override;

        private:
            std::string gameDesc_;
            logic::GameTree gameTree_;
        public:
            logic::GameTree &GetGameTree();
        };

    }  // namespace universal_poker
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
