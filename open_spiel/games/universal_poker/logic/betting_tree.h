//
// Created by dj on 10/19/19.
//

#ifndef OPEN_SPIEL_BETTING_TREE_H
#define OPEN_SPIEL_BETTING_TREE_H

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include <cstdint>
#include <vector>

namespace open_spiel::universal_poker::logic {
    constexpr uint8_t MAX_PLAYERS = 10;

    class BettingTree : public acpc_cpp::ACPCGame {
    public:
        enum ActionType {
            ACTION_DEAL = 1, ACTION_FOLD = 2, ACTION_CHECK_CALL = 4, ACTION_BET_POT = 8, ACTION_ALL_IN = 16
        };
        static constexpr ActionType ALL_ACTIONS[5] = {ACTION_DEAL, ACTION_FOLD, ACTION_CHECK_CALL, ACTION_BET_POT, ACTION_ALL_IN};

    public:
        class BettingNode : public acpc_cpp::ACPCGame::ACPCState {
            friend BettingTree;

        public:
            enum NodeType {
                NODE_TYPE_CHANCE, NODE_TYPE_CHOICE, NODE_TYPE_TERMINAL_FOLD, NODE_TYPE_TERMINAL_SHOWDOWN
            };

            BettingNode(BettingTree* bettingTree);

        public:
            NodeType GetNodeType() const;

            const uint32_t &GetPossibleActionsMask() const;
            const int GetPossibleActionCount() const;

            void ApplyChoiceAction(ActionType actionType);

            virtual void ApplyDealCards();

            std::string ToString() const;

            int GetDepth();

            std::string GetActionSequence() const;

            bool IsFinished() const;

        private:
            const BettingTree* bettingTree_;
            NodeType nodeType_;
            uint32_t possibleActions_;
            int32_t potSize_;
            int32_t allInSize_;
            std::string actionSequence_;

            uint8_t nbHoleCardsDealtPerPlayer_[MAX_PLAYERS];
            uint8_t nbBoardCardsDealt_;

            void _CalculateActionsAndNodeType();
        };


    public:
        BettingTree(const std::string &gameDef);
        uint32_t GetMaxBettingActions() const ;
    };
}


#endif //OPEN_SPIEL_BETTING_TREE_H
