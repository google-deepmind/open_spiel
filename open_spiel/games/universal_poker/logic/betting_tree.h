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
        enum ActionType{ ACTION_DEAL, ACTION_FOLD, ACTION_CHECK_CALL, ACTION_BET_POT, ACTION_ALL_IN};

    public:
        class BettingNode : public acpc_cpp::ACPCGame::ACPCState {
        friend BettingTree;

        public:
            enum NodeType{ NODE_TYPE_CHANCE,  NODE_TYPE_CHOICE,  NODE_TYPE_TERMINAL_FOLD,  NODE_TYPE_TERMINAL_SHOWDOWN};
            explicit BettingNode(BettingTree& bettingTree);

        public:
            NodeType GetNodeType() const;
            const std::vector<ActionType> &GetPossibleActions() const;
            void ApplyChoiceAction(uint32_t actionIdx);
            void ApplyDealCards();
            std::string ToString();

        private:
            const BettingTree& bettingTree_;
            NodeType nodeType_;
            std::vector<ActionType> possibleActions_ ;
            ACPCGame::ACPCState::ACPCAction acpcBetPotAction_;
            ACPCGame::ACPCState::ACPCAction acpcAllInAction_;

            uint8_t nbHoleCardsDealtPerPlayer_[MAX_PLAYERS];
            uint8_t nbBoardCardsDealt_;

            void _calculateActionsAndNodeType();
        };



    public:
        BettingTree(const std::string &gameDef);
        uint32_t GetMaxBettingActions();

    private:
        const ACPCGame::ACPCState::ACPCAction acpcCallAction_;
        const ACPCGame::ACPCState::ACPCAction acpcFoldAction_;
    };




}


#endif //OPEN_SPIEL_BETTING_TREE_H
