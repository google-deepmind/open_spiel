//
// Created by dj on 10/19/19.
//

#ifndef OPEN_SPIEL_GAME_TREE_H
#define OPEN_SPIEL_GAME_TREE_H

#include "betting_tree.h"
#include "card_set.h"

namespace open_spiel::universal_poker::logic {
    class GameTree : public BettingTree {
    public:
        class GameNode : public BettingNode {
        public:
            explicit GameNode(GameTree &gameTree);

            uint32_t GetActionCount() const;
            void ApplyAction(uint32_t actionIdx);
            std::string ToString() const;


        private:
            GameTree& gameTree_;
            CardSet deck_;
            uint32_t actionCount_ ;
            std::vector<CardSet> holeCards_;
            CardSet boardCards_;
        };

    public:
        GameTree(const std::string &gameDef);

    };
}

#endif //OPEN_SPIEL_GAME_TREE_H
