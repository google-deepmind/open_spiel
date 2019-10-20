//
// Created by dj on 10/19/19.
//

#include "game_tree.h"
#include <sstream>

namespace open_spiel::universal_poker::logic {
    GameTree::GameNode::GameNode( logic::GameTree &gameTree)
            : BettingNode(gameTree), deck_(gameTree.NumSuitsDeck(), gameTree.NumRanksDeck()), gameTree_(gameTree),
              actionCount_(GetPossibleActions().size())
    {
        for(uint8_t p=0; p < gameTree_.GetNbPlayers(); p++ ){
            holeCards_.emplace_back();
        }

        if(GetNodeType() == NODE_TYPE_CHANCE){
            actionCount_ = deck_.ToCardArray().size();
        }
    }

    void GameTree::GameNode::ApplyAction(uint32_t actionIdx) {
        if( GetNodeType() == NODE_TYPE_CHANCE ){
            BettingNode::ApplyDealCards();
            uint8_t card = deck_.ToCardArray()[actionIdx];
            deck_.RemoveCard(card);

            //Check where to add this card
            for(uint8_t p=0; p < gameTree_.GetNbPlayers(); p++ ){
                if( holeCards_[p].CountCards() < gameTree_.GetNbHoleCardsRequired() ){
                    holeCards_[p].AddCard(card);
                    break;
                }
            }

            if( boardCards_.CountCards() < gameTree_.GetNbBoardCardsRequired(GetRound())) {
                boardCards_.AddCard(card);
            }
        }
        else {
            BettingNode::ApplyChoiceAction(actionIdx);
        }

        if(GetNodeType() == NODE_TYPE_CHANCE ) {
            actionCount_ = deck_.ToCardArray().size();
        }
        else {
            actionCount_ = GetPossibleActions().size();
        }
    }

    uint32_t GameTree::GameNode::GetActionCount() const {
        return actionCount_;
    }

    std::string GameTree::GameNode::ToString() const {
            std::ostringstream buf;
            buf << BettingNode::ToString();

            for(uint8_t p=0; p < gameTree_.GetNbPlayers(); p++ ){
                buf << "P" << (int)p << "Cards: " << holeCards_[p].ToString() << std::endl;
            }
            buf << "BoardCards " << boardCards_.ToString() << std::endl;


            if( GetNodeType() == NODE_TYPE_CHANCE ) {
                buf << "PossibleCardsToDeal " << deck_.ToString();
            }


            return buf.str();
    }

    GameTree::GameTree(const std::string &gameDef) : BettingTree(gameDef) {}
}
