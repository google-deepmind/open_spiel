//
// Created by dj on 10/19/19.
//

#include <assert.h>
#include "betting_tree.h"
#include <sstream>

namespace open_spiel::universal_poker::logic {

    BettingTree::BettingTree(const std::string& gameDef)
            : ACPCGame(gameDef),
              acpcCallAction_(this, ACPCGame::ACPCState::ACPCAction::ACTION_CALL, 0),
              acpcFoldAction_(this, ACPCGame::ACPCState::ACPCAction::ACTION_FOLD, 0) {

    }

    uint32_t BettingTree::GetMaxBettingActions() {
        return IsLimitGame() ? 3 : 4;
    }

    BettingTree::BettingNode::BettingNode(BettingTree &bettingTree)
    : ACPCState(bettingTree), bettingTree_(bettingTree), nodeType_(NODE_TYPE_CHANCE), possibleActions_({ACTION_DEAL}),
      acpcBetPotAction_(&bettingTree, ACPCGame::ACPCState::ACPCAction::ACTION_INVALID, 0),
      acpcAllInAction_(&bettingTree, ACPCGame::ACPCState::ACPCAction::ACTION_INVALID, 0),
      nbBoardCardsDealt_(0), nbHoleCardsDealtPerPlayer_{0,0,0,0,0,0,0,0,0,0}
    {
    }

    BettingTree::BettingNode::NodeType BettingTree::BettingNode::GetNodeType() const {
        return nodeType_;
    }

    void BettingTree::BettingNode::ApplyChoiceAction(uint32_t actionIdx) {
        assert(nodeType_ == NODE_TYPE_CHOICE);
        assert(actionIdx >= 0 && actionIdx < possibleActions_.size());

        ActionType actionType = possibleActions_[actionIdx];
        switch(actionType)
        {
            case ACTION_FOLD:
                assert(IsValidAction(false, bettingTree_.acpcFoldAction_));
                DoAction(bettingTree_.acpcFoldAction_);
                break;
            case ACTION_CHECK_CALL:
                assert(IsValidAction(false, bettingTree_.acpcCallAction_));
                DoAction(bettingTree_.acpcCallAction_);
                break;
            case ACTION_BET_POT:
                assert(IsValidAction(false, acpcBetPotAction_));
                DoAction(acpcBetPotAction_);
                break;
            case ACTION_ALL_IN:
                assert(IsValidAction(false, acpcAllInAction_));
                DoAction(acpcAllInAction_);
                break;
            case ACTION_DEAL:
            default:
                assert(false);
                break;
        }

        _calculateActionsAndNodeType();
    }

    void BettingTree::BettingNode::ApplyDealCards() {
        assert(nodeType_ == NODE_TYPE_CHANCE);

        for(uint8_t p=0; p < bettingTree_.GetNbPlayers(); p++ ){
            if(nbHoleCardsDealtPerPlayer_[p] < bettingTree_.GetNbHoleCardsRequired() ){
                nbHoleCardsDealtPerPlayer_[p]++;
                _calculateActionsAndNodeType();
                return;
            }
        }

        if( nbBoardCardsDealt_ < bettingTree_.GetNbBoardCardsRequired(GetRound())) {
            nbBoardCardsDealt_++;
            _calculateActionsAndNodeType();
            return;
        }

        assert(false);
    }

    void BettingTree::BettingNode::_calculateActionsAndNodeType() {
        possibleActions_.clear();
        if(IsFinished()) {
            if( NumFolded() >= bettingTree_.GetNbPlayers() - 1){
                nodeType_ = NODE_TYPE_TERMINAL_FOLD;
            }
            else {
                nodeType_= NODE_TYPE_TERMINAL_SHOWDOWN;
            }

        }
        else {
            // Check for sth to deal
            for(uint8_t p=0; p < bettingTree_.GetNbPlayers(); p++ ){
                if(nbHoleCardsDealtPerPlayer_[p] < bettingTree_.GetNbHoleCardsRequired() ){
                    nodeType_ = NODE_TYPE_CHANCE;
                    possibleActions_.push_back(ACTION_DEAL);
                    return;
                }
            }
            if( nbBoardCardsDealt_ < bettingTree_.GetNbBoardCardsRequired(GetRound())) {
                nodeType_ = NODE_TYPE_CHANCE;
                possibleActions_.push_back(ACTION_DEAL);
                return;
            }

            //Check for CHOICE Actions
            nodeType_ = NODE_TYPE_CHOICE;
            if( IsValidAction(false, bettingTree_.acpcFoldAction_) ){
                possibleActions_.push_back(ACTION_FOLD);
            }
            if( IsValidAction(false, bettingTree_.acpcCallAction_) ){
                possibleActions_.push_back(ACTION_CHECK_CALL);
            }

            int32_t potSize = 0, minRaise = 0, maxRaise = 0;
            acpcBetPotAction_.SetActionAndSize(ACPCAction::ACTION_INVALID, 0);
            acpcAllInAction_.SetActionAndSize( ACPCAction::ACTION_INVALID, 0);
            if( RaiseIsValid(&minRaise, &maxRaise) ){
                if(bettingTree_.IsLimitGame()){
                    acpcBetPotAction_.SetActionAndSize(ACPCAction::ACTION_RAISE, 0);
                }
                else {
                    potSize = MaxSpend() > minRaise ? MaxSpend() : minRaise;
                    acpcBetPotAction_.SetActionAndSize( ACPCAction::ACTION_RAISE, potSize);
                    if( maxRaise > potSize ) {
                        acpcAllInAction_.SetActionAndSize(ACPCAction::ACTION_RAISE, maxRaise);
                    }
                }

                if( IsValidAction(false, acpcBetPotAction_) ){
                    possibleActions_.push_back(ACTION_BET_POT);
                }
                if( IsValidAction(false, acpcAllInAction_) ){
                    possibleActions_.push_back(ACTION_ALL_IN);
                }
            }
        }
    }

    const std::vector<BettingTree::ActionType> &BettingTree::BettingNode::GetPossibleActions() const {
        return possibleActions_;
    }

    std::string BettingTree::BettingNode::ToString() {
        std::ostringstream buf;
        buf << "STATE START" << std::endl;

        buf << "NodeType: ";
        buf << (nodeType_ == NODE_TYPE_CHANCE ? "NODE_TYPE_CHANCE" : "");
        buf << (nodeType_ == NODE_TYPE_CHOICE ? "NODE_TYPE_CHOICE" : "");
        buf << (nodeType_ == NODE_TYPE_TERMINAL_SHOWDOWN ? "NODE_TYPE_TERMINAL_SHOWDOWN" : "");
        buf << (nodeType_ == NODE_TYPE_TERMINAL_FOLD ? "NODE_TYPE_TERMINAL_FOLD" : "");
        buf << std::endl;

        buf << "PossibleActions (" << possibleActions_.size() << "): [";
        for( auto action : possibleActions_ ){
            buf << (action == ACTION_ALL_IN ? " ACTION_ALL_IN " : "");
            buf << (action == ACTION_BET_POT ? " ACTION_BET_POT " : "");
            buf << (action == ACTION_CHECK_CALL ? " ACTION_CHECK_CALL " : "");
            buf << (action == ACTION_FOLD ? " ACTION_FOLD " : "");
            buf << (action == ACTION_DEAL ? " ACTION_DEAL " : "");
        }
        buf << "]" << std::endl;

        buf << "ACPC State: " << ACPCState::ToString() << std::endl;

        buf << "STATE END" << std::endl;
        return buf.str();
    }
}


