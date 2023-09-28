
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

#include "open_spiel/games/twixt/twixtboard.h"
#include "open_spiel/games/twixt/twixtcell.h"

namespace open_spiel {
namespace twixt {

// ANSI colors
const char kAnsiRed[] = "\e[91m";
const char kAnsiBlue[] = "\e[94m";
const char kAnsiDefault[] = "\e[0m";

static std::pair<int, int> operator+(const std::pair<int, int> &l,
                                const std::pair<int, int> &r) {
  return {l.first + r.first, l.second + r.second};
}

// helper functions
inline int oppDir(int dir) { return (dir + kMaxCompass / 2) % kMaxCompass; }

inline int oppCand(int cand) { return cand < 16 ? cand <<= 4 : cand >>= 4; }

inline std::string moveToString(Move move) {
  return "[" + std::to_string(move.first) + "," + std::to_string(move.second) +
         "]";
}

// table of 8 link descriptors
static std::vector<LinkDescriptor> kLinkDescriptorTable{
    // NNE
    {{1, 2},  // offset of target peg (2 up, 1 right)
     {        // blocking/blocked links
      {{0, 1}, kENE},
      {{-1, 0}, kENE},

      {{0, 2}, kESE},
      {{0, 1}, kESE},
      {{-1, 2}, kESE},
      {{-1, 1}, kESE},

      {{0, 1}, kSSE},
      {{0, 2}, kSSE},
      {{0, 3}, kSSE}}},
    // ENE
    {{2, 1},
     {{{0, -1}, kNNE},
      {{1, 0}, kNNE},

      {{-1, 1}, kESE},
      {{0, 1}, kESE},
      {{1, 1}, kESE},

      {{0, 1}, kSSE},
      {{0, 2}, kSSE},
      {{1, 1}, kSSE},
      {{1, 2}, kSSE}}},
    // ESE
    {{2, -1},
     {{{0, -1}, kNNE},
      {{1, -1}, kNNE},
      {{0, -2}, kNNE},
      {{1, -2}, kNNE},

      {{-1, -1}, kENE},
      {{0, -1}, kENE},
      {{1, -1}, kENE},

      {{0, 1}, kSSE},
      {{1, 0}, kSSE}}},
    // SSE
    {{1, -2},
     {{{0, -1}, kNNE},
      {{0, -2}, kNNE},
      {{0, -3}, kNNE},

      {{-1, -1}, kENE},
      {{0, -1}, kENE},
      {{-1, -2}, kENE},
      {{0, -2}, kENE},

      {{-1, 0}, kESE},
      {{0, -1}, kESE}}},
    // SSW
    {{-1, -2},
     {{{-1, -1}, kENE},
      {{-2, -2}, kENE},

      {{-2, 0}, kESE},
      {{-1, 0}, kESE},
      {{-2, -1}, kESE},
      {{-1, -1}, kESE},

      {{-1, 1}, kSSE},
      {{-1, 0}, kSSE},
      {{-1, -1}, kSSE}}},
    // WSW
    {{-2, -1},
     {{{-2, -2}, kNNE},
      {{-1, -1}, kNNE},

      {{-3, 0}, kESE},
      {{-2, 0}, kESE},
      {{-1, 0}, kESE},

      {{-2, 1}, kSSE},
      {{-1, 1}, kSSE},
      {{-2, 0}, kSSE},
      {{-1, 0}, kSSE}}},
    // WNW
    {{-2, 1},
     {{{-2, 0}, kNNE},
      {{-1, 0}, kNNE},
      {{-2, -1}, kNNE},
      {{-1, -1}, kNNE},

      {{-3, 0}, kENE},
      {{-2, 0}, kENE},
      {{-1, 0}, kENE},

      {{-2, 2}, kSSE},
      {{-1, 1}, kSSE}}},
    // NNW
    {{-1, 2},
     {{{-1, 1}, kNNE},
      {{-1, 0}, kNNE},
      {{-1, -1}, kNNE},

      {{-2, 1}, kENE},
      {{-1, 1}, kENE},
      {{-2, 0}, kENE},
      {{-1, 0}, kENE},

      {{-2, 2}, kESE},
      {{-1, 1}, kESE}}}
};

Board::Board(int size, bool ansiColorOutput) {
  setSize(size);
  setAnsiColorOutput(ansiColorOutput);

  initializeCells(true);
  initializeLegalActions();
}

void Board::initializeBlockerMap(Move move, int dir, LinkDescriptor *ld) {
  Link link = {move, dir};
  for (auto &&entry : ld->blockingLinks) {
    Move fromMove = move + entry.first;
    if (!moveIsOffBoard(fromMove)) {
      LinkDescriptor *oppLd = &(kLinkDescriptorTable[entry.second]);
      Move toMove = move + entry.first + oppLd->offsets;
      if (!moveIsOffBoard(toMove)) {
        pushBlocker(link, {fromMove, entry.second});
        pushBlocker(link, {toMove, oppDir(entry.second)});
      }
    }
  }
}

void Board::updateResult(Player player, Move move) {
  // check for WIN
  bool connectedToStart = getCell(move)->isLinkedToBorder(player, kStart);
  bool connectedToEnd = getCell(move)->isLinkedToBorder(player, kEnd);
  if (connectedToStart && connectedToEnd) {
    // peg is linked to both boarder lines
    setResult(player == kRedPlayer ? kRedWin : kBlueWin);
    return;
  }

  // check if we are early in the game...
  if (getMoveCounter() < getSize() - 1) {
    // e.g. less than 5 moves played on a 6x6 board
    // => no win or draw possible, no need to update
    return;
  }

  // check if opponent (player to turn next) has any legal moves left
  if (!hasLegalActions(1 - player)) {
    setResult(kDraw);
    return;
  }
}

void Board::initializeCells(bool initBlockerMap) {
  mCell.resize(getSize(), std::vector<Cell>(getSize()));
  clearBlocker();

  for (int x = 0; x < getSize(); x++) {
    for (int y = 0; y < getSize(); y++) {
      Move move = {x, y};
      Cell *pCell = getCell(move);

      // set color to EMPTY or OFFBOARD
      if (moveIsOffBoard(move)) {
        pCell->setColor(kOffBoard);
      } else {  // regular board
        pCell->setColor(kEmpty);
        if (x == 0) {
          pCell->setLinkedToBorder(kBluePlayer, kStart);
        } else if (x == getSize() - 1) {
          pCell->setLinkedToBorder(kBluePlayer, kEnd);
        } else if (y == 0) {
          pCell->setLinkedToBorder(kRedPlayer, kStart);
        } else if (y == getSize() - 1) {
          pCell->setLinkedToBorder(kRedPlayer, kEnd);
        }

        initializeCandidates(move, pCell, initBlockerMap);
      }
    }
  }
}

void Board::initializeCandidates(Move move, Cell *pCell, bool initBlockerMap) {
  for (int dir = 0; dir < kMaxCompass; dir++) {
    LinkDescriptor *ld = &(kLinkDescriptorTable[dir]);
    Move targetMove = move + ld->offsets;
    if (!moveIsOffBoard(targetMove)) {
      if (initBlockerMap) {
        initializeBlockerMap(move, dir, ld);
      }
      pCell->setNeighbor(dir, targetMove);
      Cell *pTargetCell = getCell(targetMove);
      if (!(moveIsOnBorder(kRedPlayer, move) &&
            moveIsOnBorder(kBluePlayer, targetMove)) &&
          !(moveIsOnBorder(kBluePlayer, move) &&
            moveIsOnBorder(kRedPlayer, targetMove))) {
        pCell->setCandidate(kRedPlayer, dir);
        pCell->setCandidate(kBluePlayer, dir);
      }
    }
  }
}

void Board::initializeLegalActions() {
  int numDistinctLegalActions = getSize() * (getSize() - 2);

  mLegalActions[kRedPlayer].resize(numDistinctLegalActions);
  mLegalActions[kBluePlayer].resize(numDistinctLegalActions);

  for (int player = kRedPlayer; player < kNumPlayers; player++) {
    std::vector<Action> *la = &mLegalActions[player];
    la->clear();
    la->reserve(numDistinctLegalActions);

    for (Action a = 0; a < numDistinctLegalActions; a++) {
      la->push_back(a);
    }
  }
}

std::string Board::toString() const {
  std::string s = "";

  // head line
  s.append("     ");
  for (int y = 0; y < getSize(); y++) {
    std::string letter = "";
    letter += static_cast<int>('a') + y;
    letter += "  ";
    appendColorString(&s, kAnsiRed, letter);
  }
  s.append("\n");

  for (int y = getSize() - 1; y >= 0; y--) {
    // print "before" row
    s.append("    ");
    for (int x = 0; x < getSize(); x++) {
      appendBeforeRow(&s, {x, y});
    }
    s.append("\n");

    // print "peg" row
    getSize() - y < 10 ? s.append("  ") : s.append(" ");
    appendColorString(&s, kAnsiBlue, std::to_string(getSize() - y) + " ");
    for (int x = 0; x < getSize(); x++) {
      appendPegRow(&s, {x, y});
    }
    s.append("\n");

    // print "after" row
    s.append("    ");
    for (int x = 0; x < getSize(); x++) {
      appendAfterRow(&s, {x, y});
    }
    s.append("\n");
  }
  s.append("\n");

  if (mSwapped)
    s.append("[swapped]");

  switch (mResult) {
  case kOpen:
    break;
  case kRedWin:
    s.append("[x has won]");
    break;
  case kBlueWin:
    s.append("[o has won]");
    break;
  case kDraw:
    s.append("[draw]");
  default:
    break;
  }

  return s;
}

void Board::appendLinkChar(std::string *s, Move move, enum Compass dir,
                           std::string linkChar) const {
  if (!moveIsOffBoard(move) && getConstCell(move)->hasLink(dir)) {
    if (getConstCell(move)->getColor() == kRedColor) {
      appendColorString(s, kAnsiRed, linkChar);
    } else if (getConstCell(move)->getColor() == kBlueColor) {
      appendColorString(s, kAnsiBlue, linkChar);
    } else {
      s->append(linkChar);
    }
  }
}

void Board::appendColorString(std::string *s, std::string colorString,
                              std::string appString) const {
  s->append(getAnsiColorOutput() ? colorString : "");  // make it colored
  s->append(appString);
  s->append(getAnsiColorOutput() ? kAnsiDefault : "");  // make it default
}

void Board::appendPegChar(std::string *s, Move move) const {
  if (getConstCell(move)->getColor() == kRedColor) {
    // x
    appendColorString(s, kAnsiRed, "x");
  } else if (getConstCell(move)->getColor() == kBlueColor) {
    // o
    appendColorString(s, kAnsiBlue, "o");
  } else if (moveIsOffBoard(move)) {
    // corner
    s->append(" ");
  } else if (move.first == 0 || move.first == getSize() - 1) {
    // empty . (blue border line)
    appendColorString(s, kAnsiBlue, ".");
  } else if (move.second == 0 || move.second == getSize() - 1) {
    // empty . (red border line)
    appendColorString(s, kAnsiRed, ".");
  } else {
    // empty (non border line)
    s->append(".");
  }
}

void Board::appendBeforeRow(std::string *s, Move move) const {
  // -1, +1
  int len = s->length();
  appendLinkChar(s, move + (Move){-1, 0}, kENE, "/");
  appendLinkChar(s, move + (Move){-1, -1}, kNNE, "/");
  appendLinkChar(s, move + (Move){0, 0}, kWNW, "_");
  if (len == s->length())
    s->append(" ");

  //  0, +1
  len = s->length();
  appendLinkChar(s, move, kNNE, "|");
  if (len == s->length())
    appendLinkChar(s, move, kNNW, "|");
  if (len == s->length())
    s->append(" ");

  // +1, +1
  len = s->length();
  appendLinkChar(s, move + (Move){+1, 0}, kWNW, "\\");
  appendLinkChar(s, move + (Move){+1, -1}, kNNW, "\\");
  appendLinkChar(s, move + (Move){0, 0}, kENE, "_");
  if (len == s->length())
    s->append(" ");
}

void Board::appendPegRow(std::string *s, Move move) const {
  // -1, 0
  int len = s->length();
  appendLinkChar(s, move + (Move){-1, -1}, kNNE, "|");
  appendLinkChar(s, move + (Move){0, 0}, kWSW, "_");
  if (len == s->length())
    s->append(" ");

  //  0,  0
  appendPegChar(s, move);

  // +1, 0
  len = s->length();
  appendLinkChar(s, move + (Move){+1, -1}, kNNW, "|");
  appendLinkChar(s, move + (Move){0, 0}, kESE, "_");
  if (len == s->length())
    s->append(" ");
}

void Board::appendAfterRow(std::string *s, Move move) const {
  // -1, -1
  int len = s->length();
  appendLinkChar(s, move + (Move){+1, -1}, kWNW, "\\");
  appendLinkChar(s, move + (Move){0, -1}, kNNW, "\\");
  if (len == s->length())
    s->append(" ");

  //  0, -1
  len = s->length();
  appendLinkChar(s, move + (Move){-1, -1}, kENE, "_");
  appendLinkChar(s, move + (Move){+1, -1}, kWNW, "_");
  appendLinkChar(s, move, kSSW, "|");
  if (len == s->length())
    appendLinkChar(s, move, kSSE, "|");
  if (len == s->length())
    s->append(" ");

  // -1, -1
  len = s->length();
  appendLinkChar(s, move + (Move){-1, -1}, kENE, "/");
  appendLinkChar(s, move + (Move){0, -1}, kNNE, "/");
  if (len == s->length())
    s->append(" ");
}

void Board::undoFirstMove() {
  Cell *pCell = getCell(getMoveOne());
  pCell->setColor(kEmpty);
  // initialize Candidates but not static blockerMap
  initializeCandidates(getMoveOne(), pCell, false);
  initializeLegalActions();
}

void Board::applyAction(Player player, Action action) {
  Move move = actionToMove(player, action);

  if (getMoveCounter() == 1) {
    // it's the second move
    if (move == getMoveOne()) {
      // blue player swapped
      setSwapped(true);

      // undo the first move (peg and legal actions)
      undoFirstMove();

      // turn move 90° clockwise: [3,2] -> [5,3]
      int col = getSize() - move.second - 1;
      int row = move.first;
      move = {col, row};

    } else {
      // blue player hasn't swapped => regular move
      // remove move one from legal moves
      removeLegalAction(kRedPlayer, getMoveOne());
      removeLegalAction(kBluePlayer, getMoveOne());
    }
  }

  setPegAndLinks(player, move);

  if (getMoveCounter() == 0) {
    // do not remove the move from legal actions but store it
    // because second player might want to swap, by choosing the same move
    setMoveOne(move);
  } else {
    // otherwise remove move from legal actions
    removeLegalAction(kRedPlayer, move);
    removeLegalAction(kBluePlayer, move);
  }

  incMoveCounter();

  // Update the predicted result and update mCurrentPlayer...
  updateResult(player, move);
}

void Board::setPegAndLinks(Player player, Move move) {
  bool linkedToNeutral = false;
  bool linkedToStart = false;
  bool linkedToEnd = false;

  // set peg
  Cell *pCell = getCell(move);
  pCell->setColor(player);

  int dir = 0;
  bool newLinks = false;
  // check all candidates (neigbors that are empty or have same color)
  for (int cand = 1, dir = 0; cand <= pCell->getCandidates(player);
       cand <<= 1, dir++) {
    if (pCell->isCandidate(player, cand)) {
      Move n = pCell->getNeighbor(dir);

      Cell *pTargetCell = getCell(pCell->getNeighbor(dir));
      if (pTargetCell->getColor() == kEmpty) {
        // pCell is not a candidate for pTargetCell anymore
        // (from opponent's perspective)
        pTargetCell->deleteCandidate(1 - player, oppCand(cand));
      } else {
        // check if there are blocking links before setting link
        std::set<Link> *blockers = getBlockers((Link){move, dir});
        bool blocked = false;
        for (auto &&bl : *blockers) {
          if (getCell(bl.first)->hasLink(bl.second)) {
            blocked = true;
            break;
          }
        }

        if (!blocked) {
          // we set the link, and set the flag that there is at least one new
          // link
          pCell->setLink(dir);
          pTargetCell->setLink(oppDir(dir));

          newLinks = true;

          // check if cell we link to is linked to START border / END border
          if (pTargetCell->isLinkedToBorder(player, kStart)) {
            pCell->setLinkedToBorder(player, kStart);
            linkedToStart = true;
          } else if (pTargetCell->isLinkedToBorder(player, kEnd)) {
            pCell->setLinkedToBorder(player, kEnd);
            linkedToEnd = true;
          } else {
            linkedToNeutral = true;
          }
        } else {
          // we store the fact that these two pegs of the same color cannot be
          // linked this info is used for the ObservationTensor
          pCell->setBlockedNeighbor(cand);
          pTargetCell->setBlockedNeighbor(oppCand(cand));
        }
      }  // is not empty
    }  // is candidate
  }  // candidate range

  // check if we need to explore further
  if (newLinks) {
    if (pCell->isLinkedToBorder(player, kStart) && linkedToNeutral) {
      // case: new cell is linked to START and linked to neutral cells
      // => explore neutral graph and add all its cells to START
      exploreLocalGraph(player, pCell, kStart);
    }
    if (pCell->isLinkedToBorder(player, kEnd) && linkedToNeutral) {
      // case: new cell is linked to END and linked to neutral cells
      // => explore neutral graph and add all its cells to END
      exploreLocalGraph(player, pCell, kEnd);
    }
  }
}

void Board::exploreLocalGraph(Player player, Cell *pCell, enum Border border) {
  int dir = 0;
  for (int link = 1, dir = 0; link <= pCell->getLinks(); link <<= 1, dir++) {
    if (pCell->isLinked(link)) {
      Cell *pTargetCell = getCell(pCell->getNeighbor(dir));
      if (!pTargetCell->isLinkedToBorder(player, border)) {
        // linked neighbor is NOT yet member of PegSet
        // => add it and explore
        pTargetCell->setLinkedToBorder(player, border);
        exploreLocalGraph(player, pTargetCell, border);
      }
    }
  }
}

Move Board::getTensorMove(Move move, int turn) const {
  switch (turn) {
  case 0:
    return {move.first - 1, move.second};
    break;
  case 90:
    return {getSize() - move.second - 2, move.first};
    break;
  case 180:
    return {getSize() - move.first - 2, getSize() - move.second - 1};
    break;
  default:
    SpielFatalError("invalid turn: " + std::to_string(turn) +
                    "; should be 0, 90, 180");
  }
}

Move Board::actionToMove(open_spiel::Player player, Action action) const {
  Move move;
  if (player == kRedPlayer) {
    move.first = action / mSize + 1;  // col
    move.second = action % mSize;     // row
  } else {
    move.first = action % mSize;                 // col
    move.second = mSize - (action / mSize) - 2;  // row
  }
  return move;
}

Action Board::moveToAction(Player player, Move move) const {
  Action action;
  if (player == kRedPlayer) {
    action = (move.first - 1) * mSize + move.second;
  } else {
    action = (mSize - move.second - 2) * mSize + move.first;
  }
  return action;
}

Action Board::stringToAction(std::string s) const {
  Player player = (s.at(0) == 'x') ? kRedPlayer : kBluePlayer;
  Move move;
  move.first = static_cast<int>(s.at(1)) - static_cast<int>('a');
  move.second = getSize() - (static_cast<int>(s.at(2)) - static_cast<int>('0'));
  return moveToAction(player, move);
}

bool Board::moveIsOnBorder(Player player, Move move) const {
  if (player == kRedPlayer) {
    return ((move.second == 0 || move.second == getSize() - 1) &&
            (move.first > 0 && move.first < getSize() - 1));
  } else {
    return ((move.first == 0 || move.first == getSize() - 1) &&
            (move.second > 0 && move.second < getSize() - 1));
  }
}

bool Board::moveIsOffBoard(Move move) const {
  return (move.second < 0 || move.second > getSize() - 1 || move.first < 0 ||
          move.first > getSize() - 1 ||
          // corner case
          ((move.first == 0 || move.first == getSize() - 1) &&
           (move.second == 0 || move.second == getSize() - 1)));
}

void Board::removeLegalAction(Player player, Move move) {
  Action action = moveToAction(player, move);
  std::vector<Action> *la = &mLegalActions[player];
  std::vector<Action>::iterator it;
  it = find(la->begin(), la->end(), action);
  if (it != la->end())
    la->erase(it);
}

}  // namespace twixt
}  // namespace open_spiel
