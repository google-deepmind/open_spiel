
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

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include "open_spiel/games/twixt/twixtboard.h"
#include "open_spiel/games/twixt/twixtcell.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace twixt {

// ANSI colors
const char kAnsiRed[] = "\e[91m";
const char kAnsiBlue[] = "\e[94m";
const char kAnsiDefault[] = "\e[0m";

// helper functions
inline int OppDir(int direction) {
  return (direction + kMaxCompass / 2) % kMaxCompass;
}

inline std::string PositionToString(Position position) {
  return "[" + std::to_string(position.x) + "," + std::to_string(position.y) +
         "]";
}

// table of 8 link descriptors
static const std::vector<LinkDescriptor> kLinkDescriptorTable{
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
      {{-1, 1}, kESE}}}};

// helper class: blockerMap stores set of blocking links for each link
std::unordered_map<Link, std::set<Link>, LinkHashFunction> BlockerMap::map_ =
    {};

const std::set<Link>& BlockerMap::GetBlockers(Link link) {
  return BlockerMap::map_[link];
}

void BlockerMap::PushBlocker(Link link, Link blocked_link) {
  BlockerMap::map_[link].insert(blocked_link);
}

void BlockerMap::DeleteBlocker(Link link, Link blocked_link) {
  BlockerMap::map_[link].erase(blocked_link);
}

void BlockerMap::ClearBlocker() { BlockerMap::map_.clear(); }

Board::Board(int size, bool ansi_color_output) {
  set_size(size);
  set_ansi_color_output(ansi_color_output);

  InitializeCells(true);
  InitializeLegalActions();
}

void Board::InitializeBlockerMap(Position position, int dir,
                                 const LinkDescriptor& ld) {
  Link link = {position, dir};
  for (auto&& entry : ld.blocking_links) {
    Position fromPosition = position + entry.position;
    if (!PositionIsOffBoard(fromPosition)) {
      const LinkDescriptor& oppLd = kLinkDescriptorTable[entry.direction];
      Position toPosition = position + entry.position + oppLd.offsets;
      if (!PositionIsOffBoard(toPosition)) {
        BlockerMap::PushBlocker(link, {fromPosition, entry.direction});
        BlockerMap::PushBlocker(link, {toPosition, OppDir(entry.direction)});
      }
    }
  }
}

void Board::UpdateResult(Player player, Position position) {
  // check for WIN
  bool connected_to_start = GetCell(position).IsLinkedToBorder(player, kStart);
  bool connected_to_end = GetCell(position).IsLinkedToBorder(player, kEnd);
  if (connected_to_start && connected_to_end) {
    // peg is linked to both boarder lines
    set_result(player == kRedPlayer ? kRedWin : kBlueWin);
    return;
  }

  // check if opponent (player to turn next) has any legal moves left
  if (!HasLegalActions(1 - player)) {
    set_result(kDraw);
    return;
  }
}

void Board::InitializeCells(bool init_blocker_map) {
  cell_.resize(size(), std::vector<Cell>(size()));
  BlockerMap::ClearBlocker();

  for (int x = 0; x < size(); x++) {
    for (int y = 0; y < size(); y++) {
      Position position = {x, y};
      Cell& cell = GetCell(position);

      // set color to EMPTY or OFFBOARD
      if (PositionIsOffBoard(position)) {
        cell.set_color(kOffBoard);
      } else {  // regular board
        cell.set_color(kEmpty);
        if (x == 0) {
          cell.SetLinkedToBorder(kBluePlayer, kStart);
        } else if (x == size() - 1) {
          cell.SetLinkedToBorder(kBluePlayer, kEnd);
        } else if (y == 0) {
          cell.SetLinkedToBorder(kRedPlayer, kStart);
        } else if (y == size() - 1) {
          cell.SetLinkedToBorder(kRedPlayer, kEnd);
        }
        InitializeNeighbors(position, cell, init_blocker_map);
      }
    }
  }
}

void Board::InitializeNeighbors(Position position, Cell& cell,
                                bool init_blocker_map) {
  for (int dir = 0; dir < kMaxCompass; dir++) {
    const LinkDescriptor& ld = kLinkDescriptorTable[dir];
    Position target_position = position + ld.offsets;
    if (!PositionIsOffBoard(target_position)) {
      if (init_blocker_map) {
        InitializeBlockerMap(position, dir, ld);
      }
      cell.SetNeighbor(dir, target_position);
    }
  }
}

void Board::InitializeLegalActions() {
  int num_legal_actions_per_player = size() * (size() - 2);

  for (Player p = 0; p < kNumPlayers; p++) {
    legal_actions_[p].resize(num_legal_actions_per_player);
    legal_actions_[p].clear();
  }

  for (int col = 0; col < size(); col++) {
    for (int row = 0; row < size(); row++) {
      Position pos = {col, row};
      Action action = col * size() + row;
      if (PositionIsOffBoard(pos)) {
        continue;
      } else if (PositionIsOnBorder(kRedPlayer, pos)) {
        legal_actions_[kRedPlayer].push_back(action);
      } else if (PositionIsOnBorder(kBluePlayer, pos)) {
        legal_actions_[kBluePlayer].push_back(action);
      } else {
        legal_actions_[kRedPlayer].push_back(action);
        legal_actions_[kBluePlayer].push_back(action);
      }
    }
  }
}

std::string Board::ToString() const {
  std::string s = "";

  // head line
  s.append("     ");
  for (int y = 0; y < size(); y++) {
    std::string letter = "";
    letter += static_cast<int>('a') + y;
    letter += "  ";
    AppendColorString(s, kAnsiRed, letter);
  }
  s.append("\n");

  for (int y = size() - 1; y >= 0; y--) {
    // print "before" row
    s.append("    ");
    for (int x = 0; x < size(); x++) {
      AppendBeforeRow(s, {x, y});
    }
    s.append("\n");

    // print "peg" row
    size() - y < 10 ? s.append("  ") : s.append(" ");
    AppendColorString(s, kAnsiBlue, std::to_string(size() - y) + " ");
    for (int x = 0; x < size(); x++) {
      AppendPegRow(s, {x, y});
    }
    s.append("\n");

    // print "after" row
    s.append("    ");
    for (int x = 0; x < size(); x++) {
      AppendAfterRow(s, {x, y});
    }
    s.append("\n");
  }
  s.append("\n");

  if (swapped_) s.append("[swapped]");

  switch (result_) {
    case kOpen: {
      break;
    }
    case kRedWin: {
      s.append("[x has won]");
      break;
    }
    case kBlueWin: {
      s.append("[o has won]");
      break;
    }
    case kDraw: {
      s.append("[draw]");
      break;
    }
    default: {
      break;
    }
  }

  return s;
}

void Board::AppendLinkChar(std::string& s, Position position, enum Compass dir,
                           std::string linkChar) const {
  if (!PositionIsOffBoard(position) && GetConstCell(position).HasLink(dir)) {
    if (GetConstCell(position).color() == kRedColor) {
      AppendColorString(s, kAnsiRed, linkChar);
    } else if (GetConstCell(position).color() == kBlueColor) {
      AppendColorString(s, kAnsiBlue, linkChar);
    } else {
      s.append(linkChar);
    }
  }
}

void Board::AppendColorString(std::string& s, std::string colorString,
                              std::string appString) const {
  s.append(ansi_color_output() ? colorString : "");  // make it colored
  s.append(appString);
  s.append(ansi_color_output() ? kAnsiDefault : "");  // make it default
}

void Board::AppendPegChar(std::string& s, Position position) const {
  if (GetConstCell(position).color() == kRedColor) {
    // x
    AppendColorString(s, kAnsiRed, "x");
  } else if (GetConstCell(position).color() == kBlueColor) {
    // o
    AppendColorString(s, kAnsiBlue, "o");
  } else if (PositionIsOffBoard(position)) {
    // corner
    s.append(" ");
  } else if (position.x == 0 || position.x == size() - 1) {
    // empty . (blue border line)
    AppendColorString(s, kAnsiBlue, ".");
  } else if (position.y == 0 || position.y == size() - 1) {
    // empty . (red border line)
    AppendColorString(s, kAnsiRed, ".");
  } else {
    // empty (non border line)
    s.append(".");
  }
}

void Board::AppendBeforeRow(std::string& s, Position position) const {
  // -1, +1
  int len = s.length();
  AppendLinkChar(s, position + Position{-1, 0}, kENE, "/");
  AppendLinkChar(s, position + Position{-1, -1}, kNNE, "/");
  AppendLinkChar(s, position + Position{0, 0}, kWNW, "_");
  if (len == s.length()) s.append(" ");

  //  0, +1
  len = s.length();
  AppendLinkChar(s, position, kNNE, "|");
  if (len == s.length()) AppendLinkChar(s, position, kNNW, "|");
  if (len == s.length()) s.append(" ");

  // +1, +1
  len = s.length();
  AppendLinkChar(s, position + Position{+1, 0}, kWNW, "\\");
  AppendLinkChar(s, position + Position{+1, -1}, kNNW, "\\");
  AppendLinkChar(s, position + Position{0, 0}, kENE, "_");
  if (len == s.length()) s.append(" ");
}

void Board::AppendPegRow(std::string& s, Position position) const {
  // -1, 0
  int len = s.length();
  AppendLinkChar(s, position + Position{-1, -1}, kNNE, "|");
  AppendLinkChar(s, position + Position{0, 0}, kWSW, "_");
  if (len == s.length()) s.append(" ");

  //  0,  0
  AppendPegChar(s, position);

  // +1, 0
  len = s.length();
  AppendLinkChar(s, position + Position{+1, -1}, kNNW, "|");
  AppendLinkChar(s, position + Position{0, 0}, kESE, "_");
  if (len == s.length()) s.append(" ");
}

void Board::AppendAfterRow(std::string& s, Position position) const {
  // -1, -1
  int len = s.length();
  AppendLinkChar(s, position + Position{+1, -1}, kWNW, "\\");
  AppendLinkChar(s, position + Position{0, -1}, kNNW, "\\");
  if (len == s.length()) s.append(" ");

  //  0, -1
  len = s.length();
  AppendLinkChar(s, position + Position{-1, -1}, kENE, "_");
  AppendLinkChar(s, position + Position{+1, -1}, kWNW, "_");
  AppendLinkChar(s, position, kSSW, "|");
  if (len == s.length()) AppendLinkChar(s, position, kSSE, "|");
  if (len == s.length()) s.append(" ");

  // -1, -1
  len = s.length();
  AppendLinkChar(s, position + Position{-1, -1}, kENE, "/");
  AppendLinkChar(s, position + Position{0, -1}, kNNE, "/");
  if (len == s.length()) s.append(" ");
}

void Board::UndoFirstMove() {
  Cell& cell = GetCell(move_one());
  cell.set_color(kEmpty);
  InitializeNeighbors(move_one(), cell, false);
  InitializeLegalActions();
}

void Board::ApplyAction(Player player, Action action) {
  Position position = ActionToPosition(action);

  if (move_counter() == 1) {
    // it's the second position
    if (position == move_one()) {
      // blue player swapped
      set_swapped(true);

      // undo the first move: (remove peg and restore legal actions)
      UndoFirstMove();

      // turn position 90° clockwise:
      // [2,3]->[3,5]; [1,4]->[4,6]; [3,2]->[2,4]
      int x = position.y;
      int y = size() - position.x - 1;
      position = {x, y};

    } else {
      // blue player hasn't swapped => regular move
      // remove move one from legal moves
      RemoveLegalAction(kRedPlayer, move_one());
      RemoveLegalAction(kBluePlayer, move_one());
    }
  }

  SetPegAndLinks(player, position);

  if (move_counter() == 0) {
    // do not remove the move from legal actions but store it
    // because second player might want to swap, by choosing the same move
    set_move_one(position);
  } else {
    // otherwise remove move from legal actions
    RemoveLegalAction(kRedPlayer, position);
    RemoveLegalAction(kBluePlayer, position);
  }

  IncMoveCounter();

  // Update the predicted result and update current_player_...
  UpdateResult(player, position);
}

void Board::SetPegAndLinks(Player player, Position position) {
  bool linked_to_neutral = false;
  bool linked_to_start = false;
  bool linked_to_end = false;

  // set peg
  Cell& cell = GetCell(position);
  cell.set_color(player);

  int dir = 0;
  bool newLinks = false;
  // check all neigbors that are empty or have same color)
  for (dir = 0; dir < kMaxCompass; dir++) {
    Position target_position = position + kLinkDescriptorTable[dir].offsets;
    if (!PositionIsOffBoard(target_position)) {
      Cell& target_cell = GetCell(target_position);
      if (target_cell.color() == cell.color()) {
        // check if there are blocking links before setting link
        const std::set<Link>& blockers =
            BlockerMap::GetBlockers(Link{position, dir});
        bool blocked = false;
        for (auto& bl : blockers) {
          if (GetCell(bl.position).HasLink(bl.direction)) {
            blocked = true;
            break;
          }
        }

        if (!blocked) {
          // we set the link, and set the flag that there is at least one new
          // link
          cell.set_link(dir);
          target_cell.set_link(OppDir(dir));

          newLinks = true;

          // check if cell we link to is linked to START border / END border
          if (target_cell.IsLinkedToBorder(player, kStart)) {
            cell.SetLinkedToBorder(player, kStart);
            linked_to_start = true;
          } else if (target_cell.IsLinkedToBorder(player, kEnd)) {
            cell.SetLinkedToBorder(player, kEnd);
            linked_to_end = true;
          } else {
            linked_to_neutral = true;
          }
        } else {
          // we store the fact that these two pegs of the same color cannot be
          // linked this info is used for the ObservationTensor
          cell.SetBlockedNeighbor(dir);
          target_cell.SetBlockedNeighbor(OppDir(dir));
        }
      }  // same color
    }  // is on board
  }  // range of directions

  // check if we need to explore further
  if (newLinks) {
    std::set<Cell*> visited = {};
    if (cell.IsLinkedToBorder(player, kStart) && linked_to_neutral) {
      // case: new cell is linked to START and linked to neutral cells
      // => explore neutral graph and add all its cells to START
      ExploreLocalGraph(player, cell, kStart, visited);
    }
    if (cell.IsLinkedToBorder(player, kEnd) && linked_to_neutral) {
      // case: new cell is linked to END and linked to neutral cells
      // => explore neutral graph and add all its cells to END
      ExploreLocalGraph(player, cell, kEnd, visited);
    }
  }
}

void Board::ExploreLocalGraph(Player player, Cell& cell, enum Border border,
                              std::set<Cell*> visited) {
  visited.insert(&cell);
  for (int dir = 0; dir < kMaxCompass; dir++) {
    if (cell.HasLink(dir)) {
      Cell& target_cell = GetCell(cell.GetNeighbor(dir));
      if ((visited.find(&target_cell) == visited.end()) &&
          !target_cell.IsLinkedToBorder(player, border)) {
        // linked neighbor has not been visited yet
        // => add it and explore
        target_cell.SetLinkedToBorder(player, border);
        ExploreLocalGraph(player, target_cell, border, visited);
      }
    }
  }
}

Position Board::GetTensorPosition(Position position, bool turn) const {
  // we flip x/y and top/bottom for better readability in playthrough output
  if (turn) {
    return {size() - position.x - 1, size() - position.y - 2};
  } else {
    return {size() - position.y - 1, position.x - 1};
  }
}

Position Board::ActionToPosition(Action action) const {
  return {static_cast<int>(action) / size_, static_cast<int>(action) % size_};
}

Action Board::PositionToAction(Position position) const {
  return position.x * size() + position.y;
}

Action Board::StringToAction(std::string s) const {
  Position position;
  position.x = static_cast<int>(s.at(1)) - static_cast<int>('a');
  position.y = size() - (static_cast<int>(s.at(2)) - static_cast<int>('0'));
  return PositionToAction(position);
}

bool Board::PositionIsOnBorder(Player player, Position position) const {
  if (player == kRedPlayer) {
    return ((position.y == 0 || position.y == size() - 1) &&
            (position.x > 0 && position.x < size() - 1));
  } else {
    return ((position.x == 0 || position.x == size() - 1) &&
            (position.y > 0 && position.y < size() - 1));
  }
}

bool Board::PositionIsOffBoard(Position position) const {
  return (position.y < 0 || position.y > size() - 1 || position.x < 0 ||
          position.x > size() - 1 ||
          // corner case
          ((position.x == 0 || position.x == size() - 1) &&
           (position.y == 0 || position.y == size() - 1)));
}

void Board::RemoveLegalAction(Player player, Position position) {
  Action action = PositionToAction(position);
  std::vector<Action>& la = legal_actions_[player];
  std::vector<Action>::iterator it;
  it = find(la.begin(), la.end(), action);
  if (it != la.end()) la.erase(it);
}

}  // namespace twixt
}  // namespace open_spiel
