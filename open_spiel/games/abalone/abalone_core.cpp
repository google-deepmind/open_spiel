// Copyright 2023 DeepMind Technologies Limited
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

#include "open_spiel/games/abalone/abalone_core.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>

namespace abalone_core {

std::string StateToString(CellState state) {
  switch (state) {
  case CellState::Invalid: return " ";
  case CellState::Empty:   return ".";
  case CellState::Player0: return "1";
  case CellState::Player1: return "2";
  default:
    return "Unknown state.";
  }
}

std::string DirectionToString(Direction _dir) {
  switch (_dir) {
  case Direction::RIGHT:      return "RIGHT";
  case Direction::UP_RIGHT:   return "UP_RIGHT";
  case Direction::UP_LEFT:    return "UP_LEFT";
  case Direction::LEFT:       return "LEFT";
  case Direction::DOWN_LEFT:  return "DOWN_LEFT";
  case Direction::DOWN_RIGHT: return "DOWN_RIGHT";
  default:
    return "INVALID";
  }
}

// columns:  1        2        3        4        5        6        7        8
const CellState VALID_BOARD[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},    // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},      // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // i
};

const CellState ABALONE_INIT_CLASSIC[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},    // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},      // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},      // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1},  // i
};

const CellState ABALONE_INIT_BELGIAN_DAISY[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},    // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},      // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},    // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player0},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player0},    // i
};


void core_state::Reset(
    const CellState _init_pattern[kNumRows][kNumCols]) {
  outcome_ = CellState::Invalid;
  // current_player_ = CellState::Player0;
  turn_ = 0;

  for (auto j = 0; j < kNumRows; ++j)
    for (auto i = 0; i < kNumCols; ++i)
      board_[j][i] = _init_pattern[j][i];
}

std::tuple<bool, CellState> core_state::Eval(
    int _marbles_to_win, int _game_length, bool _marble_advantage) const {
  if (outcome_ != CellState::Invalid) {
    return std::make_tuple(true, outcome_);
  }

  int ballCount[2] = { 0, 0 };
  for (int line = 0; line < kNumRows; ++line) {
    for (int column = 0; column < kNumCols; ++column) {
      auto slot = board_[line][column];
      if (slot == CellState::Player0) {
        ballCount[0]++;
      } else if (slot == CellState::Player1) {
        ballCount[1]++;
      }
    }
  }
  auto min_balls = std::min(ballCount[0], ballCount[1]);
  if (min_balls <= kMarblesPerPlayer - _marbles_to_win) {
    if (ballCount[1] == min_balls)
      return std::make_tuple(true, CellState::Player0);
    if (ballCount[0] == min_balls)
      return std::make_tuple(true, CellState::Player1);
  }

  if (_game_length > 0 && turn_ >= _game_length) {
    if (_marble_advantage) {
      if (ballCount[0] == ballCount[1])
        return std::make_tuple(true, CellState::Empty);
      else if (ballCount[1] == min_balls)
        return std::make_tuple(true, CellState::Player0);
      else
        return std::make_tuple(true, CellState::Player1);
    } else {
      return std::make_tuple(true, CellState::Empty);
    }
  }

  return std::make_tuple(false, CellState::Invalid);
}

std::ostream& operator<<(std::ostream& _os, core_state const& _arg) {
  // os << std::setfill('0');
  // os << std::setw(2);
  _os << "m_board = " << std::endl;
  auto display_line = [&](std::string prefix, int row, int start, int end,
                          std::string postfix) {
    _os << prefix;
    for (auto i = start; i < end; ++i) {
      _os << "   " << StateToString(_arg.board_[row][i]);
    }
    _os << postfix;
    _os << std::endl;
  };
  display_line("<i>        ", 8, 4, 9, "");
  display_line("<h>      ", 7, 3, 9, "");
  display_line("<g>    ", 6, 2, 9, "");
  display_line("<f>  ", 5, 1, 9, "");
  display_line("<e>", 4, 0, 9, "");
  display_line("<d>  ", 3, 0, 8, "  <9>");
  display_line("<c>    ", 2, 0, 7, "  <8>");
  display_line("<b>      ", 1, 0, 6, "  <7>");
  display_line("<a>        ", 0, 0, 5, "  <6>");

  _os << "               <1> <2> <3> <4> <5>" << std::endl;

  _os << "current_player_ = " << _arg.ToPlay() << std::endl;
  _os << "outcome_ = " << _arg.outcome_ << std::endl;
  return _os;
}

bool Move::_IsValidSlide(const core_state& _state) const {
  auto player = _state.board_[m_start.m_row][m_start.m_column];
  if (player == CellState::Empty || player == CellState::Invalid)
    return false;
  auto rSize = std::abs(m_end.m_row - m_start.m_row) + 1;
  auto cSize = std::abs(m_end.m_column - m_start.m_column) + 1;
  if (rSize > 3)
    return false;
  if (cSize > 3)
    return false;
  auto slide_line = std::max(std::min(m_end.m_row - m_start.m_row, 1), -1);
  auto slide_column =
      std::max(std::min(m_end.m_column - m_start.m_column, 1), -1);
  auto valid_direction = false;
  for (auto dir = Direction::Direction_First;
       dir < Direction::Direction_Last; dir = Direction(dir + 1))
    if (Offsets[dir].m_row == slide_line &&
        Offsets[dir].m_column == slide_column)
      valid_direction = true;
  if (!valid_direction)
    return false;
  auto direction = Offsets[m_direction];
  auto current_row = m_start.m_row;
  auto current_column = m_start.m_column;
  auto size = std::max(cSize, rSize);
  for (auto i = 0; i < size; ++i) {
    if (_state.board_[current_row][current_column] != player)
      return false;
    auto destination_row = current_row + direction.m_row;
    auto destination_column = current_column + direction.m_column;
    if (destination_row < 0 || destination_row >= kNumRows)
      return false;
    if (destination_column < 0 || destination_column >= kNumCols)
      return false;
    if (_state.board_[destination_row][destination_column] != CellState::Empty)
      return false;
    current_row += slide_line;
    current_column += slide_column;
  }

  return true;
}

bool Move::IsValid(const core_state& _state) const {
  if (m_start.m_row < 0 || m_start.m_row >= kNumRows)
    return false;
  if (m_start.m_column < 0 || m_start.m_column >= kNumCols)
    return false;
  if (m_end.m_row < 0 || m_end.m_row >= kNumRows)
    return false;
  if (m_end.m_column < 0 || m_end.m_column >= kNumCols)
    return false;

  auto current_row = m_start.m_row;
  auto current_column = m_start.m_column;
  auto player = _state.board_[current_row][current_column];
  if (player != _state.ToPlay())
    return false;
  // if (player == CellState::Invalid)
  //   return false;
  // if (player == CellState::Empty)
  //   return false;

  auto vectorR = m_end.m_row - m_start.m_row;
  auto vectorC = m_end.m_column - m_start.m_column;
  auto direction = Offsets[m_direction];
  if (direction.m_row != vectorR || direction.m_column != vectorC)
    return _IsValidSlide(_state);

  // check in-row moves
  constexpr int kLineSize = 6;
  CellState line_content[kLineSize];
  for (auto i = 0; i < kLineSize; ++i) {
    if (current_row >= 0 && current_row < kNumRows &&
        current_column >= 0 && current_column < kNumCols) {
      line_content[i] = _state.board_[current_row][current_column];
    } else {
      line_content[i] = CellState::Invalid;
    }
    current_row += vectorR;
    current_column += vectorC;
  }

  CellState opponent = (player == CellState::Player0) ? CellState::Player1
                                                      : CellState::Player0;

  if (line_content[1] == CellState::Empty)
    return true;
  // if (line_content[1] == CellState::Invalid)
  //   return false;
  // if (line_content[1] == opponent)
  //   return false;

  if (line_content[1] == player && line_content[2] == CellState::Empty)
    return true;
  // if (line_content[1] == player && line_content[2] == CellState::Invalid)
  //   return false;

  if (line_content[1] == player && line_content[2] == opponent &&
      (line_content[3] == CellState::Invalid ||
       line_content[3] == CellState::Empty))  // 2VS1
    return true;

  if (line_content[1] == player && line_content[2] == player &&
      line_content[3] == CellState::Empty)
    return true;
  // if (line_content[1] == player && line_content[2] == player
  //     && line_content[3] == CellState::Invalid)
  //   return false;

  if (line_content[1] == player && line_content[2] == player &&
      line_content[3] == opponent &&
      (line_content[4] == CellState::Invalid ||
       line_content[4] == CellState::Empty))  // 3VS1
    return true;

  if (line_content[1] == player &&
      line_content[2] == player &&
      line_content[3] == opponent &&
      line_content[4] == opponent &&
      (line_content[5] == CellState::Invalid ||
       line_content[5] == CellState::Empty))  // 3VS2
    return true;

  return false;
}

// @param _dr, _dc: offsets from the move direction
void Move::_ApplyParallelMove(core_state& _state, int _dr, int _dc) const {
  auto slide_row = std::max(std::min(m_end.m_row - m_start.m_row, 1), -1);
  auto slide_column =
      std::max(std::min(m_end.m_column - m_start.m_column, 1), -1);
  auto rSize = std::abs(m_end.m_row - m_start.m_row);
  auto cSize = std::abs(m_end.m_column - m_start.m_column);
  auto size = std::max(rSize, cSize);
  auto r = m_start.m_row;
  auto c = m_start.m_column;
  auto player = _state.board_[r][c];
  for (int i = 0; i <= size; i++) {
    if (r < 0 || r >= kNumRows || c < 0 || c >= kNumCols)
      break;
    auto id = _state.board_[r][c];
    if (id != player)
      break;
    auto dst_r = r + _dr;
    auto dst_c = c + _dc;
    if (dst_r < 0 || dst_r >= kNumRows || dst_c < 0 || dst_c >= kNumCols)
      break;
    auto nextId = _state.board_[dst_r][dst_c];
    if (nextId != CellState::Empty)  // we could slide on kEmpty and own cells
      break;
    _state.board_[dst_r][dst_c] = player;
    _state.board_[r][c] = CellState::Empty;
    r += slide_row;
    c += slide_column;
  }
}

void Move::_ApplySingleMove(core_state& _state, int _dr, int _dc) const {
  auto r = m_start.m_row;
  auto c = m_start.m_column;
  auto nextId = CellState::Empty;
  while (r >= 0 && r < kNumRows && c >= 0 && c < kNumCols) {  // obvious
    auto currentId = _state.board_[r][c];
    if (currentId == CellState::Invalid)
      break;
    _state.board_[r][c] = nextId;
    if (currentId == CellState::Empty)
      break;
    nextId = currentId;
    r += _dr;
    c += _dc;
  }
}

void Move::Apply(core_state& _state, int _marbles_to_win, int _game_length,
                 bool _marble_advantage) const {
  auto offset = Offsets[m_direction];
  auto vr = m_end.m_row - m_start.m_row;
  auto vc = m_end.m_column - m_start.m_column;

  if (offset.m_row != vr || offset.m_column != vc) {
    _ApplyParallelMove(_state, offset.m_row, offset.m_column);
  } else {
    _ApplySingleMove(_state, offset.m_row, offset.m_column);
  }
  _state.turn_++;
  _state.outcome_ = std::get<1>(
      _state.Eval(_marbles_to_win, _game_length, _marble_advantage));

  // _state.m_turn += 1;
  // auto eval = Eval(_state);
  // _state.m_done = std::get<0>(eval);
  // _state.m_winner = std::get<1>(eval);
}

std::tuple<bool, Move> Move::FromString(const std::string& _str) {
  auto local_str = _str;
  if (local_str.length() != 4 && local_str.length() != 6)
    return std::make_tuple(false, Move());
  for (int i = 0; i < static_cast<int>(local_str.length()); ++i)
    local_str[i] = tolower(local_str[i]);
  // int rowIndex = (kNumRows - 1) - (local_str[0] - 'a');
  int rowIndex = local_str[0] - 'a';
  int colIndex = local_str[1] - '1';
  if (rowIndex < 0 || rowIndex >= kNumRows ||
      colIndex < 0 || colIndex >= kNumCols)
    return std::make_tuple(false, Move());
  auto start = Coordinate{ rowIndex, colIndex };
  // rowIndex = (kNumRows - 1) - (local_str[2] - 'a');
  rowIndex = local_str[2] - 'a';
  colIndex = local_str[3] - '1';
  if (rowIndex < 0 || rowIndex >= kNumRows ||
      colIndex < 0 || colIndex >= kNumCols)
    return std::make_tuple(false, Move());
  auto end = Coordinate{ rowIndex, colIndex };

  // inline move
  if (local_str.length() == 4) {
    auto dir = Direction_Invalid;
    auto vl = end.m_row - start.m_row;
    auto vc = end.m_column - start.m_column;
    for (auto dir = Direction::Direction_First;
         dir < Direction::Direction_Last; dir = Direction(dir + 1)) {
      auto offset = Offsets[dir];
      if (vl == offset.m_row && vc == offset.m_column) {
        return std::make_tuple(true, Move{ dir, start, end });
      }
    }
    return std::make_tuple(false, Move());
  }

  // slide move
  auto end_slide = end;
  auto slide_row = end_slide.m_row - start.m_row;
  auto slide_column = end_slide.m_column - start.m_column;
  if (std::max(std::abs(slide_row), std::abs(slide_column)) > 2)
    return std::make_tuple(false, Move());

  // clip vector to match sister's entry
  slide_row = std::max(std::min(slide_row, 1), -1);
  slide_column = std::max(std::min(slide_column, 1), -1);
  auto slide_vec = Coordinate{ slide_row, slide_column };

  rowIndex = local_str[4] - 'a';
  colIndex = local_str[5] - '1';
  if (rowIndex < 0 || rowIndex >= kNumRows ||
      colIndex < 0 || colIndex >= kNumCols)
    return std::make_tuple(false, Move());

  end = Coordinate{ rowIndex, colIndex };
  auto move_vec = Coordinate{ end.m_row - start.m_row,
                               end.m_column - start.m_column };

  for (auto dir = Direction::Direction_First;
       dir < Direction::Direction_Last; dir = Direction(dir + 1)) {
    auto offset = Offsets[dir];
    if (move_vec.m_row == offset.m_row &&
        move_vec.m_column == offset.m_column) {
      // We assume the slide direction is to the left (dir+N) of the move
      // direction:
      if (Offsets[Sisters[dir].first] == slide_vec ||
          Offsets[Sisters[dir].second] == slide_vec)
        return std::make_tuple(true, Move{ dir, start, end_slide });
      // we have to swap start / end
      return std::make_tuple(true, Move{ dir, end_slide, start });
    }
  }

  return std::make_tuple(false, Move());
}

std::string Move::ToString() const {
  auto offset = Offsets[m_direction];
  auto vl = m_end.m_row - m_start.m_row;
  auto vc = m_end.m_column - m_start.m_column;
  std::string ret;
  if (offset.m_row != vl || offset.m_column != vc) {
    char buff[100];
    snprintf(
        buff, sizeof(buff),
        "%c%c%c%c%c%c",
        static_cast<char>('a' + this->m_start.m_row),
        static_cast<char>('1' + this->m_start.m_column),
        static_cast<char>('a' + this->m_end.m_row),
        static_cast<char>('1' + this->m_end.m_column),
        static_cast<char>('a' + (this->m_start.m_row + offset.m_row)),
        static_cast<char>('1' + this->m_start.m_column + offset.m_column));
    ret = std::string(buff);
  } else {
    char buff[100];
    snprintf(
        buff, sizeof(buff),
        "%c%c%c%c",
        static_cast<char>('a' + this->m_start.m_row),
        static_cast<char>('1' + this->m_start.m_column),
        static_cast<char>('a' + this->m_end.m_row),
        static_cast<char>('1' + this->m_end.m_column));
    ret = std::string(buff);
  }
  return ret;
}

std::ostream& operator<<(std::ostream& os, const Move& _move) {
  os << _move.ToString();
  return os;
}

Move Move::ActionToMove(core_Action moveId) {
  auto remains = moveId;
  auto moveType = remains % kNumActionsPerDirection;
  remains /= kNumActionsPerDirection;
  Direction dir = Direction(remains % Direction::Direction_Last);
  remains /= Direction::Direction_Last;
  auto column = remains % kNumCols;
  remains /= kNumCols;
  auto row = remains;
  Move move;
  move.m_direction = dir;
  move.m_start.m_row = row;
  move.m_start.m_column = column;
  auto offset = Offsets[dir];
  switch (moveType) {
  case 0: {  // single move
    move.m_end.m_row = row + offset.m_row;
    move.m_end.m_column = column + offset.m_column;
    break;
  }
  case 1: {  // slideX2 right front
    auto slide = Offsets[(dir + 1) % Direction::Direction_Last];
    move.m_end.m_row = row + slide.m_row;
    move.m_end.m_column = column + slide.m_column;
    break;
  }
  case 2: {  // slideX2 right back
    auto slide = Offsets[(dir + 2) % Direction::Direction_Last];
    move.m_end.m_row = row + slide.m_row;
    move.m_end.m_column = column + slide.m_column;
    break;
  }
  case 3: {  // slideX3 right front
    auto slide = Offsets[(dir + 1) % Direction::Direction_Last];
    move.m_end.m_row = row + 2 * slide.m_row;
    move.m_end.m_column = column + 2 * slide.m_column;
    break;
  }
  case 4: {  // slideX3 right back
    auto slide = Offsets[(dir + 2) % Direction::Direction_Last];
    move.m_end.m_row = row + 2 * slide.m_row;
    move.m_end.m_column = column + 2 * slide.m_column;
    break;
  }
  }
  return move;
}

core_Action Move::MoveToAction(const Move& move) {
  auto result = move.m_start.m_row;
  result *= kNumCols;
  result += move.m_start.m_column;
  result *= Direction::Direction_Last;
  result += move.m_direction;
  result *= 5;

  auto offset = Offsets[move.m_direction];
  // test slide move
  if (move.m_start.m_row + offset.m_row != move.m_end.m_row ||
      move.m_start.m_column + offset.m_column != move.m_end.m_column) {
    // look for direction
    auto slideF = Offsets[Sisters[move.m_direction].first];
    auto slideB = Offsets[Sisters[move.m_direction].second];

    if (move.m_start.m_row + slideF.m_row == move.m_end.m_row &&
        move.m_start.m_column + slideF.m_column == move.m_end.m_column) {
      result += 1;
    } else if (move.m_start.m_row + slideB.m_row == move.m_end.m_row &&
               move.m_start.m_column + slideB.m_column ==
               move.m_end.m_column) {
      result += 2;
    } else if (
        move.m_start.m_row + slideF.m_row + slideF.m_row ==
            move.m_end.m_row &&
        move.m_start.m_column + slideF.m_column + slideF.m_column ==
            move.m_end.m_column) {
      result += 3;
    } else if (
        move.m_start.m_row + slideB.m_row + slideB.m_row ==
            move.m_end.m_row &&
        move.m_start.m_column + slideB.m_column + slideB.m_column ==
            move.m_end.m_column) {
      result += 4;
    }
  }

  return static_cast<core_Action>(result);
}

}  // namespace abalone_core
