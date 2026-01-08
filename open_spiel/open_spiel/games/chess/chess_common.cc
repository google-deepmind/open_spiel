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

#include "open_spiel/games/chess/chess_common.h"

#include <algorithm>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chess_common {
namespace {

int DiffToDestinationIndex(int diff, int board_size) {
  int destination_index = diff + board_size - 1;
  if (diff > 0) --destination_index;
  return destination_index;
}

int DestinationIndexToDiff(int destination_index, int board_size) {
  int diff = destination_index - board_size + 1;
  if (diff >= 0) ++diff;
  return diff;
}

template <typename KnightOffsets>
int OffsetToDestinationIndexImpl(const Offset& offset,
                                 const KnightOffsets& knight_offsets,
                                 int board_size) {
  // Encodes chess queen moves + knight moves.
  int move_type = -1;
  int destination_index = -1;
  if (offset.x_offset == 0) {
    // vertical moves
    move_type = 0;
    destination_index = DiffToDestinationIndex(offset.y_offset, board_size);
  } else if (offset.y_offset == 0) {
    // horizontal moves
    move_type = 1;
    destination_index = DiffToDestinationIndex(offset.x_offset, board_size);
  } else if (offset.x_offset == offset.y_offset) {
    // left downward or right upward diagonal moves.
    move_type = 2;
    destination_index = DiffToDestinationIndex(offset.x_offset, board_size);
  } else if (offset.x_offset == -offset.y_offset) {
    // left upward or right downward diagonal moves.
    move_type = 3;
    destination_index = DiffToDestinationIndex(offset.x_offset, board_size);
  } else {
    // knight moves.
    move_type = 4;
    auto itr = std::find(knight_offsets.begin(), knight_offsets.end(), offset);
    if (itr != knight_offsets.end()) {
      destination_index = std::distance(knight_offsets.begin(), itr);
    } else {
      SpielFatalError(absl::StrCat("Unexpected offset (",
                                   static_cast<int>(offset.x_offset), ", ",
                                   static_cast<int>(offset.y_offset), ")"));
    }
  }

  return move_type * 2 * (board_size - 1) + destination_index;
}

template <typename KnightOffsets>
Offset DestinationIndexToOffsetImpl(int destination_index,
                                    const KnightOffsets& knight_offsets,
                                    int board_size) {
  int move_type = destination_index / (2 * (board_size - 1));
  destination_index = destination_index % (2 * (board_size - 1));
  int8_t diff = DestinationIndexToDiff(destination_index, board_size);

  if (move_type == 0) {
    return {0, diff};
  } else if (move_type == 1) {
    return {diff, 0};
  } else if (move_type == 2) {
    return {diff, diff};
  } else if (move_type == 3) {
    return {diff, static_cast<int8_t>(-diff)};
  } else if (move_type == 4) {
    SPIEL_CHECK_GE(destination_index, 0);
    SPIEL_CHECK_LT(destination_index, knight_offsets.size());
    return knight_offsets[destination_index];
  } else {
    SpielFatalError(absl::StrCat("Unexpected move type (", move_type, ")"));
  }
}

}  // namespace

int OffsetToDestinationIndex(const Offset& offset,
                             const std::array<Offset, 8>& knight_offsets,
                             int board_size) {
  return OffsetToDestinationIndexImpl(offset, knight_offsets, board_size);
}

int OffsetToDestinationIndex(const Offset& offset,
                             const std::array<Offset, 2>& knight_offsets,
                             int board_size) {
  return OffsetToDestinationIndexImpl(offset, knight_offsets, board_size);
}

Offset DestinationIndexToOffset(int destination_index,
                                const std::array<Offset, 8>& knight_offsets,
                                int board_size) {
  return DestinationIndexToOffsetImpl(destination_index, knight_offsets,
                                      board_size);
}

Offset DestinationIndexToOffset(int destination_index,
                                const std::array<Offset, 2>& knight_offsets,
                                int board_size) {
  return DestinationIndexToOffsetImpl(destination_index, knight_offsets,
                                      board_size);
}

std::pair<Square, int> DecodeNetworkTarget(int i, int board_size,
                                           int num_actions_destinations) {
  int xy = i / num_actions_destinations;
  SPIEL_CHECK_GE(xy, 0);
  SPIEL_CHECK_LT(xy, board_size * board_size);
  int8_t x = xy / board_size;
  int8_t y = xy % board_size;
  int destination_index = i % num_actions_destinations;
  SPIEL_CHECK_GE(destination_index, 0);
  SPIEL_CHECK_LT(destination_index, num_actions_destinations);
  return std::make_pair(Square{x, y}, destination_index);
}

int EncodeNetworkTarget(const Square& from_square, int destination_index,
                        int board_size, int num_actions_destinations) {
  return (from_square.x * board_size + from_square.y) *
             num_actions_destinations +
         destination_index;
}

}  // namespace chess_common
}  // namespace open_spiel
