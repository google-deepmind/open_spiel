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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_DUEL_DEFAULT_BOARD_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_DUEL_DEFAULT_BOARD_H_

namespace open_spiel {
namespace crossword_duel {

// Board format:
//
// Tiles:
//   's' - starting location
//   '2' - double letter
//   '@' - double word
//   '3' - tripple letter
//   '#' - tripple word
//
// The board layout is followed by a blank line and then tile distribution
// where '*' is the blank tile, in this format:
//
//   <letter> <number of tiles> <value of the tile>


inline constexpr const char* kDuelDefaultBoard = R"duel(
3...#..2..#...3
.2...3...3...2.
..@...2.2...@..
...3...@...3...
#...@.2.2.@...#
.3...3...3...3.
..2...2.2...2..
2..@...s...@..2
..2...2.2...2..
.3...3...3...3.
#...@.2.2.@...#
...3...@...3...
..@...2.2...@..
.2...3...3...2.
3...#..2..#...3

A 9 1
B 2 3
C 2 4
D 4 2
E 12 1
F 2 4
G 3 3
H 2 4
I 9 1
J 1 8
K 1 5
L 4 1
M 2 3
N 6 1
O 8 1
P 4 3
Q 1 10
R 6 1
S 4 1
T 6 1
U 4 2
V 2 4
W 2 4
X 1 8
Y 2 4
Z 1 10
)duel";

}  // namespace crossword_duel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CROSSWORD_DUEL_DEFAULT_BOARD_H_

