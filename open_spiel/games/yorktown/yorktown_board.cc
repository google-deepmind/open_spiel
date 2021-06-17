// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/yorktown/yorktown_board.h"

#include <cctype>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace yorktown {

// Returns a string representation for a color
std::string ColorToString(Color c) {
  switch (c) {
    case Color::kRed:
      return "red";
    case Color::kBlue:
      return "blue";
    case Color::kEmpty:
      return "empty";
    default:
      SpielFatalError(absl::StrCat("Unknown color: ", c));
      return "This will never return.";
  }
}

// Returns the piece type given the easier char representation of a piece
absl::optional<PieceType> PieceTypeFromChar(char c) {
  switch (toupper(c)) {
    case '_':
      return PieceType::kLake;
    case 'B':
      return PieceType::kBomb;
    case 'F':
      return PieceType::kFlag;
    case 'S':
      return PieceType::kSpy;
    case '2':
      return PieceType::kScout;
    case '3':
      return PieceType::kMiner;
    case '4':
      return PieceType::kSergeant;
    case '5':
      return PieceType::kLieutenant;
    case '6':
      return PieceType::kCaptain;
    case '7':
      return PieceType::kMajor;
    case '8':
      return PieceType::kColonel;
    case '9':
      return PieceType::kGeneral;
    case '0':
      return PieceType::kMarshal;
    default:
      std::cerr << "Invalid piece type: " << c << std::endl;
      return std::nullopt;
  }
}

// Returns a string representation of the given piece type
std::string PieceTypeToString(PieceType p) {
  switch (p) {
    case PieceType::kEmpty:
      return " ";
    case PieceType::kLake:
      return "_";
    case PieceType::kBomb:
      return "B";
    case PieceType::kFlag:
      return "F";
    case PieceType::kSpy:
      return "S";
    case PieceType::kScout:
      return "2";
    case PieceType::kMiner:
      return "3";
    case PieceType::kSergeant:
      return "4";
    case PieceType::kLieutenant:
      return "5";
    case PieceType::kCaptain:
      return "6";
    case PieceType::kMajor:
      return "7";
    case PieceType::kColonel:
      return "8";
    case PieceType::kGeneral:
      return "9";
    case PieceType::kMarshal:
      return "0";
    default:
      SpielFatalError("Unknown piece.");
      return "This will never return.";
  }
}

// Maps the strados char representation to a piece type
absl::optional<PieceType> PieceTypeFromStraDos(char c) {
  switch (toupper(c)) {
    case 'A':
      return PieceType::kEmpty;
    case '_':
      return PieceType::kLake;
    case 'B':
    case 'N':
      return PieceType::kBomb;
    case 'M':
    case 'Y':
      return PieceType::kFlag;
    case 'C':
    case 'O':
      return PieceType::kSpy;
    case 'D':
    case 'P':
      return PieceType::kScout;
    case 'E':
    case 'Q':
      return PieceType::kMiner;
    case 'F':
    case 'R':
      return PieceType::kSergeant;
    case 'G':
    case 'S':
      return PieceType::kLieutenant;
    case 'H':
    case 'T':
      return PieceType::kCaptain;
    case 'I':
    case 'U':
      return PieceType::kMajor;
    case 'J':
    case 'V':
      return PieceType::kColonel;
    case 'K':
    case 'W':
      return PieceType::kGeneral;
    case 'L':
    case 'X':
      return PieceType::kMarshal;
    default:
      std::cerr << "Invalid piece type: " << c << std::endl;
      return std::nullopt;
  }
}

// Maps the piece type to a strados notation char, for this the color or the piece is also necessary
std::string PieceTypeToStradosString(PieceType p, Color c) {
  switch (p) {
    case PieceType::kEmpty:
      return "A";
    case PieceType::kLake:
      return "_";
    case PieceType::kBomb:
      return c == Color::kRed ? "B" : "N";
    case PieceType::kFlag:
      return c == Color::kRed ? "M" : "Y";
    case PieceType::kSpy:
      return c == Color::kRed ? "C" : "O";
    case PieceType::kScout:
      return c == Color::kRed ? "D" : "P";
    case PieceType::kMiner:
      return c == Color::kRed ? "E" : "Q";
    case PieceType::kSergeant:
      return c == Color::kRed ? "F" : "R";
    case PieceType::kLieutenant:
      return c == Color::kRed ? "G" : "S";
    case PieceType::kCaptain:
      return c == Color::kRed ? "H" : "T";
    case PieceType::kMajor:
      return c == Color::kRed ? "I" : "U";
    case PieceType::kColonel:
      return c == Color::kRed ? "J" : "V";
    case PieceType::kGeneral:
      return c == Color::kRed ? "K" : "W";
    case PieceType::kMarshal:
      return c == Color::kRed ? "L" : "X";
    default:
      SpielFatalError("Unknown piece.");
      return "This will never return.";
  }
}

//Method within the piece construct
std::string Piece::ToString(Color c) const {
  std::string base = PieceTypeToString(type);
  if (type == PieceType::kEmpty) return "  ";
  if (type == PieceType::kLake) return "__";
  if (c == Color::kEmpty){
    if(color == Color::kBlue) return isVisible ? "b"+base : "B"+base;
    else return isVisible ? "r"+base : "R"+base;
  }else if (c == Color::kBlue){
    if(color == Color::kBlue) return isVisible ? "b"+base : "B"+base;
    else return isVisible ? "r"+base : "R?";
  }else if (c == Color::kRed){
    if(color == Color::kBlue) return isVisible ? "b"+base : "B?";
    else return isVisible ? "r"+base : "R"+base;
  }
}

//Method within the piece construct, this time for strados notation
std::string Piece::ToStraDosString(Color c) const {
  if(type == PieceType::kLake) return PieceTypeToStradosString(type, Color::kBlue);
  if (c == Color::kEmpty){
    if(color == Color::kBlue) return isVisible ?  absl::AsciiStrToLower(PieceTypeToStradosString(type, Color::kBlue)) : PieceTypeToStradosString(type, Color::kBlue);
    else return isVisible ?  absl::AsciiStrToLower(PieceTypeToStradosString(type, Color::kRed)) : PieceTypeToStradosString(type, Color::kRed);
  }else if (c == Color::kBlue){
    if(color == Color::kBlue) return isVisible ?  absl::AsciiStrToLower(PieceTypeToStradosString(type, Color::kBlue)) : PieceTypeToStradosString(type, Color::kBlue);
    else return isVisible ?  absl::AsciiStrToLower(PieceTypeToStradosString(type, Color::kRed)) : "?";
  }else if (c == Color::kRed){
    if(color == Color::kBlue) return isVisible ?  absl::AsciiStrToLower(PieceTypeToStradosString(type, Color::kBlue)) : "?";
    else return isVisible ?  absl::AsciiStrToLower(PieceTypeToStradosString(type, Color::kRed)) : PieceTypeToStradosString(type, Color::kRed);
  }
}

// Returns a square given its file and rank (a7 -> Square Nr. 6)
absl::optional<Square> SquareFromString(const std::string &s) {
  if (s.size() != 2) return chess_common::kInvalidSquare;
  auto file = ParseFile(s[0]);
  auto rank = ParseRank(s[1]);

  if (file && rank) return Square{*file, *rank};
  return std::nullopt;
}

// String representation for the move construct 
std::string Move::ToString() const {
  std::string extra;
  if (piece == kEmptyPiece){
    return absl::StrCat("Move: ", SquareToStrategoString(from), " to ", SquareToStrategoString(to), extra);
  }
  return absl::StrCat(piece.ToString(), ": ", SquareToStrategoString(from), " to ", SquareToStrategoString(to), extra);
}

std::string Move::ToLANMove() const {
   return absl::StrCat(SquareToStrategoString(from), SquareToStrategoString(to));
}

std::string Move::ToComplexLANMove(const YorktownBoard<10> &board) const {
  std::string ret;
  absl::StrAppend(&ret, SquareToStrategoString(from));
  if (board.at(from).isVisible && board.at(from).type != PieceType::kEmpty) absl::StrAppend(&ret, "[", PieceTypeToString(board.at(from).type), "]");
  if(board.at(to).type == PieceType::kEmpty) absl::StrAppend(&ret, "-");
  else absl::StrAppend(&ret, "x");
  absl::StrAppend(&ret, SquareToStrategoString(to));
  if (board.at(to).isVisible && board.at(to).type != PieceType::kEmpty) absl::StrAppend(&ret, "[", PieceTypeToString(board.at(to).type), "]");
  return ret;
}

std::string Move::ToStraDosMove() const {
  //TODO get Movenumber
  return std::string("<move id=\"xxx\" source=\""+SquareToStrategoString(from)+"\" target=\""+SquareToStrategoString(to)+"\"/>");
}

template <uint32_t kBoardSize>
YorktownBoard<kBoardSize>::YorktownBoard()
    : to_play_(Color::kRed),
      move_number_(1),
      zobrist_hash_(0) {
  board_.fill(kEmptyPiece);
}

template <uint32_t kBoardSize>
absl::optional<YorktownBoard<kBoardSize>>

// Creates a board based on a StraDos3 string (see header)
YorktownBoard<kBoardSize>::BoardFromStraDos3(const std::string &straDos) {

  YorktownBoard board;

  std::vector<std::string> straDos_parts = absl::StrSplit(straDos, ' ');

  if (straDos_parts.size() != 3 || straDos_parts[0].size() != 100)  {
    std::cerr << "Invalid StraDos: " << straDos << std::endl;
    return std::nullopt;
  }

  // Part 1 piece configuration
  std::string &piece_configuration = straDos_parts[0];

  std::string side_to_move = straDos_parts[1];
  int move_number = std::stoi(straDos_parts[2]);

  std::array<int, kPieceTypes.size()*2> living = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  std::array<int, kPieceTypes.size()> living2 = {0,0,0,0,0,0,0,0,0,0,0,0};

  for(int8_t current_y = 0; current_y < kBoardSize; current_y++){
    for(int8_t current_x = 0; current_x < kBoardSize; current_x++){
      char piece_on_sq = piece_configuration[current_y*kBoardSize+current_x];
      auto piece_type = PieceTypeFromStraDos(toupper(piece_on_sq));
      
      if (piece_type == std::nullopt) return std::nullopt;
      
      Color color;
      if ((piece_on_sq > 'A' && piece_on_sq < 'N') || (piece_on_sq > 'a' && piece_on_sq < 'n')) color = Color::kRed;
      else if ((piece_on_sq >= 'N' && piece_on_sq < 'Z') || (piece_on_sq >= 'n' && piece_on_sq < 'z')) color= Color::kBlue;
      else color = Color::kEmpty;
      
      bool visible = true;;
      if (piece_on_sq > 'A' && piece_on_sq <= 'Z') visible = false;
      

      board.set_square(Square{current_x, current_y},
                         Piece{color, *piece_type, visible});
 
      if((int) *piece_type >= 0){
        if(color == Color::kRed){
          living[((int) *piece_type)-1]++;
        }
        if(color == Color::kBlue){
          living2[((int) *piece_type)-1]++; 
        }
      }
    }
  }

  for(int i = 0; i<kPieceTypes.size();i++){
    living[12+i]= living2[i];
  }

  board.SetLivingPieces(living);

  //Part 2 current player represented by color 

  if (side_to_move == "r") {
    board.SetToPlay(Color::kRed);
  } else if (side_to_move == "b") {
    board.SetToPlay(Color::kBlue);
  } else {
    std::cerr << "Invalid side to move: " << side_to_move << std::endl;
    return std::nullopt;
  }

  //Part 3 ply count

  board.SetMovenumber(move_number);

  return board;
}

// Method to find a vector of all pieces equals the defined piece type
template <uint32_t kBoardSize>
std::vector<Square> YorktownBoard<kBoardSize>::find(const Piece &piece) const {
  std::vector<Square> l;
  for (int8_t y = 0; y < kBoardSize; ++y) {
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Square sq{x, y};
      if (at(sq) == piece) {
        l.push_back(sq);
      }
    }
  }
  return l;
}

// Returns the neighbour field pieces of a specific square
template <uint32_t kBoardSize>
std::array<Piece, 4> YorktownBoard<kBoardSize>::neighbours(const Square &square) const {
  std::array<Piece, 4> l = {kEmptyPiece, kEmptyPiece, kEmptyPiece, kEmptyPiece};
  int x = square.x;
  int y = square.y;
  
  if(x==0){
      auto &piece = at(Square{x+1, y});
      l[0] = piece;
    }else if (x==kBoardSize-1){
      auto &piece = at(Square{x-1, y});
      l[2] = piece;
    }else if(x>0 && x<kBoardSize-1){
      auto &piece = at(Square{x-1, y});
      auto &piece2 = at(Square{x+1, y});
      l[2] = piece;
      l[0] = piece2;
    }

  if(y==0){
      auto &piece = at(Square{x, y+1});
      l[1] = piece;
    }else if (y==kBoardSize-1){
      auto &piece = at(Square{x, y-1});
      l[3] = piece;
    }else if(y>0 && y<kBoardSize-1){
      auto &piece = at(Square{x, y-1});
      auto &piece2 = at(Square{x, y+1});
      l[3] = piece;
      l[1] = piece2;
    }

  return l;
}

// Generates all legal moves of the current position and returns it at the given pointer
template <uint32_t kBoardSize>
void YorktownBoard<kBoardSize>::GenerateLegalMoves(
    const MoveYieldFn &yield) const {
  bool generating = true;

#define YIELD(move)     \
  if (!yield(move)) {   \
    generating = false; \
  }

  for (int8_t y = 0; y < kBoardSize && generating; ++y) {
    for (int8_t x = 0; x < kBoardSize && generating; ++x) {
      Square sq{x, y};
      auto &piece = at(sq);
      if (piece.type != PieceType::kEmpty && piece.type != PieceType::kLake && piece.color == to_play_) {
        switch (piece.type) {
          case PieceType::kBomb:
            break;
          case PieceType::kFlag:
            break;
          case PieceType::kSpy:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kMiner:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kSergeant:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kLieutenant:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kCaptain:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kMajor:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kColonel:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kGeneral:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kMarshal:
            GenerateStandardDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          case PieceType::kScout:
            GenerateScoutDestinations_(
                sq, to_play_,
                [&yield, &piece, &sq, &generating](const Square &to) {
                  YIELD(Move(sq, to, piece));
                });
            break;
          default:
            std::cerr << "Unknown piece type: " << static_cast<int>(piece.type)
                      << std::endl;
        }
      }
    }
  }

#undef YIELD
}


//parses a move from string to the move construct
template <uint32_t kBoardSize>
absl::optional<Move> YorktownBoard<kBoardSize>::ParseMove(
    const std::string &move) const {

  // First see if they are in the long form -
  // "anan" (eg. "e2e4")
  // StraDos Moves have a xml format
  // <move id="n" source="an" target="an"/>
  // eg. <move id="2" source="B4" target="B5"/>
 
  auto lan_move = ParseLANMove(move);
  if (lan_move) {
    return lan_move;
  }

  auto complex_lan_move = ParseComplexLANMove(move);
  if (complex_lan_move) {
    return complex_lan_move;
  }

  auto straDos_move = ParseStraDosMove(move);
  if (straDos_move) {
    return straDos_move;
  }

  return std::nullopt;
}

template <uint32_t kBoardSize>
absl::optional<Move> YorktownBoard<kBoardSize>::ParseLANMove(
    const std::string &move) const {
  SPIEL_CHECK_FALSE(move.empty());

  // Long algebraic notation moves (of the variant we care about) are in one form of
  // "anan" (eg E2E4), "an-an" (eg. "E2-E4") or an[t]xan[t] (eg. E2[C]xE3[P])
  if (move.size() == 4) {
    if (move[0] < 'a' || move[0] >= ('a' + kBoardSize+1) || move[1] < '1' ||
        move[1] >= ('1' + kBoardSize) || move[2] < 'a' ||
        move[2] >= ('a' + kBoardSize+1) || move[3] < '1' ||
        move[3] >= ('1' + kBoardSize)) {
      return std::nullopt;
    }

    auto from = SquareFromString(move.substr(0, 2));
    auto to = SquareFromString(std::string(absl::ClippedSubstr(move, 2, 2)));
    if (from && to) {
      
      std::vector<Move> candidates;
      GenerateLegalMoves(
          [&to, &from, &candidates](const Move &move) {
            if (move.from == *from && move.to == *to) {
              candidates.push_back(move);
            }
            return true;
          });

      if (candidates.empty()) {
        std::cerr << "Illegal move - " << move << " on " << DebugString()
                  << std::endl;
        return std::nullopt;
      } else if (candidates.size() > 1) {
        std::cerr << "Multiple matches (is promotion type missing?) - " << move
                  << std::endl;
        return std::nullopt;
      }
      
      return candidates[0];
    }
  } else {
    return std::nullopt;
  }
  SpielFatalError("All conditionals failed; this is a bug.");
  return std::nullopt;
}

template <uint32_t kBoardSize>
absl::optional<Move> YorktownBoard<kBoardSize>::ParseComplexLANMove(
    const std::string &move) const {
  SPIEL_CHECK_FALSE(move.empty());

  // Long algebraic notation moves (of the variant we care about) are in one form of
  // "an-an" (eg. "E2-E4") or an[t]xan[t] (eg. E2[C]xE3[P]) or an[t]-an, an-an[t]
  if(move.size() == 5){
    std::string LAN = move.substr(0, 2) + move.substr(3, 2);
    return ParseLANMove(LAN);
  }

  if(move.size() == 11){
    std::string LAN = move.substr(0, 2) + move.substr(6, 2);
    return ParseLANMove(LAN);
  }

  if(move.size() == 8){
    std::string LAN;
    if(move.substr(3,1) == "[") LAN = move.substr(0, 2) + move.substr(6, 2);
    else LAN = move.substr(0, 2) + move.substr(3, 2);
    return ParseLANMove(LAN);
  }

  return std::nullopt;
}

template <uint32_t kBoardSize>
absl::optional<Move> YorktownBoard<kBoardSize>::ParseStraDosMove(
    const std::string &move) const {
  SPIEL_CHECK_FALSE(move.empty());

      // split the move String // eg. <move id="2" source="B4" target="B5"/>
      // into "B4B5"
      std::vector<std::string> seglist = absl::StrSplit(move, '"');
      if (seglist.size() != 7) return std::nullopt;

      std::string LANmove = seglist[3] + seglist[5];
      return ParseLANMove(LANmove);

}

template <uint32_t kBoardSize>
void YorktownBoard<kBoardSize>::ApplyMove(const Move &move) {
  // Most moves are simple - we remove the moving piece from the original
  // square, and put it on the destination square.
  
  Piece moving_piece = at(move.from);
  Piece destination_piece = at(move.to);

  // We set hasMoved = true 
  moving_piece.hasMoved = true;

  if(destination_piece.type == PieceType::kLake ){
    SpielFatalError(absl::StrCat("You moved into a lake, how?!: "));
  }
  if(destination_piece == kEmptyPiece){
    set_square(move.from, kEmptyPiece);
    set_square(move.to, moving_piece);
  }
  else{
    // We fight

    // Fist both pieces are now visible to both players
    moving_piece.isVisible = true;
    destination_piece.isVisible = true;

    // Case 1: Run into a bomb
    if(moving_piece.type != PieceType::kMiner && destination_piece.type == PieceType::kBomb){
      set_square(move.from, kEmptyPiece);
      set_square(move.to, destination_piece);
      capturePiece(moving_piece);
    }
    // Case 2: Run into a bomb as a Miner
    else if(moving_piece.type == PieceType::kMiner && destination_piece.type == PieceType::kBomb){
      set_square(move.from, kEmptyPiece);
      set_square(move.to, moving_piece);
      capturePiece(destination_piece);
    }
    // Case 3: Spy catches Marshall
    else if(moving_piece.type == PieceType::kSpy && destination_piece.type == PieceType::kMarshal){
      set_square(move.from, kEmptyPiece);
      set_square(move.to, moving_piece);
      capturePiece(destination_piece);
    }
    // Case 4: Defaults
    else if(moving_piece.type > destination_piece.type){
      set_square(move.from, kEmptyPiece);
      set_square(move.to, moving_piece);
      capturePiece(destination_piece);
    }
    else if(moving_piece.type < destination_piece.type){
      set_square(move.from, kEmptyPiece);
      set_square(move.to, destination_piece);
      capturePiece(moving_piece);
    }
    else if(moving_piece.type == destination_piece.type){
      set_square(move.from, kEmptyPiece);
      set_square(move.to, kEmptyPiece);
      capturePiece(moving_piece);
      capturePiece(destination_piece);
    }

  }
  

  ++move_number_;
  SetToPlay(OppColor(to_play_));
}

// Returns the board as a string representation from a specific perspective
template <uint32_t kBoardSize>
std::string YorktownBoard<kBoardSize>::DebugString(Color c, bool onlyBoard) const {
  std::string s; 
  if(onlyBoard == false) s = absl::StrCat("Strados: ", ToStraDos3(c), "\n");
 
  absl::StrAppend(&s, "\n  ---------------------------------------------------\n");
  for (int8_t y = kBoardSize - 1; y >= 0; --y) {
    // Rank label.
    absl::StrAppend(&s, RankToString(y), " ");

    // Pieces on the rank.
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Square sq{x, y};
      absl::StrAppend(&s, "| ", at(sq).ToString(c), " ");
    }
    absl::StrAppend(&s, "|\n");
    absl::StrAppend(&s, "  ---------------------------------------------------\n");
  }

  // File labels.
  absl::StrAppend(&s, "    ");
  for (int8_t x = 0; x < kBoardSize; ++x) {
    absl::StrAppend(&s, FileToString(x), "    ");
  }

  if(onlyBoard == true) return s;

  absl::StrAppend(&s, "\n");
  absl::StrAppend(&s, "\n");

  if(c == Color::kRed) absl::StrAppend(&s,"This is the partial observation of the red player \n");
  else if(c == Color::kBlue ) absl::StrAppend(&s,"This is the partial observation of the blue player \n");
  else absl::StrAppend(&s,"This is the observers view with all information \n");

  absl::StrAppend(&s, "To play: ", to_play_ == Color::kRed ? "R" : "B", "\n");
  absl::StrAppend(&s, "Move number: ", move_number_, "\n\n");
  absl::StrAppend(&s, "Living Pieces Red: ");
  for(int i = 0; i < kPieceTypes.size(); ++i){
        absl::StrAppend(&s, std::to_string(livingPieces_[i]), " ");
      }
  absl::StrAppend(&s, "\n");
  absl::StrAppend(&s, "Living Pieces Blue: ");
  for(int i = 0; i < kPieceTypes.size(); ++i){
        absl::StrAppend(&s, std::to_string(livingPieces_[i+12]), " ");
      }

  absl::StrAppend(&s, "\n\n");

  return s;
}

// The same as DebugString but with StraDos Notation
template <uint32_t kBoardSize>
std::string YorktownBoard<kBoardSize>::DebugStringStraDos(Color c, bool onlyBoard) const {
  std::string s;
  if(onlyBoard == false) s = absl::StrCat("Strados: ", ToStraDos3(c), "\n");
  
  absl::StrAppend(&s, "\n  -----------------------------------------\n");
  for (int8_t y = kBoardSize - 1; y >= 0; --y) {
    // Rank label.
    absl::StrAppend(&s, RankToString(y), " ");

    // Pieces on the rank.
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Square sq{x, y};
      absl::StrAppend(&s, "| ", at(sq).ToStraDosString(c), " ");
    }
    absl::StrAppend(&s, "|\n");
    absl::StrAppend(&s, "  -----------------------------------------\n");
  }

  // File labels.
  absl::StrAppend(&s, "    ");
  for (int8_t x = 0; x < kBoardSize; ++x) {
    absl::StrAppend(&s, FileToString(x), "   ");
  }

  if(onlyBoard == true) return s;

  absl::StrAppend(&s, "\n");
  absl::StrAppend(&s, "\n");

  if(c == Color::kRed) absl::StrAppend(&s,"This is the partial observation of the red player \n");
  else if(c == Color::kBlue ) absl::StrAppend(&s,"This is the partial observation of the blue player \n");
  else absl::StrAppend(&s,"This is the observers view with all information \n");



  absl::StrAppend(&s, "To play: ", to_play_ == Color::kRed ? "R" : "B", "\n");
  absl::StrAppend(&s, "Move number: ", move_number_, "\n\n");

  return s;
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void YorktownBoard<kBoardSize>::GenerateStandardDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  static const std::array<Offset, 4> kOffsets = {
      {{1, 0}, {0, 1}, {0, -1}, {-1, 0}}};

  for (const auto &offset : kOffsets) {
    Square dest = sq + offset;
    if (InBoardArea(dest) && IsEmptyOrEnemy(dest, color)) {
      yield(dest);
    }
  }
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void YorktownBoard<kBoardSize>::GenerateScoutDestinations_(
    Square sq, Color color, const YieldFn &yield) const {
  GenerateRayDestinations_(sq, color, {1, 0}, yield);
  GenerateRayDestinations_(sq, color, {-1, 0}, yield);
  GenerateRayDestinations_(sq, color, {0, 1}, yield);
  GenerateRayDestinations_(sq, color, {0, -1}, yield);
}

template <uint32_t kBoardSize>
template <typename YieldFn>
void YorktownBoard<kBoardSize>::GenerateRayDestinations_(
    Square sq, Color color, Offset offset_step, const YieldFn &yield) const {
  for (Square dest = sq + offset_step; InBoardArea(dest); dest += offset_step) {
    if (IsEmpty(dest)) {
      yield(dest);
    } else if (IsEnemy(dest, color)) {
      yield(dest);
      break;
    } else {
      // We have a friendly piece or Lake.
      break;
    }
  }
}

template <uint32_t kBoardSize>
std::string YorktownBoard<kBoardSize>::ToString(Color c) const {
 std::string strados;

  // 1. encode the board.
  for (int8_t rank = 0; rank < kBoardSize ; ++rank) {
    for (int8_t file = 0; file < kBoardSize; ++file) {
      auto piece = at(Square{file, rank});
      strados += piece.ToStraDosString(c);
    }
  }

  return strados;
}

template <uint32_t kBoardSize>
std::string YorktownBoard<kBoardSize>::ToStraDos3(Color c) const {
  // Example StraDos FEBMBEFEEFbgiBHIBEDBGJDDDHCGJGDhdlIFKDDHAA__AA__AAAA__AA__AASTPQNSQPTSUPWPVrpXPURNqonNQSNVPTNQRRTYup r 20
  std::string strados = ToString(c);

  // 2. color to play.
  strados +=" " + (to_play_ == Color::kRed ? std::string("r") : std::string("b"));

  // 3. full-move clock
  strados += " " + std::to_string(move_number_);

  return strados;
}

template <uint32_t kBoardSize>
void YorktownBoard<kBoardSize>::set_square(Square sq, Piece piece) {
  static const ZobristTableU64<kBoardSize * kBoardSize, 4, 11> kZobristValues(
      /*seed=*/2765481);

  auto position = SquareToIndex_(sq);
  auto current_piece = at(sq);
  zobrist_hash_ ^=
      kZobristValues[position][static_cast<int>(current_piece.color)]
                    [static_cast<int>(current_piece.type)];
  
  zobrist_hash_ ^=
      kZobristValues[position][static_cast<int>(piece.color)]
                    [static_cast<int>(piece.type)];
 
  board_[position] = piece;
}

template <uint32_t kBoardSize>
void YorktownBoard<kBoardSize>::SetToPlay(Color c) {
  static const ZobristTableU64<2> kZobristValues(/*seed=*/284628);

  // Remove old color and add new to play.
  zobrist_hash_ ^= kZobristValues[ToInt(to_play_)];
  zobrist_hash_ ^= kZobristValues[ToInt(c)];
  to_play_ = c;
}

template <uint32_t kBoardSize>
void YorktownBoard<kBoardSize>::SetMovenumber(int move_number) {
  move_number_ = move_number;
}

template <uint32_t kBoardSize>
void YorktownBoard<kBoardSize>::capturePiece(Piece piece){
  
  if(piece.color == Color::kRed){
    for(int i = 0; i < kPieceTypes.size(); i++){
      if(piece.type == kPieceTypes[i]){
       livingPieces_[i]--;
      }
    }
  }else{
    for(int i = 0; i < kPieceTypes.size(); i++){
      if(piece.type == kPieceTypes[i]){
        capturedPieces_[i+12]++;
        livingPieces_[i+12]--;
      }
    }
  }

}

// Explicit instantiations for all board sizes we care about.
template class YorktownBoard<10>;

YorktownBoard<10> MakeDefaultBoard() {
  auto maybe_board = YorktownBoard<10>::BoardFromStraDos3(
      "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0");
  SPIEL_CHECK_TRUE(maybe_board);
  return *maybe_board;
}

YorktownBoard<10> MakeDefaultBoard(std::string strados3) {
  auto maybe_board = YorktownBoard<10>::BoardFromStraDos3(strados3);
  SPIEL_CHECK_TRUE(maybe_board);
  return *maybe_board;
}



}  // namespace yorktown
}  // namespace open_spiel
