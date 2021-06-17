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

#include "open_spiel/games/yorktown.h"
#include "open_spiel/games/yorktown/yorktown_board.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// This is based on the chess game of open_spiel

namespace open_spiel {
namespace yorktown {

namespace {
// Default Parameters.

// Facts about the game
const GameType kGameType{/*short_name=*/"yorktown",
                         /*long_name=*/"Yorktown",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/false,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {{"players", GameParameter(NumPlayers())},
                         {"strados3", GameParameter(kInitPos)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new YorktownGame(params));
}


REGISTER_SPIEL_GAME(kGameType, Factory);

/* The following methods describe a way to represent the game in so called planes.
* Each plane has a size of BoardSize x BoardSize and represents the presence
* and absence of a given piece type and colour at each square as well as other information.
* These methods are used within to calculate the InformationStateTensor.
*/

// Adds a plane to the information state vector independent from the visibility status of a
// piece, i.e. this plane represents always the piece type even if the piece is hidden.
void AddPieceTypePlane(Color color, Player player, PieceType piece_type,
                       const StandardYorktownBoard& board,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < BoardSize(); ++y) {
    for (int8_t x = 0; x < BoardSize(); ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      *value_it++ =
          ((piece_on_board.color == color && piece_on_board.type == piece_type)
               ? 1.0
               : 0.0);
    }
  }
}

// Adds a plane to the information state vector presenting the probabilities of each possible
// piece type per piece, i.e. if a piece if open the probability for the correct piece type is 1 
// but if the piece is hidden it is represented by a probability distribution over all piece types.
void AddProbabilityPieceTypePlane(Color color, Player player, PieceType piece_type,
                       const StandardYorktownBoard& board,
                       absl::Span<float>::iterator& value_it) {
  
  double countInvisible = 0;
  double countMovedInvisible = 0;
  for (int8_t y = 0; y < BoardSize(); ++y) {
    for (int8_t x = 0; x < BoardSize(); ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      if(piece_on_board.color == color){
        if(piece_on_board.isVisible == false){
           countInvisible++;
           if(piece_on_board.hasMoved == true){
             countMovedInvisible++;
           }
        }
      }
    }
  }
  
  
  for (int8_t y = 0; y < BoardSize(); ++y) {
    for (int8_t x = 0; x < BoardSize(); ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      if(piece_on_board.isVisible == true || piece_on_board.color == PlayerToColor(player) || color == PlayerToColor(player)){
        if(piece_on_board.color == color){ 
          if(piece_on_board.type == piece_type) {
            *value_it++ = 1.0; 
          }
          else *value_it++ = 0.0;
        }else *value_it++ = 0.0;
      }else{
          if(piece_type == PieceType::kFlag || piece_type == PieceType::kBomb){
              if(piece_on_board.hasMoved == true) *value_it++ = 0.0;
              else{
                for(int i = 0; i < 12; ++i){
                  if(piece_type == kPieceTypes[i] && color == Color::kRed){
                    *value_it++ = ((float) board.LivingPieces()[i]-board.find(Piece{color, piece_type, true}).size())/((float) countInvisible-countMovedInvisible);
                  }
                  if(piece_type == kPieceTypes[i] && color == Color::kBlue){
                    *value_it++ = ((float) board.LivingPieces()[i+12]-board.find(Piece{color, piece_type, true}).size())/((float) countInvisible-countMovedInvisible);
                  }
                }
              }
              // For the flag and bomb its 1/unmoved invisible pieces instead of 1/all invisible pieces
          }else{
             for(int i = 0; i < 12; ++i){
                if(piece_type == kPieceTypes[i] && color == Color::kRed){
                  *value_it++ = ((float) board.LivingPieces()[i]-board.find(Piece{color, piece_type, true}).size())/((float) countInvisible);
                }
                if(piece_type == kPieceTypes[i] && color == Color::kBlue){
                  *value_it++ = ((float) board.LivingPieces()[i+12]-board.find(Piece{color, piece_type, true}).size())/((float) countInvisible);
                }
              }
            }
          }  
        }
      }
    }


// Adds a plan representing all pieces of a color/player which are still hidden
void AddUnknownPlane(Color color, const StandardYorktownBoard& board,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < BoardSize(); ++y) {
    for (int8_t x = 0; x < BoardSize(); ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      *value_it++ =
          ((piece_on_board.isVisible == false && piece_on_board.color != color)
               ? 1.0
               : 0.0);
    }
  }
}

// Adds a uniform scalar plane scaled with min and max.
template <typename T>
void AddScalarPlane(T val, T min, T max,
                    absl::Span<float>::iterator& value_it) {
  double normalized_val = static_cast<double>(val - min) / (max - min);
  for (int i = 0; i < BoardSize() * BoardSize(); ++i)
    *value_it++ = normalized_val;
}

// Adds a binary scalar plane representing 0 or 1 on each square.
void AddBinaryPlane(bool val, absl::Span<float>::iterator& value_it) {
  AddScalarPlane<int>(val ? 1 : 0, 0, 1, value_it);
}

}  // namespace

// Constructor with a predefined starting position
YorktownState::YorktownState(std::shared_ptr<const Game> game)
    : State(game),
      start_board_(MakeDefaultBoard()),
      current_board_(start_board_) {
}

// Constructor with a string defining the starting position. The format of the string
// is called Strados3 and is defined in Yorktown_board.
YorktownState::YorktownState(std::shared_ptr<const Game> game, const std::string& strados3)
    : State(game) {
  auto maybe_board = StandardYorktownBoard::BoardFromStraDos3(strados3);
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  
}

// Applies a Move on the board and adds it to the history
void YorktownState::DoApplyAction(Action action) {
  Color c = OppColor(PlayerToColor(CurrentPlayer()));
  Move move = ActionToMove(action, Board());
  moves_history_.push_back(move);
  Board().ApplyMove(move);
  cached_legal_actions_.reset();
}

// Generates legal moves, sort them and temporary save them. 
// It is called Maybe... because it does not check it the current state is a terminal one
void YorktownState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GenerateLegalMoves([this](const Move& move) -> bool {
      //std::cout << move.ToString() << std::endl;
      //std::cout << move.ToLANMove() << std::endl;
      cached_legal_actions_->push_back(MoveToAction(move));
      return true;
    });
    absl::c_sort(*cached_legal_actions_);
  }
}

// Returns the current legal actions as a vector
std::vector<Action> YorktownState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

// Returns the color of a given player
Color PlayerToColor(Player p) {
  SPIEL_CHECK_NE(p, kInvalidPlayer);
  return static_cast<Color>(p);
}

// Construct a Move index based on a square and the destination index
int EncodeMove(const Square& from_square, int destination_index, int board_size,
               int num_actions_destinations) {
  return (from_square.x * board_size + from_square.y) *
             num_actions_destinations +
         destination_index;
}

// Reflect the board ranks for the blue player so that both players are playing from the same side
int8_t ReflectRank(Color to_play, int board_size, int8_t rank) {
  return to_play == Color::kBlue ? board_size - 1 - rank : rank;
}

// Decodes a Move into an Action. It is the counter part to the ActionToMove method.
Action MoveToAction(const Move& move) {
  Color color = move.piece.color;
  // We rotate the move to be from player p's perspective.
  Move player_move(move);

  // Rotate move to be from player p's perspective.
  player_move.from.y = ReflectRank(color, BoardSize(), player_move.from.y);
  player_move.to.y = ReflectRank(color, BoardSize(), player_move.to.y);

  // For each starting square, we enumerate 36 actions:
  // - 9 possible moves per direction (Scout can walk max. 9 into one direction)

  // In total, this results in an upper limit of  100*36 = 3600 indices.
  // Like said, thats an upper limit (eg. only 92 fields are possible (8 lakes) and 
  // not only 16 out of 80 units can walk like this. Further the amount of possible 
  // actions on a board are far less.)
  
  int starting_index =
      EncodeMove(player_move.from, 0, BoardSize(), kNumActionDestinations);
  int8_t x_diff = player_move.to.x - player_move.from.x;
  int8_t y_diff = player_move.to.y - player_move.from.y;
  Offset offset{x_diff, y_diff};
  
  std::array<Offset, 8> tmp;
  // For the normal moves, we simply encode starting and destination square.
  int destination_index =
      chess_common::OffsetToDestinationIndex(offset, tmp, BoardSize());
  SPIEL_CHECK_TRUE(destination_index >= 0 && destination_index < 100);
  return starting_index + destination_index;
  
}

// Converts an Action into a destination square as well as an index
std::pair<Square, int> ActionToDestination(int action, int board_size,
                                           int num_actions_destinations) {
  const int xy = action / num_actions_destinations;
  SPIEL_CHECK_GE(xy, 0);
  SPIEL_CHECK_LT(xy, board_size * board_size);
  const int8_t x = xy / board_size;
  const int8_t y = xy % board_size;
  const int destination_index = action % num_actions_destinations;
  SPIEL_CHECK_GE(destination_index, 0);
  SPIEL_CHECK_LT(destination_index, num_actions_destinations);
  return {Square{x, y}, destination_index};
}

// Converts an Action into a Move. It is the counter part to the MoveToAction method.
Move ActionToMove(const Action& action, const StandardYorktownBoard& board) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);

  // The encoded action represents an action encoded from color's perspective.
  Color color = board.ToPlay();
  
  auto [from_square, destination_index] =
      ActionToDestination(action, BoardSize(), kNumActionDestinations);
  
  SPIEL_CHECK_LT(destination_index, kNumActionDestinations);

  std::array<Offset, 8> tmp;
  
  Offset offset;
  offset = DestinationIndexToOffset(destination_index, tmp,
                                      BoardSize());

  Square to_square = from_square + offset;

  from_square.y = ReflectRank(color, BoardSize(), from_square.y);
  to_square.y = ReflectRank(color, BoardSize(), to_square.y);

  // This uses the current state to infer the piece type.
  Piece piece = {board.ToPlay(), board.at(from_square).type};

  Move move(from_square, to_square, piece);
  return move;
}

// Returns a string representation of an action (1800 -> a4a5)
std::string YorktownState::ActionToString(Player player, Action action) const {
  Move move = ActionToMove(action, Board());
  return move.ToLANMove();
}

// Returns a string representation of the current state
std::string YorktownState::ToString() const { return Board().ToStraDos3(); 
}

// Returns a vector with the returns for each player. Returns 0 if not in a terminal state.
std::vector<double> YorktownState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

// Returns a string representation of the current information state from the perspective of the given player
std::string YorktownState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return Board().ToStraDos3(PlayerToColor(player));
}

// Saves the information state tensor from the perspective of the given player at the given postition (values)
void YorktownState::InformationStateTensor(Player player,
                                   absl::Span<float> values) const {
 
  auto value_it = values.begin();

  /* This lines add perfect information planes for each piece type per player. 
  * for (const auto& piece_type : kPieceTypes) {
  *   AddPieceTypePlane(Color::kRed, player, piece_type, Board(), value_it);
  *   AddPieceTypePlane(Color::kBlue, player, piece_type, Board(), value_it);
  * }
  */

  // Piece configuration with probability representation
  for (const auto& piece_type : kPieceTypes) {
    AddProbabilityPieceTypePlane(Color::kRed, player, piece_type, Board(), value_it);
    AddProbabilityPieceTypePlane(Color::kBlue, player, piece_type, Board(), value_it);
  }

  AddUnknownPlane(Color::kRed, Board(), value_it);
  AddUnknownPlane(Color::kBlue, Board(), value_it);
  
  AddPieceTypePlane(Color::kEmpty, player, PieceType::kEmpty, Board(), value_it);
  AddPieceTypePlane(Color::kEmpty, player, PieceType::kLake, Board(), value_it);
 
 
  // Side to play.
  AddScalarPlane(ColorToPlayer(Board().ToPlay()), 0, 1, value_it);
 
}

// Clones the current YorktownState and returns a unique pointer to the cloned state object.
std::unique_ptr<State> YorktownState::Clone() const {
  //return CloneAndRandomizeToState();
  return std::unique_ptr<State>(new YorktownState(*this));
}

// Checks if the current state is a terminal state and if so returns the reward for each player in form of a tuple.
// TODO: Optimize
void YorktownState::UndoAction(Player player, Action action) {
  // TODO: Make this fast by storing undoing the action instead or replaying the hole game
  SPIEL_CHECK_GE(moves_history_.size(), 1);
  moves_history_.pop_back();
  history_.pop_back();
  current_board_ = start_board_;
  for (const Move& move : moves_history_) {
    current_board_.ApplyMove(move);
  }
}


void YorktownState::DebugString(){
  std::cout << Board().DebugString() << std::endl;
}

absl::optional<std::vector<double>> YorktownState::MaybeFinalReturns() const {
  
  

  // Check if the red player has lost his flag
  if(Board().LivingPieces()[0] != 1){
    return std::vector<double>{LossUtility(), WinUtility()};
    
  }

  // Check if the blue player has lost his flag
  if(Board().LivingPieces()[12] != 1){
    return std::vector<double>{WinUtility(), LossUtility()};
  }

  // Compute and cache the legal actions.
  MaybeGenerateLegalActions();
  SPIEL_CHECK_TRUE(cached_legal_actions_);
  bool have_legal_moves = !cached_legal_actions_->empty();
  

  // If we don't have legal moves we arenÄt able to move and lose the game.
  if(!have_legal_moves) {
    //std::cout << "NLmoves";
    std::vector<double> returns(NumPlayers());
    auto to_play = ColorToPlayer(Board().ToPlay());
    returns[to_play] = LossUtility();
    returns[OtherPlayer(to_play)] = WinUtility();
    return returns;
    
  }

  // Restricts the number of possible plys until it is called a draw. 
  if(Board().Movenumber() > 3000){
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  return std::nullopt;
}


int YorktownGame::MaxGameLength() const {
  // I do not have any clue how to calculate this. 
  return 3000;
}

YorktownGame::YorktownGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace yorktown
}  // namespace open_spiel
