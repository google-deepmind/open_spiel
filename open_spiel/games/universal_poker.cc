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

#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {

const absl::string_view kNoGameDef = "NoGameDef";

const GameType kGameType{
    /*short_name=*/"universal_poker",
    /*long_name=*/"Universal Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    //
    // The ACPC code uses a specific configuration file to describe the game.
    // The following has been copied from ACPC documentation:
    // """
    // Game definitions can have the following fields (case is ignored):
    //
    // gamedef - the starting tag for a game definition
    // end gamedef - ending tag for a game definition
    // stack - the stack size for each player at the start of each hand (for
    //   no-limit)
    // blind - the size of the blinds for each player (relative to the dealer)
    // raisesize - the size of raises on each round (for limit games)
    // limit - specifies a limit game
    // nolimit - specifies a no-limit game
    // numplayers - number of players in the game
    // numrounds - number of betting rounds per hand of the game
    // firstplayer - the player that acts first (relative to the dealer) on each
    //   round
    // maxraises - the maximum number of raises on each round
    // numsuits - the number of different suits in the deck
    // numranks - the number of different ranks in the deck
    // numholecards - the number of private cards to  deal to each player
    // numboardcards - the number of cards revealed on each round
    //
    // Empty lines or lines with '#' as the very first character will be ignored
    //
    // If you are creating your own game definitions, please note that game.h
    // defines some constants for maximums in games (e.g., number of rounds).
    // These may need to be changed for games outside of the what is being run
    // for the Annual Computer Poker Competition.
    // """
    {// The ACPC gamedef string.  When present, it will take precedence over
     // everything and no other argument should be provided.
     {"gamedef", GameParameter(std::string(kNoGameDef))},
     // Number of Players (up to 10)
     {"players", GameParameter(2)},
     // Betting Type "limit" "nolimit" (currently only nolimit supported)
     {"bettingType", GameParameter(std::string("nolimit"))},
     // Stack of money per Player
     {"stackPerPlayer", GameParameter(1200)},
     {"bigBlind", GameParameter(100)},
     {"smallBlind", GameParameter(100)},
     // Count of Rounds
     {"rounds", GameParameter(2)},
     // Who is the first player by round?
     {"firstPlayer", GameParameter(std::string("1 1"))},
     {"numSuits", GameParameter(4)},
     {"numRanks", GameParameter(6)},
     // Hole Cards (Private Cards) per Player
     {"numHoleCards", GameParameter(1)},
     // Board Cards (Public Cards) per Player
     {"numBoardCards", GameParameter(std::string("0 1"))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new UniversalPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// namespace universal_poker
UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game)
    : State(game),
      game_tree_(((UniversalPokerGame *)game.get())->GetGameTree()),
      game_node_(game_tree_) {}

std::string UniversalPokerState::ToString() const {
  return game_node_.ToString();
}

bool UniversalPokerState::IsTerminal() const { return game_node_.IsFinished(); }

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  return absl::StrCat("player=", player, " move=", move);
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (game_node_.GetNodeType() == logic::GameTree::GameNode::NODE_TYPE_CHANCE) {
    return kChancePlayerId;
  }

  return Player(game_node_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (Player player = 0; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = game_node_.GetTotalReward(player);
  }

  return returns;
}

void UniversalPokerState::InformationStateTensor(
    Player player, std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(game_->InformationStateTensorShape()[0]);
  std::fill(values->begin(), values->end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my card: deck_.size() bits
  //   public card: deck_.size() bits
  //   first round sequence: (max round seq length)*2 bits
  //   second round sequence: (max round seq length)*2 bits

  int offset = 0;

  // Mark who I am.
  (*values)[player] = 1;
  offset += NumPlayers();

  logic::CardSet deck(game_tree_->NumSuitsDeck(), game_tree_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = deck.ToCardArray();
  logic::CardSet holeCards = game_node_.GetHoleCardsOfPlayer(player);

  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  logic::CardSet boardCards = game_node_.GetBoardCards();
  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = boardCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  std::string actionSeq = game_node_.GetActionSequence();
  const int length = game_node_.GetActionSequence().length();
  SPIEL_CHECK_LT(length, game_->MaxGameLength());

  for (int i = 0; i < length; ++i) {
    SPIEL_CHECK_LT(offset + i + 1, values->size());
    if (actionSeq[i] == 'c') {
      // Encode call as 10.
      (*values)[offset + (2 * i)] = 1;
      (*values)[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'p') {
      // Encode raise as 01.
      (*values)[offset + (2 * i)] = 0;
      (*values)[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'a') {
      // Encode raise as 01.
      (*values)[offset + (2 * i)] = 1;
      (*values)[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'f') {
      // Encode fold as 00.
      (*values)[offset + (2 * i)] = 0;
      (*values)[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'd') {
      (*values)[offset + (2 * i)] = 0;
      (*values)[offset + (2 * i) + 1] = 0;
    } else {
      SPIEL_CHECK_EQ(actionSeq[i], 'd');
    }
  }

  // Move offset up to the next round: 2 bits per move.
  offset += game_->MaxGameLength() * 2;

  SPIEL_CHECK_EQ(offset, game_->InformationStateTensorShape()[0]);
}

void UniversalPokerState::ObservationTensor(Player player,
                                            std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  values->resize(game_->ObservationTensorShape()[0]);
  std::fill(values->begin(), values->end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my card: deck_.size() bits
  //   public card: deck_.size() bits
  //   the contribution of each player to the pot. num_players integers.

  int offset = 0;

  // Mark who I am.
  (*values)[player] = 1;
  offset += NumPlayers();

  logic::CardSet deck(game_tree_->NumSuitsDeck(), game_tree_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = deck.ToCardArray();
  logic::CardSet holeCards = game_node_.GetHoleCardsOfPlayer(player);

  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  logic::CardSet boardCards = game_node_.GetBoardCards();
  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = boardCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < NumPlayers(); p++) {
    (*values)[offset + p] = game_node_.Ante(p);
  }
  offset += NumPlayers();
  SPIEL_CHECK_EQ(offset, game_->ObservationTensorShape()[0]);
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_tree_->GetNbPlayers());
  const uint32_t pot = game_node_.MaxSpend() *
                       (game_tree_->GetNbPlayers() - game_node_.NumFolded());
  std::vector<int> money;
  for (auto p = Player{0}; p < game_tree_->GetNbPlayers(); p++) {
    money.emplace_back(game_node_.Money(p));
  }
  std::vector<std::string> sequences;
  for (auto r = 0; r <= game_node_.GetRound(); r++) {
    sequences.emplace_back(game_node_.BettingSequence(r));
  }

  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s][Private: %s][Public: "
      "%s][Sequences: %s]",
      game_node_.GetRound(), CurrentPlayer(), pot, absl::StrJoin(money, " "),
      game_node_.GetHoleCardsOfPlayer(player).ToString(),
      game_node_.GetBoardCards().ToString(), absl::StrJoin(sequences, "¦"));
}

std::string UniversalPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_tree_->GetNbPlayers());
  std::string result;

  const uint32_t pot = game_node_.MaxSpend() *
                       (game_tree_->GetNbPlayers() - game_node_.NumFolded());
  absl::StrAppend(&result, "[Round ", game_node_.GetRound(),
                  "][Player: ", CurrentPlayer(), "][Pot: ", pot, "][Money:");
  for (auto p = Player{0}; p < game_tree_->GetNbPlayers(); p++) {
    absl::StrAppend(&result, " ", game_node_.Money(p));
  }
  // Add the player's private cards
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "[Private: ",
                    game_node_.GetHoleCardsOfPlayer(player).ToString(), "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", game_node_.Ante(p));
  }
  absl::StrAppend(&result, "]");

  return result;
}

std::unique_ptr<State> UniversalPokerState::Clone() const {
  return std::unique_ptr<State>(new UniversalPokerState(*this));
}

std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  const double p = 1.0 / (double)game_node_.GetActionCount();
  std::vector<std::pair<Action, double>> outcomes(game_node_.GetActionCount(),
                                                  {0, p});

  for (uint64_t card = 0; card < game_node_.GetActionCount(); ++card) {
    outcomes[card].first = card;
  }
  return outcomes;
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  std::vector<Action> actions(game_node_.GetActionCount(), 0);

  for (uint64_t idx = 0; idx < game_node_.GetActionCount(); ++idx) {
    actions[idx] = idx;
  }

  return actions;
}

void UniversalPokerState::DoApplyAction(Action action_id) {
  game_node_.ApplyAction(action_id);
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      game_tree_(gameDesc_) {}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return std::unique_ptr<State>(new UniversalPokerState(shared_from_this()));
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)

  const int numBoardCards = game_tree_.GetTotalNbBoardCards();
  const int numHoleCards = game_tree_.GetNbHoleCardsRequired();
  const int numPlayers = game_tree_.GetNbPlayers();
  const int gameLength = MaxGameLength();

  return {(numPlayers) +
          (numBoardCards + numHoleCards) *
              (game_tree_.NumRanksDeck() * game_tree_.NumSuitsDeck()) +
          (gameLength * 2)};
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot

  const int numBoardCards = game_tree_.GetTotalNbBoardCards();
  const int numHoleCards = game_tree_.GetNbHoleCardsRequired();
  const int numPlayers = game_tree_.GetNbPlayers();

  return {(numPlayers) +
          (numBoardCards + numHoleCards) *
              (game_tree_.NumRanksDeck() * game_tree_.NumSuitsDeck()) +
          (numPlayers)};
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.

  return (double)game_tree_.StackSize(0) * (game_tree_.GetNbPlayers() - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  return -1 * (double)game_tree_.StackSize(0);
}

int UniversalPokerGame::MaxChanceOutcomes() const {
  return game_tree_.NumSuitsDeck() * game_tree_.NumRanksDeck();
}

int UniversalPokerGame::NumPlayers() const { return game_tree_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  return game_tree_.GetMaxBettingActions();
}

logic::GameTree *UniversalPokerGame::GetGameTree() { return &game_tree_; }

std::shared_ptr<const Game> UniversalPokerGame::Clone() const {
  return std::shared_ptr<const Game>(new UniversalPokerGame(*this));
}

int UniversalPokerGame::MaxGameLength() const {
  // Make a good guess here because bruteforcing the tree is far too slow
  // One Terminal Action
  int length = 1;

  // Deal Actions
  length += game_tree_.GetTotalNbBoardCards() +
            game_tree_.GetNbHoleCardsRequired() * game_tree_.GetNbPlayers();

  // Check Actions
  length += (NumPlayers() * game_tree_.GetNbRounds());

  // Bet Actions
  double maxStack = 0;
  double maxBlind = 0;
  for (uint32_t p = 0; p < NumPlayers(); p++) {
    maxStack =
        game_tree_.StackSize(p) > maxStack ? game_tree_.StackSize(p) : maxStack;
    maxBlind =
        game_tree_.BlindSize(p) > maxStack ? game_tree_.BlindSize(p) : maxBlind;
  }

  while (maxStack > maxBlind) {
    maxStack /= 2.0;         // You have always to bet the pot size
    length += NumPlayers();  // Each player has to react
  }

  return length;
}

/**
 * Parses the Game Paramters and makes a gameDesc out of it
 * @param map
 * @return
 */
std::string UniversalPokerGame::parseParameters(const GameParameters &map) {
  if (map.find("gamedef") != map.end()) {
    // We check for sanity that all parameters are empty
    if (map.size() != 1) {
      std::vector<std::string> game_parameter_keys;
      game_parameter_keys.reserve(map.size());
      for (auto const &imap : map) {
        game_parameter_keys.push_back(imap.first);
      }
      SpielFatalError(
          absl::StrCat("When loading a 'universal_poker' game, the 'gamedef' "
                       "field was present, but other fields were present too: ",
                       absl::StrJoin(game_parameter_keys, ", "),
                       "gamedef is exclusive with other paraemters."));
    }
    return ParameterValue<std::string>("gamedef");
  }

  std::string generated_gamedef = "GAMEDEF\n";
  absl::StrAppend(
      &generated_gamedef, ParameterValue<std::string>("bettingType"), "\n",
      "numPlayers = ", ParameterValue<int>("players"), "\n",
      "numRounds = ", ParameterValue<int>("rounds"), "\n",
      "numSuits = ", ParameterValue<int>("numSuits"), "\n",
      "firstPlayer = ", ParameterValue<std::string>("firstPlayer"), "\n",
      "numRanks = ", ParameterValue<int>("numRanks"), "\n",
      "numHoleCards = ", ParameterValue<int>("numHoleCards"), "\n",
      "numBoardCards = ", ParameterValue<std::string>("numBoardCards"), "\n");

  absl::StrAppend(&generated_gamedef, "stack = ");
  for (int p = 0; p < ParameterValue<int>("players"); p++) {
    absl::StrAppend(&generated_gamedef, ParameterValue<int>("stackPerPlayer"),
                    " ");
  }
  absl::StrAppend(&generated_gamedef, "\n");

  absl::StrAppend(&generated_gamedef, "blind = ");
  for (int p = 0; p < ParameterValue<int>("players"); p++) {
    if (p == 0) {
      absl::StrAppend(&generated_gamedef, ParameterValue<int>("bigBlind"), " ");
    } else if (p == 1) {
      absl::StrAppend(&generated_gamedef, ParameterValue<int>("smallBlind"),
                      " ");
    } else {
      absl::StrAppend(&generated_gamedef, "0 ");
    }
  }
  absl::StrAppend(&generated_gamedef, "\n");

  absl::StrAppend(&generated_gamedef, "END GAMEDEF\n");
  std::cerr << "Generated gamedef for Universal Poker:\n"
            << generated_gamedef << std::endl;
  return generated_gamedef;
}

}  // namespace universal_poker
}  // namespace open_spiel
