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
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {
namespace universal_poker {

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
    {// Number of Players (up to 10)
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
      gameTree_(((UniversalPokerGame *)game.get())->GetGameTree()),
      gameNode_(gameTree_) {}

std::string UniversalPokerState::ToString() const {
  return gameNode_.ToString();
}

bool UniversalPokerState::IsTerminal() const { return gameNode_.IsFinished(); }

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  return absl::StrCat("player=", player, " move=", move);
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return Player(kTerminalPlayerId);
  }
  if (gameNode_.GetNodeType() == logic::GameTree::GameNode::NODE_TYPE_CHANCE) {
    return Player(kChancePlayerId);
  }

  return Player(gameNode_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (auto player = Player{0}; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = gameNode_.GetTotalReward(player);
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

  logic::CardSet deck(gameTree_->NumSuitsDeck(), gameTree_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = deck.ToCardArray();
  logic::CardSet holeCards = gameNode_.GetHoleCardsOfPlayer(player);

  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  logic::CardSet boardCards = gameNode_.GetBoardCards();
  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = boardCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  std::string actionSeq = gameNode_.GetActionSequence();
  const int length = gameNode_.GetActionSequence().length();
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

  logic::CardSet deck(gameTree_->NumSuitsDeck(), gameTree_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = deck.ToCardArray();
  logic::CardSet holeCards = gameNode_.GetHoleCardsOfPlayer(player);

  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  logic::CardSet boardCards = gameNode_.GetBoardCards();
  for (uint32_t i = 0; i < deck.CountCards(); i++) {
    (*values)[i + offset] = boardCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += deck.CountCards();

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < NumPlayers(); p++) {
    (*values)[offset + p] = gameNode_.Ante(p);
  }
  offset += NumPlayers();
  SPIEL_CHECK_EQ(offset, game_->ObservationTensorShape()[0]);
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, gameTree_->GetNbPlayers());
  const uint32_t pot = gameNode_.MaxSpend() *
                       (gameTree_->GetNbPlayers() - gameNode_.NumFolded());
  std::vector<int> money;
  for (auto p = Player{0}; p < gameTree_->GetNbPlayers(); p++) {
    money.emplace_back(gameNode_.Money(p));
  }
  std::vector<std::string> sequences;
  for (auto r = 0; r <= gameNode_.GetRound(); r++) {
    sequences.emplace_back(gameNode_.BettingSequence(r));
  }

  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s][Private: %s][Public: "
      "%s][Sequences: %s]",
      gameNode_.GetRound(), CurrentPlayer(), pot, absl::StrJoin(money, " "),
      gameNode_.GetHoleCardsOfPlayer(player).ToString(),
      gameNode_.GetBoardCards().ToString(), absl::StrJoin(sequences, "¦"));
}

std::string UniversalPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, gameTree_->GetNbPlayers());
  std::string result;

  const uint32_t pot = gameNode_.MaxSpend() *
                       (gameTree_->GetNbPlayers() - gameNode_.NumFolded());
  absl::StrAppend(&result, "[Round ", gameNode_.GetRound(),
                  "][Player: ", CurrentPlayer(), "][Pot: ", pot, "][Money:");
  for (auto p = Player{0}; p < gameTree_->GetNbPlayers(); p++) {
    absl::StrAppend(&result, " ", gameNode_.Money(p));
  }
  // Add the player's private cards
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "[Private: ",
                    gameNode_.GetHoleCardsOfPlayer(player).ToString(), "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", gameNode_.Ante(p));
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
  const double p = 1.0 / (double)gameNode_.GetActionCount();
  std::vector<std::pair<Action, double>> outcomes(gameNode_.GetActionCount(),
                                                  {0, p});

  for (uint64_t card = 0; card < gameNode_.GetActionCount(); ++card) {
    outcomes[card].first = card;
  }
  return outcomes;
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  std::vector<Action> actions(gameNode_.GetActionCount(), 0);

  for (uint64_t idx = 0; idx < gameNode_.GetActionCount(); idx++) {
    actions[idx] = idx;
  }

  return actions;
}

void UniversalPokerState::DoApplyAction(Action action_id) {
  gameNode_.ApplyAction(action_id);
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      gameTree_(gameDesc_) {}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return std::unique_ptr<State>(new UniversalPokerState(shared_from_this()));
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)

  const int numBoardCards = gameTree_.GetTotalNbBoardCards();
  const int numHoleCards = gameTree_.GetNbHoleCardsRequired();
  const int numPlayers = gameTree_.GetNbPlayers();
  const int gameLength = MaxGameLength();

  return {(numPlayers) +
          (numBoardCards + numHoleCards) *
              (gameTree_.NumRanksDeck() * gameTree_.NumSuitsDeck()) +
          (gameLength * 2)};
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot

  const int numBoardCards = gameTree_.GetTotalNbBoardCards();
  const int numHoleCards = gameTree_.GetNbHoleCardsRequired();
  const int numPlayers = gameTree_.GetNbPlayers();

  return {(numPlayers) +
          (numBoardCards + numHoleCards) *
              (gameTree_.NumRanksDeck() * gameTree_.NumSuitsDeck()) +
          (numPlayers)};
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.

  return (double)gameTree_.StackSize(0) * (gameTree_.GetNbPlayers() - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  return -1 * (double)gameTree_.StackSize(0);
}

int UniversalPokerGame::MaxChanceOutcomes() const {
  return gameTree_.NumSuitsDeck() * gameTree_.NumRanksDeck();
}

int UniversalPokerGame::NumPlayers() const { return gameTree_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  return gameTree_.GetMaxBettingActions();
}

logic::GameTree *UniversalPokerGame::GetGameTree() { return &gameTree_; }

std::shared_ptr<const Game> UniversalPokerGame::Clone() const {
  return std::shared_ptr<const Game>(new UniversalPokerGame(*this));
}

double UniversalPokerGame::UtilitySum() const { return 0; }

int UniversalPokerGame::MaxGameLength() const {
  // Make a good guess here because bruteforcing the tree is far too slow
  // One Terminal Action
  int length = 1;

  // Deal Actions
  length += gameTree_.GetTotalNbBoardCards() +
            gameTree_.GetNbHoleCardsRequired() * gameTree_.GetNbPlayers();

  // Check Actions
  length += (NumPlayers() * gameTree_.GetNbRounds());

  // Bet Actions
  double maxStack = 0;
  double maxBlind = 0;
  for (uint32_t p = 0; p < NumPlayers(); p++) {
    maxStack =
        gameTree_.StackSize(p) > maxStack ? gameTree_.StackSize(p) : maxStack;
    maxBlind =
        gameTree_.BlindSize(p) > maxStack ? gameTree_.BlindSize(p) : maxBlind;
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
  std::string gameDesc;
  if (map.find("gameDesc") == map.end()) {
    std::ostringstream generatedDesc;

    generatedDesc << "GAMEDEF" << std::endl;
    generatedDesc << ParameterValue<std::string>("bettingType") << std::endl;
    generatedDesc << "numPlayers = " << (int)ParameterValue<int>("players")
                  << std::endl;
    generatedDesc << "numRounds = " << (int)ParameterValue<int>("rounds")
                  << std::endl;

    generatedDesc << "stack = ";
    for (int p = 0; p < ParameterValue<int>("players"); p++) {
      generatedDesc << ParameterValue<int>("stackPerPlayer") << " ";
    }
    generatedDesc << std::endl;

    generatedDesc << "blind = ";
    for (int p = 0; p < ParameterValue<int>("players"); p++) {
      if (p == 0) {
        generatedDesc << ParameterValue<int>("bigBlind") << " ";
      } else if (p == 1) {
        generatedDesc << ParameterValue<int>("smallBlind") << " ";
      } else {
        generatedDesc << "0 ";
      }
    }
    generatedDesc << std::endl;

    generatedDesc << "firstPlayer = "
                  << (std::string)ParameterValue<std::string>("firstPlayer")
                  << std::endl;
    generatedDesc << "numSuits = " << (int)ParameterValue<int>("numSuits")
                  << std::endl;
    generatedDesc << "numRanks = " << (int)ParameterValue<int>("numRanks")
                  << std::endl;
    generatedDesc << "numHoleCards = "
                  << (int)ParameterValue<int>("numHoleCards") << std::endl;
    generatedDesc << "numBoardCards = "
                  << (std::string)ParameterValue<std::string>("numBoardCards")
                  << std::endl;

    generatedDesc << "END GAMEDEF" << std::endl;

    gameDesc = generatedDesc.str();

  } else {
    gameDesc = ParameterValue<std::string>("bettingType");
  }
  return gameDesc;
}
}  // namespace universal_poker
}  // namespace open_spiel
