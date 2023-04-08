#include "open_spiel/games/schnapsen.h"

#include <algorithm>
#include <functional>

#include "spiel_globals.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace schnapsen {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"schnapsen",
    /*long_name=*/"Schnapsen",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SchnapsenGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

Player GetOtherPlayer(Player player) { return Player{(player + 1) % 2}; }

// Cards are ordered by suit, i.e. J,Q,K,T,A of the first suit, etc.
int GetSuit(int action) { return action / kSuits; }
int GetValue(int action) { return action % kSuits; }

int GetCardScore(int action) {
  switch (GetValue(action)) {
    case 0:
      return 2;
    case 1:
      return 3;
    case 2:
      return 4;
    case 3:
      return 10;
    case 4:
      return 11;
    default:
      SpielFatalError("Unknown Marker.");
  }
}

bool All(const std::array<bool, kCards>& cards) {
  return std::all_of(cards.begin(), cards.end(), [](bool v) { return v; });
}
bool Any(const std::array<bool, kCards>& cards) {
  return std::any_of(cards.begin(), cards.end(), [](bool v) { return v; });
}
bool Count(const std::array<bool, kCards>& cards) {
  return std::accumulate(cards.begin(), cards.end(), 0);
}
std::vector<Action> Actions(const std::array<bool, kCards>& cards) {
  std::vector<Action> numbers;
  for (int card = 0; card < kCards; card++) {
    if (cards[card]) {
      numbers.push_back(card);
    }
  }

  return numbers;
}

bool GetWinnerReturn(int loser_score) {
  if (loser_score == 0) {
    return 3;
  } else if (loser_score < 33) {
    return 2;
  } else {
    return 1;
  }
}

Player SchnapsenState::GetWinner() const {
  for (int i = 0; i < kNumPlayers; i++) {
    if (scores_[i] > kWinningScore) {
      return Player{i};
    }
  }
  if (All(played_cards_)) {
    return active_player_;
  }

  return kInvalidPlayer;
}

bool SchnapsenState::IsTerminal() const {
  return GetWinner() != kInvalidPlayer;
}

std::vector<double> SchnapsenState::Returns() const {
  int winner = GetWinner();
  if (winner == kInvalidPlayer) {
    return {0, 0};
  }

  int winner_return = GetWinnerReturn(scores_[GetOtherPlayer(winner)]);
  std::vector<double> returns(kNumPlayers);
  returns[winner] = winner_return;
  returns[GetOtherPlayer(winner)] = -winner_return;
  return returns;
}

Player GetTrickWinner(int first_action, int second_action, Player first_player,
                      int attout_suit) {
  if (GetSuit(first_action) != GetSuit(second_action)) {
    return GetSuit(second_action) != attout_suit ? first_player
                                                 : GetOtherPlayer(first_player);
  }

  return GetValue(first_action) > GetValue(second_action)
             ? first_player
             : GetOtherPlayer(first_player);
}

void SchnapsenState::ApplyChanceAction(Action action) {
  int cards_active_player = Count(hands_[active_player_]);
  int cards_other_player = Count(hands_[GetOtherPlayer(active_player_)]);

  SPIEL_CHECK_GE(cards_active_player, cards_other_player);
  SPIEL_CHECK_LT(cards_other_player, kHandSize);

  if (cards_active_player == 0 && cards_other_player == 0) {
    SPIEL_CHECK_EQ(attout_open_card_, kNoCard);

    attout_open_card_ = GetValue(action);
    attout_suit_ = GetSuit(action);
  } else if (cards_active_player == cards_other_player) {
    SPIEL_CHECK_LT(cards_active_player, kHandSize);
    hands_[active_player_][action] = true;
  } else {
    SPIEL_CHECK_LT(GetOtherPlayer(cards_active_player), kHandSize);
    hands_[GetOtherPlayer(active_player_)][action] = true;
  }

  if (Count(stack_cards_) == 0) {
    attout_open_card_ = kNoCard;
  } else {
    stack_cards_[action] = false;
  }
}

void SchnapsenState::ApplyPlayerAction(Action action) {
  SPIEL_CHECK_TRUE(hands_[active_player_][action]);
  SPIEL_CHECK_FALSE(played_cards_[action]);

  hands_[active_player_][action] = false;
  played_cards_[action] = true;

  if (open_card_ == kNoCard) {
    open_card_ == action;
    active_player_ = GetOtherPlayer(active_player_);
  } else {
    Player winner = GetTrickWinner(
        open_card_, action, GetOtherPlayer(active_player_), attout_suit_);
    scores_[active_player_] += GetCardScore(open_card_) + GetCardScore(action);
    open_card_ = kNoCard;
    last_trick_winner_ = winner;

    if (attout_open_card_ != kNoCard) {
      active_player_ = kChancePlayerId;
    } else {
      active_player_ = winner;
    }
  }
}

void SchnapsenState::DoApplyAction(Action action) {
  if (active_player_ == kChancePlayerId) {
    ApplyChanceAction(action);
  } else {
    ApplyPlayerAction(action);
  }
}

std::vector<Action> SchnapsenState::LegalActions() const {
  if (IsTerminal()) return {};

  if (active_player_ == kChancePlayerId) {
    return Actions(stack_cards_);
  } else {
    return Actions(hands_[active_player_]);
  }
}

Player SchnapsenState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }

  return active_player_;
  return kChancePlayerId;
}

std::vector<std::pair<Action, double>> SchnapsenState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> chance_outcomes;
  std::vector<Action> actions = Actions(stack_cards_);

  if (actions.size() == 0) {
    SPIEL_CHECK_NE(attout_open_card_, kNoCard);
    chance_outcomes.push_back({attout_suit_ * attout_open_card_, 1});
  } else {
    for (Action action : actions) {
      chance_outcomes.push_back({action, 1.0 / actions.size()});
    }
  }

  return chance_outcomes;
};

}  // namespace schnapsen
}  // namespace open_spiel