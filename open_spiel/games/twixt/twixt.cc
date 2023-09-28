#include "open_spiel/spiel_utils.h"

#include "open_spiel/games/twixt/twixt.h"
#include "open_spiel/games/twixt/twixtboard.h"
#include "open_spiel/utils/tensor_view.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>

namespace open_spiel {
namespace twixt {
namespace {

// Facts about the game.
const GameType kGameType {
		/*short_name=*/"twixt",
		/*long_name=*/"TwixT", 
		GameType::Dynamics::kSequential,
		GameType::ChanceMode::kDeterministic,
		GameType::Information::kPerfectInformation, 
		GameType::Utility::kZeroSum,
		GameType::RewardModel::kTerminal,
		/*max_num_players=*/2,
		/*min_num_players=*/2,
		/*provides_information_state_string=*/true,
		/*provides_information_state_tensor=*/false,
		/*provides_observation_string=*/true,
		/*provides_observation_tensor=*/true,
		/*parameter_specification=*/
		{
		  { "board_size", GameParameter(kDefaultBoardSize) },
		  { "ansi_color_output", GameParameter(kDefaultAnsiColorOutput) },
		  { "discount",	GameParameter(kDefaultDiscount)    }
		},
};


std::unique_ptr<Game> Factory(const GameParameters &params) {
	return std::unique_ptr < Game > (new TwixTGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

TwixTState::TwixTState(std::shared_ptr<const Game> game) :   State(game) {
	const TwixTGame &parent_game = static_cast<const TwixTGame&>(*game);
	mBoard = Board(
		parent_game.getBoardSize(),
		parent_game.getAnsiColorOutput()
	);

}

std::string TwixTState::ActionToString(open_spiel::Player player, Action action) const
{
	Move move = mBoard.actionToMove(player, action);	
	std::string s = (player == kRedPlayer) ? "x" : "o";
	s += char(int('a') + move.first);
	s.append(std::to_string(mBoard.getSize() - move.second));
	return s;

};


void TwixTState::setPegAndLinksOnTensor(absl::Span<float> values, const Cell *pCell, int offset, int turn, Move move) const {
	// we flip col/row here for better output in playthrough file 
	TensorView<3> view(values, {kNumPlanes, mBoard.getSize(), mBoard.getSize()-2}, false);
	Move tensorMove = mBoard.getTensorMove(move, turn);

	if (! pCell->hasLinks()) {
		// peg has no links -> use plane 0
		view[{0 + offset, tensorMove.second, tensorMove.first}] = 1.0;
	} else {
		// peg has links -> use plane 1
		view[{1 + offset, tensorMove.second, tensorMove.first}] = 1.0;
	}

	if (pCell->hasBlockedNeighbors()) {
		// peg has blocked neighbors on plane 1 -> use also plane 2
		view[{2 + offset, tensorMove.second, tensorMove.first}] = 1.0;
	}

}


void TwixTState::ObservationTensor (open_spiel::Player player, absl::Span<float> values) const {

	SPIEL_CHECK_GE(player, 0);
	SPIEL_CHECK_LT(player, kNumPlayers);

	const int kOpponentPlaneOffset=3;
	const int kCurPlayerPlaneOffset=0;
	int size = mBoard.getSize();

	// 6 planes of size boardSize x (boardSize-2): 
	// each plane excludes the endlines of the opponent
	// planes 0 (3) are for the unlinked pegs of the current (opponent) player
	// planes 1 (4) are for the linked pegs of the current (opponent) player
	// planes 2 (5) are for the blocked pegs on plane 1 (4) 

	// here we initialize Tensor with zeros for each state
	TensorView<3> view(values, {kNumPlanes, mBoard.getSize(), mBoard.getSize()-2}, true);

	for (int c = 0; c < size; c++) {
		for (int r = 0; r < size; r++) {
			Move move = { c, r };
			const Cell *pCell = mBoard.getConstCell(move); 
			int color = pCell->getColor();
			if (player == kRedPlayer) {
				if (color == kRedColor) {
					// no turn
					setPegAndLinksOnTensor(values, pCell, kCurPlayerPlaneOffset, 0, move);
				} else if (color == kBlueColor) {
					// 90 degr turn (blue player sits left side of red player)
					setPegAndLinksOnTensor(values, pCell, kOpponentPlaneOffset, 90, move);
				}
			} else if (player == kBluePlayer) {
				if (color == kBlueColor) {
					// 90 degr turn 
					setPegAndLinksOnTensor(values, pCell, kCurPlayerPlaneOffset, 90, move);
				} else if (color == kRedColor) {
					// 90+90 degr turn (red player sits left of blue player)
					//setPegAndLinksOnTensor(values, pCell, 5, size-c-2, size-r-1);
					setPegAndLinksOnTensor(values, pCell, kOpponentPlaneOffset, 180, move);
				}
			}
		}			
	}
}

TwixTGame::TwixTGame(const GameParameters &params) :
		Game(kGameType, params),
		mAnsiColorOutput(
				ParameterValue<bool>("ansi_color_output",kDefaultAnsiColorOutput)
		),
		mBoardSize(
				ParameterValue<int>("board_size", kDefaultBoardSize)
		),
		mDiscount(
				ParameterValue<double>("discount", kDefaultDiscount)
		) {
	if (mBoardSize < kMinBoardSize || mBoardSize > kMaxBoardSize) {
		SpielFatalError(
				"board_size out of range [" + std::to_string(kMinBoardSize) + ".."
						+ std::to_string(kMaxBoardSize) + "]: "
						+ std::to_string(mBoardSize) + "; ");
	}

	if (mDiscount <= kMinDiscount || mDiscount > kMaxDiscount) {
		SpielFatalError(
				"discount out of range [" + std::to_string(kMinDiscount)
						+ " < discount <= " + std::to_string(kMaxDiscount) + "]: "
						+ std::to_string(mDiscount) + "; ");
	}
}

}  // namespace twixt
}  // namespace open_spiel
