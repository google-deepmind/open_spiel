
#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_TWIXT_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_TWIXT_H_

#include "open_spiel/games/twixt/twixtcell.h"
#include "open_spiel/games/twixt/twixtboard.h"
#include <iostream>
#include <string>

// https://en.wikipedia.org/wiki/TwixT

namespace open_spiel {
namespace twixt {

class TwixTState: public State {
	public:

		TwixTState(std::shared_ptr<const Game> game);

		TwixTState(const TwixTState&) = default;
		TwixTState& operator=(const TwixTState&) = default;

		open_spiel::Player CurrentPlayer() const override { return mCurrentPlayer; };

		std::string ActionToString(open_spiel::Player player, Action action) const override;

		std::string ToString() const override { return mBoard.toString(); };

		bool IsTerminal() const override {
			int result = mBoard.getResult();
			return (result == kRedWin || result == kBlueWin || result == kDraw);
		};

		std::vector<double> Returns() const override {
			double reward;
			int result = mBoard.getResult();
			if (result == kOpen || result == kDraw) { return {0.0, 0.0}; }
			else {
				reward = pow(mDiscount, mBoard.getMoveCounter());
				if (result == kRedWin) { return {reward, -reward}; }
				else { return {-reward, reward}; }
			}
		};

		std::string InformationStateString(open_spiel::Player player) const override { 
 			SPIEL_CHECK_GE(player, 0);
  			SPIEL_CHECK_LT(player, kNumPlayers);			
			return ToString(); 
		};

		std::string ObservationString(open_spiel::Player player) const override {
 			SPIEL_CHECK_GE(player, 0);
  			SPIEL_CHECK_LT(player, kNumPlayers);			
			return ToString();
		};

		void ObservationTensor (open_spiel::Player player, absl::Span<float> values) const override;
		
		std::unique_ptr<State> Clone() const override {
			return std::unique_ptr < State > (new TwixTState(*this));
		};
		
		void UndoAction(open_spiel::Player, Action) override {};

		std::vector<Action> LegalActions() const override {
  		    if (IsTerminal()) return {};
			return mBoard.getLegalActions(CurrentPlayer());
		};

	protected:
		void DoApplyAction(Action move) override {
			mBoard.applyAction(CurrentPlayer(), move);
			if (mBoard.getResult() == kOpen) { setCurrentPlayer(1 - CurrentPlayer()); }
			else { setCurrentPlayer(kTerminalPlayerId); }
		};

	private:
		int mCurrentPlayer = kRedPlayer;  
		Board mBoard;
		double mDiscount = kDefaultDiscount;

		void setCurrentPlayer(int player) { mCurrentPlayer = player; }
		void setPegAndLinksOnTensor(absl::Span<float>, const Cell *, int, int, Move) const;

};


class TwixTGame: public Game {

	public:
		explicit TwixTGame(const GameParameters &params);

		std::unique_ptr<State> NewInitialState() const override {
			return std::unique_ptr<State>(new TwixTState(shared_from_this()));
		};

		int NumDistinctActions() const override { return mBoardSize*(mBoardSize-2); };

		int NumPlayers() const override { return kNumPlayers; };
		double MinUtility() const override { return -1.0; };
		virtual absl::optional<double> UtilitySum() const override { return 0.0; };
		double MaxUtility() const override { return 1.0; };

		std::vector<int> ObservationTensorShape() const override {
			static std::vector<int> shape{ kNumPlanes, mBoardSize, mBoardSize-2 };
			return shape;
		}

		int MaxGameLength() const { 
			// square - 4 corners + swap move
			return mBoardSize*mBoardSize - 4 + 1; 
		}
		bool getAnsiColorOutput() const { return mAnsiColorOutput; }
		bool getUnicodeOutput() const { return mUnicodeOutput; }
		int getBoardSize() const { return mBoardSize; }
		double getDiscount() const { return mDiscount; }

	private:
		bool mAnsiColorOutput;
		bool mUnicodeOutput;
		int  mBoardSize;
		double mDiscount;

};

}  // namespace twixt
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_TWIXT_H_
