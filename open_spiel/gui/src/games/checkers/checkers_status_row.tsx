// Copyright 2026 DeepMind Technologies Limited
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

// The checkers board's status row: whose move it is or how the game ended,
// the move number, and what the viewer is waiting for.

import type {JSX} from 'react';

import type {CheckersTurn} from './checkers_gameplay_state.ts';
import {moveNumber, resultText, sideName} from './checkers_model.ts';
import type {CheckersState, CheckersViewer} from './checkers_session.ts';

/** The status row above the board. */
export function CheckersStatusRow({
  state,
  turn,
  viewer,
}: {
  readonly state: CheckersState;
  readonly turn: CheckersTurn;
  readonly viewer: CheckersViewer;
}): JSX.Element {
  const {currentPlayerSlot, waitingOnBot, isGameMode, notMyTurn} = turn;
  const {notParticipant, currentAutoPlay} = turn;
  return (
    <div className="ogs-state-meta">
      <span data-testid="checkers-status">
        {state.isTerminal && state.returns
          ? resultText(state.returns)
          : currentPlayerSlot >= 0
            ? `${sideName(currentPlayerSlot)} to move`
            : 'Finished'}
      </span>
      <span>Move {moveNumber(state.historyPlayers)}</span>
      {waitingOnBot && (
        <span>
          {currentAutoPlay ? 'Bot is thinking…' : 'Waiting for bot step…'}
        </span>
      )}
      {isGameMode && notParticipant && !state.isTerminal && (
        <span className="checkers-spectator-notice">Spectating</span>
      )}
      {isGameMode &&
        viewer.isParticipant &&
        notMyTurn &&
        !waitingOnBot &&
        !state.isTerminal && (
          <span className="checkers-turn-notice">Waiting for your turn…</span>
        )}
      {isGameMode &&
        viewer.isActiveSeat &&
        !waitingOnBot &&
        !state.isTerminal && (
          <span className="checkers-turn-notice checkers-turn-notice--active">
            Your turn!
          </span>
        )}
    </div>
  );
}
