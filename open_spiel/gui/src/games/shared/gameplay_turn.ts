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

// Whose turn it is, as both board panels (connect four, tic-tac-toe)
// work it out: whether a bot is to move, and whether the viewer may move.

import type {SessionPayload, ViewerContext} from './board_session.ts';
import {isHumanController} from './utils.ts';

/** The turn at the position a board panel shows. */
export interface GameplayTurn {
  /** Whether the seat to move autoplays (a bot thinks rather than waits). */
  readonly currentAutoPlay: boolean;
  /** Whether a seat that is not human is to move. */
  readonly waitingOnBot: boolean;
  /** Whether the session is a game rather than an analysis. */
  readonly isGameMode: boolean;
  /** In a game, whether the viewer's seat is not the one to move. */
  readonly notMyTurn: boolean;
  /** In a game, whether the viewer holds no seat. */
  readonly notParticipant: boolean;
  /** Whether the viewer may not move here. */
  readonly turnLocked: boolean;
}

/** The turn in `session` for `viewer`. */
export function gameplayTurn(
  session: SessionPayload,
  viewer: ViewerContext,
): GameplayTurn {
  const {state, players} = session;
  const currentPlayerSlot = state.currentPlayer;
  const currentPlayerEntry =
    currentPlayerSlot >= 0
      ? players.find(player => player.slot === currentPlayerSlot)
      : undefined;
  const currentController = currentPlayerEntry?.controller;
  const currentAutoPlay = Boolean(currentPlayerEntry?.autoPlay);
  const isHumanTurn =
    currentPlayerSlot >= 0
      ? isHumanController(currentController, currentPlayerSlot)
      : false;
  const waitingOnBot =
    !state.isTerminal && currentPlayerSlot >= 0 && !isHumanTurn;

  // In game mode, only allow actions when it's the viewer's turn
  const isGameMode = session.mode === 'game';
  const notMyTurn = isGameMode && !viewer.isActiveSeat;
  const notParticipant = isGameMode && !viewer.isParticipant;
  const turnLocked = notMyTurn || notParticipant;
  return {
    currentAutoPlay,
    waitingOnBot,
    isGameMode,
    notMyTurn,
    notParticipant,
    turnLocked,
  };
}
