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

// A seat's move and the bot's reply reach a board in one frame: the server
// answers once the bot has replied too. Drawn together, both marks appear in
// the same frame and nobody sees the reply land. This hook says how many of
// the history's actions to draw: those already drawn, then each further
// action a beat after the one before (useStagedPlays), at the viewer's
// Animation speed. A page load, a step back or a single new action draws
// everything at once; Instant and reduced motion stage nothing.

import {scaleMs, useAnimationScale} from '../../motion/index.ts';
import type {OpenSpielState} from './board_session.ts';
import {
  PLAY_REST_MS,
  type StagedPlays,
  useStagedPlays,
} from './use_staged_plays.ts';

/** How long a placed mark or a dropped disc takes, ms at Normal speed. */
export const MOVE_BEAT_MS = 350;

/** How many of `history`'s actions to draw now. */
export function useStagedHistory(history: readonly number[]): StagedPlays {
  const scale = useAnimationScale();
  return useStagedPlays(
    history.map((action, index) => `${String(index)}:${String(action)}`),
    {legMs: scaleMs(MOVE_BEAT_MS, scale), restMs: scaleMs(PLAY_REST_MS, scale)},
  );
}

/**
 * `state` as it stood after its first `shown` actions, for a two-seat game
 * whose seats take turns from seat 0 (tic-tac-toe, connect four). Nothing
 * is legal there and the game is not over yet: the board only shows it on
 * the way to the live position.
 */
export function alternatingStagedState(
  state: OpenSpielState,
  shown: number,
): OpenSpielState {
  if (shown >= state.history.length) return state;
  const seat = shown % 2;
  const labels = state.historyStr.trim().split(/\s+/);
  return {
    ...state,
    history: state.history.slice(0, shown),
    historyStr: labels.slice(0, shown).join(' '),
    currentPlayer: seat,
    currentPlayerLabel: `player ${String(seat)}`,
    isTerminal: false,
    legalActions: [],
    returns: null,
    turnNumber: shown,
  };
}

/**
 * The result of a finished two-seat game in the board's words: "X wins",
 * "Red wins" or "Draw", and, for a viewer who holds a seat, "You win" or
 * "You lose" after it. `sides` names each seat, as the board does.
 */
export function boardResultText(
  returns: readonly number[],
  sides: readonly string[],
  viewerSeat: number | null,
): string {
  if (returns.length !== sides.length) {
    throw new Error(
      `${String(returns.length)} returns for ${String(sides.length)} seats`,
    );
  }
  const best = Math.max(...returns);
  const winners = returns.flatMap((value, seat) =>
    value === best ? [seat] : [],
  );
  const winner = winners.length === 1 ? winners[0] : undefined;
  if (winner === undefined) return 'Draw';
  const text = `${sides[winner] ?? ''} wins`;
  if (viewerSeat === null) return text;
  return `${text} · ${winner === viewerSeat ? 'You win' : 'You lose'}`;
}
