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

// The Euchre table while a finished trick is on show (shared/cards
// trickCollect): the finished trick in the centre, the live trick's cards
// still in their players' hands, no hand-over panel yet, trick counts that do
// not yet count the trick on show (or any queued behind it), and a turn line
// that names the trick's winner rather than the live turn.
import type {CompletedTrick} from '../../cards/library/index.ts';
import type {EuchreUIState} from './hooks/use_euchre_state.ts';
import {TEAM_NAME} from './hand_result.ts';
import {
  HIDDEN_CARD,
  sortCards,
  type CardCode,
  type PlayerPosition,
} from './types.ts';

/** The parts of the table a trick on show changes. */
export type EuchreView = Pick<
  EuchreUIState,
  | 'players'
  | 'trickPile'
  | 'isTerminal'
  | 'phase'
  | 'trumpSuit'
  | 'tricksWon'
  | 'teamOf'
  | 'statusMessage'
>;

/** The table while `held`, a finished trick, is on show, with `pending` queued behind it. */
export function euchreHeldView(
  state: EuchreView,
  held: CompletedTrick<CardCode, PlayerPosition> | null,
  pending: ReadonlyArray<CompletedTrick<CardCode, PlayerPosition>> = [],
): EuchreView {
  if (!held) return state;
  // The deal ends when the last trick does, and the served hands are then
  // the original deal (the engine prints it at game over): while that last
  // trick is on show, the seats that played it hold only what is still to
  // come (queued tricks), not the whole deal again.
  const played = new Set(held.cards.map(c => c.seat));
  const players = state.players.map(p => {
    const hand = state.isTerminal && played.has(p.position) ? [] : p.hand;
    const back = [
      ...pending
        .flatMap(t => t.cards)
        .filter(c => c.seat === p.position)
        .map(c => c.card),
      ...state.trickPile
        .filter(t => t.position === p.position)
        .map(t => t.card),
    ];
    if (back.length === 0) return hand === p.hand ? p : {...p, hand};
    return {
      ...p,
      hand: p.hidden
        ? [...hand, ...back.map(() => HIDDEN_CARD)]
        : sortCards([...hand, ...back], state.trumpSuit),
    };
  });
  // The live counts already include the trick on show and any queued
  // behind it; the table counts them when they are collected.
  const tricksWon = {...state.tricksWon};
  for (const t of [held, ...pending]) {
    const team = state.teamOf[t.winner];
    tricksWon[team] -= 1;
    if (tricksWon[team] < 0) {
      throw new Error(
        `euchre: held trick ${String(t.index)} won by ${team}, but the live count is ${String(state.tricksWon[team])}`,
      );
    }
  }
  return {
    players,
    trickPile: held.cards.map(c => ({card: c.card, position: c.seat})),
    isTerminal: false,
    phase: 'playing',
    trumpSuit: state.trumpSuit,
    tricksWon,
    teamOf: state.teamOf,
    statusMessage: `${TEAM_NAME[state.teamOf[held.winner]]} win the trick`,
  };
}

/** The table while plays that arrived together are drawn one at a time:
 *  the first `shown` plays of the trick on the table, the rest still in
 *  their players' hands, so each flies from its hand when its turn comes. A
 *  hidden hand gets a back, as it had. */
export function euchreStagedView<V extends EuchreView>(
  view: V,
  shown: number,
): V {
  if (!Number.isInteger(shown) || shown < 0) {
    throw new Error(
      `euchreStagedView: shown must be an integer >= 0, got ${String(shown)}`,
    );
  }
  if (shown >= view.trickPile.length) return view;
  const later = view.trickPile.slice(shown);
  const players = view.players.map(p => {
    const back = later.filter(t => t.position === p.position).map(t => t.card);
    if (back.length === 0) return p;
    return {
      ...p,
      hand: p.hidden
        ? [...p.hand, ...back.map(() => HIDDEN_CARD)]
        : sortCards([...p.hand, ...back], view.trumpSuit),
    };
  });
  return {...view, players, trickPile: view.trickPile.slice(0, shown)};
}
