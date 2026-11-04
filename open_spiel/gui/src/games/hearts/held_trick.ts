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

// The Hearts table while a finished trick is on show (cards/library
// trickCollect): the finished trick in the trick area, the live trick's cards
// (the next leader's, bot replies) still in their players' hands, and no game
// over yet. When the trick clears, the live state shows and those cards fly
// from the hands to the trick as any play does.
import type {CompletedTrick} from '../../cards/library/index.ts';
import type {HeartsUIState} from './hooks/use_hearts_state.ts';
import {
  HIDDEN_CARD,
  sortCards,
  type CardCode,
  type PlayerPosition,
} from './types.ts';

/** What the table draws of the state: the hands, the trick, the end. */
export type HeartsView = Pick<
  HeartsUIState,
  'players' | 'trickPile' | 'isTerminal'
>;

/** The table while `held`, a finished trick, is on show (see the file
 *  comment); `pending` are finished tricks still to show. */
export function heartsHeldView(
  state: HeartsView,
  held: CompletedTrick<CardCode, PlayerPosition> | null,
  pending: ReadonlyArray<CompletedTrick<CardCode, PlayerPosition>> = [],
): HeartsView {
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
        : sortCards([...hand, ...back]),
    };
  });
  return {
    players,
    trickPile: held.cards.map(c => ({card: c.card, position: c.seat})),
    isTerminal: false,
  };
}

/** The Hearts table while plays that arrived together are shown one at a
 *  time (games/shared/use_staged_plays.ts): the first `shown` plays of the
 *  trick on the table, the rest still in their players' hands, so each flies
 *  from its hand when its turn to be drawn comes. */
export function heartsStagedView(view: HeartsView, shown: number): HeartsView {
  if (!Number.isInteger(shown) || shown < 0) {
    throw new Error(
      `heartsStagedView: shown must be an integer >= 0, got ${String(shown)}`,
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
        : sortCards([...p.hand, ...back]),
    };
  });
  return {...view, players, trickPile: view.trickPile.slice(0, shown)};
}
