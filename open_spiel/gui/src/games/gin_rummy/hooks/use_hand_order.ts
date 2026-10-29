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

import {useCallback, useMemo, useState} from 'react';
import type {PlayerMeld} from '../types.ts';
import {RANK_CODES} from '../types.ts';

/** Rank value for sorting (A=1, 2-10, J=11, Q=12, K=13). */
function rankValue(card: string): number {
  const idx = RANK_CODES.findIndex(rank => rank === card.charAt(0));
  return idx >= 0 ? idx + 1 : 0;
}

/** Suit value for secondary sorting (s, h, d, c). */
function suitValue(card: string): number {
  return ['s', 'h', 'd', 'c'].indexOf(card.charAt(1));
}

/** Melds first (each grouped, by rank), then deadwood by rank, then suit. */
export function sortCardsWithMelds(
  cards: readonly string[],
  playerMelds: readonly PlayerMeld[],
  bestMeldIds: readonly string[],
): string[] {
  const cardSet = new Set(cards.map(c => c.toLowerCase()));
  const meldCardSet = new Set<string>();
  const meldGroups: string[][] = [];
  for (const meld of playerMelds.filter(m => bestMeldIds.includes(m.id))) {
    const meldCards = meld.cards
      .map(c => c.toLowerCase())
      .filter(c => cardSet.has(c));
    if (meldCards.length > 0) {
      meldCards.sort((a, b) => rankValue(a) - rankValue(b));
      meldGroups.push(meldCards);
      for (const c of meldCards) meldCardSet.add(c);
    }
  }
  const deadwood = cards
    .map(c => c.toLowerCase())
    .filter(c => !meldCardSet.has(c));
  deadwood.sort(
    (a, b) => rankValue(a) - rankValue(b) || suitValue(a) - suitValue(b),
  );
  const cardMap = new Map(cards.map(c => [c.toLowerCase(), c]));
  return [...meldGroups.flat(), ...deadwood].map(c => cardMap.get(c) ?? c);
}

/**
 * `order` with `moved` taken out and put back, in their order, at the gap
 * that was `index` (0..order.length) before they were taken out.
 */
export function reorder(
  order: readonly string[],
  moved: readonly string[],
  index: number,
): string[] {
  if (!(Number.isInteger(index) && index >= 0 && index <= order.length)) {
    throw new Error(
      `reorder index ${String(index)} out of range 0..${String(order.length)}`,
    );
  }
  const set = new Set(moved);
  for (const c of moved) {
    if (!order.includes(c)) {
      throw new Error(`reorder: '${c}' is not in the hand`);
    }
  }
  const before = order.slice(0, index).filter(c => !set.has(c));
  const after = order.slice(index).filter(c => !set.has(c));
  const group = order.filter(c => set.has(c));
  return [...before, ...group, ...after];
}

/** The player's order of their hand, and how to change it. */
export interface HandOrder {
  /** The hand in the player's order. */
  order: readonly string[];
  /** Cards picked (shift+click) to move together. */
  picked: ReadonlySet<string>;
  togglePick(card: string): void;
  /** Move cards to a gap in the hand; clears the picks. */
  move(cards: readonly string[], index: number): void;
  /**
   * The next card to join the hand (a drawn upcard) goes to this gap instead
   * of the end.
   */
  placeNext(card: string, index: number): void;
}

/**
 * `order` brought up to date with the hand: cards gone leave, new cards join
 * at the end, or at `pending`'s gap for that card.
 */
function reconcile(
  order: readonly string[],
  cards: readonly string[],
  pending: {card: string; index: number} | null,
): string[] {
  const now = new Set(cards);
  const next = order.filter(c => now.has(c));
  for (const c of cards) {
    if (next.includes(c)) continue;
    if (pending?.card === c) {
      next.splice(Math.min(pending.index, next.length), 0, c);
    } else {
      next.push(c);
    }
  }
  return next;
}

function sameList(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((c, i) => c === b[i]);
}

interface OrderState {
  base: readonly string[];
  /** Where the next card to join the hand goes, set by placeNext. */
  pending: {card: string; index: number} | null;
  /** The sort trigger last applied. */
  sorted: number;
}

/**
 * The player's own order of their hand. A drawn card joins at the end (or
 * where placeNext put it), a played one leaves; picks clear whenever the
 * hand changes. A change of `sortTrigger` sorts melds first. The order is
 * derived in render, so it never names a card the hand no longer holds.
 */
export function useHandOrder(
  cards: readonly string[],
  sortTrigger: number,
  playerMelds: readonly PlayerMeld[],
  bestMeldIds: readonly string[],
): HandOrder {
  const [state, setState] = useState<OrderState>({
    base: cards,
    pending: null,
    sorted: sortTrigger,
  });
  // Picks belong to one hand: any draw or play clears them.
  const hand = [...cards].sort().join(',');
  const [picks, setPicks] = useState<{hand: string; set: ReadonlySet<string>}>({
    hand,
    set: new Set(),
  });
  const {base, pending} = state;
  const reconciled = useMemo(
    () => reconcile(base, cards, pending),
    [base, cards, pending],
  );
  let order = reconciled;
  if (sortTrigger !== state.sorted) {
    order = sortCardsWithMelds(order, playerMelds, bestMeldIds);
    setState({base: order, pending: null, sorted: sortTrigger});
    setPicks({hand, set: new Set()});
  } else if (!sameList(order, state.base)) {
    // The hand changed: the order follows it, and a pending place is used up.
    setState({base: order, pending: null, sorted: sortTrigger});
  }
  const picked = picks.hand === hand ? picks.set : EMPTY;
  const togglePick = useCallback(
    (card: string) => {
      setPicks(prev => {
        const next = new Set(prev.hand === hand ? prev.set : []);
        if (!next.delete(card)) next.add(card);
        return {hand, set: next};
      });
    },
    [hand],
  );
  const move = useCallback(
    (moved: readonly string[], index: number) => {
      setState(prev => ({...prev, base: reorder(prev.base, moved, index)}));
      setPicks({hand, set: new Set()});
    },
    [hand],
  );
  const placeNext = useCallback((card: string, index: number) => {
    setState(prev => ({...prev, pending: {card, index}}));
  }, []);
  return {order, picked, togglePick, move, placeNext};
}

const EMPTY: ReadonlySet<string> = new Set();
