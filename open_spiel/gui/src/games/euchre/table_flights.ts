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

// The Euchre table's card flights: the table
// draws its own card elements, so it says where each arriving card came from
// and the shared card library's arriveFrom flies it there (the twin of the
// trick collect's useCollectElements). Two arrivals:
//  - a trick card flies from its player's hand: from its own face in the
//    hand when the viewer could see it, else from the middle of that seat's
//    hand (a hidden hand reads nothing it does not show);
//  - an up-card that is ordered up flies from where the up-card lay to the
//    dealer's hand.
// Positions are measured after every commit, so the next commit's arrivals
// leave from where the cards were last drawn. The first commit flies nothing
// (a page load shows where things are). Durations come from the card style's
// motion at the viewer's Animation speed (motionDuration); 0 flies nothing.
import {useLayoutEffect, useRef, type RefObject} from 'react';
import {arriveFrom, type ArriveBox} from '../../cards/library/index.ts';
import type {CardCode, PlayerPosition, TrickCard} from './types.ts';
import {trickKey} from './trick_seats.ts';

/** The up-card in the dealer's hand, ordered up and not yet discarded against. */
export interface Pickup {
  dealer: PlayerPosition;
  card: CardCode;
}

interface Snapshot {
  trick: Set<string>;
  pickup: string | null;
  /** Face-up hand cards by code. */
  cards: Map<string, ArriveBox>;
  /** Each seat's hand. */
  seats: Map<PlayerPosition, ArriveBox>;
  upcard: ArriveBox | null;
}

const POSITIONS: readonly PlayerPosition[] = ['bottom', 'left', 'top', 'right'];

function box(el: Element): ArriveBox | null {
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0
    ? {left: r.left, top: r.top, width: r.width, height: r.height}
    : null;
}

function snapshot(
  root: HTMLElement,
  trick: readonly TrickCard[],
  pickup: Pickup | null,
): Snapshot {
  const cards = new Map<string, ArriveBox>();
  for (const el of root.querySelectorAll(
    '.euchre-player-section [data-card-id]',
  )) {
    const b = box(el);
    if (b) cards.set(el.getAttribute('data-card-id') ?? '', b);
  }
  const seats = new Map<PlayerPosition, ArriveBox>();
  for (const p of POSITIONS) {
    const el =
      root.querySelector(`.euchre-player-section--${p} .euchre-card-fan`) ??
      root.querySelector(`.euchre-player-section--${p}`);
    const b = el && box(el);
    if (b) seats.set(p, b);
  }
  const up = root.querySelector('[data-euchre-upcard] .euchre-card');
  return {
    trick: new Set(trick.map(trickKey)),
    pickup: pickup ? `${pickup.dealer}:${pickup.card}` : null,
    cards,
    seats,
    upcard: up ? box(up) : null,
  };
}

/** Flies each card that arrived on the table since the last commit from where it was. */
export function useTableFlights(
  root: RefObject<HTMLElement | null>,
  trick: readonly TrickCard[],
  pickup: Pickup | null,
  motion: {durationMs: number; easing: string},
): void {
  const last = useRef<Snapshot | null>(null);
  // Every commit: fly what arrived since the last one, then measure.
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const prev = last.current;
    if (prev && motion.durationMs > 0) {
      const fly = (
        target: Element | null,
        from: ArriveBox | null | undefined,
      ) => {
        // A card removed mid-flight takes its animation with it.
        if (target instanceof HTMLElement && from) {
          arriveFrom(target, from, motion);
        }
      };
      for (const t of trick) {
        if (prev.trick.has(trickKey(t))) continue;
        fly(
          el.querySelector(`.euchre-trick-pile [data-card-id="${t.card}"]`),
          prev.cards.get(t.card) ?? prev.seats.get(t.position),
        );
      }
      if (pickup && prev.pickup !== `${pickup.dealer}:${pickup.card}`) {
        const hand = el.querySelector(
          `.euchre-player-section--${pickup.dealer}`,
        );
        const cards = hand
          ? [...hand.querySelectorAll('.euchre-card-fan__card')]
          : [];
        // Face up: the up-card itself; face down: the back it became.
        const target =
          hand?.querySelector(`[data-card-id="${pickup.card}"]`) ??
          cards[cards.length - 1] ??
          null;
        fly(target, prev.upcard);
      }
    }
    last.current = snapshot(el, trick, pickup);
  });
}
