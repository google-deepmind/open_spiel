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

// Trick collect (pure): the one rule every trick-taking table uses to show a
// finished trick before it goes. No React, no DOM.
//
// The served state jumps past a finished trick: the payload after the fourth
// card already has an empty trick (or the next leader's card, or several bot
// replies). So a game cannot "wait" for the trick; it serves the last finished
// trick alongside the live one (`lastTrick`, numbered from 1), and the client
// notices that number grow between two payloads it has rendered. Then the
// finished trick is held on the table (TRICK_HOLD_MS), slides together toward
// its winner and fades (TRICK_COLLECT_MS), and the live state shows.
//
// Which deltas count (detectCompletedTrick):
//  - only a live delta: no previous payload (first load, refresh, a remount
//    after reconnect), another session, a history view or a replay on either
//    side, or a shorter history (undo, reset, a new deal) shows nothing;
//  - several tricks finished in one delta (a spectated all-bot table, a gap
//    while disconnected) show the LAST one only. A seated player's card is in
//    every trick, so between two of their decisions at most one trick
//    finishes; more than one means the viewer was not watching, and replaying
//    the skipped ones would put the board seconds behind the game for tricks
//    the score already counts. The served state carries only the last one.

/** A card of a finished trick and the seat that played it. */
export interface TrickCardAt<C = string, S = string> {
  card: C;
  seat: S;
}

/** A finished trick as a game serves it. */
export interface CompletedTrick<C = string, S = string> {
  /** Its number in the deal, from 1. Grows by one per finished trick. */
  index: number;
  /** In play order. */
  cards: ReadonlyArray<TrickCardAt<C, S>>;
  winner: S;
}

/** What the detector reads from one rendered payload. */
export interface TrickFeed<C = string, S = string> {
  /** The session (or deal) the payload belongs to. */
  session: string;
  /** Actions in the served history; it only grows during live play. */
  history: number;
  /** A history view or replay is on screen: the board is being scrubbed. */
  scrubbing: boolean;
  /** The last finished trick, or null before the first. */
  last: CompletedTrick<C, S> | null;
}

/** The viewer's trick-hold setting. */
export type TrickHold = 'off' | 'short' | 'normal';
/** Every trick hold. */
export const TRICK_HOLDS: readonly TrickHold[] = ['off', 'short', 'normal'];
/** The trick hold unless the viewer picks another. */
export const DEFAULT_TRICK_HOLD: TrickHold = 'normal';
/** Each trick hold's name in the settings. */
export const TRICK_HOLD_LABELS: Readonly<Record<TrickHold, string>> = {
  off: 'Off',
  short: 'Short',
  normal: 'Normal',
};
/** How long a finished trick stays fully visible, ms. 'off' shows no hold and
 *  no collect: the trick clears as soon as the next state arrives. */
export const TRICK_HOLD_MS: Readonly<Record<TrickHold, number>> = {
  off: 0,
  short: 500,
  normal: 900,
};
/** The collect: the cards slide together toward the winner and fade, ms. */
export const TRICK_COLLECT_MS = 350;

/** How far toward the winner's seat the cards gather (0: the trick's centre,
 *  1: the seat's centre). Short of the seat, so they read as taken toward the
 *  player, not dealt into the hand. */
export const COLLECT_REACH = 0.6;
/** Size at the end of the collect, x the card's. */
export const COLLECT_END_SCALE = 0.5;
/** Progress (0..1) at which the fade starts; fully transparent at 1. */
export const COLLECT_FADE_FROM = 0.25;

/** The ms a hold setting gives; throws on anything that is not one. */
export function trickHoldMs(hold: TrickHold): number {
  if (!Object.hasOwn(TRICK_HOLD_MS, hold)) {
    throw new Error(`trick hold: unknown setting ${JSON.stringify(hold)}`);
  }
  return TRICK_HOLD_MS[hold];
}

/** The finished trick a live delta just completed, or null (rules above). */
export function detectCompletedTrick<C, S>(
  prev: TrickFeed<C, S> | null | undefined,
  next: TrickFeed<C, S> | null | undefined,
): CompletedTrick<C, S> | null {
  if (!prev || !next) return null;
  if (prev.session !== next.session) return null;
  if (prev.scrubbing || next.scrubbing) return null;
  if (next.history < prev.history) return null;
  const last = next.last;
  if (!last) return null;
  if (!Number.isInteger(last.index) || last.index < 1) {
    throw new Error(
      `trick collect: a finished trick's index must be a positive integer, got ${JSON.stringify(last.index)}`,
    );
  }
  if (last.cards.length === 0) {
    throw new Error(
      `trick collect: finished trick ${String(last.index)} has no cards`,
    );
  }
  if (last.index <= (prev.last?.index ?? 0)) return null;
  return last;
}

/** Two feeds that would detect the same thing (render-to-render identity). */
export function sameTrickFeed<C, S>(
  a: TrickFeed<C, S> | null | undefined,
  b: TrickFeed<C, S> | null | undefined,
): boolean {
  if (!a || !b) return !a && !b;
  return (
    a.session === b.session &&
    a.history === b.history &&
    a.scrubbing === b.scrubbing &&
    (a.last?.index ?? 0) === (b.last?.index ?? 0)
  );
}

/** A point, px. */
export interface XY {
  x: number;
  y: number;
}

/** Where the cards gather: from their centroid, COLLECT_REACH of the way to
 *  the winner's seat. */
export function collectGather(cards: readonly XY[], seat: XY): XY {
  if (cards.length === 0) throw new Error('collectGather: no cards');
  const c = cards.reduce(
    (a, p) => ({x: a.x + p.x / cards.length, y: a.y + p.y / cards.length}),
    {x: 0, y: 0},
  );
  return {
    x: c.x + (seat.x - c.x) * COLLECT_REACH,
    y: c.y + (seat.y - c.y) * COLLECT_REACH,
  };
}

/** Ease-out cubic: fast away, settling at the end. No overshoot. */
export function easeOutCubic(p: number): number {
  const q = 1 - Math.min(1, Math.max(0, p));
  return 1 - q * q * q;
}

/** One frame of the collect for a card centred at `from`, at progress p in
 *  0..1: its centre, its scale and its opacity. */
export function collectFrame(
  p: number,
  from: XY,
  gather: XY,
): {x: number; y: number; scale: number; opacity: number} {
  const t = Math.min(1, Math.max(0, p));
  const e = easeOutCubic(t);
  const fade =
    t <= COLLECT_FADE_FROM
      ? 0
      : (t - COLLECT_FADE_FROM) / (1 - COLLECT_FADE_FROM);
  return {
    x: from.x + (gather.x - from.x) * e,
    y: from.y + (gather.y - from.y) * e,
    scale: 1 + (COLLECT_END_SCALE - 1) * e,
    opacity: 1 - fade,
  };
}
