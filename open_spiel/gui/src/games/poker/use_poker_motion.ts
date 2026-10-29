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

// Poker table motion: what changed between two states of the table
// moves, so a player can see what the bot just did without reading the
// action history.
//
// - Chips: a seat whose contribution grew sends a chip, labelled with the
//   amount, from its plate to the pot. Seats that acted in one update (the
//   viewer's bet and the bot's call arrive together: the server answers once
//   the bot has replied) send theirs in the order they acted, each a flight
//   and a rest after the one before, so the reply is seen as a reply.
// - The deal: a board card that was not there before is dealt onto the board
//   (the flop one card after another), after the chips have landed.
// - The reveal: a hole card that was not shown before (the showdown, a new
//   hand's deal) turns face up on its plate, after the board.
//
// Every duration is a base in the card library's flight time
// (CardStyle.motion.durationMs), scaled where it is used by scaleMs() at the
// viewer's Animation speed (shared/motion), the scale motionDuration()
// applies; at Instant or under reduced motion nothing moves (the end state
// shows at once). The first state a panel shows never animates: it is the
// table as it is.
//
// The numbers follow the chip: while a chip is in the air, its
// seat's Stack and In and the pot still read as before the bet (shownNumbers);
// each changes at that chip's own animationend, so the result never shows
// before the motion. With no chip in flight (Instant, reduced motion, the
// first state) the numbers are the state's at once.
//
// A motion still in flight when the next state arrives (the bot answers
// within one flight) is carried into the next motion until its own
// animationend: a chip keeps flying to the pot and a dealt or turned card
// keeps its class, so nothing snaps to its end mid-air.
import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from 'react';
import {motionDuration, useCardStyle} from '../../cards/library/index.ts';
import {useAnimationScale} from '../../motion/index.ts';
import {PLAY_REST_MS} from '../shared/use_staged_plays.ts';
import type {PokerUiState} from './poker_state.ts';

/** A bet or call in flight from its seat's plate to the pot. */
export interface ChipFlight {
  key: string;
  slot: number;
  amount: number;
  /** Delay before it leaves its seat, base ms (it waits unseen till then). */
  delayBaseMs: number;
  /** Start (seat) and travel (to the pot), table px. */
  x: number;
  y: number;
  dx: number;
  dy: number;
}

/** Times are BASE ms (Normal speed); scale them with scaleMs(base, scale). */
export interface PokerMotion {
  /** One card flight, base ms (the card library's). */
  flightBaseMs: number;
  /** The viewer's Animation speed multiplier, captured with the change. */
  scale: number;
  /** Board index -> delay before its deal starts, base ms. */
  boardDelays: ReadonlyMap<number, number>;
  /** `${slot}:${index}` -> delay before that hole card turns up, base ms. */
  revealDelays: ReadonlyMap<string, number>;
  chips: readonly ChipFlight[];
  /** Seats whose fold came with chips still to land: drawn as not yet
   *  folded until every chip has landed. */
  heldFolds: ReadonlySet<number>;
}

interface Snapshot {
  board: string[];
  hands: string[][];
  contributions: number[];
  folded: boolean[];
  /** The seat of each seat action in the history, in order. */
  actors: number[];
}

/** Whether nothing moves: every chip has landed and every card turned. */
export function motionSettled(motion: PokerMotion): boolean {
  return (
    motion.chips.length === 0 &&
    motion.boardDelays.size === 0 &&
    motion.revealDelays.size === 0
  );
}

/** What the table shows while chips fly: a seat's Stack and In, and
 *  the pot, as they were before every chip still in the air left its seat. */
export function shownNumbers(
  state: Pick<PokerUiState, 'pot' | 'seats'>,
  motion: PokerMotion,
): {
  pot: number;
  stack: (slot: number, stack: number) => number;
  contribution: (slot: number, value: number) => number;
} {
  // Never more than the seat has in: the render that brings a new hand
  // still holds the last hand's chips until the layout effect drops them
  // (before paint), and that render must not show a negative number.
  const inFlight = new Map<number, number>();
  for (const chip of motion.chips) {
    inFlight.set(chip.slot, (inFlight.get(chip.slot) ?? 0) + chip.amount);
  }
  const bySlot = new Map<number, number>();
  let total = 0;
  inFlight.forEach((amount, slot) => {
    const seat = state.seats.find(s => s.slot === slot);
    const pending = Math.min(amount, seat?.contribution ?? 0);
    bySlot.set(slot, pending);
    total += pending;
  });
  return {
    pot: state.pot - total,
    stack: (slot, stack) => stack + (bySlot.get(slot) ?? 0),
    contribution: (slot, value) => value - (bySlot.get(slot) ?? 0),
  };
}

const NO_MOTION: PokerMotion = {
  flightBaseMs: 0,
  scale: 0,
  boardDelays: new Map(),
  revealDelays: new Map(),
  chips: [],
  heldFolds: new Set(),
};

function snapshotOf(state: PokerUiState | null): Snapshot | null {
  return state
    ? {
        board: state.board.map(card => card.code),
        hands: state.seats.map(seat => seat.cards.map(card => card.code)),
        contributions: state.seats.map(seat => seat.contribution),
        folded: state.seats.map(seat => seat.hasFolded),
        actors: state.history.flatMap(entry =>
          entry.player !== null && entry.player >= 0 ? [entry.player] : [],
        ),
      }
    : null;
}

/** The centre of `el` in `table`'s padding-box px, corrected for any
 *  ancestor scale (the table's measured width over its layout width). */
function centreIn(table: HTMLElement, el: Element): {x: number; y: number} {
  const t = table.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  const scale =
    table.offsetWidth > 0 && t.width > 0 ? t.width / table.offsetWidth : 1;
  return {
    x: (r.left + r.width / 2 - t.left) / scale - table.clientLeft,
    y: (r.top + r.height / 2 - t.top) / scale - table.clientTop,
  };
}

/**
 * The order the seats' chips leave in: by when each seat first acted among
 * the new actions; a seat that put chips in without acting (none in these
 * games) first.
 */
function chipOrder(prev: Snapshot, next: Snapshot): (slot: number) => number {
  const acted = next.actors.slice(prev.actors.length);
  return slot => acted.indexOf(slot) + 1;
}

/** A chip for each seat whose contribution grew, from its plate to the pot. */
function chipFlights(
  prev: Snapshot,
  next: Snapshot,
  table: HTMLElement,
  pot: HTMLElement,
  counter: {current: number},
  stepBaseMs: number,
): ChipFlight[] {
  const chips: ChipFlight[] = [];
  const to = centreIn(table, pot);
  const order = chipOrder(prev, next);
  const movers = next.contributions
    .map((value, slot) => ({
      slot,
      amount: value - (prev.contributions[slot] ?? 0),
    }))
    .filter(({amount}) => amount > 0)
    .sort((a, b) => order(a.slot) - order(b.slot));
  movers.forEach(({slot, amount}, index) => {
    const plate = table.querySelector(`.poker-seat--${String(slot)}`);
    if (!plate) {
      throw new Error(`poker motion: no plate for seat ${String(slot)}`);
    }
    const from = centreIn(table, plate);
    counter.current += 1;
    chips.push({
      key: `chip-${String(counter.current)}`,
      slot,
      amount,
      delayBaseMs: index * stepBaseMs,
      x: from.x,
      y: from.y,
      dx: to.x - from.x,
      dy: to.y - from.y,
    });
  });
  return chips;
}

/**
 * Cards follow the chips, one flight apart; the flop's cards half a flight
 * apart from each other, and the hole cards turned up after the board.
 */
function cardDelays(
  prev: Snapshot,
  next: Snapshot,
  flightBaseMs: number,
  chips: readonly ChipFlight[],
): {boardDelays: Map<number, number>; revealDelays: Map<string, number>} {
  const step = flightBaseMs / 2;
  // After the last chip has landed.
  let at =
    chips.length > 0
      ? Math.max(...chips.map(chip => chip.delayBaseMs)) + flightBaseMs
      : 0;
  const boardDelays = new Map<number, number>();
  next.board.forEach((code, index) => {
    if (prev.board[index] === code) return;
    boardDelays.set(index, at);
    at += step;
  });
  if (boardDelays.size > 0) at += step;
  const revealDelays = new Map<string, number>();
  next.hands.forEach((cards, slot) => {
    cards.forEach((code, index) => {
      if (prev.hands[slot]?.[index] === code) return;
      revealDelays.set(`${String(slot)}:${String(index)}`, at);
    });
  });
  return {boardDelays, revealDelays};
}

/**
 * The next motion: what is still in flight carried (every chip that has
 * not landed, and a card's motion while that card is still the same card),
 * and the new change's motion.
 */
function nextMotion(
  current: PokerMotion,
  change: {
    prev: Snapshot;
    next: Snapshot;
    chips: ChipFlight[];
    newHand: boolean;
    flightBaseMs: number;
    scale: number;
  },
): PokerMotion {
  const {prev, next, chips, newHand, flightBaseMs, scale} = change;
  const {boardDelays, revealDelays} = cardDelays(
    prev,
    next,
    flightBaseMs,
    chips,
  );
  const folds = new Set(newHand ? [] : current.heldFolds);
  if (chips.length > 0) {
    next.folded.forEach((folded, slot) => {
      if (folded && prev.folded[slot] !== true) folds.add(slot);
    });
  }
  const carriedBoard = new Map<number, number>();
  current.boardDelays.forEach((delay, index) => {
    if (!boardDelays.has(index) && prev.board[index] === next.board[index]) {
      carriedBoard.set(index, delay);
    }
  });
  const carriedReveals = new Map<string, number>();
  current.revealDelays.forEach((delay, revealKey) => {
    const [slot = -1, index = -1] = revealKey.split(':').map(Number);
    if (
      !revealDelays.has(revealKey) &&
      prev.hands[slot]?.[index] === next.hands[slot]?.[index]
    ) {
      carriedReveals.set(revealKey, delay);
    }
  });
  return {
    flightBaseMs,
    scale,
    boardDelays: new Map([...carriedBoard, ...boardDelays]),
    revealDelays: new Map([...carriedReveals, ...revealDelays]),
    // A new hand drops the last hand's chips: their amounts are no
    // longer in this hand's numbers (shownNumbers).
    chips: newHand ? chips : [...current.chips, ...chips],
    heldFolds: folds,
  };
}

/** The callbacks a landed motion calls, each dropping it from the motion. */
function useLanded(
  setMotion: (update: (m: PokerMotion) => PokerMotion) => void,
): {
  chipLanded: (key: string) => void;
  boardLanded: (index: number) => void;
  revealLanded: (key: string) => void;
} {
  // Stable: a chip's animationcancel listener is keyed on it.
  const chipLanded = useCallback(
    (key: string): void => {
      setMotion(m => {
        const chips = m.chips.filter(chip => chip.key !== key);
        // The last chip down shows the folds that came with it.
        return {
          ...m,
          chips,
          heldFolds: chips.length === 0 ? new Set<number>() : m.heldFolds,
        };
      });
    },
    [setMotion],
  );
  const boardLanded = (index: number): void => {
    setMotion(m => {
      if (!m.boardDelays.has(index)) return m;
      const boardDelays = new Map(m.boardDelays);
      boardDelays.delete(index);
      return {...m, boardDelays};
    });
  };
  const revealLanded = (key: string): void => {
    setMotion(m => {
      if (!m.revealDelays.has(key)) return m;
      const revealDelays = new Map(m.revealDelays);
      revealDelays.delete(key);
      return {...m, revealDelays};
    });
  };
  return {chipLanded, boardLanded, revealLanded};
}

/**
 * The motion for the change from the previous state to `state`. `tableRef`
 * is the `.poker-table`, `potRef` the pot inside it; a seat plate is found
 * as `.poker-seat--<slot>`.
 */
export function usePokerMotion(
  state: PokerUiState | null,
  tableRef: RefObject<HTMLElement | null>,
  potRef: RefObject<HTMLElement | null>,
): {
  motion: PokerMotion;
  chipLanded: (key: string) => void;
  boardLanded: (index: number) => void;
  revealLanded: (key: string) => void;
} {
  const cardMotion = useCardStyle().motion;
  // The scale motionDuration() applies to a card flight (0 at Instant, and
  // under reduced motion unless the card style ignores it).
  const scale = useAnimationScale({reducedMotion: cardMotion.reducedMotion});
  const flightBaseMs = cardMotion.durationMs;
  const previous = useRef<Snapshot | null>(null);
  const counter = useRef(0);
  const [motion, setMotion] = useState<PokerMotion>(NO_MOTION);
  // Keyed on what the table shows, not on object identity: a re-derived
  // state with the same cards and chips is no change.
  const snapshot = snapshotOf(state);
  const key = JSON.stringify(snapshot);

  useLayoutEffect(() => {
    const next = snapshot;
    const prev = previous.current;
    previous.current = next;
    if (!next || !prev || motionDuration(cardMotion) <= 0) {
      setMotion(NO_MOTION);
      return;
    }
    // A new hand: contributions go back to the blinds or the ante. No chip
    // travels then; the new hole cards are still dealt.
    const newHand =
      prev.contributions.length !== next.contributions.length ||
      next.contributions.some(
        (value, slot) => value < (prev.contributions[slot] ?? 0),
      );
    const table = tableRef.current;
    const pot = potRef.current;
    const chips =
      !newHand && table && pot
        ? chipFlights(
            prev,
            next,
            table,
            pot,
            counter,
            flightBaseMs + PLAY_REST_MS,
          )
        : [];
    setMotion(current =>
      nextMotion(current, {prev, next, chips, newHand, flightBaseMs, scale}),
    );
    // Runs on a new table state only (its key): a speed change mid-flight
    // leaves the flight in progress alone.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return {motion, ...useLanded(setMotion)};
}
