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

// The hearts table's cards, per slot (heartsItems), and who may do what at
// the table (rulesOf): the one source for the items and the policy.
import {
  playableTags,
  raisesPlayable,
  trickAreaItems,
  TRICK_LAYOUTS,
  type TableCard,
  type TrickLayout,
  type TrickSeat,
} from '../../../cards/library/index.ts';
import {SEAT_SLOT} from '../hearts_layout.ts';
import {
  getPointValue,
  toSharedCard,
  type CardCode,
  type PlayerPosition,
} from '../types.ts';
import {
  PASS_COUNT,
  type PassFlight,
  type SeatState,
  type TableLayoutProps,
} from './table_types.ts';

/**
 * The id of the `i`th back of a hidden hand. Backs are positional
 * (`${slot}-${i}`), so a card passed into a hand of backs would have no id
 * to fly to, and a received card none to fly from. Once a pass has left
 * your hand, the first PASS_COUNT backs of the seat it went to take new ids
 * (each with `was` naming the card you passed, so it flies there and turns
 * face down), and so do those of the seat that passed to you once your new
 * cards have arrived (each new card's `was` names one of that seat's old
 * backs, so it flies from there and turns face up). The ids hold for the
 * rest of the hand: a trick card's `was` names its back by this same rule.
 */
function backId(
  slot: string,
  seat: PlayerPosition,
  i: number,
  pass: PassFlight | null | undefined,
): string {
  const renamed =
    !!pass &&
    i < PASS_COUNT &&
    (seat === pass.to || (seat === pass.from && pass.received.length > 0));
  return renamed ? `${slot}-passed-${String(i)}` : `${slot}-${String(i)}`;
}

/** A seat as the trick area names it (the table's compass). */
export const COMPASS: Readonly<Record<PlayerPosition, TrickSeat>> = {
  top: 'n',
  right: 'e',
  bottom: 's',
  left: 'w',
};

/** The hand slots' seats ('toward' trick slots too: a card dropped there
 *  is its seat's). The overlapping trick's one slot is no seat's. */
export const SEAT_OF_SLOT = new Map<string, PlayerPosition>(
  (
    Object.entries(SEAT_SLOT) as Array<
      [PlayerPosition, (typeof SEAT_SLOT)[PlayerPosition]]
    >
  ).flatMap(([seat, s]) => [
    [s.hand, seat],
    [s.trick, seat],
  ]),
);

/** " (13 points)" for a card that scores, "" for one that does not. */
export function points(code: CardCode): string {
  const n = getPointValue(code);
  return n === 0 ? '' : n === 1 ? ' (1 point)' : ` (${String(n)} points)`;
}

/** The trick layout the table draws: each card toward its player. */
export const DEFAULT_HEARTS_TRICK_LAYOUT: TrickLayout = 'toward';

/** The trick layout drawn: the one asked for, or the table's default. */
export function trickLayoutOf(
  p: Pick<TableLayoutProps, 'trickLayout'>,
): TrickLayout {
  const t = p.trickLayout ?? DEFAULT_HEARTS_TRICK_LAYOUT;
  if (!TRICK_LAYOUTS.includes(t)) {
    throw new Error(`hearts: unknown trick layout ${JSON.stringify(t)}`);
  }
  return t;
}

/** Who may do what at the table. */
export interface TableRules {
  readonly isPassPhase: boolean;
  /** A hand takes gestures: yours, or (analysis, open hands) the one on turn. */
  readonly handActive: (seat: PlayerPosition) => boolean;
  /** On turn, only playable cards act; off turn, only pass selection does. */
  readonly canAct: (seat: PlayerPosition, code: CardCode) => boolean;
  readonly faceUp: (seat: PlayerPosition) => boolean;
  readonly isChosen: (code: CardCode) => boolean;
  /** Your card on its way to the server. */
  readonly isSending: (code: CardCode) => boolean;
}

/** Who may do what at the table (the one source for items and policy). */
export function rulesOf(p: SeatState): TableRules {
  const isPassPhase = p.phase === 'pass';
  const open = !!p.showOpponentCards;
  const handActive = (seat: PlayerPosition): boolean =>
    !!p.onCardClick &&
    (seat === 'bottom' ||
      (!!p.allHandsInteractive && open && p.currentPlayer === seat));
  return {
    isPassPhase,
    handActive,
    canAct: (seat, code) =>
      handActive(seat) &&
      (p.currentPlayer === seat
        ? (p.playableCards?.has(code) ?? true)
        : isPassPhase),
    faceUp: seat => seat === 'bottom' || open,
    isChosen: code => p.selectedCards?.has(code) ?? false,
    isSending: code => !!p.sendingCard && p.sendingCard === code,
  };
}

/** The cards of a hand the viewer may see, each with its tags. */
function openHand(
  p: SeatState,
  r: TableRules,
  seat: PlayerPosition,
  hand: readonly CardCode[],
  o: {raise: boolean; from: PlayerPosition | null},
): TableCard[] {
  const pass = p.passFlight;
  const passing =
    r.handActive(seat) && (seat === 'bottom' || p.currentPlayer === seat);
  const onTurn =
    r.handActive(seat) &&
    p.currentPlayer === seat &&
    !r.isPassPhase &&
    (p.playableCards?.size ?? 0) > 0;
  const raise =
    o.raise && onTurn && raisesPlayable(hand, code => r.canAct(seat, code));
  return hand.map(code => {
    const acts = r.canAct(seat, code);
    const tags = [
      ...((passing && r.isChosen(code)) ||
      (seat === 'bottom' && r.isSending(code))
        ? ['selected']
        : []),
      ...playableTags(raise, acts),
    ];
    const k = o.from && pass ? pass.received.indexOf(code) : -1;
    const item: TableCard = {
      id: code,
      card: toSharedCard(code),
      interactive: acts,
      tags,
    };
    return k >= 0 && o.from
      ? {...item, was: `${SEAT_SLOT[o.from].hand}-${String(k)}`}
      : item;
  });
}

/**
 * Cards per table slot (pure). A card keeps its id from hand to trick, so it
 * flies there; a hidden opponent's card arrives with `was` naming the back it
 * replaces, so it flies AND turns face up. `trick: false` when the centre
 * (pass button, game over) replaces the trick. On your turn, when only some
 * cards may go, those stand up (cards/library playable.ts); the rest keep
 * full contrast, never dimmed, and nothing stands up on a free lead.
 * `raise: false` stands nothing up: a card that may not be played is still
 * refused. `trickLayout` is the layout the recipe was re-laid with
 * (trickLayoutRecipe): required, so items and slots cannot disagree. A card
 * being played (`sendingCard`) is lifted as a chosen card is, the
 * move-pending cue; a pass that has left your hand (`passFlight`) flies
 * (backId).
 */
export function heartsItems(
  p: SeatState,
  o: {trick: boolean; raise?: boolean; trickLayout: TrickLayout},
): Record<string, TableCard[]> {
  const raiseOn = o.raise ?? true;
  const r = rulesOf(p);
  const bySeat = new Map(p.players.map(x => [x.position, x] as const));
  const hiddenSeat = (seat: PlayerPosition): boolean =>
    !r.faceUp(seat) || !!bySeat.get(seat)?.hidden;
  // Backs are renamed for a pass only where they are backs: an open hand's
  // cards keep their codes, so a passed card flies there by its id.
  const pass = p.passFlight;
  const backOf = (seat: PlayerPosition, i: number): string =>
    backId(SEAT_SLOT[seat].hand, seat, i, hiddenSeat(seat) ? pass : null);
  const items: Record<string, TableCard[]> = {};
  for (const [seat, player] of bySeat) {
    const slot = SEAT_SLOT[seat].hand;
    // A hand the server withheld is backs, even the bottom seat's: a
    // spectator has no seat whose cards it may see.
    if (hiddenSeat(seat)) {
      items[slot] = player.hand.map((_, i) => {
        const id = backOf(seat, i);
        // The card you passed that this back now is: it flies here face down.
        const was =
          seat === pass?.to && id !== `${slot}-${String(i)}`
            ? pass.sent[i]
            : undefined;
        return was === undefined ? {id, card: null} : {id, card: null, was};
      });
      continue;
    }
    // Your received cards fly in from the hidden seat that passed them: each
    // names one of that seat's backs as they were before it was renamed.
    const from =
      pass && seat === 'bottom' && hiddenSeat(pass.from) ? pass.from : null;
    items[slot] = openHand(p, r, seat, player.hand, {raise: raiseOn, from});
  }
  if (!o.trick) return items;
  // In play order (trickPile is), so an overlapping trick stacks as played.
  const trick = trickAreaItems(
    o.trickLayout,
    p.trickPile.map(t => {
      const left = bySeat.get(t.position)?.hand.length;
      const was =
        !hiddenSeat(t.position) || left === undefined
          ? undefined
          : backOf(t.position, left);
      const item: TableCard = {
        id: t.card,
        card: toSharedCard(t.card),
        interactive: false,
        ...(was === undefined ? {} : {was}),
      };
      return {seat: COMPASS[t.position], item};
    }),
  );
  return {...items, ...trick};
}
