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

// <TrickStack>: the stacked trick (trickArea.ts, stackBox) for a board that
// draws its cards as elements rather than on a <Table>: each card lies on
// the one before, a little toward the seat that played it, so the order of
// play reads from the stacking. The caller draws each card (a PlayingCard at
// `cardWidth`, with its own accessible name); this places them, in the order
// given, which is the order played. The box is the whole stack's, reserved
// even while the trick is empty, so the board never moves as it fills. No
// motion and no dimming of its own.
import type {JSX, ReactNode} from 'react';
import {itemAt} from '../checked.ts';
import {indexBoxes} from '../hand_measure.ts';
import {getDeck} from '../registry.ts';
import {useCardStyle} from '../style.ts';
import {
  PAIR_SEATS,
  SEAT_TOWARD,
  TRICK_SEATS,
  stackBox,
  type Pinwheel,
  type TrickSeat,
} from '../trick_area.ts';

/** One card of a trick stack and the seat that played it. */
export interface TrickStackCard {
  /** The seat that played it (one of `seats`). */
  seat: TrickSeat;
  /** A stable key (the card's id). */
  id: string;
  /** The card, drawn `cardWidth` wide. */
  children: ReactNode;
}

/** A trick stack: its cards and card width. */
export interface TrickStackProps {
  /** The cards of the trick IN PLAY ORDER. */
  cards: readonly TrickStackCard[];
  /** Card width, px. */
  cardWidth: number;
  /** The seats round the stack, in play order round the table (default
   *  PAIR_SEATS: across, then the near seat). */
  seats?: readonly TrickSeat[] | undefined;
  /** The pinwheel searched from (stackPinwheel's default). */
  reference?: Pinwheel | undefined;
  className?: string | undefined;
  testId?: string | undefined;
}

/** A trick's cards stacked toward their players. */
export function TrickStack({
  cards,
  cardWidth,
  seats = PAIR_SEATS,
  reference,
  className,
  testId,
}: TrickStackProps): JSX.Element {
  const style = useCardStyle();
  const deck = getDeck(style.deck);
  if (!(cardWidth > 0)) {
    throw new Error(
      `<TrickStack> cardWidth must be > 0, got ${String(cardWidth)}`,
    );
  }
  if (
    seats.length < 2 ||
    new Set(seats).size !== seats.length ||
    seats.some(k => !TRICK_SEATS.includes(k))
  ) {
    throw new Error(
      `<TrickStack> seats must be two or more distinct compass seats, got ${JSON.stringify(seats)}`,
    );
  }
  const seen = new Set<TrickSeat>();
  for (const c of cards) {
    if (!seats.includes(c.seat)) {
      throw new Error(
        `<TrickStack> card '${c.id}' is from seat '${c.seat}', not one of ${seats.join(', ')}`,
      );
    }
    if (seen.has(c.seat)) {
      throw new Error(`<TrickStack> two cards from seat '${c.seat}'`);
    }
    seen.add(c.seat);
  }
  const box = stackBox(
    seats.map(k => SEAT_TOWARD[k]),
    {cardHeight: 1 / deck.aspect, indexes: indexBoxes(deck, style)},
    reference,
  );
  return (
    <div
      className={className}
      data-testid={testId}
      data-trick-stack=""
      style={{
        position: 'relative',
        flex: 'none',
        width: box.width * cardWidth,
        height: box.height * cardWidth,
      }}
    >
      {cards.map((c, i) => {
        const p = itemAt(box.places, seats.indexOf(c.seat));
        return (
          <div
            key={c.id}
            data-seat={c.seat}
            data-play-order={i}
            style={{
              position: 'absolute',
              left: p.x * cardWidth,
              top: p.y * cardWidth,
              zIndex: i + 1,
              lineHeight: 0,
            }}
          >
            {c.children}
          </div>
        );
      })}
    </div>
  );
}
