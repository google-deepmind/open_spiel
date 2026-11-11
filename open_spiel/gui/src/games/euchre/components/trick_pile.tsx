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

import {useRef, type JSX} from 'react';
import {
  playOrder,
  towardOffset,
  useCollectElements,
} from '../../../cards/library/index.ts';
import {Card, type CardSize} from './card.tsx';
import {type TrickCard, type PlayerPosition} from '../types.ts';
import {TRICK_SEAT, trickKey} from '../trick_seats.ts';

/** The properties of TrickPile. */
export interface TrickPileProps {
  /** Cards in the current trick with player positions */
  cards: TrickCard[];
  /** Which player led the trick (their card appears first/bottom) */
  leadPosition?: PlayerPosition | undefined;
  /** Card size */
  cardSize?: CardSize | undefined;
  /** Additional CSS class */
  className?: string | undefined;
  /** A finished trick being collected (shared/cards trickCollect): these
   *  cards slide toward the winner's seat and fade. */
  collect?:
    | {key: string; toward: PlayerPosition; durationMs: number}
    | null
    | undefined;
}

// The trick laid out toward the players (shared/cards trickArea 'toward'),
// in px from the centre: each card sits on its player's side.
const TOWARD_PX = {dx: 45, dy: 35};

/**
 * Center trick pile showing cards played this trick
 */
export function TrickPile({
  cards,
  leadPosition,
  cardSize = 'md',
  className = '',
  collect = null,
}: TrickPileProps): JSX.Element {
  const pile = useRef<HTMLDivElement | null>(null);
  useCollectElements(
    collect && {
      key: collect.key,
      durationMs: collect.durationMs,
      cards: () => [
        ...(pile.current?.querySelectorAll<HTMLElement>(
          '.euchre-trick-pile__card',
        ) ?? []),
      ],
      toward: () =>
        pile.current
          ?.closest('.euchre-table')
          ?.querySelector(`.euchre-player-section--${collect.toward}`) ?? null,
    },
  );
  // Determine lead position: use provided value, or infer from first card played
  const effectiveLeadPosition = leadPosition ?? cards[0]?.position ?? 'bottom';

  // Stack in play order: the lead at the bottom, then clockwise.
  const order = playOrder(TRICK_SEAT[effectiveLeadPosition]);
  const sortedCards = [...cards].sort(
    (a, b) =>
      order.indexOf(TRICK_SEAT[a.position]) -
      order.indexOf(TRICK_SEAT[b.position]),
  );

  return (
    <div ref={pile} className={`euchre-trick-pile ${className}`}>
      {sortedCards.map((trickCard, index) => {
        const offset = towardOffset(TRICK_SEAT[trickCard.position], TOWARD_PX);

        return (
          <Card
            key={trickKey(trickCard)}
            card={trickCard.card}
            dataCardId={trickCard.card}
            faceUp
            size={cardSize}
            className={`euchre-trick-pile__card euchre-trick-pile__card--${trickCard.position}`}
            style={{
              transform: `translate(${String(offset.x)}px, ${String(offset.y)}px)`,
              zIndex: index,
            }}
          />
        );
      })}
    </div>
  );
}
