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

import {type JSX, useMemo} from 'react';
import {Card, type CardSize} from './card.tsx';
import {type CardCode, type PlayerPosition} from '../types.ts';

/** Whether a hand's cards run across or down. */
export type FanOrientation = 'horizontal' | 'vertical';

/** The properties of CardFan. */
export interface CardFanProps {
  /** Array of card codes to display */
  cards: CardCode[];
  /** Fan orientation */
  orientation?: FanOrientation | undefined;
  /** Player position (for determining selection lift direction) */
  position?: PlayerPosition | undefined;
  /** Whether to show card faces */
  faceUp?: boolean | undefined;
  /** Set of selected card codes */
  selectedCards?: Set<string> | undefined;
  /** Set of playable card codes */
  playableCards?: Set<string> | undefined;
  /** Card size */
  cardSize?: CardSize | undefined;
  /** Maximum width/height of the fan in pixels */
  maxSpread?: number | undefined;
  /** Angle spread for the fan in degrees (0 = no angle, just overlap) */
  fanAngle?: number | undefined;
  /** Whether to enable selection interaction */
  interactive?: boolean | undefined;
  /** Callback when a card is clicked */
  onCardClick?: ((card: CardCode) => void) | undefined;
  /** Additional CSS class */
  className?: string | undefined;
  /** The hand's accessible name (a group: whose hand, how many cards). */
  label?: string | undefined;
}

// Card dimensions by size (matching Card.tsx SIZE_MAP with aspect ratio 7/5 = 1.4)
const CARD_WIDTHS: Record<CardSize, number> = {
  sm: 60,
  md: 80,
  lg: 100,
};

const CARD_HEIGHTS: Record<CardSize, number> = {
  sm: 84,
  md: 112,
  lg: 140,
};

/**
 * Calculate card positions for a fan layout
 */
function calculateFanPositions(
  cardCount: number,
  orientation: FanOrientation,
  cardSize: CardSize,
  maxSpread: number,
  fanAngle: number,
): {
  positions: Array<{x: number; y: number; rotation: number; zIndex: number}>;
  actualSpan: number;
} {
  if (cardCount === 0) return {positions: [], actualSpan: 0};

  const cardWidth = CARD_WIDTHS[cardSize];
  const cardHeight = CARD_HEIGHTS[cardSize];

  const isHorizontal = orientation === 'horizontal';

  // For horizontal: spread along width, for vertical: spread along height
  const overlapDimension = isHorizontal ? cardWidth : cardHeight;

  // For Euchre (5 cards), we can show more of each card
  const minVisible = overlapDimension * 0.4;

  // Calculate spacing to fit within maxSpread
  const idealSpacing = minVisible;
  const totalIdealSpan = overlapDimension + (cardCount - 1) * idealSpacing;

  let spacing = idealSpacing;
  if (totalIdealSpan > maxSpread && cardCount > 1) {
    spacing = (maxSpread - overlapDimension) / (cardCount - 1);
    spacing = Math.max(spacing, 15); // Minimum visibility
  }

  // Calculate total span
  const totalSpan = overlapDimension + (cardCount - 1) * spacing;
  const startOffset = -totalSpan / 2 + overlapDimension / 2;

  // Calculate angle per card (reduced for vertical)
  const effectiveAngle = isHorizontal ? fanAngle : fanAngle * 0.5;
  const anglePerCard = cardCount > 1 ? effectiveAngle / (cardCount - 1) : 0;
  const startAngle = -effectiveAngle / 2;

  const positions = Array.from({length: cardCount}, (_, i) => {
    const offset = startOffset + i * spacing;
    const rotation = startAngle + i * anglePerCard;

    // Subtle arc for visual appeal
    const arcOffset =
      fanAngle > 0 ? Math.sin((i / (cardCount - 1 || 1)) * Math.PI) * 5 : 0;

    if (isHorizontal) {
      return {
        x: offset,
        y: arcOffset,
        rotation,
        zIndex: i,
      };
    } else {
      // Vertical: offset along Y axis, no rotation (cards stay upright)
      return {
        x: arcOffset,
        y: offset,
        rotation: 0, // Keep cards upright for vertical layout
        zIndex: i,
      };
    }
  });

  return {positions, actualSpan: totalSpan};
}

// Selection lift offset by position (towards center)
const SELECTION_OFFSET: Record<PlayerPosition, {x: number; y: number}> = {
  bottom: {x: 0, y: -20}, // lift up
  top: {x: 0, y: 20}, // lift down
  left: {x: 20, y: 0}, // lift right
  right: {x: -20, y: 0}, // lift left
};

/** The fan position of card `index`; every card has one. */
function fanPosition<T>(positions: readonly T[], index: number): T {
  const pos = positions[index];
  if (pos === undefined) {
    throw new Error(`euchre: no fan position for card ${String(index)}`);
  }
  return pos;
}

/** Where one card of the fan sits, lifted toward the centre when selected. */
function fanCardStyle(
  pos: {x: number; y: number; rotation: number; zIndex: number},
  isSelected: boolean,
  position: PlayerPosition,
  cardWidth: number,
  cardHeight: number,
): React.CSSProperties {
  const lift = isSelected ? SELECTION_OFFSET[position] : {x: 0, y: 0};
  return {
    position: 'absolute',
    left: '50%',
    top: '50%',
    marginLeft: -cardWidth / 2,
    marginTop: -cardHeight / 2,
    transform: `translate(${String(pos.x + lift.x)}px, ${String(pos.y + lift.y)}px) rotate(${String(pos.rotation)}deg)`,
    zIndex: pos.zIndex,
  };
}

/**
 * Card fan component for displaying overlapping cards (optimized for 5-card Euchre hands)
 */
export function CardFan({
  cards,
  orientation = 'horizontal',
  position = 'bottom',
  faceUp = true,
  selectedCards,
  playableCards,
  cardSize = 'md',
  maxSpread = 300, // Smaller default for 5 cards
  fanAngle = 0,
  interactive = false,
  onCardClick,
  className = '',
  label,
}: CardFanProps): JSX.Element {
  const {positions, actualSpan} = useMemo(
    () =>
      calculateFanPositions(
        cards.length,
        orientation,
        cardSize,
        maxSpread,
        fanAngle,
      ),
    [cards.length, orientation, cardSize, maxSpread, fanAngle],
  );

  const isHorizontal = orientation === 'horizontal';
  const cardWidth = CARD_WIDTHS[cardSize];
  const cardHeight = CARD_HEIGHTS[cardSize];

  // Container dimensions - use actual span, not maxSpread
  const containerStyle: React.CSSProperties = {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: isHorizontal ? actualSpan : cardWidth,
    height: isHorizontal ? cardHeight : actualSpan,
  };

  const handleCardClick = (card: CardCode) => {
    if (interactive && onCardClick) {
      onCardClick(card);
    }
  };

  return (
    <div
      className={`euchre-card-fan euchre-card-fan--${orientation} ${className}`}
      style={containerStyle}
      role={label ? 'group' : undefined}
      aria-label={label}
    >
      {cards.map((card, index) => {
        const pos = fanPosition(positions, index);
        const isSelected = selectedCards?.has(card) ?? false;
        const isPlayable = playableCards?.has(card) ?? interactive;

        return (
          <Card
            key={`${card}-${String(index)}`}
            card={card}
            // A face-up card carries its code: a play flies from here.
            dataCardId={faceUp ? card : undefined}
            faceUp={faceUp}
            selected={isSelected}
            playable={isPlayable}
            size={cardSize}
            rotation={pos.rotation}
            onClick={() => {
              handleCardClick(card);
            }}
            className="euchre-card-fan__card"
            style={fanCardStyle(
              pos,
              isSelected,
              position,
              cardWidth,
              cardHeight,
            )}
          />
        );
      })}
    </div>
  );
}
