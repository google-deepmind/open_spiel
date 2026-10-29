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

import {type JSX, type KeyboardEvent, type MouseEvent} from 'react';
import {
  PlayingCard,
  type Card as LibraryCard,
  type Rank as LibraryRank,
  type Suit as LibrarySuit,
} from '../../../cards/library/index.ts';
import {
  type CardCode,
  type Suit,
  type EuchreRank,
  HIDDEN_CARD,
} from '../types.ts';
import {useCardStyle} from '../euchre_gameplay_config_extras.tsx';
import {cardA11y} from './card_a11y.ts';

/** A card size preset. */
export type CardSize = 'sm' | 'md' | 'lg';

/** The properties of Card. */
export interface CardProps {
  /** Card code (e.g., "Ah", "Ks") or null/undefined for card back */
  card?: CardCode | null | undefined;
  /** Whether to show the face (true) or back (false) of the card */
  faceUp?: boolean | undefined;
  /** Whether the card is selected (raised with glow) */
  selected?: boolean | undefined;
  /** Whether the card can be clicked/played */
  playable?: boolean | undefined;
  /** Whether the card is disabled (greyed out) */
  disabled?: boolean | undefined;
  /** Card size */
  size?: CardSize | undefined;
  /** Rotation angle in degrees (for fan layouts) */
  rotation?: number | undefined;
  /** Click handler */
  onClick?: ((e: MouseEvent) => void) | undefined;
  /** Additional CSS class */
  className?: string | undefined;
  /** Inline styles for positioning */
  style?: React.CSSProperties | undefined;
  /** The card code on the element, for tests and the trick collect. */
  dataCardId?: string | undefined;
}

// Card aspect ratio is 5:7 (width:height), as the kernel decks draw it
const ASPECT_RATIO = 7 / 5;

// Size presets (width in pixels)
const SIZE_MAP: Record<CardSize, number> = {
  sm: 60,
  md: 80,
  lg: 100,
};

// Map our suit codes to the card library's
const SUIT_MAP: Record<Suit, LibrarySuit> = {
  h: 'H',
  d: 'D',
  c: 'C',
  s: 'S',
};

// Map our rank codes to the card library's (Euchre: 9-A only)
const RANK_MAP: Record<EuchreRank, LibraryRank> = {
  '9': '9',
  T: '10',
  J: 'J',
  Q: 'Q',
  K: 'K',
  A: 'A',
};

/**
 * Convert our card code to the card library's card
 * e.g., "Ah" -> {rank: 'A', suit: 'H'}, "Td" -> {rank: '10', suit: 'D'}
 */
function libraryCard(card: CardCode): LibraryCard {
  const rank = card[0] as EuchreRank;
  const suit = card[1] as Suit;
  return {rank: RANK_MAP[rank], suit: SUIT_MAP[suit]};
}

/**
 * Playing card component drawing the card library's deck (the Card style
 * setting: the simple deck by default, or the standard one)
 */
export function Card({
  card,
  faceUp = true,
  selected = false,
  playable = false,
  disabled = false,
  size = 'md',
  rotation = 0,
  onClick,
  className = '',
  style = {},
  dataCardId,
}: CardProps): JSX.Element {
  const [cardStyle] = useCardStyle();
  const width = SIZE_MAP[size];
  const height = width * ASPECT_RATIO;

  // Build CSS classes
  const classes = [
    'euchre-card',
    `euchre-card--${size}`,
    selected && 'euchre-card--selected',
    playable && !disabled && 'euchre-card--playable',
    disabled && 'euchre-card--disabled',
    !faceUp && 'euchre-card--back',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  // Combine styles
  const combinedStyle: React.CSSProperties = {
    width,
    height,
    transform: rotation !== 0 ? `rotate(${String(rotation)}deg)` : undefined,
    ...style,
  };

  const handleClick = (e: MouseEvent) => {
    if (!disabled && playable && onClick) {
      onClick(e);
    }
  };

  const interactive = playable && !disabled;
  const a11y = cardA11y(card, faceUp, interactive);
  // A role=button div answers Enter and Space as a button does.
  const handleKeyDown =
    interactive && onClick
      ? (e: KeyboardEvent<HTMLDivElement>) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onClick(e as unknown as MouseEvent);
          }
        }
      : undefined;

  // The face from the card library's deck; a card the viewer may not see
  // (or none) is the deck's back.
  const face =
    faceUp && card && card !== HIDDEN_CARD ? libraryCard(card) : null;

  return (
    <div
      className={classes}
      style={combinedStyle}
      data-card-id={dataCardId}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      {...a11y}
      tabIndex={interactive ? 0 : undefined}
    >
      <PlayingCard
        card={face}
        faceUp={face !== null}
        deck={cardStyle}
        width={width}
        a11y={null}
        noLift
      />
    </div>
  );
}
