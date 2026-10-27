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

import type {JSX} from 'react';

import {parseShown} from './card.ts';
import {backVariant, getDeck, useCardStyle} from './library/index.ts';
import {type Skin, STANDARD_SKIN} from './skins.ts';

import './cards.css';

/** What a card face is given. */
export interface CardFaceProps {
  /** The card's code, or null for a face-down card. */
  readonly code: string | null;
  readonly skin?: Skin;
  /** Makes the card a button that calls this when pressed. */
  readonly onSelect?: () => void;
  /** Draws the card raised, as the one chosen. */
  readonly selected?: boolean;
  /** Extra text for the card's name: what pressing it does. */
  readonly action?: string;
}

/**
 * The back a face-down card shows: the card library's, from the CardStyle in
 * effect (its deck and back, as the Card back picker sets them), so this
 * renderer's face-down cards match every other card table.
 */
function FaceDownArt(): JSX.Element {
  const style = useCardStyle();
  const deck = getDeck(style.deck);
  const picked = backVariant(style.back);
  const variant =
    picked !== undefined && deck.backs?.includes(picked) ? picked : undefined;
  return <>{deck.renderBack({variant})}</>;
}

/**
 * A playing card drawn in CSS: the rank and suit in the corner and the suit
 * in the middle, or the card library's back when it is face down. A card
 * that can be played is a button named by its card and what it does.
 */
export function CardFace({
  code,
  skin = STANDARD_SKIN,
  onSelect,
  selected = false,
  action,
}: CardFaceProps): JSX.Element {
  const card = parseShown(code);
  const name = card === null ? 'Face-down card' : skin.cardName(card);
  const label = action === undefined ? name : `${name}: ${action}`;
  const className =
    card === null ? 'playing-card playing-card--back' : 'playing-card';
  const tone = card === null ? undefined : skin.suitTone(card.suit);
  const glyph = card === null ? '' : skin.suitGlyph(card.suit);
  const face =
    card === null ? (
      <FaceDownArt />
    ) : (
      <>
        <span className="playing-card__index" aria-hidden="true">
          <span>{skin.rankLabel(card.rank)}</span>
          <span>{glyph}</span>
        </span>
        <span className="playing-card__pip" aria-hidden="true">
          {glyph}
        </span>
      </>
    );
  if (onSelect === undefined) {
    return (
      <span
        className={className}
        data-tone={tone}
        role="img"
        aria-label={label}
      >
        {face}
      </span>
    );
  }
  return (
    <button
      type="button"
      className={`${className} playing-card--playable`}
      data-tone={tone}
      data-selected={selected ? 'true' : undefined}
      aria-label={label}
      title={label}
      onClick={onSelect}
    >
      {face}
    </button>
  );
}
