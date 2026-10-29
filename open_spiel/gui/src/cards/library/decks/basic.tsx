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

// `basic` ("Corner index + large pip"): a deck drawn in code. Rank and suit
// only, large and readable: a tall corner index so fanned piles stay legible,
// and one large suit in the middle that reads from across the table.
import type {JSX} from 'react';

import {clampIndexScale, scaleIndex} from '../index_scale.ts';
import {cardKey, isJoker, jokerColor, suitColor} from '../model.ts';
import {StarGlyph, SuitGlyph} from '../suit_glyph.tsx';
import {DEFAULT_SUIT_GLYPHS, type SuitGlyphSet} from '../suit_glyphs/sets.ts';
import type {
  Card,
  ColorScheme,
  Deck,
  FaceOptions,
  IndexGeometry,
} from '../types.ts';
import {CARD_BACK_LABELS, CARD_BACKS, CardBack} from './backs.tsx';

// Card in viewBox units; 250x350 = the standard 5:7 (2.5 x 3.5 in) card aspect.
const W = 250;
const H = 350;

// Corner index layout, in viewBox units. The index is one column: the rank
// is text anchored at its middle on INDEX_CX, with a bounded advance ("10" is
// squeezed to RANK_W_MAX by textLength), and the suit is a path with an exact
// extent centred on the same x, so the suit sits under the rank whatever its
// width.
const INDEX_CX = 41;
const RANK_FONT = 72;
const RANK_BASELINE = 68;
const RANK_W_MAX = 58;
const SUIT_SIZE = 44;
const SUIT_X = INDEX_CX - SUIT_SIZE / 2;
// The suit sits clear of the rank's descender: a Q's tail reaches about
// 0.2 em below the baseline in the wider fallback faces, so the suit starts
// at 82 to leave a gap under it.
const SUIT_Y = 82;

// The box that must stay visible: from the top of the tallest rank glyph
// (cap height ~0.72em above the baseline, padded to 0.8em) to the bottom of
// the suit shape, and across the column wide enough for "10" (the widest
// rank, RANK_W_MAX, centred on INDEX_CX; the suit is narrower) plus 4 units
// of padding either side.
const BOX = {
  x0: INDEX_CX - RANK_W_MAX / 2 - 4,
  y0: RANK_BASELINE - 0.8 * RANK_FONT,
  x1: INDEX_CX + RANK_W_MAX / 2 + 4,
  y1: SUIT_Y + SUIT_SIZE + 3,
};

/** The basic deck's corner index box, as fractions of the card. */
export const BASIC_INDEX: IndexGeometry = {
  left: BOX.x0 / W,
  top: BOX.y0 / H,
  width: (BOX.x1 - BOX.x0) / W,
  height: (BOX.y1 - BOX.y0) / H,
};

const FONT =
  "'Roboto Condensed', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif";

function Index({
  card,
  scheme,
  glyphs,
}: {
  card: Card;
  scheme: ColorScheme;
  glyphs: SuitGlyphSet;
}): JSX.Element {
  if (isJoker(card)) {
    const color = jokerColor(card.joker);
    return (
      <g>
        <text
          x={INDEX_CX}
          y={RANK_BASELINE}
          textAnchor="middle"
          fontSize={RANK_FONT * 0.62}
          fontFamily={FONT}
          fontWeight={700}
          style={{fill: color}}
          textLength={RANK_W_MAX}
          lengthAdjust="spacingAndGlyphs"
        >
          JKR
        </text>
        <StarGlyph x={SUIT_X} y={SUIT_Y} size={SUIT_SIZE} color={color} />
      </g>
    );
  }
  const color = suitColor(card.suit, scheme);
  const wide = card.rank === '10';
  return (
    <g>
      <text
        x={INDEX_CX}
        y={RANK_BASELINE}
        textAnchor="middle"
        fontSize={RANK_FONT}
        fontFamily={FONT}
        fontWeight={700}
        style={{fill: color}}
        {...(wide
          ? {textLength: RANK_W_MAX, lengthAdjust: 'spacingAndGlyphs' as const}
          : {})}
      >
        {card.rank}
      </text>
      <SuitGlyph
        suit={card.suit}
        x={SUIT_X}
        y={SUIT_Y}
        size={SUIT_SIZE}
        color={color}
        set={glyphs}
      />
    </g>
  );
}

const CENTRE = 110;

function Middle({
  card,
  scheme,
  glyphs,
}: {
  card: Card;
  scheme: ColorScheme;
  glyphs: SuitGlyphSet;
}): JSX.Element {
  if (isJoker(card)) {
    const color = jokerColor(card.joker);
    return (
      <g>
        <StarGlyph
          x={(W - CENTRE) / 2}
          y={(H - CENTRE) / 2 - 16}
          size={CENTRE}
          color={color}
        />
        {/* Face art, scaled with the card like its pips; the joker is read
            from its index ('JKR'), which the index legibility floor holds.
            data-face-art marks it for the text-floor probe. */}
        <text
          x={W / 2}
          y={H / 2 + 76}
          textAnchor="middle"
          fontSize={30}
          fontFamily={FONT}
          fontWeight={700}
          letterSpacing={4}
          style={{fill: color}}
          data-face-art
          aria-hidden="true"
        >
          JOKER
        </text>
      </g>
    );
  }
  return (
    <SuitGlyph
      suit={card.suit}
      x={(W - CENTRE) / 2}
      y={(H - CENTRE) / 2}
      size={CENTRE}
      color={suitColor(card.suit, scheme)}
      set={glyphs}
    />
  );
}

function Face({
  card,
  scheme,
  indexScale,
  glyphs,
}: {
  card: Card;
  scheme: ColorScheme;
  indexScale: number;
  glyphs: SuitGlyphSet;
}): JSX.Element {
  // The viewBox origin is the card's top-left corner, so scale(s) grows the
  // index about that corner; the rotated twin grows about the opposite one.
  const s = clampIndexScale(BASIC_INDEX, indexScale);
  const index = (
    <g transform={`scale(${String(s)})`}>
      <Index card={card} scheme={scheme} glyphs={glyphs} />
    </g>
  );
  return (
    <svg
      viewBox={`0 0 ${String(W)} ${String(H)}`}
      width="100%"
      height="100%"
      preserveAspectRatio="none"
      style={{display: 'block'}}
      data-deck="basic"
      data-card={cardKey(card)}
      data-scheme={scheme}
      data-suit-glyphs={glyphs}
    >
      {/* Square, unstroked: the container clips the corners and draws the
          edge. */}
      <rect width={W} height={H} style={{fill: 'var(--color-card-face)'}} />
      {index}
      <g transform={`rotate(180 ${String(W / 2)} ${String(H / 2)})`}>{index}</g>
      <Middle card={card} scheme={scheme} glyphs={glyphs} />
    </svg>
  );
}

/** The basic deck: a corner index of rank over suit, and a centre pip. */
export const basicDeck: Deck = {
  id: 'basic',
  label: 'Corner index + large pip',
  description:
    'Best for table piles and tricks: a big centre suit reads across the table, and its compact corner index packs the tightest fan.',
  provenance:
    'Drawn in code, suit shapes included (the CardStyle.suitGlyphs set).',
  aspect: W / H,
  index: BASIC_INDEX,
  rotatedTwin: true,
  indexAt(scale: number): IndexGeometry {
    return scaleIndex(BASIC_INDEX, scale);
  },
  renderFace(
    card: Card,
    {scheme, indexScale = 1, suitGlyphs = DEFAULT_SUIT_GLYPHS}: FaceOptions,
  ): JSX.Element {
    return (
      <Face
        card={card}
        scheme={scheme}
        indexScale={indexScale}
        glyphs={suitGlyphs}
      />
    );
  },
  renderBack({variant}: {variant?: string | undefined}): JSX.Element {
    return <CardBack deckId="basic" variant={variant} />;
  },
  backs: CARD_BACKS,
  backLabels: CARD_BACK_LABELS,
  drawsSuitGlyphs: true,
};
