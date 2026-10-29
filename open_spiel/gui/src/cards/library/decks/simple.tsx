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

// `simple` and `simple-corner`: a flat card, rank over suit, bold,
// suit-coloured, nothing else. No corner indexes, no pips. `simple` centres
// the block, so a fanned pile needs big offsets; `simple-corner` puts the same
// block at the top-left, which fans tightly, with its 180° twin at the
// bottom-right (fan-left/fan-up piles read the mirror of IndexGeometry).
//
// Every size is a function of the content scale (CardStyle.indexScale), and
// so is the IndexGeometry: the block IS the index.
import type {JSX} from 'react';

import {MIN_INDEX_SCALE} from '../index_scale.ts';
import {cardKey, isJoker, jokerColor, suitColor} from '../model.ts';
import {StarGlyph, SuitGlyph} from '../suit_glyph.tsx';
import {DEFAULT_SUIT_GLYPHS, type SuitGlyphSet} from '../suit_glyphs/sets.ts';
import type {Card, ColorScheme, Deck, IndexGeometry} from '../types.ts';
import {CARD_BACK_LABELS, CARD_BACKS, CardBack} from './backs.tsx';

// Card in viewBox units; 250x350 = the standard 5:7 (2.5 x 3.5 in) card
// aspect, as `basic`.
const W = 250;
const H = 350;

// The block at scale 1, in viewBox units: a 76px card sets the rank at 24px
// and the suit at 18px with a 4px gap; × 250/76 gives these.
const RANK_FONT = 80;
/** Cap height padded to 0.8em above the baseline, as `basic` measures it. */
const RANK_TOP = 0.8 * RANK_FONT;
/** "10" and "JKR" are squeezed to this advance by textLength. */
const RANK_W_MAX = 80;
const GAP = 12;
const SUIT = 60;
const BLOCK_W = Math.max(RANK_W_MAX, SUIT);
const BLOCK_H = RANK_TOP + GAP + SUIT;
/** Padding around the block in the index box. */
const PAD = 4;
/**
 * Distance of the corner block from the card's top and left edges. Kept
 * small so the corner index box ends near 0.6 card widths down (148 of 150
 * units): overlapped-row tables derive their step from it (TableClearance),
 * and a tight block keeps simple-corner's overlapped rows close to the
 * corner-index decks'.
 */
const CORNER_MARGIN = 8;

/** Where the simple block sits: centred, or in the corner with a twin. */
export type SimpleVariant = 'centre' | 'corner';

/**
 * Largest content scale a variant can draw. `centre`: the block stays on the
 * card (its height binds). `corner`: the block stays clear of its rotated
 * twin along at least one axis, the rule index_scale.ts applies to corner
 * indexes. Larger scales draw at this one, and indexAt reports the geometry
 * drawn.
 */
export function simpleMaxScale(variant: SimpleVariant): number {
  if (variant === 'centre') {
    return Math.min((H - 2 * PAD) / BLOCK_H, (W - 2 * PAD) / BLOCK_W);
  }
  const clearX = (W / 2 - CORNER_MARGIN - PAD) / BLOCK_W;
  const clearY = (H / 2 - CORNER_MARGIN - PAD) / BLOCK_H;
  return Math.max(clearX, clearY);
}

interface Block {
  /** Block's horizontal centre. */
  cx: number;
  /** Block's top edge (top of the rank's padded cap height). */
  top: number;
  scale: number;
}

/**
 * The scale actually drawn: `scale` clamped to [MIN_INDEX_SCALE,
 * simpleMaxScale], as clampIndexScale does for the corner-index decks.
 */
export function simpleScale(variant: SimpleVariant, scale: number): number {
  if (!Number.isFinite(scale)) {
    throw new Error(`indexScale must be a finite number, got ${String(scale)}`);
  }
  return Math.max(MIN_INDEX_SCALE, Math.min(simpleMaxScale(variant), scale));
}

function block(variant: SimpleVariant, requested: number): Block {
  const scale = simpleScale(variant, requested);
  if (variant === 'centre') {
    return {cx: W / 2, top: (H - BLOCK_H * scale) / 2, scale};
  }
  return {cx: CORNER_MARGIN + (BLOCK_W * scale) / 2, top: CORNER_MARGIN, scale};
}

/** The box that must stay visible, as fractions of the card, at `scale`. */
export function simpleIndex(variant: SimpleVariant, scale = 1): IndexGeometry {
  const b = block(variant, scale);
  const x0 = Math.max(0, b.cx - (BLOCK_W * b.scale) / 2 - PAD);
  const y0 = Math.max(0, b.top - PAD);
  const x1 = Math.min(W, b.cx + (BLOCK_W * b.scale) / 2 + PAD);
  const y1 = Math.min(H, b.top + BLOCK_H * b.scale + PAD);
  return {
    left: x0 / W,
    top: y0 / H,
    width: (x1 - x0) / W,
    height: (y1 - y0) / H,
  };
}

const FONT =
  "'Roboto Condensed', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif";

function Content({
  card,
  scheme,
  b,
  glyphs,
}: {
  card: Card;
  scheme: ColorScheme;
  b: Block;
  glyphs: SuitGlyphSet;
}): JSX.Element {
  const s = b.scale;
  const baseline = b.top + RANK_TOP * s;
  const suitX = b.cx - (SUIT * s) / 2;
  const suitY = baseline + GAP * s;
  const squeeze = {
    textLength: RANK_W_MAX * s,
    lengthAdjust: 'spacingAndGlyphs' as const,
  };
  if (isJoker(card)) {
    const color = jokerColor(card.joker);
    return (
      <g data-simple-block="">
        <text
          x={b.cx}
          y={baseline}
          textAnchor="middle"
          fontSize={RANK_FONT * 0.62 * s}
          fontFamily={FONT}
          fontWeight={700}
          style={{fill: color}}
          {...squeeze}
        >
          JKR
        </text>
        <StarGlyph x={suitX} y={suitY} size={SUIT * s} color={color} />
      </g>
    );
  }
  const color = suitColor(card.suit, scheme);
  return (
    <g data-simple-block="">
      <text
        x={b.cx}
        y={baseline}
        textAnchor="middle"
        fontSize={RANK_FONT * s}
        fontFamily={FONT}
        fontWeight={700}
        style={{fill: color}}
        {...(card.rank === '10' ? squeeze : {})}
      >
        {card.rank}
      </text>
      <SuitGlyph
        suit={card.suit}
        x={suitX}
        y={suitY}
        size={SUIT * s}
        color={color}
        set={glyphs}
      />
    </g>
  );
}

interface FaceProps {
  card: Card;
  scheme: ColorScheme;
  deckId: string;
  variant: SimpleVariant;
  scale: number;
  glyphs: SuitGlyphSet;
}

function Face({
  card,
  scheme,
  deckId,
  variant,
  scale,
  glyphs,
}: FaceProps): JSX.Element {
  const b = block(variant, scale);
  return (
    <svg
      viewBox={`0 0 ${String(W)} ${String(H)}`}
      width="100%"
      height="100%"
      preserveAspectRatio="none"
      style={{display: 'block'}}
      data-deck={deckId}
      data-card={cardKey(card)}
      data-scheme={scheme}
      data-index-scale={simpleScale(variant, scale)}
      data-suit-glyphs={glyphs}
    >
      {/* Square, unstroked: the container clips the corners and draws the
          edge. */}
      <rect width={W} height={H} style={{fill: 'var(--color-card-face)'}} />
      <Content card={card} scheme={scheme} b={b} glyphs={glyphs} />
      {variant === 'corner' && (
        <g transform={`rotate(180 ${String(W / 2)} ${String(H / 2)})`}>
          <Content card={card} scheme={scheme} b={b} glyphs={glyphs} />
        </g>
      )}
    </svg>
  );
}

function makeSimpleDeck(
  id: string,
  label: string,
  description: string,
  variant: SimpleVariant,
): Deck {
  return {
    id,
    label,
    description,
    provenance:
      'Drawn in code, suit shapes included (the CardStyle.suitGlyphs set).',
    aspect: W / H,
    index: simpleIndex(variant, 1),
    indexAt: scale => simpleIndex(variant, scale),
    // simple-corner draws the whole block again inside rotate(180) about the
    // centre (Face), so fan-left/fan-up read its mirror; the centred block
    // has no twin, and those piles refuse it.
    rotatedTwin: variant === 'corner',
    renderFace: (
      card,
      {scheme, indexScale = 1, suitGlyphs = DEFAULT_SUIT_GLYPHS},
    ) => (
      <Face
        card={card}
        scheme={scheme}
        deckId={id}
        variant={variant}
        scale={indexScale}
        glyphs={suitGlyphs}
      />
    ),
    renderBack: ({variant: back}) => <CardBack deckId={id} variant={back} />,
    backs: CARD_BACKS,
    backLabels: CARD_BACK_LABELS,
    drawsSuitGlyphs: true,
  };
}

/** The simple deck: rank over suit, centred. The id stays 'simple' (stored
 *  settings and games name it); the picker calls it by what it draws. */
export const simpleDeck = makeSimpleDeck(
  'simple',
  'Centred rank and suit',
  'Best for cards seen one at a time (a lone turned card, a deal shown card by card): rank over suit in the middle, no corner index, so not for fans.',
  'centre',
);
/** The simple deck with its block in the corner and a rotated twin. */
export const simpleCornerDeck = makeSimpleDeck(
  'simple-corner',
  'Large corner index',
  'Best for fanned hands: the biggest rank and suit in two corners, so each card in a fan reads at a glance.',
  'corner',
);
