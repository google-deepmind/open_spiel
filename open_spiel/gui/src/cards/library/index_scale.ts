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

// Corner-index scaling (CardStyle.indexScale). A deck that can redraw
// its index scales it about the card's top-left corner (and the rotated twin
// about the bottom-right), so the index box scales the same way: every
// IndexGeometry field × s. The scale is clamped so the box stays on the card
// and the two corner indexes never meet.
import type {Deck, IndexGeometry} from './types.ts';

/** The smallest corner-index scale. */
export const MIN_INDEX_SCALE = 0.5;
/** The largest corner-index scale. */
export const MAX_INDEX_SCALE = 3;

/** The largest scale at which `index` still fits: on the card, and clear of
 *  the rotated twin in the opposite corner along at least one axis. */
export function maxIndexScale(index: IndexGeometry): number {
  const farX = index.left + index.width;
  const farY = index.top + index.height;
  const onCard = Math.min(1 / farX, 1 / farY);
  const clearOfTwin = Math.max(0.5 / farX, 0.5 / farY);
  return Math.min(MAX_INDEX_SCALE, onCard, clearOfTwin);
}

/** `scale` clamped to what `index` can draw. */
export function clampIndexScale(index: IndexGeometry, scale: number): number {
  if (!Number.isFinite(scale)) {
    throw new Error(`indexScale must be a finite number, got ${String(scale)}`);
  }
  return Math.max(MIN_INDEX_SCALE, Math.min(maxIndexScale(index), scale));
}

/** `index` drawn at `scale` (clamped), scaled about the card's top-left corner. */
export function scaleIndex(index: IndexGeometry, scale: number): IndexGeometry {
  const s = clampIndexScale(index, scale);
  return {
    left: index.left * s,
    top: index.top * s,
    width: index.width * s,
    height: index.height * s,
  };
}

/** The index box a deck draws at `scale`: its own at 1, or fixed when it cannot redraw. */
export function deckIndex(
  deck: Pick<Deck, 'index' | 'indexAt'>,
  scale = 1,
): IndexGeometry {
  return deck.indexAt ? deck.indexAt(scale) : deck.index;
}

/** Which corner index a pile leaves visible: the drawn top-left one,
 *  or its 180° twin in the bottom-right. */
export type IndexCorner = 'top-left' | 'bottom-right';

/** `index` rotated 180° about the card's centre: the twin's box. */
export function mirrorIndex(index: IndexGeometry): IndexGeometry {
  return {
    left: 1 - index.left - index.width,
    top: 1 - index.top - index.height,
    width: index.width,
    height: index.height,
  };
}

/** `index` on a card turned a quarter anticlockwise, as fractions of the
 *  turned card's bounding box (width and height swap): the top-left index
 *  lands in the bottom-left. */
export function quarterTurnIndex(index: IndexGeometry): IndexGeometry {
  return {
    left: index.top,
    top: 1 - index.left - index.width,
    width: index.height,
    height: index.width,
  };
}

/** The deck's index box at `scale` in `corner`. The bottom-right twin exists
 *  only on decks that declare rotatedTwin; asking for it on any other throws. */
export function cornerIndex(
  deck: Pick<Deck, 'index' | 'indexAt' | 'rotatedTwin'> & {
    id?: string | undefined;
  },
  scale: number,
  corner: IndexCorner,
): IndexGeometry {
  const index = deckIndex(deck, scale);
  if (corner === 'top-left') return index;
  if (!deck.rotatedTwin) {
    throw new Error(
      `deck ${deck.id ?? '(unnamed)'} draws no bottom-right index twin (Deck.rotatedTwin); fan-left and fan-up piles cannot keep it legible`,
    );
  }
  return mirrorIndex(index);
}
