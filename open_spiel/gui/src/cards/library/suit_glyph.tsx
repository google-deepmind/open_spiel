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

// Suit shapes drawn as paths in a 100x100 box, so their extent is exact (a
// text glyph's extent depends on whichever font the browser picked). The
// shapes come from a suit-glyph set (suitGlyphs/sets.ts, CardStyle.suitGlyphs).
import type {JSX} from 'react';
import type {Suit} from './types.ts';
import {
  DEFAULT_SUIT_GLYPHS,
  suitShapes,
  type SuitGlyphSet,
} from './suit_glyphs/sets.ts';

/** A suit shape filling the square (x, y, size) in the parent SVG's units,
 *  from `set` (default DEFAULT_SUIT_GLYPHS). */
export function SuitGlyph({
  suit,
  x,
  y,
  size,
  color,
  set = DEFAULT_SUIT_GLYPHS,
}: {
  suit: Suit;
  x: number;
  y: number;
  size: number;
  color: string;
  set?: SuitGlyphSet | undefined;
}): JSX.Element {
  return (
    <svg
      x={x}
      y={y}
      width={size}
      height={size}
      viewBox="0 0 100 100"
      data-suit={suit}
      data-suit-set={set}
    >
      {suitShapes(set, suit).map((s, i) =>
        'd' in s ? (
          <path key={i} d={s.d} style={{fill: color}} />
        ) : (
          <circle key={i} cx={s.cx} cy={s.cy} r={s.r} style={{fill: color}} />
        ),
      )}
    </svg>
  );
}

const STAR = 'M50 3 61 37H97L68 58 79 93 50 72 21 93 32 58 3 37H39Z';

/** A five-pointed star filling the square (x, y, size): the joker's mark. */
export function StarGlyph({
  x,
  y,
  size,
  color,
}: {
  x: number;
  y: number;
  size: number;
  color: string;
}): JSX.Element {
  return (
    <svg x={x} y={y} width={size} height={size} viewBox="0 0 100 100">
      <path d={STAR} style={{fill: color}} />
    </svg>
  );
}
