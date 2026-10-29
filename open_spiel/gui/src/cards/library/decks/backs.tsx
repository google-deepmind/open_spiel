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

// The card backs every built-in deck shares, drawn in code in a 250x350
// (5:7) box.
//
//   lattice (blue, red): a white border, a rounded field of fine crossing
//     lines turned 45° (white at 22% over the colour) and a white frame
//     inset on the field. The default, blue.
//   diamond (diamond-red, diamond-black): a white border round a field of
//     whole small diamonds, point to point, with no frame: the field is a
//     whole number of 7-unit tiles anchored at its corner, so no diamond is
//     cut at its edge, and the three cells at each corner are left empty,
//     which rounds it.
//   stripe (stripe-navy): 45° bands in two navies, edge to edge, painted as
//     one repeating gradient (a rotated pattern tile would leave a hairline
//     at every tile seam).
//
// Each card's paint server has its own id (the back id and a useId suffix):
// a url(#id) resolves to the first element with that id in the document, and
// one inside a display:none subtree is not painted, so a shared id would
// leave every other card showing that back unfilled.
import {useId, type JSX} from 'react';

const W = 250;
const H = 350;
/** The lattice's white border around the field. */
const BORDER = 12;
/** The lattice's white frame, inset on the field. */
const FRAME = 20;
/** The diamond tile, and the field's whole number of tiles each way. */
const DIAMOND_TILE = 7;
const DIAMOND_COLS = 32;
const DIAMOND_ROWS = 46;
/** The diamond field's corner: the field centred in the card. */
const DIAMOND_X = (W - DIAMOND_COLS * DIAMOND_TILE) / 2;
const DIAMOND_Y = (H - DIAMOND_ROWS * DIAMOND_TILE) / 2;
/** One stripe band's width across the stripe, in card units (two bands make
 *  one period). The CSS original's bands are a fixed 6 px; an SVG back
 *  scales with the card (the canvas motion path rasterises the same markup),
 *  so 12 units splits the difference: 7.2 px at 150 px wide, 4.6 px at 96. */
const STRIPE_BAND = 12;

type Pattern = 'lattice' | 'diamond' | 'stripe';

interface BackSpec {
  label: string;
  pattern: Pattern;
  /** The field's main colour, a token. */
  ink: string;
  /** The stripe pattern's second colour, a token. */
  ink2?: string;
}

// Insertion order is the picker's order; the first is the default.
const BACK_SPECS: Readonly<Record<string, BackSpec>> = {
  blue: {label: 'Blue lattice', pattern: 'lattice', ink: '--card-back-blue'},
  red: {label: 'Red lattice', pattern: 'lattice', ink: '--card-back-red'},
  'diamond-red': {
    label: 'Red diamonds',
    pattern: 'diamond',
    ink: '--card-back-diamond-red',
  },
  'diamond-black': {
    label: 'Black diamonds',
    pattern: 'diamond',
    ink: '--card-back-black',
  },
  'stripe-navy': {
    label: 'Navy stripes',
    pattern: 'stripe',
    ink: '--card-back-navy',
    ink2: '--card-back-navy-light',
  },
};

/** The back ids every built-in deck draws, in the picker's order. */
export const CARD_BACKS: readonly string[] = Object.keys(BACK_SPECS);
/** The back a deck draws when none is chosen. */
export const DEFAULT_CARD_BACK = 'blue';
/** Each back's name in the picker. */
export const CARD_BACK_LABELS: Readonly<Record<string, string>> =
  Object.fromEntries(CARD_BACKS.map(id => [id, BACK_SPECS[id]?.label ?? id]));

function token(name: string): string {
  return `var(${name})`;
}

function PatternDef({id, spec}: {id: string; spec: BackSpec}): JSX.Element {
  if (spec.pattern === 'lattice') {
    // A 14-unit tile turned 45°: one horizontal and one vertical line
    // through its middle, so the tiles join into a diagonal lattice.
    return (
      <pattern
        id={id}
        width={14}
        height={14}
        patternUnits="userSpaceOnUse"
        patternTransform="rotate(45)"
      >
        <rect width={14} height={14} style={{fill: token(spec.ink)}} />
        <path
          d="M0 7H14M7 0V14"
          strokeOpacity={0.22}
          strokeWidth={2}
          style={{stroke: token('--card-back-inset')}}
        />
      </pattern>
    );
  }
  if (spec.pattern === 'diamond') {
    // A 7-unit tile on white with a diamond whose points stop 0.1 short of
    // the tile's edge midpoints, so neighbours all but touch and the field
    // reads as solid colour at hand size. Anchored at the
    // field's corner, so the field holds whole tiles only.
    return (
      <pattern
        id={id}
        x={DIAMOND_X}
        y={DIAMOND_Y}
        width={DIAMOND_TILE}
        height={DIAMOND_TILE}
        patternUnits="userSpaceOnUse"
      >
        <rect
          width={DIAMOND_TILE}
          height={DIAMOND_TILE}
          style={{fill: token('--card-back-paper')}}
        />
        <path
          d="M3.5 0.1L6.9 3.5L3.5 6.9L0.1 3.5Z"
          style={{fill: token(spec.ink)}}
        />
      </pattern>
    );
  }
  // stripe: a gradient repeating along the up-right diagonal (as the CSS
  // original's repeating-linear-gradient(45deg)), so the bands fall to the
  // right; hard stops at the half make two flat bands per period.
  const period = (2 * STRIPE_BAND) / Math.SQRT2;
  return (
    <linearGradient
      id={id}
      gradientUnits="userSpaceOnUse"
      spreadMethod="repeat"
      x1={0}
      y1={period}
      x2={period}
      y2={0}
    >
      <stop offset={0} style={{stopColor: token(spec.ink)}} />
      <stop offset={0.5} style={{stopColor: token(spec.ink)}} />
      <stop offset={0.5} style={{stopColor: token(spec.ink2 ?? spec.ink)}} />
      <stop offset={1} style={{stopColor: token(spec.ink2 ?? spec.ink)}} />
    </linearGradient>
  );
}

/** The paper-coloured staircase over a diamond field's corner cells: three
 *  cells a corner (the corner and its two neighbours) left empty. */
function diamondCorners(): string {
  const t = DIAMOND_TILE;
  const x0 = DIAMOND_X;
  const y0 = DIAMOND_Y;
  const x1 = x0 + DIAMOND_COLS * t;
  const y1 = y0 + DIAMOND_ROWS * t;
  const n = (v: number) => String(v);
  // Each: from the corner, two cells along x, one cell in, back one, one
  // more cell in, back to the edge.
  return [
    `M${n(x0)} ${n(y0)}h${n(2 * t)}v${n(t)}h${n(-t)}v${n(t)}h${n(-t)}Z`,
    `M${n(x1)} ${n(y0)}h${n(-2 * t)}v${n(t)}h${n(t)}v${n(t)}h${n(t)}Z`,
    `M${n(x0)} ${n(y1)}h${n(2 * t)}v${n(-t)}h${n(-t)}v${n(-t)}h${n(-t)}Z`,
    `M${n(x1)} ${n(y1)}h${n(-2 * t)}v${n(-t)}h${n(t)}v${n(-t)}h${n(t)}Z`,
  ].join('');
}
const DIAMOND_CORNERS = diamondCorners();

/** The back of deck `deckId` in variant `variant` (default DEFAULT_CARD_BACK).
 *  Throws, naming the deck and its backs, on an unknown variant. */
export function CardBack({
  deckId,
  variant,
}: {
  deckId: string;
  variant?: string | undefined;
}): JSX.Element {
  const key = variant ?? DEFAULT_CARD_BACK;
  const spec = BACK_SPECS[key];
  if (spec === undefined) {
    throw new Error(
      `${deckId} deck has no back variant '${key}'; have: ${CARD_BACKS.join(', ')}`,
    );
  }
  // useId's characters are not all valid in an id a url() names.
  const uid = useId().replace(/[^\w-]/g, '');
  const pid = `card-back-${key}-${uid}`;
  return (
    <svg
      viewBox={`0 0 ${String(W)} ${String(H)}`}
      width="100%"
      height="100%"
      preserveAspectRatio="none"
      style={{display: 'block'}}
      data-deck={deckId}
      data-back={key}
    >
      <defs>
        <PatternDef id={pid} spec={spec} />
      </defs>
      {/* Square, unstroked: the container clips the corners and draws the
          edge. */}
      <rect width={W} height={H} style={{fill: token('--card-back-paper')}} />
      {spec.pattern === 'stripe' && (
        <rect width={W} height={H} fill={`url(#${pid})`} />
      )}
      {spec.pattern === 'diamond' && (
        <>
          <rect
            x={DIAMOND_X}
            y={DIAMOND_Y}
            width={DIAMOND_COLS * DIAMOND_TILE}
            height={DIAMOND_ROWS * DIAMOND_TILE}
            fill={`url(#${pid})`}
          />
          <path
            d={DIAMOND_CORNERS}
            style={{fill: token('--card-back-paper')}}
          />
        </>
      )}
      {spec.pattern === 'lattice' && (
        <>
          <rect
            x={BORDER}
            y={BORDER}
            width={W - 2 * BORDER}
            height={H - 2 * BORDER}
            rx={8}
            fill={`url(#${pid})`}
          />
          <rect
            x={FRAME}
            y={FRAME}
            width={W - 2 * FRAME}
            height={H - 2 * FRAME}
            rx={5}
            strokeOpacity={0.7}
            strokeWidth={2}
            style={{fill: 'none', stroke: token('--card-back-inset')}}
          />
        </>
      )}
    </svg>
  );
}
