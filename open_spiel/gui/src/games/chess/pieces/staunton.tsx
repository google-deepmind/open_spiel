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

// The default piece set: classic Staunton pieces drawn in code for this
// client (staunton_paths.ts). White pieces are white with dark lines, black
// pieces dark with light detail lines; the colours are tokens.

import type {JSX} from 'react';

import type {PieceArtProps, PieceColor, PieceSetModule} from './piece_types.ts';
import {
  STAUNTON_PATHS,
  STAUNTON_STROKE,
  STAUNTON_VIEW_BOX,
  type StauntonPaint,
} from './staunton_paths.ts';

/** A colour token's value as a paint: `var(--name)`, or the value itself. */
type Token = (name: string) => string;

function cssVar(name: string): string {
  return `var(--${name})`;
}

/**
 * The fill and stroke of a path painted `paint`, the colours from `token`;
 * null when not drawn.
 */
function paintProps(
  paint: StauntonPaint,
  color: PieceColor,
  token: Token = cssVar,
): {fill: string; stroke: string} | null {
  const fill = token(`chess-piece-${color}`);
  const line = token(`chess-piece-${color}-line`);
  const edge = token('chess-piece-white-line');
  switch (paint) {
    case 'body':
      return {fill, stroke: edge};
    case 'line':
      return {fill: 'none', stroke: line};
    case 'inner':
      return color === 'black' ? {fill: 'none', stroke: line} : null;
    case 'dot':
      return {fill: line, stroke: 'none'};
    case 'edge':
      return {fill: 'none', stroke: edge};
    default: {
      const unreachable: never = paint;
      throw new Error(`Unhandled paint: ${String(unreachable)}`);
    }
  }
}

/** One Staunton piece, filling its box. */
export function StauntonPiece({role, color}: PieceArtProps): JSX.Element {
  return (
    <svg
      className="chess-piece-art"
      viewBox={STAUNTON_VIEW_BOX}
      aria-hidden="true"
      focusable="false"
    >
      <g
        strokeWidth={STAUNTON_STROKE}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {STAUNTON_PATHS[role].map(([d, paint], index) => {
          const props = paintProps(paint, color);
          return props === null ? null : <path key={index} d={d} {...props} />;
        })}
      </g>
    </svg>
  );
}

/**
 * The value of the colour token `name` on the page, as `src/styles/tokens.css`
 * sets it.
 *
 * @throws Error when the page sets no such token.
 */
function pageToken(name: string): string {
  const value = window
    .getComputedStyle(document.documentElement)
    .getPropertyValue(`--${name}`)
    .trim();
  if (value === '') {
    throw new Error(`The page sets no colour token --${name} (tokens.css).`);
  }
  return value;
}

function escapeAttribute(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;');
}

/**
 * One Staunton piece as an SVG data URL, in the page's colour tokens as they
 * are when called (a board drawing images calls it again on a theme change).
 */
export function stauntonImageUrl({role, color}: PieceArtProps): string {
  const paths = STAUNTON_PATHS[role].flatMap(([d, paint]) => {
    const props = paintProps(paint, color, pageToken);
    return props === null
      ? []
      : [
          `<path d="${escapeAttribute(d)}" fill="${escapeAttribute(props.fill)}"` +
            ` stroke="${escapeAttribute(props.stroke)}"/>`,
        ];
  });
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${STAUNTON_VIEW_BOX}">` +
    `<g stroke-width="${String(STAUNTON_STROKE)}" stroke-linecap="round"` +
    ` stroke-linejoin="round">${paths.join('')}</g></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

/** The Staunton set's registry module. */
export const STAUNTON_PIECES: PieceSetModule = {
  label: 'Staunton',
  Piece: StauntonPiece,
  imageUrl: stauntonImageUrl,
};
