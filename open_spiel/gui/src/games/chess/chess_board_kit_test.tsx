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
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {stubResizeObserver} from '../../testing/dom.ts';
import {type Rendered, render} from '../../testing/render.tsx';
import {
  ChessBoardFrame,
  PromotionPicker,
  chessStatusText,
  pieceImageUrl,
  pieceSetCss,
  squareColors,
} from './chess_board_kit.tsx';
import {readChessFrame} from './chess_board_slot.ts';
import {chessFrame} from './chess_test_fixtures.ts';
import {registerPieceSet} from './pieces/piece_sets.ts';
import type {PieceArtProps, PieceColor} from './pieces/piece_types.ts';
import {stauntonImageUrl} from './pieces/staunton.tsx';

let rendered: Rendered | null = null;
const unregister: Array<() => void> = [];

beforeEach(() => {
  stubResizeObserver();
});

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  for (const remove of unregister.splice(0)) remove();
  vi.unstubAllGlobals();
  document.documentElement.removeAttribute('style');
});

function NoArt(): JSX.Element {
  return <span />;
}

function shownFrame(): ReturnType<typeof readChessFrame> {
  const frame = chessFrame();
  if (frame.presentation === null) throw new Error('No presentation.');
  return readChessFrame(frame, frame.presentation);
}

describe('the chess board kit', () => {
  it('says whose move it is', () => {
    expect(chessStatusText(shownFrame())).toBe('Move 2 · White to move');
  });

  it('draws the status line over a square the size of the panel', () => {
    const sizes: number[] = [];
    rendered = render(
      <ChessBoardFrame frame={shownFrame()}>
        {size => {
          sizes.push(size);
          return <div data-testid="board" />;
        }}
      </ChessBoardFrame>,
    );
    expect(document.querySelector('[role="status"]')?.textContent).toBe(
      'Move 2 · White to move',
    );
    expect(document.querySelector('[data-testid="board"]')).not.toBeNull();
    expect(sizes.every(size => size >= 0)).toBe(true);
  });

  it('offers the four promotion pieces, and a press outside cancels', () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();
    const promoting: PieceColor = 'white';
    rendered = render(
      <PromotionPicker
        color={promoting}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
    );
    const buttons = [...document.querySelectorAll('button')];
    expect(buttons.map(button => button.getAttribute('aria-label'))).toEqual([
      'Promote to queen',
      'Promote to rook',
      'Promote to bishop',
      'Promote to knight',
    ]);
    buttons[3]?.click();
    expect(onSelect).toHaveBeenCalledWith('n');
    document
      .querySelector<HTMLElement>('[data-testid="promotion-cancel"]')
      ?.click();
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});

describe('the chess board kit: piece sets and squares', () => {
  it('gives the squares the theme colours', () => {
    expect(squareColors('green')).toEqual({
      light: 'var(--chess-green-light)',
      dark: 'var(--chess-green-dark)',
    });
  });

  it("styles each piece with the set's image", () => {
    unregister.push(
      registerPieceSet('kit_images', {
        label: 'Kit images',
        Piece: NoArt,
        imageUrl: ({role, color}: PieceArtProps) => `/p/${color}-${role}.svg`,
      }),
    );
    const css = pieceSetCss('kit_images', '.board');
    expect(css.split('\n')).toHaveLength(12);
    expect(css).toContain(
      '.board piece.black.knight{background-image:url("/p/black-knight.svg")}',
    );
  });

  it('refuses a set that draws no images, naming it', () => {
    unregister.push(
      registerPieceSet('kit_no_images', {label: 'No images', Piece: NoArt}),
    );
    expect(() => pieceImageUrl('kit_no_images', 'pawn', 'white')).toThrow(
      'Piece set "No images" draws no images; pick one that does.',
    );
  });

  it('draws a Staunton piece as an image in the page colours', () => {
    const style = document.documentElement.style;
    style.setProperty('--chess-piece-white', '#fafafa');
    style.setProperty('--chess-piece-white-line', '#111');
    style.setProperty('--chess-piece-black', '#222');
    style.setProperty('--chess-piece-black-line', '#eee');
    const url = stauntonImageUrl({role: 'queen', color: 'white'});
    expect(url.startsWith('data:image/svg+xml,')).toBe(true);
    const svg = decodeURIComponent(url.slice('data:image/svg+xml,'.length));
    expect(svg).toContain('fill="#fafafa"');
    expect(svg).not.toContain('var(');
  });

  it('refuses a Staunton image when the page sets no colours', () => {
    expect(() => stauntonImageUrl({role: 'pawn', color: 'black'})).toThrow(
      /sets no colour token --chess-piece-black/,
    );
  });
});
