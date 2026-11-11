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

import {act} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {History} from '../../actions/history.tsx';
import type {Frame} from '../../contract/contract.ts';
import {RENDERERS} from '../../renderers/registry.ts';
import {ANALYSIS_VIEW_SEAT_KEY} from '../../game/bottom_seat.ts';
import {stubResizeObserver} from '../../testing/dom.ts';
import {session} from '../../testing/frames.ts';
import {enter, labelled} from '../../testing/input.ts';
import {type Rendered, render} from '../../testing/render.tsx';
import {SetViewSetting} from '../../testing/view_settings.tsx';
import {Perspective} from '../../view/perspective.tsx';
import {ChessConfigSection} from './chess_config_section.tsx';
import {ChessRenderer} from './chess_renderer.tsx';
import {
  chessFrame,
  chessHistory,
  chessPresentation,
} from './chess_test_fixtures.ts';

const SIZE = 400;
const SQUARE = SIZE / 8;
let rendered: Rendered | null = null;

beforeEach(() => {
  stubResizeObserver();
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(SIZE);
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(SIZE);
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(
    new DOMRect(0, 0, SIZE, SIZE),
  );
  HTMLElement.prototype.setPointerCapture = vi.fn();
});

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

function show(
  shown: Frame,
  play: ((action: number) => void) | null = vi.fn(),
): Rendered {
  const {presentation} = shown;
  if (presentation === null) throw new Error('No presentation.');
  rendered = render(
    <ChessRenderer frame={shown} presentation={presentation} play={play} />,
  );
  return rendered;
}

function board(): HTMLElement {
  const found = document.querySelector('.cg-board');
  if (!(found instanceof HTMLElement)) throw new Error('No board.');
  return found;
}

// The centre of `square` on a board seen from white's side.
function centre(square: string): {clientX: number; clientY: number} {
  const file = 'abcdefgh'.indexOf(square[0] ?? '');
  const rank = Number(square[1]);
  return {
    clientX: (file + 0.5) * SQUARE,
    clientY: (8 - rank + 0.5) * SQUARE,
  };
}

function pointer(type: string, square: string): void {
  act(() => {
    board().dispatchEvent(
      new MouseEvent(type, {bubbles: true, button: 0, ...centre(square)}),
    );
  });
}

/** Lets a sent move's promise settle, as the board resyncs after it. */
async function settle(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

function pieceOn(square: string): Element | null {
  return document.querySelector(`.piece[data-square="${square}"]:not(.ghost)`);
}

describe('the chess board', () => {
  it('draws the position, the last move and the strips', () => {
    show(chessFrame());
    expect(document.querySelectorAll('.cg-board .piece')).toHaveLength(32);
    expect(pieceOn('e4')?.className).toContain('pawn white');
    expect(
      [...document.querySelectorAll('.square.last-move')].map(square =>
        square.getAttribute('data-square'),
      ),
    ).toEqual(['d7', 'd5']);
    expect(
      document
        .querySelector('[data-testid="chess-strip-white"]')
        ?.getAttribute('aria-label'),
    ).toBe('White, Human, to move');
    expect(
      document.querySelector('[data-testid="chess-status"]')?.textContent,
    ).toBe('Move 2');
    expect(document.querySelectorAll('.coords .coord')).toHaveLength(16);
  });

  it('plays a move pressed on two squares', async () => {
    const play = vi.fn();
    show(chessFrame(), play);
    pointer('pointerdown', 'e4');
    pointer('pointerup', 'e4');
    expect(
      document.querySelector('.square.selected')?.getAttribute('data-square'),
    ).toBe('e4');
    expect(
      [...document.querySelectorAll('.square.move-dest')]
        .map(square => square.getAttribute('data-square'))
        .sort(),
    ).toEqual(['d5', 'e5']);
    expect(
      document
        .querySelector('.square.move-dest.oc')
        ?.getAttribute('data-square'),
    ).toBe('d5');
    pointer('pointerdown', 'd5');
    expect(play).toHaveBeenCalledWith(10);
    // The piece stays where it was put while the move is with the server.
    expect(pieceOn('d5')?.className).toContain('pawn white');
    expect(pieceOn('e4')).toBeNull();
    await settle();
  });
});

describe('the chess board: gestures', () => {
  it('plays a move dragged to its square', async () => {
    const play = vi.fn();
    show(chessFrame(), play);
    pointer('pointerdown', 'g1');
    pointer('pointermove', 'g2');
    expect(document.querySelector('.piece.ghost')).not.toBeNull();
    expect(document.querySelector('.piece.dragging')).not.toBeNull();
    pointer('pointermove', 'f3');
    pointer('pointerup', 'f3');
    expect(play).toHaveBeenCalledWith(12);
    expect(pieceOn('f3')?.className).toContain('knight white');
    await settle();
  });

  it('puts a piece dropped off its moves back', () => {
    const play = vi.fn();
    show(chessFrame(), play);
    pointer('pointerdown', 'g1');
    pointer('pointermove', 'g4');
    pointer('pointerup', 'g4');
    expect(play).not.toHaveBeenCalled();
    expect(pieceOn('g1')?.className).toContain('knight white');
  });

  it("cannot pick up the other side's piece or move when it cannot play", () => {
    const play = vi.fn();
    show(chessFrame(), play);
    pointer('pointerdown', 'd5');
    expect(document.querySelector('.square.selected')).toBeNull();
    rendered?.unmount();
    show(chessFrame(), null);
    pointer('pointerdown', 'e4');
    expect(document.querySelector('.square.selected')).toBeNull();
    expect(document.querySelector('.cg-wrap.view-only')).not.toBeNull();
  });
});

describe('the chess board: promotion, check and seats', () => {
  it('asks which piece a pawn becomes, and plays the choice', async () => {
    const presentation = chessPresentation(
      {fen: '4k3/1P6/8/8/8/8/8/4K3 w - - 0 1', chessLastMove: null},
      [
        {id: 20, from: 'b7', to: 'b8', promotion: 'q'},
        {id: 21, from: 'b7', to: 'b8', promotion: 'n'},
        {id: 22, from: 'b7', to: 'b8', promotion: 'r'},
        {id: 23, from: 'b7', to: 'b8', promotion: 'b'},
      ],
    );
    const play = vi.fn();
    show(chessFrame({history: []}, presentation), play);
    pointer('pointerdown', 'b7');
    pointer('pointerup', 'b7');
    pointer('pointerdown', 'b8');
    expect(play).not.toHaveBeenCalled();
    const knight = document.querySelector('[aria-label="Promote to Knight"]');
    if (!(knight instanceof HTMLButtonElement)) throw new Error('No choice.');
    act(() => {
      knight.click();
    });
    expect(play).toHaveBeenCalledWith(21);
    await settle();
    expect(document.querySelector('.promotion-overlay')).toBeNull();
  });

  it('marks the king in check and names the result', () => {
    const fen = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3';
    const presentation = chessPresentation(
      {
        fen,
        chessCheck: 'white',
        chessLastMove: {from: 'd8', to: 'h4'},
        chessOutcome: {checkmate: true, stalemate: false, threefold: false},
      },
      [],
    );
    show(
      chessFrame(
        {
          status: 'terminal',
          to_move: {kind: 'terminal', seat: null},
          returns: [1, -1],
          legal_actions: [],
          history: chessHistory(['f3', 'e5', 'g4', 'Qh4#']),
        },
        presentation,
      ),
      null,
    );
    expect(
      document.querySelector('.square.check')?.getAttribute('data-square'),
    ).toBe('e1');
    expect(
      document.querySelector('[data-testid="chess-status"]')?.textContent,
    ).toBe('0-1 · Black wins by checkmate');
  });

  it('turns the board to a black seat', () => {
    show(chessFrame({viewer: {kind: 'seat', seat: 0}}));
    expect(document.querySelector('.cg-wrap.orientation-black')).not.toBeNull();
    expect(
      document
        .querySelector('[data-testid="chess-strip-black"]')
        ?.getAttribute('aria-label'),
    ).toBe('Black, Human (you)');
  });
});

describe('the chess board in analysis', () => {
  it('turns the board in analysis to the seat View › Perspective chooses', () => {
    const shown = chessFrame({viewer: {kind: 'oracle', seat: null}});
    const {presentation} = shown;
    if (presentation === null) throw new Error('No presentation.');
    rendered = render(
      <>
        <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={null} />
        <Perspective viewer={{kind: 'oracle', seat: null}} numSeats={2} />
        <ChessRenderer
          frame={shown}
          presentation={presentation}
          play={vi.fn()}
        />
      </>,
    );
    const {container} = rendered;
    enter(labelled(container, 'Perspective'), '1');
    expect(document.querySelector('.cg-wrap.orientation-white')).not.toBe(null);
    // Seat 0 plays Black.
    enter(labelled(container, 'Perspective'), '0');
    expect(document.querySelector('.cg-wrap.orientation-black')).not.toBe(null);
    enter(labelled(container, 'Perspective'), '1');
    expect(document.querySelector('.cg-wrap.orientation-white')).not.toBe(null);
  });

  it('rests on Seat 0 drawn from White until a seat is chosen', () => {
    const shown = chessFrame({viewer: {kind: 'oracle', seat: null}});
    const {presentation} = shown;
    if (presentation === null) throw new Error('No presentation.');
    const board = (
      <ChessRenderer frame={shown} presentation={presentation} play={vi.fn()} />
    );
    rendered = render(
      <>
        <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={null} />
        <Perspective viewer={{kind: 'oracle', seat: null}} numSeats={2} />
        <ChessConfigSection />
        {board}
      </>,
    );
    const {container} = rendered;
    // Perspective names the first seat, and chess keeps View from until a
    // seat is chosen.
    expect(
      (labelled(container, 'Perspective') as HTMLSelectElement).value,
    ).toBe('0');
    expect(document.querySelector('.cg-wrap.orientation-white')).not.toBe(null);
    enter(labelled(container, 'Perspective'), '0');
    expect(document.querySelector('.cg-wrap.orientation-black')).not.toBe(null);
    const viewFrom = [...container.querySelectorAll('label')]
      .find(label => label.querySelector('span')?.textContent === 'View from')
      ?.querySelector('select');
    if (!(viewFrom instanceof HTMLSelectElement)) {
      throw new Error('No View from.');
    }
    enter(viewFrom, 'white');
    expect(document.querySelector('.cg-wrap.orientation-white')).not.toBe(null);
  });
});

describe('the chess move list', () => {
  it('pairs the moves a turn to a row, in figurine notation, and shows a position', () => {
    const shown = chessFrame({history: chessHistory(['e4', 'd5', 'Nf3'])});
    const live = session(shown);
    rendered = render(<History session={live} />);
    const rows = document.querySelectorAll(
      '.chess-history-row:not(.chess-history-row--header)',
    );
    expect(rows).toHaveLength(2);
    expect(rows[1]?.textContent).toBe('2.♞f3');
    expect(
      document.querySelector('.chess-history-move--current')?.textContent,
    ).toBe('♞f3');
    const initial = document.querySelector('.chess-history-initial');
    if (!(initial instanceof HTMLButtonElement)) throw new Error('No Initial.');
    act(() => {
      initial.click();
    });
    expect(live.commands.goTo).toHaveBeenCalledWith(0);
  });

  it('is registered with the chess renderer', () => {
    expect(RENDERERS.options('chess').History).toBeDefined();
    expect(RENDERERS.options('chess').keepsState).toBe(true);
  });
});
