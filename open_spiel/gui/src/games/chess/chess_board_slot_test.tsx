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

import {type JSX, act} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {GameplayBoard} from '../../board/board.tsx';
import type {Frame} from '../../contract/contract.ts';
import {replaceRenderer} from '../../renderers/registry.ts';
import type {Renderer} from '../../renderers/renderer.ts';
import {stubResizeObserver} from '../../testing/dom.ts';
import {session} from '../../testing/frames.ts';
import {type Rendered, render} from '../../testing/render.tsx';
import {
  type ChessBoardProps,
  actionFor,
  readChessFrame,
  readChessViewer,
} from './chess_board_slot.ts';
import {chessRenderer} from './chess_renderer.tsx';
import {AFTER_E4_D5, MOVES, chessFrame} from './chess_test_fixtures.ts';

let rendered: Rendered | null = null;
let restore: Renderer | null = null;

beforeEach(() => {
  stubResizeObserver();
});

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  vi.unstubAllGlobals();
  if (restore !== null) replaceRenderer('chess', restore);
  restore = null;
});

// The board of the guide's worked example (src/INTERFACES.md): the position
// as text and one button per legal move.
function FakeBoard({frame, viewer, act: play}: ChessBoardProps): JSX.Element {
  return (
    <div data-testid="fake-board" data-orientation={viewer.orientation}>
      <p>
        {frame.fen} · {frame.turn} to move
      </p>
      {frame.moves.map(move => (
        <button
          key={move.action}
          type="button"
          disabled={!viewer.canMove}
          onClick={() => void play(move.action)}
        >
          {move.from}-{move.to}
        </button>
      ))}
    </div>
  );
}

function headerOf(shown: Frame): Parameters<typeof readChessFrame> {
  const {presentation} = shown;
  if (presentation === null) throw new Error('No presentation.');
  return [shown, presentation];
}

describe('the chess board slot', () => {
  it('draws chess with a registered board, whose act plays a legal move', () => {
    restore = replaceRenderer('chess', chessRenderer(FakeBoard));
    const shown = session(chessFrame());
    rendered = render(<GameplayBoard session={shown} />);
    const board = document.querySelector('[data-testid="fake-board"]');
    expect(board?.textContent).toContain(`${AFTER_E4_D5} · white to move`);
    expect(document.querySelector('.cg-board')).toBeNull();
    const buttons = [...document.querySelectorAll('button')];
    expect(buttons.map(button => button.textContent)).toEqual(
      MOVES.map(({from, to}) => `${from}-${to}`),
    );
    act(() => {
      buttons[1]?.click();
    });
    expect(shown.commands.act).toHaveBeenCalledWith(MOVES[1].id);
  });
});

describe('the chess board slot: act and the viewer', () => {
  it('rejects a move while the viewer cannot move, sending nothing', async () => {
    const Spy = vi.fn((props: ChessBoardProps) => <FakeBoard {...props} />);
    restore = replaceRenderer('chess', chessRenderer(Spy));
    // A move in flight: the session is busy.
    const shown = session(chessFrame(), true);
    rendered = render(<GameplayBoard session={shown} />);
    const props = Spy.mock.calls.at(-1)?.[0];
    if (props === undefined) throw new Error('The board was not drawn.');
    expect(props.viewer.canMove).toBe(false);
    await expect(props.act(MOVES[0].id)).rejects.toThrow(
      'You cannot move at this position.',
    );
    expect(shown.commands.act).not.toHaveBeenCalled();
  });

  it('turns the board for the oracle as View from picks', () => {
    expect(readChessViewer(chessFrame(), true, 'black').orientation).toBe(
      'black',
    );
    const seat0 = chessFrame({viewer: {kind: 'seat', seat: 0}});
    expect(readChessViewer(seat0, false, 'white').orientation).toBe('black');
  });

  it('draws the built-in board again once put back', () => {
    const previous = replaceRenderer('chess', chessRenderer(FakeBoard));
    replaceRenderer('chess', previous);
    rendered = render(<GameplayBoard session={session(chessFrame())} />);
    expect(document.querySelector('[data-testid="fake-board"]')).toBeNull();
    expect(document.querySelector('.chess-panel')).not.toBeNull();
  });

  it('refuses to replace a renderer that does not exist', () => {
    expect(() => replaceRenderer('no_such', chessRenderer(FakeBoard))).toThrow(
      /no renderer "no_such".*chess/,
    );
  });

  it('reads the moves, the last move and the side to move', () => {
    const frame = readChessFrame(...headerOf(chessFrame()));
    expect(frame.turn).toBe('white');
    expect(frame.lastMove).toEqual({from: 'd7', to: 'd5'});
    expect(frame.check).toBeNull();
    expect(actionFor(frame, 'e4', 'd5')).toBe(10);
    expect(actionFor(frame, 'e4', 'e6')).toBeNull();
  });

  it('turns the board to the viewer: seat 0 plays black', () => {
    const seat0 = chessFrame({viewer: {kind: 'seat', seat: 0}});
    expect(readChessViewer(seat0, false)).toMatchObject({
      side: 'black',
      orientation: 'black',
      isOracle: false,
      canMove: false,
    });
    const oracle = readChessViewer(chessFrame(), true);
    expect(oracle).toMatchObject({
      side: null,
      orientation: 'white',
      isOracle: true,
      canMove: true,
    });
  });
});
