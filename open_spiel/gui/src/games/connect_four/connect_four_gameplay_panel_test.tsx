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
import {afterAll, afterEach, beforeAll, describe, expect, it, vi} from 'vitest';
import {stubResizeObserver} from '../../testing/dom.ts';
import {type Rendered, only, render} from '../../testing/render.tsx';
import {
  buildBoard,
  connectFourDimensions,
  connectFourStatusText,
  createConnectFourGameplayPanel,
  winningCells,
} from './connect_four_gameplay_panel.tsx';
import type {SessionContextValue} from '../shared/board_session.ts';

// Setup offers `rows` and `columns`, and the engine plays them, so the
// gameplay panel must draw the grid from the session's effective game
// parameters instead of a hard-coded 6 x 7, place every piece, and accept a
// click on every legal column.

beforeAll(() => {
  stubResizeObserver();
});

afterAll(() => {
  vi.unstubAllGlobals();
});

interface Options {
  gameParams?: Record<string, unknown>;
  history: number[];
  legal: number[];
}

function makeSession(opts: Options): {
  useSession: () => SessionContextValue;
  submitAction: ReturnType<typeof vi.fn>;
} {
  const submitAction = vi.fn(() => Promise.resolve());
  const value = {
    session: {
      sessionId: 'sess-c4',
      mode: 'analysis',
      gameParams: opts.gameParams,
      players: [
        {slot: 0, controller: 'human', autoPlay: false},
        {slot: 1, controller: 'human', autoPlay: false},
      ],
      state: {
        history: opts.history,
        historyStr: '',
        currentPlayer: opts.history.length % 2,
        currentPlayerLabel: opts.history.length % 2 === 0 ? 'x' : 'o',
        turnNumber: opts.history.length + 1,
        isTerminal: false,
        legalActions: opts.legal.map(action => ({
          action,
          label: String(action),
        })),
      },
    },
    error: null,
    isBusy: false,
    viewer: {seat: 0, isParticipant: true, isActiveSeat: true, viewSeat: 0},
    submitAction,
  } as unknown as SessionContextValue;
  return {useSession: () => value, submitAction};
}

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
});

function mount(tree: Parameters<typeof render>[0]): Rendered {
  rendered = render(tree);
  return rendered;
}

function click(element: HTMLElement | null): void {
  if (element === null) throw new Error('No such element.');
  act(() => {
    element.click();
  });
}

interface RenderedPanel extends Rendered {
  board: HTMLElement;
  cells: () => HTMLElement[];
  cellAt: (row: number, column: number) => HTMLElement | null;
  submitAction: ReturnType<typeof vi.fn>;
}

function renderPanel(opts: Options): RenderedPanel {
  const {useSession, submitAction} = makeSession(opts);
  const Panel = createConnectFourGameplayPanel(useSession);
  const utils = mount(<Panel />);
  const board = only(utils.container, '[data-testid="c4-board"]');
  const cells = (): HTMLElement[] =>
    Array.from(board.querySelectorAll<HTMLElement>('.cf-cell'));
  const cellAt = (row: number, column: number): HTMLElement | null =>
    board.querySelector<HTMLElement>(
      `.cf-cell[data-row="${String(row)}"][data-column="${String(column)}"]`,
    );
  return {...utils, board, cells, cellAt, submitAction};
}

function range(n: number): number[] {
  return Array.from({length: n}, (_, i) => i);
}

describe('connectFourDimensions', () => {
  it('reads rows and columns from the effective game params', () => {
    expect(connectFourDimensions({rows: 8, columns: 9, x_in_row: 4})).toEqual({
      rows: 8,
      columns: 9,
    });
  });

  it('uses the engine defaults (6 x 7) for absent keys', () => {
    expect(connectFourDimensions(undefined)).toEqual({rows: 6, columns: 7});
    expect(connectFourDimensions({columns: 5})).toEqual({rows: 6, columns: 5});
  });

  it('throws on a malformed dimension rather than guessing', () => {
    expect(() => connectFourDimensions({rows: '8'})).toThrow(/rows/);
    expect(() => connectFourDimensions({columns: 0})).toThrow(/columns/);
    expect(() => connectFourDimensions({rows: 6.5})).toThrow(/rows/);
  });
});

describe('buildBoard', () => {
  it('stacks pieces from the bottom of a non-default grid', () => {
    const board = buildBoard([8, 8, 0], {rows: 4, columns: 9});
    expect(board).toHaveLength(4);
    expect(board[0]).toHaveLength(9);
    expect(board[3]?.[8]).toBe('X');
    expect(board[2]?.[8]).toBe('O');
    expect(board[3]?.[0]).toBe('X');
  });

  it('throws on an action outside the grid instead of dropping it', () => {
    expect(() => buildBoard([7], {rows: 6, columns: 7})).toThrow(/outside/);
    expect(() => buildBoard([-1], {rows: 6, columns: 7})).toThrow(/outside/);
  });

  it('throws on a move into a full column', () => {
    expect(() => buildBoard([0, 0, 0, 0, 0], {rows: 4, columns: 4})).toThrow(
      /full column/,
    );
  });
});

describe('ConnectFourGameplayPanel — board size follows the game params', () => {
  it('renders the default 6 x 7 grid when no size params are set', () => {
    const {board, cells} = renderPanel({history: [], legal: range(7)});
    expect(board.dataset['rows']).toBe('6');
    expect(board.dataset['columns']).toBe('7');
    expect(cells()).toHaveLength(42);
  });

  it('renders an 8 x 9 grid and draws a piece in column index 8', () => {
    const {board, cells, cellAt} = renderPanel({
      gameParams: {rows: 8, columns: 9, x_in_row: 4},
      history: [8, 8],
      legal: range(9),
    });
    expect(board.dataset['rows']).toBe('8');
    expect(board.dataset['columns']).toBe('9');
    expect(board.style.getPropertyValue('--cf-rows')).toBe('8');
    expect(board.style.getPropertyValue('--cf-columns')).toBe('9');
    expect(cells()).toHaveLength(72);
    expect(cellAt(7, 8)?.className).toMatch(/cf-cell--player-one/);
    expect(cellAt(6, 8)?.className).toMatch(/cf-cell--player-two/);
  });

  it('submits a click on a column beyond the old 7-column limit', () => {
    const {cellAt, submitAction} = renderPanel({
      gameParams: {rows: 8, columns: 9},
      history: [],
      legal: range(9),
    });
    click(cellAt(0, 8));
    expect(submitAction).toHaveBeenCalledWith(8);
  });

  it('renders a 4 x 5 grid and ignores a click on an illegal column', () => {
    const {cells, cellAt, submitAction} = renderPanel({
      gameParams: {rows: 4, columns: 5},
      history: [0, 0, 0, 0],
      legal: [1, 2, 3, 4],
    });
    expect(cells()).toHaveLength(20);
    expect(cellAt(0, 0)?.className).toMatch(/cf-cell--player-two/);
    click(cellAt(0, 0));
    expect(submitAction).not.toHaveBeenCalled();
    click(cellAt(3, 4));
    expect(submitAction).toHaveBeenCalledWith(4);
  });
});

describe('ConnectFourGameplayPanel — a UI/engine mismatch is shown in the panel', () => {
  function renderRaw(opts: Options): Rendered {
    const {useSession} = makeSession(opts);
    const Panel = createConnectFourGameplayPanel(useSession);
    return mount(<Panel />);
  }

  it('shows the error in place of the board for a move outside the grid', () => {
    const utils = renderRaw({
      gameParams: {rows: 6, columns: 7},
      history: [8],
      legal: range(7),
    });
    expect(
      utils.container.querySelector('[data-testid="c4-board"]'),
    ).toBeNull();
    const alert = only(utils.container, '[role="alert"]');
    expect(alert.textContent).toMatch(/Cannot draw this board/);
    expect(alert.textContent).toMatch(
      /history action 8 .* outside a 7-column board/,
    );
  });

  it('shows the error for a malformed size parameter', () => {
    const utils = renderRaw({
      gameParams: {rows: 'eight'},
      history: [],
      legal: [],
    });
    expect(
      utils.container.querySelector('[data-testid="c4-board"]'),
    ).toBeNull();
    expect(only(utils.container, '[role="alert"]').textContent).toMatch(
      /rows="eight"/,
    );
  });
});

describe('connect four at game over', () => {
  it('finds the winning four, and none in a full board without one', () => {
    const board = buildBoard([0, 1, 0, 1, 0, 1, 0], {rows: 6, columns: 7});
    expect([...winningCells(board)].sort()).toEqual([
      '2-0',
      '3-0',
      '4-0',
      '5-0',
    ]);
    expect(winningCells(buildBoard([0, 1, 2], {rows: 6, columns: 7}))).toEqual(
      new Set(),
    );
  });

  it('says the colour to move, and who won with the viewer result', () => {
    const state = {
      history: [0],
      historyStr: '0',
      currentPlayer: 1,
      currentPlayerLabel: 'player 1',
      isTerminal: false,
      legalActions: [],
      returns: null,
      turnNumber: 1,
    };
    expect(connectFourStatusText(state, 0)).toBe('Yellow to move');
    expect(
      connectFourStatusText(
        {...state, isTerminal: true, returns: [-1, 1], currentPlayer: -4},
        0,
      ),
    ).toBe('Yellow wins · You lose');
  });
});
