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

import type {Frame} from '../../contract/contract.ts';
import {stubResizeObserver} from '../../testing/dom.ts';
import {actions, frame} from '../../testing/frames.ts';
import {type Rendered, only, render} from '../../testing/render.tsx';
import {CONNECT_FOUR_RENDERER} from '../connect_four/connect_four_renderer.tsx';
import {TIC_TAC_TOE_RENDERER} from '../tic_tac_toe/tic_tac_toe_renderer.tsx';
import {boardSession} from './board_session.ts';

let rendered: Rendered | undefined;

beforeEach(() => {
  stubResizeObserver();
});

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  vi.unstubAllGlobals();
});

interface Position {
  readonly game: string;
  readonly history: readonly number[];
  readonly legal: readonly number[];
  readonly viewer?: Frame['viewer'];
  readonly returns?: readonly number[];
  readonly parameters?: Record<string, number>;
}

function view(position: Position): Record<string, unknown> {
  const terminal = position.returns !== undefined;
  const toMove = position.history.length % 2;
  return {
    history: [...position.history],
    historyStr: position.history.join(', '),
    currentPlayer: terminal ? -4 : toMove,
    currentPlayerLabel: terminal ? 'terminal' : `player ${String(toMove)}`,
    isTerminal: terminal,
    returns: position.returns === undefined ? null : [...position.returns],
    turnNumber: position.history.length,
    gameParams: position.parameters ?? {},
  };
}

function framed(position: Position): Frame {
  const terminal = position.returns !== undefined;
  return frame({
    viewer: position.viewer ?? {kind: 'seat', seat: 0},
    status: terminal ? 'terminal' : 'playing',
    to_move: terminal
      ? {kind: 'terminal', seat: null}
      : {kind: 'seat', seat: position.history.length % 2},
    legal_actions: actions(9).filter(action =>
      position.legal.includes(action.id),
    ),
    returns: position.returns === undefined ? null : [...position.returns],
    presentation: {
      renderer: position.game,
      view: view(position),
      actions: [],
    },
  });
}

function draw(
  position: Position,
  play: ((action: number) => void) | null = vi.fn(),
): HTMLElement {
  const shown = framed(position);
  if (shown.presentation === null) throw new Error('No presentation.');
  const Renderer =
    position.game === 'connect_four'
      ? CONNECT_FOUR_RENDERER.render
      : TIC_TAC_TOE_RENDERER.render;
  rendered = render(
    <Renderer frame={shown} presentation={shown.presentation} play={play} />,
  );
  return rendered.container;
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

describe('board renderers', () => {
  it('draw their own status row and keep their state', () => {
    for (const entry of [CONNECT_FOUR_RENDERER, TIC_TAC_TOE_RENDERER]) {
      expect(entry.drawsStatus).toBe(true);
      expect(entry.keepsState).toBe(true);
    }
  });

  it('reads the view into the session the panels read', () => {
    const shown = framed({game: 'tic_tac_toe', history: [4], legal: [0, 1]});
    if (shown.presentation === null) throw new Error('No presentation.');
    const session = boardSession(shown, shown.presentation);
    expect(session.mode).toBe('game');
    expect(session.state.history).toEqual([4]);
    expect(session.state.currentPlayerLabel).toBe('player 1');
    expect(session.state.legalActions.map(action => action.action)).toEqual([
      0, 1,
    ]);
    expect(session.players.map(player => player.controller)).toEqual([
      'human',
      'human',
    ]);
  });

  it('refuses a view without the fields the panels read', () => {
    const shown = framed({game: 'tic_tac_toe', history: [], legal: []});
    expect(() =>
      boardSession(shown, {renderer: 'tic_tac_toe', view: {}, actions: []}),
    ).toThrow(/isTerminal/);
  });
});

describe('tic_tac_toe renderer', () => {
  it('shows the turn, marks the history, and plays a pressed cell', () => {
    const play = vi.fn();
    const container = draw(
      {game: 'tic_tac_toe', history: [4, 0], legal: [1, 2, 3, 5, 6, 7, 8]},
      play,
    );
    const meta = only(container, '.ogs-state-meta');
    // The mark to move and the viewer's own, never "Waiting on player 0"
    // beside "Your turn!".
    expect(meta.textContent).toBe('X to moveTurn: 2You play XYour turn!');
    expect(only(container, '[data-cell="4"]').textContent).toBe('X');
    expect(only(container, '[data-cell="0"]').textContent).toBe('O');
    expect(only(container, '[data-cell="0"]').className).toMatch(
      /tto-cell--last/,
    );
    click(only(container, '[data-cell="8"]'));
    expect(play).toHaveBeenCalledWith(8);
  });

  it('locks the board for the seat not to move', () => {
    const play = vi.fn();
    const container = draw(
      {game: 'tic_tac_toe', history: [4], legal: [0, 1, 2, 3, 5, 6, 7, 8]},
      play,
    );
    expect(only(container, '.ogs-state-meta').textContent).toContain(
      'Waiting for your turn…',
    );
    expect(container.querySelectorAll('.tto-cell--legal')).toHaveLength(0);
    click(only(container, '[data-cell="0"]'));
    expect(play).not.toHaveBeenCalled();
  });

  it('says a spectator is spectating', () => {
    const container = draw({
      game: 'tic_tac_toe',
      history: [],
      legal: [],
      viewer: {kind: 'spectator', seat: null},
    });
    expect(only(container, '.ogs-state-meta').textContent).toContain(
      'Spectating',
    );
  });

  it('says who won once the game is over, not the raw returns', () => {
    const container = draw({
      game: 'tic_tac_toe',
      history: [0, 3, 1, 4, 2],
      legal: [],
      returns: [1, -1],
    });
    expect(only(container, '.ogs-state-meta').textContent).toBe(
      'X wins · You winTurn: 5',
    );
    expect(only(container, '[data-testid="game-result"]').textContent).toBe(
      'X wins · You win',
    );
    expect(container.querySelectorAll('.tto-cell:not(:disabled)')).toHaveLength(
      0,
    );
  });
});

describe('connect_four renderer', () => {
  it('drops a disc in the pressed column', () => {
    const play = vi.fn();
    const container = draw(
      {game: 'connect_four', history: [3, 3], legal: [0, 1, 2, 3, 4, 5, 6]},
      play,
    );
    const board = only(container, '[data-testid="c4-board"]');
    expect(board.className).toMatch(/cf-board--interactive/);
    expect(only(board, '[data-row="5"][data-column="3"]').className).toMatch(
      /cf-cell--player-one/,
    );
    expect(only(board, '[data-row="4"][data-column="3"]').className).toMatch(
      /cf-cell--player-two/,
    );
    click(only(board, '[data-row="0"][data-column="6"]'));
    expect(play).toHaveBeenCalledWith(6);
  });

  it('follows the board size parameters', () => {
    const container = draw({
      game: 'connect_four',
      history: [],
      legal: [0, 1, 2, 3, 4],
      parameters: {rows: 4, columns: 5, x_in_row: 3},
    });
    const board = only(container, '[data-testid="c4-board"]');
    expect(board.querySelectorAll('.cf-cell')).toHaveLength(20);
  });

  it('cannot be played when the viewer cannot move', () => {
    const container = draw(
      {game: 'connect_four', history: [], legal: [0, 1, 2, 3, 4, 5, 6]},
      null,
    );
    const board = only(container, '[data-testid="c4-board"]');
    expect(board.className).not.toMatch(/cf-board--interactive/);
  });
});
