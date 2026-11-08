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

// Connect four and tic-tac-toe played to the end on their own Gameplay
// panels, through the real server: seat 0 plays every move by pressing the
// board (a column's hole, an empty cell), never the legal list; seat 1 is
// Random. Every frame seat 0 receives carries the view the panels read and
// nothing else, the same view the oracle and a spectator get, since neither
// game hides anything.
//
// With GRID_SCREENSHOTS set to a directory, the page is captured there at
// GRID_VIEWPORT_WIDTH (default 1280) by 800 pixels: the opening, mid-game
// and the end, in the light theme, and the end in the dark theme.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {type App, RANDOM} from './app.ts';
import {expect, test} from './fixtures.ts';
import {isViewer} from './frames.ts';

const SHOTS = process.env['GRID_SCREENSHOTS'] ?? '';
const WIDTH = Number(process.env['GRID_VIEWPORT_WIDTH'] ?? '1280');
const VIEW_FIELDS = [
  'currentPlayer',
  'currentPlayerLabel',
  'gameParams',
  'history',
  'historyStr',
  'isTerminal',
  'returns',
  'turnNumber',
];

test.use({viewport: {width: WIDTH, height: 800}});

interface Game {
  readonly name: string;
  /** The most moves seat 0 can make. */
  readonly moves: number;
  /** The board's test id. */
  readonly board: string;
  /** The places seat 0 may press now. */
  readonly playable: (board: Locator) => Locator;
  /** The move after which the mid-game shot is taken. */
  readonly midgame: number;
  /** Each seat's mark or colour, as the status names the seats. */
  readonly sides: readonly [string, string];
}

/** The result as the board says it to seat 0: "X wins · You win", "Draw". */
function resultFor(game: Game, returns: readonly number[] | null): string {
  const [first = 0, second = 0] = returns ?? [];
  if (first > second) return `${game.sides[0]} wins · You win`;
  if (second > first) return `${game.sides[1]} wins · You lose`;
  return 'Draw';
}

const CONNECT_FOUR: Game = {
  name: 'connect_four',
  moves: 21,
  board: 'c4-board',
  playable: board => board.locator('.cf-cell--interactive[data-row="0"]'),
  midgame: 4,
  sides: ['Red', 'Yellow'],
};
const TIC_TAC_TOE: Game = {
  name: 'tic_tac_toe',
  moves: 5,
  board: 'ttt-board',
  playable: board => board.locator('button.tto-cell--legal'),
  midgame: 2,
  sides: ['X', 'O'],
};

function viewOf(frame: Frame): Record<string, unknown> {
  const view = frame.presentation?.view;
  if (view === null || typeof view !== 'object' || Array.isArray(view)) {
    throw new Error('The frame has no view.');
  }
  return view as Record<string, unknown>;
}

// Whether seat 0 is to move at `frame`, or the game is over.
function settled(frame: Frame, game: Game): boolean {
  return (
    frame.session.game === game.name &&
    isViewer(frame.viewer, 0) &&
    (frame.status === 'terminal' ||
      (frame.to_move.kind === 'seat' && frame.to_move.seat === 0))
  );
}

/** Seat 0's frames hold the panels' view, and only it. */
function checkFrames(app: App, game: Game): void {
  const seen = app.frames
    .all()
    .filter(frame => frame.session.game === game.name);
  expect(seen.length).toBeGreaterThan(0);
  for (const frame of seen) {
    expect(frame.omniscient).toBe(true);
    expect(frame.presentation?.renderer).toBe(game.name);
    const view = viewOf(frame);
    expect(Object.keys(view).sort()).toEqual(VIEW_FIELDS);
    expect(view['history']).toEqual(
      frame.history.slice(0, frame.ply).map(entry => entry.action),
    );
  }
}

async function shot(page: Page, name: string): Promise<void> {
  if (SHOTS === '') return;
  // The cells ease their colours over 150 ms when the theme changes.
  await page.waitForTimeout(500);
  await page.screenshot({
    path: path.join(SHOTS, `kernel-${String(WIDTH)}-${name}.png`),
  });
}

async function press(page: Page, target: Locator): Promise<void> {
  await Promise.all([
    page.waitForResponse(
      response =>
        response.request().method() === 'POST' &&
        new URL(response.url()).pathname.endsWith('/actions'),
    ),
    target.click(),
  ]);
}

/** Plays seat 0 on the board, at random, until the game ends. */
async function playBoardToEnd(
  app: App,
  page: Page,
  game: Game,
): Promise<Frame> {
  const gameplay = await app.gameplay();
  const board = gameplay.getByTestId(game.board);
  const status = gameplay.locator('.ogs-state-meta');
  let after = 0;
  let ply = -1;
  for (let move = 0; move <= game.moves; move++) {
    const frame = await app.frames.waitFor(
      shown => settled(shown, game) && shown.ply > ply,
      after,
      `seat 0's turn after ply ${String(ply)}`,
    );
    if (frame.status === 'terminal') {
      await expect(status).toContainText(resultFor(game, frame.returns));
      await expect(game.playable(board)).toHaveCount(0);
      return frame;
    }
    await expect(status).toContainText(
      `${game.sides[0]} to moveTurn: ${String(frame.ply)}You play ${game.sides[0]}Your turn!`,
    );
    if (move === 0) await shot(page, `${game.name}-01-opening`);
    if (move === game.midgame) await shot(page, `${game.name}-02-midgame`);
    ply = frame.ply;
    const cells = game.playable(board);
    await expect(cells.first()).toBeVisible();
    const count = await cells.count();
    expect(count).toBe((frame.legal_actions ?? []).length);
    after = app.frames.count;
    await press(page, cells.nth(Math.floor(Math.random() * count)));
  }
  throw new Error(`${game.name} did not end in ${String(game.moves)} moves.`);
}

async function playAgainstRandom(
  app: App,
  page: Page,
  game: Game,
): Promise<Frame> {
  await app.openGame(game.name);
  await app.chooseTheme('Light');
  await app.setViewer(0);
  await app.setController(1, RANDOM, true);
  const end = await playBoardToEnd(app, page, game);
  expect(end.status).toBe('terminal');
  expect(end.history.map(entry => entry.seat)).toEqual(
    end.history.map((_, index) => index % 2),
  );
  checkFrames(app, game);
  // The oracle and a spectator get the very view seat 0 got.
  const seatView = viewOf(end);
  for (const viewer of ['oracle', 'spectator'] as const) {
    const shown = await app.setViewer(viewer);
    expect(viewOf(shown)).toEqual(seatView);
  }
  await app.setViewer(0);
  return end;
}

async function endShots(app: App, page: Page, name: string): Promise<void> {
  await shot(page, `${name}-03-end`);
  await app.chooseTheme('Dark');
  await shot(page, `${name}-04-end-dark`);
  await app.chooseTheme('Light');
}

test('tic_tac_toe: seat 0 marks cells against Random to the end', async ({
  app,
  page,
}) => {
  const end = await playAgainstRandom(app, page, TIC_TAC_TOE);
  const gameplay = await app.gameplay();
  await expect(gameplay.getByTestId('game-result')).toHaveText(
    resultFor(TIC_TAC_TOE, end.returns),
  );
  await expect(gameplay.locator('.tto-cell:not(:disabled)')).toHaveCount(0);
  await endShots(app, page, 'tic_tac_toe');
});

test('connect_four: seat 0 drops discs against Random to the end', async ({
  app,
  page,
}) => {
  const end = await playAgainstRandom(app, page, CONNECT_FOUR);
  const gameplay = await app.gameplay();
  const board = gameplay.getByTestId('c4-board');
  // Every disc played is on the board, in its player's colour.
  const discs = end.history.length;
  const red = board.locator('.cf-cell--player-one');
  const yellow = board.locator('.cf-cell--player-two');
  await expect(red).toHaveCount(Math.ceil(discs / 2));
  await expect(yellow).toHaveCount(Math.floor(discs / 2));
  await expect(board).not.toHaveClass(/cf-board--interactive/);
  await endShots(app, page, 'connect_four');
});
