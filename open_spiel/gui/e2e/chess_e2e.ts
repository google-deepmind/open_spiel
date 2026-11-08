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

// Chess played on its own board, through the board's own gestures: a
// whole game as White against Random, every move pressed on two squares or
// dragged, a promotion chosen in the board's promotion choice, never the
// legal list; every frame the seat receives is checked to hold the board's
// four fields and nothing more. The board at each phase of a game
// (opening, a piece picked up, mid-game, check, the View panel's Board
// section, the promotion choice, after the promotion, checkmate) is
// captured to CHESS_SCREENSHOTS at CHESS_VIEWPORT_WIDTH (1280) by 800.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {type App, RANDOM} from './app.ts';
import {expect, test} from './fixtures.ts';

const WHITE = 1;
const BLACK = 0;
// The most moves White makes before the game must be over: OpenSpiel's chess
// ends a game by the fifty-move rule well before this.
const MAX_MOVES = 600;
const FILES = 'abcdefgh';
const SCREENSHOTS = process.env['CHESS_SCREENSHOTS'] ?? '';
const VIEW_FIELDS = ['chessCheck', 'chessLastMove', 'chessOutcome', 'fen'];

interface Hint {
  readonly from: string;
  readonly to: string;
  readonly promotion?: string;
}

function board(app: App): Locator {
  return app.page.locator('.chess-board .cg-board');
}

/** The centre of `square` on the page, for the side the board is seen from. */
async function centre(
  app: App,
  square: string,
): Promise<{x: number; y: number}> {
  const box = await board(app).boundingBox();
  if (box === null) throw new Error('The board is not on screen.');
  const black =
    (await app.page.locator('.cg-wrap.orientation-black').count()) > 0;
  const file = FILES.indexOf(square[0] ?? '');
  const rank = Number(square[1]);
  const column = black ? 7 - file : file;
  const row = black ? rank - 1 : 8 - rank;
  const step = box.width / 8;
  return {x: box.x + (column + 0.5) * step, y: box.y + (row + 0.5) * step};
}

async function pressSquare(app: App, square: string): Promise<void> {
  const {x, y} = await centre(app, square);
  await app.page.mouse.click(x, y);
}

async function dragPiece(app: App, from: string, to: string): Promise<void> {
  const start = await centre(app, from);
  const end = await centre(app, to);
  const {mouse} = app.page;
  await mouse.move(start.x, start.y);
  await mouse.down();
  await mouse.move((start.x + end.x) / 2, (start.y + end.y) / 2, {steps: 4});
  await mouse.move(end.x, end.y, {steps: 4});
  await mouse.up();
}

function hints(frame: Frame): Hint[] {
  const actions = frame.presentation?.actions;
  if (actions === null || actions === undefined) {
    throw new Error(`Frame at ply ${String(frame.ply)} hints no moves.`);
  }
  return actions.map(({hint}) => hint as Hint);
}

function fen(frame: Frame): string {
  const view = frame.presentation?.view as {fen?: unknown} | undefined;
  if (typeof view?.fen !== 'string') throw new Error('The frame has no FEN.');
  return view.fen;
}

/** The squares of a FEN that hold a piece. */
function occupied(position: string): Set<string> {
  const squares = new Set<string>();
  (position.split(' ')[0] ?? '').split('/').forEach((row, index) => {
    let file = 0;
    for (const letter of row) {
      if (/\d/.test(letter)) {
        file += Number(letter);
      } else {
        squares.add(`${FILES[file] ?? '?'}${String(8 - index)}`);
        file += 1;
      }
    }
  });
  return squares;
}

// A capture or a promotion when there is one, so that the game ends sooner;
// otherwise any move.
function chooseMove(frame: Frame, pick: () => number): Hint {
  const taken = occupied(fen(frame));
  const moves = hints(frame).filter(
    move => move.promotion === undefined || move.promotion === 'q',
  );
  const sharp = moves.filter(
    move => move.promotion !== undefined || taken.has(move.to),
  );
  const pool = sharp.length > 0 ? sharp : moves;
  const move = pool[Math.floor(pick() * pool.length)];
  if (move === undefined) throw new Error('No legal move to choose.');
  return move;
}

/** Plays `move` on the board, pressing or dragging, and waits for it. */
async function play(
  app: App,
  move: Hint,
  drag: boolean,
  plies: number,
): Promise<Frame> {
  const before = app.frames.count;
  if (drag) {
    await dragPiece(app, move.from, move.to);
  } else {
    await pressSquare(app, move.from);
    await pressSquare(app, move.to);
  }
  if (move.promotion !== undefined) {
    await app.page.getByRole('button', {name: 'Promote to Queen'}).click();
  }
  return app.frames.waitFor(
    frame => frame.num_plies > plies,
    before,
    `the frame after ${move.from}${move.to}`,
  );
}

// A seeded generator, so that a failing game can be played again.
function generator(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

/** Each frame holds the board's four fields; returns how many it checked. */
function expectBoardFieldsOnly(frames: readonly Frame[]): number {
  let checked = 0;
  for (const frame of frames) {
    if (frame.session.game !== 'chess' || frame.presentation === null) continue;
    const view = frame.presentation.view as Record<string, unknown>;
    expect(Object.keys(view).sort()).toEqual(VIEW_FIELDS);
    // Chess hides nothing, so every viewer is omniscient.
    expect(frame.omniscient).toBe(true);
    checked++;
  }
  return checked;
}

test('chess: White plays a whole game against Random on the board', async ({
  app,
}) => {
  test.setTimeout(30 * 60_000);
  await app.openGame('chess');
  await app.setViewer(WHITE);
  await app.setController(BLACK, RANDOM, true);
  await expect(board(app)).toBeVisible();
  await expect(app.page.getByTestId('chess-strip-white')).toContainText(
    'Human · you',
  );
  const pick = generator(20261021);
  let frame = await app.frames.waitFor(
    shown => shown.to_move.kind === 'seat' && shown.to_move.seat === WHITE,
    0,
    'white to move',
  );
  let promotions = 0;
  for (let moves = 0; moves < MAX_MOVES && frame.status !== 'terminal';) {
    if (frame.to_move.kind === 'seat' && frame.to_move.seat === WHITE) {
      const move = chooseMove(frame, pick);
      if (move.promotion !== undefined) promotions++;
      frame = await play(app, move, moves % 2 === 1, frame.num_plies);
      moves++;
    } else {
      const plies = frame.num_plies;
      frame = await app.frames.waitFor(
        shown => shown.num_plies > plies || shown.status === 'terminal',
        Math.max(0, app.frames.count - 1),
        'Random to move',
      );
    }
  }
  expect(frame.status).toBe('terminal');
  process.stdout.write(
    `chess game: ${String(frame.num_plies)} plies, ${String(promotions)} promotions, returns ${JSON.stringify(frame.returns)}\n`,
  );
  expect(frame.history.map(entry => entry.seat)).toEqual(
    frame.history.map((_, index) => (index % 2 === 0 ? WHITE : BLACK)),
  );
  // The result is the line under the board, and no piece can be picked up.
  await expect(app.page.getByTestId('chess-status')).toHaveText(
    /^(1-0 · White wins|0-1 · Black wins|½-½ · Draw)/,
  );
  await expect(app.page.locator('.cg-wrap.view-only')).toHaveCount(1);
  // The move list pairs the moves a turn to a row.
  const rows = app.page.locator(
    '.chess-history-row:not(.chess-history-row--header)',
  );
  await expect(rows).toHaveCount(Math.ceil(frame.history.length / 2));
  expect(expectBoardFieldsOnly(app.frames.all())).toBeGreaterThan(10);
});

async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // Pieces in flight settle before the shot.
  await page.waitForTimeout(700);
  await page.screenshot({path: path.join(SCREENSHOTS, `kernel-${name}.png`)});
}

/** Plays `moves` (from, to) on the board as whoever is to move. */
async function playLine(app: App, moves: readonly string[]): Promise<void> {
  for (const text of moves) {
    const plies = app.frames.all().at(-1)?.num_plies ?? 0;
    await play(
      app,
      {from: text.slice(0, 2), to: text.slice(2, 4)},
      false,
      plies,
    );
  }
}

test('chess: the board at each phase of a game', async ({app, page}) => {
  test.setTimeout(10 * 60_000);
  const width = Number(process.env['CHESS_VIEWPORT_WIDTH'] ?? '1280');
  await page.setViewportSize({width, height: 800});
  await app.openGame('chess');
  await app.setViewer('oracle');
  await app.showSection('Setup', 'Players');
  await expect(board(app)).toBeVisible();
  await capture(page, '01-opening');
  // A piece picked up shows its moves.
  await pressSquare(app, 'g1');
  await expect(page.locator('.square.move-dest')).toHaveCount(2);
  await capture(page, '02-selected');
  await pressSquare(app, 'g1');
  await expect(page.locator('.square.selected')).toHaveCount(0);
  await playLine(app, ['e2e4', 'd7d5', 'e4d5', 'd8d5', 'b1c3']);
  await expect(page.locator('.square.last-move')).toHaveCount(2);
  await capture(page, '03-midgame');
  await playLine(app, ['d5e5']);
  await expect(page.locator('.square.check')).toHaveCount(1);
  await capture(page, '04-check');
  const section = await app.showSection('View', 'Board');
  await expect(section.getByText('View from')).toBeVisible();
  await capture(page, '05-view');
  await app.resetGame();
  await app.showSection('Setup', 'Players');
  await app.frames.waitFor(
    frame => frame.num_plies === 0,
    Math.max(0, app.frames.count - 1),
    'the reset game',
  );
  await playLine(app, [
    'e2e4',
    'd7d5',
    'e4d5',
    'c7c6',
    'd5c6',
    'g8f6',
    'c6b7',
    'c8d7',
  ]);
  await dragPiece(app, 'b7', 'a8');
  await expect(
    page.getByRole('button', {name: 'Promote to Queen'}),
  ).toBeVisible();
  await capture(page, '06-promotion');
  const before = app.frames.count;
  await page.getByRole('button', {name: 'Promote to Queen'}).click();
  await app.frames.waitFor(frame => frame.num_plies === 9, before, 'bxa8=Q');
  await expect(page.getByTestId('chess-strip-white')).toContainText('+16');
  await capture(page, '07-promoted');
  await app.resetGame();
  await app.showSection('Setup', 'Players');
  await app.frames.waitFor(
    frame => frame.num_plies === 0,
    Math.max(0, app.frames.count - 1),
    'the reset game',
  );
  await playLine(app, ['f2f3', 'e7e5', 'g2g4', 'd8h4']);
  await expect(page.getByTestId('chess-status')).toHaveText(
    '0-1 · Black wins by checkmate',
  );
  await capture(page, '08-checkmate');
});
