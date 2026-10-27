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

// A whole game of checkers played on the checkers board: seat 0 clicks one
// of its movable men and then one of the landings the board marks, every
// jump of a multiple capture on its own, against Random until the game ends.
//
// With CHECKERS_SCREENSHOTS set to a directory, the page is captured there
// at CHECKERS_VIEWPORT_WIDTH (default 1280) by 800 pixels: the opening, a
// man selected, mid-game with the last turn marked, a multiple capture, a
// king, the end, the second seat's view and the View panel's Board section.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {type App, RANDOM} from './app.ts';
import {expect, test} from './fixtures.ts';

const GAME = 'checkers';
// A checkers game ends within a few hundred plies: the game draws one that
// runs too long.
const MAX_TURNS = 400;
const SCREENSHOTS = process.env['CHECKERS_SCREENSHOTS'] ?? '';
const ENDING = /^(Black wins|White wins|Draw)$/;

async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // The last move's slide settles, and no man is hovered, before the shot.
  await page.mouse.move(0, 0);
  await page.waitForTimeout(400);
  await page.screenshot({path: path.join(SCREENSHOTS, `kernel-${name}.png`)});
}

/** Seat 0's men the board lets it pick up now. */
function movable(board: Locator): Locator {
  return board.locator('.checkers-piece--movable');
}

/** The landings the board marks for the man picked up. */
function landings(board: Locator): Locator {
  return board.locator('.checkers-square--destination');
}

/** Waits until seat 0 is to move on a frame after `after`, or the end. */
async function waitForTurn(app: App, after: number): Promise<Frame> {
  return app.frames.waitFor(
    frame =>
      frame.session.game === GAME &&
      frame.ply === frame.num_plies &&
      (frame.status === 'terminal' ||
        (frame.to_move.kind === 'seat' && frame.to_move.seat === 0)),
    after,
    'seat 0 to move, or the end',
  );
}

/** Plays one move or jump on the board: a man, then its first landing. */
async function playOnBoard(app: App, board: Locator): Promise<void> {
  await movable(board).first().click();
  const target = landings(board).first();
  await expect(target).toBeVisible();
  const before = app.frames.count;
  await Promise.all([
    app.page.waitForResponse(
      response =>
        response.request().method() === 'POST' &&
        new URL(response.url()).pathname.endsWith('/actions'),
    ),
    target.click(),
  ]);
  await waitForTurn(app, before);
}

/**
 * Plays seat 0's moves on the board until the game ends, capturing the
 * mid-game, the first multiple capture and the first king on the way.
 */
async function playToEnd(
  app: App,
  page: Page,
  board: Locator,
): Promise<{multipleCapture: boolean; king: boolean}> {
  let multipleCapture = false;
  let king = false;
  let midGame = false;
  for (let turn = 0; turn < MAX_TURNS; turn++) {
    const frame = await waitForTurn(app, 0);
    if (frame.status === 'terminal') break;
    await expect(movable(board).first()).toBeVisible();
    const history = frame.history;
    const last = history.at(-1);
    if (last?.seat === 0) {
      // Seat 0 is to move again: a multiple capture, and only the jumping
      // man may move on.
      await expect(movable(board)).toHaveCount(1);
      if (!multipleCapture) await capture(page, '04-multiple-capture');
      multipleCapture = true;
    }
    if (!midGame && frame.num_plies >= 20) {
      midGame = true;
      await expect(
        board.locator('.checkers-square--last-move').first(),
      ).toBeVisible();
      await capture(page, '03-midgame');
    }
    if (!king && (await board.locator('.checkers-piece--king').count()) > 0) {
      king = true;
      await expect(board.locator('.checkers-king-base').first()).toBeVisible();
      await capture(page, '05-king');
    }
    await playOnBoard(app, board);
  }
  return {multipleCapture, king};
}

/** The board's View section sets the theme and the coordinates. */
async function checkViewSection(app: App, page: Page): Promise<void> {
  // The board's own View section, before it is changed.
  const section = await app.showSection('View', 'Board');
  await expect(section.getByText('Show coordinates')).toBeVisible();
  await capture(page, '08-view');
  await section.getByLabel('Show coordinates').check();
  await section.getByLabel('Theme').selectOption('modern');
  const shown = await app.gameplay();
  await expect(shown.locator('.checkers-coords--left')).toBeVisible();
  await expect(shown.getByTestId('checkers-board')).toHaveClass(
    /checkers-board--modern/,
  );
  await capture(page, '09-modern-coordinates');
}

test('checkers: seat 0 plays Random to the end on the board', async ({
  app,
  page,
}) => {
  test.setTimeout(420_000);
  const width = Number(process.env['CHECKERS_VIEWPORT_WIDTH'] ?? '1280');
  await page.setViewportSize({width, height: 800});
  await app.openGame(GAME);
  await app.setViewer(0);
  await app.setController(1, RANDOM, true);
  const gameplay = await app.gameplay();
  const board = gameplay.getByTestId('checkers-board');
  const status = gameplay.getByTestId('checkers-status');
  await expect(board).toBeVisible();
  await expect(status).toHaveText('Black to move');
  await expect(gameplay.getByText('Your turn!')).toBeVisible();
  await expect(board.locator('.checkers-square')).toHaveCount(64);
  await expect(board.locator('.checkers-piece--black')).toHaveCount(12);
  await expect(board.locator('.checkers-piece--white')).toHaveCount(12);
  // The opening: the four Black men that can step forward. a1 is dark.
  await expect(movable(board)).toHaveCount(4);
  await expect(board.locator('[data-row="0"][data-col="0"]')).toHaveClass(
    /checkers-square--dark/,
  );
  await capture(page, '01-opening');

  // A man picked up, its landings marked; a click elsewhere puts it down.
  await movable(board).first().click();
  await expect(board.locator('.checkers-square--selected')).toHaveCount(1);
  await expect(landings(board).first()).toBeVisible();
  await capture(page, '02-selected');
  await board.locator('.checkers-square--light').first().click();
  await expect(board.locator('.checkers-square--selected')).toHaveCount(0);

  const {multipleCapture, king} = await playToEnd(app, page, board);

  const end = app.frames.all().at(-1);
  if (end === undefined) throw new Error('No frame has arrived.');
  expect(end.status).toBe('terminal');
  expect(end.returns).toHaveLength(2);
  expect(end.seats[1]?.controller).toBe(RANDOM.name);
  await expect(status).toHaveText(ENDING);
  await expect(movable(board)).toHaveCount(0);
  // Every ply seat 0 played was a move or a jump on the board.
  expect(end.history.filter(entry => entry.seat === 0).length).toBeGreaterThan(
    0,
  );
  await capture(page, '06-end');

  await checkViewSection(app, page);
  console.log(
    `checkers: ${String(end.num_plies)} plies, returns ` +
      `${JSON.stringify(end.returns)}, a multiple capture: ` +
      `${String(multipleCapture)}, a king: ${String(king)}`,
  );
});

test('checkers: White sees the board turned, and a spectator cannot move', async ({
  app,
  page,
}) => {
  const width = Number(process.env['CHECKERS_VIEWPORT_WIDTH'] ?? '1280');
  await page.setViewportSize({width, height: 800});
  await app.openGame(GAME);
  await app.setViewer(1);
  const gameplay = await app.gameplay();
  const board = gameplay.getByTestId('checkers-board');
  // White's own men at the bottom: h1 is drawn top left.
  await expect(board.locator('.checkers-square').first()).toHaveAttribute(
    'data-row',
    '0',
  );
  await expect(board.locator('.checkers-square').first()).toHaveAttribute(
    'data-col',
    '7',
  );
  await expect(movable(board)).toHaveCount(0);
  await expect(gameplay.getByText('Waiting for your turn…')).toBeVisible();
  await capture(page, '07-white-seat');
  await app.setViewer('spectator');
  await expect(board.locator('.checkers-square').first()).toHaveAttribute(
    'data-row',
    '7',
  );
  await expect(movable(board)).toHaveCount(0);
  await expect(gameplay.getByText('Spectating')).toBeVisible();
});
