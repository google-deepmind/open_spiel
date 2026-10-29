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

// A hand of gin rummy that reaches the wall with seat 0 to move: the stock
// is down to its last two cards, and seat 0 may only pass (the hand ends in
// a draw) or knock with the upcard. Both seats are humans played through the
// server's API until then, neither knocking; a draw from the upcard instead
// of the stock steers the wall to seat 0. Seat 0 then passes with the
// board's Pass button, and the hand ends in a draw with no score.
//
// At the wall the Pass button must be enabled and must play the wall's
// pass; otherwise no move can be sent and the hand cannot end.
// GIN_WALL_SEED picks the deal.

/// <reference types="node" />

import process from 'node:process';

import type {Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';
import {
  GAME,
  MAX_MOVES,
  button,
  legal,
  press,
  readView,
  seatAct,
  seatDiscard,
  seatFrame,
} from './gin_board.ts';

/** The cards left in the stock when the wall is reached. */
const WALL_STOCK = 2;

/** The stock's size in `frame`. */
function stockSize(frame: Frame): number {
  const struct = readView(frame).view['struct'] as Record<string, unknown>;
  const size = struct['stock_size'];
  if (typeof size !== 'number') throw new Error('No stock size.');
  return size;
}

/**
 * The next move of a seat that never knocks, through the API. The stock draw
 * that would leave the wall to seat 1 is a draw from the upcard instead.
 */
function nextMove(frame: Frame, seat: number): number {
  const view = readView(frame);
  const pass = legal(frame, 'Pass');
  const stock = legal(frame, 'Draw stock');
  const upcard = legal(frame, 'Draw upcard');
  if (view.phase === 'FirstUpcard') {
    const move = pass ?? stock;
    if (move === null) throw new Error('No first-upcard move.');
    return move;
  }
  if (view.phase === 'Draw') {
    // The wall falls to the seat after the one whose stock draw leaves
    // WALL_STOCK cards: seat 0 must not make that draw.
    const size = stockSize(frame);
    const steer = seat === 0 && size === WALL_STOCK + 1;
    const move = steer ? upcard : (stock ?? upcard);
    if (move === null) throw new Error('No draw.');
    return move;
  }
  if (view.phase === 'Discard') return seatDiscard(frame, view, seat);
  throw new Error(
    `Seat ${String(seat)} at phase ${view.phase} before seat 0's wall.`,
  );
}

/**
 * Plays both seats through the API until seat 0 is to move at the wall.
 *
 * @returns That frame, as the server gives it to seat 0.
 */
async function playToWall(page: Page, id: string): Promise<Frame> {
  for (let move = 0; move < 2 * MAX_MOVES; move++) {
    const server = await seatFrame(page, id, 0);
    if (server.status === 'terminal') {
      throw new Error('The hand ended before seat 0 reached the wall.');
    }
    const seat = server.to_move.seat;
    if (seat === null) throw new Error('No seat to move.');
    if (seat === 0 && readView(server).phase === 'Wall') return server;
    const frame = seat === 0 ? server : await seatFrame(page, id, seat);
    await seatAct(page, id, seat, nextMove(frame, seat));
  }
  throw new Error(`No wall in ${String(2 * MAX_MOVES)} moves.`);
}

/**
 * Opens a hand between two humans, seat 0 on the page.
 *
 * @returns The session's id.
 */
async function openHand(app: App): Promise<string> {
  const first = await app.openSession(
    {
      game: GAME,
      parameters: {oklahoma: true},
      seats: [
        {controller: 'human', config: {}, autoplay: false},
        {controller: 'human', config: {}, autoplay: false},
      ],
      chance_policy: 'auto',
      seed: Number(process.env['GIN_WALL_SEED'] ?? '5'),
    },
    0,
  );
  return first.session.id;
}

test("gin_rummy: seat 0 passes at the wall with the board's Pass", async ({
  app,
  page,
}) => {
  test.setTimeout(300_000);
  await page.setViewportSize({width: 1280, height: 800});
  const id = await openHand(app);
  const board = (await app.gameplay()).locator('.gin-render-wrapper');
  await expect(board).toBeVisible();
  const wall = await playToWall(page, id);
  expect(stockSize(wall)).toBe(WALL_STOCK);
  const pass = legal(wall, 'Pass');
  expect(pass, 'Pass is legal at the wall').not.toBeNull();
  // The moves went through the API, off the page: the page is shown the
  // session afresh, at the wall for seat 0.
  const before = Math.max(0, app.frames.count - 1);
  await page.reload();
  await app.frames.waitFor(
    frame =>
      frame.session.id === id &&
      isViewer(frame.viewer, 0) &&
      frame.num_plies === wall.num_plies,
    before,
    'the wall on the page',
  );
  const passButton = button(board, 'Pass');
  await expect(passButton, 'Pass at the wall').toBeEnabled();
  // At the wall Pass ends the hand in a draw, and its title says so.
  await expect(passButton).toHaveAttribute('title', /draw/);
  // Which wall this deal reaches: Pass alone, or Pass beside a legal knock.
  const knock = legal(wall, 'Knock');
  test.info().annotations.push({
    type: 'wall',
    description: `ply ${String(wall.num_plies)}, knock ${knock === null ? 'not legal' : 'legal'}`,
  });
  const sent: unknown[] = [];
  page.on('request', request => {
    if (request.method() === 'POST' && request.url().includes('/actions')) {
      sent.push(request.postDataJSON());
    }
  });
  await press(page, {target: passButton, what: 'pass', double: false});
  const end = await app.frames.waitFor(
    frame =>
      frame.session.id === id &&
      isViewer(frame.viewer, 0) &&
      frame.status === 'terminal',
    0,
    'the end of the hand',
  );
  expect(end.num_plies).toBe(wall.num_plies + 1);
  // One move was sent: the Pass.
  expect(sent).toHaveLength(1);
  expect(sent[0]).toMatchObject({action: pass});
  expect(end.returns).toEqual([0, 0]);
  await expect(button(board, 'Result')).toBeVisible();
});
