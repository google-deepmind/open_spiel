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

// A whole hand of euchre played on the euchre board against three Random
// seats, on the real server: North makes every move with the board's own
// gestures (the bidding panel's buttons, a click on a card of its hand),
// never the legal list. Every frame North receives is checked to hold no
// card of another seat's hand, not even the up-card in the dealer's; the
// other seats' calls and plays reach it under their public labels, and the
// dealer's discard and the deal stay hidden.
//
// With EUCHRE_SCREENSHOTS set to a directory, the table is captured there
// at EUCHRE_VIEWPORT_WIDTH (1280 by default) by 800 pixels: the first round
// of bidding, the second, the lone-hand choice when North makes trump, a
// trick in play, a finished trick on show, the hand over and the View panel.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';

const GAME = 'euchre';
const NORTH = 0;
// North's moves in one hand: a call or two, a lone-hand choice, a discard
// and five cards; far fewer than this.
const MAX_MOVES = 20;
const SCREENSHOTS = process.env['EUCHRE_SCREENSHOTS'] ?? '';
const SEED = Number(process.env['EUCHRE_SEED'] ?? '4');
// North orders up the up-card, unless EUCHRE_BID is 'pass'.
const ORDER_UP = process.env['EUCHRE_BID'] !== 'pass';
const COMPASS = ['N', 'E', 'S', 'W'] as const;
const HIDDEN_DISCARD = 'Dealer discards (hidden)';

/** The parts of the euchre struct this test reads. */
interface EuchreStruct {
  readonly hands: Readonly<Record<string, ReadonlyArray<string | null>>>;
  readonly phase: string;
  readonly upcard: string | null;
  readonly dealer: string;
  readonly discarding: boolean;
  readonly currentTrick: readonly unknown[];
}

function structOf(frame: Frame): EuchreStruct {
  if (frame.presentation?.renderer !== GAME) {
    throw new Error('The frame has no euchre presentation.');
  }
  const view = frame.presentation.view as {struct?: EuchreStruct};
  if (view.struct === undefined) throw new Error('The view has no struct.');
  return view.struct;
}

/**
 * Checks North is sent its own hand and, of every other hand, only its
 * size: one null per card, with no card among them, public or not.
 */
function expectNorthsFrame(frame: Frame): void {
  const struct = structOf(frame);
  COMPASS.forEach((compass, seat) => {
    const hand = struct.hands[compass] ?? [];
    if (seat === NORTH) expect(hand).not.toContain(null);
    else expect(hand.every(card => card === null)).toBe(true);
  });
  expect(frame.state_text).toBeNull();
}

/**
 * Checks North reads every other seat's call and play under its public
 * label, the dealer's discard as hidden, and no chance outcome.
 */
function expectPublicHistory(frame: Frame): void {
  const others = frame.history.filter(
    entry => entry.seat !== null && entry.seat !== NORTH,
  );
  expect(others.length).toBeGreaterThan(5);
  for (const entry of others) {
    expect(entry.action).toBeNull();
    expect(entry.public_label).not.toBeNull();
  }
  for (const entry of frame.history.filter(each => each.seat === null)) {
    expect(entry.public_label).toBeNull();
  }
  const discards = others.filter(e => e.public_label === HIDDEN_DISCARD);
  expect(discards.length).toBeLessThanOrEqual(1);
}

async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // The board settles its cards in flight before the shot.
  await page.waitForTimeout(900);
  await page.screenshot({path: path.join(SCREENSHOTS, `kernel-${name}.png`)});
}

// Milliseconds a press waits for its move to be sent before pressing again:
// a card still flying into place, or a trick still on show, may take the
// first press.
const PRESS_WAIT_MS = 4000;
const PRESS_TRIES = 6;

async function press(
  page: Page,
  target: Locator,
  what: string,
  card: boolean,
): Promise<void> {
  await expect(target, what).toBeVisible();
  for (let attempt = 0; attempt < PRESS_TRIES; attempt++) {
    const posted = page
      .waitForResponse(
        response =>
          response.request().method() === 'POST' &&
          new URL(response.url()).pathname.endsWith('/actions'),
        {timeout: PRESS_WAIT_MS},
      )
      .then(
        () => true,
        () => false,
      );
    // A fanned card shows only its left edge under the next one.
    if (card) await target.click({position: {x: 10, y: 40}});
    else await target.click();
    if (await posted) return;
  }
  throw new Error(
    `${what}: no move was sent after ${String(PRESS_TRIES)} presses.`,
  );
}

function biddingButton(board: Locator, text: string): Locator {
  return board.locator('.euchre-bidding-btn', {
    hasText: new RegExp(`^${text}$`),
  });
}

/**
 * North's move at `frame` as the board offers it: order up (or pass) in the first round, name
 * the first suit offered in the second, play with partner, and the first
 * card of its hand it may play (or discard).
 */
function chooseMove(
  board: Locator,
  frame: Frame,
): {target: Locator; what: string; card: boolean} {
  const {phase} = structOf(frame);
  if (phase === 'bidding1') {
    return ORDER_UP
      ? {
          target: biddingButton(board, 'Order Up'),
          what: 'order up',
          card: false,
        }
      : {
          target: biddingButton(board, 'Pass'),
          what: 'pass the up-card',
          card: false,
        };
  }
  if (phase === 'bidding2') {
    const suits = board.locator('.euchre-bidding-btn--suit');
    return {target: suits.first(), what: 'name trump', card: false};
  }
  if (phase === 'alone') {
    return {
      target: biddingButton(board, 'Play with Partner'),
      what: 'keep partner',
      card: false,
    };
  }
  const card = board.locator(
    '.euchre-player-section--bottom .euchre-card--playable',
  );
  return {target: card.first(), what: 'play or discard a card', card: true};
}

async function shoot(
  page: Page,
  frame: Frame,
  captured: Set<string>,
): Promise<void> {
  const struct = structOf(frame);
  const name =
    struct.phase === 'bidding1'
      ? '01-bidding1'
      : struct.phase === 'bidding2'
        ? '02-bidding2'
        : struct.phase === 'alone'
          ? '03-alone'
          : struct.phase === 'playing' && struct.currentTrick.length > 0
            ? '04-trick'
            : null;
  if (name === null || captured.has(name)) return;
  captured.add(name);
  await capture(page, name);
}

/** Plays North's moves to the end of the hand; returns the shots taken. */
async function playHand(app: App, board: Locator): Promise<Set<string>> {
  let plies = -1;
  const captured = new Set<string>();
  for (let move = 0; move < MAX_MOVES; move++) {
    const frame = await app.frames.waitFor(
      next =>
        next.session.game === GAME &&
        isViewer(next.viewer, NORTH) &&
        (next.status === 'terminal' ||
          (next.num_plies > plies &&
            next.to_move.seat === NORTH &&
            next.legal_actions !== null &&
            next.legal_actions.length > 0)),
      0,
      "North's turn, or the end of the hand",
    );
    if (frame.status === 'terminal') return captured;
    await shoot(app.page, frame, captured);
    const chosen = chooseMove(board, frame);
    await press(app.page, chosen.target, chosen.what, chosen.card);
    plies = frame.num_plies;
    if (chosen.card && !captured.has('05-held')) {
      // A finished trick stays on show before it is collected.
      const held = app.page.locator(
        '.euchre-gameplay-wrapper[data-trick-phase="hold"]',
      );
      const shown = await held.waitFor({timeout: 2500}).then(
        () => true,
        () => false,
      );
      if (shown) {
        captured.add('05-held');
        await capture(app.page, '05-held');
      }
    }
  }
  throw new Error(`The hand did not end in ${String(MAX_MOVES)} moves.`);
}

test('euchre: North plays a whole hand on the board against Random', async ({
  app,
  page,
}) => {
  test.setTimeout(300_000);
  const width = Number(process.env['EUCHRE_VIEWPORT_WIDTH'] ?? '1280');
  await page.setViewportSize({width, height: 800});
  await app.openSession(
    {
      game: GAME,
      parameters: {},
      seats: [
        {controller: 'human', config: {}, autoplay: false},
        {controller: 'random', config: {}, autoplay: true},
        {controller: 'random', config: {}, autoplay: true},
        {controller: 'random', config: {}, autoplay: true},
      ],
      chance_policy: 'auto',
      seed: SEED,
    },
    NORTH,
  );
  const board = (await app.gameplay()).locator('.euchre-gameplay-wrapper');
  await expect(board).toBeVisible();
  await expect(board.locator('.euchre-player-name--bottom')).toHaveText(
    'North',
  );
  await playHand(app, board);
  const end = await app.frames.waitFor(
    frame =>
      frame.session.game === GAME &&
      isViewer(frame.viewer, NORTH) &&
      frame.status === 'terminal',
    0,
    'the end of the hand',
  );
  expect(end.returns).toHaveLength(4);
  expectPublicHistory(end);
  await expect(board.locator('.euchre-game-over__title')).toHaveText(
    'Hand Over',
  );
  await capture(page, '06-result');
  // The other hands are backs on the page, as they were on the wire.
  for (const position of ['left', 'top', 'right']) {
    const hand = board.locator(`.euchre-player-section--${position}`);
    await expect(hand.getByRole('img', {name: / of /})).toHaveCount(0);
  }
  for (const frame of app.frames.all()) {
    if (frame.session.game === GAME && isViewer(frame.viewer, NORTH)) {
      expectNorthsFrame(frame);
    }
  }
  // The board's own View section: Card style, Trick hold, Show all cards.
  const section = await app.showSection('View', 'Euchre');
  await expect(section.getByText('Card style')).toBeVisible();
  await expect(section.getByText('Trick hold')).toBeVisible();
  await capture(page, '07-view');
});
