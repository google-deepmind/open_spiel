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

// A whole hand of hearts played on the hearts board against Random,
// on the real server: seat 0 picks its three pass cards by clicking them in
// its hand and passes with the board's Pass button, then plays every card
// by clicking it, never the legal list. Every frame seat 0 receives is
// checked to hold none of the other seats' cards; the other seats' plays
// reach it under their public labels and their passes stay hidden.
//
// With HEARTS_SCREENSHOTS set to a directory, the page is captured there at
// HEARTS_VIEWPORT_WIDTH (default 1280) by 800 pixels: the pass picks, the
// cards received, mid-trick on seat 0's turn, the end of the hand and the
// View panel's Hearts section.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';

const GAME = 'hearts';
// One pass and thirteen plays, with room for presses a held trick refuses.
const MAX_MOVES = 40;
const SCREENSHOTS = process.env['HEARTS_SCREENSHOTS'] ?? '';
const COMPASS = ['N', 'E', 'S', 'W'];
const PASS_HIDDEN = 'Passes a card (hidden)';

interface Struct {
  readonly hands: Readonly<Record<string, ReadonlyArray<string | null>>>;
  readonly phase: string;
  readonly passDirection: string;
  readonly currentTrick: readonly unknown[];
}

function structOf(frame: Frame): Struct {
  const view = frame.presentation?.view as {struct?: Struct} | undefined;
  if (view?.struct === undefined) throw new Error('No hearts struct.');
  return view.struct;
}

function seat0(frame: Frame): boolean {
  return frame.session.game === GAME && isViewer(frame.viewer, 0);
}

/** The card codes seat 0 may play or pass at `frame`. */
function legalCodes(frame: Frame): Set<string> {
  const codes = new Set<string>();
  for (const {hint} of frame.presentation?.actions ?? []) {
    const code = (hint as {cardCode?: unknown} | null)?.cardCode;
    if (typeof code === 'string') codes.add(code);
  }
  return codes;
}

/** Checks no frame for seat 0 holds a card of another seat's hand. */
function expectPrivate(frames: readonly Frame[]): number {
  let checked = 0;
  for (const frame of frames) {
    if (!seat0(frame) || frame.presentation === null) continue;
    expect(frame.state_text).toBeNull();
    const {hands} = structOf(frame);
    for (const [seat, compass] of COMPASS.entries()) {
      if (seat === 0) continue;
      expect(
        hands[compass]?.every(card => card === null),
        compass,
      ).toBe(true);
    }
    checked++;
  }
  return checked;
}

/** Checks seat 0 reads the other seats' plays and not their passes. */
function expectPublicHistory(frame: Frame): void {
  const others = frame.history.filter(
    entry => entry.seat !== null && entry.seat !== 0,
  );
  const passes = others.filter(entry => entry.public_label === PASS_HIDDEN);
  const plays = others.filter(entry => entry.public_label !== PASS_HIDDEN);
  expect(passes).toHaveLength(9);
  expect(plays).toHaveLength(39);
  for (const entry of others) expect(entry.action).toBeNull();
  for (const entry of plays) {
    expect(entry.public_label).toMatch(/^([2-9]|10|[TJQKA])[CDHS]$/);
  }
  for (const entry of frame.history.filter(e => e.seat === null)) {
    expect(entry.public_label).toBeNull();
  }
}

/** Waits until no card on the table has moved for two looks in a row. */
async function settled(page: Page): Promise<void> {
  const boxes = (): Promise<string> =>
    page
      .locator('[data-testid="hearts-cards"] [data-card-id]')
      .evaluateAll(cards =>
        cards
          .map(card => {
            const r = card.getBoundingClientRect();
            return `${card.getAttribute('data-card-id') ?? ''}:${String(Math.round(r.x))},${String(Math.round(r.y))}`;
          })
          .join(' '),
      );
  let last = '';
  for (let look = 0; look < 40; look++) {
    const now = await boxes();
    if (now === last) return;
    last = now;
    await page.waitForTimeout(400);
  }
  throw new Error('The table did not settle in 16 s.');
}

async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // The pointer leaves the table (no hover lift) and the board settles its
  // cards in flight and any held trick before the shot.
  await page.mouse.move(2, 2);
  await page.waitForTimeout(900);
  await settled(page);
  await page.screenshot({path: path.join(SCREENSHOTS, `kernel-${name}.png`)});
}

function southCard(table: Locator, code: string): Locator {
  return table.locator(`[data-slot="south"][data-card-id="${code}"]`);
}

// Milliseconds a press waits for its move to be sent before pressing again:
// a held trick refuses a play until it has been collected.
const PRESS_WAIT_MS = 4000;
const PRESS_TRIES = 5;

async function press(page: Page, target: Locator, what: string): Promise<void> {
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
    await target.click({force: true});
    if (await posted) return;
  }
  throw new Error(
    `${what}: no move was sent after ${String(PRESS_TRIES)} presses.`,
  );
}

/** Seat 0's pass: three cards clicked in its hand, then Pass. */
async function pass(
  app: App,
  page: Page,
  table: Locator,
  frame: Frame,
): Promise<void> {
  const ids = await table
    .locator('[data-slot="south"][data-card-id]')
    .evaluateAll(cards => cards.map(card => card.getAttribute('data-card-id')));
  const picks = ids.filter((id): id is string => id !== null).slice(-3);
  for (const code of picks) {
    await southCard(table, code).click({force: true});
    await expect(southCard(table, code)).toHaveAttribute(
      'data-tags',
      /selected/,
    );
  }
  const button = table.locator('.hearts-pass-button');
  await expect(button).toBeEnabled();
  await capture(page, '01-pass');
  await button.click();
  await app.frames.waitFor(
    next => seat0(next) && next.num_plies >= frame.num_plies + 3,
    0,
    "seat 0's pass",
  );
  for (const code of picks) await expect(southCard(table, code)).toHaveCount(0);
}

/** Plays seat 0's hand to the end; returns the phases captured. */
async function playHand(
  app: App,
  page: Page,
  table: Locator,
): Promise<Set<string>> {
  const seen = new Set<string>();
  let plies = -1;
  let plays = 0;
  for (let move = 0; move < MAX_MOVES; move++) {
    const frame = await app.frames.waitFor(
      next =>
        seat0(next) &&
        (next.status === 'terminal' ||
          (next.num_plies > plies &&
            next.to_move.seat === 0 &&
            next.legal_actions !== null)),
      0,
      'a frame where seat 0 moves, or the end of the hand',
    );
    if (frame.status === 'terminal') return seen;
    const struct = structOf(frame);
    if (struct.phase === 'passing') {
      await pass(app, page, table, frame);
      seen.add('pass');
      plies = frame.num_plies + 2;
      continue;
    }
    if (!seen.has('received')) {
      seen.add('received');
      await page.waitForTimeout(1500);
      await capture(page, '02-received');
    }
    const legal = legalCodes(frame);
    const playable = table.locator(
      '[data-slot="south"][data-interactive="true"]',
    );
    await expect(playable.first()).toBeVisible({timeout: 15_000});
    const codes = await playable.evaluateAll(cards =>
      cards.map(card => card.getAttribute('data-card-id')),
    );
    const code = codes.find(c => c !== null && legal.has(c));
    if (code === undefined || code === null) {
      throw new Error(`No playable card is legal: ${codes.join(' ')}`);
    }
    if (plays >= 2 && struct.currentTrick.length >= 2 && !seen.has('mid')) {
      seen.add('mid');
      // The held trick is collected and the staged plays have all landed.
      const want = struct.currentTrick
        .map(entry => (entry as {card: string}).card)
        .sort()
        .join(' ');
      await expect
        .poll(
          () =>
            table
              .locator('[data-slot^="trick-"][data-card-id]')
              .evaluateAll(cards =>
                cards
                  .map(card => card.getAttribute('data-card-id') ?? '')
                  .sort()
                  .join(' '),
              ),
          {timeout: 20_000},
        )
        .toBe(want);
      await capture(page, '03-midtrick');
    }
    await press(page, southCard(table, code), `play ${code}`);
    plays++;
    plies = frame.num_plies;
  }
  throw new Error(`The hand did not end in ${String(MAX_MOVES)} moves.`);
}

test('hearts: seat 0 passes and plays a whole hand on the board against Random', async ({
  app,
  page,
}, testInfo) => {
  test.setTimeout(420_000);
  const width = Number(process.env['HEARTS_VIEWPORT_WIDTH'] ?? '1280');
  await page.setViewportSize({width, height: 800});
  const random = {controller: 'random', config: {}, autoplay: true};
  const first = await app.openSession(
    {
      game: GAME,
      parameters: {},
      seats: [
        {controller: 'human', config: {}, autoplay: false},
        random,
        random,
        random,
      ],
      chance_policy: 'auto',
      seed: 3,
    },
    0,
  );
  expect(structOf(first).passDirection).not.toBe('none');
  const board = (await app.gameplay()).locator('.hearts-gameplay-wrapper');
  const table = board.locator('[data-testid="hearts-table"]');
  await expect(table).toBeVisible();
  const seen = await playHand(app, page, table);
  expect([...seen].sort()).toEqual(['mid', 'pass', 'received']);
  const end = await app.frames.waitFor(
    frame => seat0(frame) && frame.status === 'terminal',
    0,
    'the end of the hand',
  );
  expect(end.returns).toHaveLength(4);
  expectPublicHistory(end);
  await expect(board.locator('.hearts-game-over')).toBeVisible();
  await expect(board.locator('.hearts-game-over__badge').first()).toHaveText(
    'Winner!',
  );
  await capture(page, '04-result');
  await page.screenshot({path: testInfo.outputPath('hearts-light.png')});
  await app.chooseTheme('Dark');
  await app.gameplay();
  await page.screenshot({path: testInfo.outputPath('hearts-dark.png')});
  await app.chooseTheme('Light');
  // The board's own View section.
  const section = await app.showSection('View', 'Hearts');
  await expect(section.getByText('Show all cards')).toBeVisible();
  await expect(section.getByText('Hand shape')).toBeVisible();
  await capture(page, '05-view');
  expect(expectPrivate(app.frames.all())).toBeGreaterThan(10);
});
