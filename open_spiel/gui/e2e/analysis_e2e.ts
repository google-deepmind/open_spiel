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

// Analysis: a game page opens as the
// oracle, every hand on the wire, and View › Perspective only chooses the
// seat the board is drawn from, fetching nothing. The board's own Show all
// cards turns the other hands face up. Chess turns to the seat chosen. The
// catalogue says which games are drawn as a board, and filters by it.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';

const SCREENSHOTS = process.env['ANALYSIS_SCREENSHOTS'] ?? '';

async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // Cards in flight settle before the shot.
  await page.waitForTimeout(700);
  await page.screenshot({path: path.join(SCREENSHOTS, `${name}.png`)});
}

const HUMAN = {controller: 'human', config: {}, autoplay: false};

/** Opens a four-seat game of `game` in analysis, every seat a person. */
async function openAnalysis(app: App, game: string): Promise<void> {
  await app.openSession(
    {
      game,
      parameters: {},
      seats: [HUMAN, HUMAN, HUMAN, HUMAN],
      chance_policy: 'auto',
      seed: 4,
    },
    'oracle',
  );
}

/** Turns the board's Show all cards, in its View section, on or off. */
async function showAllCards(
  app: App,
  section: string,
  on: boolean,
): Promise<void> {
  const view = await app.showSection('View', section);
  const toggle = view.getByRole('checkbox', {name: 'Show all cards'});
  await expect(toggle).toBeEnabled();
  await toggle.setChecked(on);
}

function euchreHand(board: Locator, position: string): Locator {
  return board.locator(`.euchre-player-section--${position} [data-card-id]`);
}

test('euchre analysis: sit at any seat, and show every hand or not', async ({
  app,
  page,
}) => {
  await page.setViewportSize({width: 1280, height: 800});
  await openAnalysis(app, 'euchre');
  const board = (await app.gameplay()).locator('.euchre-gameplay-wrapper');
  await expect(board.locator('.euchre-player-name--bottom')).toHaveText(
    'North',
  );
  await expect(euchreHand(board, 'bottom')).toHaveCount(5);
  // At rest the Perspective names the seat the board shows.
  const perspective = (await app.showSection('View', 'Perspective')).getByRole(
    'combobox',
    {name: 'Perspective'},
  );
  await expect(perspective).toHaveValue('0');
  await expect(perspective.locator('option:checked')).toHaveText('Seat 0');
  await app.gameplay();
  await capture(page, 'euchre-analysis-perspective-rest');

  const frames = app.frames.count;
  await app.sitAt(2);
  await app.gameplay();
  await expect(board.locator('.euchre-player-name--bottom')).toHaveText(
    'South',
  );
  await expect(euchreHand(board, 'bottom')).toHaveCount(5);
  for (const position of ['left', 'top', 'right']) {
    await expect(euchreHand(board, position)).toHaveCount(0);
  }
  // Sitting down fetched nothing; the address keeps the seat.
  expect(app.frames.count).toBe(frames);
  expect(new URL(page.url()).searchParams.get('viewSeat')).toBe('2');
  await capture(page, 'euchre-analysis-seat2-reveal-off');

  await showAllCards(app, 'Euchre', true);
  await app.gameplay();
  for (const position of ['bottom', 'left', 'top', 'right']) {
    await expect(euchreHand(board, position)).toHaveCount(5);
  }
  await capture(page, 'euchre-analysis-seat2-reveal-on');

  // The setting carries to another seat.
  await app.sitAt(1);
  await app.gameplay();
  await expect(board.locator('.euchre-player-name--bottom')).toHaveText('East');
  await expect(euchreHand(board, 'top')).toHaveCount(5);
  await showAllCards(app, 'Euchre', false);
  expect(
    app.frames.all().every(frame => isViewer(frame.viewer, 'oracle')),
  ).toBe(true);
  expect(app.frames.count).toBe(frames);
});

test('hearts analysis: sit at any seat, and show every hand or not', async ({
  app,
  page,
}) => {
  await page.setViewportSize({width: 1280, height: 800});
  await openAnalysis(app, 'hearts');
  const table = (await app.gameplay()).locator('[data-testid="hearts-table"]');
  // A face-up card's id is its code ("Qs"); a back's names its slot.
  const hand = (slot: string): Locator =>
    table.locator(
      `[data-slot="${slot}"][data-card-id]:not([data-card-id*="-"])`,
    );
  await expect(hand('south')).toHaveCount(13);
  const first = await hand('south').evaluateAll(cards =>
    cards.map(card => card.getAttribute('data-card-id')).sort(),
  );

  const frames = app.frames.count;
  await app.sitAt(1);
  await app.gameplay();
  await expect(hand('south')).toHaveCount(13);
  const east = await hand('south').evaluateAll(cards =>
    cards.map(card => card.getAttribute('data-card-id')).sort(),
  );
  expect(east).not.toEqual(first);
  for (const slot of ['west', 'north', 'east']) {
    await expect(hand(slot)).toHaveCount(0);
  }
  await capture(page, 'hearts-analysis-seat1-reveal-off');

  await showAllCards(app, 'Hearts', true);
  await app.gameplay();
  for (const slot of ['south', 'west', 'north', 'east']) {
    await expect(hand(slot)).toHaveCount(13);
  }
  // Seat 0's hand is now on the right of seat 1.
  const right = await hand('east').evaluateAll(cards =>
    cards.map(card => card.getAttribute('data-card-id')).sort(),
  );
  expect(right).toEqual(first);
  await capture(page, 'hearts-analysis-seat1-reveal-on');
  await showAllCards(app, 'Hearts', false);
  expect(app.frames.count).toBe(frames);
});

test('chess analysis: the board turns to the seat chosen', async ({
  app,
  page,
}) => {
  await page.setViewportSize({width: 1280, height: 800});
  await app.openGame('chess');
  const wrap = (await app.gameplay()).getByTestId('chess-board');
  await expect(wrap).toHaveClass(/orientation-white/);
  const frames = app.frames.count;
  // Seat 0 plays Black.
  await app.sitAt(0);
  await app.gameplay();
  await expect(wrap).toHaveClass(/orientation-black/);
  await capture(page, 'chess-analysis-seat0');
  await app.sitAt(1);
  await app.gameplay();
  await expect(wrap).toHaveClass(/orientation-white/);
  await capture(page, 'chess-analysis-seat1');
  expect(app.frames.count).toBe(frames);
});

test('the catalogue badges board games and filters by them', async ({page}) => {
  await page.setViewportSize({width: 1280, height: 900});
  await page.goto('/');
  const games = page.getByRole('list', {name: 'Games'});
  const row = (name: string): Locator =>
    games.locator('.catalogue__game').filter({
      has: page.locator('.catalogue__name', {hasText: new RegExp(`^${name}$`)}),
    });
  await expect(row('chess').locator('.catalogue__badge')).toHaveText(
    'Board UI',
  );
  await expect(row('euchre').locator('.catalogue__badge')).toHaveText(
    'Board UI',
  );
  await expect(row('go').locator('.catalogue__badge')).toHaveText('Text only');
  // The badges line up in one column, as wide as the widest.
  const boxes = await games.locator('.catalogue__badge').evaluateAll(badges =>
    badges.map(badge => {
      const box = badge.getBoundingClientRect();
      return [Math.round(box.left), Math.round(box.width)];
    }),
  );
  expect(new Set(boxes.map(([left]) => left)).size).toBe(1);
  expect(new Set(boxes.map(([, width]) => width)).size).toBe(1);
  // Every game is under exactly one filter, the unservable ones included.
  const counted = await page
    .getByRole('button', {name: /\(\d+\)$/})
    .allTextContents();
  const total = counted
    .map(label => Number(/\((\d+)\)$/.exec(label)?.[1]))
    .reduce((a, b) => a + b, 0);
  expect(counted).toHaveLength(3);
  expect(total).toBe(boxes.length);
  await capture(page, 'lobby-all');

  await page.getByRole('button', {name: /^Board UI \(\d+\)$/}).click();
  await expect(row('go')).toHaveCount(0);
  const badges = await games.locator('.catalogue__badge').allTextContents();
  expect(badges.length).toBeGreaterThanOrEqual(9);
  expect(new Set(badges)).toEqual(new Set(['Board UI']));
  await capture(page, 'lobby-board-ui');

  await page.getByRole('button', {name: /^Text only \(\d+\)$/}).click();
  await expect(row('chess')).toHaveCount(0);
  await expect(row('go')).toHaveCount(1);
  await capture(page, 'lobby-text-only');
});
