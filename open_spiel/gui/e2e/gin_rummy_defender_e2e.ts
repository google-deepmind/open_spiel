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

// A hand of gin rummy with seat 0 on the defending side: seat 1, a human
// played through the server's API, knocks as soon as it may and lays down
// the engine's first plan; seat 0 lays off onto its melds with the board's
// gestures (a double-click on each card, in the engine's order), lays its
// own melds with Done, and reads the hand's result. The first Done is
// refused by the server after its lay-offs were played: the page then shows
// where the hand stands, says so once, over the board without moving it,
// and the next Done finishes the hand.

/// <reference types="node" />

import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';
import {
  GAME,
  MAX_MOVES,
  type GinView,
  bestMelded,
  plansOf,
  readView,
  points,
  legal,
  button,
  handCard,
  turnLine,
  chooseMove,
  capture,
  press,
} from './gin_board.ts';

/** Seat 1's frame from the server, as seat 1 sees it. */
async function seatFrame(page: Page, id: string, seat: number): Promise<Frame> {
  const response = await page.request.get(
    `/api/sessions/${id}?viewer=${String(seat)}`,
  );
  expect(response.ok(), await response.text()).toBe(true);
  return (await response.json()) as Frame;
}

async function seatAct(
  page: Page,
  id: string,
  seat: number,
  action: number,
): Promise<Frame> {
  const response = await page.request.post(
    `/api/sessions/${id}/actions?viewer=${String(seat)}`,
    {data: {action, ply: null}},
  );
  expect(response.ok(), await response.text()).toBe(true);
  return (await response.json()) as Frame;
}

/** Seat 1's discard: its highest card outside its best melds. */
function seatDiscard(frame: Frame, view: GinView, seat: number): number {
  const hand = view.hands[seat] ?? [];
  const melded = bestMelded(view.view, seat);
  const choices = hand
    .filter(code => legal(frame, code) !== null)
    .sort(
      (a, b) =>
        Number(melded.has(a)) - Number(melded.has(b)) || points(b) - points(a),
    );
  const action = choices[0] === undefined ? null : legal(frame, choices[0]);
  if (action === null) throw new Error(`Seat ${String(seat)} cannot discard.`);
  return action;
}

/**
 * Plays seat 1's turn through the API, a human knocking as soon as it may
 * and laying down the engine's first plan; stops when seat 1 is not to move.
 */
async function seatOneTurn(page: Page, id: string): Promise<void> {
  for (let move = 0; move < 20; move++) {
    const frame = await seatFrame(page, id, 1);
    if (frame.status === 'terminal' || frame.to_move.seat !== 1) return;
    const view = readView(frame);
    const knock = legal(frame, 'Knock');
    const pass = legal(frame, 'Pass');
    if (knock !== null) {
      await seatAct(page, id, 1, knock);
    } else if (view.stage === 'discard') {
      await seatAct(page, id, 1, seatDiscard(frame, view, 1));
    } else if (view.stage === 'melds') {
      for (const step of plansOf(view.view)[0] ?? []) {
        await seatAct(page, id, 1, step.action);
      }
      const passAction = view.view['ginLaydownPassAction'];
      if (typeof passAction !== 'number') {
        throw new Error('Seat 1 cannot pass.');
      }
      await seatAct(page, id, 1, passAction);
    } else if (pass !== null) {
      await seatAct(page, id, 1, pass);
    } else if (legal(frame, 'Draw stock') !== null) {
      await seatAct(page, id, 1, legal(frame, 'Draw stock') ?? -1);
    } else {
      await seatAct(page, id, 1, seatDiscard(frame, view, 1));
    }
  }
  throw new Error("Seat 1's turn did not end in 20 moves.");
}

/**
 * At seat 0's lay-off stage: stages the largest set of lay-offs whose cards
 * are in none of its staged melds, by double-clicking them in the engine's
 * order, and checks the turn line counts them.
 *
 * @returns The cards staged.
 */
async function stageLayoffs(
  page: Page,
  board: Locator,
  view: GinView,
): Promise<string[]> {
  const melded = bestMelded(view.view, 0);
  const plans = plansOf(view.view)
    .filter(plan =>
      plan.every(
        step => step.cards.length === 1 && !melded.has(step.cards[0] ?? ''),
      ),
    )
    .sort((a, b) => b.length - a.length);
  const cards = (plans[0] ?? []).map(step => step.cards[0] ?? '');
  for (const code of cards) {
    await handCard(board, code).dblclick({force: true});
    await page.waitForTimeout(300);
  }
  if (cards.length > 0) {
    await expect(turnLine(board)).toHaveText(
      new RegExp(`^Laying off ${String(cards.length)} cards?|lay off`),
    );
  }
  return cards;
}

// Draw upcard: never legal at a lay-off, so the server refuses it (409).
const NOT_LEGAL_AT_LAYOFF = 52;

/**
 * Presses Done with the lay-offs staged, the server refusing the pass that
 * closes them, after the lay-offs before it were played; checks the page
 * shows the position the server is at, with one notice over the board that
 * leaves the cards where they were.
 */
async function refusedLayoffs(
  app: App,
  page: Page,
  board: Locator,
  frame: Frame,
): Promise<void> {
  const pass = readView(frame).view['ginLaydownPassAction'];
  if (typeof pass !== 'number') throw new Error('Seat 0 cannot pass.');
  let armed = true;
  const pattern = '**/api/sessions/*/actions*';
  await page.route(pattern, async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    if (!armed || body['action'] !== pass) {
      await route.continue();
      return;
    }
    armed = false;
    const postData = JSON.stringify({...body, action: NOT_LEGAL_AT_LAYOFF});
    await route.continue({postData});
  });
  const top = (await board.boundingBox())?.y;
  await press(page, {
    target: button(board, 'Done'),
    what: 'done',
    double: false,
  });
  const notice = page.locator('.gin-notice');
  await expect(notice).toHaveText(
    /^The lay-down was not accepted: .*not legal.*\. The table now shows where the hand stands; check your melds, then press Done\.$/,
  );
  await expect(notice).not.toContainText('..');
  await page.unroute(pattern);
  // One surface: the Gameplay panel's own error notice says nothing.
  await expect(
    page.locator('.gameplay__notice[data-tone="error"]'),
  ).toHaveCount(0);
  expect((await board.boundingBox())?.y).toBe(top);
  const server = await seatFrame(page, frame.session.id, 0);
  expect(server.num_plies).toBeGreaterThan(frame.num_plies);
  await app.frames.waitFor(
    next =>
      next.session.id === frame.session.id &&
      isViewer(next.viewer, 0) &&
      next.num_plies === server.num_plies,
    0,
    'the page at the position the server is at',
  );
  await capture(page, '07a-layoff-refused');
}

/** Plays seat 1's turn off the page, then shows the page the session afresh. */
async function seatOneOffPage(app: App, page: Page, id: string): Promise<void> {
  await seatOneTurn(page, id);
  const plies = (await seatFrame(page, id, 0)).num_plies;
  const before = Math.max(0, app.frames.count - 1);
  await page.reload();
  await app.frames.waitFor(
    next =>
      next.session.id === id &&
      isViewer(next.viewer, 0) &&
      next.num_plies === plies,
    before,
    'the page after seat 1 moved',
  );
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
      parameters: {},
      seats: [
        {controller: 'human', config: {}, autoplay: false},
        {controller: 'human', config: {}, autoplay: false},
      ],
      chance_policy: 'auto',
      seed: Number(process.env['GIN_DEFENDER_SEED'] ?? '11'),
    },
    0,
  );
  return first.session.id;
}

test('gin_rummy: seat 0 lays off and melds after seat 1 knocks', async ({
  app,
  page,
  health,
}) => {
  test.setTimeout(420_000);
  // The refusal this test asks for, and the browser's line about it.
  health.allow(/409/);
  await page.setViewportSize({width: 1280, height: 800});
  const id = await openHand(app);
  const board = (await app.gameplay()).locator('.gin-render-wrapper');
  await expect(board).toBeVisible();
  let laidOff: string[] | null = null;
  let refused = false;
  for (let move = 0; move < MAX_MOVES; move++) {
    const server = await seatFrame(page, id, 0);
    if (server.status === 'terminal') break;
    if (server.to_move.seat === 1) {
      await seatOneOffPage(app, page, id);
      continue;
    }
    const frame = await app.frames.waitFor(
      next =>
        next.session.id === id &&
        isViewer(next.viewer, 0) &&
        next.num_plies === server.num_plies,
      0,
      "seat 0's turn on the page",
    );
    const view = readView(frame);
    if (view.stage === 'layoffs') {
      const staged = await stageLayoffs(page, board, view);
      laidOff = [...(laidOff ?? []), ...staged];
      await capture(page, '07-layoffs');
      if (!refused && staged.length > 0) {
        refused = true;
        await refusedLayoffs(app, page, board, frame);
        continue;
      }
    }
    const chosen = chooseMove(board, frame, view, false);
    await press(page, chosen);
    if (chosen.what === 'done') {
      // Done walks the lay-offs, then the melds: it ends when the turn
      // leaves seat 0 or the hand is over.
      await app.frames.waitFor(
        next =>
          next.session.id === id &&
          isViewer(next.viewer, 0) &&
          next.num_plies > frame.num_plies &&
          (next.status === 'terminal' || next.to_move.seat !== 0),
        0,
        "the end of seat 0's lay-down",
      );
    }
  }
  expect(
    laidOff,
    'seat 1 knocked and seat 0 reached its lay-offs',
  ).not.toBeNull();
  expect(refused, 'a lay-off was staged, and its Done refused').toBe(true);
  const end = await app.frames.waitFor(
    frame =>
      frame.session.id === id &&
      isViewer(frame.viewer, 0) &&
      frame.status === 'terminal',
    0,
    'the end of the hand',
  );
  // Every staged lay-off was laid: none of them is left in seat 0's hand.
  const hand = readView(end).hands[0] ?? [];
  for (const code of laidOff ?? []) expect(hand).not.toContain(code);
  await expect(button(board, 'Result')).toBeVisible();
  await capture(page, '08-defender-result');
});
