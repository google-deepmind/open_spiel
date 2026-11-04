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

// A whole hand of gin rummy played on the gin board against Random, on the
// real server: seat 0 plays every move with the board's own gestures (Pass,
// a double-click on the stock, a double-click on a card of its grid, Knock,
// a double-click to stage the knocker's discard, Done), never the legal
// list, and knocks as soon as it can. Every frame seat 0 receives is checked
// to hold none of seat 1's hidden cards; seat 1's moves reach it under their
// public labels, which draw the board's last-turn cue, and the cards dealt
// and drawn from the stock stay hidden.
//
// The knocker's lay-down is checked before it is sent: a composition that
// leaves more deadwood than the knock card holds Done back and says why,
// and nothing is sent until it is mended. gin_rummy_defender_e2e.ts plays
// the other side of the knock: the lay-offs.
//
// With GIN_SCREENSHOTS set to a directory, the table is captured there at
// 1280 by 800 pixels: the opening deal, mid-hand with a meld, the lay-down,
// the result, the View panel and the felt texture.

/// <reference types="node" />

import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {type App, RANDOM} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';
import {
  GAME,
  MAX_MOVES,
  type GinView,
  allMelds,
  bestMelds,
  readView,
  points,
  button,
  handCard,
  turnLine,
  chooseMove,
  expectLiveIsTurnLine,
  capture,
  press,
  record,
} from './gin_board.ts';

const HIDDEN = 'XX';

/**
 * Checks no frame for seat 0 shows a card of seat 1's hand while that hand
 * is closed, nor any analysis or lay-down plan of seat 1's.
 */
function expectNoHiddenCards(frames: readonly Frame[]): number {
  let checked = 0;
  for (const frame of frames) {
    if (frame.session.game !== GAME || !isViewer(frame.viewer, 0)) continue;
    const view = readView(frame);
    const analysed = Object.keys(
      record(view.view['ginAnalysisMeldsBySeat'], 'melds'),
    );
    if (view.visible.has(1)) continue;
    expect(view.hands[1]?.every(card => card === HIDDEN)).toBe(true);
    expect(analysed).not.toContain('1');
    checked++;
  }
  return checked;
}

/** Whether seat 1's last turn, as seat 0 reads it, drew from a pile. */
function opponentDrewLast(frame: Frame): boolean {
  const turn: string[] = [];
  for (const entry of [...frame.history].reverse()) {
    if (entry.seat === null) continue;
    if (entry.seat !== 1) break;
    turn.push(entry.public_label ?? '');
  }
  return turn.some(label => /Action: Draw (upcard|stock)$/.test(label));
}

/**
 * Checks seat 0 reads each of seat 1's moves under its public label, and of
 * the chance outcomes only the first upcard, dealt after 20 hidden cards.
 */
function expectPublicHistory(frame: Frame): void {
  const opponent = frame.history.filter(entry => entry.seat === 1);
  expect(opponent.length).toBeGreaterThan(5);
  for (const entry of opponent) {
    expect(entry.action).toBeNull();
    expect(entry.public_label).toMatch(/^Player: 1 Action: /);
  }
  const publicChance = frame.history.flatMap((entry, index) =>
    entry.seat === null && entry.public_label !== null ? [index] : [],
  );
  expect(publicChance).toEqual([20]);
}

/**
 * At the knocker's discard stage, with the discard staged and the best melds
 * staged with it: takes back the heaviest meld whose loss leaves more
 * deadwood than the knock card, checks Done is held back with the reason on
 * the turn line and that nothing is sent, then stages the meld (and the
 * discard) again. False when no meld's loss would go over the card.
 */
/** The stats region's Deadwood line. */
function deadwoodShown(board: Locator): Locator {
  return board.locator('[data-region-id="stats"]').getByText(/^Deadwood:/);
}

/** Where the supply row's stats and the player's grid panel are drawn. */
async function fixedBoxes(board: Locator): Promise<string> {
  const boxes = await Promise.all(
    ['stats', 'me-panel'].map(id =>
      board.locator(`[data-region-id="${id}"]`).boundingBox(),
    ),
  );
  return JSON.stringify(boxes);
}

async function overKnockHeldBack(
  page: Page,
  board: Locator,
  view: GinView,
  discard: string,
): Promise<boolean> {
  const hand = (view.hands[0] ?? []).filter(code => code !== discard);
  const melds = allMelds(view.view, 0);
  const best = bestMelds(view.view, 0).filter(m => !m.includes(discard));
  const melded = new Set(best.flat());
  const deadwood = hand
    .filter(code => !melded.has(code))
    .reduce((sum, code) => sum + points(code), 0);
  const choices = best
    .map(meld => ({
      meld,
      over: deadwood + meld.reduce((sum, code) => sum + points(code), 0),
      // A card of this meld and of no other: one double-click unstages it.
      card: meld.find(code => melds.filter(m => m.includes(code)).length === 1),
    }))
    .filter(choice => choice.over > view.knockCard && choice.card !== undefined)
    .sort((a, b) => b.over - a.over);
  // The Deadwood readout is what the staged lay-down leaves: the hand less
  // the staged discard and the staged (best) melds, not the engine's count
  // of every card not yet laid.
  await expect(deadwoodShown(board)).toHaveText(
    `Deadwood: ${String(deadwood)}`,
  );
  const choice = choices[0];
  if (choice?.card === undefined) return false;
  const card = choice.card;
  const done = button(board, 'Done');
  let posts = 0;
  const count = (request: {method: () => string; url: () => string}): void => {
    if (request.method() === 'POST' && request.url().includes('/actions')) {
      posts++;
    }
  };
  page.on('request', count);
  await handCard(board, card).dblclick({force: true});
  await expect(done).toBeDisabled();
  await expect(deadwoodShown(board)).toHaveText(
    `Deadwood: ${String(choice.over)}`,
  );
  await expect(turnLine(board)).toHaveText(
    `Deadwood ${String(choice.over)} is over the knock card (${String(view.knockCard)}): stage more melds or discard another card, then Done.`,
  );
  await capture(page, '03a-over-knock');
  await done.click({force: true});
  // Restaging: the card's cycle may pass through "staged as the discard",
  // which a meld holding it then clears; the chosen discard is restaged.
  for (let tries = 0; tries < 6; tries++) {
    const text = (await turnLine(board).textContent()) ?? '';
    if (text.startsWith(`Discarding ${discard}.`) && (await done.isEnabled())) {
      break;
    }
    const target = text.startsWith('Double-click your discard')
      ? discard
      : card;
    await handCard(board, target).dblclick({force: true});
    await page.waitForTimeout(300);
  }
  await expect(turnLine(board)).toHaveText(
    `Discarding ${discard}. Stage melds, then Done.`,
  );
  await expect(done).toBeEnabled();
  page.off('request', count);
  expect(posts, 'moves sent while Done was held back').toBe(0);
  return true;
}

/**
 * Waits until seat 0's lay-down, sent as one sequence of actions, is played:
 * the turn has left seat 0 or the hand is over.
 *
 * @returns The plies played by then.
 */
async function laidDown(app: App, after: number): Promise<number> {
  const frame = await app.frames.waitFor(
    next =>
      next.session.game === GAME &&
      isViewer(next.viewer, 0) &&
      next.num_plies > after &&
      (next.status === 'terminal' || next.to_move.seat !== 0),
    0,
    "the end of seat 0's lay-down",
  );
  return frame.num_plies;
}

/** Captures the opening deal and a mid-hand meld, once each. */
async function captureStage(
  page: Page,
  view: GinView,
  captured: Set<string>,
): Promise<void> {
  if (view.phase === 'FirstUpcard' && !captured.has('opening')) {
    captured.add('opening');
    await capture(page, '01-opening');
  }
  if (
    view.phase === 'Discard' &&
    view.melded.size > 0 &&
    !captured.has('mid')
  ) {
    captured.add('mid');
    await capture(page, '02-midhand');
  }
}

/**
 * Plays seat 0's moves to the end of the hand; returns what it saw. boxes[0]
 * is set to the stats and grid boxes (fixedBoxes) at each of its turns.
 */
async function playHand(
  app: App,
  page: Page,
  board: Locator,
  boxes: string[],
): Promise<ReadonlySet<string>> {
  let plies = -1;
  const captured = new Set<string>();
  for (let move = 0; move < MAX_MOVES; move++) {
    const frame = await app.frames.waitFor(
      next =>
        next.session.game === GAME &&
        isViewer(next.viewer, 0) &&
        (next.status === 'terminal' ||
          (next.num_plies > plies &&
            next.to_move.seat === 0 &&
            next.legal_actions !== null)),
      0,
      'a frame where seat 0 moves, or the end of the hand',
    );
    if (frame.status === 'terminal') return captured;
    boxes[0] = await fixedBoxes(board);
    const view = readView(frame);
    if (
      view.phase === 'Draw' &&
      !captured.has('cue') &&
      opponentDrewLast(frame)
    ) {
      // The other seat's draw and discard are marked on the table.
      captured.add('cue');
      await expect(board.locator('.gin-move-cue').first()).toBeVisible();
    }
    await captureStage(page, view, captured);
    const chosen = chooseMove(board, frame, view);
    if (chosen.what.startsWith('stage ')) {
      // Staging the discard sends nothing; the composition then shows the
      // discard and the melds, and Done lays them down, then passes.
      await chosen.target.dblclick({force: true});
      await expect(button(board, 'Done')).toBeEnabled();
      const discard = chosen.what.split(' ')[1] ?? '';
      await expectLiveIsTurnLine(board, `${discard} staged`);
      if (await overKnockHeldBack(page, board, view, discard)) {
        captured.add('held-back');
        await expectLiveIsTurnLine(board, 'the lay-down restaged');
      }
      if (!captured.has('knock')) {
        captured.add('knock');
        await capture(page, '03-knock');
      }
      await press(page, {
        target: button(board, 'Done'),
        what: 'done',
        double: false,
      });
      plies = await laidDown(app, frame.num_plies);
      continue;
    }
    if (chosen.what === 'done' && !captured.has('knock')) {
      captured.add('knock');
      await capture(page, '03-knock');
    }
    await press(page, chosen);
    if (chosen.what === 'knock') {
      await expectLiveIsTurnLine(board, 'after Knock');
      captured.add('knock-status');
    }
    plies =
      chosen.what === 'done'
        ? await laidDown(app, frame.num_plies)
        : frame.num_plies;
  }
  throw new Error(`The hand did not end in ${String(MAX_MOVES)} moves.`);
}

test('gin_rummy: seat 0 plays a whole hand on the board against Random', async ({
  app,
  page,
}) => {
  test.setTimeout(420_000);
  // GIN_VIEWPORT_WIDTH narrows the page, to compare the board at another
  // panel width.
  const width = Number(process.env['GIN_VIEWPORT_WIDTH'] ?? '1280');
  await page.setViewportSize({width, height: 800});
  await app.openSession(
    {
      game: GAME,
      parameters: {oklahoma: true},
      seats: [
        {controller: 'human', config: {}, autoplay: false},
        {controller: 'random', config: {}, autoplay: false},
      ],
      chance_policy: 'auto',
      seed: 7,
    },
    0,
  );
  const board = (await app.gameplay()).locator('.gin-render-wrapper');
  await expect(board).toBeVisible();
  // Random starts once the page has the first frame, so the page sees the
  // first upcard before anyone can take it.
  await app.setController(1, RANDOM, true);
  const boxes: string[] = [];
  const seen = await playHand(app, page, board, boxes);
  expect(seen.has('cue')).toBe(true);
  expect(seen.has('knock-status')).toBe(true);
  const end = await app.frames.waitFor(
    frame =>
      frame.session.game === GAME &&
      isViewer(frame.viewer, 0) &&
      frame.status === 'terminal',
    0,
    'the end of the hand',
  );
  expect(end.history.length).toBeGreaterThan(20);
  expectPublicHistory(end);
  await expect(button(board, 'Result')).toBeVisible();
  // The hand's end reveals the opponent's hand in the room kept for it: the
  // supply row and the player's own grid stay where they were.
  expect(await fixedBoxes(board)).toBe(boxes[0]);
  await capture(page, '04-result');
  // The board's own View sections: Visuals, Gin Rummy, Perspective,
  // Animation.
  const visuals = await app.showSection('View', 'Visuals');
  await expect(visuals.getByText('Card style')).toBeVisible();
  await capture(page, '05-view');
  // The felt texture, an SVG noise tile, paints over the felt colour.
  await visuals.getByLabel('Table texture').selectOption('felt');
  await expect(board.locator('.gin-render-surface')).toHaveCSS(
    'background-image',
    /data:image\/svg\+xml/,
  );
  await capture(page, '06-felt');
  expect(expectNoHiddenCards(app.frames.all())).toBeGreaterThan(5);
});
