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

// Actions against the running server: History before the first action and
// while a request is in flight, and chance outcomes chosen from Legal.

import type {Request} from '@playwright/test';

import type {CreateSession, Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {expect, test} from './fixtures.ts';

const HUMAN_SEAT = {controller: 'human', config: {}, autoplay: false};

/** Kuhn poker with the deal chosen in the browser. */
const EXPOSED_KUHN: CreateSession = {
  game: 'kuhn_poker',
  parameters: {},
  seats: [HUMAN_SEAT, HUMAN_SEAT],
  chance_policy: 'expose',
  seed: 1,
};

/** Plays `moves` first legal actions from Actions › Legal, as the oracle. */
async function playMoves(app: App, moves: number): Promise<Frame> {
  const legal = await app.legal();
  let frame: Frame | null = null;
  for (let move = 0; move < moves; move++) {
    const before = app.frames.count;
    await app.legalButtons(legal).first().click();
    frame = await app.frames.waitFor(
      next => next.num_plies === move + 1,
      before,
      `the frame after move ${String(move + 1)}`,
    );
  }
  if (frame === null) throw new Error('No move was played.');
  return frame;
}

test('History says "No actions yet." to a seat before the first action', async ({
  app,
}) => {
  const first = await app.openSession(EXPOSED_KUHN, 0);
  expect(first.omniscient).toBe(false);
  expect(first.history).toEqual([]);
  const history = await app.showSection('Actions', 'History');
  await expect(history.getByText('No actions yet.')).toBeVisible();
  await expect(history.getByRole('list', {name: 'History'})).toHaveCount(0);
});

test('History ignores a jump while a request is in flight', async ({
  app,
  page,
}) => {
  await app.openGame('tic_tac_toe');
  await playMoves(app, 3);
  const history = await app.showSection('Actions', 'History');
  const rows = history.getByRole('list', {name: 'History'}).getByRole('button');
  // The initial position, then each move.
  await expect(rows).toHaveCount(4);
  const first = history.getByRole('button', {name: /^#1\b/});

  // Hold the first replay request until the jump has been tried.
  let release: () => void = () => undefined;
  const held = new Promise<void>(resolve => {
    release = resolve;
  });
  await page.route(
    '**/api/sessions/*/plies/*',
    async route => {
      await held;
      await route.continue();
    },
    {times: 1},
  );
  const replays: Request[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.includes('/plies/')) {
      replays.push(request);
    }
  });
  await (
    await app.replay()
  )
    .getByRole('button', {name: 'Previous move'})
    .click();
  await expect(first).toHaveAttribute('aria-disabled', 'true');
  // Forced: the click is made, as a person's would be, though the row says
  // it is disabled.
  await first.click({force: true});
  expect(replays.map(request => new URL(request.url()).pathname)).toEqual([
    expect.stringMatching(/\/plies\/2$/),
  ]);

  const before = app.frames.count;
  release();
  await app.frames.waitFor(frame => frame.ply === 2, before, 'ply 2');
  await expect(first).not.toHaveAttribute('aria-disabled');
  expect(replays).toHaveLength(1);
});

test('Legal offers the chance outcomes the browser picks, with their odds', async ({
  app,
}) => {
  const first = await app.openSession(EXPOSED_KUHN, 'oracle');
  expect(first.to_move.kind).toBe('chance');
  expect(first.chance_outcomes).toHaveLength(3);
  const legal = await app.legal();
  const list = legal.getByRole('list', {name: 'Legal (3)'});
  const outcomes = list.getByRole('button');
  await expect(outcomes).toHaveCount(3);
  for (const outcome of await outcomes.all()) {
    await expect(outcome).toContainText('p=0.33');
  }

  const before = app.frames.count;
  await outcomes.first().click();
  const dealt = await app.frames.waitFor(
    frame => frame.num_plies === 1,
    before,
    'the frame after the first deal',
  );
  expect(dealt.history[0]?.action).toBe(first.chance_outcomes?.[0]?.id);
  const second = legal
    .getByRole('list', {name: 'Legal (2)'})
    .getByRole('button');
  await expect(second).toHaveCount(2);
  for (const outcome of await second.all()) {
    await expect(outcome).toContainText('p=0.50');
  }
});
