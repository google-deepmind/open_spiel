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

// Setup › Players against the running server: stepping a controller,
// retrying one that stopped, a seat list that keeps its group tall enough,
// and an unsaved draft dropped by a move.

import type {Locator} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {type App, RANDOM} from './app.ts';
import {expect, test} from './fixtures.ts';

const HUMAN_SEAT = {controller: 'human', config: {}, autoplay: false};

/** Plays the first legal action from Actions › Legal; returns the frame. */
async function playFirst(app: App): Promise<Frame> {
  const legal = await app.legal();
  const before = app.frames.count;
  const plies = app.frames.all().at(-1)?.num_plies ?? 0;
  await app.legalButtons(legal).first().click();
  return app.frames.waitFor(
    frame => frame.num_plies === plies + 1,
    before,
    'the frame after the move',
  );
}

function stepButton(row: Locator): Locator {
  return row.getByRole('button', {name: 'Step', exact: true});
}

test('Players: Step lets a waiting controller move, on its turn only', async ({
  app,
}) => {
  await app.openGame('tic_tac_toe');
  await app.setController(1, RANDOM, false);
  const row = await app.seat(1);
  await expect(stepButton(row)).toBeDisabled();
  await expect(stepButton(row)).toHaveAttribute(
    'title',
    'Not this player’s turn',
  );

  await playFirst(app);
  const seatRow = await app.seat(1);
  await expect(stepButton(seatRow)).toBeEnabled();
  const before = app.frames.count;
  await stepButton(seatRow).click();
  const stepped = await app.frames.waitFor(
    frame => frame.num_plies === 2,
    before,
    'the frame after Player 1 steps',
  );
  expect(stepped.history[1]?.seat).toBe(1);
  await expect(stepButton(seatRow)).toBeDisabled();
});

test('Players: a stopped seat says why and offers Retry, which plays it', async ({
  app,
  page,
}) => {
  await app.openGame('tic_tac_toe');
  await app.setController(1, RANDOM, false);
  // No controller the server offers fails on tic_tac_toe, so the answer to
  // the next move is changed to the frame of a seat that stopped.
  await page.route(
    '**/api/sessions/*/actions?*',
    async route => {
      const response = await route.fetch();
      const frame = (await response.json()) as Frame;
      const failed: Frame = {
        ...frame,
        status: 'paused',
        seats: frame.seats.map(seat =>
          seat.seat === 1
            ? {...seat, failure: {error: 'ValueError', message: 'No move.'}}
            : seat,
        ),
      };
      await route.fulfill({response, json: failed});
    },
    {times: 1},
  );
  await playFirst(app);
  const row = await app.seat(1);
  await expect(row.getByRole('alert')).toContainText(
    'Player 1’s controller stopped with ValueError. Play is paused.',
  );
  await expect(row.getByRole('alert')).toContainText('No move.');
  await expect(stepButton(row)).toHaveCount(0);
  const retry = row.getByRole('button', {name: 'Retry', exact: true});
  await expect(retry).toBeEnabled();

  const before = app.frames.count;
  await retry.click();
  const played = await app.frames.waitFor(
    frame => frame.num_plies === 2,
    before,
    'the frame after the retried seat moves',
  );
  expect(played.seats[1]?.failure).toBeNull();
  expect(played.history[1]?.seat).toBe(1);
  await expect(retry).toHaveCount(0);
  await expect(row.getByRole('alert')).toHaveCount(0);
});

test('Players: the Setup group grows to show every seat', async ({app}) => {
  // As many as the column has room for once the rest keeps its reserve.
  const seats = 7;
  await app.openSession(
    {
      game: 'kuhn_poker',
      parameters: {players: seats},
      seats: Array.from({length: seats}, () => HUMAN_SEAT),
      chance_policy: 'auto',
      seed: 1,
    },
    'oracle',
  );
  const group = await app.showPanel('Setup');
  await app.showSection('Setup', 'Position');
  const floor = (await group.boundingBox())?.height ?? 0;
  const players = await app.showSection('Setup', 'Players');
  const list = players.getByRole('list', {name: 'Seats'});
  await expect(list.getByRole('listitem')).toHaveCount(seats);
  await expect
    .poll(async () => (await group.boundingBox())?.height ?? 0)
    .toBeGreaterThan(floor);
  // Every seat, collapsed, is in sight without scrolling.
  const scroll = players.locator('.players__scroll');
  await expect
    .poll(() =>
      scroll.evaluate(element => element.scrollHeight - element.clientHeight),
    )
    .toBeLessThanOrEqual(1);
  const last = list.getByRole('listitem').last();
  await expect(last).toBeInViewport({ratio: 1});
});

test('Players: a move drops an unsaved draft', async ({app}) => {
  await app.openGame('tic_tac_toe');
  const row = await app.seat(1);
  await row.getByRole('button', {name: 'Settings', exact: true}).click();
  const controller = row.getByRole('combobox', {name: 'Controller'});
  await controller.selectOption({label: RANDOM.label});
  const state = row.getByRole('status');
  await expect(state).toHaveText('Unsaved changes');

  await playFirst(app);
  const after = await app.seat(1);
  await expect(after.getByRole('status')).toHaveText('No changes');
  await expect(after.getByRole('combobox', {name: 'Controller'})).toHaveValue(
    'human',
  );
});
