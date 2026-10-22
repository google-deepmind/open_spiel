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

// Each panel doing its one job in the running client: the catalogue, Setup
// with its Players, Actions, State with the replay control, copying, the
// theme, saved layouts and popping a section out.

import type {Frame} from '../src/contract/contract.ts';
import {type App, HUMAN, RANDOM} from './app.ts';
import {expect, test} from './fixtures.ts';

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

function stateText(frame: Frame): string {
  if (frame.state_text === null) throw new Error('The oracle sees the state.');
  return frame.state_text;
}

test('the catalogue opens a game', async ({app, page}) => {
  await page.goto('/');
  await page.getByRole('link', {name: /^Tic Tac Toe\b/}).click();
  await expect(page).toHaveURL(/\/game\/tic_tac_toe$/);
  await expect(app.dockTab('Gameplay')).toBeVisible();
});

test('Setup changes a parameter and starts a new game', async ({app}) => {
  const first = await app.openGame('go');
  const options = await app.showSection('Setup', 'Game options');
  // A change applies at once: there is nothing to press.
  await expect(options.getByRole('button')).toHaveCount(0);
  await expect(options).toContainText('Changing one starts a new game.');
  const next = await app.startWithParameter('board_size', 'Board Size', '13');
  expect(next.session.id).not.toBe(first.session.id);
  expect(next.session.num_seats).toBe(first.session.num_seats);
  expect(next.legal_actions).toHaveLength(13 * 13 + 1);
});

test('Setup sets the number of players, and Players lists the new seats', async ({
  app,
}) => {
  const first = await app.openGame('kuhn_poker');
  expect(first.session.num_seats).toBe(2);
  const options = await app.showSection('Setup', 'Game options');
  const players = options.getByRole('spinbutton', {
    name: 'Players',
    exact: true,
  });
  await expect(players).toHaveValue('2');
  await expect(players).toHaveAttribute('min', '2');
  await expect(players).toHaveAttribute('max', '10');
  const next = await app.startWithParameter('players', 'Players', '3');
  expect(next.session.id).not.toBe(first.session.id);
  expect(next.session.num_seats).toBe(3);
  expect(next.seats).toHaveLength(3);
  const seats = (await app.showSection('Setup', 'Players')).getByRole('list', {
    name: 'Seats',
  });
  await expect(seats.getByRole('listitem')).toHaveCount(3);
  await expect(seats).toContainText('Player 2');
});

test('Setup offers no Game options for a game with a fixed number of seats and no other parameter', async ({
  app,
}) => {
  await app.openGame('tic_tac_toe');
  const group = await app.showPanel('Setup');
  await expect(
    group.getByRole('tablist', {name: 'Setup sections'}).getByRole('tab'),
  ).toHaveText(['Position', 'Players']);
});

test('Players switches a seat to another controller and back', async ({
  app,
}) => {
  await app.openGame('tic_tac_toe');
  await app.setViewer('oracle');
  await app.setController(1, RANDOM, false);
  const row = await app.seat(1);
  await expect(row.locator('.seat__controller')).toHaveText('Random');
  await expect(
    row.getByRole('button', {name: 'Step', exact: true}),
  ).toBeVisible();
  await app.setController(1, HUMAN, false);
  await expect(row.locator('.seat__controller')).toHaveText('Human');
  await expect(
    row.getByRole('button', {name: 'Step', exact: true}),
  ).toHaveCount(0);
});

test('Actions: a long legal list is windowed, with no filter', async ({
  app,
}) => {
  await app.openGame('go');
  await app.setViewer('oracle');
  // 19x19 go: 361 points and a pass, past the windowing threshold.
  const frame = await app.startWithParameter('board_size', 'Board Size', '19');
  expect(frame.legal_actions).toHaveLength(362);
  const legal = await app.showSection('Actions', 'Legal');
  await expect(legal.getByRole('searchbox')).toHaveCount(0);
  const list = legal.getByRole('list', {name: 'Legal (362)'});
  const rows = list.getByRole('listitem');
  await expect(rows.first()).toHaveAttribute('aria-setsize', '362');
  expect(await rows.count()).toBeLessThan(120);
  await expect(
    list.getByRole('listitem').filter({hasText: 'PASS'}),
  ).toHaveCount(0);
  await list.evaluate(element => {
    element.scrollTop = element.scrollHeight;
  });
  await expect(rows.filter({hasText: /B PASS$/})).toHaveAttribute(
    'aria-posinset',
    '362',
  );
});

test('State shows the text; History steps back and goes to the end', async ({
  app,
}) => {
  await app.openGame('tic_tac_toe');
  await app.setViewer('oracle');
  const live = await playMoves(app, 2);
  const text = async () =>
    (await app.showPanel('State')).getByRole('group', {
      name: 'Text',
      exact: true,
    });
  await expect(await text()).toHaveText(stateText(live));

  let before = app.frames.count;
  await (
    await app.replay()
  )
    .getByRole('button', {name: 'Previous move'})
    .click();
  const earlier = await app.frames.waitFor(
    frame => frame.ply === live.num_plies - 1,
    before,
    'the previous position',
  );
  await expect(await text()).toHaveText(stateText(earlier));
  expect(stateText(earlier)).not.toBe(stateText(live));

  before = app.frames.count;
  await (await app.replay()).getByRole('button', {name: 'Go to end'}).click();
  await app.frames.waitFor(
    frame => frame.ply === live.num_plies,
    before,
    'the live position',
  );
  await expect(await text()).toHaveText(stateText(live));
});

test('History plays on from an earlier position', async ({app}) => {
  await app.openGame('tic_tac_toe');
  await app.setViewer('oracle');
  const live = await playMoves(app, 3);
  const history = await app.showSection('Actions', 'History');
  let before = app.frames.count;
  await history
    .getByRole('list', {name: 'History'})
    .getByRole('button', {name: /^#1/})
    .click();
  await app.frames.waitFor(frame => frame.ply === 1, before, 'ply 1');
  // The later moves stay listed until a move is played here.
  await expect(history.getByRole('button', {name: /^#3/})).toBeVisible();
  const legal = await app.showSection('Actions', 'Legal');
  before = app.frames.count;
  await app.legalButtons(legal).last().click();
  const branched = await app.frames.waitFor(
    frame => frame.num_plies === 2 && frame.ply === 2,
    before,
    'the new line',
  );
  expect(branched.history[0]?.action).toBe(live.history[0]?.action);
  expect(branched.history[1]?.action).not.toBe(live.history[1]?.action);
});

test('a copy icon copies', async ({app, page}) => {
  await app.openGame('tic_tac_toe');
  await app.setViewer('oracle');
  const live = await playMoves(app, 1);
  const state = await app.showPanel('State');
  await state.getByRole('button', {name: 'Copy to clipboard'}).click();
  await expect(state.getByRole('button', {name: 'Copied'})).toBeVisible();
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toBe(stateText(live));
});

test('the settings switch the theme, and it is kept', async ({app, page}) => {
  await app.openGame('tic_tac_toe');
  const html = page.locator('html');
  await app.chooseTheme('Dark');
  await expect(html).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(html).toHaveAttribute('data-theme', 'dark');
  await app.chooseTheme('Light');
  await expect(html).toHaveAttribute('data-theme', 'light');
});

test('a saved layout survives a reload', async ({app, page}) => {
  await app.openGame('tic_tac_toe');
  // State in front of Actions is the arrangement saved.
  const state = await app.showPanel('State');
  await expect(
    state.getByRole('tablist', {name: 'State sections'}),
  ).toBeVisible();
  await app.saveLayoutAs('e2e layout');

  // A reload builds the built-in arrangement, Actions in front of State.
  await page.reload();
  await expect(app.dockTab('Gameplay')).toBeVisible();
  const stateTabs = page.getByRole('tablist', {name: 'State sections'});
  await expect(stateTabs).toBeHidden();
  const saved = app.savedLayout('e2e layout');
  await expect(saved).toBeVisible();
  await saved.click();
  await expect(stateTabs).toBeVisible();
});

test('a section pops out into a panel of its own and docks back', async ({
  app,
  page,
}) => {
  await app.openGame('tic_tac_toe');
  await app.showSection('Actions', 'Text');
  await page.getByRole('button', {name: 'Pop out Text'}).click();
  await expect(app.dockTab('Text')).toBeVisible();
  await expect(page.getByText('From Actions')).toBeVisible();
  await page.getByRole('button', {name: 'Dock Text into Actions'}).click();
  await expect(app.dockTab('Text')).toHaveCount(0);
  const tabs = page.getByRole('tablist', {name: 'Actions sections'});
  await expect(tabs.getByRole('tab', {name: 'Text'})).toHaveAttribute(
    'aria-selected',
    'true',
  );
});

test('Session opens from Panel options, closed at first', async ({
  app,
  page,
}) => {
  await app.openGame('tic_tac_toe');
  await expect(app.dockTab('Session')).toHaveCount(0);
  await app.openPanelOptions();
  const drawer = page.getByRole('dialog', {name: 'Panels'});
  await drawer.getByRole('button', {name: 'Session', exact: true}).click();
  await expect(app.dockTab('Session')).toBeVisible();
  await expect(drawer.getByText('All panels are open')).toBeVisible();
});
