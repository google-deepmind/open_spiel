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

// Whole games played through the page: tic-tac-toe against each of the
// server's controllers to the end, and moves of 9x9 go from the legal list.

import type {Frame} from '../src/contract/contract.ts';
import {type App, type ControllerChoice, MCTS, RANDOM} from './app.ts';
import {expect, test} from './fixtures.ts';

// Seat 0 moves at most five times in tic-tac-toe.
const TIC_TAC_TOE_MOVES = 5;

/** Checks that `end` is over, and that the seats took turns from seat 0. */
function expectFinished(end: Frame, opponent: ControllerChoice): void {
  expect(end.status).toBe('terminal');
  expect(end.returns).toHaveLength(2);
  expect(end.seats[1]?.controller).toBe(opponent.name);
  expect(end.history.map(entry => entry.seat)).toEqual(
    end.history.map((_, index) => index % 2),
  );
}

async function playAgainst(
  app: App,
  opponent: ControllerChoice,
): Promise<void> {
  await app.setController(1, opponent, true);
  const end = await app.playToEnd('tic_tac_toe', TIC_TAC_TOE_MOVES);
  expectFinished(end, opponent);
  const legal = await app.legal();
  await expect(
    legal.getByText('Game finished. No legal actions.'),
  ).toBeVisible();
  // The tic-tac-toe board says the result itself, over the board, to seat 0.
  const [first = 0, second = 0] = end.returns ?? [];
  await expect((await app.gameplay()).getByTestId('game-result')).toHaveText(
    first > second
      ? 'X wins · You win'
      : second > first
        ? 'O wins · You lose'
        : 'Draw',
  );
}

test('tic_tac_toe: seat 0 plays Random, then MCTS, to the end', async ({
  app,
}) => {
  await app.openGame('tic_tac_toe');
  await app.setViewer(0);
  await playAgainst(app, RANDOM);

  // Resetting keeps the seats; the opponent then changes to MCTS.
  const before = app.frames.count;
  await app.resetGame();
  await app.frames.waitFor(
    frame => frame.num_plies === 0 && frame.status === 'playing',
    before,
    'the restarted game',
  );
  await playAgainst(app, MCTS);
});

/** Plays the legal action `label` from Actions › Legal. */
async function playLabel(app: App, label: string): Promise<Frame> {
  const legal = await app.showSection('Actions', 'Legal');
  const before = app.frames.count;
  await app
    .legalButtons(legal)
    .filter({hasText: new RegExp(`${label}$`)})
    .click();
  return app.frames.waitFor(
    frame => frame.history.at(-1)?.label === label,
    before,
    `the frame after ${label}`,
  );
}

test('go: 9x9 moves from the legal list', async ({app}) => {
  await app.openGame('go');
  // Go opens on the 9x9 board: 81 points and a pass.
  const start = await app.setViewer('oracle');
  expect(start.session.parameters['board_size']).toBe(9);
  expect(start.legal_actions).toHaveLength(82);

  await playLabel(app, 'B d4');
  await playLabel(app, 'W e5');
  const end = await playLabel(app, 'B c3');
  expect(end.history.map(entry => entry.label)).toEqual([
    'B d4',
    'W e5',
    'B c3',
  ]);
  if (end.state_text === null) throw new Error('The oracle sees the state.');
  const board = await app.gameplay();
  await expect(board.getByRole('region', {name: 'State'})).toHaveText(
    end.state_text,
  );
});
