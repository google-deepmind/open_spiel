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

// Gin rummy against a checkpoint in one step, as REMOTE_BOTS.md's recipe
// does it: `python -m open_spiel.python.gui.play` starts the bot server with
// a checkpoint and the GUI server, opens a session whose seat 1 is a Remote
// bot, and prints the address. The bot is the stand-in checkpoint bot, so
// nothing beyond OpenSpiel is needed. The browser opens the address as seat
// 0, finds seat 1 a Remote bot with the checkpoint in its Checkpoint field,
// changes the field to a second checkpoint in analysis, and plays the hand to
// the end as seat 0; the bot's notice, which names the checkpoint that
// answered, shows the second one played. Stopping the launcher stops both
// servers: the GUI's address and the bot server's (which the launcher's
// stderr names) both refuse connections afterwards.
//
// With E2E_PLAY_BOT set (a MODULE:FACTORY importable by E2E_PYTHON) and
// E2E_PLAY_CHECKPOINT naming its checkpoint, the same hand is played against
// that bot, a trained checkpoint say; E2E_PLAY_CHECKPOINT_2, if set, is the
// checkpoint the field is changed to, and otherwise the field is left as it is.
//
// The stand-in plays on the launcher's default state, seat_view, and the
// launcher prints no full-state note. A given bot is launched with
// `--state full_state`, as the gin recipe does (the gin checkpoint bot takes
// only the full state), and the launcher prints its one-line note that the
// bot sees what its seat may not.

/// <reference types="node" />

import {type ChildProcess, spawn} from 'node:child_process';
import {mkdirSync, mkdtempSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import process from 'node:process';

import type {Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {expect, test} from './fixtures.ts';
import {MAX_MOVES} from './gin_board.ts';
import {CLIENT_ROOT, REPOSITORY_ROOT} from './server.ts';

const GAME = 'gin_rummy';
const STAND_IN = 'open_spiel.python.gui.stand_in_checkpoint_bot:make_bot';
const PLAY = /^Play (http:\/\/\S+)$/m;
// The launcher's one line when it sends the full state of a game that hides
// information.
const FULL_STATE_NOTE = /^Note: --state full_state .*gin_rummy/m;
// The bot server's ready line, as the launcher copies it to its stderr.
const BOT_SERVING = /^\[bot server\] Serving the .* bot at (http:\/\/\S+)$/m;
// Milliseconds: the launcher starts two servers.
const START_TIMEOUT_MS = 150_000;

/** Writes a stand-in checkpoint for gin rummy; returns its "<dir>@<step>". */
function writeCheckpoint(root: string, name: string, step: number): string {
  const directory = path.join(root, name);
  mkdirSync(path.join(directory, String(step)), {recursive: true});
  writeFileSync(
    path.join(directory, String(step), 'stand_in.json'),
    JSON.stringify({game: GAME}),
  );
  return `${directory}@${String(step)}`;
}

interface Launcher {
  readonly child: ChildProcess;
  readonly playUrl: string;
  readonly output: string[];
}

/** Starts the launcher; resolves with the address it prints. */
async function launch(
  bot: string,
  checkpoint: string,
  state: 'seat_view' | 'full_state' | null,
): Promise<Launcher> {
  const child = spawn(
    process.env['E2E_PYTHON'] ?? 'python3',
    [
      '-m',
      'open_spiel.python.gui.play',
      '--game',
      `${GAME}(oklahoma=true)`,
      '--bot',
      bot,
      '--checkpoint',
      checkpoint,
      '--port',
      '0',
      '--client-dir',
      path.join(CLIENT_ROOT, 'dist'),
      ...(state === null ? [] : ['--state', state]),
    ],
    {
      cwd: REPOSITORY_ROOT,
      env: {...process.env, PYTHONUNBUFFERED: '1'},
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  const output: string[] = [];
  child.stderr.on('data', (chunk: Buffer) => {
    output.push(chunk.toString());
  });
  const playUrl = await new Promise<string>((resolve, reject) => {
    let stdout = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`The launcher printed no address:\n${output.join('')}`));
    }, START_TIMEOUT_MS);
    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
      const match = PLAY.exec(stdout);
      if (match?.[1] !== undefined) {
        clearTimeout(timer);
        resolve(match[1]);
      }
    });
    child.once('exit', code => {
      clearTimeout(timer);
      reject(
        new Error(
          `The launcher exited with ${String(code)}:\n${output.join('')}`,
        ),
      );
    });
  });
  return {child, playUrl, output};
}

/** Stops the launcher with SIGTERM; resolves with its exit code. */
function stop(launcher: Launcher): Promise<number | null> {
  const {child} = launcher;
  if (child.exitCode !== null) return Promise.resolve(child.exitCode);
  return new Promise(resolve => {
    child.once('exit', code => {
      resolve(code);
    });
    child.kill('SIGTERM');
  });
}

/**
 * Stops the launcher, which must exit 0; then the GUI server's port and the
 * bot server's (named in the launcher's stderr) both refuse connections.
 */
async function stopAndExpectBothGone(
  launcher: Launcher,
  page: Page,
): Promise<void> {
  // The page stops polling before the servers go.
  await page.goto('about:blank');
  expect(await stop(launcher)).toBe(0);
  await expect(page.request.get(launcher.playUrl)).rejects.toThrow(
    /ECONNREFUSED/,
  );
  const botUrl = BOT_SERVING.exec(launcher.output.join(''))?.[1];
  expect(
    botUrl,
    "the bot server's address in the launcher's output",
  ).toBeDefined();
  await expect(page.request.get(`${botUrl ?? ''}/health`)).rejects.toThrow(
    /ECONNREFUSED/,
  );
}

/**
 * The bot seat is sent `sent`, as every viewer reads in its settings, and
 * the launcher printed its full-state note exactly when that is full_state.
 */
function expectStateSent(
  opened: Frame,
  launcher: Launcher,
  sent: 'seat_view' | 'full_state',
): void {
  expect(opened.seats[1]?.config['state']).toBe(sent);
  const output = launcher.output.join('');
  if (sent === 'full_state') {
    expect(output).toMatch(FULL_STATE_NOTE);
  } else {
    expect(output).not.toMatch(FULL_STATE_NOTE);
  }
}

test('play launcher: gin rummy against a checkpoint in one step, to the end', async ({
  app,
  page,
}) => {
  test.setTimeout(400_000);
  const given = process.env['E2E_PLAY_BOT'] ?? '';
  const bot = given === '' ? STAND_IN : given;
  let first: string;
  let second: string | null;
  if (given === '') {
    const root = mkdtempSync(path.join(tmpdir(), 'gin-stand-in-'));
    first = writeCheckpoint(root, 'first', 5);
    second = writeCheckpoint(root, 'second', 9);
  } else {
    first = process.env['E2E_PLAY_CHECKPOINT'] ?? '';
    expect(first, "E2E_PLAY_CHECKPOINT names the bot's checkpoint").not.toBe(
      '',
    );
    second = process.env['E2E_PLAY_CHECKPOINT_2'] ?? null;
  }
  // The stand-in on the default (seat_view); a given bot, the gin
  // checkpoint bot say, on the full state it needs.
  const state = given === '' ? null : 'full_state';
  const launcher = await launch(bot, first, state);
  try {
    const url = new URL(launcher.playUrl);
    expect(url.pathname).toBe(`/game/${GAME}`);
    expect(url.searchParams.get('viewer')).toBe('0');
    const session = url.searchParams.get('session') ?? '';
    const api = new URL(`/api/sessions/${session}`, url);

    // The address the launcher prints opens the session as seat 0.
    const before = app.frames.count;
    await page.goto(launcher.playUrl);
    await expect(app.dockTab('Gameplay')).toBeVisible();
    const opened: Frame = await app.frames.waitFor(
      frame => frame.session.id === session,
      before,
      'the first frame of the launched session',
    );
    expect(opened.session.game).toBe(GAME);
    expect(opened.seats[1]?.controller).toBe('remote');
    expectStateSent(opened, launcher, state ?? 'seat_view');
    expect(opened.seats[0]?.controller).toBe('human');

    // Seat 1's settings carry the checkpoint; in analysis, change it.
    await app.setViewer('oracle');
    const row = await app.seat(1);
    await row.getByRole('button', {name: 'Settings', exact: true}).click();
    const field = row.getByRole('textbox', {name: 'Checkpoint'});
    await expect(field).toHaveValue(first);
    if (second !== null) {
      const next = second;
      await field.fill(next);
      const saved = app.frames.count;
      await row
        .getByRole('button', {name: 'Save settings for Player 1'})
        .click();
      await app.frames.waitFor(
        frame =>
          frame.seats[1]?.controller === 'remote' &&
          frame.seats[1].config['checkpoint'] === next &&
          frame.seats[1].status === 'ready',
        saved,
        'a frame with Player 1 on the second checkpoint',
      );
    }

    // Seat 0 plays the hand to the end; the bot answers every turn. Seat 0
    // plays its first legal action and the deal is unseeded, so the hand
    // may end after the bot's first turn (a draw and a discard): the bot
    // has played, and no seat failed (below).
    await app.setViewer(0);
    const end = await app.playToEnd(GAME, MAX_MOVES);
    expect(end.status).toBe('terminal');
    expect(
      end.history.filter(entry => entry.seat === 1).length,
    ).toBeGreaterThan(0);
    const oracle = (await (
      await page.request.get(`${api.toString()}?viewer=oracle`)
    ).json()) as Frame;
    expect(oracle.status).toBe('terminal');
    expect(oracle.seats[1]?.failure ?? null).toBeNull();
    if (given === '') {
      expect(oracle.seats[1]?.notice).toBe('stand-in second@9');
    }
  } finally {
    await stopAndExpectBothGone(launcher, page);
  }
});
