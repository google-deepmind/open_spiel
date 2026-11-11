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

// A Remote bot seat, set up in Setup › Players as REMOTE_BOTS.md's recipe does:
// the reference bot server runs on this machine at a port of its own, its
// address goes in the seat's URL, and the seat plays it. The bot's notice
// shows in its seat's row to the oracle and to that seat, and to no one
// else.

/// <reference types="node" />

import {type ChildProcess, spawn} from 'node:child_process';
import process from 'node:process';

import type {Frame} from '../src/contract/contract.ts';
import {expect, test} from './fixtures.ts';
import {REPOSITORY_ROOT} from './server.ts';

const NOTICE = 'Recovered from a legal-set mismatch.';
const READY = /at (http:\/\/\S+)/;

interface BotServer {
  readonly url: string;
  readonly child: ChildProcess;
  /** The bot server's request log so far, one line per request. */
  readonly log: string[];
}

/** How many of the bot server's logged requests are `GET path`. */
function gets(bot: BotServer, path: string): number {
  return bot.log.filter(line => line.includes(`"GET ${path} `)).length;
}

/** Starts the reference bot server on a free port. */
async function startBot(): Promise<BotServer> {
  const child = spawn(
    process.env['E2E_PYTHON'] ?? 'python3',
    [
      '-m',
      'open_spiel.python.gui.remote_bot_server',
      '--port',
      '0',
      '--bot',
      'random',
      '--notice',
      NOTICE,
    ],
    {cwd: REPOSITORY_ROOT, stdio: ['ignore', 'pipe', 'pipe']},
  );
  const log: string[] = [];
  let pending = '';
  child.stderr.on('data', (chunk: Buffer) => {
    const lines = (pending + chunk.toString()).split('\n');
    pending = lines.pop() ?? '';
    log.push(...lines);
  });
  const url = await new Promise<string>((resolve, reject) => {
    let output = '';
    child.stdout.on('data', (chunk: Buffer) => {
      output += chunk.toString();
      const match = READY.exec(output);
      if (match?.[1] !== undefined) resolve(match[1]);
    });
    child.on('exit', code => {
      reject(new Error(`The bot server exited with ${String(code)}.`));
    });
  });
  return {url, child, log};
}

test('Remote bot: a seat plays a bot server; its notice reaches the seat and analysis only', async ({
  app,
}) => {
  const bot = await startBot();
  try {
    await app.openGame('kuhn_poker');
    const row = await app.seat(1);
    await row.getByRole('button', {name: 'Settings', exact: true}).click();
    await row
      .getByRole('combobox', {name: 'Controller'})
      .selectOption({label: 'Remote bot'});
    await row.getByRole('textbox', {name: 'URL'}).fill(bot.url);
    await row.getByRole('checkbox', {name: 'Autoplay'}).setChecked(true);
    const puts: string[] = [];
    app.page.on('request', request => {
      if (request.method() === 'PUT') puts.push(request.url());
    });
    const before = app.frames.count;
    await row.getByRole('button', {name: 'Save settings for Player 1'}).click();
    await app.frames.waitFor(
      frame =>
        frame.seats[1]?.controller === 'remote' &&
        frame.seats[1].status === 'ready',
      before,
      'a frame with Player 1 played by the remote bot',
    );
    // One save is one seat change, and the bot server is checked once.
    await app.page.waitForTimeout(500);
    expect(puts).toHaveLength(1);
    expect([gets(bot, '/health'), gets(bot, '/info')]).toEqual([1, 1]);

    const legal = await app.legal();
    const played = app.frames.count;
    await app.legalButtons(legal).first().click();
    const answered: Frame = await app.frames.waitFor(
      frame => frame.history.some(entry => entry.seat === 1),
      played,
      'the frame after the remote bot moves',
    );
    expect(answered.seats[1]?.notice).toBe(NOTICE);
    await expect((await app.seat(1)).locator('.seat__notice')).toHaveText(
      NOTICE,
    );

    for (const [viewer, seen] of [
      [1, NOTICE],
      [0, null],
      ['spectator', null],
    ] as const) {
      const frame = await app.setViewer(viewer);
      expect(frame.seats[1]?.notice ?? null).toBe(seen);
      const notice = (await app.seat(1)).locator('.seat__notice');
      if (seen === null) {
        await expect(notice).toHaveCount(0);
      } else {
        await expect(notice).toHaveText(seen);
      }
    }
  } finally {
    bot.child.kill();
  }
});
