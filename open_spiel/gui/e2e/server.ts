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

// Starts the GUI server, `python -m open_spiel.python.gui`, on a free port
// and stops it again.

/// <reference types="node" />

import {type ChildProcess, spawn} from 'node:child_process';
import {createWriteStream} from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import {fileURLToPath} from 'node:url';

/** The client's directory, open_spiel/gui. */
export const CLIENT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
);

/** The repository root, from which `open_spiel` is importable. */
export const REPOSITORY_ROOT = path.resolve(CLIENT_ROOT, '..', '..');

// What the server prints once it is listening.
const READY = /Serving OpenSpiel games at (http:\/\/\S+)/;

// Milliseconds. The server imports OpenSpiel and registers every game first.
const START_TIMEOUT_MS = 120_000;
const STOP_TIMEOUT_MS = 10_000;

/** A running server. */
export interface RunningServer {
  /** Where it serves, ending in a slash. */
  readonly url: string;
  /** Stops it and waits for it to exit. */
  stop(): Promise<void>;
}

/** How to start the server. */
export interface ServerOptions {
  /** The Python interpreter, which must import `pyspiel`. */
  readonly python: string;
  /** The built client the server serves. */
  readonly clientDir: string;
  /** Where the server's output is written. */
  readonly logFile: string;
}

/**
 * Starts the server on 127.0.0.1 with a port the system picks, and resolves
 * once it says where it listens.
 *
 * @throws Error when the server exits, or says nothing, before it listens.
 */
export async function startServer(
  options: ServerOptions,
): Promise<RunningServer> {
  const child = spawn(
    options.python,
    [
      '-m',
      'open_spiel.python.gui',
      '--host',
      '127.0.0.1',
      '--port',
      '0',
      '--client-dir',
      options.clientDir,
    ],
    {
      cwd: REPOSITORY_ROOT,
      env: {...process.env, PYTHONUNBUFFERED: '1'},
      stdio: ['ignore', 'pipe', 'pipe'],
      // Its own process group, so that stopping it stops anything it started.
      detached: true,
    },
  );
  // A test run that dies must not leave the server behind.
  const killOnExit = (): void => {
    signalGroup(child, 'SIGKILL');
  };
  process.once('exit', killOnExit);
  const log = createWriteStream(options.logFile);
  child.stdout.pipe(log, {end: false});
  child.stderr.pipe(log, {end: false});
  const url = await readyUrl(child, options.logFile);
  return {
    url,
    stop: async () => {
      process.removeListener('exit', killOnExit);
      await stopChild(child);
      log.end();
    },
  };
}

function readyUrl(child: ChildProcess, logFile: string): Promise<string> {
  return new Promise((resolve, reject) => {
    let output = '';
    const timer = setTimeout(() => {
      signalGroup(child, 'SIGKILL');
      reject(
        new Error(
          `The server did not start within ${String(START_TIMEOUT_MS)} ms; ` +
            `its output is in ${logFile}.`,
        ),
      );
    }, START_TIMEOUT_MS);
    child.stdout?.on('data', (chunk: Buffer) => {
      output += chunk.toString();
      const match = READY.exec(output);
      if (match?.[1] !== undefined) {
        clearTimeout(timer);
        resolve(match[1].endsWith('/') ? match[1] : `${match[1]}/`);
      }
    });
    child.once('error', error => {
      clearTimeout(timer);
      reject(error);
    });
    child.once('exit', code => {
      clearTimeout(timer);
      reject(
        new Error(
          `The server exited with code ${String(code)} before it listened; ` +
            `its output is in ${logFile}.`,
        ),
      );
    });
  });
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise<void>(resolve => {
    child.once('exit', () => {
      resolve();
    });
  });
  signalGroup(child, 'SIGTERM');
  const timer = setTimeout(() => {
    signalGroup(child, 'SIGKILL');
  }, STOP_TIMEOUT_MS);
  await exited;
  clearTimeout(timer);
  // Whatever outlived it no longer writes to the log.
  child.stdout?.destroy();
  child.stderr?.destroy();
}

/** Sends `signal` to the process group `child` leads, if it still exists. */
function signalGroup(child: ChildProcess, signal: NodeJS.Signals): void {
  if (child.pid === undefined) return;
  try {
    process.kill(-child.pid, signal);
  } catch (error) {
    // The group has already exited.
    if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error;
  }
}

/**
 * Checks that `url` is a GUI server serving a built client: its health route
 * answers, and its page is the client's.
 *
 * @throws Error naming what is missing.
 */
export async function checkServer(url: string): Promise<void> {
  const health = await fetch(new URL('api/health', url));
  if (!health.ok) {
    throw new Error(
      `${url} answers its health route with ${String(health.status)}.`,
    );
  }
  const page = await fetch(url);
  const html = await page.text();
  if (!page.ok || !html.includes('id="root"')) {
    throw new Error(
      `${url} serves no built client; run \`npm run build\` first.`,
    );
  }
}
