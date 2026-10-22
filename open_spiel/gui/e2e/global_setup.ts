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

// Gives the tests a server: the one E2E_BASE_URL names, or a fresh one
// started on a free port and stopped after the run. The tests read its
// address from E2E_BASE_URL.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import {CLIENT_ROOT, checkServer, startServer} from './server.ts';

/** The environment variable holding the server's address. */
export const BASE_URL_VARIABLE = 'E2E_BASE_URL';

/** Starts the server unless one is given; returns the function that stops it. */
export default async function globalSetup(): Promise<() => Promise<void>> {
  const given = process.env[BASE_URL_VARIABLE];
  if (given !== undefined && given !== '') {
    await checkServer(given);
    return async () => {
      // The server was given, so it is not ours to stop.
    };
  }
  const server = await startServer({
    python: process.env['E2E_PYTHON'] ?? 'python3',
    clientDir: path.join(CLIENT_ROOT, 'dist'),
    logFile:
      process.env['E2E_SERVER_LOG'] ?? path.join(CLIENT_ROOT, 'e2e-server.log'),
  });
  try {
    await checkServer(server.url);
  } catch (error) {
    await server.stop();
    throw error;
  }
  process.env[BASE_URL_VARIABLE] = server.url;
  return () => server.stop();
}
