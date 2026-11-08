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

// Adds dist/THIRD_PARTY_NOTICES.txt to a build: the licence text of every
// third-party package listed in licenses/DEPENDENCIES.md, which those
// licences ask to accompany every copy of the client.

/// <reference types="node" />

import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';

import type {Plugin} from 'vite';

/** The notices file the build adds to its output directory. */
export const NOTICES_NAME = 'THIRD_PARTY_NOTICES.txt';

/** The list of third-party packages, relative to the client's root. */
export const DEPENDENCIES_PATH = 'licenses/DEPENDENCIES.md';

/** One package of the list. */
export interface Notice {
  readonly name: string;
  readonly version: string;
  readonly licence: string;
  /** The path of its licence text, relative to the client's root. */
  readonly text: string;
}

/** The rows of the one table in `markdown`, after its header and rule. */
export function parseNotices(markdown: string): Notice[] {
  const rows = markdown
    .split('\n')
    .filter(line => line.startsWith('|'))
    .slice(2);
  return rows.map(row => {
    const [name = '', version = '', licence = '', text = ''] = row
      .trim()
      .replace(/^\||\|$/g, '')
      .split('|')
      .map(cell => cell.trim());
    return {name, version, licence, text};
  });
}

/** The notices file: each package with its licence text. */
export function noticesText(licensed: ReadonlyArray<[Notice, string]>): string {
  const sections = licensed.map(([notice, text]) => {
    const title = `${notice.name} ${notice.version} (${notice.licence})`;
    return `${title}\n${'-'.repeat(title.length)}\n\n${text.trimEnd()}\n`;
  });
  return [
    'The OpenSpiel GUI client contains the following third-party software.\n',
    ...sections,
  ].join('\n');
}

/** The Vite plugin that writes the notices file in `vite build`. */
export function thirdPartyNotices(): Plugin {
  let root = '';
  return {
    name: 'third-party-notices',
    apply: 'build',
    configResolved(config) {
      root = config.root;
    },
    async generateBundle() {
      const read = (path: string): Promise<string> =>
        readFile(resolve(root, path), 'utf8');
      const notices = parseNotices(await read(DEPENDENCIES_PATH));
      const licensed = await Promise.all(
        notices.map(async (notice): Promise<[Notice, string]> => [
          notice,
          await read(notice.text),
        ]),
      );
      this.emitFile({
        type: 'asset',
        fileName: NOTICES_NAME,
        source: noticesText(licensed),
      });
    },
  };
}
