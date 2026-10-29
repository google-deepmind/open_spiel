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

// @vitest-environment node

/// <reference types="node" />

import {mkdir, mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';

import {build} from 'vite';
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import {
  DEPENDENCIES_PATH,
  NOTICES_NAME,
  noticesText,
  parseNotices,
  thirdPartyNotices,
} from './third_party_notices.ts';

const TABLE = `# Third-party packages

| Package | Version | Licence | Licence text |
| ------- | ------- | ------- | ------------ |
| tiny    | 1.0.0   | MIT     | licenses/tiny/LICENSE |
`;

describe('parseNotices', () => {
  it('reads the rows after the header and its rule', () => {
    expect(parseNotices(TABLE)).toEqual([
      {
        name: 'tiny',
        version: '1.0.0',
        licence: 'MIT',
        text: 'licenses/tiny/LICENSE',
      },
    ]);
  });
});

describe('noticesText', () => {
  it('titles each licence text with its package', () => {
    const [notice] = parseNotices(TABLE);
    if (notice === undefined) throw new Error('no row');
    expect(noticesText([[notice, 'MIT License\n\n']])).toBe(
      'The OpenSpiel GUI client contains the following third-party software.\n' +
        '\ntiny 1.0.0 (MIT)\n----------------\n\nMIT License\n',
    );
  });
});

describe('thirdPartyNotices in vite build', () => {
  let root: string;

  async function put(path: string, content: string): Promise<void> {
    await mkdir(dirname(join(root, path)), {recursive: true});
    await writeFile(join(root, path), content);
  }

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), 'third-party-notices-'));
    await put('index.html', '<script type="module" src="./main.ts"></script>');
    await put('main.ts', "document.title = 'client';\n");
    await put(DEPENDENCIES_PATH, TABLE);
    await put('licenses/tiny/LICENSE', 'MIT License\n');
  });

  afterEach(async () => {
    await rm(root, {recursive: true, force: true});
  });

  it('writes every listed licence text into the build', async () => {
    await build({
      root,
      configFile: false,
      logLevel: 'silent',
      plugins: [thirdPartyNotices()],
    });
    const notices = await readFile(join(root, 'dist', NOTICES_NAME), 'utf8');
    expect(notices).toContain(
      'tiny 1.0.0 (MIT)\n----------------\n\nMIT License',
    );
  });
});
