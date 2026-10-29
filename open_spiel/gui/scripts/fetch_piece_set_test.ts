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

/// <reference types="node" />

import {Buffer} from 'node:buffer';
import {existsSync} from 'node:fs';
import {mkdir, mkdtemp, readFile, readdir, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';

import {describe, expect, it, vi} from 'vitest';

import {
  type PieceSetSource,
  SOURCES,
  fetchPieceSet,
  fetchedSetNotices,
  setDirectory,
  sha256,
} from './fetch_piece_set.ts';

function fakeSource(
  contents: Readonly<Record<string, string>>,
): PieceSetSource {
  return {
    id: 'fake',
    description: 'a fake set',
    files: Object.entries(contents).map(([saveAs, text]) => ({
      saveAs,
      url: `https://example.invalid/${saveAs}`,
      sha256: sha256(Buffer.from(text)),
    })),
  };
}

function fakeFetcher(served: Readonly<Record<string, string>>) {
  return vi.fn((url: string) => {
    const name = url.slice(url.lastIndexOf('/') + 1);
    const text = served[name];
    if (text === undefined) throw new Error(`No ${url}.`);
    return Promise.resolve(new Uint8Array(Buffer.from(text)));
  });
}

async function scratch(): Promise<string> {
  return join(await mkdtemp(join(tmpdir(), 'fetch-pieces-')), 'set');
}

const FILES = {'white-king.svg': 'a piece', 'LICENSE.txt': 'licence'};

describe('fetchPieceSet', () => {
  it('writes every file once all match their pins', async () => {
    const directory = await scratch();
    const outcome = await fetchPieceSet(
      fakeSource(FILES),
      directory,
      fakeFetcher(FILES),
    );
    expect(outcome).toBe('fetched');
    expect((await readdir(directory)).sort()).toEqual([
      'LICENSE.txt',
      'white-king.svg',
    ]);
    expect(await readFile(join(directory, 'LICENSE.txt'), 'utf8')).toBe(
      'licence',
    );
  });

  it('refuses a file whose digest differs, writing nothing', async () => {
    const directory = await scratch();
    const served = {...FILES, 'LICENSE.txt': 'tampered'};
    await expect(
      fetchPieceSet(fakeSource(FILES), directory, fakeFetcher(served)),
    ).rejects.toThrow(
      /Refusing fake: LICENSE\.txt from https:\/\/example\.invalid\/LICENSE\.txt has sha256 [0-9a-f]{64}, not the pinned [0-9a-f]{64}/,
    );
    expect(existsSync(directory)).toBe(false);
  });

  it('downloads nothing when the set is already there', async () => {
    const directory = await scratch();
    await fetchPieceSet(fakeSource(FILES), directory, fakeFetcher(FILES));
    const again = fakeFetcher(FILES);
    expect(await fetchPieceSet(fakeSource(FILES), directory, again)).toBe(
      'present',
    );
    expect(again).not.toHaveBeenCalled();
  });

  it('refuses a directory holding other files', async () => {
    const directory = await scratch();
    await fetchPieceSet(fakeSource(FILES), directory, fakeFetcher(FILES));
    await writeFile(join(directory, 'white-king.svg'), 'edited');
    await expect(
      fetchPieceSet(fakeSource(FILES), directory, fakeFetcher(FILES)),
    ).rejects.toThrow(/not fake's pinned ones/);
  });
});

describe('fetchPieceSet, refusing', () => {
  it('refuses a directory of other files before downloading', async () => {
    const directory = await scratch();
    await mkdir(directory);
    await writeFile(join(directory, 'notes.txt'), 'mine');
    const fetcher = fakeFetcher(FILES);
    await expect(
      fetchPieceSet(fakeSource(FILES), directory, fetcher),
    ).rejects.toThrow(/not fake's pinned ones/);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('names the file that failed to download, writing nothing', async () => {
    const directory = await scratch();
    const fetcher = fakeFetcher({'white-king.svg': 'a piece'});
    await expect(
      fetchPieceSet(fakeSource(FILES), directory, fetcher),
    ).rejects.toThrow(
      "Could not fetch fake's LICENSE.txt from https://example.invalid/" +
        'LICENSE.txt. Nothing was written.',
    );
    expect(existsSync(directory)).toBe(false);
  });

  it('leaves no partial directory when the rename fails', async () => {
    const directory = await scratch();
    const served = fakeFetcher(FILES);
    // Something else fills the directory while the files download, so the
    // rename into place fails.
    const fetcher = vi.fn(async (url: string) => {
      await mkdir(directory, {recursive: true});
      await writeFile(join(directory, 'late.txt'), 'late');
      return served(url);
    });
    await expect(
      fetchPieceSet(fakeSource(FILES), directory, fetcher),
    ).rejects.toThrow();
    const siblings = await readdir(dirname(directory));
    expect(siblings).toEqual(['set']);
  });
});

describe('SOURCES', () => {
  const cburnett = SOURCES['cburnett'];

  it('pins twelve pieces, their source pages and a licence for cburnett', () => {
    expect(cburnett).toBeDefined();
    const names = (cburnett?.files ?? []).map(file => file.saveAs);
    for (const color of ['white', 'black']) {
      for (const role of [
        'king',
        'queen',
        'rook',
        'bishop',
        'knight',
        'pawn',
      ]) {
        expect(names).toContain(`${color}-${role}.svg`);
        expect(names).toContain(`${color}-${role}.svg.source.wikitext`);
      }
    }
    expect(names).toContain('LICENSE-BSD.wikitext');
    expect(new Set(names).size).toBe(names.length);
  });

  it('pins Wikimedia Commons originals and fixed page revisions', () => {
    for (const file of cburnett?.files ?? []) {
      expect(file.sha256).toMatch(/^[0-9a-f]{64}$/);
      if (file.saveAs.endsWith('.svg')) {
        expect(file.url).toMatch(
          /^https:\/\/upload\.wikimedia\.org\/wikipedia\/commons\/[0-9a-f]\/[0-9a-f]{2}\/Chess_[kqrbnp][ld]t45\.svg$/,
        );
      } else {
        expect(file.url).toMatch(
          /^https:\/\/commons\.wikimedia\.org\/w\/index\.php\?action=raw&oldid=\d+$/,
        );
      }
    }
  });

  it('fetches into piece_sets/<id>/ beside the client', () => {
    expect(setDirectory('cburnett')).toMatch(/\/piece_sets\/cburnett$/);
  });
});

describe('fetchedSetNotices', () => {
  async function emitted(directory: string): Promise<Record<string, string>> {
    const plugin = fetchedSetNotices(
      {fake: fakeSource(FILES)},
      () => directory,
    );
    const files: Record<string, string> = {};
    const emitFile = (file: {fileName: string; source: Uint8Array}) => {
      files[file.fileName] = Buffer.from(file.source).toString();
    };
    const hook = plugin.generateBundle;
    if (typeof hook !== 'function') throw new Error('No generateBundle.');
    await Reflect.apply(hook, {emitFile}, []);
    return files;
  }

  it("copies a fetched set's licence into the build, but not its pieces", async () => {
    const directory = await scratch();
    await fetchPieceSet(fakeSource(FILES), directory, fakeFetcher(FILES));
    expect(await emitted(directory)).toEqual({
      'piece_sets/fake/LICENSE.txt': 'licence',
    });
  });

  it('copies nothing when the set was not fetched', async () => {
    expect(await emitted(await scratch())).toEqual({});
  });
});
