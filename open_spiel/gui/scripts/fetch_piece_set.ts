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

// Fetches a chess piece set the client does not ship, for the user's own
// copy:
//
//   npm run fetch-pieces -- cburnett
//
// Each file comes from a pinned URL and must match its pinned sha256; on any
// mismatch nothing is written. The set's twelve SVG files, named
// `<colour>-<role>.svg`, and the licence text and source pages downloaded
// with them, go to piece_sets/<id>/, which git ignores. The client offers
// the set once that directory is there (src/games/chess/pieces/
// fetched_sets.tsx).

/// <reference types="node" />

import {createHash} from 'node:crypto';
import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import process from 'node:process';
import {setTimeout as sleep} from 'node:timers/promises';

import type {Plugin} from 'vite';

import {isEntryPoint} from './entry_point.ts';

/** One file of a set: where it is saved, where it comes from, its digest. */
export interface PinnedFile {
  readonly saveAs: string;
  readonly url: string;
  readonly sha256: string;
}

/** A piece set to fetch: what it is and its pinned files. */
export interface PieceSetSource {
  readonly id: string;
  readonly description: string;
  readonly files: readonly PinnedFile[];
}

/** Downloads `url`'s bytes. */
export type Fetcher = (url: string) => Promise<Uint8Array>;

/** Whether `error` is a missing file or directory. */
function isMissing(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT';
}

function pinned(saveAs: string, url: string, sha256: string): PinnedFile {
  return {saveAs, url, sha256};
}

const COMMONS = 'https://upload.wikimedia.org/wikipedia/commons';
const REVISION = 'https://commons.wikimedia.org/w/index.php?action=raw&oldid=';

/**
 * Every set this script fetches, by id. cburnett: Colin M.L. Burnett's
 * pieces, from their Wikimedia Commons originals. Each SVG is the file's
 * current upload; each `.source.wikitext` is a fixed revision of the file's
 * Commons page; LICENSE-BSD.wikitext is a fixed revision of the Commons
 * licence page those pages name.
 */
export const SOURCES: Readonly<Record<string, PieceSetSource>> = {
  cburnett: {
    id: 'cburnett',
    description: "Colin M.L. Burnett's chess pieces, from Wikimedia Commons",
    files: [
      pinned(
        'white-king.svg',
        `${COMMONS}/4/42/Chess_klt45.svg`,
        '56f55c784843b1ac272b8745d740aa2a3e6c585513ef889978916f88e5d0b70b',
      ),
      pinned(
        'white-queen.svg',
        `${COMMONS}/1/15/Chess_qlt45.svg`,
        'b72b864e2a5b6c8f8afb7f260130c10e649ff063f4ef58190c00a35c56364327',
      ),
      pinned(
        'white-rook.svg',
        `${COMMONS}/7/72/Chess_rlt45.svg`,
        '4d42ab45afd862c704eb9b35317102d453a7a6b9b71d40f18958c8eadc829e4b',
      ),
      pinned(
        'white-bishop.svg',
        `${COMMONS}/b/b1/Chess_blt45.svg`,
        '1d7beace24d455c923ee80d27125963eaf0287b956c5576dbf790c97ac0b97eb',
      ),
      pinned(
        'white-knight.svg',
        `${COMMONS}/7/70/Chess_nlt45.svg`,
        '5486791207156f7ae8b8678187648df45085d726334c2862e73b077dea00641e',
      ),
      pinned(
        'white-pawn.svg',
        `${COMMONS}/4/45/Chess_plt45.svg`,
        'cc7de30708dcec8f4d593a89d10893d5f9c063682039a1c441e86c44cf2096db',
      ),
      pinned(
        'black-king.svg',
        `${COMMONS}/f/f0/Chess_kdt45.svg`,
        '025eea92e0ef8eb1fd06b1c58d0d112948f08bf66cea6b5d003659569949b41c',
      ),
      pinned(
        'black-queen.svg',
        `${COMMONS}/4/47/Chess_qdt45.svg`,
        '70191a3fbc729ef629661e2419a66ab8024c49277aab8ccae3a5ef61372ab802',
      ),
      pinned(
        'black-rook.svg',
        `${COMMONS}/f/ff/Chess_rdt45.svg`,
        '6abf617a9e26902e0734d85897c9ca55e29d7be2928142aa21032c38967e34ba',
      ),
      pinned(
        'black-bishop.svg',
        `${COMMONS}/9/98/Chess_bdt45.svg`,
        'ba67da76ce919addc60ecb8b46801def073dd54149b2c038a2d07a16d904d5e4',
      ),
      pinned(
        'black-knight.svg',
        `${COMMONS}/e/ef/Chess_ndt45.svg`,
        '735cc58315b123a56632d4877a6b976c827481fa97bf9a5c8f459ec969bc2549',
      ),
      pinned(
        'black-pawn.svg',
        `${COMMONS}/c/c7/Chess_pdt45.svg`,
        '4413bf7c18a341f9723d97e6f92c985e30b6167b037e80842cea59b7541bb074',
      ),
      pinned(
        'white-king.svg.source.wikitext',
        `${REVISION}1122093902`,
        '8a47817fd6a0c0569f368a222a46fad45b460d1811054b86cf86b5c776964c68',
      ),
      pinned(
        'white-queen.svg.source.wikitext',
        `${REVISION}1122094769`,
        'f41f1eee0622799c49b66a0c59f975459b2cee204ea9d207396fa1de8a9c1ac8',
      ),
      pinned(
        'white-rook.svg.source.wikitext',
        `${REVISION}1122094886`,
        '6a60db8b112f394487b4e34cec14736d87b01c011a2e0d4df1c45e09d706819a',
      ),
      pinned(
        'white-bishop.svg.source.wikitext',
        `${REVISION}1122092843`,
        'fecc88dee7cf912e775bc067b8c4998aab032592e550729ae8cb9712a4e390a5',
      ),
      pinned(
        'white-knight.svg.source.wikitext',
        `${REVISION}1122094287`,
        'e1f7455dc4ab9681ec17e139bd1f6bca801322cf72f1ada878ea653c34d9a4a6',
      ),
      pinned(
        'white-pawn.svg.source.wikitext',
        `${REVISION}1218934628`,
        'fddbf619453078c26e56d6d12e858f7430a4ffa1f649c2331f030ebf621f9487',
      ),
      pinned(
        'black-king.svg.source.wikitext',
        `${REVISION}1122093852`,
        '4b1042e81e2d33108209dd2435b0dce40a0067b6883718a83665eb62659921b9',
      ),
      pinned(
        'black-queen.svg.source.wikitext',
        `${REVISION}997858038`,
        'e2aa7ab76080d00f609585d699f153405f10cf7a3d4f96904034f47a4a4d9ac6',
      ),
      pinned(
        'black-rook.svg.source.wikitext',
        `${REVISION}997857576`,
        'dcc21f559648ba650abff8c18419b3562fb10ad8730964f82cb9d2e096d09af4',
      ),
      pinned(
        'black-bishop.svg.source.wikitext',
        `${REVISION}1122092772`,
        'fdf10cec64d2a32fd021dd1a86c33776300d836b3bd5924a8314baf3d3f97307',
      ),
      pinned(
        'black-knight.svg.source.wikitext',
        `${REVISION}1122094232`,
        '53d17790269690121062e4b9238b4eb35a775db4ab28e631f86e1cd7182ba97d',
      ),
      pinned(
        'black-pawn.svg.source.wikitext',
        `${REVISION}1030603507`,
        'b3bab3e4d7df20ff7d68200fa764fccefd4b4a4e7b34fc427e8db464b5cdc295',
      ),
      pinned(
        'LICENSE-BSD.wikitext',
        `${REVISION}198717010`,
        '714cfcbe70ed81f3ac036f131665ef7b49c4a16821610d5c787dc0364b48464b',
      ),
    ],
  },
};

/** The directory a set `id` is fetched into: piece_sets/<id>/ in the client. */
export function setDirectory(id: string): string {
  return resolve(import.meta.dirname, '..', 'piece_sets', id);
}

/** The sha256 of `bytes`, in hex. */
export function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

// Wikimedia asks for a descriptive User-Agent and answers too many requests
// in a row with 429; the files are fetched one at a time, a moment apart.
const USER_AGENT = 'open_spiel-gui-fetch-pieces (OpenSpiel GUI piece fetcher)';
const PAUSE_MS = 500;
const RETRIES = 5;
/** How long one request may take before it is abandoned. */
export const TIMEOUT_MS = 30_000;

/**
 * Downloads `url` over HTTPS, waiting and retrying while it is rate limited;
 * a request that takes longer than TIMEOUT_MS fails.
 */
export async function httpFetch(url: string): Promise<Uint8Array> {
  for (let attempt = 0; ; attempt++) {
    await sleep(PAUSE_MS);
    const response = await fetch(url, {
      headers: {'User-Agent': USER_AGENT},
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (response.ok) return new Uint8Array(await response.arrayBuffer());
    if (response.status !== 429 || attempt === RETRIES) {
      throw new Error(`GET ${url}: HTTP ${String(response.status)}`);
    }
    await sleep(5000 * (attempt + 1));
  }
}

/**
 * Downloads every file of `source` with `fetcher`, checks each against its
 * pinned sha256 and, when all match, writes them into `directory`. When
 * `directory` already holds the set, unchanged, nothing is downloaded.
 *
 * @returns 'fetched', or 'present' when the set was already there.
 * @throws Error naming the file, the URL and both digests on a mismatch,
 *     or the file and the URL when its download fails, having written
 *     nothing; or, before downloading, when `directory` holds anything but
 *     the pinned set.
 */
export async function fetchPieceSet(
  source: PieceSetSource,
  directory: string,
  fetcher: Fetcher = httpFetch,
): Promise<'fetched' | 'present'> {
  const present = await Promise.all(
    source.files.map(async file => {
      try {
        return sha256(await readFile(join(directory, file.saveAs)));
      } catch (error: unknown) {
        if (isMissing(error)) return null;
        throw error;
      }
    }),
  );
  if (present.every((digest, i) => digest === source.files[i]?.sha256)) {
    return 'present';
  }
  const held = await readdir(directory).catch((error: unknown) => {
    if (isMissing(error)) return [];
    throw error;
  });
  if (held.length > 0) {
    throw new Error(
      `${directory} holds files that are not ${source.id}'s pinned ones; ` +
        'remove it and fetch again.',
    );
  }
  const downloaded: Array<[PinnedFile, Uint8Array]> = [];
  for (const file of source.files) {
    let bytes: Uint8Array;
    try {
      bytes = await fetcher(file.url);
    } catch (error: unknown) {
      throw new Error(
        `Could not fetch ${source.id}'s ${file.saveAs} from ${file.url}. ` +
          'Nothing was written.',
        {cause: error},
      );
    }
    const digest = sha256(bytes);
    if (digest !== file.sha256) {
      throw new Error(
        `Refusing ${source.id}: ${file.saveAs} from ${file.url} has sha256 ` +
          `${digest}, not the pinned ${file.sha256}. Nothing was written.`,
      );
    }
    downloaded.push([file, bytes]);
  }
  // Written beside the directory and renamed into place, so the client never
  // sees half a set.
  // A failure removes the partial directory, so nothing is left behind.
  const partial = `${directory}.partial-${String(process.pid)}`;
  try {
    await mkdir(partial, {recursive: true});
    for (const [file, bytes] of downloaded) {
      await writeFile(join(partial, file.saveAs), bytes);
    }
    await mkdir(dirname(directory), {recursive: true});
    await rename(partial, directory);
  } finally {
    await rm(partial, {recursive: true, force: true});
  }
  return 'fetched';
}

/**
 * The Vite plugin that, in `vite build`, copies each fetched set's licence
 * text and source pages (every pinned file but the pieces) into the output
 * as piece_sets/<id>/, beside the pieces the build carries, whenever the
 * set's directory is there. `sources` and `directoryOf` are for a test.
 */
export function fetchedSetNotices(
  sources: Readonly<Record<string, PieceSetSource>> = SOURCES,
  directoryOf: (id: string) => string = setDirectory,
): Plugin {
  return {
    name: 'fetched-set-notices',
    apply: 'build',
    async generateBundle() {
      for (const source of Object.values(sources)) {
        const directory = directoryOf(source.id);
        for (const file of source.files) {
          if (file.saveAs.endsWith('.svg')) continue;
          let bytes: Uint8Array;
          try {
            bytes = await readFile(join(directory, file.saveAs));
          } catch (error: unknown) {
            if (isMissing(error)) continue;
            throw error;
          }
          this.emitFile({
            type: 'asset',
            fileName: `piece_sets/${source.id}/${file.saveAs}`,
            source: bytes,
          });
        }
      }
    },
  };
}

if (isEntryPoint(import.meta.filename)) {
  const id = process.argv[2] ?? '';
  const source = SOURCES[id];
  if (source === undefined) {
    console.error(
      `usage: npm run fetch-pieces -- <set>; the sets are ${Object.keys(SOURCES).join(', ')}.`,
    );
    process.exitCode = 2;
  } else {
    const directory = setDirectory(id);
    const outcome = await fetchPieceSet(source, directory);
    console.log(
      outcome === 'present'
        ? `${id} is already in ${directory}.`
        : `Fetched ${source.description} into ${directory}. Rebuild or ` +
            'restart the client and pick the set in View › Board › Pieces.',
    );
  }
}
