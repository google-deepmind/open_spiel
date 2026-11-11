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

// Piece sets the user fetches: `npm run fetch-pieces -- <id>`
// (scripts/fetch_piece_set.ts) downloads a set's twelve SVG files into
// piece_sets/<id>/, a directory git ignores, as `<colour>-<role>.svg`, with
// each file's source page and the set's licence text. The client ships none
// of these files; a set is offered only when its directory holds all of
// them when the client is built or served. An incomplete directory leaves
// that one set out, with an error on the console, and the client runs on.

import type {JSX} from 'react';

import {
  PIECE_COLORS,
  PIECE_ROLES,
  type PieceArtProps,
  type PieceSetModule,
} from './piece_types.ts';

/** The file a fetched set holds for one piece. */
export function pieceFileName({role, color}: PieceArtProps): string {
  return `${color}-${role}.svg`;
}

/** The source page a fetched set holds for the piece file `name`. */
export function sourcePageName(name: string): string {
  return `${name}.source.wikitext`;
}

/**
 * The set `label` drawn from the files `urls`, each a file's URL by its path
 * (as a glob import gives them); null when there are none. A set is complete
 * when it holds every piece's file and source page, and `licence`.
 *
 * @throws Error naming the missing files when the set is incomplete.
 */
export function fetchedPieceSet(
  label: string,
  urls: Readonly<Record<string, string>>,
  licence: string,
): PieceSetModule | null {
  const byName = new Map(
    Object.entries(urls).map(([path, url]) => [
      path.slice(path.lastIndexOf('/') + 1),
      url,
    ]),
  );
  if (byName.size === 0) return null;
  const pieceFiles = PIECE_COLORS.flatMap(color =>
    PIECE_ROLES.map(role => pieceFileName({role, color})),
  );
  const missing = [
    ...pieceFiles,
    ...pieceFiles.map(sourcePageName),
    licence,
  ].filter(name => !byName.has(name));
  if (missing.length > 0) {
    throw new Error(
      `Piece set "${label}" lacks ${missing.join(', ')}; remove its ` +
        'directory and fetch it again (npm run fetch-pieces).',
    );
  }
  function FetchedPiece(piece: PieceArtProps): JSX.Element {
    return (
      <img
        className="chess-piece-art"
        src={byName.get(pieceFileName(piece))}
        alt=""
        draggable={false}
      />
    );
  }
  return {
    label,
    Piece: FetchedPiece,
    imageUrl: piece => {
      const url = byName.get(pieceFileName(piece));
      if (url === undefined) throw new Error(`No ${pieceFileName(piece)}.`);
      return url;
    },
  };
}

/** Every set that may be fetched: its id and the licence file it holds. */
const FETCHABLE: ReadonlyArray<{
  readonly id: string;
  readonly licence: string;
}> = [{id: 'cburnett', licence: 'LICENSE-BSD.wikitext'}];

/** The files under piece_sets/, each URL by its path, by set id. */
function filesFound(): Record<string, Record<string, string>> {
  return {
    cburnett: {
      ...import.meta.glob('/piece_sets/cburnett/*.svg', {
        eager: true,
        query: '?url',
        import: 'default',
      }),
      ...import.meta.glob('/piece_sets/cburnett/*.wikitext', {
        eager: true,
        query: '?url',
        import: 'default',
      }),
    },
  };
}

/**
 * The fetched sets whose files are all present, by id. A set whose
 * directory is incomplete is left out, with an error on the console naming
 * what it lacks, so the rest of the client still runs.
 */
export function fetchedPieceSets(
  found: Readonly<
    Record<string, Readonly<Record<string, string>>>
  > = filesFound(),
): Array<[string, PieceSetModule]> {
  return FETCHABLE.flatMap(({id, licence}): Array<[string, PieceSetModule]> => {
    try {
      const module = fetchedPieceSet(id, found[id] ?? {}, licence);
      return module === null ? [] : [[id, module]];
    } catch (error: unknown) {
      console.error(`Piece set "${id}" is not offered.`, error);
      return [];
    }
  });
}
