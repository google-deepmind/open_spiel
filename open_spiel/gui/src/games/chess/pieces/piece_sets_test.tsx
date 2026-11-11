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

import {existsSync} from 'node:fs';
import {resolve} from 'node:path';

import type {JSX} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {type Rendered, render} from '../../../testing/render.tsx';
import {PIECE_SET_KEY} from '../chess_config_section.tsx';
import {ChessPieceArt} from '../chess_pieces.tsx';
import {
  fetchedPieceSet,
  fetchedPieceSets,
  pieceFileName,
  sourcePageName,
} from './fetched_sets.tsx';
import {
  DEFAULT_PIECE_SET,
  getPieceSet,
  pieceSetIds,
  registerPieceSet,
} from './piece_sets.ts';
import {
  PIECE_COLORS,
  PIECE_ROLES,
  type PieceArtProps,
  type PieceColor,
  type PieceRole,
} from './piece_types.ts';

let rendered: Rendered | null = null;
let unregister: (() => void) | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  unregister?.();
  unregister = null;
  window.localStorage.clear();
});

function FakePiece({role, color}: PieceArtProps): JSX.Element {
  return (
    <span className="chess-piece-art" data-testid="fake-piece">
      {color} {role}
    </span>
  );
}

// A piece's props, spread rather than written as a colour attribute.
function piece(role: PieceRole, color: PieceColor): PieceArtProps {
  return {role, color};
}

const ALL_PIECES = PIECE_COLORS.flatMap(color =>
  PIECE_ROLES.map(role => ({role, color})),
);

describe('the piece-set registry', () => {
  it('draws with the Staunton set by default', () => {
    expect(DEFAULT_PIECE_SET).toBe('staunton');
    expect(pieceSetIds()[0]).toBe('staunton');
    rendered = render(<ChessPieceArt {...piece('knight', 'black')} />);
    const svg = rendered.container.querySelector('svg.chess-piece-art');
    expect(svg?.getAttribute('viewBox')).toBe('0 0 100 100');
    expect(svg?.querySelectorAll('path').length).toBeGreaterThan(0);
  });

  it('draws every Staunton piece in both colours', () => {
    const {Piece} = getPieceSet('staunton');
    for (const piece of ALL_PIECES) {
      const one = render(<Piece {...piece} />);
      expect(one.container.querySelectorAll('svg path').length).toBeGreaterThan(
        0,
      );
      one.unmount();
    }
  });

  it('takes a set as one module and draws with it once picked', () => {
    unregister = registerPieceSet('fake', {label: 'Fake', Piece: FakePiece});
    expect(pieceSetIds()).toContain('fake');
    expect(getPieceSet('fake').label).toBe('Fake');
    window.localStorage.setItem(`open_spiel.gui.${PIECE_SET_KEY}`, 'fake');
    rendered = render(<ChessPieceArt {...piece('queen', 'white')} />);
    expect(
      rendered.container.querySelector('[data-testid="fake-piece"]')
        ?.textContent,
    ).toBe('white queen');
  });

  it('removes a set registered for a test', () => {
    const remove = registerPieceSet('gone', {label: 'Gone', Piece: FakePiece});
    remove();
    expect(pieceSetIds()).not.toContain('gone');
    expect(() => getPieceSet('gone')).toThrow(/no piece set "gone"/);
  });

  it('refuses an id taken or blank', () => {
    expect(() => {
      registerPieceSet('staunton', {label: 'Again', Piece: FakePiece});
    }).toThrow('Piece set "staunton" is already registered.');
    expect(() => {
      registerPieceSet(' ', {label: 'Blank', Piece: FakePiece});
    }).toThrow('A piece set has a blank id.');
  });

  it('fails loud on an unknown id, naming the sets', () => {
    expect(() => getPieceSet('no_such')).toThrow(
      /There is no piece set "no_such"; the sets are staunton/,
    );
  });

  it('offers cburnett exactly when its files were fetched', () => {
    // The client ships no cburnett files; a checkout has them only once the
    // user runs npm run fetch-pieces -- cburnett.
    const fetched = existsSync(
      resolve(import.meta.dirname, '../../../../piece_sets/cburnett'),
    );
    expect(pieceSetIds().includes('cburnett')).toBe(fetched);
  });
});

describe('a fetched piece set', () => {
  const DIRECTORY = '/piece_sets/cburnett';

  const LICENCE = 'LICENSE-BSD.wikitext';

  function fetchedDirectory(): Record<string, string> {
    const names = [
      ...ALL_PIECES.flatMap(piece => [
        pieceFileName(piece),
        sourcePageName(pieceFileName(piece)),
      ]),
      LICENCE,
    ];
    return Object.fromEntries(
      names.map(name => [`${DIRECTORY}/${name}`, `/assets/${name}`]),
    );
  }

  function without(suffix: string): Record<string, string> {
    return Object.fromEntries(
      Object.entries(fetchedDirectory()).filter(
        ([path]) => !path.endsWith(suffix),
      ),
    );
  }

  it('is offered only when its directory holds files', () => {
    expect(fetchedPieceSet('cburnett', {}, LICENCE)).toBeNull();
    expect(
      fetchedPieceSet('cburnett', fetchedDirectory(), LICENCE)?.label,
    ).toBe('cburnett');
  });

  it('draws each piece from its own file', () => {
    const module = fetchedPieceSet('cburnett', fetchedDirectory(), LICENCE);
    if (module === null) throw new Error('No set.');
    const {Piece} = module;
    rendered = render(<Piece {...piece('bishop', 'black')} />);
    const image = rendered.container.querySelector('img.chess-piece-art');
    expect(image?.getAttribute('src')).toBe('/assets/black-bishop.svg');
    expect(image?.getAttribute('alt')).toBe('');
  });

  it('refuses a directory missing pieces, naming them', () => {
    expect(() =>
      fetchedPieceSet('cburnett', without('/white-king.svg'), LICENCE),
    ).toThrow('Piece set "cburnett" lacks white-king.svg; remove its');
  });

  it('refuses a directory without its licence or a source page', () => {
    expect(() =>
      fetchedPieceSet('cburnett', without(LICENCE), LICENCE),
    ).toThrow(`lacks ${LICENCE};`);
    expect(() =>
      fetchedPieceSet(
        'cburnett',
        without('/black-pawn.svg.source.wikitext'),
        LICENCE,
      ),
    ).toThrow('lacks black-pawn.svg.source.wikitext;');
  });

  it('leaves out only an incomplete set, with an error on the console', () => {
    const errors = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    expect(fetchedPieceSets({cburnett: without(LICENCE)})).toEqual([]);
    const call: unknown[] = errors.mock.calls[0] ?? [];
    expect(call[0]).toBe('Piece set "cburnett" is not offered.');
    expect(call[1]).toBeInstanceOf(Error);
    expect(String(call[1])).toContain(`lacks ${LICENCE};`);
    errors.mockRestore();
    expect(
      fetchedPieceSets({cburnett: fetchedDirectory()}).map(([id]) => id),
    ).toEqual(['cburnett']);
  });
});
