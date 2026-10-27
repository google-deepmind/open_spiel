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

import {act} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {Presentation} from '../contract/contract.ts';
import {type Rendered, render} from '../testing/render.tsx';
import {button} from '../testing/render.tsx';
import {GridBoard} from './grid_board.tsx';
import {readGridMoves, readGridView} from './grid_view.ts';
import {pieceSetFor, registerPieceSet} from './piece_sets.ts';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

function grid(fields: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    rows: 2,
    columns: 2,
    cells: [
      ['x', null],
      [null, 'o'],
    ],
    piece_set: 'marks',
    mode: 'claim',
    highlights: [[0, 0]],
    ...fields,
  };
}

function presentation(hints: unknown[], view = grid()): Presentation {
  return {
    renderer: 'grid',
    view: {grid: view},
    actions: hints.map((hint, id) => ({id, hint})),
  };
}

// Renders the board of `shown`, playing through `play`.
function show(
  shown: Presentation,
  play: ((action: number) => void) | null,
  labels = new Map<number, string>(),
): HTMLElement {
  rendered = render(
    <GridBoard
      view={readGridView((shown.view as {grid: unknown}).grid)}
      moves={readGridMoves(shown)}
      play={play}
      labels={labels}
    />,
  );
  return rendered.container;
}

function cell(container: HTMLElement, name: string): HTMLElement {
  const found = container.querySelector(`[aria-label^="${name}:"]`);
  if (!(found instanceof HTMLElement)) throw new Error(`No cell ${name}.`);
  return found;
}

describe('readGridView', () => {
  it('reads the view and its optional fields', () => {
    const view = readGridView(grid({flipped: true, column_labels: ['a', 'b']}));
    expect(view.cells[1]).toEqual([null, 'o']);
    expect(view.flipped).toBe(true);
    expect(view.checkered).toBe(false);
    expect(view.columnLabels).toEqual(['a', 'b']);
    expect(view.rowLabels).toBeNull();
    expect(view.alerts).toEqual([]);
    expect(readGridView(grid({alerts: [[1, 0]]})).alerts).toEqual([[1, 0]]);
  });

  it('names the field at fault', () => {
    expect(() => readGridView(grid({mode: 'fly'}))).toThrow(
      'grid.mode is "fly"',
    );
    expect(() => readGridView(grid({cells: [['x']]}))).toThrow(
      'grid.cells is not 2 rows of 2.',
    );
    expect(() => readGridView(grid({rows: undefined}))).toThrow('grid.rows');
  });
});

describe('readGridMoves', () => {
  it('reads each kind of move, skipping hints it does not place', () => {
    expect(
      readGridMoves(
        presentation([
          {cell: [0, 1]},
          {column: 1},
          {from: [0, 0], to: [1, 0]},
          {button: 'Pass'},
        ]),
      ),
    ).toEqual([
      {kind: 'claim', action: 0, cell: [0, 1]},
      {kind: 'drop', action: 1, column: 1},
      {kind: 'move', action: 2, from: [0, 0], to: [1, 0]},
    ]);
  });
});

describe('piece sets', () => {
  it('draws codes by set, and refuses unknown codes and sets', () => {
    expect(pieceSetFor('chess').piece('K')).toMatchObject({tone: 'light'});
    expect(pieceSetFor('chess').piece('n').name).toBe('black knight');
    expect(() => pieceSetFor('chess').piece('z')).toThrow('no piece "z"');
    expect(() => pieceSetFor('none')).toThrow('no piece set "none"');
    expect(() => {
      registerPieceSet(pieceSetFor('marks'));
    }).toThrow('already registered');
  });
});

describe('GridBoard', () => {
  it('claims a cell with one press, and offers nothing it cannot play', () => {
    const play = vi.fn();
    const container = show(presentation([{cell: [0, 1]}]), play);
    expect(container.querySelectorAll('button')).toHaveLength(1);
    expect(
      cell(container, 'row 1, column 1').getAttribute('data-highlight'),
    ).toBe('true');
    act(() => {
      cell(container, 'row 1, column 2').click();
    });
    expect(play).toHaveBeenCalledWith(0);
  });

  it('offers no cell to a viewer who cannot play', () => {
    const container = show(presentation([{cell: [0, 1]}]), null);
    expect(container.querySelector('button')).toBeNull();
  });

  it('drops into a column from any of its cells', () => {
    const play = vi.fn();
    const container = show(
      presentation([{column: 0}], grid({mode: 'drop'})),
      play,
    );
    act(() => {
      cell(container, 'row 2, column 1').click();
    });
    expect(play).toHaveBeenCalledWith(0);
    expect(container.querySelectorAll('button')).toHaveLength(2);
  });

  it('selects a piece, marks its targets, and moves on a second press', () => {
    const play = vi.fn();
    const container = show(
      presentation(
        [
          {from: [0, 0], to: [0, 1]},
          {from: [0, 0], to: [1, 0]},
        ],
        grid({mode: 'move', column_labels: ['a', 'b'], row_labels: ['2', '1']}),
      ),
      play,
    );
    act(() => {
      cell(container, 'a2').click();
    });
    expect(cell(container, 'a2').getAttribute('data-selected')).toBe('true');
    expect(cell(container, 'a1').getAttribute('data-target')).toBe('true');
    act(() => {
      cell(container, 'a1').click();
    });
    expect(play).toHaveBeenCalledWith(1);
  });

  it('asks which move to play when several join the same cells', () => {
    const play = vi.fn();
    const container = show(
      presentation(
        [
          {from: [0, 0], to: [0, 1]},
          {from: [0, 0], to: [0, 1]},
        ],
        grid({mode: 'move'}),
      ),
      play,
      new Map([
        [0, 'promote to queen'],
        [1, 'promote to knight'],
      ]),
    );
    act(() => {
      cell(container, 'row 1, column 1').click();
    });
    act(() => {
      cell(container, 'row 1, column 2').click();
    });
    expect(play).not.toHaveBeenCalled();
    act(() => {
      button('promote to knight', container).click();
    });
    expect(play).toHaveBeenCalledWith(1);
  });
});

describe('GridBoard alerts', () => {
  it('marks the alerted cells', () => {
    const container = show(presentation([], grid({alerts: [[1, 1]]})), null);
    expect(cell(container, 'row 2, column 2').getAttribute('data-alert')).toBe(
      'true',
    );
    expect(cell(container, 'row 1, column 1').hasAttribute('data-alert')).toBe(
      false,
    );
  });
});
