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
import {afterEach, describe, expect, it} from 'vitest';

import type {Presentation} from '../contract/contract.ts';
import {defineRenderers} from '../renderers/renderer.ts';
import {frame, session} from '../testing/frames.ts';
import {type Rendered, render} from '../testing/render.tsx';
import {GameplayBoard} from './board.tsx';

const TIC_TAC_TOE: Presentation = {
  renderer: 'grid',
  view: {
    grid: {
      rows: 3,
      columns: 3,
      cells: [
        [null, null, null],
        [null, 'x', null],
        [null, null, null],
      ],
      piece_set: 'marks',
      mode: 'claim',
      highlights: [],
    },
    lines: ['O to move.'],
  },
  actions: [
    {id: 0, hint: {cell: [0, 0]}},
    {id: 1, hint: {cell: [0, 1]}},
  ],
};

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

function show(tree: Parameters<typeof render>[0]): HTMLElement {
  rendered = render(tree);
  return rendered.container;
}

describe('GameplayBoard presentation', () => {
  it('draws the presentation with its renderer, and plays on it', () => {
    const shown = session(
      frame({presentation: TIC_TAC_TOE, state_text: 'text'}),
    );
    const container = show(<GameplayBoard session={shown} />);
    expect(container.querySelector('pre')).toBeNull();
    expect(container.textContent).toContain('O to move.');
    act(() => {
      container
        .querySelector<HTMLElement>('[aria-label^="row 1, column 2"]')
        ?.click();
    });
    expect(shown.commands.act).toHaveBeenCalledWith(1);
  });

  it('offers no move where the viewer cannot play', () => {
    const shown = session(
      frame({
        presentation: TIC_TAC_TOE,
        viewer: {kind: 'spectator', seat: null},
      }),
    );
    const container = show(<GameplayBoard session={shown} />);
    expect(container.querySelector('button')).toBeNull();
  });

  it('falls back to the text when no renderer has the name', () => {
    const shown = session(
      frame({
        presentation: {...TIC_TAC_TOE, renderer: 'unknown'},
        state_text: 'text',
      }),
    );
    const container = show(<GameplayBoard session={shown} />);
    expect(container.querySelector('pre')?.textContent).toBe('text');
  });

  it('shows a renderer that throws as an error over the text', () => {
    const broken = defineRenderers([
      {
        name: 'grid',
        render: () => {
          throw new Error('bad view');
        },
      },
    ]);
    const shown = session(
      frame({presentation: TIC_TAC_TOE, state_text: 'text'}),
    );
    const errors = console.error;
    console.error = () => undefined;
    try {
      const container = show(
        <GameplayBoard session={shown} renderers={broken} />,
      );
      expect(container.querySelector('[role="alert"]')?.textContent).toBe(
        'The board could not be drawn: bad view',
      );
      expect(container.querySelector('pre')?.textContent).toBe('text');
    } finally {
      console.error = errors;
    }
  });
});

describe('defineRenderers', () => {
  it('refuses a repeated or blank name', () => {
    const render = (): never => {
      throw new Error('unused');
    };
    expect(() =>
      defineRenderers([
        {name: 'grid', render},
        {name: 'grid', render},
      ]),
    ).toThrow('defined twice');
    expect(() => defineRenderers([{name: ' ', render}])).toThrow('blank name');
  });
});
