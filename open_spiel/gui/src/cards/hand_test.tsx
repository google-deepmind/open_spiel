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

import {type Rendered, render} from '../testing/render.tsx';
import {CardFace} from './card_face.tsx';
import {DEFAULT_CARD_BACK} from './library/decks/backs.tsx';
import {CardStyleProvider} from './library/index.ts';
import {Hand} from './hand.tsx';
import {Pile, Trick} from './pile.tsx';

let rendered: Rendered | null = null;

function show(tree: Parameters<typeof render>[0]): HTMLElement {
  rendered = render(tree);
  return rendered.container;
}

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

describe('CardFace', () => {
  it('draws rank and suit, named for assistive technology', () => {
    const container = show(<CardFace code="Th" />);
    const card = container.querySelector('[aria-label="Ten of hearts"]');
    expect(card?.getAttribute('data-tone')).toBe('red');
    expect(card?.textContent).toBe('10♥♥');
    expect(container.querySelector('button')).toBeNull();
  });

  it('draws a hidden card face down', () => {
    const container = show(<CardFace code={null} />);
    expect(
      container.querySelector('[aria-label="Face-down card"]'),
    ).not.toBeNull();
  });

  it("draws a face-down card with the card library's back, default blue lattice", () => {
    const container = show(<CardFace code={null} />);
    const back = container.querySelector(
      '[aria-label="Face-down card"] svg[data-back]',
    );
    expect(back?.getAttribute('data-back')).toBe(DEFAULT_CARD_BACK);
  });

  it('draws the back the Card back picker chose', () => {
    const container = show(
      <CardStyleProvider value={{deck: 'basic', back: 'stripe-navy'}}>
        <CardFace code={null} />
      </CardStyleProvider>,
    );
    const back = container.querySelector(
      '[aria-label="Face-down card"] svg[data-back]',
    );
    expect(back?.getAttribute('data-deck')).toBe('basic');
    expect(back?.getAttribute('data-back')).toBe('stripe-navy');
  });
});

describe('Hand', () => {
  it('makes buttons of the playable cards only, which play their action', () => {
    const play = vi.fn();
    const container = show(
      <Hand
        label="Your hand"
        cards={['As', 'Kd', null]}
        playable={new Map([['Kd', 9]])}
        play={play}
      />,
    );
    const buttons = container.querySelectorAll('button');
    expect(buttons).toHaveLength(1);
    expect(buttons[0]?.getAttribute('aria-label')).toBe(
      'King of diamonds: play',
    );
    act(() => {
      buttons[0]?.click();
    });
    expect(play).toHaveBeenCalledWith(9);
  });

  it('plays nothing when the viewer cannot play', () => {
    const container = show(
      <Hand
        label="Hand"
        cards={['As']}
        playable={new Map([['As', 1]])}
        play={null}
      />,
    );
    expect(container.querySelector('button')).toBeNull();
  });

  it('hands every shown card to onCard when given one', () => {
    const onCard = vi.fn();
    const container = show(
      <Hand
        label="Hand"
        cards={['As', '2c']}
        playable={new Map([['2c', 4]])}
        onCard={onCard}
        selected={new Set(['As'])}
      />,
    );
    const [ace, two] = container.querySelectorAll('button');
    expect(ace?.getAttribute('data-selected')).toBe('true');
    act(() => {
      ace?.click();
      two?.click();
    });
    expect(onCard.mock.calls).toEqual([
      ['As', undefined],
      ['2c', 4],
    ]);
  });
});

describe('Pile and Trick', () => {
  it('shows a pile by its top card and size, the top playing topAction', () => {
    const play = vi.fn();
    const container = show(
      <Pile label="Stock" cards={[null, null]} topAction={52} play={play} />,
    );
    expect(container.textContent).toContain('2 cards');
    act(() => {
      container.querySelector('button')?.click();
    });
    expect(play).toHaveBeenCalledWith(52);
  });

  it('shows each trick card under its player', () => {
    const container = show(
      <Trick
        plays={[
          {label: 'North', card: 'Qs'},
          {label: 'East', card: null},
        ]}
      />,
    );
    expect(
      [...container.querySelectorAll('figcaption')].map(c => c.textContent),
    ).toEqual(['North', 'East']);
  });
});
