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

import {afterEach, describe, expect, it} from 'vitest';

import {ANALYSIS_VIEW_SEAT_KEY} from '../game/bottom_seat.ts';
import {enter, labelled} from '../testing/input.ts';
import {type Rendered, only, render} from '../testing/render.tsx';
import {SetViewSetting} from '../testing/view_settings.tsx';
import {Perspective} from './perspective.tsx';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

describe('Perspective', () => {
  it('offers the seats alone in analysis', () => {
    rendered = render(
      <>
        <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={null} />
        <Perspective viewer={{kind: 'oracle', seat: null}} numSeats={3} />
      </>,
    );
    const select = labelled(rendered.container, 'Perspective');
    if (!(select instanceof HTMLSelectElement)) throw new Error('No select.');
    expect([...select.options].map(option => option.textContent)).toEqual([
      'Select a seat',
      'Seat 0',
      'Seat 1',
      'Seat 2',
    ]);
    expect(select.value).toBe('0');
    expect(rendered.container.textContent).not.toMatch(/Oracle|Spectator/);
    enter(select, '2');
    expect(select.value).toBe('2');
  });

  it('notes that seats appear later when there are none', () => {
    rendered = render(
      <Perspective viewer={{kind: 'oracle', seat: null}} numSeats={0} />,
    );
    expect(rendered.container.textContent).toContain(
      'Seat options will appear once players or hands are detected in the current session.',
    );
  });

  it('is locked to the seat in a game, with the reason', () => {
    rendered = render(
      <Perspective viewer={{kind: 'seat', seat: 1}} numSeats={2} />,
    );
    const {container} = rendered;
    expect(container.querySelector('select')).toBeNull();
    expect(
      only(container, '[data-testid="perspective-locked"]').textContent,
    ).toBe('Seat 1 (you)');
    expect(
      only(container, '[data-testid="perspective-lock-reason"]').textContent,
    ).toBe('Locked to your seat during a game');
  });

  it('is locked for a spectator, with the reason', () => {
    rendered = render(
      <Perspective viewer={{kind: 'spectator', seat: null}} numSeats={2} />,
    );
    const {container} = rendered;
    expect(
      only(container, '[data-testid="perspective-locked"]').textContent,
    ).toBe('Spectating');
    expect(
      only(container, '[data-testid="perspective-lock-reason"]').textContent,
    ).toBe('Spectating: the perspective is fixed during a game');
  });
});
