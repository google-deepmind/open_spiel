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

import type {JSX} from 'react';
import {act} from 'react';
import type {Root} from 'react-dom/client';
import {createRoot} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {ChanceMode, Frame} from '../contract/contract.ts';
import {only} from '../testing/render.tsx';
import {ReplayControl} from './replay_control.tsx';
import {recordedFrame} from './testing/recorded_frames.ts';

let container: HTMLElement;
let root: Root;
let onGoTo: ReturnType<typeof vi.fn<(ply: number) => void>>;

function show(frame: Frame, busy = false, chanceMode?: ChanceMode): void {
  const control: JSX.Element = (
    <ReplayControl
      frame={frame}
      busy={busy}
      chanceMode={
        chanceMode ??
        (frame.session.game === 'kuhn_poker'
          ? 'explicit_stochastic'
          : 'deterministic')
      }
      onGoTo={onGoTo}
    />
  );
  act(() => {
    root.render(control);
  });
}

function stepButton(label: string): HTMLButtonElement {
  const element = only(container, `button[aria-label="${label}"]`);
  if (!(element instanceof HTMLButtonElement)) throw new Error(label);
  return element;
}

function slider(): HTMLInputElement {
  const element = only(container, 'input[type="range"]');
  if (!(element instanceof HTMLInputElement)) throw new Error('slider');
  return element;
}

// The step buttons, by label, mapped to whether each is available.
function available(): Record<string, boolean> {
  return Object.fromEntries(
    ['Go to start', 'Previous move', 'Next move', 'Go to end'].map(label => [
      label,
      stepButton(label).getAttribute('aria-disabled') === 'false',
    ]),
  );
}

function position(): string | null {
  return slider().getAttribute('aria-valuetext');
}

// Moves the slider as a user does, through the value setter React watches.
function slide(index: number): void {
  act(() => {
    Reflect.set(HTMLInputElement.prototype, 'value', String(index), slider());
    slider().dispatchEvent(new Event('input', {bubbles: true}));
  });
}

function toggleSkipChance(): void {
  act(() => {
    only(container, 'input[type="checkbox"]').click();
  });
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  onGoTo = vi.fn<(ply: number) => void>();
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
});

describe('ReplayControl controls', () => {
  it('lays out its controls in order, named for assistive technology', async () => {
    show(await recordedFrame('kuhn_poker/oracle/live'));
    const group = only(container, '[role="group"]');
    expect(group.getAttribute('aria-label')).toBe('Replay');
    expect(
      Array.from(
        group.children,
        child => child.getAttribute('aria-label') ?? child.textContent,
      ),
    ).toEqual([
      'Go to start',
      'Previous move',
      'Replay position',
      '4 / 4',
      'Next move',
      'Go to end',
    ]);
    expect(
      Array.from(group.querySelectorAll('button'), node => node.textContent),
    ).toEqual(['|<', '<', '>', '>|']);
    expect(stepButton('Next move').title).toBe('Next move');
    expect(container.querySelector('[role="status"]')).toBeNull();
  });

  it('offers Skip chance only in a game with chance', async () => {
    show(await recordedFrame('tic_tac_toe/oracle/2'));
    expect(container.querySelector('input[type="checkbox"]')).toBeNull();
    expect(position()).toBe('step 2 of 3');
    show(
      await recordedFrame('kuhn_poker/oracle/3'),
      false,
      'sampled_stochastic',
    );
    const label = only(container, 'label');
    expect(label.textContent).toBe('Skip chance');
    expect(label.title).toBe('Skip over chance actions (card deals, etc.)');
  });

  it('says why nothing steps before anything is played', async () => {
    show(await recordedFrame('tic_tac_toe/oracle/new'));
    const reason = 'Nothing has been played yet.';
    expect(only(container, '[role="group"]').title).toBe(reason);
    expect(stepButton('Go to start').title).toBe(reason);
    expect(slider().getAttribute('aria-disabled')).toBe('true');
    expect(position()).toBe('step 0 of 0');
    expect(Object.values(available())).toEqual([false, false, false, false]);
  });
});

describe('ReplayControl stepping', () => {
  it('at the live position, offers only the way back', async () => {
    // kuhn_poker's history: two deals, then three seat actions.
    show(await recordedFrame('kuhn_poker/oracle/live'));
    expect(position()).toBe('step 4 of 4');
    expect(available()).toEqual({
      'Go to start': true,
      'Previous move': true,
      'Next move': false,
      'Go to end': false,
    });
    stepButton('Next move').click();
    stepButton('Previous move').click();
    stepButton('Go to start').click();
    expect(onGoTo.mock.calls).toEqual([[4], [0]]);
  });

  it('in a replay, steps both ways', async () => {
    show(await recordedFrame('kuhn_poker/spectator/3'));
    expect(position()).toBe('step 2 of 4');
    expect(Object.values(available())).toEqual([true, true, true, true]);
    for (const label of [
      'Go to start',
      'Previous move',
      'Next move',
      'Go to end',
    ]) {
      stepButton(label).click();
    }
    expect(onGoTo.mock.calls).toEqual([[0], [2], [4], [5]]);
  });

  it('skips chance unless told not to, for every viewer', async () => {
    for (const viewer of ['oracle', '0', '1', 'spectator']) {
      const frame = await recordedFrame(`kuhn_poker/${viewer}/0`);
      show(frame);
      expect(position()).toBe('step 0 of 4');
      stepButton('Next move').click();
      expect(onGoTo).toHaveBeenLastCalledWith(2);

      toggleSkipChance();
      expect(position()).toBe('step 0 of 5');
      stepButton('Next move').click();
      expect(onGoTo).toHaveBeenLastCalledWith(1);
      toggleSkipChance();
    }
  });

  it('counts a chance position among its stops when it is shown', async () => {
    const frame = await recordedFrame('kuhn_poker/oracle/0');
    show({...frame, ply: 1});
    expect(position()).toBe('step 1 of 5');
    stepButton('Previous move').click();
    stepButton('Next move').click();
    expect(onGoTo.mock.calls).toEqual([[0], [2]]);
  });

  it('refuses a frame whose history is not its plies', async () => {
    const frame = await recordedFrame('kuhn_poker/oracle/live');
    expect(() => {
      show({...frame, num_plies: 4});
    }).toThrow('A frame of 4 plies carries 5 history entries.');
  });
});

describe('ReplayControl while busy', () => {
  it('keeps every button focusable and inert', async () => {
    show(await recordedFrame('kuhn_poker/oracle/3'), true);
    expect(Object.values(available())).toEqual([false, false, false, false]);
    const next = stepButton('Next move');
    next.focus();
    next.click();
    expect(onGoTo).not.toHaveBeenCalled();
    expect(next.disabled).toBe(false);
    expect(document.activeElement).toBe(next);
  });

  it('holds Skip chance as it is', async () => {
    show(await recordedFrame('kuhn_poker/oracle/3'), true);
    const box = only(container, 'input[type="checkbox"]');
    expect(box instanceof HTMLInputElement && box.disabled).toBe(true);
  });

  it('holds slider moves while busy and sends the latest once idle', async () => {
    const frame = await recordedFrame('kuhn_poker/oracle/3');
    show(frame, true);
    slide(4);
    expect(position()).toBe('step 4 of 4');
    slide(1);
    expect(position()).toBe('step 1 of 4');
    expect(onGoTo).not.toHaveBeenCalled();
    // Idle again: the latest move goes out, alone, and shows until the
    // session has shown it.
    show(frame);
    expect(onGoTo.mock.calls).toEqual([[2]]);
    expect(position()).toBe('step 1 of 4');
    show(frame, true);
    expect(position()).toBe('step 1 of 4');
    show(await recordedFrame('kuhn_poker/oracle/0'));
    expect(position()).toBe('step 0 of 4');
  });

  it('sends a drag across every stop as one request per round trip', async () => {
    const frame = await recordedFrame('kuhn_poker/oracle/0');
    show(frame);
    toggleSkipChance();
    // The first move goes out at once, and the session is busy until it
    // returns; the rest of the drag waits for it.
    slide(1);
    show(frame, true);
    for (const index of [2, 3, 4, 5, 4, 3]) slide(index);
    expect(onGoTo.mock.calls).toEqual([[1]]);
    show(frame);
    expect(onGoTo.mock.calls).toEqual([[1], [3]]);
  });

  it('shows the frame when a move leaves the session idle', async () => {
    show(await recordedFrame('kuhn_poker/oracle/3'));
    slide(4);
    expect(onGoTo).toHaveBeenLastCalledWith(5);
    expect(position()).toBe('step 2 of 4');
  });
});
