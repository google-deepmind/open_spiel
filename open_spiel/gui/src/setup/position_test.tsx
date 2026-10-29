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

import {type JSX, act, useState} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {Frame} from '../contract/contract.ts';
import {ContractError} from '../session/errors.ts';
import {press} from '../testing/dom.ts';
import type {Rendered} from '../testing/render.tsx';
import {button, only, render} from '../testing/render.tsx';
import {Position} from './position.tsx';
import {boardFrame} from './testing/fixtures.ts';
import {enter} from './testing/input.ts';

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
});

type OnReset = ReturnType<typeof vi.fn<(history: readonly number[]) => void>>;

function show(
  frame: Frame = boardFrame(),
  busy = false,
): {container: HTMLElement; onReset: OnReset} {
  const onReset = vi.fn<(history: readonly number[]) => void>();
  rendered = render(
    <Position frame={frame} busy={busy} error={null} onReset={onReset} />,
  );
  return {container: rendered.container, onReset};
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

describe('Position', () => {
  it('says what its controls do and starts with Set state closed', () => {
    const {container} = show();
    expect(only(container, '.setup-card__description').textContent).toBe(
      'Set state or reset. Each replaces the current position.',
    );
    expect(container.querySelector('textarea')).toBeNull();
    expect(button('Set state', container).getAttribute('aria-expanded')).toBe(
      'false',
    );
  });

  it('opens Set state and replays the typed actions', () => {
    const {container, onReset} = show();
    click(button('Set state', container));
    const field = only(container, 'textarea') as HTMLTextAreaElement;
    expect(field.placeholder).toBe('e.g. [1, 4, 9] or 1 4 9');
    expect(only(container, 'label').textContent).toBe(
      'Action sequence (comma or space separated, brackets ok)',
    );
    enter(field, '[0, 4, 8]');
    click(button('Apply', container));
    expect(onReset).toHaveBeenCalledExactlyOnceWith([0, 4, 8]);
  });

  it('reads an episode object', () => {
    const {container, onReset} = show();
    click(button('Set state', container));
    enter(only(container, 'textarea'), '{"actions": [2, 7]}');
    click(button('Apply', container));
    expect(onReset).toHaveBeenCalledExactlyOnceWith([2, 7]);
  });

  it('says what is wrong with the actions and replays nothing', () => {
    const {container, onReset} = show();
    click(button('Set state', container));
    const field = only(container, 'textarea');
    enter(field, '0 x');
    click(button('Apply', container));
    expect(onReset).not.toHaveBeenCalled();
    const alert = only(container, '[role="alert"]');
    expect(alert.textContent).toBe('x is not an action id.');
    expect(field.getAttribute('aria-invalid')).toBe('true');
    expect(field.getAttribute('aria-describedby')).toBe(alert.id);
  });
});

describe('Position answers', () => {
  it('says Applied once the server takes the history, and closes', () => {
    let answer: (busy: boolean, error: ContractError | null) => void = () =>
      undefined;
    function Shown(): JSX.Element {
      const [state, setState] = useState<{
        busy: boolean;
        error: ContractError | null;
      }>({busy: false, error: null});
      answer = (busy, error) => {
        setState({busy, error});
      };
      return (
        <Position
          frame={boardFrame()}
          busy={state.busy}
          error={state.error}
          onReset={() => {
            answer(true, null);
          }}
        />
      );
    }
    rendered = render(<Shown />);
    const {container} = rendered;
    click(button('Set state', container));
    enter(only(container, 'textarea'), '1');
    click(button('Apply', container));
    expect(button('Applying…', container).disabled).toBe(true);
    act(() => {
      answer(false, null);
    });
    expect(container.querySelector('textarea')).toBeNull();
    click(button('Set state', container));
    expect(only(container, '[role="status"]').textContent).toBe('Applied');

    enter(only(container, 'textarea'), '9');
    click(button('Apply', container));
    const refusal = new ContractError(
      {
        contract_version: 8,
        code: 'illegal_action',
        message: 'Ply 0 is illegal.',
        seat: null,
        ply: 0,
      },
      409,
    );
    act(() => {
      answer(false, refusal);
    });
    expect(only(container, '[role="alert"]').textContent).toBe(
      'Ply 0 is illegal.',
    );
  });

  it('closes Set state', () => {
    const {container} = show();
    click(button('Set state', container));
    click(button('Close', container));
    expect(container.querySelector('textarea')).toBeNull();
  });
});

describe('Position reset', () => {
  it('resets the game once the reset is confirmed', () => {
    const {container, onReset} = show(boardFrame({history: [3]}));
    click(button('Reset', container));
    const dialog = only(document.body, '[role="dialog"]');
    expect(dialog.textContent).toContain('Reset Game?');
    expect(dialog.textContent).toContain(
      'This will clear all history and start a new game.',
    );
    click(button('Reset', dialog));
    expect(onReset).toHaveBeenCalledExactlyOnceWith([]);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('resets nothing when the reset is cancelled or escaped', () => {
    const {container, onReset} = show();
    click(button('Reset', container));
    expect(only(document.body, '[role="dialog"]').textContent).toContain(
      'This will start a new game.',
    );
    click(button('Cancel', only(document.body, '[role="dialog"]')));
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    click(button('Reset', container));
    press(document.body, 'Escape');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(onReset).not.toHaveBeenCalled();
  });

  it('changes nothing while a request is in flight', () => {
    const {container} = show(boardFrame(), true);
    expect(button('Set state', container).disabled).toBe(true);
    expect(button('Reset', container).disabled).toBe(true);
  });
});
