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

import type {
  CreateSession,
  Frame,
  ParameterSpec,
} from '../contract/contract.ts';
import {ContractError} from '../session/errors.ts';
import type {Rendered} from '../testing/render.tsx';
import {only, render} from '../testing/render.tsx';
import {GameOptions, type GameOptionsProps} from './game_options.tsx';
import {
  BOARD_SIZE,
  DEFAULTS,
  ENTRY,
  PLAYERS,
  RULES,
  TITLE,
  boardFrame,
} from './testing/fixtures.ts';
import {enter} from './testing/input.ts';

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
});

function show(props: Partial<GameOptionsProps> = {}): {
  container: HTMLElement;
  onCreate: ReturnType<typeof vi.fn<(body: CreateSession) => void>>;
} {
  const onCreate = vi.fn<(body: CreateSession) => void>();
  rendered = render(
    <GameOptions
      entry={ENTRY}
      frame={boardFrame()}
      busy={false}
      error={null}
      onCreate={onCreate}
      {...props}
    />,
  );
  return {container: rendered.container, onCreate};
}

// The control labelled `name`.
function control(container: HTMLElement, name: string): HTMLElement {
  const label = Array.from(container.querySelectorAll('label')).find(
    candidate => candidate.textContent === name,
  );
  const element = document.getElementById(label?.htmlFor ?? '');
  if (element === null) throw new Error(`No control is labelled ${name}.`);
  return element;
}

// Commits a typed value, as leaving its field does.
function commit(element: HTMLElement): void {
  act(() => {
    element.dispatchEvent(new FocusEvent('focusout', {bubbles: true}));
  });
}

const SEATS = [
  {controller: 'human', config: {}, autoplay: false},
  {controller: 'random', config: {depth: 2}, autoplay: true},
];

describe('GameOptions controls', () => {
  it('heads the section and says that a change starts a new game', () => {
    const {container} = show();
    expect(only(container, 'h3').textContent).toBe('Game Options');
    expect(only(container, '.setup-card__description').textContent).toBe(
      'OpenSpiel game parameters. Changing one starts a new game.',
    );
    expect(container.querySelector('button')).toBeNull();
  });

  it('offers each parameter, named for people, through a control of its type', () => {
    const {container} = show();
    const shape = (name: string): string => {
      const element = control(container, name);
      return element instanceof HTMLInputElement
        ? `${element.type} ${String(element.type === 'checkbox' ? element.checked : element.value)}`
        : `${element.tagName.toLowerCase()} ${(element as HTMLSelectElement).value}`;
    };
    expect(
      ['Board Size', 'Komi', 'Rules', 'Suicide', 'Title'].map(shape),
    ).toEqual([
      'number 9',
      'number 7.5',
      'select area',
      'checkbox false',
      'text Board',
    ]);
    const size = control(container, 'Board Size') as HTMLInputElement;
    expect([size.min, size.max, size.step]).toEqual(['5', '19', '1']);
    expect((control(container, 'Komi') as HTMLInputElement).step).toBe('any');
    const rules = control(container, 'Rules') as HTMLSelectElement;
    expect(Array.from(rules.options, option => option.value)).toEqual([
      'area',
      'territory',
    ]);
  });

  it('names a board size choice by its sides', () => {
    const size: ParameterSpec = {
      ...RULES,
      name: 'board_size',
      type: 'int',
      default: 9,
      choices: [9, 13, 19],
    };
    const {container} = show({
      entry: {...ENTRY, parameters: [size]},
      frame: boardFrame({parameters: {board_size: 9}}),
    });
    const select = control(container, 'Board Size') as HTMLSelectElement;
    expect(Array.from(select.options, option => option.text)).toEqual([
      '9×9',
      '13×13',
      '19×19',
    ]);
  });

  it('keeps the lines of text that spans them', () => {
    const layout = {...TITLE, name: 'layout', default: 'a..\n..b'};
    const {container} = show({
      entry: {...ENTRY, parameters: [layout]},
      frame: boardFrame({parameters: {layout: 'a..\n..b'}}),
    });
    const area = control(container, 'Layout');
    expect(area).toBeInstanceOf(HTMLTextAreaElement);
    expect((area as HTMLTextAreaElement).value).toBe('a..\n..b');
  });
});

describe('GameOptions descriptions', () => {
  it('refuses a choice without options', () => {
    const rules = {...RULES, choices: null};
    expect(() => show({entry: {...ENTRY, parameters: [rules]}})).toThrow(
      'Parameter rules is a choice without options.',
    );
  });

  it('describes a parameter under its control', () => {
    const {container} = show();
    const describedBy = control(container, 'Board Size').getAttribute(
      'aria-describedby',
    );
    expect(document.getElementById(describedBy ?? '')?.textContent).toBe(
      'The width of the board.',
    );
    expect(control(container, 'Komi').hasAttribute('aria-describedby')).toBe(
      false,
    );
  });
});

describe('GameOptions starting a new game', () => {
  it('starts a new game once a value is set, keeping the seats', () => {
    const {container, onCreate} = show();
    const size = control(container, 'Board Size');
    enter(size, '13');
    commit(size);
    expect(onCreate).toHaveBeenCalledExactlyOnceWith({
      game: 'board',
      parameters: {board_size: 13},
      seats: SEATS,
      chance_policy: 'expose',
      seed: null,
    });
    enter(control(container, 'Rules'), 'territory');
    act(() => {
      control(container, 'Suicide').click();
    });
    expect(onCreate).toHaveBeenLastCalledWith(
      expect.objectContaining({
        parameters: {board_size: 13, rules: 'territory', suicide: true},
      }),
    );
  });

  it('starts nothing for a value the session already has', () => {
    const {container, onCreate} = show();
    const size = control(container, 'Board Size');
    enter(size, '9');
    commit(size);
    expect(onCreate).not.toHaveBeenCalled();
  });

  it('applies a typed number once it is committed, not at every keystroke', () => {
    const {container, onCreate} = show();
    const komi = control(container, 'Komi');
    enter(komi, '6');
    enter(komi, '6.');
    enter(komi, '6.5');
    expect(onCreate).not.toHaveBeenCalled();
    expect(komi.isConnected).toBe(true);
    act(() => {
      komi.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
      );
    });
    expect(onCreate).toHaveBeenCalledOnce();
    expect(onCreate.mock.calls[0]?.[0].parameters).toEqual({komi: 6.5});
  });

  it('applies text once it is committed, not at every keystroke', () => {
    const {container, onCreate} = show();
    const title = control(container, 'Title');
    enter(title, 'Bo');
    enter(title, 'Box');
    expect(onCreate).not.toHaveBeenCalled();
    act(() => {
      title.dispatchEvent(new FocusEvent('focusout', {bubbles: true}));
    });
    expect(onCreate.mock.calls[0]?.[0].parameters).toEqual({title: 'Box'});
  });

  it('says what is wrong with a value and sends nothing', () => {
    const {container, onCreate} = show();
    const size = control(container, 'Board Size');
    enter(size, '21');
    expect(size.getAttribute('aria-invalid')).toBe('true');
    expect(only(container, '[role="alert"]').textContent).toBe(
      'Invalid game options: board_size: Enter a number from 5 to 19.',
    );
    commit(size);
    expect(onCreate).not.toHaveBeenCalled();

    enter(size, '19');
    expect(size.getAttribute('aria-invalid')).toBe('false');
    expect(container.querySelector('[role="alert"]')).toBeNull();
    commit(size);
    expect(onCreate).toHaveBeenCalledOnce();
  });
});

describe('GameOptions refusals', () => {
  it('explains a refused change under the control that was changed', () => {
    let answer: (busy: boolean, error: ContractError | null) => void = () =>
      undefined;
    const onCreate = vi.fn<(body: CreateSession) => void>();
    function Shown(): JSX.Element {
      const [state, setState] = useState<{
        busy: boolean;
        error: ContractError | null;
      }>({busy: false, error: null});
      answer = (busy, error) => {
        setState({busy, error});
      };
      return (
        <GameOptions
          entry={ENTRY}
          frame={boardFrame()}
          busy={state.busy}
          error={state.error}
          onCreate={body => {
            onCreate(body);
            answer(true, null);
          }}
        />
      );
    }
    rendered = render(<Shown />);
    const {container} = rendered;
    enter(control(container, 'Board Size'), '13');
    commit(control(container, 'Board Size'));
    const refusal = new ContractError(
      {
        contract_version: 7,
        code: 'bad_request',
        message: 'The game refuses a board of 13.',
        seat: null,
        ply: null,
      },
      400,
    );
    act(() => {
      answer(false, refusal);
    });
    const size = control(container, 'Board Size');
    const notice = only(container, '[role="alert"]');
    expect(notice.textContent).toBe(
      'Not applied: The game refuses a board of 13.',
    );
    expect(size.getAttribute('aria-describedby')).toContain(notice.id);

    enter(size, '9');
    expect(container.querySelector('[role="alert"]')).toBeNull();
  });
});

describe('GameOptions sessions', () => {
  it('starts again from the values of a new session', () => {
    let showFrame: (frame: Frame) => void = () => undefined;
    function Shown(): JSX.Element {
      const [frame, setFrame] = useState(boardFrame());
      showFrame = setFrame;
      return (
        <GameOptions
          entry={ENTRY}
          frame={frame}
          busy={false}
          error={null}
          onCreate={() => undefined}
        />
      );
    }
    rendered = render(<Shown />);
    const {container} = rendered;
    enter(control(container, 'Komi'), '0.5');
    act(() => {
      showFrame(
        boardFrame({id: 'session-2', parameters: {...DEFAULTS, komi: 6}}),
      );
    });
    expect((control(container, 'Komi') as HTMLInputElement).value).toBe('6.0');
  });

  it("leaves out a fixed seat count and keeps the session's", () => {
    const numPlayers = {...PLAYERS, name: 'num_players'};
    const {container, onCreate} = show({
      entry: {
        ...ENTRY,
        min_seats: 3,
        max_seats: 3,
        num_players: 3,
        parameters: [BOARD_SIZE, numPlayers],
      },
      frame: boardFrame({parameters: {board_size: 9, num_players: 3}}),
    });
    expect(
      Array.from(
        container.querySelectorAll('label'),
        label => label.textContent,
      ),
    ).toEqual(['Board Size']);
    enter(control(container, 'Board Size'), '13');
    commit(control(container, 'Board Size'));
    expect(onCreate.mock.calls[0]?.[0].parameters).toEqual({
      board_size: 13,
      num_players: 3,
    });
    expect(onCreate.mock.calls[0]?.[0].seats).toHaveLength(3);
  });
});

describe('GameOptions players', () => {
  const kuhnPoker = {
    ...ENTRY,
    name: 'kuhn_poker',
    max_seats: 10,
    parameters: [PLAYERS],
  };
  const kuhnFrame = (players = 2): Frame => {
    const frame = boardFrame({parameters: {players}});
    return {...frame, session: {...frame.session, game: 'kuhn_poker'}};
  };

  it('offers a range of seats as Players, bounded to the range', () => {
    const {container} = show({entry: kuhnPoker, frame: kuhnFrame()});
    const players = control(container, 'Players') as HTMLInputElement;
    expect([players.type, players.value, players.min, players.max]).toEqual([
      'number',
      '2',
      '2',
      '10',
    ]);
  });

  it('starts a game with as many seats once Players is committed', () => {
    const {container, onCreate} = show({entry: kuhnPoker, frame: kuhnFrame()});
    const players = control(container, 'Players');
    enter(players, '3');
    expect(onCreate).not.toHaveBeenCalled();
    commit(players);
    expect(onCreate).toHaveBeenCalledExactlyOnceWith({
      game: 'kuhn_poker',
      parameters: {players: 3},
      seats: [...SEATS, {controller: 'human', config: {}, autoplay: false}],
      chance_policy: 'expose',
      seed: null,
    });
  });

  it('refuses a number of seats outside the range', () => {
    const {container, onCreate} = show({entry: kuhnPoker, frame: kuhnFrame()});
    const players = control(container, 'Players');
    enter(players, '11');
    commit(players);
    expect(onCreate).not.toHaveBeenCalled();
    expect(only(container, '[role="alert"]').textContent).toBe(
      'Invalid game options: players: Enter a number from 2 to 10.',
    );
  });

  it("shows the session's seats where the parameter means the game's own count", () => {
    const {container} = show({
      entry: {...kuhnPoker, parameters: [{...PLAYERS, default: 0}]},
      frame: kuhnFrame(0),
    });
    expect((control(container, 'Players') as HTMLInputElement).value).toBe('2');
  });
});

describe('GameOptions for a spectator', () => {
  it('shows the values, read only', () => {
    const {container} = show({
      frame: boardFrame({viewer: {kind: 'spectator', seat: null}}),
    });
    expect(container.querySelector('input, select, button')).toBeNull();
    expect(only(container, '.setup-card__description').textContent).toBe(
      'The parameters this game is played with.',
    );
    const rows = Array.from(container.querySelectorAll('dt'), term => [
      term.textContent,
      term.nextElementSibling?.textContent,
    ]);
    expect(rows).toEqual([
      ['Board Size', '9'],
      ['Komi', '7.5'],
      ['Players', '2'],
      ['Rules', 'area'],
      ['Suicide', 'false'],
      ['Title', 'Board'],
    ]);
  });

  it('lets a seat of a game that hides information change the options', () => {
    const {container} = show({
      frame: boardFrame({
        viewer: {kind: 'seat', seat: 1},
        hidesInformation: true,
      }),
    });
    expect(control(container, 'Komi')).toBeInstanceOf(HTMLInputElement);
  });
});

describe('GameOptions for a game without values to edit', () => {
  it('refuses a game with no options, whose section is left out', () => {
    expect(() => show({entry: {...ENTRY, parameters: []}})).toThrow(
      'board has no options to show.',
    );
  });

  it('refuses a game whose only parameter is a fixed seat count', () => {
    const fixed = {...ENTRY, max_seats: 2, parameters: [PLAYERS]};
    expect(() => show({entry: fixed})).toThrow('board has no options to show.');
  });

  it('refuses the entry of another game', () => {
    expect(() => show({entry: {...ENTRY, name: 'chess'}})).toThrow(
      'A session of board is not a game of chess.',
    );
  });

  it('refuses a session that lacks a parameter', () => {
    const frame = boardFrame({parameters: {board_size: 9}});
    expect(() => show({entry: {...ENTRY, parameters: [TITLE]}, frame})).toThrow(
      'The session has no value for parameter title.',
    );
  });
});
