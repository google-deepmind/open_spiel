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

import {describe, expect, it} from 'vitest';

import type {SeatView} from '../contract/contract.ts';
import type {SeatDraft} from './seat_draft.ts';
import {
  chooseController,
  controllerNamed,
  editField,
  editJson,
  isChanged,
  readSeatDraft,
  refusalOf,
  sameConfig,
  savedConfig,
  savedDraft,
} from './seat_draft.ts';
import {
  CHECKPOINT,
  CONTROLLERS,
  HUMAN_CONTROLLER,
  MCTS,
  RANDOM,
  SCRIPTED,
  seatView,
} from './testing/seats.ts';

const MCTS_SEAT = seatView(1, MCTS, {
  config: {max_simulations: 50, uct_c: 1.5, solve: false},
  autoplay: true,
});

function choose(draft: SeatDraft, name: string, seat = MCTS_SEAT): SeatDraft {
  return chooseController(draft, name, seat, CONTROLLERS);
}

function changed(draft: SeatDraft, seat: SeatView): boolean {
  return isChanged(readSeatDraft(draft), savedConfig(seat, CONTROLLERS));
}

describe('controllerNamed', () => {
  it('finds a controller by name', () => {
    expect(controllerNamed(CONTROLLERS, 'mcts')).toBe(MCTS);
  });

  it('refuses a name the session does not list', () => {
    expect(() => controllerNamed(CONTROLLERS, 'alpha_zero')).toThrow(
      'The session lists no controller alpha_zero.',
    );
  });
});

describe('savedConfig', () => {
  it('fills in the settings a seat was given without, in the form’s order', () => {
    const seat = seatView(1, MCTS, {config: {solve: false, uct_c: 1}});
    const {config} = savedConfig(seat, CONTROLLERS);
    expect(Object.entries(config)).toEqual([
      ['max_simulations', 1000],
      ['uct_c', 1],
      ['solve', false],
    ]);
  });

  it('keeps settings no form can show as they are', () => {
    const seat = seatView(0, SCRIPTED, {config: {moves: '1 2'}});
    expect(savedConfig(seat, CONTROLLERS).config).toEqual({moves: '1 2'});
  });
});

describe('required settings', () => {
  const draft = choose(savedDraft(seatView(1), CONTROLLERS), 'checkpoint');

  it('fills in no default for a required setting', () => {
    const seat = seatView(1, CHECKPOINT, {config: {checkpoint: 'c'}});
    expect(savedConfig(seat, CONTROLLERS).config).toEqual({
      deterministic: true,
      temperature: 1,
      checkpoint: 'c',
    });
  });

  it('refuses a draft missing a required setting, naming it', () => {
    const reading = readSeatDraft(draft);
    expect(reading).toEqual({
      ok: false,
      errors: {
        kind: 'form',
        errors: {checkpoint: 'This setting is required.'},
      },
    });
    if (reading.ok) throw new Error('The draft was read.');
    expect(refusalOf(draft, reading.errors)).toBe(
      'Not saved: Checkpoint is required.',
    );
  });

  it('says which settings are missing and which are not valid', () => {
    const bad = editField(draft, 'temperature', '-1');
    const reading = readSeatDraft(bad);
    if (reading.ok) throw new Error('The draft was read.');
    expect(refusalOf(bad, reading.errors)).toBe(
      'Not saved: Checkpoint is required; Temperature is not valid.',
    );
  });

  it('reads a draft with the required setting filled in', () => {
    expect(readSeatDraft(editField(draft, 'checkpoint', 'c'))).toEqual({
      ok: true,
      config: {
        controller: 'checkpoint',
        config: {checkpoint: 'c', deterministic: true, temperature: 1},
        autoplay: true,
      },
    });
  });
});

describe('savedDraft', () => {
  it('drafts the saved settings in a form', () => {
    expect(savedDraft(MCTS_SEAT, CONTROLLERS)).toEqual({
      controller: 'mcts',
      settings: {
        kind: 'form',
        fields: expect.any(Array) as unknown,
        drafts: {max_simulations: '50', uct_c: '1.5', solve: false},
      },
      autoplay: true,
    });
  });

  it('drafts settings no form can show as JSON', () => {
    const seat = seatView(0, SCRIPTED, {config: {moves: '1 2'}});
    expect(savedDraft(seat, CONTROLLERS).settings).toEqual({
      kind: 'json',
      text: '{\n  "moves": "1 2"\n}',
    });
  });
});

describe('chooseController', () => {
  const human = savedDraft(seatView(1), CONTROLLERS);

  it('starts another controller from its defaults', () => {
    const draft = choose(savedDraft(MCTS_SEAT, CONTROLLERS), 'random');
    expect(draft).toEqual({
      controller: 'random',
      settings: {kind: 'form', fields: [], drafts: {}},
      autoplay: true,
    });
    expect(readSeatDraft(choose(draft, 'mcts', seatView(1)))).toEqual({
      ok: true,
      config: {
        controller: 'mcts',
        config: {max_simulations: 1000, uct_c: 2, solve: true},
        autoplay: true,
      },
    });
  });

  it('brings back the saved settings with the saved controller', () => {
    const saved = savedDraft(MCTS_SEAT, CONTROLLERS);
    expect(choose(choose(saved, 'random'), 'mcts')).toEqual(saved);
  });

  it('autoplays a controller chosen in place of a human', () => {
    expect(choose(human, 'random', seatView(1)).autoplay).toBe(true);
  });

  it('keeps autoplay off from one controller to another', () => {
    const manual = {...choose(human, 'random', seatView(1)), autoplay: false};
    expect(choose(manual, 'mcts', seatView(1)).autoplay).toBe(false);
  });

  it('brings back the saved autoplay with the saved controller', () => {
    const saved = savedDraft(MCTS_SEAT, CONTROLLERS);
    const manual = {...choose(saved, 'random'), autoplay: false};
    expect(choose(manual, 'mcts').autoplay).toBe(true);
  });

  it('never autoplays a human', () => {
    const draft = choose(savedDraft(MCTS_SEAT, CONTROLLERS), 'human');
    expect(draft.autoplay).toBe(false);
  });
});

describe('readSeatDraft', () => {
  it('reports every setting that cannot be read', () => {
    let draft = savedDraft(MCTS_SEAT, CONTROLLERS);
    draft = editField(draft, 'max_simulations', '0');
    draft = editField(draft, 'uct_c', 'high');
    expect(readSeatDraft(draft)).toEqual({
      ok: false,
      errors: {
        kind: 'form',
        errors: {
          max_simulations: 'A whole number from 1 to 10000.',
          uct_c: 'A number from 0 to 100.',
        },
      },
    });
  });

  it('never sends a human that autoplays', () => {
    const draft = {...savedDraft(seatView(0), CONTROLLERS), autoplay: true};
    const reading = readSeatDraft(draft);
    expect(reading.ok && reading.config.autoplay).toBe(false);
  });

  it.each([
    ['', {}],
    [
      '{"moves": "1 2", "fast": true, "depth": 2}',
      {moves: '1 2', fast: true, depth: 2},
    ],
  ])('reads JSON settings %j', (text, config) => {
    const draft = editJson(
      savedDraft(seatView(0, SCRIPTED), CONTROLLERS),
      text,
    );
    expect(readSeatDraft(draft)).toEqual({
      ok: true,
      config: {controller: 'scripted', config, autoplay: false},
    });
  });

  it.each([
    ['{', /^The settings are not JSON: /],
    [
      '[1]',
      /^The settings are a JSON object of booleans, finite numbers and strings\.$/,
    ],
    ['{"moves": [1]}', /JSON object of booleans/],
    ['{"moves": null}', /JSON object of booleans/],
    ['3', /JSON object of booleans/],
    ['{"moves": 1e400}', /JSON object of booleans/],
  ])('refuses JSON settings %j', (text, message) => {
    const draft = editJson(
      savedDraft(seatView(0, SCRIPTED), CONTROLLERS),
      text,
    );
    const reading = readSeatDraft(draft);
    expect(reading.ok).toBe(false);
    expect(
      !reading.ok && reading.errors.kind === 'json' && reading.errors.error,
    ).toMatch(message);
  });
});

describe('editing', () => {
  it('refuses to edit settings through the wrong editor', () => {
    expect(() => editJson(savedDraft(MCTS_SEAT, CONTROLLERS), '{}')).toThrow();
    expect(() =>
      editField(savedDraft(seatView(0, SCRIPTED), CONTROLLERS), 'moves', ''),
    ).toThrow();
  });
});

describe('isChanged', () => {
  it('is false for the saved settings', () => {
    expect(changed(savedDraft(MCTS_SEAT, CONTROLLERS), MCTS_SEAT)).toBe(false);
  });

  it('is false for a seat saved without its settings, which take defaults', () => {
    const seat = seatView(1, MCTS);
    expect(changed(savedDraft(seat, CONTROLLERS), seat)).toBe(false);
    const draft = editField(savedDraft(seat, CONTROLLERS), 'uct_c', '3');
    expect(changed(draft, seat)).toBe(true);
  });

  it('is false again once an edit is undone', () => {
    let draft = savedDraft(MCTS_SEAT, CONTROLLERS);
    draft = editField(draft, 'uct_c', '3');
    expect(changed(draft, MCTS_SEAT)).toBe(true);
    draft = editField(draft, 'uct_c', '1.50');
    expect(changed(draft, MCTS_SEAT)).toBe(false);
  });

  it('is true for another controller, autoplay or unreadable settings', () => {
    const saved = savedDraft(MCTS_SEAT, CONTROLLERS);
    expect(changed(choose(saved, 'random'), MCTS_SEAT)).toBe(true);
    expect(changed({...saved, autoplay: false}, MCTS_SEAT)).toBe(true);
    expect(changed(editField(saved, 'uct_c', ''), MCTS_SEAT)).toBe(true);
  });

  it('compares JSON settings whatever the order of their keys', () => {
    const seat = seatView(0, SCRIPTED, {config: {a: 1, b: 'x'}});
    const draft = editJson(savedDraft(seat, CONTROLLERS), '{"b": "x", "a": 1}');
    expect(changed(draft, seat)).toBe(false);
    expect(changed(editJson(draft, '{"b": "x"}'), seat)).toBe(true);
    expect(changed(editJson(draft, '{"b": "x", "c": 1}'), seat)).toBe(true);
  });
});

describe('sameConfig', () => {
  it('compares settings by value, whatever their order', () => {
    const first = {controller: 'mcts', config: {a: 1, b: true}, autoplay: true};
    expect(sameConfig(first, {...first, config: {b: true, a: 1}})).toBe(true);
    expect(sameConfig(first, {...first, config: {a: 1, b: false}})).toBe(false);
    expect(sameConfig(first, {...first, autoplay: false})).toBe(false);
    expect(sameConfig(first, {...first, controller: RANDOM.name})).toBe(false);
    expect(
      sameConfig(first, {...first, controller: HUMAN_CONTROLLER.name}),
    ).toBe(false);
  });
});
