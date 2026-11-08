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

import type {CatalogueEntry, ParameterSpec} from '../contract/contract.ts';
import {
  type Draft,
  type ParameterValue,
  draftOf,
  formatValue,
  isCheckbox,
  isMultiline,
  newGame,
  parameterLabel,
  parseDraft,
  parseDrafts,
  seatCountParameter,
  seatCountSpec,
  sessionDrafts,
  sessionParameters,
  setupParameters,
} from './parameters.ts';
import {
  BOARD_SIZE,
  DEFAULTS,
  ENTRY,
  KOMI,
  PLAYERS,
  RULES,
  SUICIDE,
  TITLE,
  boardFrame,
} from './testing/fixtures.ts';

const BOOL_CHOICE: ParameterSpec = {
  ...SUICIDE,
  kind: 'choice',
  choices: [false, true],
};
const PLAYERS_RANGE: ParameterSpec = {
  ...PLAYERS,
  kind: 'range',
  minimum: 2,
  maximum: 4,
};
// A game shaped like tiny_hanabi, whose player count is num_players.
const NUM_PLAYERS: ParameterSpec = {...PLAYERS, name: 'num_players'};
const HANABI = {...ENTRY, parameters: [KOMI, NUM_PLAYERS]};
// Kuhn poker and tic-tac-toe as the catalogue lists them.
const KUHN_POKER: CatalogueEntry = {
  ...ENTRY,
  name: 'kuhn_poker',
  min_seats: 2,
  max_seats: 10,
  num_players: 2,
  parameters: [PLAYERS],
};
const TIC_TAC_TOE: CatalogueEntry = {
  ...ENTRY,
  name: 'tic_tac_toe',
  min_seats: 2,
  max_seats: 2,
  num_players: 2,
  parameters: [],
};

describe('seatCountParameter', () => {
  it.each(['players', 'num_players', 'numPlayers'])('finds %s', name => {
    const entry = {...ENTRY, parameters: [KOMI, {...PLAYERS, name}]};
    expect(seatCountParameter(entry)).toBe(name);
  });

  it('finds none in a game whose player count is not a parameter', () => {
    expect(seatCountParameter({...ENTRY, parameters: [KOMI]})).toBeNull();
  });

  it('picks the one whose default is a seat count', () => {
    const sentinel = {...PLAYERS, default: 0};
    const entry = {...ENTRY, parameters: [NUM_PLAYERS, sentinel]};
    expect(seatCountParameter(entry)).toBe('num_players');
  });

  it('refuses a game with two player counts', () => {
    expect(() =>
      seatCountParameter({...ENTRY, parameters: [PLAYERS, NUM_PLAYERS]}),
    ).toThrow('board has two player counts: players, num_players.');
  });
});

describe('seatCountSpec', () => {
  it('offers a free players parameter over the seat counts', () => {
    expect(seatCountSpec(ENTRY)).toEqual(PLAYERS_RANGE);
  });

  it('offers a free num_players parameter over the seat counts', () => {
    expect(seatCountSpec(HANABI)).toEqual({
      ...PLAYERS_RANGE,
      name: 'num_players',
    });
  });

  it("defaults to the game's seats where the parameter means its own count", () => {
    const sentinel = {
      ...ENTRY,
      num_players: 3,
      parameters: [{...PLAYERS, default: 0}],
    };
    expect(seatCountSpec(sentinel)).toEqual({...PLAYERS_RANGE, default: 3});
  });

  it('offers none for a game with a fixed number of seats', () => {
    expect(seatCountSpec({...ENTRY, max_seats: 2})).toBeNull();
  });

  it('offers none for a game the server cannot load', () => {
    expect(seatCountSpec({...ENTRY, num_players: null})).toBeNull();
  });

  it('offers none for a players parameter that is not a whole number', () => {
    const named = {...ENTRY, parameters: [{...TITLE, name: 'players'}]};
    expect(seatCountSpec(named)).toBeNull();
    const listed: ParameterSpec = {...PLAYERS, kind: 'choice', choices: [2, 4]};
    expect(seatCountSpec({...ENTRY, parameters: [listed]})).toBeNull();
  });

  it('offers none for a game without a seat-count parameter', () => {
    expect(seatCountSpec({...ENTRY, parameters: [KOMI]})).toBeNull();
  });
});

describe('setupParameters', () => {
  it('offers Kuhn poker its players, from 2 to 10', () => {
    const [players, ...others] = setupParameters(KUHN_POKER);
    expect(others).toEqual([]);
    expect(players).toMatchObject({
      name: 'players',
      kind: 'range',
      default: 2,
      minimum: 2,
      maximum: 10,
    });
  });

  it('offers tic-tac-toe no seat count', () => {
    expect(setupParameters(TIC_TAC_TOE)).toEqual([]);
  });

  it('names every seat-count parameter Players', () => {
    expect(
      ['players', 'num_players', 'numPlayers'].map(parameterLabel),
    ).toEqual(['Players', 'Players', 'Players']);
    expect(parameterLabel('board_size')).toBe('Board Size');
  });

  it('offers the seat count among the other parameters', () => {
    expect(setupParameters(ENTRY)).toEqual([
      BOARD_SIZE,
      KOMI,
      PLAYERS_RANGE,
      RULES,
      SUICIDE,
      TITLE,
    ]);
    expect(setupParameters(HANABI)).toEqual([
      KOMI,
      {...PLAYERS_RANGE, name: 'num_players'},
    ]);
  });

  it('leaves out a fixed seat count', () => {
    const fixed = {...ENTRY, max_seats: 2};
    expect(setupParameters(fixed).map(spec => spec.name)).toEqual([
      'board_size',
      'komi',
      'rules',
      'suicide',
      'title',
    ]);
    expect(setupParameters({...fixed, parameters: [PLAYERS]})).toEqual([]);
  });

  it('offers only the board size of Go', () => {
    const go = {...ENTRY, name: 'go'};
    expect(setupParameters(go)).toEqual([BOARD_SIZE]);
    expect(() => setupParameters({...go, parameters: [KOMI]})).toThrow(
      'go has no parameter board_size.',
    );
  });
});

describe('sessionParameters', () => {
  it("holds an offered seat count as the session's seats", () => {
    const sentinel = {...ENTRY, parameters: [{...PLAYERS, default: 0}]};
    const frame = boardFrame({parameters: {...DEFAULTS, players: 0}});
    expect(sessionParameters(sentinel, frame)).toEqual(DEFAULTS);
  });

  it('keeps a seat count it does not offer as the session has it', () => {
    const frame = boardFrame({parameters: {...DEFAULTS, players: 0}});
    expect(sessionParameters({...ENTRY, max_seats: 2}, frame)).toEqual({
      ...DEFAULTS,
      players: 0,
    });
  });
});

describe('drafts', () => {
  it('edits a free boolean with a checkbox and the rest as text', () => {
    expect(isCheckbox(SUICIDE)).toBe(true);
    const boolChoice: ParameterSpec = {
      ...SUICIDE,
      kind: 'choice',
      choices: [true],
      default: true,
    };
    expect(isCheckbox(boolChoice)).toBe(false);
    expect(draftOf(boolChoice, true)).toBe('true');
    expect(draftOf(SUICIDE, true)).toBe(true);
    expect(draftOf(BOARD_SIZE, 13)).toBe('13');
  });

  it('edits text that spans lines in a text area', () => {
    expect(isMultiline({...TITLE, default: 'a.\n.b'})).toBe(true);
    expect(isMultiline(TITLE)).toBe(false);
    expect(isMultiline(RULES)).toBe(false);
  });

  it('writes a real number with its point', () => {
    expect(formatValue(KOMI, 7)).toBe('7.0');
    expect(formatValue(KOMI, 6.5)).toBe('6.5');
    expect(formatValue(BOARD_SIZE, 7)).toBe('7');
  });

  it('reads back every draft it writes', () => {
    for (const spec of setupParameters(ENTRY)) {
      expect(parseDraft(spec, draftOf(spec, spec.default))).toEqual({
        ok: true,
        value: spec.default,
      });
    }
  });

  it('starts from the session values', () => {
    const specs = setupParameters(ENTRY);
    expect(sessionDrafts(specs, {...DEFAULTS, komi: 6, suicide: true})).toEqual(
      {
        board_size: '9',
        komi: '6.0',
        players: '2',
        rules: 'area',
        suicide: true,
        title: 'Board',
      },
    );
    expect(() => sessionDrafts(specs, {komi: 6})).toThrow(
      'The session has no value for parameter board_size.',
    );
    expect(() => sessionDrafts(specs, {...DEFAULTS, rules: 'stones'})).toThrow(
      'The session\'s rules, "stones", is not one of its choices.',
    );
  });
});

describe('parseDraft', () => {
  const readable: Array<{
    spec: ParameterSpec;
    draft: Draft;
    value: ParameterValue;
  }> = [
    {spec: BOARD_SIZE, draft: ' 13 ', value: 13},
    {spec: BOARD_SIZE, draft: '+5', value: 5},
    {spec: KOMI, draft: '-0.5', value: -0.5},
    {spec: KOMI, draft: '1e2', value: 100},
    {spec: KOMI, draft: '2.', value: 2},
    {spec: KOMI, draft: '.5', value: 0.5},
    {spec: RULES, draft: 'territory', value: 'territory'},
    {spec: TITLE, draft: ' a b ', value: ' a b '},
    {spec: TITLE, draft: '', value: ''},
    {spec: SUICIDE, draft: true, value: true},
    {spec: BOOL_CHOICE, draft: 'true', value: true},
  ];
  it.each(readable)('reads $spec.name from $draft', ({spec, draft, value}) => {
    expect(parseDraft(spec, draft)).toEqual({ok: true, value});
  });

  const unreadable: Array<{
    spec: ParameterSpec;
    draft: Draft;
    problem: string;
  }> = [
    {spec: BOARD_SIZE, draft: '9.5', problem: 'Enter a whole number.'},
    {spec: BOARD_SIZE, draft: '', problem: 'Enter a whole number.'},
    {spec: BOARD_SIZE, draft: '1e1', problem: 'Enter a whole number.'},
    {
      spec: BOARD_SIZE,
      draft: '99999999999999999999',
      problem: 'Enter a whole number.',
    },
    {spec: BOARD_SIZE, draft: '4', problem: 'Enter a number from 5 to 19.'},
    {spec: BOARD_SIZE, draft: '20', problem: 'Enter a number from 5 to 19.'},
    {spec: KOMI, draft: '', problem: 'Enter a number.'},
    {spec: KOMI, draft: '1e400', problem: 'Enter a number.'},
    {spec: KOMI, draft: 'seven', problem: 'Enter a number.'},
    {spec: KOMI, draft: '0x10', problem: 'Enter a number.'},
    {spec: KOMI, draft: '0b11', problem: 'Enter a number.'},
    {spec: KOMI, draft: 'Infinity', problem: 'Enter a number.'},
    {spec: KOMI, draft: '.', problem: 'Enter a number.'},
    {spec: RULES, draft: 'stone', problem: 'Choose one of the listed values.'},
    {spec: BOOL_CHOICE, draft: 'yes', problem: 'Enter true or false.'},
  ];
  it.each(unreadable)(
    'refuses $spec.name from $draft',
    ({spec, draft, problem}) => {
      expect(parseDraft(spec, draft)).toEqual({ok: false, problem});
    },
  );

  it('refuses a range without bounds', () => {
    expect(() => parseDraft({...BOARD_SIZE, maximum: null}, '9')).toThrow(
      'Parameter board_size is a range without bounds.',
    );
  });
});

describe('parseDrafts', () => {
  const specs = [BOARD_SIZE, KOMI];

  it('reads every draft', () => {
    expect(parseDrafts(specs, {board_size: '13', komi: '6.5'})).toEqual({
      values: {board_size: 13, komi: 6.5},
      problems: new Map(),
    });
  });

  it('gives no values while a draft has a problem', () => {
    expect(parseDrafts(specs, {board_size: '3', komi: 'x'})).toEqual({
      values: null,
      problems: new Map([
        ['board_size', 'Enter a number from 5 to 19.'],
        ['komi', 'Enter a number.'],
      ]),
    });
  });

  it('refuses drafts that miss a parameter', () => {
    expect(() => parseDrafts(specs, {komi: '6.5'})).toThrow(
      'No draft holds parameter board_size.',
    );
  });
});

describe('newGame', () => {
  const frame = boardFrame();
  const seats = [
    {controller: 'human', config: {}, autoplay: false},
    {controller: 'random', config: {depth: 2}, autoplay: true},
  ];

  it('sends the values other than the default and keeps the seats', () => {
    expect(
      newGame(ENTRY, frame, {...DEFAULTS, board_size: 13, komi: 7.5}),
    ).toEqual({
      game: 'board',
      parameters: {board_size: 13},
      seats,
      chance_policy: 'expose',
      seed: null,
    });
  });

  it('adds human seats for more players and drops seats for fewer', () => {
    const three = newGame(ENTRY, frame, {...DEFAULTS, players: 3});
    expect(three.parameters).toEqual({players: 3});
    expect(three.seats).toEqual([
      ...seats,
      {controller: 'human', config: {}, autoplay: false},
    ]);
    const one = newGame({...ENTRY, min_seats: 1}, frame, {
      ...DEFAULTS,
      players: 1,
    });
    expect(one.seats).toEqual(seats.slice(0, 1));
  });

  it('follows num_players in a game that names its player count so', () => {
    const three = newGame(HANABI, frame, {komi: 7.5, num_players: 3});
    expect(three.parameters).toEqual({num_players: 3});
    expect(three.seats).toEqual([
      ...seats,
      {controller: 'human', config: {}, autoplay: false},
    ]);
  });

  it("keeps the session's seat count when the values name none", () => {
    const three = boardFrame({parameters: {...DEFAULTS, players: 3}});
    const body = newGame(ENTRY, three, {board_size: 13});
    expect(body.parameters).toEqual({board_size: 13, players: 3});
    expect(body.seats).toEqual([
      ...seats,
      {controller: 'human', config: {}, autoplay: false},
    ]);
  });

  it('keeps the seats while players holds no seat count', () => {
    const sentinel = {...ENTRY, parameters: [{...PLAYERS, default: -1}]};
    const body = newGame(sentinel, frame, {players: -1});
    expect(body.parameters).toEqual({});
    expect(body.seats).toEqual(seats);
  });

  it('refuses a frame of another game', () => {
    expect(() => newGame({...ENTRY, name: 'chess'}, frame, DEFAULTS)).toThrow(
      'A session of board is not a game of chess.',
    );
  });
});
