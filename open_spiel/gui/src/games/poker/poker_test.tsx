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

import {historyRendererProps} from '../../actions/history_renderer.ts';
import type {Frame, HistoryEntry} from '../../contract/contract.ts';
import {type Game, GameContext} from '../../game/game_context.ts';
import {frame, session} from '../../testing/frames.ts';
import {type Rendered, button, render} from '../../testing/render.tsx';
import {describeHistoryActions, parseBettingHistory} from './poker_betting.ts';
import {heldTable} from './poker_gameplay_panel.tsx';
import {POKER_RENDERER, newHandSession} from './poker_renderer.tsx';
import {pokerSession} from './poker_session.ts';
import {type PokerUiState, derivePokerState} from './poker_state.ts';
import type {PokerMotion} from './use_poker_motion.ts';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

const KUHN = {title: 'Kuhn Poker', betting: 'limit', holeCards: 1};
const LEDUC = {title: "Leduc Hold'em", betting: 'limit', holeCards: 1};

// A chance entry: a card dealt, its label withheld.
function deal(): HistoryEntry {
  return {
    seat: null,
    action: null,
    label: null,
    rewards: null,
    public_label: null,
  };
}

function move(seat: number, label: string): HistoryEntry {
  return {seat, action: 0, label, rewards: null, public_label: null};
}

interface Table {
  readonly game: Record<string, unknown>;
  readonly boardSizes: number[];
  readonly struct: Record<string, unknown>;
  readonly labels: string[];
  readonly history: HistoryEntry[];
  readonly legal: Array<[number, string]>;
  readonly terminal?: boolean;
  readonly returns?: number[];
}

// A frame of `table` as seat 0 sees it.
function pokerFrame(table: Table): Frame {
  const terminal = table.terminal ?? false;
  return frame({
    session: {...frame().session, game: 'kuhn_poker'},
    viewer: {kind: 'seat', seat: 0},
    omniscient: false,
    status: terminal ? 'terminal' : 'playing',
    to_move: terminal
      ? {kind: 'terminal', seat: null}
      : {kind: 'seat', seat: 0},
    history: table.history,
    legal_actions: terminal
      ? []
      : table.legal.map(([id]) => ({id, label: `game ${String(id)}`})),
    returns: table.returns ?? null,
    presentation: {
      renderer: 'poker',
      view: {
        struct: table.struct,
        historyActionStrings: table.labels,
        pokerGame: {...table.game, boardSizes: table.boardSizes},
      },
      actions: terminal
        ? []
        : table.legal.map(([id, label]) => ({id, hint: {label}})),
    },
  });
}

function struct(fields: Record<string, unknown>): Record<string, unknown> {
  return {
    board_cards: '',
    player_hands: ['Ks', ''],
    player_contributions: [1, 1],
    starting_stacks: [],
    pot_size: 2,
    current_player: 0,
    betting_history: '',
    blinds: [],
    ante: 1,
    best_hand_rank_types: ['High Card', ''],
    best_five_card_hands: ['Ks', ''],
    ...fields,
  };
}

// Kuhn, seat 0 to open.
const KUHN_OPEN: Table = {
  game: KUHN,
  boardSizes: [0],
  struct: struct({}),
  labels: ['Chance outcome (hidden)', 'Chance outcome (hidden)'],
  history: [deal(), deal()],
  legal: [
    [0, 'player=0 move=Call'],
    [1, 'player=0 move=Bet2'],
  ],
};

// Leduc, seat 0 facing seat 1's raise after its own bet.
const LEDUC_RAISED: Table = {
  game: LEDUC,
  boardSizes: [0, 1],
  struct: struct({
    player_hands: ['Qh', ''],
    player_contributions: [3, 5],
    pot_size: 8,
    betting_history: 'r3r5',
  }),
  labels: [
    'Chance outcome (hidden)',
    'Chance outcome (hidden)',
    'player=0 move=Bet3',
    'player=1 move=Bet5',
  ],
  history: [deal(), deal(), move(0, 'Raise'), move(1, 'Raise')],
  legal: [
    [0, 'player=0 move=Fold'],
    [1, 'player=0 move=Call'],
  ],
};

function show(
  table: Table,
  play: ((action: number) => void) | null = null,
  game: Game | null = null,
): HTMLElement {
  const shown = pokerFrame(table);
  const presentation = shown.presentation;
  if (presentation === null) throw new Error('No presentation.');
  const Board = POKER_RENDERER.render;
  rendered = render(
    <GameContext value={game}>
      <Board frame={shown} presentation={presentation} play={play} />
    </GameContext>,
  );
  return rendered.container;
}

function texts(container: HTMLElement, selector: string): string[] {
  return [...container.querySelectorAll(selector)].map(
    element => element.textContent,
  );
}

describe('the poker board, kuhn_poker', () => {
  it('shows the hand bar, the ante and no stacks', () => {
    const container = show(KUHN_OPEN);
    // Kuhn has one betting round and no board: no Hold'em street.
    expect(texts(container, '.poker-handbar__title > *')).toEqual([
      'Kuhn Poker',
      'Betting',
    ]);
    expect(texts(container, '.poker-handbar__meta span')).toEqual(['Ante 1']);
    expect(texts(container, '.poker-seat__metrics')).toEqual(['In 1', 'In 1']);
  });

  it("shows the seat its own card and the other's as an empty slot", () => {
    const container = show(KUHN_OPEN);
    const plates = container.querySelectorAll('.poker-seat__cards');
    expect(
      plates[0]?.querySelector('.poker-card')?.getAttribute('aria-label'),
    ).toBe('King of Spades');
    expect(plates[1]?.querySelectorAll('.poker-card--empty')).toHaveLength(1);
    expect(
      container.querySelectorAll('.poker-board__cards .poker-card'),
    ).toHaveLength(0);
  });

  it("names the moves in poker's words and plays the one pressed", () => {
    const play = vi.fn();
    const container = show(KUHN_OPEN, play);
    expect(texts(container, '.poker-action')).toEqual(['Check', 'Bet 1']);
    // A fixed-limit game has one bet size, so no sizing box.
    expect(container.querySelector('.poker-betbox')).toBeNull();
    act(() => {
      button('Bet 1', container).click();
    });
    expect(play).toHaveBeenCalledWith(1);
  });

  it('offers no press when the viewer cannot play', () => {
    const container = show(KUHN_OPEN, null);
    expect(button('Check', container).disabled).toBe(true);
  });

  it('says how a hand won by fold ended, showing no card', () => {
    const container = show({
      ...KUHN_OPEN,
      struct: struct({
        player_contributions: [2, 1],
        pot_size: 3,
        current_player: -4,
        betting_history: 'r2f',
      }),
      labels: [...KUHN_OPEN.labels, 'player=0 move=Bet2', 'player=1 move=Fold'],
      history: [deal(), deal(), move(0, 'Bet'), move(1, 'Pass')],
      terminal: true,
      returns: [1, -1],
    });
    // Seat 0 is the viewer's own.
    expect(texts(container, '.poker-terminal strong')).toEqual([
      'You win +1 by fold',
    ]);
    expect(texts(container, '.poker-handbar__street')).toEqual(['Won by fold']);
    expect(texts(container, '.poker-history li')).toEqual([
      'DealChance outcome (hidden)',
      'DealChance outcome (hidden)',
      'player 0Bet 1',
      'player 1Fold',
    ]);
  });
});

describe('the poker board, its viewer and its History', () => {
  it("marks the viewer's own seat and no other", () => {
    const container = show(KUHN_OPEN);
    expect(texts(container, '.poker-seat__you')).toEqual(['You']);
    const plates = [...container.querySelectorAll('.poker-seat')];
    expect(plates.map(plate => plate.getAttribute('aria-label'))).toEqual([
      'player 0 (you), to act',
      'player 1',
    ]);
  });

  it('lists Actions › History in the words of the table', () => {
    const terminal = pokerFrame({
      ...KUHN_OPEN,
      struct: struct({
        player_contributions: [2, 2],
        pot_size: 4,
        current_player: -4,
        betting_history: 'r2c',
      }),
      labels: [...KUHN_OPEN.labels, 'player=0 move=Bet2', 'player=1 move=Call'],
      // The engine names Kuhn's call "Bet" and its fold "Pass".
      history: [deal(), deal(), move(0, 'Bet'), move(1, 'Bet')],
      terminal: true,
      returns: [2, -2],
    });
    const List = POKER_RENDERER.History;
    if (List === undefined) throw new Error('No poker History list.');
    rendered = render(
      <List {...historyRendererProps(terminal, false, vi.fn())} />,
    );
    expect(texts(rendered.container, '.history__label')).toEqual([
      'Deal Chance outcome (hidden)',
      'Deal Chance outcome (hidden)',
      'player 0 Bet 1',
      'player 1 Call 1',
    ]);
  });
});

describe('the poker board, leduc_poker', () => {
  it('names a raise to its street total and the call it owes', () => {
    const container = show(LEDUC_RAISED, vi.fn());
    expect(texts(container, '.poker-action')).toEqual(['Fold', 'Call 2']);
    expect(texts(container, '.poker-history li').slice(2)).toEqual([
      'player 0Bet 2',
      'player 1Raise to 4',
    ]);
    expect(container.querySelectorAll('.poker-board__cards > *')).toHaveLength(
      1,
    );
  });

  it('turns the public card on the flop, its deal named', () => {
    const container = show({
      ...LEDUC_RAISED,
      struct: struct({
        board_cards: 'Qs',
        player_hands: ['Qh', ''],
        player_contributions: [5, 5],
        pot_size: 10,
        betting_history: 'r3r5c/',
        best_hand_rank_types: ['Pair', ''],
        best_five_card_hands: ['QhQs', ''],
      }),
      labels: [
        ...LEDUC_RAISED.labels,
        'player=0 move=Call',
        'player=-1 move=Deal Qs',
      ],
      history: [...LEDUC_RAISED.history, move(0, 'Call'), deal()],
      legal: [
        [1, 'player=0 move=Call'],
        [2, 'player=0 move=Bet9'],
      ],
    });
    // One public card starts Leduc's second round, not a flop.
    expect(texts(container, '.poker-handbar__street')).toEqual(['Round 2']);
    expect(texts(container, '.poker-action')).toEqual(['Check', 'Bet 4']);
    expect(texts(container, '.poker-seat__best')).toEqual(['Pair']);
    expect(texts(container, '.poker-history li').at(-1)).toBe('DealQs');
  });
});

describe('the outcome waits for the chips', () => {
  const finished = (): PokerUiState => {
    const shown = pokerFrame({
      ...KUHN_OPEN,
      struct: struct({
        player_contributions: [2, 1],
        pot_size: 3,
        current_player: -4,
        betting_history: 'r2f',
      }),
      labels: [...KUHN_OPEN.labels, 'player=0 move=Bet2', 'player=1 move=Fold'],
      history: [deal(), deal(), move(0, 'Bet'), move(1, 'Pass')],
      terminal: true,
      returns: [1, -1],
    });
    if (shown.presentation === null) throw new Error('No presentation.');
    const state = derivePokerState(pokerSession(shown, shown.presentation));
    if (state === null) throw new Error('No table.');
    return state;
  };
  const flying: PokerMotion = {
    flightBaseMs: 240,
    scale: 1,
    boardDelays: new Map(),
    revealDelays: new Map(),
    chips: [
      {key: 'c', slot: 0, amount: 1, delayBaseMs: 0, x: 0, y: 0, dx: 1, dy: 1},
    ],
    heldFolds: new Set([1]),
  };

  it('holds back the result, the winner and the fold while a chip flies', () => {
    const held = heldTable(finished(), flying, [0]);
    expect(held.ending).toBeNull();
    expect(held.currentStreet).toBe('Betting');
    expect(held.seats.map(seat => seat.isWinner)).toEqual([false, false]);
    expect(held.seats.map(seat => seat.hasFolded)).toEqual([false, false]);
  });

  it('shows them once everything has landed', () => {
    const state = finished();
    const settled = {...flying, chips: [], heldFolds: new Set<number>()};
    expect(heldTable(state, settled, [0])).toBe(state);
    expect(state.currentStreet).toBe('Won by fold');
  });
});

describe('a new hand', () => {
  function game(): Game {
    const shown = pokerFrame(KUHN_OPEN);
    return {session: session(shown)} as unknown as Game;
  }

  it('deals the same table again', () => {
    const table = game();
    const container = show(
      {...KUHN_OPEN, terminal: true, returns: [1, -1]},
      null,
      table,
    );
    act(() => {
      button(
        'New hand',
        container.querySelector('.poker-terminal') ?? container,
      ).click();
    });
    expect(table.session.commands.reset).toHaveBeenCalledOnce();
  });

  it('seats a third player as the second plays', () => {
    const shown = pokerFrame(KUHN_OPEN);
    const random = {controller: 'random', config: {}, autoplay: false};
    const created = newHandSession(
      {
        ...shown,
        seats: [shown.seats[0], {...random, seat: 1, failure: null}],
      } as Frame,
      {seats: 3},
    );
    expect(created.parameters['players']).toBe(3);
    expect(created.seats).toEqual([
      {controller: 'human', config: {}, autoplay: false},
      random,
      random,
    ]);
  });
});

describe('the poker session and betting', () => {
  it('reads the adapter hints as the moves and fails on a short history', () => {
    const shown = pokerFrame(KUHN_OPEN);
    const presentation = shown.presentation;
    if (presentation === null) throw new Error('No presentation.');
    expect(pokerSession(shown, presentation).state.legalActions).toEqual([
      {action: 0, label: 'player=0 move=Call'},
      {action: 1, label: 'player=0 move=Bet2'},
    ]);
    const short = {
      ...presentation,
      view: {...(presentation.view as object), historyActionStrings: []},
    };
    expect(() => pokerSession(shown, short)).toThrow(
      /labels 0 history entries/,
    );
  });

  it("describes hold'em betting in street chips", () => {
    const rows = [
      {label: 'player=1 move=Bet300', player: 1},
      {label: 'player=0 move=Call', player: 0},
      {label: 'player=0 move=Call', player: 0},
      {label: 'player=1 move=Bet700', player: 1},
    ];
    const words = describeHistoryActions(
      rows,
      parseBettingHistory('r300c/cr700'),
      {
        blinds: [100, 50],
        startingStacks: [20000, 20000],
        ante: 0,
      },
    );
    expect([...words.values()]).toEqual([
      'Raise to 300',
      'Call 200',
      'Check',
      'Bet 400',
    ]);
  });

  it('marks a showdown winner and its cards', () => {
    const shown = pokerFrame({
      ...KUHN_OPEN,
      struct: struct({
        player_hands: ['Ks', 'Qs'],
        current_player: -4,
        betting_history: 'cc',
        best_hand_rank_types: ['High Card', 'High Card'],
        best_five_card_hands: ['Ks', 'Qs'],
      }),
      labels: [...KUHN_OPEN.labels, 'player=0 move=Call', 'player=1 move=Call'],
      history: [deal(), deal(), move(0, 'Pass'), move(1, 'Pass')],
      terminal: true,
      returns: [1, -1],
    });
    const presentation = shown.presentation;
    if (presentation === null) throw new Error('No presentation.');
    const state = derivePokerState(pokerSession(shown, presentation));
    expect(state?.ending).toBe('showdown');
    expect(state?.seats.map(seat => seat.isWinner)).toEqual([true, false]);
    expect(state?.history.slice(2).map(entry => entry.action)).toEqual([
      'Check',
      'Check',
    ]);
  });
});
