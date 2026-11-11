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

// The Hearts table while a finished trick is on show: the finished trick in
// the trick area, the live trick's cards back in their players' hands.
import {describe, expect, it} from 'vitest';
import {
  heartsHeldView,
  heartsStagedView,
  type HeartsView,
} from './held_trick.ts';
import {HIDDEN_CARD, type CardCode, type PlayerPosition} from './types.ts';
import type {CompletedTrick} from '../../cards/library/index.ts';

const live: HeartsView = {
  players: [
    {name: 'You', position: 'bottom', hand: ['3c', 'Ah'] as CardCode[]},
    {
      name: 'W',
      position: 'left',
      hand: [HIDDEN_CARD, HIDDEN_CARD],
      hidden: true,
    },
    {
      name: 'N',
      position: 'top',
      hand: [HIDDEN_CARD, HIDDEN_CARD],
      hidden: true,
    },
    {name: 'E', position: 'right', hand: [HIDDEN_CARD], hidden: true},
  ],
  // East won the last trick and has led the next one.
  trickPile: [{card: '7d', position: 'right'}],
  isTerminal: false,
};
const held: CompletedTrick<CardCode, PlayerPosition> = {
  index: 3,
  cards: [
    {card: '2s', seat: 'top'},
    {card: 'Ks', seat: 'right'},
    {card: '4s', seat: 'bottom'},
    {card: '9s', seat: 'left'},
  ],
  winner: 'right',
};

describe('heartsHeldView', () => {
  it('is the live view when nothing is held', () => {
    expect(heartsHeldView(live, null)).toBe(live);
  });

  it('shows the finished trick and returns the live trick to the hands', () => {
    const v = heartsHeldView(live, held);
    expect(v.trickPile).toEqual([
      {card: '2s', position: 'top'},
      {card: 'Ks', position: 'right'},
      {card: '4s', position: 'bottom'},
      {card: '9s', position: 'left'},
    ]);
    const east = v.players.find(p => p.position === 'right');
    expect(east?.hand).toHaveLength(2);
    expect(v.players.find(p => p.position === 'top')?.hand).toHaveLength(2);
  });

  it('a face-up hand gets the card back in sorted order', () => {
    const open: HeartsView = {
      ...live,
      trickPile: [{card: '5c', position: 'bottom'}],
    };
    const v = heartsHeldView(open, held);
    expect(v.players.find(p => p.position === 'bottom')?.hand).toEqual([
      '3c',
      '5c',
      'Ah',
    ]);
  });

  it('keeps the cards of tricks queued behind it in the hands too', () => {
    const queued: CompletedTrick<CardCode, PlayerPosition> = {
      index: 4,
      cards: [
        {card: '7d', seat: 'right'},
        {card: '8d', seat: 'bottom'},
        {card: '9d', seat: 'left'},
        {card: 'Td', seat: 'top'},
      ],
      winner: 'top',
    };
    const v = heartsHeldView(
      {...live, trickPile: [{card: 'Jc', position: 'top'}]},
      held,
      [queued],
    );
    expect(v.trickPile.map(t => t.card)).toEqual(['2s', 'Ks', '4s', '9s']);
    expect(v.players.find(p => p.position === 'bottom')?.hand).toEqual([
      '3c',
      '8d',
      'Ah',
    ]);
    // North: its queued 10 and its live jack, as backs.
    expect(v.players.find(p => p.position === 'top')?.hand).toHaveLength(4);
    expect(v.players.find(p => p.position === 'right')?.hand).toHaveLength(2);
  });

  it('defers game over until the last trick has been shown', () => {
    expect(
      heartsHeldView({...live, trickPile: [], isTerminal: true}, held)
        .isTerminal,
    ).toBe(false);
  });

  it('the last trick on show: its players hold nothing, not the original deal the engine serves at game over', () => {
    const over: HeartsView = {
      ...live,
      trickPile: [],
      isTerminal: true,
      players: live.players.map(p => ({
        ...p,
        hand: p.hidden
          ? Array.from({length: 13}, () => HIDDEN_CARD)
          : (['3c', '5c', 'Ah'] as CardCode[]),
      })),
    };
    const v = heartsHeldView(over, held);
    expect(v.players.every(p => p.hand.length === 0)).toBe(true);
    expect(v.trickPile).toHaveLength(4);
  });
});

describe('heartsStagedView', () => {
  const view: HeartsView = {
    players: [
      {name: 'You', position: 'bottom', hand: ['As'] as CardCode[]},
      {name: 'W', position: 'left', hand: [HIDDEN_CARD], hidden: true},
      {name: 'N', position: 'top', hand: ['3h'] as CardCode[]},
    ] as HeartsView['players'],
    trickPile: [
      {card: '2c', position: 'left'},
      {card: '5c', position: 'top'},
    ],
    isTerminal: false,
  };

  it('keeps the later plays in their hands until their turn to be drawn', () => {
    const one = heartsStagedView(view, 1);
    expect(one.trickPile).toEqual([{card: '2c', position: 'left'}]);
    expect(one.players.find(p => p.position === 'top')?.hand).toContain('5c');
    expect(one.players.find(p => p.position === 'left')?.hand).toEqual([
      HIDDEN_CARD,
    ]);
    const none = heartsStagedView(view, 0);
    expect(none.players.find(p => p.position === 'left')?.hand).toEqual([
      HIDDEN_CARD,
      HIDDEN_CARD,
    ]);
    expect(heartsStagedView(view, 2)).toBe(view);
    expect(() => heartsStagedView(view, -1)).toThrow(/integer >= 0/);
  });
});
