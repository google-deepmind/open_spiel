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
import {createPortal} from 'react-dom';

import {Drawer} from '../../../controls/drawer.tsx';
import {
  GIN_HAND_KIND_LABEL,
  ginCardText,
  ginPointsSum,
  type GinHandResult,
  type GinSeatNamer,
  type GinSeatResult,
} from '../gin_hand_result.ts';

/** A seat's name mid-sentence ('you', not 'You'). */
function inSentence(name: string): string {
  return name === 'You' ? 'you' : name;
}

/** Whose deadwood: 'your', "Player 1's". */
function possessive(name: string): string {
  return name === 'You' ? 'your' : `${name}'s`;
}

function cards(list: readonly string[]): string {
  return list.map(ginCardText).join(' ');
}

/** What happened, in one sentence: who knocked and how it went. */
function headline(result: GinHandResult, name: GinSeatNamer): string {
  const {knocker: knockerSeat, winner: winnerSeat} = result;
  if (result.kind === 'draw' || knockerSeat === null || winnerSeat === null) {
    return result.stockExhausted
      ? 'No one knocked before the stock reached its last two cards, so the hand is a draw: no score.'
      : 'The hand ended without a knock, so it is a draw: no score.';
  }
  const knocker = name(knockerSeat);
  const defender = name(1 - knockerSeat);
  const winner = name(winnerSeat);
  const how =
    result.kind === 'gin'
      ? `${knocker} went gin.`
      : result.kind === 'knock'
        ? `${knocker} knocked.`
        : `${knocker} knocked and ${inSentence(defender)} undercut.`;
  const pts = result.points === null ? '' : ` ${String(result.points)}`;
  const outcome =
    winner === 'You'
      ? `You win${pts}.`
      : name(1 - winnerSeat) === 'You'
        ? `You lose${pts}.`
        : `${winner} wins${pts}.`;
  return `${how} ${outcome}`;
}

function laidOff(result: GinHandResult): string {
  if (result.layoffs.length > 0) return cards(result.layoffs);
  return result.kind === 'gin' ? 'none (no lay-offs against gin)' : 'none';
}

function deadwoodText(s: GinSeatResult): string {
  const count = s.deadwood === null ? 'not shown' : String(s.deadwood);
  if (s.deadwoodCards === null) return count;
  return s.deadwoodCards.length > 0
    ? `${cards(s.deadwoodCards)} = ${count}`
    : `none = ${count}`;
}

function SeatSection({
  result,
  seat,
  name,
}: {
  result: GinHandResult;
  seat: number;
  name: GinSeatNamer;
}): JSX.Element | null {
  const s = result.seats[seat];
  if (s === undefined) return null;
  const role = seat === result.knocker ? 'knocked' : 'defended';
  return (
    <section
      className="gin-hand-result__seat"
      data-testid={`gin-hand-result-seat-${String(seat)}`}
    >
      <h3>
        {name(seat)} ({role})
      </h3>
      <dl>
        <dt>Melds</dt>
        <dd>
          {s.melds.length > 0 ? s.melds.map(m => cards(m)).join(' · ') : 'none'}
        </dd>
        {seat !== result.knocker && (
          <>
            <dt>Laid off</dt>
            <dd>{laidOff(result)}</dd>
          </>
        )}
        <dt>Deadwood</dt>
        <dd>{deadwoodText(s)}</dd>
      </dl>
    </section>
  );
}

/**
 * The full accounting of a finished gin hand, in the shared drawer: what
 * happened, the points as a sum, then each player's melds, lay-offs and
 * deadwood (the knocker first). Plain text: a definition list per player.
 * Portalled to the body: the board's container-query ancestors contain
 * layout, which would make the fixed drawer a box inside the board.
 */
export function HandResult({
  result,
  name,
  onClose,
}: {
  readonly result: GinHandResult;
  readonly name: GinSeatNamer;
  readonly onClose: () => void;
}): JSX.Element {
  const sum = ginPointsSum(result, seat => possessive(name(seat)));
  const order =
    result.knocker === null ? [] : [result.knocker, 1 - result.knocker];
  return createPortal(
    <Drawer
      title={`Hand result: ${GIN_HAND_KIND_LABEL[result.kind]}`}
      onClose={onClose}
      testId="gin-hand-result"
      className="gin-hand-result"
    >
      <p
        className="gin-hand-result__headline"
        data-testid="gin-hand-result-headline"
      >
        {headline(result, name)}
      </p>
      {sum !== null && <p data-testid="gin-hand-result-sum">Points: {sum}</p>}
      {sum !== null && result.multiplier > 1 && (
        <p data-testid="gin-hand-result-multiplier">
          {`The score is multiplied by ${String(result.multiplier)}: the first upcard was a spade (Oklahoma, spades double).`}
        </p>
      )}
      {order.map(seat => (
        <SeatSection key={seat} result={result} seat={seat} name={name} />
      ))}
    </Drawer>,
    document.body,
  );
}
