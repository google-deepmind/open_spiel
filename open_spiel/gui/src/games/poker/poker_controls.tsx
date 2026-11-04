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

// The poker table's controls: Fold, Check or Call, and the bet with its
// sizing box, or at the end of a hand its result and New hand. Split from
// poker_gameplay_panel.tsx.

import {type JSX, useCallback, useMemo, useState} from 'react';

import type {PokerSeat, PokerUiState} from './poker_state.ts';
import {clamp, formatChips} from './poker_table_parts.tsx';

/** The words and amounts the controls show for the seat to act. */
interface BetWords {
  callText: string;
  betVerb: string;
  base: number;
  seatStack: number;
  halfPotTarget: number;
  potTarget: number;
}

// Poker's own words, in this street's chips (the engine counts the whole
// hand and says "Call" for a check): "Check" when nothing is owed, "Call
// 50", "Bet 100" when nobody has bet this street, "Raise to 300"
// otherwise, "all-in" when it takes the whole stack.
function betWords(state: PokerUiState): BetWords {
  const currentSeat: PokerSeat | undefined =
    state.currentPlayer >= 0 ? state.seats[state.currentPlayer] : undefined;
  const highestContribution = Math.max(
    0,
    ...state.seats.map(seat => seat.contribution),
  );
  const callAmount = currentSeat
    ? Math.max(0, highestContribution - currentSeat.contribution)
    : 0;
  const seatStack = currentSeat?.startingStack ?? 0;
  const callPaid =
    currentSeat && seatStack > 0
      ? Math.min(callAmount, seatStack - currentSeat.contribution)
      : callAmount;
  const callAllIn =
    currentSeat !== undefined &&
    seatStack > 0 &&
    callAmount > 0 &&
    callPaid + currentSeat.contribution >= seatStack;
  const potAfterCall = state.pot + callAmount;
  return {
    callText:
      callAmount > 0
        ? `Call ${formatChips(callPaid)}${callAllIn ? ' all-in' : ''}`
        : 'Check',
    betVerb: state.streetLevel <= state.streetBase ? 'Bet' : 'Raise to',
    base: state.streetBase,
    seatStack,
    halfPotTarget: Math.ceil(highestContribution + potAfterCall / 2),
    potTarget: Math.ceil(highestContribution + potAfterCall),
  };
}

/** The bet the sizing box has chosen, and how to choose another. */
function useBetSize(state: PokerUiState): {
  betAmount: number;
  setBetAmount: (amount: number) => void;
  selectedBet: PokerUiState['actionGroups']['bets'][number] | undefined;
  setPresetBet: (target: number) => void;
  minBet: number;
  maxBet: number;
} {
  const betActions = state.actionGroups.bets;
  const minBet = betActions[0]?.amount ?? 0;
  const maxBet = betActions[betActions.length - 1]?.amount ?? 0;
  // The amount chosen, kept while the legal range holds it; otherwise the
  // smallest legal bet (none when there is no bet to make).
  const [chosen, setBetAmount] = useState(minBet);
  const betAmount =
    betActions.length === 0
      ? 0
      : chosen >= minBet && chosen <= maxBet
        ? chosen
        : minBet;
  const selectedBet = useMemo(() => {
    if (betActions.length === 0) {
      return undefined;
    }
    const normalized = clamp(Math.round(betAmount), minBet, maxBet);
    return betActions.find(action => action.amount === normalized);
  }, [betActions, betAmount, minBet, maxBet]);
  const setPresetBet = useCallback(
    (target: number) => {
      const first = betActions[0];
      if (first === undefined) {
        return;
      }
      const legal = betActions.reduce((best, candidate) => {
        const bestDistance = Math.abs(best.amount - target);
        const candidateDistance = Math.abs(candidate.amount - target);
        return candidateDistance < bestDistance ? candidate : best;
      }, first);
      setBetAmount(legal.amount);
    },
    [betActions],
  );
  return {betAmount, setBetAmount, selectedBet, setPresetBet, minBet, maxBet};
}

/** What the controls are drawn from. */
export interface PokerControlsProps {
  readonly state: PokerUiState;
  readonly canAct: boolean;
  readonly isBusy: boolean;
  /** Whether the game bets in fixed sizes, so offers no sizing box. */
  readonly fixedLimit: boolean;
  readonly showdownText: string;
  readonly submitIfReady: (action: number | undefined) => void;
  readonly startNewHand: () => void;
}

function Presets({
  size,
  words,
  canAct,
}: {
  readonly size: ReturnType<typeof useBetSize>;
  readonly words: BetWords;
  readonly canAct: boolean;
}): JSX.Element {
  const {setPresetBet, minBet, maxBet} = size;
  const presets: Array<[string, number]> = [
    ['Min', minBet],
    ['1/2 pot', words.halfPotTarget],
    ['Pot', words.potTarget],
    ['All in', maxBet],
  ];
  return (
    <>
      {presets.map(([text, target]) => (
        <button
          disabled={!canAct}
          key={text}
          onClick={() => {
            setPresetBet(target);
          }}
          type="button"
        >
          {text}
        </button>
      ))}
    </>
  );
}

function BetBox({
  size,
  words,
  canAct,
}: {
  readonly size: ReturnType<typeof useBetSize>;
  readonly words: BetWords;
  readonly canAct: boolean;
}): JSX.Element {
  const {betAmount, setBetAmount, selectedBet, minBet, maxBet} = size;
  const {base} = words;
  return (
    <div className="poker-betbox">
      <div className="poker-betbox__presets">
        <Presets size={size} words={words} canAct={canAct} />
        <input
          aria-label="Bet size"
          id="poker-bet-amount"
          max={maxBet - base}
          min={minBet - base}
          onChange={event => {
            setBetAmount(
              clamp(Number(event.target.value) + base, minBet, maxBet),
            );
          }}
          step={1}
          type="number"
          value={betAmount - base}
        />
      </div>
      <input
        aria-label="Bet amount"
        className="poker-betbox__slider"
        max={maxBet - base}
        min={minBet - base}
        onChange={event => {
          setBetAmount(Number(event.target.value) + base);
        }}
        step={1}
        type="range"
        value={betAmount - base}
      />
      {!selectedBet && (
        <div className="poker-betbox__hint">
          Choose a legal amount from {formatChips(minBet - base)} to{' '}
          {formatChips(maxBet - base)}.
        </div>
      )}
    </div>
  );
}

function PrimaryButtons({
  props,
  size,
  words,
}: {
  readonly props: PokerControlsProps;
  readonly size: ReturnType<typeof useBetSize>;
  readonly words: BetWords;
}): JSX.Element {
  const {state, canAct, submitIfReady} = props;
  const {fold, call, bets} = state.actionGroups;
  const {selectedBet} = size;
  const betButtonLabel = selectedBet
    ? `${words.betVerb} ${formatChips(selectedBet.amount - words.base)}${
        words.seatStack > 0 && selectedBet.amount >= words.seatStack
          ? ' all-in'
          : ''
      }`
    : 'Choose size';
  return (
    <div className="poker-controls__primary">
      {fold && (
        <button
          className="poker-action poker-action--fold"
          disabled={!canAct}
          onClick={() => {
            submitIfReady(fold.action);
          }}
          type="button"
        >
          Fold
        </button>
      )}
      {call && (
        <button
          className="poker-action poker-action--call"
          disabled={!canAct}
          onClick={() => {
            submitIfReady(call.action);
          }}
          type="button"
        >
          {words.callText}
        </button>
      )}
      {bets.length > 0 && (
        <button
          className="poker-action poker-action--bet"
          disabled={!canAct || !selectedBet}
          onClick={() => {
            submitIfReady(selectedBet?.action);
          }}
          type="button"
        >
          {betButtonLabel}
        </button>
      )}
    </div>
  );
}

/** The controls under the table. */
export function PokerControls(props: PokerControlsProps): JSX.Element {
  const {state, canAct, isBusy, fixedLimit, showdownText, startNewHand} = props;
  const size = useBetSize(state);
  const words = betWords(state);
  return (
    <section
      className={`poker-controls${
        state.isTerminal ? ' poker-controls--terminal' : ''
      }`}
      aria-label="Poker actions"
    >
      {state.isTerminal ? (
        <div className="poker-terminal">
          <strong>{showdownText}</strong>
          <button
            className="poker-action poker-action--bet"
            disabled={isBusy}
            onClick={startNewHand}
            type="button"
          >
            New hand
          </button>
        </div>
      ) : (
        <>
          <PrimaryButtons props={props} size={size} words={words} />
          {state.actionGroups.bets.length > 0 && !fixedLimit && (
            <BetBox size={size} words={words} canAct={canAct} />
          )}
        </>
      )}
    </section>
  );
}
