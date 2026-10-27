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

// The poker board: the hand bar (the game, the street, blinds or ante,
// stacks, seats and New hand), the table (the board's cards, the pot and a
// plate per seat, with chips flying to the pot), the controls, and the
// Action History.

import {
  type JSX,
  type RefObject,
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';

import {useStickToBottom} from '../../actions/use_stick_to_bottom.ts';
import {PokerControls} from './poker_controls.tsx';
import {type SessionPayload, usePokerSession} from './poker_session.ts';
import {
  type PokerUiState,
  derivePokerState,
  historyEntryText,
  roundName,
} from './poker_state.ts';
import {
  CardView,
  ChipFlightView,
  SeatView,
  clamp,
  formatBlindText,
  formatChips,
  formatStackText,
  terminalSummary,
  winningCards,
} from './poker_table_parts.tsx';
import {useBoardInDealOrder} from './use_board_in_deal_order.ts';
import {
  type PokerMotion,
  motionSettled,
  shownNumbers,
  usePokerMotion,
} from './use_poker_motion.ts';

import './poker.css';

// The seat counts the hand bar offers.
const SEAT_COUNTS = [2, 3, 4, 5, 6];

function HandBar({
  state,
  title,
  seatCount,
  setSeatCount,
  isBusy,
  startNewHand,
}: {
  readonly state: PokerUiState;
  readonly title: string;
  readonly seatCount: number;
  readonly setSeatCount: (count: number) => void;
  readonly isBusy: boolean;
  readonly startNewHand: () => void;
}): JSX.Element {
  const stacks = formatStackText(state.seats.map(seat => seat.startingStack));
  return (
    <div className="poker-handbar">
      <div className="poker-handbar__title">
        <strong>{title}</strong>
        <span className="poker-handbar__street">{state.currentStreet}</span>
      </div>
      <div className="poker-handbar__meta">
        <span>{formatBlindText(state.blinds, state.ante)}</span>
        {stacks !== null && <span>{stacks}</span>}
      </div>
      <div className="poker-new-hand" aria-label="New hand setup">
        <label htmlFor="poker-seat-count">Seats</label>
        <select
          id="poker-seat-count"
          onChange={event => {
            setSeatCount(Number(event.target.value));
          }}
          value={seatCount}
        >
          {SEAT_COUNTS.map(count => (
            <option key={count} value={count}>
              {count}
            </option>
          ))}
        </select>
        <button disabled={isBusy} onClick={startNewHand} type="button">
          New hand
        </button>
      </div>
    </div>
  );
}

function BoardCards({
  state,
  motion,
  boardLanded,
  playing,
  slots,
}: {
  readonly state: PokerUiState;
  readonly motion: ReturnType<typeof usePokerMotion>['motion'];
  readonly boardLanded: (index: number) => void;
  readonly playing: ReadonlySet<string>;
  readonly slots: number;
}): JSX.Element {
  return (
    <div className="poker-board__cards">
      {Array.from({length: slots}, (_, index) => {
        const card = state.board[index];
        const delayBaseMs = motion.boardDelays.get(index);
        return (
          <CardView
            card={card}
            key={card?.code ?? `empty-${String(index)}`}
            motion={
              card && delayBaseMs !== undefined
                ? {
                    kind: 'deal',
                    delayBaseMs,
                    motion,
                    onLanded: () => {
                      boardLanded(index);
                    },
                  }
                : undefined
            }
            plays={card !== undefined && playing.has(card.code)}
          />
        );
      })}
    </div>
  );
}

/**
 * The table as it may be drawn while chips fly and cards turn: the hand's
 * outcome (the result, the winner, a showdown's hand names, a fold that came
 * with the chips) waits until everything has landed, so the opponent's
 * action is seen before the result it leads to.
 */
export function heldTable(
  state: PokerUiState,
  motion: PokerMotion,
  boardSizes: readonly number[],
): PokerUiState {
  if (motionSettled(motion)) return state;
  const turning = new Set(
    [...motion.revealDelays.keys()].map(key => Number(key.split(':')[0])),
  );
  return {
    ...state,
    ending: null,
    returns: null,
    currentStreet: state.isTerminal
      ? roundName(state.board, boardSizes)
      : state.currentStreet,
    seats: state.seats.map(seat => ({
      ...seat,
      returnValue: undefined,
      isWinner: false,
      hasFolded: seat.hasFolded && !motion.heldFolds.has(seat.slot),
      bestHand: turning.has(seat.slot) ? undefined : seat.bestHand,
    })),
  };
}

function Table({
  state,
  session,
  motionState,
  tableRef,
  potRef,
  viewerSeat,
}: {
  readonly state: PokerUiState;
  readonly session: SessionPayload;
  readonly motionState: ReturnType<typeof usePokerMotion>;
  readonly tableRef: RefObject<HTMLDivElement | null>;
  readonly potRef: RefObject<HTMLDivElement | null>;
  readonly viewerSeat: number | null;
}): JSX.Element {
  const {motion, chipLanded, boardLanded, revealLanded} = motionState;
  // The pot, Stack and In as shown: a chip in the air has not reached the
  // pot yet. Controls use the state's numbers.
  const shown = shownNumbers(state, motion);
  const playing = winningCards(state.seats, state.ending);
  const slots = session.game.boardSizes.reduce((sum, size) => sum + size, 0);
  const hasStacks = state.seats.some(seat => seat.startingStack > 0);
  return (
    <section
      className={`poker-table-stage poker-table-stage--${String(state.seats.length)}`}
      aria-label="Poker table"
    >
      <div
        className={`poker-table poker-table--${String(state.seats.length)}${
          state.isTerminal ? ' poker-table--terminal' : ''
        }`}
        ref={tableRef}
      >
        <div className="poker-board">
          <BoardCards
            state={state}
            motion={motion}
            boardLanded={boardLanded}
            playing={playing}
            slots={slots}
          />
          <div className="poker-pot" ref={potRef}>
            <span>Pot</span>
            <strong>{formatChips(shown.pot)}</strong>
          </div>
        </div>
        {state.seats.map(seat => (
          <SeatView
            key={seat.slot}
            motion={motion}
            revealLanded={revealLanded}
            seat={seat}
            holeCards={session.game.holeCards}
            hasStacks={hasStacks}
            shown={{
              stack: shown.stack(seat.slot, seat.stack),
              contribution: shown.contribution(seat.slot, seat.contribution),
            }}
            totalSeats={state.seats.length}
            playing={playing}
            isViewer={seat.slot === viewerSeat}
          />
        ))}
        {motion.chips.map(chip => (
          <ChipFlightView
            chip={chip}
            motion={motion}
            key={chip.key}
            onLanded={chipLanded}
          />
        ))}
      </div>
    </section>
  );
}

function History({state}: {readonly state: PokerUiState}): JSX.Element {
  // Keyed by session in the parent, so a new hand starts pinned.
  const {ref, onScroll} = useStickToBottom<HTMLElement>(state.history.length);
  return (
    <section
      className="poker-history"
      aria-label="Action history"
      ref={ref}
      onScroll={onScroll}
    >
      <h3>Action History</h3>
      {state.history.length === 0 ? (
        <p>No actions yet.</p>
      ) : (
        <ol>
          {state.history.map((entry, index) => {
            const row = historyEntryText(entry, state.seats);
            return (
              <li key={`${entry.label}-${String(index)}`}>
                <span title={row.who}>{row.who}</span>
                <span>{row.what}</span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

/** The derived table, or the reason the hand's record cannot be drawn. */
function useDerived(session: SessionPayload): {
  state: PokerUiState | null;
  error: string | null;
} {
  // A record the table cannot be drawn from (a history row that disagrees
  // with the betting string, a fold by no seat) is said on the panel, in
  // full, not a blank pane.
  return useMemo(() => {
    try {
      return {state: derivePokerState(session), error: null};
    } catch (caught) {
      return {
        state: null,
        error: `This hand's record cannot be drawn: ${caught instanceof Error ? caught.message : String(caught)}`,
      };
    }
  }, [session]);
}

/** The viewer's own seat, marked on the table; null for a spectator or the
 *  oracle. */
function seatOf(viewer: {participantId: string | null}): number | null {
  return viewer.participantId === null ? null : Number(viewer.participantId);
}

/** The result line under the table; empty until the outcome may be shown. */
function showdownLine(state: PokerUiState, viewerSeat: number | null): string {
  return state.ending === null
    ? ''
    : terminalSummary(state.seats, state.returns, state.ending, viewerSeat);
}

/** The poker board, drawn from the session of the renderer it is in. */
export function PokerGameplay(): JSX.Element {
  const {session, error, submitAction, isBusy, createSession, viewer} =
    usePokerSession();
  const derived = useDerived(session);
  const pokerState = useBoardInDealOrder(derived.state, session.sessionId);
  const tableRef = useRef<HTMLDivElement>(null);
  const potRef = useRef<HTMLDivElement>(null);
  const motionState = usePokerMotion(pokerState, tableRef, potRef);
  const viewerSeat = seatOf(viewer);
  const [chosenSeats, setSeatCount] = useState<number | null>(null);
  const seatCount = chosenSeats ?? pokerState?.seats.length ?? 2;
  const submitIfReady = useCallback(
    (action: number | undefined) => {
      if (typeof action !== 'number' || isBusy) {
        return;
      }
      void submitAction(action);
    },
    [isBusy, submitAction],
  );
  const startNewHand = useCallback(() => {
    setSeatCount(null);
    createSession({seats: clamp(Math.round(seatCount), 2, 6)});
  }, [createSession, seatCount]);
  if (!pokerState) {
    return (
      <div className="ogs-panel poker-error">
        <div className="ogs-error" role="alert">
          {derived.error ?? error ?? 'Poker session unavailable.'}
        </div>
      </div>
    );
  }
  const {boardSizes} = session.game;
  const shownState = heldTable(pokerState, motionState.motion, boardSizes);
  // Only the seat to act may act; another seat's turn offers no presses.
  const canAct =
    viewer.canAct &&
    !pokerState.isTerminal &&
    pokerState.currentPlayer >= 0 &&
    session.state.legalActions.length > 0 &&
    !isBusy;
  return (
    <div className="poker-gameplay">
      {error !== null && (
        <div className="poker-notice" role="alert">
          {error}
        </div>
      )}
      <HandBar
        state={shownState}
        title={session.game.title}
        seatCount={seatCount}
        setSeatCount={setSeatCount}
        isBusy={isBusy}
        startNewHand={startNewHand}
      />
      <Table
        state={shownState}
        session={session}
        motionState={motionState}
        tableRef={tableRef}
        potRef={potRef}
        viewerSeat={viewerSeat}
      />
      <PokerControls
        state={pokerState}
        canAct={canAct}
        isBusy={isBusy}
        fixedLimit={session.game.betting === 'limit'}
        showdownText={showdownLine(shownState, viewerSeat)}
        submitIfReady={submitIfReady}
        startNewHand={startNewHand}
      />
      <History key={session.sessionId} state={pokerState} />
    </div>
  );
}
