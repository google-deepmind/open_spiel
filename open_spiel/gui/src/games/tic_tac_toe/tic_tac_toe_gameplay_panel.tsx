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

import {type FC, type JSX, useCallback, useMemo} from 'react';

import type {
  OpenSpielState,
  SessionContextValue,
  ViewerContext,
} from '../shared/board_session.ts';
import {GameplayNotice} from '../shared/gameplay_notice.tsx';
import {GameplayShell} from '../shared/gameplay_shell.tsx';
import {type GameplayTurn, gameplayTurn} from '../shared/gameplay_turn.ts';
import {
  alternatingStagedState,
  boardResultText,
  useStagedHistory,
} from '../shared/use_staged_history.ts';
import '../shared/open_spiel_generic.css';
import './tic_tac_toe.css';

type UseSession = () => SessionContextValue;

const BOARD_SIZE = 3;

/** Each seat's mark: seat 0 plays X, seat 1 plays O. */
const MARKS = ['X', 'O'] as const;

/** The seat's mark, as the board and its status name the seat. */
function markOf(seat: number): string {
  const mark = MARKS[seat];
  if (mark === undefined) {
    throw new Error(`tic_tac_toe has no seat ${String(seat)}`);
  }
  return mark;
}

/**
 * A cell's name: where it is, in the engine's own coordinates, as the
 * History writes its moves ("x(1,1)" is the centre, "(row,column)" from 0).
 */
export function cellName(index: number): string {
  return `(${String(Math.floor(index / BOARD_SIZE))},${String(index % BOARD_SIZE)})`;
}

/** What the status says: the result, or whose mark is to move. */
export function ticTacToeStatusText(
  state: OpenSpielState,
  viewerSeat: number | null,
): string {
  if (state.isTerminal && state.returns) {
    return boardResultText(state.returns, MARKS, viewerSeat);
  }
  return state.currentPlayer >= 0
    ? `${markOf(state.currentPlayer)} to move`
    : 'Finished';
}

function buildBoard(state: OpenSpielState): string[] {
  const board = Array.from({length: BOARD_SIZE * BOARD_SIZE}, () => '');
  state.history.forEach((action, index) => {
    const mark = index % 2 === 0 ? 'X' : 'O';
    if (action >= 0 && action < board.length) {
      board[action] = mark;
    }
  });
  return board;
}

function parseActionLabel(value: string): string {
  return value.replace(':', '');
}

/** One move of the history, as a tic-tac-toe move list shows it. */
export interface TicTacToeHistoryEntry {
  turn: number;
  player: 'X' | 'O';
  label: string;
  row: number;
  col: number;
}

/** The history as a move list: who marked which cell. */
export function buildTicTacToeHistory(
  state: OpenSpielState,
): TicTacToeHistoryEntry[] {
  const labels = state.historyStr ? state.historyStr.split(/\s+/) : [];
  return state.history.map((action, index) => {
    const player = index % 2 === 0 ? 'X' : 'O';
    const text = labels[index];
    const label = text !== undefined ? parseActionLabel(text) : String(action);
    const row = Math.floor(action / BOARD_SIZE);
    const col = action % BOARD_SIZE;
    return {
      turn: index + 1,
      player,
      label,
      row,
      col,
    };
  });
}

/** The status row: whose turn it is, and what the viewer may do. */
function TicTacToeStatus({
  state,
  turn,
  viewer,
}: {
  readonly state: OpenSpielState;
  readonly turn: GameplayTurn;
  readonly viewer: ViewerContext;
}): JSX.Element {
  const {waitingOnBot, isGameMode, notMyTurn, notParticipant} = turn;
  // A seated viewer's own result, as checkers and chess say it.
  const seat = isGameMode ? viewer.seat : null;
  return (
    <div className="ogs-state-meta">
      <span data-testid="ttt-status">{ticTacToeStatusText(state, seat)}</span>
      <span>Turn: {state.turnNumber}</span>
      {seat !== null && !state.isTerminal && (
        <span>You play {markOf(seat)}</span>
      )}
      {waitingOnBot && (
        <span>
          {turn.currentAutoPlay ? 'Bot is thinking…' : 'Waiting for bot step…'}
        </span>
      )}
      {isGameMode && notParticipant && !state.isTerminal && (
        <span className="tto-spectator-notice">Spectating</span>
      )}
      {isGameMode &&
        viewer.isParticipant &&
        notMyTurn &&
        !waitingOnBot &&
        !state.isTerminal && (
          <span className="tto-turn-notice">Waiting for your turn…</span>
        )}
      {isGameMode &&
        viewer.isActiveSeat &&
        !waitingOnBot &&
        !state.isTerminal && (
          <span className="tto-turn-notice tto-turn-notice--active">
            Your turn!
          </span>
        )}
    </div>
  );
}

/** The nine cells, each a button that marks it. */
function TicTacToeBoard({
  board,
  state,
  turn,
  isBusy,
  legalActions,
  onCellClick,
}: {
  readonly board: readonly string[];
  readonly state: OpenSpielState;
  readonly turn: GameplayTurn;
  readonly isBusy: boolean;
  readonly legalActions: ReadonlySet<number>;
  readonly onCellClick: (index: number) => void;
}): JSX.Element {
  const {waitingOnBot, turnLocked} = turn;
  const lastActionIndex = state.history.at(-1) ?? null;
  return (
    <div className="tto-board-wrapper">
      <div className="tto-board" data-testid="ttt-board">
        {board.map((cell, index) => {
          const isLegal = legalActions.has(index);
          const isLast = lastActionIndex === index;
          const disabled =
            Boolean(cell) ||
            !isLegal ||
            isBusy ||
            state.isTerminal ||
            waitingOnBot ||
            turnLocked;
          const canPlay =
            isLegal &&
            !cell &&
            !state.isTerminal &&
            !waitingOnBot &&
            !turnLocked;

          return (
            <button
              key={index}
              className={`tto-cell${
                isLast ? ' tto-cell--last' : ''
              }${canPlay ? ' tto-cell--legal' : ''}`}
              disabled={disabled}
              onClick={() => {
                onCellClick(index);
              }}
              aria-label={`Play at ${cellName(index)}`}
              data-cell={index}
            >
              {cell}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** The panel's content, drawn once the session has a position. */
function TicTacToeContent({
  useSession,
}: {
  readonly useSession: UseSession;
}): JSX.Element {
  const {session, error, isBusy, submitAction, viewer} = useSession();
  // The seat's move and a bot's reply, arriving together, are drawn one
  // after the other: the position after the seat's move is shown first.
  const staged = useStagedHistory(session.state.history);
  const state = alternatingStagedState(session.state, staged.shown);
  const shownSession = state === session.state ? session : {...session, state};

  const board = useMemo(() => buildBoard(state), [state]);
  const legalActions = useMemo(
    () => new Set(state.legalActions.map(action => action.action)),
    [state.legalActions],
  );
  const shownViewer = {
    ...viewer,
    isActiveSeat: viewer.isActiveSeat && state === session.state,
  };
  const turn = gameplayTurn(shownSession, shownViewer);
  const {waitingOnBot, turnLocked} = turn;

  const onCellClick = useCallback(
    (index: number) => {
      if (
        legalActions.has(index) &&
        !board[index] &&
        !isBusy &&
        !state.isTerminal &&
        !waitingOnBot &&
        !turnLocked
      ) {
        void submitAction(index);
      }
    },
    [
      legalActions,
      board,
      isBusy,
      waitingOnBot,
      state.isTerminal,
      submitAction,
      turnLocked,
    ],
  );

  return (
    <>
      {error && <GameplayNotice tone="error">{error}</GameplayNotice>}
      <TicTacToeStatus state={state} turn={turn} viewer={shownViewer} />
      <TicTacToeBoard
        board={board}
        state={state}
        turn={turn}
        isBusy={isBusy}
        legalActions={legalActions}
        onCellClick={onCellClick}
      />
      {state.isTerminal && state.returns && (
        <GameplayNotice tone="turn" testId="game-result">
          {ticTacToeStatusText(
            state,
            session.mode === 'game' ? viewer.seat : null,
          )}
        </GameplayNotice>
      )}
    </>
  );
}

/** The tic-tac-toe Gameplay panel, reading its session from `useSession`. */
export function createTicTacToeGameplayPanel(useSession: UseSession): FC {
  const GameplayPanel: FC = () => (
    <GameplayShell className="ogs-panel">
      <TicTacToeContent useSession={useSession} />
    </GameplayShell>
  );
  GameplayPanel.displayName = 'TicTacToeGameplayPanel';
  return GameplayPanel;
}
