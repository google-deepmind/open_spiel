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

import {
  type CSSProperties,
  type FC,
  type JSX,
  type MouseEvent,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

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
import './connect_four.css';

type UseSession = () => SessionContextValue;

/** The engine's default number of rows (connect_four.cc). */
export const DEFAULT_ROWS = 6;
/** The engine's default number of columns (connect_four.cc). */
export const DEFAULT_COLUMNS = 7;

/** The size of the grid the engine plays. */
export interface ConnectFourDimensions {
  rows: number;
  columns: number;
}

function readDimension(
  gameParams: Record<string, unknown> | undefined,
  key: 'rows' | 'columns',
  fallback: number,
): number {
  const value = gameParams?.[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new Error(
      `connect_four: game parameter ${key}=${JSON.stringify(value)} is not a positive integer`,
    );
  }
  return value;
}

/**
 * The grid the engine plays, from the session's effective game parameters
 * (`session.gameParams`, sent by the adapter from the loaded game). Setup
 * offers `rows` and `columns`, so the board must never assume 6 x 7; an
 * absent key means the engine default.
 */
export function connectFourDimensions(
  gameParams: Record<string, unknown> | undefined,
): ConnectFourDimensions {
  return {
    rows: readDimension(gameParams, 'rows', DEFAULT_ROWS),
    columns: readDimension(gameParams, 'columns', DEFAULT_COLUMNS),
  };
}

type BoardCell = '' | 'X' | 'O';

/**
 * Replay the action history (action = column) onto a rows x columns grid.
 * An action outside the grid or into a full column means the UI and the
 * engine disagree about the board, so it throws rather than drop the piece.
 */
export function buildBoard(
  history: readonly number[],
  {rows, columns}: ConnectFourDimensions,
): BoardCell[][] {
  const board: BoardCell[][] = Array.from({length: rows}, () =>
    Array.from({length: columns}, (): BoardCell => ''),
  );

  history.forEach((action, index) => {
    if (!Number.isInteger(action) || action < 0 || action >= columns) {
      throw new Error(
        `connect_four: history action ${String(action)} (move ${String(index + 1)}) is outside a ${String(columns)}-column board`,
      );
    }
    for (let row = rows - 1; row >= 0; row--) {
      const cells = board[row];
      if (cells !== undefined && !cells[action]) {
        cells[action] = index % 2 === 0 ? 'X' : 'O';
        return;
      }
    }
    throw new Error(
      `connect_four: history action ${String(action)} (move ${String(index + 1)}) plays into a full column of a ${String(rows)}-row board`,
    );
  });

  return board;
}

/** Each seat's colour: seat 0 drops red discs, seat 1 yellow. */
const COLOURS = ['Red', 'Yellow'] as const;

function colourOf(seat: number): string {
  const colour = COLOURS[seat];
  if (colour === undefined) {
    throw new Error(`connect_four has no seat ${String(seat)}`);
  }
  return colour;
}

/** What the status says: the result, or whose colour is to move. */
export function connectFourStatusText(
  state: OpenSpielState,
  viewerSeat: number | null,
): string {
  if (state.isTerminal && state.returns) {
    return boardResultText(state.returns, COLOURS, viewerSeat);
  }
  return state.currentPlayer >= 0
    ? `${colourOf(state.currentPlayer)} to move`
    : 'Finished';
}

// The four directions a line of four can run: across, up, and both diagonals.
const LINE_STEPS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

/**
 * The cells ("row-column") of every line of four or more discs of one
 * colour: the winning line once the game is won, none in a draw.
 */
export function winningCells(board: BoardCell[][]): Set<string> {
  const cells = new Set<string>();
  const at = (row: number, col: number): BoardCell => board[row]?.[col] ?? '';
  board.forEach((cellsInRow, row) => {
    cellsInRow.forEach((cell, col) => {
      if (cell === '') return;
      for (const [dr, dc] of LINE_STEPS) {
        const line = [0, 1, 2, 3].map(k => [row + dr * k, col + dc * k]);
        if (line.every(([r = -1, c = -1]) => at(r, c) === cell)) {
          line.forEach(([r = -1, c = -1]) =>
            cells.add(`${String(r)}-${String(c)}`),
          );
        }
      }
    });
  });
  return cells;
}

/** The cell ("row-column") the last action dropped its disc into, if any. */
function lastDrop(
  history: readonly number[],
  board: BoardCell[][],
): {key: string; row: number} | null {
  const column = history.at(-1);
  if (column === undefined) return null;
  const row = board.findIndex(cells => cells[column] !== '');
  return row < 0 ? null : {key: `${String(row)}-${String(column)}`, row};
}

/** One move of the history, as a connect four move list shows it. */
export interface ConnectFourHistoryEntry {
  turn: number;
  player: 'X' | 'O';
  column: number;
  label: string;
}

/** The history as a move list: who dropped into which column. */
export function buildConnectFourHistory(
  state: OpenSpielState,
): ConnectFourHistoryEntry[] {
  const labels =
    state.historyStr.trim().length > 0 ? state.historyStr.split(/\s+/) : [];
  return state.history.map((action, index) => {
    const player = index % 2 === 0 ? 'X' : 'O';
    const label = labels[index] ?? `Column ${String(action + 1)}`;
    return {
      turn: index + 1,
      player,
      column: action,
      label,
    };
  });
}

interface Layout extends ConnectFourDimensions {
  board: BoardCell[][] | null;
  error: string | null;
}

// A disagreement between the UI and the engine (malformed size param, a move
// outside the grid or into a full column) is shown IN the panel instead of
// the board: thrown during render it would take the board down, and the
// player would see the game's text in its place. Nothing is guessed and no
// board is drawn.
function boardLayout(
  gameParams: Record<string, unknown>,
  history: readonly number[],
): Layout {
  try {
    const dims = connectFourDimensions(gameParams);
    return {...dims, board: buildBoard(history, dims), error: null};
  } catch (err) {
    return {
      rows: DEFAULT_ROWS,
      columns: DEFAULT_COLUMNS,
      board: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/** The status row: whose turn it is, and what the viewer may do. */
function ConnectFourStatus({
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
      <span data-testid="c4-status">{connectFourStatusText(state, seat)}</span>
      <span>Turn: {state.turnNumber}</span>
      {seat !== null && !state.isTerminal && (
        <span>You play {colourOf(seat)}</span>
      )}
      {waitingOnBot && (
        <span>
          {turn.currentAutoPlay ? 'Bot is thinking…' : 'Waiting for bot step…'}
        </span>
      )}
      {isGameMode && notParticipant && !state.isTerminal && (
        <span>Spectating</span>
      )}
      {isGameMode &&
        viewer.isParticipant &&
        notMyTurn &&
        !waitingOnBot &&
        !state.isTerminal && <span>Waiting for your turn…</span>}
      {isGameMode &&
        viewer.isActiveSeat &&
        !waitingOnBot &&
        !state.isTerminal && (
          <span style={{fontWeight: 'bold'}}>Your turn!</span>
        )}
    </div>
  );
}

/**
 * The board's width: as wide as its container, no taller than it, and at
 * most 640 px, kept current as the container is resized.
 */
function useBoardWidth(
  container: RefObject<HTMLDivElement | null>,
  rows: number,
  columns: number,
  hasBoard: boolean,
): number | null {
  const [boardWidth, setBoardWidth] = useState<number | null>(null);
  useEffect(() => {
    const element = container.current;
    if (!element) {
      return;
    }
    const computeBoardWidth = (): void => {
      const rect = element.getBoundingClientRect();
      const {width, height} = rect;
      if (width <= 0 || height <= 0) {
        return;
      }
      const maxWidthFromHeight = height * (columns / rows);
      const nextWidth = Math.min(width, maxWidthFromHeight, 640);
      setBoardWidth(prev =>
        prev !== null && Math.abs(prev - nextWidth) < 0.5 ? prev : nextWidth,
      );
    };

    computeBoardWidth();

    if (typeof ResizeObserver === 'undefined') {
      return;
    }

    const resizeObserver = new ResizeObserver(computeBoardWidth);
    resizeObserver.observe(element);
    return () => {
      resizeObserver.disconnect();
    };
  }, [container, rows, columns, hasBoard]);
  return boardWidth;
}

/**
 * The board's click: a press on a hole plays its column; a press between
 * holes plays the column under the pointer.
 */
function useBoardClick(
  boardRef: RefObject<HTMLDivElement | null>,
  columns: number,
  canPlay: boolean,
  onColumnClick: (column: number) => void,
): (event: MouseEvent<HTMLDivElement>) => void {
  return useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (!canPlay) {
        return;
      }
      const target = event.target instanceof HTMLElement ? event.target : null;
      const columnElement = target?.closest<HTMLElement>('[data-column]');
      if (columnElement) {
        const columnValue = Number(columnElement.dataset['column']);
        if (!Number.isNaN(columnValue)) {
          onColumnClick(columnValue);
          return;
        }
      }
      const boardElement = boardRef.current;
      if (!boardElement) {
        return;
      }
      const rect = boardElement.getBoundingClientRect();
      const relativeX = event.clientX - rect.left;
      const widthPerColumn = rect.width / columns;
      if (widthPerColumn <= 0) {
        return;
      }
      const computedColumn = Math.max(
        0,
        Math.min(columns - 1, Math.floor(relativeX / widthPerColumn)),
      );
      onColumnClick(computedColumn);
    },
    [boardRef, canPlay, onColumnClick, columns],
  );
}

function discClass(cell: BoardCell): string {
  return cell === 'X'
    ? ' cf-cell--player-one'
    : cell === 'O'
      ? ' cf-cell--player-two'
      : '';
}

function cellClass(
  cell: BoardCell,
  interactive: boolean,
  winning: boolean,
  dropping: boolean,
): string {
  // A disc still dropping is drawn by its own element, falling into the
  // empty hole.
  return `cf-cell${dropping ? '' : discClass(cell)}${
    interactive ? ' cf-cell--interactive' : ''
  }${winning ? ' cf-cell--winning' : ''}`;
}

/** The grid of holes, each holding its disc. */
function ConnectFourGrid({
  board,
  rows,
  columns,
  canPlay,
  legalActions,
  onColumnClick,
  winning,
  drop,
}: {
  readonly board: BoardCell[][];
  readonly rows: number;
  readonly columns: number;
  readonly canPlay: boolean;
  readonly legalActions: ReadonlySet<number>;
  readonly onColumnClick: (column: number) => void;
  /** The cells of the winning line; empty while the game is on. */
  readonly winning: ReadonlySet<string>;
  /** The disc the last move dropped, which falls into its hole. */
  readonly drop: {key: string; row: number; ply: number} | null;
}): JSX.Element {
  const boardRef = useRef<HTMLDivElement | null>(null);
  const boardContainerRef = useRef<HTMLDivElement | null>(null);
  const boardWidth = useBoardWidth(boardContainerRef, rows, columns, true);
  const handleBoardClick = useBoardClick(
    boardRef,
    columns,
    canPlay,
    onColumnClick,
  );
  const style: CSSProperties & Record<string, number | string> = {
    '--cf-rows': rows,
    '--cf-columns': columns,
    ...(boardWidth !== null ? {width: `${String(boardWidth)}px`} : {}),
  };
  return (
    <div className="cf-board-wrapper">
      <div ref={boardContainerRef} className="cf-board-container">
        <div
          ref={boardRef}
          className={`cf-board${canPlay ? ' cf-board--interactive' : ''}`}
          onClick={handleBoardClick}
          style={style}
          data-testid="c4-board"
          data-rows={rows}
          data-columns={columns}
        >
          {board.map((row, rowIndex) =>
            row.map((cell, columnIndex) => {
              const key = `${String(rowIndex)}-${String(columnIndex)}`;
              const dropping = drop !== null && drop.key === key;
              return (
                <div
                  key={key}
                  className={cellClass(
                    cell,
                    canPlay && legalActions.has(columnIndex),
                    winning.has(key),
                    dropping,
                  )}
                  data-column={columnIndex}
                  data-row={rowIndex}
                  data-disc={cell === '' ? undefined : cell}
                >
                  {dropping && (
                    <span
                      key={drop.ply}
                      aria-hidden="true"
                      className={`cf-disc-drop${discClass(cell)}`}
                      style={{'--cf-drop-rows': drop.row + 1} as CSSProperties}
                    />
                  )}
                </div>
              );
            }),
          )}
        </div>
      </div>
    </div>
  );
}

/** The panel's content, drawn once the session has a position. */
function ConnectFourContent({
  useSession,
}: {
  readonly useSession: UseSession;
}): JSX.Element {
  const {session, error, isBusy, submitAction, viewer} = useSession();
  const {gameParams} = session;
  // The seat's disc and a bot's reply, arriving together, are dropped one
  // after the other: the position after the seat's move is shown first.
  const staged = useStagedHistory(session.state.history);
  const state = alternatingStagedState(session.state, staged.shown);
  const shownSession = state === session.state ? session : {...session, state};
  const shownViewer = {
    ...viewer,
    isActiveSeat: viewer.isActiveSeat && state === session.state,
  };
  const layout = useMemo(
    () => boardLayout(gameParams, state.history),
    [gameParams, state.history],
  );
  const {rows, columns, board} = layout;
  const legalActions = useMemo(
    () => new Set(state.legalActions.map(action => action.action)),
    [state.legalActions],
  );
  const turn = gameplayTurn(shownSession, shownViewer);
  const {waitingOnBot, turnLocked} = turn;
  const canPlay = !state.isTerminal && !waitingOnBot && !isBusy && !turnLocked;
  const winning = useMemo(
    () =>
      board !== null && state.isTerminal
        ? winningCells(board)
        : new Set<string>(),
    [board, state.isTerminal],
  );
  // Only a disc that has just arrived falls; a page load or a replay jump
  // draws the board at rest.
  const ply = state.history.length;
  const [seen, setSeen] = useState({ply, dropPly: null as number | null});
  const dropPly =
    seen.ply === ply ? seen.dropPly : ply === seen.ply + 1 ? ply : null;
  if (seen.ply !== ply) setSeen({ply, dropPly});
  const landed = board !== null ? lastDrop(state.history, board) : null;
  const drop = landed !== null && dropPly === ply ? {...landed, ply} : null;

  const onColumnClick = useCallback(
    (column: number) => {
      if (
        legalActions.has(column) &&
        !isBusy &&
        !state.isTerminal &&
        !waitingOnBot
      ) {
        void submitAction(column);
      }
    },
    [legalActions, isBusy, waitingOnBot, state.isTerminal, submitAction],
  );

  return (
    <>
      {error && <GameplayNotice tone="error">{error}</GameplayNotice>}
      <ConnectFourStatus state={state} turn={turn} viewer={shownViewer} />
      {layout.error !== null && (
        <div className="ogs-error" role="alert" data-testid="c4-board-error">
          Cannot draw this board: {layout.error}
        </div>
      )}
      {board !== null && (
        <ConnectFourGrid
          board={board}
          rows={rows}
          columns={columns}
          canPlay={canPlay}
          legalActions={legalActions}
          onColumnClick={onColumnClick}
          winning={winning}
          drop={drop}
        />
      )}
    </>
  );
}

/** The connect four Gameplay panel, reading its session from `useSession`. */
export function createConnectFourGameplayPanel(useSession: UseSession): FC {
  const GameplayPanel: FC = () => (
    <GameplayShell className="ogs-panel">
      <ConnectFourContent useSession={useSession} />
    </GameplayShell>
  );
  GameplayPanel.displayName = 'ConnectFourGameplayPanel';
  return GameplayPanel;
}
