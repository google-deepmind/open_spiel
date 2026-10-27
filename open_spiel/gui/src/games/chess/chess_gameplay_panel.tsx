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

import {type JSX, useCallback, useMemo, useState} from 'react';

import {HUMAN} from '../../players/seat_draft.ts';
import {ChessBoard, type PendingPromotion} from './chess_board.tsx';
import {
  useBoardOrientation,
  useBoardTheme,
  useShowCoordinates,
} from './chess_config_section.tsx';
import {ChessPlayerStrip} from './chess_player_strip.tsx';
import type {
  ChessViewer,
  MoveHint,
  SessionContextValue,
  SessionPayload,
} from './chess_session.ts';
import {
  SLOT_OF,
  gameResult,
  materialDiff,
  other,
  playerName,
  type Side,
  type SideMaterial,
} from './chess_table.ts';
import type {PromotionChoice} from './promotion_dialog.tsx';
import {useChessBoard} from './use_chess_board.ts';

type UseSession = () => SessionContextValue;

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

/**
 * Build the map from a move's squares (and promotion piece) to its action.
 */
function buildActionMap(moves: readonly MoveHint[]): Map<string, number> {
  const byCoordinate = new Map<string, number>();
  for (const move of moves) {
    byCoordinate.set(
      `${move.from}${move.to}${move.promotion ?? ''}`,
      move.action,
    );
  }
  return byCoordinate;
}

/** The properties of the board's content. */
interface ChessContentProps {
  useSession: UseSession;
}

/**
 * The moves made on the board: each resolved to its action and sent, a
 * promotion first asking for its piece, and the board told when a move
 * attempt settles.
 */
function useMoveHandlers(
  moves: readonly MoveHint[],
  submitAction: (action: number) => Promise<void>,
  onSent: (from: string, to: string) => void,
) {
  // Track pending promotion (when pawn reaches back rank)
  const [pendingPromotion, setPendingPromotion] =
    useState<PendingPromotion | null>(null);

  // Bumped when a move attempt settles, so the board resets to the
  // server's position if the move did not land (chess_board_input.ts).
  const [boardResync, setBoardResync] = useState(0);
  const resyncBoard = useCallback(() => {
    setBoardResync(n => n + 1);
  }, []);

  // Build the action map for move resolution
  const actionMap = useMemo(() => buildActionMap(moves), [moves]);

  // Resolve a move to its action and send it.
  const send = useCallback(
    (from: string, to: string, promotion?: string) => {
      const actionId = actionMap.get(`${from}${to}${promotion ?? ''}`);
      if (actionId === undefined) {
        resyncBoard();
      } else {
        onSent(from, to);
        void submitAction(actionId).finally(resyncBoard);
      }
    },
    [actionMap, submitAction, resyncBoard, onSent],
  );

  // Handle regular move from the board (non-promotion)
  const handleMove = useCallback(
    (from: string, to: string) => {
      send(from, to);
    },
    [send],
  );

  // Handle promotion move - shows promotion dialog
  const handlePromotionMove = useCallback((from: string, to: string) => {
    // Determine color based on destination rank
    // White promotes on rank 8, black promotes on rank 1
    const destRank = to[1];
    const color = destRank === '8' ? 'white' : 'black';
    setPendingPromotion({from, to, color});
  }, []);

  // Handle promotion piece selection
  const handlePromotionComplete = useCallback(
    (from: string, to: string, piece: PromotionChoice) => {
      send(from, to, piece);
      setPendingPromotion(null);
    },
    [send],
  );

  // Handle promotion cancellation
  const handleCancelPromotion = useCallback(() => {
    setPendingPromotion(null);
    resyncBoard();
  }, [resyncBoard]);

  return {
    pendingPromotion,
    boardResync,
    handleMove,
    handlePromotionMove,
    handlePromotionComplete,
    handleCancelPromotion,
  };
}

/** Whose turn it is, and whether the viewer may move now. */
function turnState(
  session: SessionPayload,
  viewer: ChessViewer,
  isBusy: boolean,
) {
  // Determine current player and turn state
  const currentPlayerSlot = session.state.currentPlayer;
  const currentPlayerEntry =
    currentPlayerSlot >= 0
      ? session.players.find(player => player.slot === currentPlayerSlot)
      : undefined;
  const currentAutoPlay = Boolean(currentPlayerEntry?.autoPlay);
  const isHumanTurn =
    currentPlayerSlot >= 0 && currentPlayerEntry?.controller === HUMAN;

  // Game mode restrictions
  const isGameMode = session.mode === 'game';
  const notMyTurn = isGameMode && !viewer.isActiveSeat;
  const notParticipant = isGameMode && !viewer.isParticipant;
  const turnLocked = notMyTurn || notParticipant;
  const canMove = isHumanTurn && !turnLocked && !isBusy;

  // Bot state
  const waitingOnBot =
    !session.state.isTerminal && currentPlayerSlot >= 0 && !isHumanTurn;
  return {
    currentPlayerSlot,
    currentAutoPlay,
    isGameMode,
    canMove,
    waitingOnBot,
  };
}

/** A move the viewer sent: its squares, and the position it was made at. */
interface SentMove {
  readonly from: string;
  readonly to: string;
  readonly atTurn: number;
}

/**
 * The turn as the board shows it. While the viewer's move is in flight (the
 * server answers once the bot has replied too) the move is shown as made:
 * the other side is to move, and a bot there is thinking.
 */
function shownTurn(
  turn: ReturnType<typeof turnState>,
  session: SessionPayload,
): ReturnType<typeof turnState> {
  const slot = turn.currentPlayerSlot;
  if (slot < 0) return turn;
  const next = slot === SLOT_OF.white ? SLOT_OF.black : SLOT_OF.white;
  const entry = session.players.find(player => player.slot === next);
  return {
    ...turn,
    currentPlayerSlot: next,
    currentAutoPlay: Boolean(entry?.autoPlay),
    waitingOnBot: entry?.controller !== HUMAN,
    canMove: false,
  };
}

/** One side's strip: who plays it, its material and whether it is to move. */
function SideStrip({
  side,
  session,
  viewer,
  material,
  turn,
}: {
  side: Side;
  session: SessionPayload;
  viewer: ChessViewer;
  material: SideMaterial;
  turn: ReturnType<typeof turnState>;
}): JSX.Element {
  const slot = SLOT_OF[side];
  const entry = session.players.find(player => player.slot === slot);
  let word: string | null = null;
  // A paused game waits on a failed controller: no one is to move.
  if (
    !session.state.isTerminal &&
    session.status !== 'paused' &&
    turn.currentPlayerSlot === slot
  ) {
    word = turn.waitingOnBot
      ? turn.currentAutoPlay
        ? 'thinking…'
        : 'waiting for a bot step'
      : 'to move';
  }
  return (
    <ChessPlayerStrip
      side={side}
      name={playerName(entry, session.availableControllers)}
      isViewer={turn.isGameMode && viewer.isParticipant && viewer.seat === slot}
      material={material}
      turn={word}
    />
  );
}

function ChessContent({useSession}: ChessContentProps): JSX.Element {
  const {session, isBusy, submitAction, viewer} = useSession();
  const moves = session.state.moveHints;
  const [sent, setSent] = useState<SentMove | null>(null);
  const atTurn = session.state.turnNumber;
  const onSent = useCallback(
    (from: string, to: string) => {
      setSent({from, to, atTurn});
    },
    [atTurn],
  );
  const handlers = useMoveHandlers(moves, submitAction, onSent);
  // In flight until the next position arrives or the request ends.
  const inFlight = sent !== null && sent.atTurn === atTurn && isBusy;

  // Board settings
  // The side drawn at the bottom follows View › Perspective in analysis and
  // the viewer's own seat in a game (useChessPerspective, in the renderer).
  const [boardOrientation] = useBoardOrientation();
  const [boardTheme] = useBoardTheme();
  const [showCoordinates] = useShowCoordinates();

  const liveTurn = turnState(session, viewer, isBusy);
  const turn = inFlight ? shownTurn(liveTurn, session) : liveTurn;
  const fen = session.state.text || START_FEN;

  // Get the board configuration
  const {config} = useChessBoard({
    fen,
    // Disable moves while promotion dialog is open
    canMove: turn.canMove && !handlers.pendingPromotion,
    // The last move, as the server reads it from the history; the viewer's
    // own while it is in flight.
    lastMove: inFlight
      ? {from: sent.from, to: sent.to}
      : session.state.lastMove,
    check: session.state.check,
    moves,
    orientation: boardOrientation,
    showCoordinates,
    onMove: handlers.handleMove,
    onPromotionMove: handlers.handlePromotionMove,
  });

  const material = materialDiff(fen);
  const result = session.state.isTerminal
    ? gameResult(fen, session.state.outcome, session.state.returns)
    : null;
  // The far side on top, as the board is turned (the viewer's seat at the
  // bottom in a game).
  const bottomSide: Side = boardOrientation;
  const strip = (side: Side) => (
    <SideStrip
      side={side}
      session={session}
      viewer={viewer}
      material={material[side]}
      turn={turn}
    />
  );

  return (
    <>
      <div className="chess-board-wrapper">
        <ChessBoard
          config={config}
          className={`chess-board theme-${boardTheme}`}
          pendingPromotion={handlers.pendingPromotion}
          onPromotion={handlers.handlePromotionComplete}
          onCancelPromotion={handlers.handleCancelPromotion}
          resync={handlers.boardResync}
          above={strip(other(bottomSide))}
          below={strip(bottomSide)}
        />
      </div>
      <div className="chess-panel__status" data-testid="chess-status">
        {result ? (
          <>
            <span className="chess-panel__score">{result.score}</span> ·{' '}
            {result.text}
          </>
        ) : (
          <>Move {Math.floor(session.state.turnNumber / 2) + 1}</>
        )}
      </div>
    </>
  );
}

/** The chess Gameplay panel's content: the board, its strips and status. */
export function ChessGameplay({
  useSession,
}: {
  useSession: UseSession;
}): JSX.Element {
  return (
    <div className="chess-panel gameplay-shell">
      <ChessContent useSession={useSession} />
    </div>
  );
}
