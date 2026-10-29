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

// The checkers Gameplay panel's state: the last turn and its slide, whose
// turn it is and whether the viewer may move, and the man picked up with
// where it may go.

import {useCallback, useMemo, useState} from 'react';

import {
  type PieceType,
  getPiecePlayer,
  isHumanController,
  squareKey,
} from './checkers_board_parts.ts';
import {type LastTurn, type SquareCoord, lastTurn} from './checkers_model.ts';
import type {
  CheckersState,
  LegalAction,
  SessionContextValue,
} from './checkers_session.ts';

/** The last turn, at rest until the next one, and the leg that slides. */
export interface LastTurnMarks {
  turnError: string | null;
  lastPath: Set<string>;
  lastCaptured: Set<string>;
  slideLeg: LastTurn['lastLeg'];
  historyLength: number;
}

/**
 * The last turn, at rest until the next one: every square the piece stood
 * on, and the squares of the men it took. A history the model cannot read
 * is said in the notice slot, not thrown: a throw here would blank the
 * whole board, not just the last-move marks.
 */
export function useLastTurn(state: CheckersState): LastTurnMarks {
  const {turn, turnError} = useMemo(() => {
    try {
      return {
        turn: lastTurn(state.historyActionStrings, state.historyPlayers),
        turnError: null,
      };
    } catch (e) {
      return {
        turn: null,
        turnError: `Last move not shown: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  }, [state.historyActionStrings, state.historyPlayers]);
  const lastPath = useMemo(
    () => new Set((turn?.path ?? []).map(s => squareKey(s.row, s.col))),
    [turn],
  );
  const lastCaptured = useMemo(
    () => new Set((turn?.captured ?? []).map(s => squareKey(s.row, s.col))),
    [turn],
  );
  // Slide a move that arrived while the board is open (the history grew
  // since the last render), not the position it opened on. The slide
  // stays keyed to that length until the history changes again, so an
  // unrelated re-render does not cut it short; a shorter history (a new
  // game, a step back) slides nothing and resets the comparison.
  const historyLength = state.history.length;
  const [seenLength, setSeenLength] = useState(historyLength);
  const [slideAt, setSlideAt] = useState<number | null>(null);
  if (historyLength !== seenLength) {
    setSlideAt(historyLength > seenLength ? historyLength : null);
    setSeenLength(historyLength);
  }
  const slideLeg = slideAt === historyLength ? (turn?.lastLeg ?? null) : null;
  return {turnError, lastPath, lastCaptured, slideLeg, historyLength};
}

/** Whose turn it is, and what the viewer may do on it. */
export interface CheckersTurn {
  currentPlayerSlot: number;
  currentAutoPlay: boolean;
  waitingOnBot: boolean;
  isGameMode: boolean;
  notMyTurn: boolean;
  notParticipant: boolean;
  canInteract: boolean;
}

/** Whose turn it is, from the session and the viewer. */
export function checkersTurn(s: SessionContextValue): CheckersTurn {
  const {session, isBusy, viewer} = s;
  const {state, players} = session;
  const currentPlayerSlot = state.currentPlayer;
  const currentPlayerEntry =
    currentPlayerSlot >= 0
      ? players.find(player => player.slot === currentPlayerSlot)
      : undefined;
  const currentController = currentPlayerEntry?.controller;
  const currentAutoPlay = Boolean(currentPlayerEntry?.autoPlay);
  const isHumanTurn =
    currentPlayerSlot >= 0
      ? isHumanController(currentController, currentPlayerSlot)
      : false;
  const waitingOnBot =
    !state.isTerminal && currentPlayerSlot >= 0 && !isHumanTurn;

  const isGameMode = session.mode === 'game';
  const notMyTurn = isGameMode && !viewer.isActiveSeat;
  const notParticipant = isGameMode && !viewer.isParticipant;
  const turnLocked = notMyTurn || notParticipant;

  const canInteract =
    !isBusy && !state.isTerminal && !waitingOnBot && !turnLocked;
  return {
    currentPlayerSlot,
    currentAutoPlay,
    waitingOnBot,
    isGameMode,
    notMyTurn,
    notParticipant,
    canInteract,
  };
}

/** The man picked up, where it may go, and the board's click. */
export interface CheckersMoves {
  selectedSquare: SquareCoord | null;
  movableSources: Set<string>;
  destinationMap: Map<string, number>;
  /** Every legal move is a jump: the side to move must capture. */
  mustCapture: boolean;
  /** Why the last press picked nothing up (a man with no move); else null. */
  notice: string | null;
  onSquareClick: (row: number, col: number) => void;
}

/** Whether every one of `legalActions` is a jump (a capture is forced). */
export function captureIsForced(legalActions: readonly LegalAction[]): boolean {
  return (
    legalActions.length > 0 &&
    legalActions.every(action => action.actionType === 'jump')
  );
}

function destinationsOf(
  selectedSquare: SquareCoord | null,
  legalActions: readonly LegalAction[],
): Map<string, number> {
  const map = new Map<string, number>();
  if (!selectedSquare) return map;
  const selectedMoves = legalActions.filter(
    a =>
      a.fromSquare?.row === selectedSquare.row &&
      a.fromSquare.col === selectedSquare.col,
  );
  for (const move of selectedMoves) {
    if (move.toSquare) {
      map.set(squareKey(move.toSquare.row, move.toSquare.col), move.action);
    }
  }
  return map;
}

/** A man is picked up with a click, and played with a click on a landing. */
export function useCheckersMoves(
  state: CheckersState,
  board: PieceType[][],
  turn: CheckersTurn,
  submitAction: (action: number) => Promise<void>,
): CheckersMoves {
  const [selectedSquare, setSelectedSquare] = useState<SquareCoord | null>(
    null,
  );
  const movableSources = useMemo(() => {
    const sources = state.movableSources ?? [];
    return new Set(sources.map(s => squareKey(s.row, s.col)));
  }, [state.movableSources]);
  const destinationMap = useMemo(
    () => destinationsOf(selectedSquare, state.legalActions),
    [selectedSquare, state.legalActions],
  );
  const mustCapture = useMemo(
    () => captureIsForced(state.legalActions),
    [state.legalActions],
  );
  // The notice belongs to the position it was given at.
  const [notice, setNotice] = useState<{at: number; text: string} | null>(null);
  const at = state.history.length;
  const {canInteract, currentPlayerSlot} = turn;

  const onSquareClick = useCallback(
    (row: number, col: number) => {
      setNotice(null);
      if (!canInteract) return;

      const key = squareKey(row, col);
      const piece = board[row]?.[col];

      // If clicking a destination square while a piece is selected
      const actionId = destinationMap.get(key);
      if (selectedSquare && actionId !== undefined) {
        void submitAction(actionId);
        setSelectedSquare(null);
        return;
      }

      // If clicking on a movable piece
      if (piece && movableSources.has(key)) {
        const piecePlayer = getPiecePlayer(piece);
        if (piecePlayer === currentPlayerSlot) {
          setSelectedSquare({row, col});
          return;
        }
      }

      // A man of the side to move that cannot move: say why, so the press
      // is seen to have landed.
      if (piece && getPiecePlayer(piece) === currentPlayerSlot) {
        setNotice({
          at,
          text: mustCapture
            ? 'This piece cannot move; a capture is forced.'
            : 'This piece has no legal move.',
        });
      }

      // Otherwise, deselect
      setSelectedSquare(null);
    },
    [
      canInteract,
      board,
      selectedSquare,
      destinationMap,
      movableSources,
      currentPlayerSlot,
      submitAction,
      mustCapture,
      at,
    ],
  );
  return {
    selectedSquare,
    movableSources,
    destinationMap,
    mustCapture,
    notice: notice !== null && notice.at === at ? notice.text : null,
    onSquareClick,
  };
}
