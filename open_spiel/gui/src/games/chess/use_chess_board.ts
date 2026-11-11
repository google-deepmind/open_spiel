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

import {useCallback, useMemo} from 'react';

import {scaleMs, useAnimationScale} from '../../motion/index.ts';
import type {BoardConfig} from './chess_board_config.ts';
import type {MoveHint} from './chess_session.ts';

/** A piece's slide, ms at Normal speed; the viewer's Animation speed scales
 *  it, and Instant or reduced motion turns it off. */
export const CHESS_MOVE_MS = 250;

/** What the board's configuration is built from. */
export interface UseChessBoardProps {
  fen: string;
  canMove: boolean;
  lastMove?: {from: string; to: string} | null;
  /** The side in check, as the server reads the position. */
  check: 'white' | 'black' | null;
  /** The legal moves' squares, as the server hints them. */
  moves: readonly MoveHint[];
  /** Board orientation - which side is at the bottom */
  orientation?: 'white' | 'black';
  /** Whether to show coordinates on the board */
  showCoordinates?: boolean;
  /** Called for non-promotion moves */
  onMove: (from: string, to: string) => void;
  /** Called when a pawn reaches the promotion rank - must show promotion dialog */
  onPromotionMove: (from: string, to: string) => void;
}

/** The board's configuration. */
export interface UseChessBoardReturn {
  config: BoardConfig;
}

/** Each square's destinations, and the side to move, at `fen`. */
function legalDests(
  fen: string,
  canMove: boolean,
  moves: readonly MoveHint[],
): {dests: Map<string, string[]>; turnColor: 'white' | 'black'} {
  const destsMap = new Map<string, string[]>();
  // The side to move is the FEN's second field.
  const turn: 'white' | 'black' =
    fen.trim().split(/\s+/)[1] === 'b' ? 'black' : 'white';
  if (canMove) {
    for (const move of moves) {
      const existing = destsMap.get(move.from);
      if (existing === undefined) destsMap.set(move.from, [move.to]);
      else if (!existing.includes(move.to)) existing.push(move.to);
    }
  }
  return {dests: destsMap, turnColor: turn};
}

/**
 * Hook that generates the board configuration from game state.
 * Handles legal move computation, turn management, and move callbacks.
 */
export function useChessBoard({
  fen,
  canMove,
  lastMove,
  check,
  moves,
  orientation = 'white',
  showCoordinates = true,
  onMove,
  onPromotionMove,
}: UseChessBoardProps): UseChessBoardReturn {
  // Compute legal moves and turn color from FEN
  const {dests, turnColor} = useMemo(
    () => legalDests(fen, canMove, moves),
    [fen, canMove, moves],
  );

  // Handle move completion
  const handleMove = useCallback(
    (orig: string, dest: string) => {
      // Check if this is a promotion move
      const move = moves.find(m => m.from === orig && m.to === dest);
      if (move?.promotion !== undefined && move.promotion !== null) {
        // This is a promotion - delegate to promotion handler
        onPromotionMove(orig, dest);
        return;
      }
      // Regular move
      onMove(orig, dest);
    },
    [moves, onMove, onPromotionMove],
  );

  const moveMs = scaleMs(CHESS_MOVE_MS, useAnimationScale());

  // Build the board configuration
  const config: BoardConfig = useMemo(
    () => ({
      fen,
      orientation,
      turnColor,
      check: check ?? undefined,
      lastMove: lastMove ? [lastMove.from, lastMove.to] : undefined,
      viewOnly: !canMove,
      highlight: {
        lastMove: true,
        check: true,
      },
      animation: {
        enabled: moveMs > 0,
        duration: moveMs,
      },
      movable: {
        color: canMove ? turnColor : undefined,
        dests: canMove ? dests : undefined,
        showDests: true,
        events: {
          after: handleMove,
        },
      },
      draggable: {
        enabled: true,
        showGhost: true,
      },
      selectable: {
        enabled: true,
      },
      coordinates: showCoordinates,
    }),
    [
      fen,
      orientation,
      turnColor,
      check,
      lastMove,
      canMove,
      dests,
      handleMove,
      showCoordinates,
      moveMs,
    ],
  );

  return {config};
}
