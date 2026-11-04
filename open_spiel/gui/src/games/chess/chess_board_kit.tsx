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

// What a chess board drawn in place of the built-in one (chess_board_slot.ts)
// needs besides its props: the panel's layout (the status line over the
// largest square that fits), the viewer's View › Board settings, the line
// saying whose move it is and how the game ended, the choice of a promotion
// piece, and the pieces of the chosen set as images. Nothing here depends on
// a board library. src/INTERFACES.md, "Replacing the chess board", is the
// guide.

import {
  type JSX,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import {useAnimationScale} from '../../motion/use_animation_speed.ts';
import type {ChessFrame} from './chess_board_slot.ts';
import {
  type BoardTheme,
  useBoardTheme,
  usePieceSet,
  useShowCoordinates,
} from './chess_config_section.tsx';
import {ChessPieceArt} from './chess_pieces.tsx';
import {SIDE_NAME, gameResult} from './chess_table.ts';
import {getPieceSet} from './pieces/piece_sets.ts';
import {
  PIECE_COLORS,
  PIECE_ROLES,
  type PieceColor,
  type PieceRole,
} from './pieces/piece_types.ts';

/** The viewer's View › Board and Animation settings, as a board reads them. */
export interface ChessBoardSettings {
  /** The squares' colours: the tokens `--chess-<theme>-light`, `-dark`. */
  readonly theme: BoardTheme;
  /** Whether to draw the files and ranks. */
  readonly showCoordinates: boolean;
  /** The piece set's id in the registry (pieces/piece_sets.ts). */
  readonly pieceSet: string;
  /** The animation scale: multiply a duration by it; 0 draws no motion. */
  readonly animationScale: number;
}

/**
 * The viewer's View › Board settings (View from reaches a board as
 * `viewer.orientation`); re-renders when one changes.
 */
export function useChessBoardSettings(): ChessBoardSettings {
  const [theme] = useBoardTheme();
  const [showCoordinates] = useShowCoordinates();
  const [pieceSet] = usePieceSet();
  const animationScale = useAnimationScale();
  return useMemo(
    () => ({theme, showCoordinates, pieceSet, animationScale}),
    [theme, showCoordinates, pieceSet, animationScale],
  );
}

/** The light and dark squares' colours for `theme`, as CSS values. */
export function squareColors(theme: BoardTheme): {
  readonly light: string;
  readonly dark: string;
} {
  return {
    light: `var(--chess-${theme}-light)`,
    dark: `var(--chess-${theme}-dark)`,
  };
}

/**
 * The line under the built-in board, for a board that draws its own status:
 * the move number and the side to move (and check), or, once the game is
 * over, the score and how it ended ("1-0 · White wins by checkmate").
 */
export function chessStatusText(frame: ChessFrame): string {
  const {state} = frame.session;
  if (frame.isTerminal) {
    const result = gameResult(frame.fen, state.outcome, state.returns);
    return `${result.score} · ${result.text}`;
  }
  const move = `Move ${String(Math.floor(frame.ply / 2) + 1)}`;
  const toMove = `${SIDE_NAME[frame.turn]} to move`;
  return frame.check === null
    ? `${move} · ${toMove}`
    : `${move} · ${toMove} · Check`;
}

/** `chessStatusText(frame)` as a status line, announced when it changes. */
export function ChessStatus({
  frame,
}: {
  readonly frame: ChessFrame;
}): JSX.Element {
  return (
    <p className="chess-kit-status" role="status">
      {chessStatusText(frame)}
    </p>
  );
}

/** The pieces a pawn may become, as FEN letters. */
export type PromotionLetter = 'q' | 'r' | 'b' | 'n';

const PROMOTIONS: ReadonlyArray<{
  readonly letter: PromotionLetter;
  readonly role: PieceRole;
}> = [
  {letter: 'q', role: 'queen'},
  {letter: 'r', role: 'rook'},
  {letter: 'b', role: 'bishop'},
  {letter: 'n', role: 'knight'},
];

/** The properties of the promotion choice. */
export interface PromotionPickerProps {
  /** The promoting pawn's colour. */
  readonly color: PieceColor;
  /** Called with the piece chosen, a FEN letter. */
  readonly onSelect: (letter: PromotionLetter) => void;
  /** Called when the viewer dismisses the choice. */
  readonly onCancel: () => void;
}

/**
 * The choice of the piece a pawn becomes: a dimmed layer over its nearest
 * positioned ancestor (the board) with one button per piece, drawn in the
 * viewer's piece set. A press outside the buttons cancels.
 */
export function PromotionPicker({
  color,
  onSelect,
  onCancel,
}: PromotionPickerProps): JSX.Element {
  return (
    <div className="chess-kit-promotion">
      <div
        className="promotion-overlay"
        data-testid="promotion-cancel"
        onClick={onCancel}
      />
      <div
        className="chess-kit-promotion__choices"
        role="group"
        aria-label="Promote to"
      >
        {PROMOTIONS.map(({letter, role}) => (
          <button
            key={letter}
            type="button"
            className="chess-kit-promotion__choice"
            aria-label={`Promote to ${role}`}
            onClick={() => {
              onSelect(letter);
            }}
          >
            <span className="promotion-circle">
              <span className="promotion-piece">
                <ChessPieceArt role={role} color={color} />
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Piece `role` of colour `color` in the set `pieceSet` as an image URL.
 *
 * @throws Error when there is no set `pieceSet`, or it offers no images.
 */
export function pieceImageUrl(
  pieceSet: string,
  role: PieceRole,
  color: PieceColor,
): string {
  const {label, imageUrl} = getPieceSet(pieceSet);
  if (imageUrl === undefined) {
    throw new Error(
      `Piece set "${label}" draws no images; pick one that does.`,
    );
  }
  return imageUrl({role, color});
}

/**
 * CSS giving each of the twelve pieces of `pieceSet` its image as a
 * background, for a board drawing a piece as an element with the classes of
 * its colour and role (chessground's `piece.white.king`) under `scope`.
 */
export function pieceSetCss(pieceSet: string, scope: string): string {
  return PIECE_COLORS.flatMap(color =>
    PIECE_ROLES.map(role => {
      const url = pieceImageUrl(pieceSet, role, color);
      return `${scope} piece.${color}.${role}{background-image:url("${url}")}`;
    }),
  ).join('\n');
}

/** The properties of ChessBoardFrame. */
export interface ChessBoardFrameProps {
  readonly frame: ChessFrame;
  /** The board, given the side of its square in pixels (0 before layout). */
  readonly children: (size: number) => ReactNode;
}

/**
 * The Gameplay panel's layout for a replacement board, as the built-in board
 * has it: the status line (ChessStatus), then the largest square that fits
 * in the rest of the panel, resized with it. The square is positioned, so a
 * PromotionPicker drawn in it covers the board.
 */
export function ChessBoardFrame({
  frame,
  children,
}: ChessBoardFrameProps): JSX.Element {
  const space = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(0);
  useEffect(() => {
    const element = space.current;
    if (element === null) return;
    const update = () => {
      setSize(
        Math.max(
          0,
          Math.floor(Math.min(element.clientWidth, element.clientHeight)),
        ),
      );
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, []);
  return (
    <div className="chess-kit-frame">
      <ChessStatus frame={frame} />
      <div ref={space} className="chess-kit-frame__space">
        <div
          className="chess-kit-frame__board"
          style={{width: size, height: size}}
        >
          {children(size)}
        </div>
      </div>
    </div>
  );
}
