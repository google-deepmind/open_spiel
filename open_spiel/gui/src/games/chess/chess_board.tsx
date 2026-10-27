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
  type JSX,
  type ReactNode,
  type RefObject,
  useEffect,
  useRef,
  useState,
} from 'react';

import type {BoardConfig} from './chess_board_config.ts';
import {ChessBoardSurface} from './chess_board_surface.tsx';
import type {PieceColor} from './chess_pieces.tsx';
import {PromotionDialog, type PromotionChoice} from './promotion_dialog.tsx';

/** A pawn moved to its last rank, waiting for the piece it becomes. */
export interface PendingPromotion {
  from: string;
  to: string;
  color: PieceColor;
}

/** The properties of the board and its players' strips. */
export interface ChessBoardProps {
  config: BoardConfig;
  className?: string;
  /** Called when a promotion move needs to be completed */
  onPromotion?: (from: string, to: string, piece: PromotionChoice) => void;
  /** Current pending promotion (if any) */
  pendingPromotion?: PendingPromotion | null;
  /** Called to cancel promotion */
  onCancelPromotion?: () => void;
  /** Bump when a move attempt settles (sent, refused, unresolved or a
   *  promotion cancelled): the board then shows the server's position if
   *  its pieces differ (chess_board_input.ts). */
  resync?: number;
  /** Drawn above and below the board at the board's width (the players'
   *  strips, the far side on top); the board gives up their height. */
  above?: ReactNode;
  below?: ReactNode;
}

/**
 * The board, sized to the largest square its container holds once the
 * players' strips have their height, with the strips above and below at
 * its width and the promotion choice over it.
 */
export function ChessBoard({
  config,
  className,
  onPromotion,
  pendingPromotion,
  onCancelPromotion,
  resync = 0,
  above,
  below,
}: ChessBoardProps): JSX.Element {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const aboveRef = useRef<HTMLDivElement>(null);
  const belowRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<number>(0);

  // Set up resize observer on wrapper
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (wrapper === null) return;
    // Compute square size from container dimensions: the LAYOUT size, not a
    // transformed one.
    const updateSize = () => {
      // The strips keep their height; the board takes the rest.
      const strips =
        (aboveRef.current?.offsetHeight ?? 0) +
        (belowRef.current?.offsetHeight ?? 0);
      // Use the smaller dimension, floor to avoid sub-pixel issues
      setSize(
        Math.max(
          0,
          Math.floor(
            Math.min(wrapper.clientWidth, wrapper.clientHeight - strips),
          ),
        ),
      );
    };
    updateSize();
    const resizeObserver = new ResizeObserver(updateSize);
    resizeObserver.observe(wrapper);
    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  const handlePromotionSelect = (piece: PromotionChoice) => {
    if (pendingPromotion && onPromotion) {
      onPromotion(pendingPromotion.from, pendingPromotion.to, piece);
    }
  };

  const handlePromotionCancel = () => {
    onCancelPromotion?.();
  };

  return (
    <div ref={wrapperRef} className={`${className ?? ''} chess-board-frame`}>
      <Strip stripRef={aboveRef} size={size}>
        {above}
      </Strip>
      {size > 0 && (
        <div
          className="chess-board__square"
          style={{width: size, height: size}}
        >
          <ChessBoardSurface config={config} size={size} resync={resync} />
          {pendingPromotion && (
            <PromotionDialog
              file={pendingPromotion.to[0] ?? 'a'}
              color={pendingPromotion.color}
              boardSize={size}
              onSelect={handlePromotionSelect}
              onCancel={handlePromotionCancel}
            />
          )}
        </div>
      )}
      <Strip stripRef={belowRef} size={size}>
        {below}
      </Strip>
    </div>
  );
}

/** A players' strip at the board's width; nothing when it has no content. */
function Strip({
  stripRef,
  size,
  children,
}: {
  stripRef: RefObject<HTMLDivElement | null>;
  size: number;
  children: ReactNode;
}): JSX.Element | null {
  if (children === undefined || children === null) return null;
  return (
    <div
      ref={stripRef}
      className="chess-board__strip"
      style={size > 0 ? {width: size} : undefined}
    >
      {children}
    </div>
  );
}
