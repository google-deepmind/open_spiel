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

// How the board is played: a piece is picked up by a press and moved by a
// press on one of its marked squares, or dragged there. A piece moved stays
// where it was put while the move is with the server; the board shows the
// server's position again when the position changes, when the board becomes
// movable again with the move not made (refused, or a promotion cancelled),
// and when its owner asks (a resync with the board movable).

import {
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useState,
} from 'react';

import {
  type BoardConfig,
  readPlacement,
  squareAt,
} from './chess_board_config.ts';

/** A move made on the board and not yet answered by the server. */
interface Pending {
  readonly orig: string;
  readonly dest: string;
  /** The position it was made at. */
  readonly fen: string;
  /** Whether it slides (a press) or lands where it was dropped (a drag). */
  readonly slides: boolean;
  /** Whether the board has since become unmovable (the move is sent). */
  readonly sent: boolean;
}

/** A piece being dragged: where it started, and the pointer, in pixels. */
export interface Drag {
  readonly orig: string;
  readonly startX: number;
  readonly startY: number;
  readonly x: number;
  readonly y: number;
  /** Whether the pointer has moved far enough to be a drag, not a press. */
  readonly active: boolean;
  /** Whether a release without a drag puts the piece down again. */
  readonly deselects: boolean;
}

/** The board's input state and its pointer handlers. */
export interface BoardInput {
  readonly selected: string | null;
  readonly dests: readonly string[];
  readonly drag: Drag | null;
  /** The move on show while it is with the server, or null. */
  readonly pending: {
    readonly orig: string;
    readonly dest: string;
    readonly slides: boolean;
  } | null;
  readonly onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
  readonly onPointerMove: (event: ReactPointerEvent<HTMLElement>) => void;
  readonly onPointerUp: (event: ReactPointerEvent<HTMLElement>) => void;
  readonly onPointerCancel: () => void;
}

/** How far, in pixels, a press moves before it is a drag. */
const DRAG_THRESHOLD = 4;

function movableAt(config: BoardConfig, square: string): boolean {
  const {color} = config.movable;
  if (config.viewOnly || color === undefined) return false;
  const letter = readPlacement(config.fen).get(square);
  if (letter === undefined) return false;
  const white = letter === letter.toUpperCase();
  return white === (color === 'white');
}

function destsOf(config: BoardConfig, square: string | null): string[] {
  if (square === null || config.viewOnly) return [];
  return [...(config.movable.dests?.get(square) ?? [])];
}

/** The input's state and how to change it, as the handlers use them. */
interface InputState {
  readonly config: BoardConfig;
  readonly boardRef: RefObject<HTMLElement | null>;
  readonly size: number;
  readonly selected: string | null;
  readonly drag: Drag | null;
  readonly setSelected: (square: string | null) => void;
  readonly setDrag: (drag: Drag | null) => void;
  readonly setPending: (pending: Pending | null) => void;
}

/** Where a pointer event is on the board: pixels and the square. */
function pointAt(
  state: InputState,
  event: ReactPointerEvent<HTMLElement>,
): {x: number; y: number; square: string | null} | null {
  const box = state.boardRef.current?.getBoundingClientRect();
  if (box === undefined || state.size <= 0 || box.width <= 0) return null;
  const x = event.clientX - box.left;
  const y = event.clientY - box.top;
  const step = box.width / 8;
  return {
    x: (x * state.size) / box.width,
    y: (y * state.size) / box.height,
    square: squareAt(
      Math.floor(x / step),
      Math.floor(y / step),
      state.config.orientation,
    ),
  };
}

/** Plays `orig` to `dest`, keeping the piece there while it is sent. */
function playMove(
  state: InputState,
  orig: string,
  dest: string,
  slides: boolean,
): void {
  const {config} = state;
  state.setPending({orig, dest, fen: config.fen, slides, sent: false});
  state.setSelected(null);
  state.setDrag(null);
  config.movable.events.after(orig, dest);
}

function onDown(
  state: InputState,
  event: ReactPointerEvent<HTMLElement>,
): void {
  const {config, selected} = state;
  if (event.button !== 0 || config.viewOnly) return;
  const at = pointAt(state, event);
  const square = at?.square ?? null;
  if (at === null || square === null) return;
  if (selected !== null && destsOf(config, selected).includes(square)) {
    playMove(state, selected, square, true);
    return;
  }
  if (!movableAt(config, square)) {
    state.setSelected(null);
    return;
  }
  event.currentTarget.setPointerCapture(event.pointerId);
  state.setSelected(square);
  if (!config.draggable.enabled) return;
  state.setDrag({
    orig: square,
    startX: at.x,
    startY: at.y,
    x: at.x,
    y: at.y,
    active: false,
    deselects: selected === square,
  });
}

function onMove(
  state: InputState,
  event: ReactPointerEvent<HTMLElement>,
): void {
  const {drag} = state;
  if (drag === null) return;
  const at = pointAt(state, event);
  if (at === null) return;
  const far =
    Math.hypot(at.x - drag.startX, at.y - drag.startY) > DRAG_THRESHOLD;
  state.setDrag({...drag, x: at.x, y: at.y, active: drag.active || far});
}

function onUp(state: InputState, event: ReactPointerEvent<HTMLElement>): void {
  const {drag, config} = state;
  if (drag === null) return;
  const at = pointAt(state, event);
  state.setDrag(null);
  if (!drag.active) {
    if (drag.deselects) state.setSelected(null);
    return;
  }
  const dest = at?.square ?? null;
  if (dest !== null && destsOf(config, drag.orig).includes(dest)) {
    playMove(state, drag.orig, dest, false);
  } else if (dest !== drag.orig) {
    state.setSelected(null);
  }
}

/**
 * The move with the server as the position now stands: dropped once the
 * position changes, and once the board is movable again after it was sent.
 */
function pendingNow(
  pending: Pending | null,
  config: BoardConfig,
): Pending | null {
  if (pending?.fen !== config.fen) return null;
  if (config.viewOnly && !pending.sent) return {...pending, sent: true};
  if (!config.viewOnly && pending.sent) return null;
  return pending;
}

/**
 * The board's input: selection, drag and the move with the server, for
 * `config`, the board element `boardRef` and its `size` in pixels.
 */
export function useBoardInput(
  config: BoardConfig,
  boardRef: RefObject<HTMLElement | null>,
  size: number,
  resync: number,
): BoardInput {
  const [selected, setSelected] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [seenResync, setSeenResync] = useState(resync);
  const [seenFen, setSeenFen] = useState(config.fen);

  // Adjustments to the position shown, made while rendering.
  if (seenFen !== config.fen) {
    setSeenFen(config.fen);
    setSelected(null);
    setDrag(null);
  }
  if (seenResync !== resync) {
    setSeenResync(resync);
    if (!config.viewOnly) setPending(null);
  }
  const current = pendingNow(pending, config);
  if (current !== pending) setPending(current);
  if (config.viewOnly && (selected !== null || drag !== null)) {
    setSelected(null);
    setDrag(null);
  }

  const state: InputState = {
    config,
    boardRef,
    size,
    selected,
    drag,
    setSelected,
    setDrag,
    setPending,
  };
  return {
    selected,
    dests: config.movable.showDests ? destsOf(config, selected) : [],
    drag,
    pending: current,
    onPointerDown: event => {
      onDown(state, event);
    },
    onPointerMove: event => {
      onMove(state, event);
    },
    onPointerUp: event => {
      onUp(state, event);
    },
    onPointerCancel: () => {
      setDrag(null);
    },
  };
}
