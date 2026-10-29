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

import type {JSX} from 'react';

import {
  type Cell,
  type GridMove,
  type GridView,
  sameCell,
} from './grid_view.ts';
import {type PieceSet, pieceSetFor} from './piece_sets.ts';
import {type Press, useGridSelection} from './use_grid_selection.ts';

import './grid.css';

/** What the shared board is given. */
export interface GridBoardProps {
  readonly view: GridView;
  /** The legal actions placed on the board, from `readGridMoves`. */
  readonly moves: readonly GridMove[];
  /** Plays an action; null when the viewer cannot play. */
  readonly play: ((action: number) => void) | null;
  /** The legal actions' labels, by id, to name a choice between moves. */
  readonly labels: ReadonlyMap<number, string>;
}

/**
 * The shared board: a grid of cells holding pieces. How the viewer moves
 * follows the view's mode: in `claim` a press on a cell plays the action
 * that claims it, in `drop` a press anywhere in a column drops into it, and
 * in `move` a press selects a piece that can move, its targets are marked,
 * and a press on a target plays the move. Several moves between the same two
 * cells (a promotion) are offered as a choice by their labels.
 */
export function GridBoard({
  view,
  moves,
  play,
  labels,
}: GridBoardProps): JSX.Element {
  const {selected, choice, targets, pressFor, playMove} = useGridSelection(
    view,
    play === null ? [] : moves,
    play,
  );
  const pieces = pieceSetFor(view.pieceSet);
  return (
    <div className="grid-board">
      <div
        className="grid-board__cells"
        role="group"
        aria-label="Board"
        data-mode={view.mode}
        style={{gridTemplateColumns: `repeat(${String(view.columns)}, 1fr)`}}
      >
        {order(view.rows, view.flipped).flatMap(row =>
          order(view.columns, view.flipped).map(column => {
            const cell: Cell = [row, column];
            return (
              <GridCell
                key={`${String(row)}-${String(column)}`}
                view={view}
                cell={cell}
                pieces={pieces}
                selected={selected !== null && sameCell(selected, cell)}
                target={targets.some(
                  move => move.kind === 'move' && sameCell(move.to, cell),
                )}
                press={pressFor(cell)}
              />
            );
          }),
        )}
      </div>
      {choice !== null && (
        <div
          className="grid-board__choice"
          role="group"
          aria-label="Choose a move"
        >
          {choice.map(move => (
            <button
              type="button"
              key={move.action}
              onClick={() => {
                playMove(move);
              }}
            >
              {labels.get(move.action) ?? `Action ${String(move.action)}`}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// The indices 0 to count - 1, reversed when the board is flipped.
function order(count: number, flipped: boolean): number[] {
  const indices = Array.from({length: count}, (_, index) => index);
  return flipped ? indices.reverse() : indices;
}

// A cell's name: its labels when the view gives them ("e4"), else its row
// and column counted from one.
function cellName(view: GridView, [row, column]: Cell): string {
  const rowLabel = view.rowLabels?.[row];
  const columnLabel = view.columnLabels?.[column];
  if (rowLabel !== undefined && columnLabel !== undefined) {
    return `${columnLabel}${rowLabel}`;
  }
  return `row ${String(row + 1)}, column ${String(column + 1)}`;
}

function GridCell({
  view,
  cell,
  pieces,
  selected,
  target,
  press,
}: {
  readonly view: GridView;
  readonly cell: Cell;
  readonly pieces: PieceSet;
  readonly selected: boolean;
  readonly target: boolean;
  readonly press: Press;
}): JSX.Element {
  const [row, column] = cell;
  const code = view.cells[row]?.[column] ?? null;
  const piece = code === null ? null : pieces.piece(code);
  const name = `${cellName(view, cell)}: ${piece?.name ?? 'empty'}`;
  const attributes = {
    className: 'grid-board__cell',
    'data-square':
      view.checkered && (row + column) % 2 === 1 ? 'dark' : 'light',
    'data-highlight': view.highlights.some(lit => sameCell(lit, cell))
      ? 'true'
      : undefined,
    'data-alert': view.alerts.some(alert => sameCell(alert, cell))
      ? 'true'
      : undefined,
    'data-selected': selected ? 'true' : undefined,
    'data-target': target ? 'true' : undefined,
  };
  const content =
    piece === null ? null : (
      <span
        className="grid-board__piece"
        data-tone={piece.tone}
        aria-hidden="true"
      >
        {piece.glyph}
      </span>
    );
  if (press === undefined) {
    return (
      <div {...attributes} role="img" aria-label={name}>
        {content}
      </div>
    );
  }
  return (
    <button
      type="button"
      {...attributes}
      aria-label={name}
      title={name}
      onClick={press}
    >
      {content}
    </button>
  );
}
