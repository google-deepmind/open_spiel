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

// The shared board's view and moves, as an adapter sends them. The view is
// the presentation's "grid" object:
//
//   {"rows": 8, "columns": 8, "cells": [["r", null, ...], ...],
//    "piece_set": "chess", "mode": "move", "highlights": [[6, 4]],
//    "alerts": [[7, 4]], "checkered": true, "flipped": false,
//    "row_labels": ["8", ...], "column_labels": ["a", ...]}
//
// `cells` is row by row from the top, each a piece code or null; the last
// five fields are optional. `alerts` are cells drawn as a warning, such as a
// king in check. A move is hinted by mode: {"from": [r, c],
// "to": [r, c]} moves a piece, {"column": c} drops into a column, and
// {"cell": [r, c]} claims a cell.

import type {Presentation} from '../contract/contract.ts';
import {
  array,
  boolean,
  field,
  integer,
  nullable,
  optionalField,
  record,
  string,
  StructError,
} from '../renderers/struct_reader.ts';

/** A cell: its row from the top and its column from the left. */
export type Cell = readonly [number, number];

/** How a viewer moves on the board. */
export type GridMode = 'move' | 'drop' | 'claim' | 'none';

/** A board as the adapter presents it. */
export interface GridView {
  readonly rows: number;
  readonly columns: number;
  readonly cells: ReadonlyArray<ReadonlyArray<string | null>>;
  readonly pieceSet: string;
  readonly mode: GridMode;
  readonly highlights: readonly Cell[];
  readonly alerts: readonly Cell[];
  readonly checkered: boolean;
  readonly flipped: boolean;
  readonly rowLabels: readonly string[] | null;
  readonly columnLabels: readonly string[] | null;
}

/** A legal action placed on the board. */
export type GridMove =
  | {
      readonly kind: 'move';
      readonly action: number;
      readonly from: Cell;
      readonly to: Cell;
    }
  | {readonly kind: 'drop'; readonly action: number; readonly column: number}
  | {readonly kind: 'claim'; readonly action: number; readonly cell: Cell};

const MODES: readonly GridMode[] = ['move', 'drop', 'claim', 'none'];

function isMode(text: string): text is GridMode {
  return (MODES as readonly string[]).includes(text);
}

function readCell(value: unknown, path: string): Cell {
  const [row, column, ...rest] = array(value, path, integer);
  if (row === undefined || column === undefined || rest.length > 0) {
    throw new StructError(`${path} is not a [row, column] pair.`);
  }
  return [row, column];
}

function readLabels(value: unknown, path: string): string[] {
  return array(value, path, string);
}

/** Reads a grid view; throws a `StructError` naming any field at fault. */
export function readGridView(value: unknown, path = 'grid'): GridView {
  const rows = field(value, path, 'rows', integer);
  const columns = field(value, path, 'columns', integer);
  const cells = field(value, path, 'cells', (rowsValue, rowsPath) =>
    array(rowsValue, rowsPath, (row, rowPath) =>
      array(row, rowPath, (cell, cellPath) => nullable(cell, cellPath, string)),
    ),
  );
  if (cells.length !== rows || cells.some(row => row.length !== columns)) {
    throw new StructError(
      `${path}.cells is not ${String(rows)} rows of ${String(columns)}.`,
    );
  }
  const mode = field(value, path, 'mode', string);
  if (!isMode(mode)) {
    throw new StructError(
      `${path}.mode is "${mode}", not one of ${MODES.join(', ')}.`,
    );
  }
  return {
    rows,
    columns,
    cells,
    pieceSet: field(value, path, 'piece_set', string),
    mode,
    highlights: field(value, path, 'highlights', (list, listPath) =>
      array(list, listPath, readCell),
    ),
    alerts: optionalField(
      value,
      path,
      'alerts',
      (list, listPath) => array(list, listPath, readCell),
      [],
    ),
    checkered: optionalField(value, path, 'checkered', boolean, false),
    flipped: optionalField(value, path, 'flipped', boolean, false),
    rowLabels: optionalField(value, path, 'row_labels', readLabels, null),
    columnLabels: optionalField(value, path, 'column_labels', readLabels, null),
  };
}

function readMove(
  action: number,
  hint: unknown,
  path: string,
): GridMove | null {
  if (hint === null) return null;
  const fields = record(hint, path);
  if ('from' in fields) {
    return {
      kind: 'move',
      action,
      from: field(hint, path, 'from', readCell),
      to: field(hint, path, 'to', readCell),
    };
  }
  if ('column' in fields) {
    return {kind: 'drop', action, column: field(hint, path, 'column', integer)};
  }
  if ('cell' in fields) {
    return {kind: 'claim', action, cell: field(hint, path, 'cell', readCell)};
  }
  return null;
}

/** The legal actions of `presentation` that are placed on the board. */
export function readGridMoves(presentation: Presentation): GridMove[] {
  const moves: GridMove[] = [];
  for (const [index, {id, hint}] of (presentation.actions ?? []).entries()) {
    const move = readMove(id, hint, `actions[${String(index)}].hint`);
    if (move !== null) moves.push(move);
  }
  return moves;
}

/** Whether two cells are the same. */
export function sameCell(a: Cell, b: Cell): boolean {
  return a[0] === b[0] && a[1] === b[1];
}
