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

import type {HistoryEntry} from '../contract/contract.ts';

/**
 * One row of the history: the positions after plies `first` to `last`. Ply 0
 * is the initial position, and a row spans several plies only when it stands
 * for a run of actions withheld from the viewer.
 */
export interface HistoryRow {
  readonly first: number;
  readonly last: number;
  readonly label: string;
}

function hiddenLabel(seat: number | null): string {
  return seat === null
    ? 'Hidden chance outcome'
    : `Hidden action of seat ${String(seat)}`;
}

/** What the viewer may read of `entry`: its label, or the public label of a
 *  withheld action the game makes public (a card discarded face up); null
 *  when the action is hidden. */
export function readableLabel(entry: HistoryEntry): string | null {
  return entry.label ?? entry.public_label;
}

/**
 * The rows for `history`, after a row for the initial position. A run of
 * hidden actions by the same seat, or of hidden chance outcomes, is one
 * row: a deal of 52 hidden cards is one row, not 52. A withheld action the
 * game makes public has its own row, under its public label.
 */
export function historyRows(
  history: readonly HistoryEntry[],
): readonly HistoryRow[] {
  const rows: HistoryRow[] = [{first: 0, last: 0, label: 'Initial'}];
  history.forEach((entry, index) => {
    const ply = index + 1;
    const label = readableLabel(entry);
    if (label !== null) {
      rows.push({first: ply, last: ply, label});
      return;
    }
    const previous = history[index - 1];
    const run = rows.at(-1);
    if (
      run !== undefined &&
      previous !== undefined &&
      readableLabel(previous) === null &&
      previous.seat === entry.seat
    ) {
      rows[rows.length - 1] = {...run, last: ply};
    } else {
      rows.push({first: ply, last: ply, label: hiddenLabel(entry.seat)});
    }
  });
  return rows;
}

/** A row as the history shows it, a turn column and a label column. */
export interface RowText {
  /** "#3", or "#1–5" for a run. */
  readonly turn: string;
  /** "e4", or "Hidden … ×5" for a run. */
  readonly label: string;
}

/** The two columns of `row`: "#3" and "e4", or "#1–5" and "Hidden … ×5". */
export function rowText({first, last, label}: HistoryRow): RowText {
  if (first === last) return {turn: `#${String(first)}`, label};
  const count = String(last - first + 1);
  return {
    turn: `#${String(first)}–${String(last)}`,
    label: `${label} ×${count}`,
  };
}
