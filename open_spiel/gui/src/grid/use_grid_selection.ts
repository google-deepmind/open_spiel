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

import {useState} from 'react';

import {
  type Cell,
  type GridMove,
  type GridView,
  sameCell,
} from './grid_view.ts';

/** What pressing a cell does; undefined when it does nothing. */
export type Press = (() => void) | undefined;

/** The board's selection, and what a press on each cell does. */
export interface GridSelection {
  /** The piece chosen to move, in `move` mode. */
  readonly selected: Cell | null;
  /** Several moves between the same two cells, to choose among. */
  readonly choice: readonly GridMove[] | null;
  /** The moves of the selected piece. */
  readonly targets: readonly GridMove[];
  readonly pressFor: (cell: Cell) => Press;
  readonly playMove: (move: GridMove) => void;
}

/**
 * Click-select then click-move: the board's selection state and the press
 * handler of each cell, for `moves`, the actions the viewer may play now.
 */
export function useGridSelection(
  view: GridView,
  moves: readonly GridMove[],
  play: ((action: number) => void) | null,
): GridSelection {
  const [selected, setSelected] = useState<Cell | null>(null);
  const [choice, setChoice] = useState<readonly GridMove[] | null>(null);
  const targets =
    selected === null
      ? []
      : moves.filter(
          move => move.kind === 'move' && sameCell(move.from, selected),
        );

  const playMove = (move: GridMove): void => {
    setSelected(null);
    setChoice(null);
    play?.(move.action);
  };

  const pressFor = (cell: Cell): Press => {
    const options = cellOptions(view.mode, moves, cell);
    if (view.mode !== 'move') {
      const [only] = options;
      return only === undefined
        ? undefined
        : () => {
            playMove(only);
          };
    }
    const toCell = targets.filter(
      move => move.kind === 'move' && sameCell(move.to, cell),
    );
    if (toCell.length > 0) {
      return () => {
        if (toCell.length === 1 && toCell[0] !== undefined) playMove(toCell[0]);
        else setChoice(toCell);
      };
    }
    if (options.length > 0) {
      return () => {
        setChoice(null);
        setSelected(
          selected !== null && sameCell(selected, cell) ? null : cell,
        );
      };
    }
    if (selected === null) return undefined;
    return () => {
      setSelected(null);
      setChoice(null);
    };
  };

  return {selected, choice, targets, pressFor, playMove};
}

// The moves a press on `cell` starts or plays, by mode.
function cellOptions(
  mode: GridView['mode'],
  moves: readonly GridMove[],
  cell: Cell,
): GridMove[] {
  switch (mode) {
    case 'claim':
      return moves.filter(
        move => move.kind === 'claim' && sameCell(move.cell, cell),
      );
    case 'drop':
      return moves.filter(
        move => move.kind === 'drop' && move.column === cell[1],
      );
    case 'move':
      return moves.filter(
        move => move.kind === 'move' && sameCell(move.from, cell),
      );
    case 'none':
      return [];
    default: {
      const unreachable: never = mode;
      throw new Error(`Unhandled mode: ${JSON.stringify(unreachable)}`);
    }
  }
}
