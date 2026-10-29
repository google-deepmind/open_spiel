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

import {type ChangeEvent, type JSX, useCallback} from 'react';

import {useChessFlag, useChessSetting} from './chess_settings.ts';
import {
  DEFAULT_PIECE_SET,
  getPieceSet,
  pieceSetIds,
} from './pieces/piece_sets.ts';

import './chess_settings.css';

/** Which side of the board is drawn at the bottom. */
export type BoardOrientation = 'white' | 'black';
/** The board's square colours. */
export type BoardTheme = 'brown' | 'blue' | 'green';

/** The board orientation setting's key. */
export const BOARD_ORIENTATION_KEY = 'chess.boardOrientation';
/** The board theme setting's key. */
export const BOARD_THEME_KEY = 'chess.boardTheme';
/** The coordinates setting's key. */
export const SHOW_COORDINATES_KEY = 'chess.showCoordinates';
/** The piece set setting's key. */
export const PIECE_SET_KEY = 'chess.pieceSet';

const ORIENTATIONS: readonly BoardOrientation[] = ['white', 'black'];
const THEMES: readonly BoardTheme[] = ['brown', 'blue', 'green'];

/**
 * Hook to get/set the board orientation setting.
 */
export function useBoardOrientation(): [
  BoardOrientation,
  (value: BoardOrientation) => void,
] {
  return useChessSetting(BOARD_ORIENTATION_KEY, ORIENTATIONS, 'white');
}

/**
 * Hook to get/set the board color theme.
 */
export function useBoardTheme(): [BoardTheme, (value: BoardTheme) => void] {
  return useChessSetting(BOARD_THEME_KEY, THEMES, 'blue');
}

/**
 * Hook to get/set whether to show board coordinates.
 */
export function useShowCoordinates(): [boolean, (value: boolean) => void] {
  return useChessFlag(SHOW_COORDINATES_KEY, true);
}

/**
 * Hook to get/set the piece set the pieces are drawn with, by its id in the
 * piece-set registry (pieces/piece_sets.ts).
 */
export function usePieceSet(): [string, (value: string) => void] {
  return useChessSetting(PIECE_SET_KEY, pieceSetIds(), DEFAULT_PIECE_SET);
}

function isOrientation(value: string): value is BoardOrientation {
  return ORIENTATIONS.some(orientation => orientation === value);
}

function isTheme(value: string): value is BoardTheme {
  return THEMES.some(theme => theme === value);
}

/**
 * Chess-specific settings section for the View panel: every control here
 * is a display-only client setting (board orientation, theme, pieces,
 * coordinates) with no OpenSpiel param equivalent.
 */
export function ChessConfigSection(): JSX.Element {
  const [orientation, setOrientation] = useBoardOrientation();
  const [theme, setTheme] = useBoardTheme();
  const [showCoordinates, setShowCoordinates] = useShowCoordinates();
  const [pieceSet, setPieceSet] = usePieceSet();

  const handleOrientationChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (isOrientation(e.target.value)) setOrientation(e.target.value);
    },
    [setOrientation],
  );

  const handleThemeChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (isTheme(e.target.value)) setTheme(e.target.value);
    },
    [setTheme],
  );

  const handlePieceSetChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      setPieceSet(e.target.value);
    },
    [setPieceSet],
  );

  const handleCoordinatesChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setShowCoordinates(e.target.checked);
    },
    [setShowCoordinates],
  );

  return (
    <section className="ogs-config-section chess-config-section">
      <div className="ogs-config-header">
        <h3>Board</h3>
        <p>Customize how the chess board is displayed.</p>
      </div>
      <label className="ogs-config-field">
        <span>View from</span>
        <select value={orientation} onChange={handleOrientationChange}>
          <option value="white">White&apos;s perspective</option>
          <option value="black">Black&apos;s perspective</option>
        </select>
      </label>
      <label className="ogs-config-field">
        <span>Theme</span>
        <select value={theme} onChange={handleThemeChange}>
          <option value="brown">Brown</option>
          <option value="blue">Blue</option>
          <option value="green">Green</option>
        </select>
      </label>
      <label className="ogs-config-field">
        <span>Pieces</span>
        <select value={pieceSet} onChange={handlePieceSetChange}>
          {pieceSetIds().map(id => (
            <option key={id} value={id}>
              {getPieceSet(id).label}
            </option>
          ))}
        </select>
      </label>
      <label className="ogs-config-field ogs-config-field--checkbox">
        <span>Show coordinates</span>
        <input
          type="checkbox"
          checked={showCoordinates}
          onChange={handleCoordinatesChange}
        />
      </label>
    </section>
  );
}
