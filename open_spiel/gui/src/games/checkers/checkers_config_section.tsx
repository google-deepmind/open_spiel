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

// The checkers board's settings: the hooks the board reads them with, and
// the View panel's Board section that sets them.

import {type ChangeEvent, type JSX, useCallback} from 'react';

import {useCheckersSetting} from './checkers_settings.ts';

import './checkers_settings.css';

type Setting<T> = [T, (value: T | ((prev: T) => T)) => void];

/** The board theme setting's key. */
export const BOARD_THEME_KEY = 'checkers.boardTheme';
/** The coordinates setting's key. */
export const SHOW_COORDINATES_KEY = 'checkers.showCoordinates';
/** The legal-move highlight setting's key. */
export const HIGHLIGHT_LEGAL_MOVES_KEY = 'checkers.highlightLegalMoves';

/** A board theme: Classic Wood or Modern. */
export type BoardTheme = 'classic' | 'modern';

function isBoardTheme(value: string): value is BoardTheme {
  return value === 'classic' || value === 'modern';
}

/**
 * Hook to get/set the board theme; 'classic' by default.
 */
export function useBoardTheme(): Setting<BoardTheme> {
  return useCheckersSetting<BoardTheme>(BOARD_THEME_KEY, 'classic');
}

/**
 * Hook to get/set whether to show board coordinates; on by default, as
 * chess's are, since the History names squares ("g5f4").
 */
export function useShowCoordinates(): Setting<boolean> {
  return useCheckersSetting<boolean>(SHOW_COORDINATES_KEY, true);
}

/**
 * Hook to get/set whether to highlight legal moves.
 */
export function useHighlightLegalMoves(): Setting<boolean> {
  return useCheckersSetting<boolean>(HIGHLIGHT_LEGAL_MOVES_KEY, true);
}

/**
 * Checkers-specific settings section for the View panel: display-only
 * client settings (board theme, coordinates, legal-move highlight). The
 * game's parameters (`rows`, `columns`) are set in Setup.
 */
export function CheckersConfigSection(): JSX.Element {
  const [boardTheme, setBoardTheme] = useBoardTheme();
  const [showCoordinates, setShowCoordinates] = useShowCoordinates();
  const [highlightLegalMoves, setHighlightLegalMoves] =
    useHighlightLegalMoves();

  const handleThemeChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (isBoardTheme(e.target.value)) setBoardTheme(e.target.value);
    },
    [setBoardTheme],
  );

  const handleCoordinatesChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setShowCoordinates(e.target.checked);
    },
    [setShowCoordinates],
  );

  const handleHighlightChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setHighlightLegalMoves(e.target.checked);
    },
    [setHighlightLegalMoves],
  );

  return (
    <section className="checkers-config-section">
      <div className="ogs-config-header">
        <h3>Board</h3>
        <p>Customize how the checkers board is displayed.</p>
      </div>
      <label className="ogs-config-field">
        <span>Theme</span>
        <select value={boardTheme} onChange={handleThemeChange}>
          <option value="classic">Classic Wood</option>
          <option value="modern">Modern</option>
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
      <label className="ogs-config-field ogs-config-field--checkbox">
        <span>Highlight legal moves</span>
        <input
          type="checkbox"
          checked={highlightLegalMoves}
          onChange={handleHighlightChange}
        />
      </label>
    </section>
  );
}
