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

import {type JSX, useId} from 'react';

import type {Viewer} from '../contract/contract.ts';
import {viewerOptions} from '../game/game_defaults.ts';
import {viewerParameter} from '../session/client.ts';

import './view.css';

/** The properties of Perspective. */
export interface PerspectiveProps {
  /** Who the game is shown to. */
  readonly viewer: Viewer;
  /** The game's seats, which the perspectives are offered for. */
  readonly numSeats: number;
  /** Shows the game to `viewer`. */
  readonly onChange: (viewer: Viewer) => void;
}

/**
 * Whose view of the game the page shows: a seat, the oracle, who sees
 * everything, or a spectator, who sees only what is public. A perspective is
 * what the board shows, not who may change the game.
 */
export function Perspective({
  viewer,
  numSeats,
  onChange,
}: PerspectiveProps): JSX.Element {
  const id = useId();
  const options = viewerOptions(numSeats);
  return (
    <section className="view-card" aria-label="Perspective">
      <div className="view-field">
        <label htmlFor={id}>Perspective</label>
        <select
          id={id}
          className="view-field__select"
          value={viewerParameter(viewer)}
          onChange={event => {
            const chosen = options.find(
              option => option.value === event.currentTarget.value,
            );
            if (chosen === undefined) {
              throw new Error(`No perspective "${event.currentTarget.value}".`);
            }
            onChange(chosen.viewer);
          }}
        >
          <option value="" disabled>
            Select a seat
          </option>
          {options.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </section>
  );
}
