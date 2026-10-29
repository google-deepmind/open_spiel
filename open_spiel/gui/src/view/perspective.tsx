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
import {
  perspectiveLock,
  sessionModeOf,
  useAnalysisViewSeat,
} from '../game/bottom_seat.ts';

import './view.css';

/** The properties of Perspective. */
export interface PerspectiveProps {
  /** Who the page is shown to: the oracle in analysis, else a seat. */
  readonly viewer: Viewer;
  /** The game's seats, which the select offers. */
  readonly numSeats: number;
}

/**
 * View › Perspective. In analysis the viewer sits
 * at any seat: the select chooses the seat the board draws at the bottom
 * and fetches nothing, and the board's own "Show all cards" reveals the
 * other hands. In a game the perspective is the viewer's own seat, shown as
 * a fact with the reason it is locked.
 */
export function Perspective({viewer, numSeats}: PerspectiveProps): JSX.Element {
  const id = useId();
  const [analysisViewSeat, setAnalysisViewSeat] = useAnalysisViewSeat();
  const isParticipant = viewer.kind === 'seat' && viewer.seat !== null;
  const lock = perspectiveLock(
    {mode: sessionModeOf(viewer)},
    {seat: viewer.seat, isParticipant},
  );
  const options = Array.from({length: numSeats}, (_, seat) => ({
    value: seat,
    label:
      seat === viewer.seat
        ? `Seat ${String(seat)} (you)`
        : `Seat ${String(seat)}`,
  }));
  // Until a seat is chosen the select names one, the viewer's
  // own seat when it has one (the address's viewSeat is already the chosen
  // seat here), else the first. The oracle holds no seat, so analysis rests
  // on Seat 0. Chess keeps its View from until a seat is
  // chosen, so it may rest on Seat 0 drawn from White.
  const preferred =
    options.find(option => option.value === viewer.seat)?.value ??
    options[0]?.value ??
    null;
  const value =
    analysisViewSeat !== null
      ? String(analysisViewSeat)
      : preferred !== null
        ? String(preferred)
        : '';
  return (
    <section className="view-card" aria-label="Perspective">
      {lock.switchable ? (
        <>
          <div className="view-field">
            <label htmlFor={id}>Perspective</label>
            <select
              id={id}
              className="view-field__select"
              data-testid="perspective-select"
              value={value}
              disabled={options.length === 0}
              onChange={event => {
                const seat = Number(event.currentTarget.value);
                if (!Number.isInteger(seat)) {
                  throw new Error(
                    `No seat "${event.currentTarget.value}" to sit at.`,
                  );
                }
                setAnalysisViewSeat(seat);
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
          {options.length === 0 && (
            <p className="view-note">
              Seat options will appear once players or hands are detected in the
              current session.
            </p>
          )}
        </>
      ) : (
        <LockedPerspective
          seat={isParticipant ? viewer.seat : null}
          reason={lock.reason}
        />
      )}
    </section>
  );
}

// A game's Perspective: the viewer's own seat, as a fact, with the reason it
// cannot change.
function LockedPerspective({
  seat,
  reason,
}: {
  readonly seat: number | null;
  readonly reason: string | null;
}): JSX.Element {
  return (
    <>
      <div className="view-field">
        <span>Perspective</span>
        <span data-testid="perspective-locked">
          {seat !== null ? `Seat ${String(seat)} (you)` : 'Spectating'}
        </span>
      </div>
      <p className="view-note" data-testid="perspective-lock-reason">
        {reason}
      </p>
    </>
  );
}
