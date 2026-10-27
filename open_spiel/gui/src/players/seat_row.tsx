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
import {useId, useState} from 'react';

import type {
  ControllerFailure,
  Frame,
  SeatConfig,
  SeatView,
} from '../contract/contract.ts';
import {preparingText} from './preparing.ts';
import {HUMAN, controllerNamed, savedConfig} from './seat_draft.ts';
import {SeatEditor} from './seat_editor.tsx';
import {SeatSummary} from './seat_summary.tsx';
import {useSeatDraft} from './use_seat_draft.ts';

/** The properties of a SeatRow. */
export interface SeatRowProps {
  readonly seat: SeatView;
  /** The frame shown, whose session lists the controllers. */
  readonly frame: Frame;
  /** Whether the viewer may change who plays the seat. */
  readonly editable: boolean;
  readonly busy: boolean;
  readonly onSave: (config: SeatConfig) => void;
  /** Lets the seat's controller make one move. */
  readonly onStep: () => void;
}

/**
 * One seat as a card: its name, who plays it and, on the right, what can be
 * done with it. A controller that waits to be advanced can be stepped on its
 * turn, as can an autoplaying one left to move (after an undo, or past the
 * plies one request plays), and a failed one retried. While the server
 * builds a seat's controller the card says it is preparing, in the place of
 * the controller's name, and its Step waits, so that nothing moves when it is
 * ready. The settings stay collapsed until asked for; opening an editable
 * seat's settings starts its draft afresh from what the server holds.
 */
export function SeatRow({
  seat,
  frame,
  editable,
  busy,
  onSave,
  onStep,
}: SeatRowProps): JSX.Element {
  const id = useId();
  const controllers = frame.session.controllers;
  // A spectator's read-only settings start open; any other seat's closed.
  const [open, setOpen] = useState(!editable);
  // A move or a new game drops an unsaved draft.
  const moment = `${frame.session.id} ${String(frame.num_plies)}`;
  const state = useSeatDraft(seat, controllers, moment, onSave);
  const name = `Player ${String(seat.seat)}`;
  const {label} = controllerNamed(controllers, seat.controller);
  const preparing = seat.status === 'preparing';
  return (
    <li className="seat" aria-labelledby={`${id}-name`}>
      {seat.failure !== null && (
        <SeatFailure name={name} seat={seat} failure={seat.failure} />
      )}
      <div className="seat__row">
        <div className="seat__details">
          <span id={`${id}-name`} className="seat__name">
            {name}
          </span>
          <span
            className="seat__controller"
            aria-live="polite"
            title={preparing ? (seat.message ?? undefined) : undefined}
          >
            {preparing ? preparingText(controllers, seat) : label}
          </span>
        </div>
        <div className="seat__actions">
          {editable && (
            <SeatCommands
              seat={seat}
              frame={frame}
              busy={busy}
              onRetry={() => {
                onSave(savedConfig(seat, controllers));
                if (!seat.autoplay && seat.status === 'ready') onStep();
              }}
              onStep={onStep}
            />
          )}
          <SettingsToggle
            name={name}
            editable={editable}
            open={open}
            controls={`${id}-settings`}
            onToggle={() => {
              if (!open) state.reset();
              setOpen(!open);
            }}
          />
        </div>
      </div>
      <div id={`${id}-settings`} className="seat__settings" hidden={!open}>
        {open &&
          (editable ? (
            <SeatEditor
              name={name}
              controllers={controllers}
              state={state}
              busy={busy}
            />
          ) : (
            <SeatSummary seat={seat} controllers={controllers} />
          ))}
      </div>
    </li>
  );
}

interface SettingsToggleProps {
  /** The seat's name, such as "Player 0". */
  readonly name: string;
  readonly editable: boolean;
  readonly open: boolean;
  /** The id of the settings it shows and hides. */
  readonly controls: string;
  readonly onToggle: () => void;
}

// Shows and hides a seat's settings: a Settings button where the seat can be
// changed, and a chevron, named for the seat, where it cannot.
function SettingsToggle({
  name,
  editable,
  open,
  controls,
  onToggle,
}: SettingsToggleProps): JSX.Element {
  return (
    <button
      type="button"
      className={
        editable ? 'button button--ghost button--compact' : 'seat__chevron'
      }
      aria-expanded={open}
      aria-controls={controls}
      aria-label={
        editable ? undefined : `${open ? 'Hide' : 'Show'} settings for ${name}`
      }
      onClick={onToggle}
    >
      {editable
        ? open
          ? 'Hide settings'
          : 'Settings'
        : open
          ? '\u25B2'
          : '\u25BC'}
    </button>
  );
}

interface SeatCommandsProps {
  readonly seat: SeatView;
  readonly frame: Frame;
  readonly busy: boolean;
  readonly onRetry: () => void;
  readonly onStep: () => void;
}

// Retry for a failed seat on its turn, and for one whose controller could
// not be built at any time, which builds it again; Step for a controller
// that waits to be advanced, which it may be only on its turn at the live
// position once built, and for an autoplaying one on its turn there, which
// autoplay left to move.
function SeatCommands({
  seat,
  frame,
  busy,
  onRetry,
  onStep,
}: SeatCommandsProps): JSX.Element | null {
  if (seat.controller === HUMAN) return null;
  const onTurn =
    frame.ply === frame.num_plies &&
    frame.to_move.kind === 'seat' &&
    frame.to_move.seat === seat.seat;
  const failed = seat.failure !== null;
  const preparing = seat.status === 'preparing';
  const canStep = onTurn && frame.status === 'playing' && !preparing;
  return (
    <>
      {failed && (onTurn || seat.status === 'unavailable') && (
        <button
          type="button"
          className="button"
          disabled={busy}
          onClick={onRetry}
        >
          Retry
        </button>
      )}
      {(!seat.autoplay || onTurn) && !failed && (
        <button
          type="button"
          className="button"
          disabled={busy || !canStep}
          title={
            preparing
              ? preparingText(frame.session.controllers, seat)
              : canStep
                ? 'Step this player'
                : 'Not this player’s turn'
          }
          onClick={onStep}
        >
          Step
        </button>
      )}
    </>
  );
}

interface SeatFailureProps {
  /** The seat's name, such as "Player 0". */
  readonly name: string;
  readonly seat: SeatView;
  readonly failure: ControllerFailure;
}

// Why a seat stopped, above its row: a controller that failed in play, or
// one that could not be built, whose reason every viewer may read.
function SeatFailure({name, seat, failure}: SeatFailureProps): JSX.Element {
  const unavailable = seat.status === 'unavailable';
  const message = unavailable ? seat.message : failure.message;
  return (
    <div className="seat__failure" role="alert">
      <p>
        {unavailable
          ? `${name}’s controller could not be prepared (${failure.error}). Play is paused.`
          : `${name}’s controller stopped with ${failure.error}. Play is paused.`}
      </p>
      {message !== null && (
        <pre className="seat__failure-message">{message}</pre>
      )}
    </div>
  );
}
