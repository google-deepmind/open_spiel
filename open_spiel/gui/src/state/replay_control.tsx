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

import type {ChangeEvent, JSX} from 'react';
import {useEffect, useEffectEvent, useState} from 'react';

import type {ChanceMode, Frame} from '../contract/contract.ts';
import {replayStops} from './replay_stops.ts';

import './state.css';

/** The properties of a ReplayControl. */
export interface ReplayControlProps {
  /** The position shown, live or a replay. */
  frame: Frame;
  /** Whether the session has a request in flight. */
  busy: boolean;
  /**
   * How the game uses chance. Skip chance is offered in every mode but
   * `deterministic`.
   */
  chanceMode: ChanceMode;
  /**
   * Shows the position after `ply` plies, the session busy until it is shown;
   * the session's `goTo`.
   */
  onGoTo: (ply: number) => void;
}

const NOTHING_PLAYED = 'Nothing has been played yet.';
const SKIP_CHANCE_TITLE = 'Skip over chance actions (card deals, etc.)';

/**
 * A button that shows the position after `target` plies, or is unavailable
 * when `target` is null. `title` replaces the label as the tooltip.
 */
function StepButton({
  label,
  text,
  target,
  title,
  onGoTo,
}: {
  label: string;
  text: string;
  target: number | null;
  title: string | undefined;
  onGoTo: (ply: number) => void;
}): JSX.Element {
  return (
    <button
      type="button"
      className="replay__button"
      aria-label={label}
      title={title ?? label}
      aria-disabled={target === null}
      onClick={() => {
        if (target !== null) onGoTo(target);
      }}
    >
      {text}
    </button>
  );
}

/**
 * A slider over `stops`, at stop `current`, and the stop count beside it.
 * `onMove` receives the ply of the stop it is moved to.
 */
function ReplaySlider({
  stops,
  current,
  unavailable,
  onMove,
}: {
  stops: readonly number[];
  current: number;
  unavailable: boolean;
  onMove: (ply: number) => void;
}): JSX.Element {
  const last = stops.length - 1;
  const move = (event: ChangeEvent<HTMLInputElement>): void => {
    const ply = stops[Number(event.currentTarget.value)];
    if (ply === undefined) throw new RangeError('The slider left its range.');
    onMove(ply);
  };
  return (
    <>
      <input
        type="range"
        className="replay__slider"
        aria-label="Replay position"
        aria-valuetext={`step ${String(current)} of ${String(last)}`}
        aria-disabled={unavailable}
        min={0}
        max={last}
        value={current}
        onChange={move}
      />
      <span className="replay__count" aria-hidden="true">
        {current} / {last}
      </span>
    </>
  );
}

/**
 * Sends slider moves to `onGoTo`, one request at a time: a move while `busy`
 * is held, a later move replaces it, and the move held is sent once the
 * session is idle. Returns where the slider was moved to, until the session
 * has shown it, and the function that moves it.
 */
function useSliderMoves(
  busy: boolean,
  onGoTo: (ply: number) => void,
): [number | null, (ply: number) => void] {
  const [moved, setMoved] = useState<number | null>(null);
  const [held, setHeld] = useState<number | null>(null);
  // A held move on its way out: set when the session falls idle, sent by the
  // effect below, and cleared once the session is busy with it.
  const [release, setRelease] = useState<{ply: number} | null>(null);
  // Adjusted during render, as the session's busy flag changes.
  if (!busy && held !== null) {
    setRelease({ply: held});
    setHeld(null);
  }
  if (busy && release !== null) setRelease(null);
  if (!busy && moved !== null && held === null && release === null) {
    setMoved(null);
  }
  const send = useEffectEvent((ply: number) => {
    onGoTo(ply);
  });
  useEffect(() => {
    if (release !== null) send(release.ply);
  }, [release]);

  const move = (ply: number): void => {
    setMoved(ply);
    if (busy) setHeld(ply);
    else onGoTo(ply);
  };
  return [moved, move];
}

/**
 * Steps through the live game's positions: one row of buttons to the first,
 * previous, next and last position around a slider over the same stops,
 * and, in a game with chance, a checkbox that skips the positions where
 * chance acts. The row of the position shown in the history says where the
 * control stands.
 *
 * The buttons are marked aria-disabled rather than disabled, so that a
 * button that becomes unavailable keeps the keyboard focus, as Next does at
 * the live position or while its own request is in flight. A slider move
 * is sent at once when the session is idle; while it is busy, the control
 * keeps only the latest move and sends it when the session is idle again, so
 * a drag sends one request per round trip, not one per stop it passes. Until
 * the session has shown it, the slider and count show where the slider was
 * moved to.
 */
export function ReplayControl({
  frame,
  busy,
  chanceMode,
  onGoTo,
}: ReplayControlProps): JSX.Element {
  const [skipChance, setSkipChance] = useState(true);
  const [moved, move] = useSliderMoves(busy, onGoTo);

  const live = frame.num_plies;
  if (frame.history.length !== live) {
    throw new RangeError(
      `A frame of ${String(live)} plies carries ${String(frame.history.length)} history entries.`,
    );
  }
  const shown = moved !== null && moved <= live ? moved : frame.ply;
  const stops = replayStops(frame.history, shown, skipChance);
  const current = stops.indexOf(shown);
  const last = stops.length - 1;
  const reason = live === 0 ? NOTHING_PLAYED : undefined;
  // The ply of stop `index`, or null when it is the current stop, out of
  // range, or a request is in flight.
  const plyOfStop = (index: number): number | null =>
    busy || index === current ? null : (stops[index] ?? null);
  const buttonProps = {title: reason, onGoTo};

  return (
    <>
      <div className="replay" role="group" aria-label="Replay" title={reason}>
        <StepButton
          label="Go to start"
          text="|<"
          target={plyOfStop(0)}
          {...buttonProps}
        />
        <StepButton
          label="Previous move"
          text="<"
          target={plyOfStop(current - 1)}
          {...buttonProps}
        />
        <ReplaySlider
          stops={stops}
          current={current}
          unavailable={reason !== undefined}
          onMove={move}
        />
        <StepButton
          label="Next move"
          text=">"
          target={plyOfStop(current + 1)}
          {...buttonProps}
        />
        <StepButton
          label="Go to end"
          text=">|"
          target={plyOfStop(last)}
          {...buttonProps}
        />
      </div>
      {chanceMode !== 'deterministic' && (
        <label className="replay__skip-chance" title={SKIP_CHANCE_TITLE}>
          <input
            type="checkbox"
            checked={skipChance}
            disabled={busy}
            onChange={event => {
              setSkipChance(event.currentTarget.checked);
            }}
          />
          Skip chance
        </label>
      )}
    </>
  );
}
