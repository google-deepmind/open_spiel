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

import {NoFrame, Note} from '../actions/note.tsx';
import type {
  BotOutput,
  Frame,
  HistoryEntry,
  PolicyEntry,
} from '../contract/contract.ts';

import './bot_output.css';

/** A bot's output and the ply it is about. */
export interface ShownOutput {
  /** The ply's number in the History, from 1. */
  readonly ply: number;
  /**
   * The ply's history entry, which played `entry.action` (always present when
   * `bot_output` is non-null).
   */
  readonly entry: HistoryEntry;
  readonly output: BotOutput;
}

/**
 * The bot output to show at `frame`'s position: that of the latest ply up to
 * the position whose entry has one, so stepping through the History shows
 * each bot move's output until the next. Null when no such ply reached it.
 */
export function shownOutput(frame: Frame): ShownOutput | null {
  for (let index = frame.ply - 1; index >= 0; index--) {
    const entry = frame.history[index];
    if (entry !== undefined && entry.bot_output !== null) {
      return {ply: index + 1, entry, output: entry.bot_output};
    }
  }
  return null;
}

/**
 * `policy`, most probable first, ties by action id. Entries whose ids are
 * withheld keep the server's order among themselves.
 */
export function sortedPolicy(
  policy: readonly PolicyEntry[],
): readonly PolicyEntry[] {
  return [...policy].sort(
    (a, b) => b.prob - a.prob || (a.action ?? 0) - (b.action ?? 0),
  );
}

/** A probability as a percentage, to one decimal place. */
export function percent(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

/** A value estimate, signed, to three decimal places. */
export function signedValue(value: number): string {
  return `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
}

/**
 * The Bot output panel: what a bot said about its move at the position the
 * History is on, as the server lets the viewer see it: its policy over the
 * legal actions, most probable first with the move played marked, its value
 * estimate and what part of it decided. A bot's output comes only with its
 * move in full: to an omniscient viewer and to the bot's own seat. In a game
 * that hides information no other viewer receives it, since it tells of the
 * bot's hand, and the panel says so.
 */
export function BotOutputView({frame}: {frame: Frame | null}): JSX.Element {
  if (frame === null) return <NoFrame />;
  const shown = shownOutput(frame);
  if (shown === null) {
    // Said only where a bot plays: with no bot, nothing is withheld.
    const botPlays = frame.seats.some(seat => seat.controller !== 'human');
    return (
      <Note>
        {frame.omniscient || !botPlays
          ? 'No bot output up to this position.'
          : "No bot output up to this position. In a game that hides information, a bot's output goes only to its own seat."}
      </Note>
    );
  }
  const {ply, entry, output} = shown;
  const seat = String(entry.seat ?? '');
  return (
    <section className="bot-output" aria-label="Bot output">
      <p className="bot-output__move">
        #{String(ply)} · Seat {seat} played{' '}
        <span className="bot-output__played">
          {entry.label ?? entry.public_label ?? 'a hidden move'}
        </span>
      </p>
      <dl className="bot-output__facts">
        <div className="bot-output__fact">
          <dt>Decided by</dt>
          <dd>{output.decided_by ?? 'Not given'}</dd>
        </div>
        <div className="bot-output__fact">
          <dt>Value</dt>
          <dd>
            {output.value === null ? 'Not given' : signedValue(output.value)}
          </dd>
        </div>
      </dl>
      {output.policy.length === 0 ? (
        <Note>The bot gave no policy for this move.</Note>
      ) : (
        <ol className="bot-output__policy" aria-label="Policy">
          {sortedPolicy(output.policy).map((item, index) => {
            const chosen = item.action !== null && item.action === entry.action;
            return (
              <li
                key={item.action ?? `withheld-${String(index)}`}
                className="bot-output__row"
                data-chosen={chosen ? 'true' : undefined}
                aria-current={chosen ? 'true' : undefined}
              >
                <span className="bot-output__label">{item.label}</span>
                <span className="bot-output__prob">{percent(item.prob)}</span>
                <span className="bot-output__bar" aria-hidden="true">
                  <span
                    className="bot-output__fill"
                    style={{width: percent(item.prob)}}
                  />
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
