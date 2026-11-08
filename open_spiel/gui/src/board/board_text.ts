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

import type {Frame} from '../contract/contract.ts';

/** The text a board shows, and what that text is. */
export interface BoardText {
  /** What the text is: "State", or a seat's observation or information state. */
  readonly title: string;
  /** The game's text, verbatim. */
  readonly text: string;
}

/**
 * The fullest text `frame` gives its viewer: the full state; else the seat's
 * information state, which recalls everything the seat has seen; else its
 * observation of the current position; null when the frame carries none.
 * The server has already withheld what the viewer may not see, so this only
 * chooses among the texts that arrived, and shows the full state only to a
 * viewer the frame calls omniscient.
 */
export function boardText(frame: Frame): BoardText | null {
  const {viewer} = frame;
  if (frame.omniscient && frame.state_text !== null) {
    return {title: 'State', text: frame.state_text};
  }
  if (viewer.kind !== 'seat') return null;
  const seat = `Seat ${String(viewer.seat)}`;
  if (frame.information_state_text !== null) {
    return {
      title: `${seat}'s information state`,
      text: frame.information_state_text,
    };
  }
  if (frame.observation_text !== null) {
    return {title: `${seat}'s observation`, text: frame.observation_text};
  }
  return null;
}

/**
 * Why `frame` carries no text for its viewer, as a sentence. The server sends
 * the full state to every omniscient viewer, the oracle among them, so a
 * frame without text is shown to a seat or to a spectator of a game that
 * hides information; an oracle never reaches here.
 */
export function missingTextReason({viewer}: Frame): string {
  return viewer.kind === 'seat'
    ? `The game gives seat ${String(viewer.seat)} no text.`
    : 'The game hides information, so a spectator is shown no text.';
}

// Fifteen significant digits, the most a double always round-trips: every
// integer is written exactly, and a sum such as 0.1 + 0.2 still reads as 0.3.
const RETURN = new Intl.NumberFormat('en', {
  maximumSignificantDigits: 15,
  useGrouping: false,
});

/** Each seat's return, as a sentence: "Returns: seat 0: 1, seat 1: -1." */
export function returnsSentence(returns: readonly number[]): string {
  const seats = returns.map(
    (value, seat) => `seat ${String(seat)}: ${RETURN.format(value)}`,
  );
  return `Returns: ${seats.join(', ')}.`;
}
