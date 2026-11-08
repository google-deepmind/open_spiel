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

import type {
  CatalogueEntry,
  CreateSession,
  Frame,
  ParameterSpec,
  SeatConfig,
} from '../contract/contract.ts';
import {nameLabel} from '../form/name_label.ts';

/** A game parameter's value, as the wire format carries it. */
export type ParameterValue = ParameterSpec['default'];

/** A game's parameter values, by name. */
export type ParameterValues = Readonly<Record<string, ParameterValue>>;

/**
 * What a parameter's control holds while it is edited: a checkbox's state,
 * or the text of a field or of the chosen option.
 */
export type Draft = boolean | string;

/** A draft read as a parameter value, or why it cannot be read as one. */
export type ParsedDraft =
  | {readonly ok: true; readonly value: ParameterValue}
  | {readonly ok: false; readonly problem: string};

// OpenSpiel's names for the parameter that sets the number of players.
const SEAT_COUNT_NAMES: ReadonlySet<string> = new Set([
  'players',
  'num_players',
  'numPlayers',
]);

/** The seat a new game gives each player it adds. */
export const HUMAN_SEAT: SeatConfig = {
  controller: 'human',
  config: {},
  autoplay: false,
};

/**
 * The name of `entry`'s parameter that sets its number of players, or null
 * when it has none. OpenSpiel names it `players`, `num_players` or
 * `numPlayers`; a game whose count follows from another parameter has none.
 * Of two such names, the one whose default is a seat count sets it.
 *
 * @throws Error when that leaves more than one.
 */
export function seatCountParameter(entry: CatalogueEntry): string | null {
  const named = entry.parameters.filter(spec =>
    SEAT_COUNT_NAMES.has(spec.name),
  );
  const counting = named.filter(spec => isSeatCount(entry, spec.default));
  const [found, ...others] = counting.length > 0 ? counting : named;
  if (found === undefined) return null;
  if (others.length > 0) {
    const names = [found, ...others].map(spec => spec.name).join(', ');
    throw new Error(`${entry.name} has two player counts: ${names}.`);
  }
  return found.name;
}

/**
 * The seat-count parameter as Game options offers it: a whole number from
 * the game's fewest seats to its most, defaulting to the seats the game has
 * with its defaults. Some games default the parameter itself to a value
 * outside that range, meaning "the game's own count", so the catalogue's
 * `num_players` is the default instead. Null when the game has no seat-count
 * parameter, a fixed number of seats, a seat count that is not a whole
 * number or a list, or no default seats because it cannot be loaded.
 */
export function seatCountSpec(entry: CatalogueEntry): ParameterSpec | null {
  const name = seatCountParameter(entry);
  const spec = entry.parameters.find(listed => listed.name === name);
  const {min_seats: fewest, max_seats: most, num_players: seats} = entry;
  if (
    spec === undefined ||
    fewest >= most ||
    seats === null ||
    spec.type !== 'int' ||
    spec.kind === 'choice'
  ) {
    return null;
  }
  return {
    ...spec,
    kind: 'range',
    default: seats,
    minimum: fewest,
    maximum: most,
  };
}

function isSeatCount(
  entry: CatalogueEntry,
  value: ParameterValue | undefined,
): value is number {
  return (
    typeof value === 'number' &&
    value >= entry.min_seats &&
    value <= entry.max_seats
  );
}

// The parameters a game offers once it is open, by game, where it offers
// fewer than it has: Go's komi, handicap and length are set when a game is
// created, and only its board size changes a game in play.
const SETUP_PARAMETERS: Readonly<Record<string, readonly string[]>> = {
  go: ['board_size'],
};

/**
 * The parameters the Setup panel offers for `entry`: its own, with its seat
 * count as `seatCountSpec` offers it, and less any its game sets only when
 * it is created. A seat count `seatCountSpec` does not offer is left out,
 * so a game whose only parameter is a fixed seat count offers none.
 *
 * @throws Error when such a parameter is not one of the game's.
 */
export function setupParameters(entry: CatalogueEntry): ParameterSpec[] {
  const seatCountName = seatCountParameter(entry);
  const seatCount = seatCountSpec(entry);
  const offered = entry.parameters.flatMap(spec => {
    if (spec.name !== seatCountName) return [spec];
    return seatCount === null ? [] : [seatCount];
  });
  const shown = SETUP_PARAMETERS[entry.name];
  if (shown === undefined) return offered;
  for (const name of shown) {
    if (!offered.some(spec => spec.name === name)) {
      throw new Error(`${entry.name} has no parameter ${name}.`);
    }
  }
  return offered.filter(spec => shown.includes(spec.name));
}

/**
 * A parameter's name for people, such as "Board Size" for `board_size`; each
 * of OpenSpiel's names for the seat count is "Players".
 */
export function parameterLabel(name: string): string {
  return SEAT_COUNT_NAMES.has(name) ? 'Players' : nameLabel(name);
}

/**
 * How a list shows one of `spec`'s values: a square board's size as its two
 * sides, such as "9×9", and any other value as its text.
 */
export function choiceLabel(
  spec: ParameterSpec,
  value: ParameterValue,
): string {
  if (spec.name === 'board_size' && typeof value === 'number') {
    return `${String(value)}×${String(value)}`;
  }
  return formatValue(spec, value);
}

/** Whether `spec` is edited with a checkbox rather than as text. */
export function isCheckbox(spec: ParameterSpec): boolean {
  return spec.type === 'bool' && spec.kind !== 'choice';
}

/** Whether `spec` is edited in a number box. */
export function isNumber(spec: ParameterSpec): boolean {
  return (
    (spec.type === 'int' || spec.type === 'float') && spec.kind !== 'choice'
  );
}

/**
 * Whether `spec` is edited in a text area: a string whose default spans
 * lines, such as a board layout, which a one-line field would flatten.
 */
export function isMultiline(spec: ParameterSpec): boolean {
  return spec.type === 'string' && String(spec.default).includes('\n');
}

/** `value` as the text a field shows; a real number keeps its point. */
export function formatValue(
  spec: ParameterSpec,
  value: ParameterValue,
): string {
  if (typeof value === 'number' && spec.type === 'float') {
    return Number.isInteger(value) ? value.toFixed(1) : String(value);
  }
  return String(value);
}

/** The draft that holds `value` in `spec`'s control. */
export function draftOf(spec: ParameterSpec, value: ParameterValue): Draft {
  return isCheckbox(spec) ? value === true : formatValue(spec, value);
}

/** Reads `draft` as a value of `spec`. */
export function parseDraft(spec: ParameterSpec, draft: Draft): ParsedDraft {
  const typed = typeof draft === 'boolean' ? draft : parseText(spec, draft);
  if (typed === null) return {ok: false, problem: TYPE_PROBLEMS[spec.type]};
  if (spec.kind === 'range') {
    const {minimum, maximum} = spec;
    if (minimum === null || maximum === null) {
      throw new Error(`Parameter ${spec.name} is a range without bounds.`);
    }
    if (typeof typed !== 'number' || typed < minimum || typed > maximum) {
      return {
        ok: false,
        problem: `Enter a number from ${String(minimum)} to ${String(maximum)}.`,
      };
    }
  }
  if (spec.kind === 'choice' && spec.choices?.includes(typed) !== true) {
    return {ok: false, problem: 'Choose one of the listed values.'};
  }
  return {ok: true, value: typed};
}

const TYPE_PROBLEMS: Record<ParameterSpec['type'], string> = {
  bool: 'Enter true or false.',
  int: 'Enter a whole number.',
  float: 'Enter a number.',
  string: 'Enter text.',
};

const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function parseText(spec: ParameterSpec, text: string): ParameterValue | null {
  const trimmed = text.trim();
  switch (spec.type) {
    case 'bool':
      return trimmed === 'true' ? true : trimmed === 'false' ? false : null;
    case 'int': {
      const value = Number(trimmed);
      return /^[+-]?\d+$/.test(trimmed) && Number.isSafeInteger(value)
        ? value
        : null;
    }
    case 'float': {
      // Decimal notation only: Number() would also read '', '0x10' and
      // 'Infinity'.
      const value = Number(trimmed);
      return DECIMAL.test(trimmed) && Number.isFinite(value) ? value : null;
    }
    case 'string':
      return text;
    default: {
      const unreachable: never = spec.type;
      throw new Error(`Unhandled parameter type: ${String(unreachable)}`);
    }
  }
}

/**
 * The draft of each of `specs`, keyed by name, holding its value in a
 * session whose parameters, defaults included, are `parameters`.
 *
 * @throws Error when the session has no value for one of `specs`, or a value
 *     outside a choice's options.
 */
export function sessionDrafts(
  specs: readonly ParameterSpec[],
  parameters: ParameterValues,
): Record<string, Draft> {
  return Object.fromEntries(
    specs.map(spec => {
      const value = sessionValue(spec, parameters);
      // A list cannot show a value outside its options.
      if (spec.choices !== null && !spec.choices.includes(value)) {
        throw new Error(
          `The session's ${spec.name}, ${JSON.stringify(value)}, is not one of its choices.`,
        );
      }
      return [spec.name, draftOf(spec, value)];
    }),
  );
}

/**
 * The value of `spec` in a session whose parameters, defaults included, are
 * `parameters`.
 *
 * @throws Error when the session has no value for it.
 */
export function sessionValue(
  spec: ParameterSpec,
  parameters: ParameterValues,
): ParameterValue {
  const value = parameters[spec.name];
  if (value === undefined) {
    throw new Error(`The session has no value for parameter ${spec.name}.`);
  }
  return value;
}

/** Every draft of a form read at once. */
export interface ParsedDrafts {
  /** The value of every parameter, or null when some draft has a problem. */
  readonly values: ParameterValues | null;
  /** The problem of each draft that cannot be read, by parameter name. */
  readonly problems: ReadonlyMap<string, string>;
}

/**
 * Reads the draft of each of `specs` from `drafts`, keyed by name.
 *
 * @throws Error when `drafts` has no draft for one of `specs`.
 */
export function parseDrafts(
  specs: readonly ParameterSpec[],
  drafts: Readonly<Record<string, Draft>>,
): ParsedDrafts {
  const values: Record<string, ParameterValue> = {};
  const problems = new Map<string, string>();
  for (const spec of specs) {
    const parsed = parseDraft(spec, requireDraft(spec, drafts));
    if (parsed.ok) values[spec.name] = parsed.value;
    else problems.set(spec.name, parsed.problem);
  }
  return {values: problems.size === 0 ? values : null, problems};
}

/**
 * The draft of `spec` in `drafts`, keyed by name.
 *
 * @throws Error when `drafts` has none.
 */
export function requireDraft(
  spec: ParameterSpec,
  drafts: Readonly<Record<string, Draft>>,
): Draft {
  const draft = drafts[spec.name];
  if (draft === undefined) {
    throw new Error(`No draft holds parameter ${spec.name}.`);
  }
  return draft;
}

/**
 * The parameters of `frame`'s session, defaults included, as Game options
 * shows them: an offered seat count is the session's number of seats, which
 * a game whose parameter defaults to "its own count" does not hold.
 *
 * @throws Error when `frame` is not a session of `entry`'s game.
 */
export function sessionParameters(
  entry: CatalogueEntry,
  frame: Frame,
): ParameterValues {
  requireSameGame(entry, frame);
  const {parameters, num_seats: seats} = frame.session;
  const seatCount = seatCountSpec(entry);
  return seatCount === null
    ? parameters
    : {...parameters, [seatCount.name]: seats};
}

/**
 * Checks that `entry` describes the game `frame` is a session of.
 *
 * @throws Error when it describes another game.
 */
export function requireSameGame(entry: CatalogueEntry, frame: Frame): void {
  if (frame.session.game !== entry.name) {
    throw new Error(
      `A session of ${frame.session.game} is not a game of ${entry.name}.`,
    );
  }
}

/**
 * The request that starts a new game of `entry` with `values`, keeping the
 * seats of `frame`, its chance policy and its seat count. Only values other
 * than the default are sent. When `values` names another seat count, seats
 * past the old count are human and seats past the new one are dropped.
 *
 * @throws Error when `frame` is not a session of `entry`'s game.
 */
export function newGame(
  entry: CatalogueEntry,
  frame: Frame,
  values: ParameterValues,
): CreateSession {
  requireSameGame(entry, frame);
  const name = seatCountParameter(entry);
  // A seat count Game options does not offer is the session's.
  const sessionCount =
    name === null ? undefined : frame.session.parameters[name];
  const sent: ParameterValues =
    name === null || name in values || sessionCount === undefined
      ? values
      : {...values, [name]: sessionCount};
  const parameters: Record<string, ParameterValue> = {};
  for (const spec of entry.parameters) {
    const value = sent[spec.name];
    if (value !== undefined && value !== spec.default) {
      parameters[spec.name] = value;
    }
  }
  const players = name === null ? undefined : sent[name];
  const seatCount = isSeatCount(entry, players)
    ? players
    : frame.session.num_seats;
  const seats = Array.from({length: seatCount}, (_, seat): SeatConfig => {
    const kept = frame.seats[seat];
    if (kept === undefined) return HUMAN_SEAT;
    const {controller, config, autoplay} = kept;
    return {controller, config, autoplay};
  });
  return {
    game: entry.name,
    parameters,
    seats,
    chance_policy: frame.session.chance_policy,
    seed: null,
  };
}
