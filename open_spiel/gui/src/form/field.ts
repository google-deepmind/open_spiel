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

// A form field: one named setting, the values it admits, and the text a user
// types for it.

/** A value a setting takes on the wire. */
export type FieldValue = boolean | number | string;

/** What a field holds while it is edited: a box's state, or typed text. */
export type FieldDraft = boolean | string;

/** What a required field that is left empty says. */
export const REQUIRED_ERROR = 'This setting is required.';

interface FieldBase {
  /** The setting's name, as the server takes it. */
  readonly name: string;
  /** What the form calls the setting. */
  readonly label: string;
  readonly description: string | null;
  /** Whether the setting must be sent: the server has no default for it. */
  readonly required: boolean;
  /** The setting's default; null only for a required setting. */
  readonly default: FieldValue | null;
}

/** A setting that is on or off. A box is always sent, so never missing. */
export interface BooleanField extends FieldBase {
  readonly kind: 'boolean';
  readonly default: boolean | null;
}

/** A setting that takes a number, optionally a whole one, within bounds. */
export interface NumberField extends FieldBase {
  readonly kind: 'integer' | 'number';
  readonly default: number | null;
  /** Values offered as one-click choices beside the box; none if absent. */
  readonly presets?: readonly number[];
  /** The least value admitted, inclusive, or null for no bound. */
  readonly minimum: number | null;
  /** The greatest value admitted, inclusive, or null for no bound. */
  readonly maximum: number | null;
}

/** A setting that takes text, optionally bounded in length and patterned. */
export interface StringField extends FieldBase {
  readonly kind: 'string';
  readonly default: string | null;
  /** Values offered as one-click choices beside the box; none if absent. */
  readonly presets?: readonly string[];
  /** The fewest characters admitted, or null for no bound. */
  readonly minLength: number | null;
  /** The most characters admitted, or null for no bound. */
  readonly maxLength: number | null;
  /**
   * A regular expression the text must contain a match of, as JSON Schema's
   * `pattern` does, or null for none.
   */
  readonly pattern: string | null;
}

/** A setting that takes one of a list of values. */
export interface ChoiceField extends FieldBase {
  readonly kind: 'choice';
  readonly choices: readonly FieldValue[];
}

/** A setting a form edits, of any kind; `kind` tells which. */
export type Field = BooleanField | NumberField | StringField | ChoiceField;

/** A draft read as its field's value, or why it cannot be. */
export type FieldReading =
  | {readonly ok: true; readonly value: FieldValue}
  | {readonly ok: false; readonly error: string};

/**
 * Returns the draft that shows `value`, or the field's default when `value`
 * is undefined. A choice's draft is the index of its value in `choices`, and
 * the empty string when `value` is not one of them. A required field with
 * neither starts empty: unticked, unchosen, or with no text.
 */
export function draftOf(field: Field, value?: FieldValue): FieldDraft {
  const shown = value ?? field.default;
  switch (field.kind) {
    case 'boolean':
      return shown === true;
    case 'choice': {
      const index = shown === null ? -1 : field.choices.indexOf(shown);
      return index < 0 ? '' : String(index);
    }
    case 'integer':
    case 'number':
    case 'string':
      return shown === null ? '' : String(shown);
    default: {
      const unreachable: never = field;
      throw new Error(`Unhandled field kind: ${String(unreachable)}`);
    }
  }
}

/**
 * Reads `draft` as a value of `field`. A required field left empty is
 * refused with `REQUIRED_ERROR`.
 *
 * @throws Error when the draft is not of the field's kind: a box's state for
 *     a boolean field, and text for any other.
 */
export function readDraft(field: Field, draft: FieldDraft): FieldReading {
  if (field.kind === 'boolean') {
    if (typeof draft !== 'boolean') {
      throw new Error(`The box ${field.name} has a text draft.`);
    }
    return {ok: true, value: draft};
  }
  if (typeof draft !== 'string') {
    throw new Error(`The field ${field.name} has a box's draft.`);
  }
  const empty = (field.kind === 'string' ? draft : draft.trim()) === '';
  if (field.required && empty) {
    return {ok: false, error: REQUIRED_ERROR};
  }
  switch (field.kind) {
    case 'string':
      return readString(field, draft);
    case 'choice': {
      const value = draft === '' ? undefined : field.choices[Number(draft)];
      return value === undefined
        ? {ok: false, error: 'Choose one of the options.'}
        : {ok: true, value};
    }
    case 'integer':
    case 'number':
      return readNumber(field, draft);
    default: {
      const unreachable: never = field;
      throw new Error(`Unhandled field kind: ${String(unreachable)}`);
    }
  }
}

/** Says, as a sentence, which numbers `field` admits. */
export function admittedNumbers(field: NumberField): string {
  const what = field.kind === 'integer' ? 'A whole number' : 'A number';
  const {minimum, maximum} = field;
  if (minimum !== null && maximum !== null) {
    return `${what} from ${String(minimum)} to ${String(maximum)}.`;
  }
  if (minimum !== null) return `${what}, at least ${String(minimum)}.`;
  if (maximum !== null) return `${what}, at most ${String(maximum)}.`;
  return `${what}.`;
}

/** Says, as a sentence, which text `field` admits; null for any text. */
export function admittedText(field: StringField): string | null {
  const {minLength, maxLength, pattern} = field;
  const characters = (count: number): string =>
    count === 1 ? '1 character' : `${String(count)} characters`;
  let length: string | null = null;
  if (minLength !== null && maxLength !== null) {
    length =
      minLength === maxLength
        ? `exactly ${characters(minLength)}`
        : `${String(minLength)} to ${characters(maxLength)}`;
  } else if (minLength !== null) {
    length = `at least ${characters(minLength)}`;
  } else if (maxLength !== null) {
    length = `at most ${characters(maxLength)}`;
  }
  const parts = [
    ...(length === null ? [] : [`of ${length}`]),
    ...(pattern === null ? [] : [`matching ${pattern}`]),
  ];
  return parts.length === 0 ? null : `Text ${parts.join(', ')}.`;
}

/** A text field's length bounds and pattern. */
export type TextConstraints = Pick<
  StringField,
  'minLength' | 'maxLength' | 'pattern'
>;

/**
 * Whether `text` is admitted by length bounds and a pattern. Lengths count
 * characters (code points), as JSON Schema does, not UTF-16 units.
 */
export function admitsText(
  constraints: TextConstraints,
  text: string,
): boolean {
  const {minLength, maxLength, pattern} = constraints;
  const length = Array.from(text).length;
  return (
    (minLength === null || length >= minLength) &&
    (maxLength === null || length <= maxLength) &&
    (pattern === null || new RegExp(pattern, 'u').test(text))
  );
}

function readString(field: StringField, text: string): FieldReading {
  if (admitsText(field, text)) return {ok: true, value: text};
  const admitted = admittedText(field);
  if (admitted === null) {
    throw new Error(`The field ${field.name} refused text it admits.`);
  }
  return {ok: false, error: admitted};
}

// A number in decimal notation, optionally with an exponent, as a person
// types it: not hexadecimal, binary or `Infinity`, which `Number` also reads.
const DECIMAL = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

function readNumber(field: NumberField, text: string): FieldReading {
  const trimmed = text.trim();
  const value = Number(trimmed);
  const admitted =
    DECIMAL.test(trimmed) &&
    Number.isFinite(value) &&
    (field.kind === 'number' || Number.isSafeInteger(value)) &&
    (field.minimum === null || value >= field.minimum) &&
    (field.maximum === null || value <= field.maximum);
  return admitted
    ? {ok: true, value}
    : {ok: false, error: admittedNumbers(field)};
}
