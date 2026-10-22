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

interface FieldBase {
  /** The setting's name, as the server takes it. */
  readonly name: string;
  /** What the form calls the setting. */
  readonly label: string;
  readonly description: string | null;
  readonly default: FieldValue;
}

/** A setting that is on or off. */
export interface BooleanField extends FieldBase {
  readonly kind: 'boolean';
  readonly default: boolean;
}

/** A setting that takes a number, optionally a whole one, within bounds. */
export interface NumberField extends FieldBase {
  readonly kind: 'integer' | 'number';
  readonly default: number;
  /** Values offered as one-click choices beside the box; none if absent. */
  readonly presets?: readonly number[];
  /** The least value admitted, inclusive, or null for no bound. */
  readonly minimum: number | null;
  /** The greatest value admitted, inclusive, or null for no bound. */
  readonly maximum: number | null;
}

/** A setting that takes any text. */
export interface StringField extends FieldBase {
  readonly kind: 'string';
  readonly default: string;
  /** Values offered as one-click choices beside the box; none if absent. */
  readonly presets?: readonly string[];
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
 * the empty string when `value` is not one of them.
 */
export function draftOf(field: Field, value?: FieldValue): FieldDraft {
  const shown = value ?? field.default;
  switch (field.kind) {
    case 'boolean':
      return shown === true;
    case 'choice': {
      const index = field.choices.indexOf(shown);
      return index < 0 ? '' : String(index);
    }
    case 'integer':
    case 'number':
    case 'string':
      return String(shown);
    default: {
      const unreachable: never = field;
      throw new Error(`Unhandled field kind: ${String(unreachable)}`);
    }
  }
}

/**
 * Reads `draft` as a value of `field`.
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
  switch (field.kind) {
    case 'string':
      return {ok: true, value: draft};
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
