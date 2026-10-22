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
import {useId} from 'react';

import type {
  BooleanField,
  Field,
  FieldDraft,
  NumberField,
  StringField,
} from './field.ts';
import {FieldInfo} from './field_info.tsx';

import './form.css';

/** The properties of a FieldForm. */
export interface FieldFormProps {
  readonly fields: readonly Field[];
  /** Every field's draft, by field name. */
  readonly drafts: Readonly<Record<string, FieldDraft>>;
  /** Why a field's draft cannot be read, by field name. */
  readonly errors: Readonly<Record<string, string>>;
  readonly onChange: (name: string, draft: FieldDraft) => void;
}

/**
 * One row per field: its label on the left, with its description behind an
 * "i" toggle, and its control on the right: a checkbox, a number or text
 * box, or a list of choices. A box with presets offers them as chips under
 * it. A field's error is shown under its row.
 *
 * @throws Error when a field has no draft.
 */
export function FieldForm({
  fields,
  drafts,
  errors,
  onChange,
}: FieldFormProps): JSX.Element {
  const id = useId();
  return (
    <div className="field-form">
      {fields.map((field, index) => {
        const draft = drafts[field.name];
        if (draft === undefined) {
          throw new Error(`The field ${field.name} has no draft.`);
        }
        return (
          <FieldRow
            key={field.name}
            id={`${id}-${String(index)}`}
            field={field}
            draft={draft}
            error={errors[field.name] ?? null}
            onChange={next => {
              onChange(field.name, next);
            }}
          />
        );
      })}
    </div>
  );
}

interface FieldRowProps {
  readonly id: string;
  readonly field: Field;
  readonly draft: FieldDraft;
  readonly error: string | null;
  readonly onChange: (draft: FieldDraft) => void;
}

function FieldRow({
  id,
  field,
  draft,
  error,
  onChange,
}: FieldRowProps): JSX.Element {
  const errorId = `${id}-error`;
  const control: ControlAttributes = {
    id,
    'aria-describedby': error === null ? undefined : errorId,
    'aria-invalid': error !== null,
  };
  return (
    <div className="field-form__field" data-kind={field.kind}>
      <span className="field-form__label">
        <label htmlFor={id}>{field.label}</label>
        {field.description !== null && (
          <FieldInfo description={field.description} />
        )}
      </span>
      {field.kind === 'boolean' ? (
        <input
          {...control}
          className="field-form__check"
          type="checkbox"
          checked={draft === true}
          onChange={event => {
            onChange(event.currentTarget.checked);
          }}
        />
      ) : (
        <Input
          field={field}
          control={control}
          draft={String(draft)}
          onChange={onChange}
        />
      )}
      {error !== null && (
        <p id={errorId} className="field-form__error">
          {error}
        </p>
      )}
    </div>
  );
}

// The attributes that identify and describe a field's control.
interface ControlAttributes {
  readonly id: string;
  readonly 'aria-describedby': string | undefined;
  readonly 'aria-invalid': boolean;
}

interface InputProps {
  readonly field: Exclude<Field, BooleanField>;
  readonly control: ControlAttributes;
  readonly draft: string;
  readonly onChange: (draft: string) => void;
}

// The control of a field whose draft is text.
function Input({field, control, draft, onChange}: InputProps): JSX.Element {
  if (field.kind === 'choice') {
    return (
      <select
        {...control}
        className="control"
        value={draft}
        onChange={event => {
          onChange(event.currentTarget.value);
        }}
      >
        {draft === '' && <option value="">Choose…</option>}
        {field.choices.map((choice, index) => (
          <option key={String(index)} value={String(index)}>
            {String(choice)}
          </option>
        ))}
      </select>
    );
  }
  const numeric = field.kind !== 'string';
  const box = (
    <input
      {...control}
      className="control"
      type={numeric ? 'number' : 'text'}
      min={numeric ? (field.minimum ?? undefined) : undefined}
      max={numeric ? (field.maximum ?? undefined) : undefined}
      step={numeric ? (field.kind === 'integer' ? 1 : 'any') : undefined}
      value={draft}
      onChange={event => {
        onChange(event.currentTarget.value);
      }}
    />
  );
  const presets = field.presets ?? [];
  if (presets.length === 0) return box;
  return (
    <div className="field-form__with-presets">
      {box}
      <Presets field={field} draft={draft} onPick={onChange} />
    </div>
  );
}

// A box's presets as chips; a click puts that value in the box.
function Presets({
  field,
  draft,
  onPick,
}: {
  readonly field: NumberField | StringField;
  readonly draft: string;
  readonly onPick: (draft: string) => void;
}): JSX.Element {
  const presets: ReadonlyArray<number | string> = field.presets ?? [];
  return (
    <div
      className="field-form__presets"
      role="group"
      aria-label={`${field.label} presets`}
    >
      {presets.map(preset => (
        <button
          key={String(preset)}
          type="button"
          className="field-form__preset"
          aria-pressed={String(preset) === draft.trim()}
          onClick={() => {
            onPick(String(preset));
          }}
        >
          {String(preset)}
        </button>
      ))}
    </div>
  );
}
