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

import type {ParameterSpec} from '../contract/contract.ts';
import {
  type Draft,
  choiceLabel,
  formatValue,
  isCheckbox,
  isMultiline,
  isNumber,
  parameterLabel,
} from './parameters.ts';

import './setup.css';

/** The properties of a ParameterField. */
export interface ParameterFieldProps {
  spec: ParameterSpec;
  draft: Draft;
  /** Whether the draft cannot be read as a value. */
  invalid: boolean;
  /** Why a change to this parameter was not applied, or null. */
  notice: string | null;
  onChange: (draft: Draft) => void;
  /**
   * Called when a number or text field loses the focus, or Enter is pressed
   * in a one-line field.
   */
  onCommit: () => void;
}

/**
 * One game parameter as a card row: its name on the left and its control on
 * the right, its description beneath, and why a change to it was not
 * applied, if one was not. The control is a checkbox for a boolean, a list
 * for a choice, a number box for a number, a text area for text that spans
 * lines, and a text field for any other text.
 */
export function ParameterField({
  spec,
  draft,
  invalid,
  notice,
  onChange,
  onCommit,
}: ParameterFieldProps): JSX.Element {
  const id = useId();
  const descriptionId = spec.description === null ? null : `${id}-description`;
  const noticeId = notice === null ? null : `${id}-notice`;
  const describedBy = [descriptionId, noticeId].filter(part => part !== null);
  return (
    <div className="setup-field">
      <label className="setup-field__label" htmlFor={id}>
        {parameterLabel(spec.name)}
      </label>
      <Control
        spec={spec}
        draft={draft}
        onChange={onChange}
        onCommit={onCommit}
        attributes={{
          id,
          'aria-describedby':
            describedBy.length > 0 ? describedBy.join(' ') : undefined,
          'aria-invalid': invalid,
        }}
      />
      {descriptionId !== null && (
        <p id={descriptionId} className="setup-field__description">
          {spec.description}
        </p>
      )}
      {noticeId !== null && (
        <p id={noticeId} className="setup-field__notice" role="alert">
          {notice}
        </p>
      )}
    </div>
  );
}

// The attributes every kind of control takes from its field.
interface ControlAttributes {
  readonly id: string;
  readonly 'aria-describedby': string | undefined;
  readonly 'aria-invalid': boolean;
}

function Control({
  spec,
  draft,
  onChange,
  onCommit,
  attributes,
}: Omit<ParameterFieldProps, 'invalid' | 'notice'> & {
  attributes: ControlAttributes;
}): JSX.Element {
  if (isCheckbox(spec)) {
    return (
      <input
        {...attributes}
        className="setup-field__checkbox"
        type="checkbox"
        checked={draft === true}
        onChange={event => {
          onChange(event.currentTarget.checked);
        }}
      />
    );
  }
  if (spec.kind === 'choice') {
    return (
      <ChoiceControl
        spec={spec}
        draft={draft}
        onChange={onChange}
        attributes={attributes}
      />
    );
  }
  // A typed value is committed when its field loses the focus, and when
  // Enter is pressed in a one-line field.
  const typed = {
    ...attributes,
    className: 'setup-input',
    value: String(draft),
    onBlur: onCommit,
    onChange: (event: {currentTarget: {value: string}}): void => {
      onChange(event.currentTarget.value);
    },
  };
  const onKeyDown = (event: {key: string}): void => {
    if (event.key === 'Enter') onCommit();
  };
  if (isNumber(spec)) {
    return (
      <input
        {...typed}
        type="number"
        min={spec.minimum ?? undefined}
        max={spec.maximum ?? undefined}
        step={spec.type === 'int' ? 1 : 'any'}
        onKeyDown={onKeyDown}
      />
    );
  }
  if (isMultiline(spec)) {
    return (
      <textarea
        {...typed}
        spellCheck={false}
        rows={String(draft).split('\n').length}
      />
    );
  }
  return (
    <input {...typed} type="text" spellCheck={false} onKeyDown={onKeyDown} />
  );
}

function ChoiceControl({
  spec,
  draft,
  onChange,
  attributes,
}: Pick<ParameterFieldProps, 'spec' | 'draft' | 'onChange'> & {
  attributes: ControlAttributes;
}): JSX.Element {
  if (spec.choices === null) {
    throw new Error(`Parameter ${spec.name} is a choice without options.`);
  }
  return (
    <select
      {...attributes}
      className="setup-input"
      value={String(draft)}
      onChange={event => {
        onChange(event.currentTarget.value);
      }}
    >
      {spec.choices.map(choice => {
        const text = formatValue(spec, choice);
        return (
          <option key={text} value={text}>
            {choiceLabel(spec, choice)}
          </option>
        );
      })}
    </select>
  );
}
