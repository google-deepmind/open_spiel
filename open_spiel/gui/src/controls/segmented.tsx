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

import './controls.css';

/** One choice of a Segmented control. */
export interface SegmentedOption<Value extends string> {
  value: Value;
  label: string;
}

/** The properties of a Segmented control. */
export interface SegmentedProps<Value extends string> {
  /** The visible caption, which also names the group. */
  label: string;
  value: Value;
  options: ReadonlyArray<SegmentedOption<Value>>;
  onChange: (value: Value) => void;
}

/**
 * A choice of one of a few values, shown as a row of buttons with the chosen
 * one pressed.
 */
export function Segmented<Value extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedProps<Value>): JSX.Element {
  const labelId = useId();
  if (!options.some(option => option.value === value)) {
    throw new Error(
      `${label}: ${JSON.stringify(value)} is not one of the options.`,
    );
  }
  return (
    <div className="segmented" role="group" aria-labelledby={labelId}>
      <span id={labelId} className="segmented__label">
        {label}
      </span>
      <span className="segmented__options">
        {options.map(option => (
          <button
            key={option.value}
            type="button"
            className="button segmented__option"
            aria-pressed={option.value === value}
            onClick={() => {
              onChange(option.value);
            }}
          >
            {option.label}
          </button>
        ))}
      </span>
    </div>
  );
}
