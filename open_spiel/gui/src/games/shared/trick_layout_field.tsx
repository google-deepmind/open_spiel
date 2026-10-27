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

// The trick layout for trick-taking boards as a config field: the cards of
// the trick set toward their players, or overlapping in the middle
// (cards/library/trick_area.ts). A game keeps the setting (its own key and
// default, e.g. Hearts per table) and offers it with this field in its View
// menu or Config panel.
import type {JSX} from 'react';
import {
  TRICK_LAYOUTS,
  TRICK_LAYOUT_LABELS,
  type TrickLayout,
} from '../../cards/library/index.ts';

/** The trick layout field's props: the layout shown and how to choose. */
export interface TrickLayoutFieldProps {
  value: TrickLayout;
  onChange: (next: TrickLayout) => void;
}

/** The trick layout as a config field. */
export function TrickLayoutField({
  value,
  onChange,
}: TrickLayoutFieldProps): JSX.Element {
  return (
    <label
      className="ogs-config-field"
      title="Where the cards of the trick lie: each toward its player, or overlapping in the middle"
    >
      <span>Trick layout</span>
      <select
        value={value}
        onChange={e => {
          const next = e.target.value as TrickLayout;
          if (!TRICK_LAYOUTS.includes(next)) {
            throw new Error(
              `trick layout: unknown option ${JSON.stringify(next)}`,
            );
          }
          onChange(next);
        }}
        data-testid="trick-layout-select"
      >
        {TRICK_LAYOUTS.map(l => (
          <option key={l} value={l}>
            {TRICK_LAYOUT_LABELS[l]}
          </option>
        ))}
      </select>
    </label>
  );
}
