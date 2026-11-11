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

// The trick hold (trickHold.ts) as a config field: a game's View menu or
// Config panel.
import type {JSX} from 'react';
import {
  TRICK_HOLDS,
  TRICK_HOLD_LABELS,
  type TrickHold,
} from '../../cards/library/index.ts';
import {useTrickHold} from './trick_hold.ts';

/** The setting as a config field (a game's View menu or Config panel). */
export function TrickHoldField(): JSX.Element {
  const [hold, setHold] = useTrickHold();
  return (
    <label
      className="ogs-config-field"
      title="How long a finished trick stays on the table before it is collected"
    >
      <span>Trick hold</span>
      <select
        value={hold}
        onChange={e => {
          setHold(e.target.value as TrickHold);
        }}
        data-testid="trick-hold-select"
      >
        {TRICK_HOLDS.map(h => (
          <option key={h} value={h}>
            {TRICK_HOLD_LABELS[h]}
          </option>
        ))}
      </select>
    </label>
  );
}
