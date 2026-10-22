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

import type {Controller, SeatView} from '../contract/contract.ts';
import {fieldsOf} from '../form/json_schema.ts';
import {HUMAN, controllerNamed, savedConfig} from './seat_draft.ts';

/** The properties of a SeatSummary. */
export interface SeatSummaryProps {
  readonly seat: SeatView;
  readonly controllers: readonly Controller[];
}

/**
 * The settings of a seat the viewer may not change, as label and value rows:
 * whether a person or a bot plays it, its controller, whether it autoplays,
 * then each setting, defaults included, labelled and ordered as the editor
 * shows them. Settings no form can show are shown as their JSON.
 */
export function SeatSummary({
  seat,
  controllers,
}: SeatSummaryProps): JSX.Element {
  const controller = controllerNamed(controllers, seat.controller);
  const fields = fieldsOf(controller.config_schema);
  const human = seat.controller === HUMAN;
  const rows: Array<readonly [term: string, value: string]> = [
    ['Seat', human ? 'Human' : 'Bot'],
    ['Controller', controller.label],
    ...(human ? [] : [['Autoplay', seat.autoplay ? 'On' : 'Off'] as const]),
  ];
  if (fields !== null) {
    const {config} = savedConfig(seat, controllers);
    for (const field of fields) {
      const value = config[field.name];
      rows.push([
        field.label,
        value === undefined
          ? '\u2014'
          : typeof value === 'boolean'
            ? value
              ? 'Yes'
              : 'No'
            : String(value),
      ]);
    }
  }
  const json =
    fields === null && Object.keys(seat.config).length > 0
      ? JSON.stringify(seat.config, null, 2)
      : null;
  return (
    <div className="seat__summary">
      <dl className="seat__summary-rows">
        {rows.map(([term, value], index) => (
          <div key={String(index)} className="seat__summary-row">
            <dt>{term}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {json !== null && <pre className="seat__summary-json">{json}</pre>}
    </div>
  );
}
