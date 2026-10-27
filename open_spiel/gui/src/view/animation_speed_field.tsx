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

import {type JSX, useId} from 'react';

import {
  ANIMATION_SPEEDS,
  ANIMATION_SPEED_LABELS,
  type AnimationSpeed,
} from '../motion/animation_speed.ts';
import {
  useAnimationSpeed,
  useReducedMotion,
} from '../motion/use_animation_speed.ts';

import './view.css';

const SPEED_HELP =
  'How fast cards and pieces move when anyone plays, a bot included. Slow ' +
  'makes each move easy to follow; Instant shows the result at once.';

/**
 * View › Animation: how fast a board that animates its moves plays them.
 * One setting for every game on this browser.
 */
export function AnimationSpeedField(): JSX.Element {
  const id = useId();
  const [speed, setSpeed] = useAnimationSpeed();
  const reduced = useReducedMotion();
  return (
    <div className="view">
      <section className="view-card" aria-label="Animation">
        <div className="view-field" title={SPEED_HELP}>
          <label htmlFor={id}>Animation speed</label>
          <select
            id={id}
            className="view-field__select"
            value={speed}
            data-testid="animation-speed-select"
            onChange={event => {
              const chosen = ANIMATION_SPEEDS.find(
                value => value === event.currentTarget.value,
              );
              if (chosen === undefined) {
                throw new Error(
                  `No animation speed "${event.currentTarget.value}".`,
                );
              }
              setSpeed(chosen);
            }}
          >
            {ANIMATION_SPEEDS.map((value: AnimationSpeed) => (
              <option key={value} value={value}>
                {ANIMATION_SPEED_LABELS[value]}
              </option>
            ))}
          </select>
        </div>
        <p className="view-note" data-testid="animation-speed-scope">
          {reduced
            ? 'Your system asks for reduced motion, so moves show at once. ' +
              'Pauses that show a result, such as a finished trick, still ' +
              'follow this setting.'
            : 'Applies to every game on this browser.'}
        </p>
      </section>
    </div>
  );
}
