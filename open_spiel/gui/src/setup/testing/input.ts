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

import {act} from 'react';

/**
 * Sets the value of a field, list or text area as a user would, inside
 * act(). React tracks a control's value through the setter on its prototype,
 * so the value is set through that setter; assigning it would not reach
 * onChange.
 */
export function enter(element: HTMLElement, value: string): void {
  const event = element instanceof HTMLSelectElement ? 'change' : 'input';
  act(() => {
    if (
      !Reflect.set(
        Object.getPrototypeOf(element) as object,
        'value',
        value,
        element,
      )
    ) {
      throw new Error(`A ${element.tagName} holds no value.`);
    }
    element.dispatchEvent(new Event(event, {bubbles: true}));
  });
}
