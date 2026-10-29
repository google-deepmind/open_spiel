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

type Editable = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;

/**
 * Sets a control's value as a user would. React tracks the value it last
 * rendered, so the value is set through the prototype's setter, past
 * React's own, for React to see the change when the events arrive.
 */
export function enter(element: Editable, value: string): void {
  const prototype = Object.getPrototypeOf(element) as object;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
  if (descriptor?.set === undefined) {
    throw new Error('The control has no value.');
  }
  descriptor.set.call(element, value);
  act(() => {
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));
  });
}

/** The one control under `parent` whose label reads `label`; throws otherwise. */
export function labelled(parent: ParentNode, label: string): Editable {
  const found = Array.from(
    parent.querySelectorAll<Editable>('input, select, textarea'),
  ).filter(control =>
    Array.from(control.labels ?? []).some(
      element => element.textContent.trim() === label,
    ),
  );
  const [control] = found;
  if (found.length !== 1 || control === undefined) {
    throw new Error(
      `Expected one control labelled ${label}, found ${String(found.length)}.`,
    );
  }
  return control;
}
