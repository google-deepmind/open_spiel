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

// Stubs for the browser APIs jsdom lacks, such as navigator.clipboard and
// document.execCommand.

const stubbed: Array<[object, string]> = [];

/** Defines `target[name]` as `value` until restoreProperties(). */
export function stubProperty(
  target: object,
  name: string,
  value: unknown,
): void {
  if (Object.hasOwn(target, name)) {
    throw new Error(`${name} is already an own property; it is not a stub.`);
  }
  Object.defineProperty(target, name, {value, configurable: true});
  stubbed.push([target, name]);
}

/** Removes every property stubProperty() defined. */
export function restoreProperties(): void {
  for (const [target, name] of stubbed.splice(0)) {
    Reflect.deleteProperty(target, name);
  }
}
