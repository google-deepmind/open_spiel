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

// Words a label writes in capitals, as people write them.
const ACRONYMS: ReadonlyMap<string, string> = new Map([['url', 'URL']]);

/**
 * A setting's name for people: its words capitalised, so `board_size` and
 * `boardSize` read "Board Size", and `url` reads "URL".
 */
export function nameLabel(name: string): string {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .split(/[_\s]+/)
    .filter(word => word !== '')
    .map(
      word =>
        ACRONYMS.get(word.toLowerCase()) ??
        word.charAt(0).toUpperCase() + word.slice(1),
    )
    .join(' ');
}
