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

// Vite bundles the stylesheets a module imports; to TypeScript they are
// imports with no exports.
declare module '*.css';

// An SVG file imported with ?raw is its markup, as a string (Vite).
declare module '*.svg?raw' {
  const markup: string;
  // eslint-disable-next-line no-restricted-syntax -- Vite's ?raw import is a default export.
  export default markup;
}

// Vite's glob import, which Vite expands at build time: each file matching
// `pattern`, by path, as its URL or as the module itself; none when nothing
// matches.
interface ImportMeta {
  glob(
    pattern: string,
    options: {eager: true; query: '?url'; import: 'default'},
  ): Record<string, string>;
  glob(pattern: string, options: {eager: true}): Record<string, unknown>;
}
