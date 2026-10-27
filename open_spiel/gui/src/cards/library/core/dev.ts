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

// Whether the client runs in development (the Vite dev server or a test),
// where the card library warns about a game's likely mistakes; a production
// build drops those warnings. Vite replaces `import.meta.env.DEV` at build
// time.

declare global {
  interface ImportMeta {
    readonly env: {readonly DEV: boolean};
  }
}

/** Whether the client runs in development rather than a production build. */
export const DEV: boolean = import.meta.env.DEV;
