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

// A notice drawn over the euchre table (an error the viewer should see),
// absolutely placed so that it never moves the board when it comes and goes.

import type {JSX, ReactNode} from 'react';

/** The properties of GameplayNotice. */
export interface GameplayNoticeProps {
  /** How the notice reads; only errors are drawn by the euchre board. */
  readonly tone: 'error';
  readonly children: ReactNode;
}

/** A notice over the table. */
export function GameplayNotice({
  tone,
  children,
}: GameplayNoticeProps): JSX.Element {
  return (
    <div className={`euchre-notice euchre-notice--${tone}`} role="alert">
      {children}
    </div>
  );
}
