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

import {useLayoutEffect} from 'react';

import {useViewSetting} from '../games/shared/view_settings.ts';

/**
 * Sets the view setting `name` to `value` while mounted, as a choice made in
 * the View panel would: render it beside the board under test.
 */
export function SetViewSetting({
  name,
  value,
}: {
  readonly name: string;
  readonly value: unknown;
}): null {
  const [, set] = useViewSetting<unknown>(name, null);
  useLayoutEffect(() => {
    set(value);
  }, [set, value]);
  return null;
}
