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

import {type JSX, type ReactNode, useMemo, useState} from 'react';

import {
  GameplayNoticeLayer,
  GameplayNoticeLayerProvider,
  type GameplayNoticePlacement,
} from './gameplay_notice.tsx';

interface GameplayShellProps {
  /** Class name for the outer panel container */
  readonly className?: string;
  /** The panel's content: its status row and board. */
  readonly children: ReactNode;
  /**
   * Edge of the table the notice stack (errors, "Bot is thinking…") is drawn
   * over. `top` by default; a game whose critical controls sit along the top
   * edge picks `bottom`.
   */
  readonly noticePlacement?: GameplayNoticePlacement;
}

/**
 * Shared shell for gameplay panels: the notice layer. `<GameplayNotice>`
 * anywhere in the children is drawn over the table and takes no layout
 * space, so a status or error never moves the board.
 *
 * The client's Gameplay panel draws the loading and error states, so the
 * children are rendered only once there is a position to show.
 */
export function GameplayShell({
  className = 'ogs-panel',
  children,
  noticePlacement = 'top',
}: GameplayShellProps): JSX.Element {
  const [noticeLayer, setNoticeLayer] = useState<HTMLElement | null>(null);
  const noticeContext = useMemo(() => ({layer: noticeLayer}), [noticeLayer]);

  return (
    <div className={`${className} gameplay-shell`}>
      <GameplayNoticeLayerProvider value={noticeContext}>
        {children}
        <GameplayNoticeLayer
          placement={noticePlacement}
          layerRef={setNoticeLayer}
        />
      </GameplayNoticeLayerProvider>
    </div>
  );
}
