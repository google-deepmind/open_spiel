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

// <FreeLayer>: game content in a free layer. Owned by the motion module.
import {
  useContext,
  useLayoutEffect,
  useState,
  type JSX,
  type ReactNode,
} from 'react';
import {createPortal} from 'react-dom';
import type {Frame} from './service.ts';
import {tableLayer, viewportLayer} from './service.ts';
import {SceneRuntimeContext} from '../react/runtime.ts';
import {TableIdContext} from '../react/card_scene.tsx';

/** The props of <FreeLayer>. */
export interface FreeLayerProps {
  frame?: Frame;
  interactive?: boolean;
  tableId?: string;
  children: ReactNode;
}

/** Game content in a free layer: a canvas, a confetti node, a menu. 'viewport'
 *  portals to one fixed, full-window layer per document (pointer-events: none
 *  unless interactive). 'table' is absolutely positioned over the Table root:
 *  the nearest Table, or `tableId`, or the scene's only Table. */
export function FreeLayer({
  frame = 'viewport',
  interactive = false,
  tableId,
  children,
}: FreeLayerProps): JSX.Element | null {
  const given: unknown = frame;
  if (given !== 'viewport' && given !== 'table') {
    throw new Error(
      `<FreeLayer> frame must be 'viewport' or 'table', got ${JSON.stringify(frame)}`,
    );
  }
  const rt = useContext(SceneRuntimeContext);
  const nearest = useContext(TableIdContext);
  if (frame === 'table' && !rt) {
    throw new Error(
      "<FreeLayer frame='table'> must be inside a <Table> or <CardScene>",
    );
  }
  const [tableHost, setTableHost] = useState<HTMLElement | null>(null);
  useLayoutEffect(() => {
    if (frame !== 'table' || !rt) return;
    // Resolve now and on every commit: the Table may mount after this layer.
    const resolve = () => {
      const tables = rt.tables();
      if (tables.length === 0) return; // nothing mounted yet
      const id =
        tableId ??
        nearest ??
        (tables.length === 1 ? (tables[0]?.tableId ?? null) : null);
      if (id === null) {
        throw new Error(
          `<FreeLayer frame='table'> needs a tableId: the scene has ${String(tables.length)} tables (${tables.map(t => t.tableId).join(', ')})`,
        );
      }
      const root = rt.table(id).root; // throws, naming the mounted tables, for an unknown id
      setTableHost(prev =>
        prev?.parentElement === root ? prev : tableLayer(root),
      );
    };
    resolve();
    return rt.onCommit(resolve);
  }, [frame, rt, tableId, nearest]);
  // Inherited pointer-events: children take input only when interactive;
  // `display: contents` gives the switch no box of its own.
  const content = (
    <div
      data-free-layer-content={frame}
      style={{
        display: 'contents',
        pointerEvents: interactive ? 'auto' : 'none',
      }}
    >
      {children}
    </div>
  );
  if (frame === 'viewport') {
    if (typeof document === 'undefined') {
      throw new Error('<FreeLayer> needs a DOM document');
    }
    return createPortal(content, viewportLayer(document));
  }
  return tableHost ? createPortal(content, tableHost) : null;
}
