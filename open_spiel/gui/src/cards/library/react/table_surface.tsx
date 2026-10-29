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

// The frame and surface a table draws in: its slots (with their captions
// and cues), its regions, its foot and the drop marks.
import {useMemo, type CSSProperties, type JSX, type RefObject} from 'react';
import {Descriptions} from '../a11y/describe.ts';
import {LiveRegion} from '../a11y/live_region.tsx';
import {sceneInternals} from '../core/scene.ts';
import {DropMarks} from '../dnd/drop_marks.tsx';
import {libraryTagSheet, tagScope} from '../style.ts';
import {CAPTION_HEIGHT, CUE_WIDTH} from '../table_spec.ts';
import {CardWidthContext} from './card_scene.tsx';
import type {SceneRuntimeImpl} from './scene_runtime.ts';
import type {TableProps} from './table.tsx';
import {groupSlotRuns, SlotView, type Renderer} from './table_slot.tsx';

type SceneSlot = ReturnType<typeof sceneInternals>['slots'][number];

// The frame and surface the table draws in, with its slots, regions, foot
// and (for an implicit scene) its live region and descriptions.
function frameStyle(
  contain: boolean,
  overflow: boolean,
  responsive: boolean,
  centred: boolean,
): CSSProperties {
  return {
    width: '100%',
    height: contain ? '100%' : undefined,
    overflow: overflow ? 'auto' : 'visible',
    ...(responsive ? {contain: 'inline-size', minWidth: 0} : {}),
    ...(centred ? {display: 'flex', alignItems: 'center'} : {}),
  };
}

/** The table's frame and surface. */
export function TableFrame({
  props,
  renderer: r,
  frameRef,
  surfaceRef,
  boxWidth,
  own,
  live,
}: {
  props: TableProps;
  renderer: Renderer;
  frameRef: RefObject<HTMLDivElement | null>;
  surfaceRef: RefObject<HTMLDivElement | null>;
  boxWidth: number;
  own: SceneRuntimeImpl | null;
  live: string;
}): JSX.Element {
  const {scene, cardWidth, tableId, style} = r;
  const {fit = 'width', foot, testId} = props;
  const {overflow} = sceneInternals(scene);
  const scope = tagScope(style);
  const sheet = useMemo(() => libraryTagSheet(scope, style), [scope, style]);
  // 'contain' centres the table in its box on both axes (as object-fit
  // does); 'width' keeps it at the top. A scrolling table starts top-left.
  const centred = fit === 'contain' && !overflow;
  // A table fitted to its container takes the width it is given and asks
  // for none: its drawn width must not become its container's minimum (a
  // grid or flex column, a shrink-to-fit box), or the container would keep
  // the width the table was drawn at and the table would fit that, never
  // shrinking when the window does. A table at an explicit card width has a
  // size of its own, and keeps it.
  const responsive = props.cardWidth === undefined;
  const frame = frameStyle(fit === 'contain', overflow, responsive, centred);
  // A scrolling table with no tab stop inside (inert, or nothing
  // interactive) is itself the tab stop, so a keyboard can scroll it (axe
  // scrollable-region-focusable). With a tab stop, focus scrolls it. An
  // unmeasured box (width 0) is not known to scroll yet.
  const scrollStop =
    overflow && boxWidth > 0 && ![...r.a11y.values()].some(n => n.tabbable);
  const surface: CSSProperties & Record<string, string | number | undefined> = {
    position: 'relative',
    width: scene.width,
    height: scene.height,
    margin: overflow ? undefined : '0 auto',
    flex: 'none',
    '--card-w': `${String(cardWidth)}px`,
  };
  return (
    <div
      ref={frameRef}
      data-testid={testId}
      data-table
      data-table-overflow={overflow ? 'true' : 'false'}
      tabIndex={scrollStop ? 0 : undefined}
      style={frame}
    >
      {sheet && <style>{sheet}</style>}
      <div
        ref={surfaceRef}
        data-table-surface
        data-table-id={tableId}
        data-cs={scope}
        data-card-width={cardWidth}
        data-table-variant={scene.variant + 1}
        style={surface}
      >
        <SlotRuns r={r} captions={props.captions} />
        <Regions r={r} regions={props.regions} />
        <DropMarks drag={r.rt.drag} tableId={tableId} scene={scene} />
      </div>
      {foot !== undefined && foot !== null && (
        <div
          data-table-foot
          style={{
            position: 'sticky',
            left: 0,
            bottom: 0,
            zIndex: 1,
            boxSizing: 'border-box',
          }}
        >
          {foot}
        </div>
      )}
      {own && <LiveRegion text={live} />}
      {own && <Descriptions scene={own} />}
    </div>
  );
}

// A slot's caption above it and its continuation cue beside it.
function SlotBands({
  l,
  r,
  caption,
}: {
  l: SceneSlot;
  r: Renderer;
  caption: string | undefined;
}): JSX.Element {
  const s = l.spec;
  const {cardWidth} = r;
  const band = Math.round(CAPTION_HEIGHT * cardWidth);
  const cueBand = CUE_WIDTH * cardWidth;
  return (
    <>
      {caption !== undefined && (
        <div
          key={`${s.id}:caption`}
          data-slot-caption={s.id}
          title={caption}
          style={{
            position: 'absolute',
            left: s.x * cardWidth,
            top: s.y * cardWidth - band,
            width: l.cw,
            height: band,
            lineHeight: `${String(band)}px`,
            fontSize: Math.max(12, Math.round(band * 0.8)),
            textAlign: 'center',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            pointerEvents: 'none',
          }}
        >
          {caption}
        </div>
      )}
      {s.cue && (
        <div
          key={`${s.id}:cue`}
          data-slot-cue={s.cue}
          aria-hidden
          style={{
            position: 'absolute',
            left:
              s.cue === 'continues'
                ? s.x * cardWidth + l.cw
                : s.x * cardWidth - cueBand,
            top: s.y * cardWidth + l.ch / 2 - cueBand / 2,
            width: cueBand,
            height: cueBand,
            lineHeight: `${String(cueBand)}px`,
            fontSize: Math.max(12, Math.round(cueBand)),
            textAlign: 'center',
            opacity: 0.7,
            pointerEvents: 'none',
          }}
        >
          {s.cue === 'continues' ? '↵' : '↪'}
        </div>
      )}
    </>
  );
}

// The slots, consecutive slots of one group in a group element.
function SlotRuns({
  r,
  captions,
}: {
  r: Renderer;
  captions: TableProps['captions'];
}): JSX.Element {
  const {slots} = sceneInternals(r.scene);
  return (
    <>
      {groupSlotRuns(slots, r.policy).map(run => {
        const drawn = run.slots.map(l => {
          const s = l.spec;
          const caption =
            s.caption === undefined
              ? undefined
              : (captions?.[s.id] ?? s.caption);
          return [
            <SlotBands key={`${s.id}:bands`} l={l} r={r} caption={caption} />,
            <SlotView key={s.id} l={l} r={r} />,
          ];
        });
        const first = run.slots[0];
        if (run.group === undefined || first === undefined) return drawn[0];
        // A group element has no box of its own (the slots are placed on
        // the surface), so it only names its slots.
        const label = r.policy?.groupLabel?.(run.group);
        return (
          <div
            key={`group:${run.group}:${first.spec.id}`}
            data-slot-group={run.group}
            role={label === undefined ? undefined : 'group'}
            aria-label={label}
          >
            {drawn}
          </div>
        );
      })}
    </>
  );
}

// The regions: each a box on the surface holding the game's content.
function Regions({
  r,
  regions,
}: {
  r: Renderer;
  regions: TableProps['regions'];
}): JSX.Element {
  const {scene, cardWidth, rt} = r;
  const regionClass = new Map(
    sceneInternals(scene).regionSpecs.map(x => [x.id, x.className] as const),
  );
  return (
    <>
      {scene.regions.map(region => {
        // A region with nothing in it this render is still a box
        // on the surface, drawn over the slots and regions before
        // it: it takes no pointer, so it never swallows the click
        // meant for what lies under it (an empty region laid over
        // a filled one). A drop target keeps the pointer.
        const content = regions?.[region.id];
        const empty =
          content === undefined || content === null || content === false;
        return (
          <div
            key={`region:${region.id}`}
            className={regionClass.get(region.id)}
            data-region-id={region.id}
            data-occludes={region.occludes ? 'true' : 'false'}
            data-droppable={region.droppable ? 'true' : undefined}
            data-drop-legal={
              rt.libraryTagsFor(`region:${region.id}`).includes('drop-legal')
                ? 'true'
                : undefined
            }
            style={
              {
                position: 'absolute',
                left: region.box.x,
                top: region.box.y,
                width: region.box.width,
                height: region.box.height,
                transform: region.box.rotate
                  ? `rotate(${String(region.box.rotate)}deg)`
                  : undefined,
                boxSizing: 'border-box',
                '--card-w': `${String(cardWidth)}px`,
                pointerEvents: empty && !region.droppable ? 'none' : undefined,
              } as CSSProperties
            }
          >
            <CardWidthContext.Provider value={cardWidth}>
              {content}
            </CardWidthContext.Provider>
          </div>
        );
      })}
    </>
  );
}
